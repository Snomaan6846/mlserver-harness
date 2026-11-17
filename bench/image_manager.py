"""Image manager: three-tier sourcing + derived image builder.

Tiers (in priority order):
  1. SOURCE: build from a local MLServer repo checkout
  2. OVERRIDE: user-provided image reference (tag or digest)
  3. DEFAULT: cluster template (ISVC mode) or floating tag (Docker)

Derived images (profiling layer for diagnose) are built on top of
whatever base is resolved, cached by content hash in the registry.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Packages needed per activity (only diagnose needs extras)
ACTIVITY_PACKAGES: dict[str, list[str]] = {
    "benchmark": [],  # stock image is fine
    "diagnose": [
        "py-spy==0.3.14",
        "memray==1.14.0",
        "psutil==7.2.2",
    ],
    "reference": [],  # uses separate Triton image
    "compare": [],  # post-processing only
}


@dataclass
class ImageSource:
    """User's image sourcing preference."""

    mode: str = "default"  # "default" | "override" | "source"
    repo_path: str = ""  # for source mode: path to MLServer checkout
    dockerfile: str = "Dockerfile"  # for source mode
    override_ref: str = ""  # for override mode: image reference
    registry: str = "quay.io"
    namespace: str = ""  # defaults to whoami or config


class ImageManager:
    """Resolves, builds, and caches images for benchmark activities."""

    def __init__(self, source: ImageSource | None = None):
        self.source = source or ImageSource()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_image(self, activity: str, engine: str = "docker") -> str:
        """Get the digest-pinned image for an activity.

        Args:
            activity: "benchmark" | "diagnose" | "reference" | "compare"
            engine: "docker" | "cluster"

        Returns:
            Digest-pinned image reference (never a floating tag).
        """
        base = self.resolve_base(engine)

        if not self._needs_derived(activity):
            return base

        return self.get_or_build_derived(base, activity)

    def resolve_base(self, engine: str = "docker") -> str:
        """Resolve the base image based on the source mode and engine."""
        if self.source.mode == "source":
            return self._build_from_source()
        elif self.source.mode == "override":
            return self._pin_override(self.source.override_ref)
        elif engine == "cluster":
            return self._get_cluster_template_image()
        else:
            return self._get_default_image()

    def get_or_build_derived(self, base: str, activity: str) -> str:
        """Get or build a derived image with activity-specific packages."""
        packages = ACTIVITY_PACKAGES.get(activity, [])
        if not packages:
            return base

        # Compute cache key
        cache_key = self._derive_cache_key(base, packages)
        tag = (
            f"{self.source.registry}/{self.source.namespace}/"
            f"mlserver-bench:{activity}-{cache_key}"
        )

        # Check if already exists in registry
        if self._image_exists(tag):
            return tag

        # Build derived image
        return self._build_derived(base, packages, tag)

    # ------------------------------------------------------------------
    # Tier 1: Build from source
    # ------------------------------------------------------------------

    def _build_from_source(self) -> str:
        """Build MLServer from a source repo checkout."""
        repo = Path(self.source.repo_path)
        if not repo.is_dir():
            raise ValueError(f"Source repo not found: {repo}")

        dockerfile = repo / self.source.dockerfile
        if not dockerfile.is_file():
            raise ValueError(f"Dockerfile not found: {dockerfile}")

        # Get git SHA for tagging
        sha = self._get_git_sha(repo)

        # Build
        tag = (
            f"{self.source.registry}/{self.source.namespace}/"
            f"mlserver-bench:base-{sha[:12]}"
        )
        result = subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile),
                "-t",
                tag,
                str(repo),
            ],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Source build failed: {result.stderr[-400:]}")

        # Push
        self._push_image(tag)

        # Pin to digest
        return self._pin_to_digest(tag)

    # ------------------------------------------------------------------
    # Tier 2: Override image
    # ------------------------------------------------------------------

    def _pin_override(self, ref: str) -> str:
        """Pin an override image to its digest."""
        if "@sha256:" in ref:
            return ref  # already pinned
        return self._pin_to_digest(ref)

    # ------------------------------------------------------------------
    # Tier 3: Default (cluster template or floating tag)
    # ------------------------------------------------------------------

    def _get_cluster_template_image(self) -> str:
        """Read the image from the cluster's serving runtime template."""
        # Try to read from the cluster
        result = subprocess.run(
            [
                "oc",
                "-n",
                "redhat-ods-applications",
                "get",
                "template",
                "mlserver-runtime-template",
                "-o",
                "jsonpath={.objects[0].spec.containers[0].image}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        # Fallback to default
        return self._get_default_image()

    def _get_default_image(self) -> str:
        """Get the default floating tag, resolved to a digest."""
        from .config import DEFAULT_IMAGES
        from .docker_server import resolve_digest

        # We don't know the target here — use CPU as default
        # The engine will override for GPU
        image = DEFAULT_IMAGES["cpu"]
        return resolve_digest(image)

    # ------------------------------------------------------------------
    # Derived image builder
    # ------------------------------------------------------------------

    def _build_derived(self, base: str, packages: list[str], tag: str) -> str:
        """Build a derived image on top of the base."""
        dockerfile_content = (
            f"FROM {base}\n" f"RUN pip install --no-cache-dir {' '.join(packages)}\n"
        )

        # Write to a temp build context
        import tempfile

        with tempfile.TemporaryDirectory() as build_dir:
            df_path = Path(build_dir) / "Dockerfile"
            df_path.write_text(dockerfile_content)

            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-f",
                    str(df_path),
                    "-t",
                    tag,
                    build_dir,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Derived image build failed: {result.stderr[-400:]}"
                )

        # Push
        self._push_image(tag)

        return tag

    def _needs_derived(self, activity: str) -> bool:
        """Check if the activity requires a derived image."""
        return bool(ACTIVITY_PACKAGES.get(activity))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive_cache_key(self, base: str, packages: list[str]) -> str:
        """Compute a deterministic cache key for a derived image."""
        content = json.dumps(
            {
                "base": base,
                "packages": sorted(packages),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _image_exists(self, tag: str) -> bool:
        """Check if an image already exists in the local registry."""
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0

    def _push_image(self, tag: str) -> None:
        """Push an image to the registry."""
        result = subprocess.run(
            ["docker", "push", tag],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            # Push failure is non-fatal (local use still works)
            pass

    def _pin_to_digest(self, tag: str) -> str:
        """Resolve a tag to its digest-pinned reference."""
        if "@sha256:" in tag:
            return tag

        # Pull if not present
        subprocess.run(
            ["docker", "pull", tag],
            capture_output=True,
            timeout=600,
        )

        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{index .RepoDigests 0}}",
                tag,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        digest = result.stdout.strip()
        if "@sha256:" in digest:
            return digest

        # Fallback: use image ID
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", tag],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ident = result.stdout.strip()
        if ident.startswith("sha256:"):
            repo = tag.rsplit(":", 1)[0]
            return f"{repo}@{ident}"

        return tag

    def _get_git_sha(self, repo: Path) -> str:
        """Get the current git SHA of a repo."""
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
