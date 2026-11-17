"""Run configuration: schema, load profiles, validation.

A RunConfig fully determines a benchmark run — same config + same images +
same models => same matrix (determinism at the orchestration level).
"""

import re
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import json

# Run ids land in container names and Kubernetes resource names, so
# they must be DNS-1123-label safe (lowercase alphanumerics + dashes).
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def sanitize_run_id(value: str) -> str:
    """Coerce an arbitrary string into a run-id-safe token."""
    cleaned = re.sub(r"[^a-z0-9-]", "-", value.lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:48]


def generate_run_id(prefix: str = "") -> str:
    """Unique, DNS-1123-safe run identity.

    Sortable (timestamp first) and collision-proof (random suffix), so
    two runs can never produce the same container names or be confused
    with each other — even against containers leaked by a crashed run.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    token = secrets.token_hex(3)
    base = f"{prefix}-{stamp}-{token}" if prefix else f"{stamp}-{token}"
    return sanitize_run_id(base)


FRAMEWORKS_CPU = ["sklearn", "xgboost", "lightgbm", "onnx"]
FRAMEWORKS_GPU = ["onnx"]
SUITES = [
    "protocol",
    "batching",
    "workers",
    "caching",
    "multimodel",
    "lifecycle",
    "saturation",
    "processing",
    "framework",
    "middleware",
    "codec",
    "grpc",
    "hardware",
    "faults",
]
TIERS = ["small", "medium", "large"]

# Concurrency ladders per load profile.
LOAD_PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "concurrency": [1, 8],
        "qps_fractions": [0.75],
        "batch_sizes": [1],
        "warmup_seconds": 5,
        "window_seconds": 10,
        "stability_windows": 2,
    },
    "standard": {
        "concurrency": [1, 2, 4, 8, 16, 32, 64],
        "qps_fractions": [0.5, 0.9, 1.1],
        "batch_sizes": [1, 32],
        "warmup_seconds": 15,
        "window_seconds": 30,
        "stability_windows": 3,
    },
    "exhaustive": {
        "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
        "qps_fractions": [0.25, 0.5, 0.75, 0.9, 1.0, 1.1],
        "batch_sizes": [1, 8, 32, 128, 512],
        "warmup_seconds": 30,
        "window_seconds": 60,
        "stability_windows": 5,
    },
}

# Default images: floating odh-master tags — always the latest codebase.
# Override for R&D via RunConfig.image / --image / make IMAGE=...
DEFAULT_IMAGES = {
    "cpu": "quay.io/opendatahub/mlserver:odh-master",
    "gpu": "quay.io/opendatahub/odh-mlserver-cuda:odh-master",
}


@dataclass
class RunConfig:
    target: str = "cpu"  # cpu | gpu
    engine: str = "local"  # local (auto-detect runtime) | cluster
    rounds: int = 1  # replicates per cell (multi-round statistics)
    frameworks: list[str] = field(default_factory=lambda: list(FRAMEWORKS_CPU))
    suites: list[str] = field(default_factory=lambda: list(SUITES))
    tiers: list[str] = field(default_factory=lambda: list(TIERS))
    load_profile: str = "standard"  # quick | standard | exhaustive | custom
    custom: dict[str, Any] = field(default_factory=dict)
    image: str = ""  # resolved at runtime if empty
    warm_workers: bool = True  # deprecated no-op knob, kept for schema
    cache_enabled: str = "false"  # false | true | both
    include_real: bool = False  # add pinned downloadable reference models
    baseline: str | None = None
    output_dir: str = ""
    repo_root: str = ""
    # Unique run identity: threaded into container names, Kubernetes
    # resource names and result artifacts so every run is
    # distinguishable. Auto-generated at execute() time when empty.
    run_id: str = ""

    # ------------------------------------------------------------------
    def profile(self) -> dict[str, Any]:
        if self.load_profile == "custom":
            merged = dict(LOAD_PROFILES["standard"])
            merged.update(self.custom)
            return merged
        return LOAD_PROFILES[self.load_profile]

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.target not in ("cpu", "gpu"):
            problems.append(f"target must be cpu or gpu, got {self.target!r}")
        allowed = FRAMEWORKS_GPU if self.target == "gpu" else FRAMEWORKS_CPU
        for framework in self.frameworks:
            if framework not in allowed:
                problems.append(
                    f"framework {framework!r} not valid for target "
                    f"{self.target!r} (allowed: {allowed})"
                )
        for suite in self.suites:
            if suite not in SUITES:
                problems.append(f"unknown suite {suite!r} (allowed: {SUITES})")
        for tier in self.tiers:
            if tier not in TIERS:
                problems.append(f"unknown tier {tier!r} (allowed: {TIERS})")
        if self.load_profile not in (*LOAD_PROFILES, "custom"):
            problems.append(f"unknown load profile {self.load_profile!r}")
        if self.cache_enabled not in ("false", "true", "both"):
            problems.append("cache_enabled must be false | true | both")
        if self.run_id and not _RUN_ID_RE.match(self.run_id):
            problems.append(
                f"run_id must be lowercase alphanumerics/dashes "
                f"(DNS-1123 safe), got {self.run_id!r}"
            )
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls(**json.loads(path.read_text()))
