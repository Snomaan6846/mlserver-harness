"""ISVC deployment mode: create/patch InferenceServices for benchmarking.

Implements the minimal-ownership design from docs/ISVC_DEPLOYMENT_PLAN.md:
we own only the PVC and the ISVC. Everything else — Deployment, Service,
rollout, probes, Route lifecycle, pod cleanup — is handled by KServe and
odh-model-controller.

Usage:
    from bench.isvc import ISVCDeployment
    with ISVCDeployment(name, model_dir, pvc_name, ...) as isvc:
        isvc.wait_ready()
        # benchmark against isvc.service_dns or isvc.route_url
"""

import json
import subprocess
import time
from typing import Any

ISVC_API_VERSION = "serving.kserve.io/v1beta1"
RUNTIME_API_VERSION = "serving.kserve.io/v1alpha1"
DEFAULT_NAMESPACE = "mlserver-perf"


def _oc(
    namespace: str, *args: str, check: bool = True, stdin: str | None = None
) -> subprocess.CompletedProcess:
    cmd = ["oc", "-n", namespace] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        input=stdin,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"oc {args[:3]} failed: {result.stderr.strip()[:400]}")
    return result


class ISVCDeployment:
    """Manages a single InferenceService for benchmark trials."""

    def __init__(
        self,
        name: str,
        model_dir: str,
        pvc_name: str = "models-pvc",
        namespace: str = DEFAULT_NAMESPACE,
        model_format: str = "onnx",
        runtime: str = "mlserver-runtime",
        image: str | None = None,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        deployment_mode: str = "RawDeployment",
        exposed: bool = False,
    ):
        self.name = name
        self.model_dir = model_dir
        self.pvc_name = pvc_name
        self.namespace = namespace
        self.model_format = model_format
        self.runtime = runtime
        self.image = image
        self.env = env or {}
        self.resources = resources
        self.deployment_mode = deployment_mode
        self.exposed = exposed
        self._created = False

    # -- Properties ---------------------------------------------------------

    @property
    def service_name(self) -> str:
        """The KServe-created Service name: <isvc>-predictor."""
        return f"{self.name}-predictor"

    @property
    def service_dns(self) -> str:
        """Cluster-internal DNS for the predictor."""
        return f"{self.service_name}.{self.namespace}.svc.cluster.local"

    @property
    def rest_url(self) -> str:
        return f"http://{self.service_dns}:8080"

    @property
    def grpc_target(self) -> str:
        return f"{self.service_dns}:8081"

    @property
    def deployment_name(self) -> str:
        return f"{self.name}-predictor"

    @property
    def pod_label(self) -> str:
        """Label selector for the predictor pods."""
        return f"serving.kserve.io/inferenceservice={self.name}"

    # -- Lifecycle ----------------------------------------------------------

    def create(self) -> None:
        """Apply the InferenceService."""
        spec = self._build_spec()
        result = _oc(
            self.namespace,
            "apply",
            "-f",
            "-",
            stdin=json.dumps(spec),
        )
        if result.returncode != 0:
            raise RuntimeError(f"ISVC creation failed: {result.stderr[:400]}")
        self._created = True

    def delete(self) -> None:
        """Delete the ISVC (KServe garbage-collects owned resources)."""
        if self._created:
            _oc(
                self.namespace,
                "delete",
                "isvc",
                self.name,
                "--ignore-not-found",
                check=False,
            )
            self._created = False

    def patch_env(self, env: dict[str, str]) -> None:
        """Patch predictor env vars — triggers a fresh pod via the
        reconciler (the fresh-pod discipline for ISVC mode)."""
        patch = {
            "spec": {
                "predictor": {
                    "model": {
                        "env": [{"name": k, "value": v} for k, v in sorted(env.items())]
                    }
                }
            }
        }
        _oc(
            self.namespace,
            "patch",
            "isvc",
            self.name,
            "--type",
            "merge",
            "-p",
            json.dumps(patch),
        )

    def set_exposed(self, exposed: bool) -> None:
        """Toggle the visibility label for Route creation/deletion."""
        labels = self._get_labels()
        if exposed:
            labels["networking.kserve.io/visibility"] = "exposed"
        else:
            labels.pop("networking.kserve.io/visibility", None)

        patch = {"metadata": {"labels": labels}}
        _oc(
            self.namespace,
            "patch",
            "isvc",
            self.name,
            "--type",
            "merge",
            "-p",
            json.dumps(patch),
        )

    # -- Readiness ----------------------------------------------------------

    def wait_ready(self, timeout: float = 300) -> str:
        """Wait for the ISVC to report Ready. Returns the route URL
        (or empty string for cluster-local)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = _oc(
                self.namespace,
                "get",
                "isvc",
                self.name,
                "-o",
                "jsonpath={.status.conditions[-1].type}",
                check=False,
            )
            if result.stdout.strip() == "Ready":
                url_result = _oc(
                    self.namespace,
                    "get",
                    "isvc",
                    self.name,
                    "-o",
                    "jsonpath={.status.url}",
                    check=False,
                )
                return url_result.stdout.strip()
            time.sleep(3)
        raise TimeoutError(f"ISVC {self.name} not ready within {timeout}s")

    def get_pod_uids(self) -> list[str]:
        """Current predictor pod UIDs (for the fresh-pod proof)."""
        result = _oc(
            self.namespace,
            "get",
            "pods",
            "-l",
            self.pod_label,
            "-o",
            "jsonpath={.items[*].metadata.uid}",
            check=False,
        )
        return result.stdout.split()

    def wait_fresh_pod(self, old_uids: list[str], timeout: float = 300) -> str:
        """Wait for a new pod (different UID) to be ready — proving the
        old process is gone. Returns the new pod's UID."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            uids = self.get_pod_uids()
            new_uids = [u for u in uids if u not in old_uids]
            if new_uids and self._pods_ready(new_uids):
                # Prove old pods are gone
                still_old = [u for u in old_uids if u in uids]
                if not still_old:
                    return new_uids[0]
            time.sleep(3)
        raise TimeoutError(f"Fresh pod for {self.name} not ready within {timeout}s")

    def _pods_ready(self, uids: list[str]) -> bool:
        """Check if any pod with the given UIDs is Ready."""
        for uid in uids:
            result = _oc(
                self.namespace,
                "get",
                "pods",
                "-l",
                self.pod_label,
                "-o",
                f"jsonpath={{.items[?(@.metadata.uid=='{uid}')]"
                f".status.conditions[?(@.type=='Ready')].status}}",
                check=False,
            )
            if result.stdout.strip() == "True":
                return True
        return False

    def get_route_url(self) -> str:
        """Get the route URL (if exposed)."""
        result = _oc(
            self.namespace,
            "get",
            "isvc",
            self.name,
            "-o",
            "jsonpath={.status.url}",
            check=False,
        )
        return result.stdout.strip()

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> "ISVCDeployment":
        self.create()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.delete()

    # -- Internal -----------------------------------------------------------

    def _build_spec(self) -> dict[str, Any]:
        """Build the InferenceService manifest."""
        model_spec: dict[str, Any] = {
            "modelFormat": {"name": self.model_format},
            "runtime": self.runtime,
            "storageUri": f"pvc://{self.pvc_name}/{self.model_dir}",
        }

        if self.image:
            model_spec["image"] = self.image

        if self.env:
            model_spec["env"] = [
                {"name": k, "value": v} for k, v in sorted(self.env.items())
            ]

        if self.resources:
            model_spec["resources"] = self.resources

        metadata: dict[str, Any] = {
            "name": self.name,
            "namespace": self.namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": self.deployment_mode,
            },
        }

        if self.exposed:
            metadata["labels"] = {
                "networking.kserve.io/visibility": "exposed",
            }

        return {
            "apiVersion": ISVC_API_VERSION,
            "kind": "InferenceService",
            "metadata": metadata,
            "spec": {
                "predictor": {
                    "model": model_spec,
                },
            },
        }

    def _get_labels(self) -> dict[str, str]:
        result = _oc(
            self.namespace,
            "get",
            "isvc",
            self.name,
            "-o",
            "jsonpath={.metadata.labels}",
            check=False,
        )
        try:
            return json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def verify_prerequisites(namespace: str = DEFAULT_NAMESPACE) -> list[str]:
    """Verify RHOAI + ISVC CRD + mlserver templates are present.
    Returns a list of problems (empty = ready)."""
    problems: list[str] = []

    # ISVC CRD
    result = _oc(
        namespace,
        "get",
        "crd",
        "inferenceservices.serving.kserve.io",
        check=False,
    )
    if result.returncode != 0:
        problems.append("ISVC CRD not found — is RHOAI installed?")

    # mlserver templates
    result = _oc(
        "redhat-ods-applications",
        "get",
        "templates",
        check=False,
    )
    if result.returncode != 0 or "mlserver" not in result.stdout:
        problems.append(
            "mlserver serving runtime templates not found in " "redhat-ods-applications"
        )

    return problems


def create_pvc(
    name: str = "models-pvc",
    namespace: str = DEFAULT_NAMESPACE,
    size: str = "10Gi",
) -> None:
    """Create the models PVC if it doesn't exist."""
    result = _oc(namespace, "get", "pvc", name, check=False)
    if result.returncode == 0:
        return  # already exists

    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "resources": {"requests": {"storage": size}},
        },
    }
    _oc(namespace, "apply", "-f", "-", stdin=json.dumps(pvc))
