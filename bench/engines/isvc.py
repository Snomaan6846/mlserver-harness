"""ISVC/Cluster engine adapter: runs servers as RHOAI InferenceServices.

Implements the Engine interface using oc CLI commands against an
OpenShift cluster with RHOAI installed. Servers are InferenceServices
in RawDeployment mode with PVC model storage.

The platform (KServe + odh-model-controller) owns Deployments,
Services, Routes, probes, and pod lifecycle — we only create/patch
the ISVC and the PVC.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import Engine, ServerHandle

DEFAULT_NAMESPACE = "mlserver-perf"
# ISVC names are Kubernetes resource names (DNS-1123 labels).
CONTAINER_PREFIX = "mlserver-bench"


def _oc(
    namespace: str,
    *args: str,
    check: bool = True,
    stdin: str | None = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess:
    cmd = ["oc", "-n", namespace] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"oc {args[:3]} failed: {result.stderr.strip()[:400]}")
    return result


class EngineError(RuntimeError):
    """Raised when the ISVC engine cannot proceed."""

    pass


class ISVCEngine(Engine):
    """Runs servers as InferenceServices on OpenShift/RHOAI."""

    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        pvc_name: str = "models-pvc",
        runtime_cpu: str = "mlserver-runtime",
        runtime_gpu: str = "mlserver-cuda-runtime",
        network_path: str = "incluster",
        load_generator: str = "host",
        run_id: str = "",
        **kwargs: Any,
    ):
        self.namespace = namespace
        self.pvc_name = pvc_name
        self.runtime_cpu = runtime_cpu
        self.runtime_gpu = runtime_gpu
        self.network_path = network_path  # incluster | route
        self.load_generator = load_generator  # host | client-pod
        # Unique run identity for ISVC names: two runs (or a retry
        # after a crash) can never produce colliding resource names.
        from ..config import generate_run_id, sanitize_run_id

        self.run_id = sanitize_run_id(run_id) or generate_run_id("isvc")
        self._counter = 0
        self._port_forward_procs: list[subprocess.Popen] = []

    def _next_name(self) -> str:
        self._counter += 1
        return f"{CONTAINER_PREFIX}-{self.run_id}-{self._counter}"

    # -- Prerequisites -----------------------------------------------------

    def verify_prerequisites(self) -> list[str]:
        """Check RHOAI + ISVC CRD + mlserver templates. Returns problems."""
        problems = []

        # ISVC CRD
        result = _oc(
            self.namespace,
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
                "mlserver serving runtime templates not found in "
                "redhat-ods-applications"
            )

        return problems

    def ensure_pvc(self, size: str = "10Gi") -> None:
        """Create the models PVC if it doesn't exist."""
        result = _oc(self.namespace, "get", "pvc", self.pvc_name, check=False)
        if result.returncode == 0:
            return

        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": self.pvc_name,
                "namespace": self.namespace,
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {"requests": {"storage": size}},
            },
        }
        _oc(
            self.namespace,
            "apply",
            "-f",
            "-",
            stdin=json.dumps(pvc),
        )

    def upload_model(self, local_model_dir: str, remote_model_name: str) -> None:
        """Upload a model directory to the PVC via a model-loader pod."""
        # Deploy a temporary model-loader pod
        loader_name = "mlserver-bench-model-loader"
        _oc(
            self.namespace,
            "delete",
            "pod",
            loader_name,
            "--ignore-not-found",
            check=False,
        )

        loader_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": loader_name,
                "namespace": self.namespace,
            },
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "loader",
                        "image": "registry.access.redhat.com/ubi9/ubi:latest",
                        "command": ["/bin/bash", "-c", "sleep 3600"],
                        "volumeMounts": [
                            {
                                "name": "models",
                                "mountPath": "/mnt/models",
                            }
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "models",
                        "persistentVolumeClaim": {
                            "claimName": self.pvc_name,
                        },
                    }
                ],
            },
        }
        _oc(
            self.namespace,
            "apply",
            "-f",
            "-",
            stdin=json.dumps(loader_spec),
        )

        # Wait for the loader to be ready
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            result = _oc(
                self.namespace,
                "get",
                "pod",
                loader_name,
                "-o",
                "jsonpath={.status.phase}",
                check=False,
            )
            if result.stdout.strip() == "Running":
                break
            time.sleep(2)

        # Copy model files
        for file_path in sorted(Path(local_model_dir).iterdir()):
            if file_path.name in ("MANIFEST.json", "README.md"):
                continue
            remote_path = f"/mnt/models/{remote_model_name}/{file_path.name}"
            _oc(
                self.namespace,
                "exec",
                loader_name,
                "--",
                "mkdir",
                "-p",
                f"/mnt/models/{remote_model_name}",
            )
            # Stream file via stdin (oc cp needs tar)
            result = subprocess.run(
                [
                    "oc",
                    "-n",
                    self.namespace,
                    "exec",
                    "-i",
                    loader_name,
                    "--",
                    "sh",
                    "-c",
                    f"cat > {remote_path}",
                ],
                stdin=open(file_path, "rb"),
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise EngineError(
                    f"Failed to upload {file_path.name}: "
                    f"{result.stderr.decode()[:200]}"
                )

        # Clean up loader
        _oc(
            self.namespace,
            "delete",
            "pod",
            loader_name,
            "--ignore-not-found",
            check=False,
        )

    # -- Server lifecycle ------------------------------------------------

    def start_server(
        self,
        name: str,
        image: str,
        model_repo: str,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        target: str = "cpu",
    ) -> ServerHandle:
        # Honor the caller's name (restart reuses handle.name —
        # discarding it left the recreated resource unreachable by
        # the caller). Empty name falls back to a run-scoped name.
        isvc_name = name or self._next_name()
        runtime = self.runtime_gpu if target == "gpu" else self.runtime_cpu

        # Determine model format from the model repo
        model_format = self._detect_model_format(model_repo)

        model_spec: dict[str, Any] = {
            "modelFormat": {"name": model_format},
            "runtime": runtime,
            "storageUri": f"pvc://{self.pvc_name}/{model_repo}",
        }

        if image:
            model_spec["image"] = image

        if env:
            model_spec["env"] = [
                {"name": k, "value": v} for k, v in sorted(env.items())
            ]

        if resources:
            model_spec["resources"] = resources

        isvc_spec = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": isvc_name,
                "namespace": self.namespace,
                "annotations": {
                    "serving.kserve.io/deploymentMode": "RawDeployment",
                },
            },
            "spec": {
                "predictor": {"model": model_spec},
            },
        }

        # Add route label if network_path == "route"
        if self.network_path == "route":
            isvc_spec["metadata"]["labels"] = {
                "networking.kserve.io/visibility": "exposed",
            }

        _oc(
            self.namespace,
            "apply",
            "-f",
            "-",
            stdin=json.dumps(isvc_spec),
        )

        service_name = f"{isvc_name}-predictor"
        handle = ServerHandle(
            name=isvc_name,
            engine="isvc",
            rest_url=f"http://{service_name}.{self.namespace}.svc:8080",
            grpc_target=f"{service_name}.{self.namespace}.svc:8081",
            metrics_url=(f"http://{service_name}.{self.namespace}.svc:8082/metrics"),
            image=image,
            model_repo=model_repo,
            env=env or {},
            metadata={
                "target": target,
                "service_name": service_name,
                "pod_label": f"serving.kserve.io/inferenceservice={isvc_name}",
                "network_path": self.network_path,
                "load_generator": self.load_generator,
            },
        )
        return handle

    def stop_server(self, handle: ServerHandle) -> None:
        _oc(
            self.namespace,
            "delete",
            "isvc",
            handle.name,
            "--ignore-not-found",
            check=False,
        )
        # Wait for owned resources to be garbage-collected
        time.sleep(5)

    def wait_ready(self, handle: ServerHandle, timeout: float = 300) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = _oc(
                self.namespace,
                "get",
                "isvc",
                handle.name,
                "-o",
                "jsonpath={.status.conditions[-1].type}",
                check=False,
            )
            if result.stdout.strip() == "Ready":
                return True
            time.sleep(3)
        return False

    def get_server_pids(self, handle: ServerHandle) -> list[int]:
        output = self.exec_command(
            handle,
            [
                "sh",
                "-c",
                "for p in $(ls /proc/ 2>/dev/null | grep -E '^[0-9]+$'); "
                "do if grep -q mlserver /proc/$p/cmdline 2>/dev/null; "
                "then echo $p; fi; done",
            ],
        )
        return [int(pid) for pid in output.split() if pid.isdigit()]

    def wait_fresh_server(
        self,
        handle: ServerHandle,
        old_identifiers: list[str],
        timeout: float = 300,
    ) -> str:
        """Patch the ISVC env to trigger a reconciler rollout.

        The KServe controller creates a new Deployment generation →
        new pod with new UID. We wait for the new pod and prove the
        old one is gone.
        """
        # Trigger rollout by patching env (adds a timestamp annotation
        # to force a new generation even if env is identical)
        patch = {
            "spec": {
                "predictor": {
                    "model": {
                        "env": (
                            handle.env
                            and [
                                {"name": k, "value": v}
                                for k, v in sorted(handle.env.items())
                            ]
                        )
                        or [],
                        "annotations": {
                            "mlserver-bench.io/rollout": str(time.time()),
                        },
                    }
                }
            }
        }
        _oc(
            self.namespace,
            "patch",
            "isvc",
            handle.name,
            "--type",
            "merge",
            "-p",
            json.dumps(patch),
        )

        # Wait for a new pod (different UID from old_identifiers)
        deadline = time.monotonic() + timeout
        pod_label = handle.metadata.get(
            "pod_label", f"serving.kserve.io/inferenceservice={handle.name}"
        )
        while time.monotonic() < deadline:
            result = _oc(
                self.namespace,
                "get",
                "pods",
                "-l",
                pod_label,
                "-o",
                "jsonpath={.items[*].metadata.uid}",
                check=False,
            )
            current_uids = result.stdout.split()
            new_uids = [uid for uid in current_uids if uid not in old_identifiers]
            if new_uids and not any(uid in current_uids for uid in old_identifiers):
                # New pod exists and old pods are gone
                if self._pod_ready(new_uids[0], pod_label):
                    return new_uids[0]
            time.sleep(3)

        raise TimeoutError(f"Fresh pod for {handle.name} not ready within {timeout}s")

    def _pod_ready(self, uid: str, pod_label: str) -> bool:
        result = _oc(
            self.namespace,
            "get",
            "pods",
            "-l",
            pod_label,
            "-o",
            f"jsonpath={{.items[?(@.metadata.uid=='{uid}')]"
            f".status.conditions[?(@.type=='Ready')].status}}",
            check=False,
        )
        return result.stdout.strip() == "True"

    # -- Command execution ------------------------------------------------

    def exec_command(
        self,
        handle: ServerHandle,
        command: list[str],
        timeout: float = 60,
    ) -> str:
        """Execute a command in the ISVC's predictor pod."""
        pod_label = handle.metadata.get("pod_label", "")

        # Find the running pod
        result = _oc(
            self.namespace,
            "get",
            "pods",
            "-l",
            pod_label,
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        pod_name = result.stdout.strip()
        if not pod_name:
            return ""

        # Execute in the pod
        result = _oc(
            self.namespace,
            "exec",
            pod_name,
            "--",
            *command,
            check=False,
            timeout=timeout,
        )
        return result.stdout

    def get_logs(self, handle: ServerHandle) -> str:
        result = _oc(
            self.namespace,
            "logs",
            f"deployment/{handle.metadata.get('service_name', handle.name)}",
            check=False,
            timeout=30,
        )
        return result.stdout

    # -- Resource discovery ------------------------------------------------

    def get_pod_uids(self, handle: ServerHandle) -> list[str]:
        pod_label = handle.metadata.get("pod_label", "")
        result = _oc(
            self.namespace,
            "get",
            "pods",
            "-l",
            pod_label,
            "-o",
            "jsonpath={.items[*].metadata.uid}",
            check=False,
        )
        return result.stdout.split()

    def gpu_available(self) -> bool:
        result = _oc(
            self.namespace,
            "get",
            "nodes",
            "-o",
            "jsonpath={.items[0].status.allocatable." "nvidia\\.com/gpu}",
            check=False,
        )
        return bool(result.stdout.strip())

    def gpu_guard(self, handle: ServerHandle) -> None:
        """Verify CUDAExecutionProvider is actively executing."""
        logs = self.get_logs(handle)
        if "CUDAExecutionProvider" not in logs:
            raise EngineError(
                "CUDAExecutionProvider not active in ISVC logs. "
                "Check GPU resource request and node availability."
            )

    # -- Network access (port-forwarding for host load gen) ------------

    def setup_network(self, handle: ServerHandle) -> None:
        """Set up network access based on network_path and load_generator.

        - incluster + host: start oc port-forward
        - incluster + client-pod: no-op (client runs in-cluster)
        - route + any: no-op (route is already accessible)
        """
        if self.network_path == "incluster" and self.load_generator == "host":
            self._start_port_forward(handle)

    def teardown_network(self, handle: ServerHandle) -> None:
        """Stop port-forward processes."""
        for proc in self._port_forward_procs:
            if proc.poll() is None:
                proc.terminate()
        self._port_forward_procs.clear()

    def _start_port_forward(self, handle: ServerHandle) -> None:
        """Start oc port-forward for REST, gRPC, and metrics."""
        service_name = handle.metadata.get("service_name", "")

        for local_port, remote_port in [
            (18080, 8080),
            (18081, 8081),
            (18082, 8082),
        ]:
            proc = subprocess.Popen(
                [
                    "oc",
                    "-n",
                    self.namespace,
                    "port-forward",
                    f"svc/{service_name}",
                    f"{local_port}:{remote_port}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._port_forward_procs.append(proc)

        # Update handle URLs to localhost
        handle.rest_url = "http://127.0.0.1:18080"
        handle.grpc_target = "127.0.0.1:18081"
        handle.metrics_url = "http://127.0.0.1:18082/metrics"

        # Wait for port-forward to be established
        time.sleep(3)

    # -- Helpers -----------------------------------------------------------

    def _detect_model_format(self, model_repo: str) -> str:
        """Detect the model format from the model directory name."""
        name_lower = model_repo.lower()
        if "sklearn" in name_lower:
            return "sklearn"
        elif "xgboost" in name_lower:
            return "xgboost"
        elif "lightgbm" in name_lower:
            return "lightgbm"
        else:
            return "onnx"  # default
