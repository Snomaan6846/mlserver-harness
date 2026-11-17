"""Container engine adapter: runs servers as local containers.

Works with any OCI container runtime: Docker, Podman, or any
docker-compatible CLI. The runtime is auto-detected at initialization
(docker → podman → nerdctl → error).

Wraps the container lifecycle behind the Engine interface so all
activities (benchmark, diagnose, reference) work identically on
local containers and ISVC/cluster.
"""

import re
import shutil
import subprocess
import time
from typing import Any

from .base import Engine, ServerHandle

REST_PORT = 18080
GRPC_PORT = 18081
METRICS_PORT = 18082
CONTAINER_PREFIX = "mlserver-bench"


def _detect_runtime() -> str:
    """Auto-detect the available container runtime binary.

    Priority: podman → docker → nerdctl → error.
    Podman is preferred because:
      - It's the RHEL/Fedora/OpenShift default
      - Rootless by default (safer)
      - Better SELinux integration
      - It's what RHOAI environments standardize on

    If both podman and docker are installed, podman wins.
    If only docker is available, docker is used.
    """
    for runtime in ("podman", "docker", "nerdctl"):
        if shutil.which(runtime):
            return runtime
    raise RuntimeError(
        "No container runtime found. Install one of: podman, docker, " "or nerdctl."
    )


def _get_runtime() -> str:
    """Get the container runtime (cached)."""
    global _CACHED_RUNTIME
    if _CACHED_RUNTIME is None:
        _CACHED_RUNTIME = _detect_runtime()
    return _CACHED_RUNTIME


_CACHED_RUNTIME: str | None = None


def _docker(
    *args: str, check: bool = True, timeout: float = 120
) -> subprocess.CompletedProcess:
    """Run a container runtime command (docker/podman/nerdctl)."""
    runtime = _get_runtime()
    result = subprocess.run(
        [runtime, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"{runtime} {args[:3]} failed: {result.stderr.strip()[:400]}"
        )
    return result


class ContainerEngine(Engine):
    """Runs servers as local containers (Docker, Podman, or nerdctl).

    Auto-detects the available runtime at initialization. Works
    identically regardless of which runtime is installed.
    """

    def __init__(self, run_id: str = "", **kwargs: Any):
        self.runtime = _get_runtime()
        # Unique run identity for container names: explicit run_id or
        # a generated fallback, so no two engine instances (or two
        # runs) can ever produce colliding container names.
        from ..config import generate_run_id, sanitize_run_id

        self.run_id = sanitize_run_id(run_id) or generate_run_id("eng")
        self._counter = 0

    def _next_name(self) -> str:
        self._counter += 1
        return f"{CONTAINER_PREFIX}-{self.run_id}-srv-{self._counter}"

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
        # Honor the caller's name: the restart path reuses
        # handle.name, and discarding it meant the restarted container
        # ran under a fresh name the caller could never stop (leak).
        # Empty name falls back to a run-scoped unique name.
        container_name = name or self._next_name()

        _docker(
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "--security-opt",
            "label=disable",
            "-p",
            f"{REST_PORT}:8080",
            "-p",
            f"{GRPC_PORT}:8081",
            "-p",
            f"{METRICS_PORT}:8082",
            *(["--gpus", "all"] if target == "gpu" else []),
            "-v",
            f"{model_repo}:/mnt/models",
            "-e",
            "MLSERVER_MODELS_DIR=/mnt/models",
            "-e",
            "MLSERVER_LOAD_MODELS_AT_STARTUP=true",
            "-e",
            "MLSERVER_GRPC_MAX_MESSAGE_LENGTH=33554432",
            *[
                arg
                for key, value in (env or {}).items()
                for arg in ("-e", f"{key}={value}")
            ],
            image,
        )

        handle = ServerHandle(
            name=container_name,
            engine="docker",
            rest_url=f"http://127.0.0.1:{REST_PORT}",
            grpc_target=f"127.0.0.1:{GRPC_PORT}",
            metrics_url=f"http://127.0.0.1:{METRICS_PORT}/metrics",
            image=image,
            model_repo=model_repo,
            env=env or {},
            metadata={"target": target},
        )
        return handle

    def stop_server(self, handle: ServerHandle) -> None:
        _docker("rm", "-f", handle.name, check=False, timeout=30)

    def wait_ready(self, handle: ServerHandle, timeout: float = 300) -> bool:
        import aiohttp
        import asyncio

        async def poll() -> bool:
            url = f"{handle.rest_url}/v2/health/ready"
            backoff = 0.5
            deadline = time.monotonic() + timeout
            async with aiohttp.ClientSession() as session:
                while time.monotonic() < deadline:
                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status == 200:
                                return True
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.6, 5.0)
            return False

        return asyncio.run(poll())

    def get_server_pids(self, handle: ServerHandle) -> list[int]:
        """Find PIDs by scanning /proc inside the container."""
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
        """Docker containers are recreated — just restart and wait."""
        # For Docker, "fresh" means stopping and starting a new container
        # with the same configuration. The caller re-creates via
        # start_server and this method is less relevant (containers
        # are already fresh by construction).
        # For env-var changes, we recreate the container.
        self.stop_server(handle)
        new_handle = self.start_server(
            handle.name,
            handle.image,
            handle.model_repo,
            env=handle.env,
            target=handle.metadata.get("target", "cpu"),
        )
        # Copy new handle properties back
        handle.rest_url = new_handle.rest_url
        handle.grpc_target = new_handle.grpc_target
        handle.metrics_url = new_handle.metrics_url
        self.wait_ready(handle, timeout)
        return handle.name

    # -- Command execution ------------------------------------------------

    def exec_command(
        self,
        handle: ServerHandle,
        command: list[str],
        timeout: float = 60,
    ) -> str:
        result = _docker("exec", handle.name, *command, check=False, timeout=timeout)
        return result.stdout

    def get_logs(self, handle: ServerHandle) -> str:
        result = _docker("logs", handle.name, check=False)
        return result.stdout

    # -- Resource discovery ------------------------------------------------

    def get_pod_uids(self, handle: ServerHandle) -> list[str]:
        """For Docker, container IDs serve as unique identifiers."""
        result = _docker("inspect", "--format", "{{.Id}}", handle.name, check=False)
        return [result.stdout.strip()] if result.stdout.strip() else []

    def gpu_available(self) -> bool:
        result = _docker(
            "info",
            "--format",
            "{{.Runtimes.nvidia}}",
            check=False,
            timeout=10,
        )
        return "nvidia" in result.stdout.lower()

    def gpu_guard(self, handle: ServerHandle) -> None:
        """Verify CUDAExecutionProvider in the server logs."""
        logs = self.get_logs(handle)
        match = re.search(r"providers?\s*[:=]\s*\[?[\"']?([^\"'\]]+)", logs, re.I)
        providers = match.group(1) if match else ""
        if "CUDAExecutionProvider" not in providers:
            from .base import EngineError

            raise EngineError(
                f"CUDAExecutionProvider not active (found: "
                f"{providers or 'none'}). Check --gpus and driver."
            )

    # -- Network (no-op for Docker; ports are mapped at start) ------

    def setup_network(self, handle: ServerHandle) -> None:
        pass  # ports already mapped

    def teardown_network(self, handle: ServerHandle) -> None:
        pass  # nothing to clean up
