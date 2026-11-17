"""MLServer Docker container lifecycle + GPU ExecutionProvider guard.

Owns the container from `docker run` to teardown (signal-safe), polls the
V2 health endpoints with exponential backoff, and enforces the
anti-silent-fallback guard for GPU targets.
"""

import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

REST_PORT = 18080
GRPC_PORT = 18081
METRICS_PORT = 18082
CONTAINER_PREFIX = "mlserver-bench"


class ServerError(RuntimeError):
    pass


class GPUProviderFallback(ServerError):
    """CUDAExecutionProvider was not actively executing — never record
    CPU-fallback numbers as GPU results."""

    status = "FAILED_GPU_PROVIDER_FALLBACK"


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=300
    )
    if check and result.returncode != 0:
        raise ServerError(f"docker {args[:3]} failed: {result.stderr.strip()[:400]}")
    return result


class MLServerContainer:
    def __init__(
        self,
        name: str,
        image: str,
        model_repo: Path,
        target: str = "cpu",
        env: dict[str, str] | None = None,
        parallel_workers: int | None = None,
        metrics_http: Any = None,
        cpu_limit: float | None = None,
    ):
        self.name = name
        self.image = image
        self.model_repo = model_repo
        self.target = target
        self.env = env or {}
        self.parallel_workers = parallel_workers
        # CFS quota (--cpus): degrades THIS container deterministically
        # without loading the host — the graded-control mechanism.
        # Host CPU hogs (the previous mechanism) triggered connection
        # resets on the contended side (F-06: 2/2 graded attempts
        # aborted; every un-hogged run clean).
        self.cpu_limit = cpu_limit
        self._proc_poll = None
        self._started = False

    # -- lifecycle --------------------------------------------------------

    def start(self, timeout: float = 180.0) -> None:
        _docker(
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
            "--security-opt",
            "label=disable",
            "-p",
            f"{REST_PORT}:8080",
            "-p",
            f"{GRPC_PORT}:8081",
            "-p",
            f"{METRICS_PORT}:8082",
            *(["--gpus", "all"] if self.target == "gpu" else []),
            *(["--cpus", str(self.cpu_limit)] if self.cpu_limit else []),
            "-v",
            f"{self.model_repo}:/mnt/models",
            "-e",
            "MLSERVER_MODELS_DIR=/mnt/models",
            "-e",
            "MLSERVER_LOAD_MODELS_AT_STARTUP=true",
            "-e",
            "MLSERVER_GRPC_MAX_MESSAGE_LENGTH=33554432",
            *[
                arg
                for key, value in {
                    **self.env,
                    **(
                        {"MLSERVER_PARALLEL_WORKERS": str(self.parallel_workers)}
                        if self.parallel_workers is not None
                        else {}
                    ),
                }.items()
                for arg in ("-e", f"{key}={value}")
            ],
            self.image,
        )
        self._started = True
        try:
            self._wait_healthy(timeout)
            if self.target == "gpu":
                self.gpu_guard()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self._started:
            _docker("rm", "-f", self.name, check=False)
            self._started = False

    def __enter__(self) -> "MLServerContainer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- health -----------------------------------------------------------

    def _wait_healthy(self, timeout: float) -> None:
        import aiohttp
        import asyncio

        from .asyncutils import run_coroutine_blocking

        async def poll() -> None:
            backoff = 0.5
            deadline = time.monotonic() + timeout
            url = f"http://127.0.0.1:{REST_PORT}/v2/health/ready"
            async with aiohttp.ClientSession() as session:
                while time.monotonic() < deadline:
                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            if resp.status == 200:
                                return
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.6, 5.0)
            raise ServerError(
                f"{self.name}: server not healthy within {timeout}s — "
                f"logs:\n{self.logs()[-2000:]}"
            )

        run_coroutine_blocking(poll())

    def logs(self) -> str:
        result = _docker("logs", self.name, check=False)
        if result.returncode != 0:
            # Container already removed (--rm after death, or cleaned
            # up) — logs are unavailable, not an error.
            return (
                f"<logs unavailable: container {self.name!r} is gone — "
                "it likely exited and was auto-removed>"
            )
        return result.stdout

    # -- GPU anti-silent-fallback guard ------------------------------------

    def gpu_guard(self) -> None:
        """Fail fast unless CUDAExecutionProvider is the ACTIVE provider.

        The GPU image always contains CPUExecutionProvider as fallback, so
        a missing driver / bad visibility / opset issue silently degrades
        to CPU. We verify from the ACTIVE provider line — the
        "loaded with execution providers" message emitted when each
        model finishes loading — not from any "requested providers"
        debug line (a request for CUDA says nothing about whether ORT
        actually activated it).
        """
        logs = self.logs()
        # Anchor on the load-completion line: the LAST model's active
        # providers (earlier lines could be requests or partial loads).
        active_matches = re.findall(
            r"loaded with execution providers:\s*\[([^\]]*)\]", logs, re.I
        )
        if active_matches:
            providers = active_matches[-1]
        else:
            # Fallback to any provider line (older images).
            match = re.search(r"providers?\s*[:=]\s*\[?[\"']?([^\"'\]]+)", logs, re.I)
            providers = match.group(1) if match else ""
        if "CUDAExecutionProvider" not in providers:
            raise GPUProviderFallback(
                f"server did not activate CUDAExecutionProvider "
                f"(providers: {providers or 'not found in logs'}) — "
                f"check --gpus device visibility, driver and CUDA libs.\n"
                f"--- last logs ---\n{logs[-3000:]}"
            )
        if "CPUExecutionProvider" in providers and "cuda" not in providers.lower():
            # Provider line lists CPU only.
            raise GPUProviderFallback(
                f"only CPUExecutionProvider active: {providers}\n"
                f"--- last logs ---\n{logs[-3000:]}"
            )
        # Device-presence check (only when an NVIDIA GPU is visible on
        # THIS host; skipped on CPU-only driver shells). At guard time
        # no requests have run yet, so utilization SHOULD be ~0 —
        # the check is that a device exists and reports a parsable
        # utilization value, not that it is busy. Busy-time utilization
        # is telemetry's job (pynvml sampling during the load window).
        if shutil.which("nvidia-smi"):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if (
                result.returncode != 0
                or "no devices" in (result.stderr + result.stdout).lower()
            ):
                raise GPUProviderFallback(
                    "nvidia-smi reports no usable GPU on this host; "
                    "refusing to record CPU-fallback numbers as GPU results"
                )
            try:
                int(result.stdout.strip().splitlines()[0])
            except (ValueError, IndexError):
                raise GPUProviderFallback(
                    f"nvidia-smi returned an unparsable utilization "
                    f"value ({result.stdout.strip()!r}) — cannot confirm "
                    "a usable GPU"
                )


# ---------------------------------------------------------------------------
# Stale-container cleanup (make clean / runner start)
# ---------------------------------------------------------------------------


def clean_stale_containers(run_id: str = "") -> list[str]:
    """Remove stale benchmark containers.

    run_id="" sweeps ALL benchmark containers (make clean / run start
    — assumes exclusive host per the serial discipline). A specific
    run_id sweeps only that run's containers, so a run's teardown can
    never remove a concurrent run's servers.
    """
    prefix = f"{CONTAINER_PREFIX}-{run_id}" if run_id else CONTAINER_PREFIX
    result = _docker(
        "ps",
        "-a",
        "--filter",
        f"name={prefix}",
        "--format",
        "{{.Names}}",
        check=False,
    )
    names = [name for name in result.stdout.split() if name.strip()]
    for name in names:
        _docker("rm", "-f", name, check=False)
    return names


def detect_default_image(target: str, configured: str) -> str:
    """Configured override wins; else the floating odh-master default
    (docker auto-pulls the latest when not present locally)."""
    if configured:
        return configured
    from .config import DEFAULT_IMAGES

    return DEFAULT_IMAGES[target]


def resolve_digest(image: str) -> str:
    """Resolve a floating tag to its immutable digest reference.

    The benchmark NEVER runs a floating tag: at run start the tag is
    resolved (pull if needed) and every subsequent reference — container
    runs, receipts, manifests — uses repo@sha256:<digest>.
    """
    if "@sha256:" in image:
        return image  # already pinned
    result = _docker(
        "inspect",
        "--format",
        "{{index .RepoDigests 0}}",
        image,
        check=False,
    )
    digest = result.stdout.strip()
    if result.returncode != 0 or not digest or "@sha256:" not in digest:
        # Not present locally (or no repo digest): pull, then re-inspect.
        _docker("pull", image, check=False)
        result = _docker(
            "inspect",
            "--format",
            "{{index .RepoDigests 0}}",
            image,
            check=False,
        )
        digest = result.stdout.strip()
    if result.returncode == 0 and digest and "@sha256:" in digest:
        return digest
    # Locally-built image without a repo digest: pin by image ID.
    ident = _docker("inspect", "--format", "{{.Id}}", image, check=False).stdout.strip()
    if ident.startswith("sha256:"):
        repo = image.rsplit(":", 1)[0] if ":" in image else image
        return f"{repo}@{ident}"
    return image


def signal_safe_cleanup(*containers: MLServerContainer) -> None:
    """try/finally signal-safe teardown registered for SIGINT/SIGTERM."""

    def handler(signum: int, frame: Any) -> None:
        for container in containers:
            container.stop()
        clean_stale_containers()
        raise SystemExit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Image feature probes
# ---------------------------------------------------------------------------


def probe_settings(container: str, *names: str) -> dict[str, bool]:
    """Probe which server-level Settings fields the image supports.

    The published odh-master image can lag the local MLServer checkout
    (fork features like rest_binary_tensor land in the image only after
    a build). Suites call this against the RUNNING container and skip
    variants the image cannot honour — measuring an unapplied env var
    would silently duplicate the baseline.
    """
    import json

    probe_list = list(names)
    script = (
        "from mlserver.settings import Settings\n"
        "s = Settings()\n"
        "import json\n"
        f"print(json.dumps({{n: hasattr(s, n) for n in {probe_list!r}}}))\n"
    )
    result = _docker("exec", container, "python3", "-c", script, check=False)
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {name: False for name in names}


def probe_onnx_settings(container: str, *names: str) -> dict[str, bool]:
    """Probe which OnnxSettings (parameters.extra) fields the image's
    mlserver_onnx supports (e.g. cuda_graph, session_options)."""
    import json

    probe_list = list(names)
    script = (
        "from mlserver_onnx.settings import OnnxSettings\n"
        "s = OnnxSettings()\n"
        "import json\n"
        f"print(json.dumps({{n: hasattr(s, n) for n in {probe_list!r}}}))\n"
    )
    result = _docker("exec", container, "python3", "-c", script, check=False)
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {name: False for name in names}


def probe_import(container: str, module: str) -> bool:
    """Probe whether a Python module is importable inside the image
    (e.g. zmq for the zmq inference transport)."""
    result = _docker(
        "exec",
        container,
        "python3",
        "-c",
        f"import importlib.util; print(importlib.util.find_spec('{module}') "
        "is not None)",
        check=False,
    )
    return result.stdout.strip().endswith("True")


def probe_image_import(image: str, module: str) -> bool:
    """Probe whether a Python module is importable in a one-off
    container of the given image.

    Use BEFORE starting a server whose settings require the module
    (e.g. parallel_inference_transport=zmq crashes at startup when the
    zmq extra is missing — a post-start probe would burn the health
    timeout on a dead container).
    """
    result = _docker(
        "run",
        "--rm",
        "--security-opt",
        "label=disable",
        "--entrypoint",
        "python3",
        image,
        "-c",
        f"import importlib.util; print(importlib.util.find_spec('{module}') "
        "is not None)",
        check=False,
    )
    return result.stdout.strip().endswith("True")
