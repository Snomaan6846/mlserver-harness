"""Full-stack telemetry: Prometheus /metrics + system + GPU.

Runs as a background asyncio task during each measurement; aligns samples
to the run window. GPU telemetry via pynvml when available (guarded).
"""

import asyncio
import re
import time
from typing import Any

from .docker_server import METRICS_PORT


class TelemetryRecorder:
    """Samples Prometheus /metrics + host stats every `interval` seconds.

    Measurement hygiene: the /metrics session is REUSED across scrapes
    (a fresh session per sample paid TCP+TLS setup on every tick), and
    the docker-stats subprocess runs in an executor thread — it blocks
    1-2 s per call and used to stall the load-generator's event loop,
    silently adding latency spikes to whatever it was measuring.
    """

    def __init__(self, interval: float = 1.0, container_name: str | None = None):
        self.interval = interval
        self.container_name = container_name
        self.records: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._session: Any = None

    async def _ensure_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )
        return self._session

    async def _sample_prometheus(self) -> dict[str, Any]:
        url = f"http://127.0.0.1:{METRICS_PORT}/metrics"
        try:
            session = await self._ensure_session()
            async with session.get(url) as response:
                text = await response.text()
        except Exception:
            return {}
        return parse_prometheus(text)

    async def _sample_system(self) -> dict[str, Any]:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        record = {
            "host_cpu_percent": cpu,
            "host_ram_used_mb": memory.used / 1e6,
            "host_ram_percent": memory.percent,
        }
        if self.container_name:
            # Blocking subprocess -> executor thread (never the
            # load-generator's loop).
            stats = await asyncio.to_thread(
                _docker_container_stats, self.container_name
            )
            record.update(stats)
        record.update(_gpu_sample())
        return record

    async def _loop(self) -> None:
        while not self._stop.is_set():
            record = {"t": time.time()}
            record.update(await self._sample_prometheus())
            record.update(await self._sample_system())
            self.records.append(record)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._session is not None and not self._session.closed:
            await self._session.close()
        return self.records


def parse_prometheus(text: str) -> dict[str, float]:
    """Extract the latency-decomposition-relevant series.

    MLServer exposes (names abbreviated): model_infer_request_duration
    histograms, batch_request_queue / parallel_request_queue sizes, and
    request success/failure counters. We keep the latest value per
    series; histogram buckets are aggregated into a count + sum.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[1]
        try:
            number = float(value)
        except ValueError:
            continue
        # Strip label sets for aggregation: name{...} -> name
        base = re.sub(r"\{.*\}", "", name)
        if base.endswith("_sum") or base.endswith("_count"):
            values[base] = number
        elif base.endswith("_bucket"):
            continue  # buckets aggregated via _count/_sum
        else:
            values[base] = number
    return values


def _docker_container_stats(name: str) -> dict[str, Any]:

    from .docker_server import _docker

    result = _docker(
        "stats",
        "--no-stream",
        "--format",
        "{{.CPUPerc}} {{.MemUsage}}",
        name,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        cpu_percent, mem_usage = result.stdout.split()
        cpu = float(cpu_percent.rstrip("%"))
        used, _, total = mem_usage.partition("/")
        return {
            "container_cpu_percent": cpu,
            "container_mem_used": _parse_size(used),
            "container_mem_total": _parse_size(total),
        }
    except (ValueError, IndexError):
        return {}


def _parse_size(text: str) -> float:
    text = text.strip()
    try:
        if text.endswith("GiB"):
            return float(text[:-3]) * 1024
        if text.endswith("MiB"):
            return float(text[:-3])
        if text.endswith("KiB"):
            return float(text[:-3]) / 1024
        if text.endswith("B"):
            return float(text[:-1]) / (1024 * 1024)
    except ValueError:
        pass
    return 0.0


def _gpu_sample() -> dict[str, Any]:
    """pynvml telemetry, guarded — absent on CPU-only hosts."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        return {
            "gpu_sm_percent": utilization.gpu,
            "gpu_mem_controller_percent": utilization.memory,
            "gpu_vram_used_mb": memory.used / 1e6,
            "gpu_vram_total_mb": memory.total / 1e6,
            "gpu_power_w": power,
            "gpu_temp_c": temperature,
            "gpu_sm_clock_mhz": clocks,
        }
    except Exception:
        return {}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the telemetry time series to per-run aggregates."""
    if not records:
        return {}
    keys: set[str] = set()
    for record in records:
        keys.update(record.keys())
    keys -= {"t"}
    summary: dict[str, Any] = {"samples": len(records)}
    for key in sorted(keys):
        values = [
            record[key]
            for record in records
            if isinstance(record.get(key), (int, float))
        ]
        if not values:
            continue
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_max"] = max(values)
    return summary


def decompose_latency(
    prometheus_summary: dict[str, Any], client_result: dict[str, Any]
) -> dict[str, Any]:
    """Split E2E client latency into queue vs compute components using
    Prometheus counters (deltas are computed by the caller when a
    baseline snapshot exists)."""
    # Metric names as MLServer actually exposes them (no mlserver_
    # prefix; parse_prometheus keys are the base series names, and
    # summarize() suffixes them with _mean/_max).
    return {
        "client_p50_ms": client_result.get("latency_ms_p50"),
        "client_p99_ms": client_result.get("latency_ms_p99"),
        "infer_duration_sum": prometheus_summary.get(
            "model_infer_request_duration_sum_mean"
        ),
        "infer_duration_count": prometheus_summary.get(
            "model_infer_request_duration_count_max"
        ),
        "batch_queue_mean": prometheus_summary.get("batch_request_queue_mean"),
        "parallel_queue_mean": prometheus_summary.get("parallel_request_queue_mean"),
        "gpu_sm_percent_mean": prometheus_summary.get("gpu_sm_percent_mean"),
    }
