"""Layer 5: Latency decomposition — where does each request's time go?

Combines Prometheus /metrics from MLServer (server-side stage timing)
with the client-side measurement to produce per-request attribution.
"""

import asyncio
import time
from typing import Any

import numpy as np

from bench import telemetry as telemetry_mod


async def collect_prometheus_baseline(
    metrics_port: int = 8082, base_url: str = "http://127.0.0.1"
) -> dict[str, Any]:
    """Snapshot Prometheus counters BEFORE the load window — the baseline
    for computing deltas during measurement."""
    import aiohttp

    url = f"{base_url}:{metrics_port}/metrics"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.get(url) as response:
                text = await response.text()
        return telemetry_mod.parse_prometheus(text)
    except Exception:
        return {}


async def collect_prometheus_after(
    metrics_port: int = 8082, base_url: str = "http://127.0.0.1"
) -> dict[str, Any]:
    """Snapshot Prometheus counters AFTER the load window."""
    return await collect_prometheus_baseline(metrics_port, base_url)


def compute_stage_attribution(
    baseline: dict[str, float],
    after: dict[str, float],
    request_count: int,
    client_latencies_ms: list[float] | None = None,
) -> dict[str, Any]:
    """Decompose per-request latency into serving stages using the
    Prometheus counter deltas.

    MLServer exposes (names may be prefixed with 'mlserver_'):
      - model_infer_request_duration_seconds_{sum,count} — total infer time
      - batch_request_queue — adaptive batching queue depth
      - parallel_request_queue — worker dispatch queue depth
    """
    attribution: dict[str, Any] = {
        "layer": "latency",
        "request_count": request_count,
    }

    def delta(name: str) -> float:
        return after.get(name, 0) - baseline.get(name, 0)

    # Server-side infer time (mean per request)
    infer_sum = delta("mlserver_model_infer_request_duration_seconds_sum")
    infer_count = delta("mlserver_model_infer_request_duration_seconds_count")
    if infer_count > 0 and request_count > 0:
        attribution["server_infer_ms_per_req"] = round(
            infer_sum / request_count * 1000, 2
        )
        attribution["server_infer_count"] = int(infer_count)

    # Queue depths (mean of the sampled values — from telemetry if available)
    # These are gauges, not counters, so we take the after value as
    # representative of steady state
    for queue_name in (
        "mlserver_batch_request_queue",
        "mlserver_parallel_request_queue",
    ):
        value = after.get(queue_name)
        if value is not None:
            attribution[f"{queue_name}_steady"] = round(value, 3)

    # Client-side decomposition
    if client_latencies_ms:
        latencies = np.array(client_latencies_ms)
        attribution["client_p50_ms"] = round(float(np.percentile(latencies, 50)), 2)
        attribution["client_p95_ms"] = round(float(np.percentile(latencies, 95)), 2)
        attribution["client_p99_ms"] = round(float(np.percentile(latencies, 99)), 2)
        attribution["client_mean_ms"] = round(float(np.mean(latencies)), 2)

        # Overhead = client latency - server infer time (approximation:
        # network + serialization + protocol handling)
        server_ms = attribution.get("server_infer_ms_per_req", 0)
        if server_ms > 0:
            attribution["serving_overhead_ms"] = round(
                attribution["client_p50_ms"] - server_ms, 2
            )
            attribution["serving_overhead_pct"] = round(
                (attribution["client_p50_ms"] - server_ms)
                / attribution["client_p50_ms"]
                * 100,
                1,
            )

    # Error counts
    for counter in (
        "mlserver_model_infer_request_success",
        "mlserver_model_infer_request_failure",
    ):
        val = delta(counter)
        if val != 0:
            attribution[counter] = int(val)

    return attribution


async def run_with_load(
    send_fn,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
    metrics_port: int = 8082,
    base_url: str = "http://127.0.0.1",
) -> dict[str, Any]:
    """Run a load window and capture Prometheus deltas around it."""
    baseline = await collect_prometheus_baseline(metrics_port, base_url)

    # Drive load
    from .. import loadgen

    result = await loadgen.closed_loop(concurrency, duration_s, warmup_s, send_fn)

    after = await collect_prometheus_after(metrics_port, base_url)

    summary = result.summary()
    latencies = result.latencies_ms().tolist() if result.samples else []

    attribution = compute_stage_attribution(
        baseline, after, summary.get("ok", 0), latencies
    )
    attribution["load_summary"] = summary
    attribution["prometheus_baseline"] = {
        k: v for k, v in baseline.items() if isinstance(v, (int, float))
    }
    attribution["prometheus_after"] = {
        k: v for k, v in after.items() if isinstance(v, (int, float))
    }

    return attribution


def collect_all(
    metrics_port: int = 8082,
    base_url: str = "http://127.0.0.1",
) -> dict[str, Any]:
    """Collect latency decomposition (without driving load — snapshot only).

    For a full attribution with load, use run_with_load()."""
    baseline = asyncio.run(collect_prometheus_baseline(metrics_port, base_url))
    return {
        "layer": "latency",
        "prometheus_snapshot": {
            k: v for k, v in baseline.items() if isinstance(v, (int, float))
        },
        "timestamp": time.time(),
    }
