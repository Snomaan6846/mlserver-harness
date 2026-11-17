"""Load generation: closed-loop concurrency, open-loop arrival rates.

Closed-loop: N async workers each send back-to-back (classic saturation
sweep). Open-loop: requests are issued on a fixed schedule (constant
interval or Poisson/exponential inter-arrival) regardless of completion —
the anti-coordinated-omission pattern that measures true queueing delay.

Every request records (start_ns, end_ns, status, bytes). Percentiles via
numpy on the measured window only.
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Sample:
    start_ns: int
    end_ns: int
    ok: bool
    bytes_out: int
    bytes_in: int
    status: str = ""


@dataclass
class LoadResult:
    samples: list[Sample] = field(default_factory=list)
    wall_seconds: float = 0.0
    target_rate: float | None = None
    mode: str = "closed"

    # -- metrics ----------------------------------------------------------

    def latencies_ms(self, window: tuple[float, float] | None = None) -> np.ndarray:
        values = []
        for sample in self.samples:
            if not sample.ok:
                continue
            if window is not None and not _in_window(sample, window):
                continue
            values.append((sample.end_ns - sample.start_ns) / 1e6)
        return np.asarray(values)

    def summary(self, window: tuple[float, float] | None = None) -> dict[str, Any]:
        """Summarise results.

        With ``window`` (seconds relative to run start), ONLY samples
        whose start falls inside the window are counted — requests,
        ok/failed, throughput and byte totals all refer to the window.
        Without a window, the whole run (warmup included) is summarised.
        """
        if window is not None:
            samples = [s for s in self.samples if _in_window(s, window)]
        else:
            samples = self.samples
        latencies = self.latencies_ms(window)
        ok = sum(1 for s in samples if s.ok)
        failed = len(samples) - ok
        seconds = window[1] - window[0] if window else self.wall_seconds
        out: dict[str, Any] = {
            "mode": self.mode,
            "target_rate": self.target_rate,
            "requests": len(samples),
            "ok": ok,
            "failed": failed,
            "error_rate": (failed / len(samples)) if samples else 0.0,
            "throughput_rps": ok / seconds if seconds > 0 else 0.0,
        }
        if latencies.size:
            percentiles = np.percentile(latencies, [50, 90, 95, 99, 99.9])
            out.update(
                {
                    "latency_ms_min": float(latencies.min()),
                    "latency_ms_mean": float(latencies.mean()),
                    "latency_ms_stddev": float(latencies.std()),
                    "latency_ms_p50": float(percentiles[0]),
                    "latency_ms_p90": float(percentiles[1]),
                    "latency_ms_p95": float(percentiles[2]),
                    "latency_ms_p99": float(percentiles[3]),
                    "latency_ms_p999": float(percentiles[4]),
                    "latency_ms_max": float(latencies.max()),
                    "bytes_out": sum(s.bytes_out for s in samples),
                    "bytes_in": sum(s.bytes_in for s in samples),
                }
            )
        else:
            out.update(
                {
                    "bytes_out": sum(s.bytes_out for s in samples),
                    "bytes_in": sum(s.bytes_in for s in samples),
                }
            )
        # Client-CPU guard annotations (set by run_closed)
        client_cpu = getattr(self, "client_cpu_percent", None)
        if client_cpu is not None:
            out["client_cpu_percent"] = client_cpu
            if getattr(self, "generator_bound", False):
                out["generator_bound"] = True
        return out


def _in_window(sample: "Sample", window: tuple[float, float]) -> bool:
    """Window membership by request START time (arrival), in seconds
    relative to the run's t0."""
    seconds = (sample.start_ns - sample._start_ref) / 1e9
    return window[0] <= seconds <= window[1]


# Attach a wall-clock reference for windowing. t0_ns MUST be
# time.monotonic_ns() — a float-seconds value loses nanosecond
# precision at monotonic-clock magnitudes (float64 exact integers stop
# at ~2^53 ns ≈ 104 days of clock), which made every window filter
# silently match nothing (the vacuous-stability-gate bug).
def _with_ref(samples: list[Sample], t0_ns: int) -> list[Sample]:
    for sample in samples:
        sample._start_ref = t0_ns  # type: ignore[attr-defined]
    return samples


# ---------------------------------------------------------------------------
# Senders (protocol-agnostic: caller supplies an async send function)
# ---------------------------------------------------------------------------

SendFn = Callable[[int], "asyncio.Future[tuple[bool, int, int]]"]
"""async send(seq) -> (ok, bytes_out, bytes_in)"""


async def closed_loop(
    concurrency: int,
    seconds: float,
    warmup_seconds: float,
    send: SendFn,
    error_threshold: float = 0.001,
) -> LoadResult:
    """Fixed concurrent clients, back-to-back requests."""
    t0 = time.monotonic()
    t0_ns = time.monotonic_ns()
    samples: list[Sample] = []
    lock = asyncio.Lock()
    stop_at = t0 + warmup_seconds + seconds

    async def worker(worker_id: int) -> None:
        seq = worker_id
        while time.monotonic() < stop_at:
            started = time.monotonic_ns()
            try:
                ok, bytes_out, bytes_in = await send(seq)
                status = ""
            except Exception as error:
                ok, bytes_out, bytes_in = False, 0, 0
                status = str(error)[:120]
            ended = time.monotonic_ns()
            async with lock:
                samples.append(Sample(started, ended, ok, bytes_out, bytes_in, status))
            seq += concurrency

    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*workers)
    wall = time.monotonic() - t0
    result = LoadResult(samples=samples, wall_seconds=wall, mode="closed")
    _with_ref(result.samples, t0_ns)
    _error_guard(result, error_threshold)
    return result


async def open_loop(
    target_qps: float,
    seconds: float,
    warmup_seconds: float,
    send: SendFn,
    poisson: bool = True,
    error_threshold: float = 0.001,
    max_inflight: int = 100_000,
) -> LoadResult:
    """Requests issued on a schedule regardless of completion.

    Constant rate (deterministic interval) or Poisson (exponential
    inter-arrival, seeded). Each request's latency therefore includes the
    queueing delay the server actually caused — coordinated omission is
    eliminated by construction.
    """
    t0 = time.monotonic()
    t0_ns = time.monotonic_ns()
    samples: list[Sample] = []
    lock = asyncio.Lock()
    rng = random.Random(20261114)
    stop_at = t0 + warmup_seconds + seconds
    inflight: set[asyncio.Task] = set()

    async def one(seq: int, scheduled_ns: int) -> None:
        # Latency is measured from the SCHEDULED send time (wrk2
        # method), not the actual task start: if the client itself is
        # late (loop lag) or the server backs up, that delay belongs
        # to the request's measured latency. Coordinated omission is
        # eliminated by construction.
        started = scheduled_ns
        try:
            ok, bytes_out, bytes_in = await send(seq)
            status = ""
        except Exception as error:
            ok, bytes_out, bytes_in = False, 0, 0
            status = str(error)[:120]
        ended = time.monotonic_ns()
        async with lock:
            samples.append(Sample(started, ended, ok, bytes_out, bytes_in, status))

    next_due_ns = t0_ns
    interval_ns = 1e9 / target_qps
    seq = 0
    while True:
        now = time.monotonic()
        if now >= stop_at:
            break
        delay_ns = (
            -math.log(1 - rng.random()) / target_qps * 1e9 if poisson else interval_ns
        )
        next_due_ns += delay_ns
        due_in_s = (next_due_ns - time.monotonic_ns()) / 1e9
        if due_in_s > 0.001:
            await asyncio.sleep(min(due_in_s, stop_at - now))
        if time.monotonic() >= stop_at:
            break
        task = asyncio.create_task(one(seq, next_due_ns))
        inflight.add(task)
        task.add_done_callback(inflight.discard)
        if len(inflight) > max_inflight:
            raise RuntimeError(
                "open-loop inflight cap exceeded — target QPS is far past "
                "saturation; aborting to avoid unbounded queue growth"
            )
        seq += 1

    if inflight:
        await asyncio.gather(*inflight)
    wall = time.monotonic() - t0
    result = LoadResult(
        samples=samples,
        wall_seconds=wall,
        target_rate=target_qps,
        mode="poisson" if poisson else "constant",
    )
    _with_ref(result.samples, t0_ns)
    _error_guard(result, error_threshold)
    return result


def _error_guard(result: LoadResult, threshold: float) -> None:
    if result.samples:
        rate = sum(1 for s in result.samples if not s.ok) / len(result.samples)
        if rate > threshold:
            raise RuntimeError(
                f"error rate {rate:.2%} exceeds threshold {threshold:.2%} — "
                f"aborting (first error: "
                f"{next((s.status for s in result.samples if not s.ok), '?')})"
            )


# ---------------------------------------------------------------------------
# Stability gate (Triton perf_analyzer style sliding windows)
# ---------------------------------------------------------------------------


def stability_gate(
    result: LoadResult,
    warmup_seconds: float,
    window_seconds: float = 5.0,
    tolerance: float = 0.05,
    min_windows: int = 2,
) -> dict[str, Any]:
    """Split the measured window into consecutive slices; throughput and
    p95 must stabilize within +/-tolerance across windows."""
    windows = []
    t0 = warmup_seconds
    while t0 + window_seconds <= result.wall_seconds:
        stats = result.summary((t0, t0 + window_seconds))
        if stats.get("ok"):
            windows.append(stats)
        t0 += window_seconds
    if len(windows) < min_windows:
        return {
            "stable": False,
            "reason": "insufficient windows",
            "windows": len(windows),
        }
    throughputs = [w["throughput_rps"] for w in windows]
    p95s = [w.get("latency_ms_p95", 0.0) for w in windows]
    tp_spread = (max(throughputs) - min(throughputs)) / max(
        np.median(throughputs), 1e-9
    )
    p95_spread = (max(p95s) - min(p95s)) / max(np.median(p95s), 1e-9)
    # bool(): numpy comparisons yield np.bool_, which json.dumps
    # rejects ("Object of type bool_ is not JSON serializable")
    stable = bool(tp_spread <= tolerance and p95_spread <= tolerance)
    return {
        "stable": stable,
        "throughput_spread": float(tp_spread),
        "p95_spread": float(p95_spread),
        "tolerance": tolerance,
        "windows": [
            {"t": index, "rps": w["throughput_rps"], "p95_ms": w.get("latency_ms_p95")}
            for index, w in enumerate(windows)
        ],
    }
