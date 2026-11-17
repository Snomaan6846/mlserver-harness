"""Findings engine: rule-based automated diagnosis.

Evaluates all collected layers against a rule set and emits
severity-ranked findings with evidence and recommended actions.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Rule definitions: (name, layer, condition_fn, severity, recommendation)
# condition_fn receives the full diagnostic data dict and returns
# (triggered: bool, evidence: str)
# ---------------------------------------------------------------------------

RULES: list[dict[str, Any]] = [
    {
        "name": "frontend_bound_serving",
        "description": "Frontend thread saturated while GPU is idle — "
        "serialization/parsing is the bottleneck, not the model",
        "severity": "HIGH",
        "action": "Enable binary tensor transport (rest_binary_tensor) "
        "and/or worker processing (rest_worker_binary) to "
        "distribute serialization off the event loop",
        "check": lambda d: _check_frontend_bound(d),
    },
    {
        "name": "queue_dominated_latency",
        "description": "More than 50% of server latency is queue wait — "
        "batching or worker dispatch is misconfigured",
        "severity": "HIGH",
        "action": "Tune max_batch_size / max_batch_time for adaptive "
        "batching, or increase parallel_workers",
        "check": lambda d: _check_queue_dominated(d),
    },
    {
        "name": "memory_churn",
        "description": "High allocation rate causing GC pressure — "
        "objects created and destroyed per request",
        "severity": "MEDIUM",
        "action": "Look for Python list/tensor materialization in the "
        "hot path; consider ndarray-native pipeline (roadmap WI-02)",
        "check": lambda d: _check_memory_churn(d),
    },
    {
        "name": "gil_contention",
        "description": "Multi-threaded Python with high GIL contention — "
        "threads fighting for the interpreter lock",
        "severity": "HIGH",
        "action": "Enable worker processing to distribute GIL holders "
        "across processes; reduce thread count in the frontend",
        "check": lambda d: _check_gil_contention(d),
    },
    {
        "name": "gpu_underutilized",
        "description": "GPU SM utilization below 50% — model is too small "
        "for the GPU or the serving path is starved",
        "severity": "MEDIUM",
        "action": "Increase batch size to better utilize the GPU; "
        "verify the frontend is not the bottleneck (see "
        "frontend_bound_serving)",
        "check": lambda d: _check_gpu_underutilized(d),
    },
    {
        "name": "memory_bound_inference",
        "description": "GPU memory bandwidth near saturation — the model "
        "is memory-bound, not compute-bound",
        "severity": "MEDIUM",
        "action": "Consider quantization (FP16/INT8) to reduce memory "
        "traffic, or a model with better arithmetic intensity",
        "check": lambda d: _check_memory_bound(d),
    },
    {
        "name": "thermal_throttling",
        "description": "GPU is actively throttling — performance is being "
        "limited by temperature or power caps",
        "severity": "HIGH",
        "action": "Check cooling, verify power limits are not set "
        "artificially low, consider node rescheduling",
        "check": lambda d: _check_thermal(d),
    },
    {
        "name": "thread_thrashing",
        "description": "Excessive context switches — too many threads "
        "competing for too few cores",
        "severity": "MEDIUM",
        "action": "Reduce thread count (OMP_NUM_THREADS, ORT "
        "intra_op threads); consider CPU affinity pinning",
        "check": lambda d: _check_thread_thrashing(d),
    },
    {
        "name": "fd_exhaustion_risk",
        "description": "File descriptor count approaching the system "
        "limit — connection or file handle leak",
        "severity": "HIGH",
        "action": "Check for connection pool leaks; increase ulimit "
        "if legitimate; verify HTTP client connection reuse",
        "check": lambda d: _check_fd_exhaustion(d),
    },
    {
        "name": "offcpu_blocked",
        "description": "Significant off-CPU time — threads blocked on "
        "locks, I/O, or condition variables",
        "severity": "MEDIUM",
        "action": "Review off-CPU stacks for lock contention or I/O "
        "blocking; consider async I/O or finer-grained locks",
        "check": lambda d: _check_offcpu(d),
    },
    {
        "name": "swap_active",
        "description": "Memory swapping detected — severe performance "
        "degradation from page faults",
        "severity": "CRITICAL",
        "action": "Increase memory limits or reduce model size; "
        "swapping makes all latency measurements unreliable",
        "check": lambda d: _check_swap(d),
    },
    {
        "name": "iowait_heavy",
        "description": "High I/O wait percentage — disk I/O is " "blocking the CPU",
        "severity": "MEDIUM",
        "action": "Check if model loading or logging is hitting disk; "
        "consider tmpfs for hot data",
        "check": lambda d: _check_iowait(d),
    },
]


# ---------------------------------------------------------------------------
# Check functions — each returns (triggered: bool, evidence: str)
# ---------------------------------------------------------------------------


def _check_frontend_bound(data: dict[str, Any]) -> tuple[bool, str]:
    gpu = data.get("gpu", {})
    process = data.get("process", {})
    oncpu = data.get("oncpu", {})

    gpu_util = gpu.get("nvidia_smi", {}).get("gpu_utilization_pct", -1)
    if gpu_util < 0:
        gpu_util = gpu.get("dcgm", {}).get("sm_active_pct_mean", -1)

    thread_cpu = process.get("per_thread_cpu", {})
    total_cpu = thread_cpu.get("total_cpu_pct", 0)

    if gpu_util >= 0 and gpu_util < 50 and total_cpu > 300:
        top = (oncpu.get("top_functions") or [{}])[0]
        top_func = top.get("function", "unknown")
        top_pct = top.get("cpu_pct", 0)
        return True, (
            f"GPU utilization {gpu_util:.0f}% (<50%) while process CPU "
            f"is {total_cpu:.0f}% — top function '{top_func}' at "
            f"{top_pct:.1f}% of CPU samples"
        )
    return False, ""


def _check_queue_dominated(data: dict[str, Any]) -> tuple[bool, str]:
    latency = data.get("latency", {})
    server_ms = latency.get("server_infer_ms_per_req", 0)
    client_p50 = latency.get("client_p50_ms", 0)

    if server_ms > 0 and client_p50 > 0:
        overhead = client_p50 - server_ms
        overhead_pct = overhead / client_p50 * 100
        if overhead_pct > 50:
            return True, (
                f"Client p50 {client_p50:.1f}ms but server infer only "
                f"{server_ms:.1f}ms — {overhead_pct:.0f}% of latency "
                f"is outside the model"
            )
    return False, ""


def _check_memory_churn(data: dict[str, Any]) -> tuple[bool, str]:
    memory = data.get("memory", {})
    rss = memory.get("rss_timeline", {})
    growth = rss.get("rss_growth_rate_kb_per_s", 0)
    gc = memory.get("gc", {}).get("gc_stats", {})

    gc_counts = gc.get("counts", []) if gc else []
    if growth > 1000:  # > 1 MB/s growth
        return True, (
            f"RSS growing at {growth:.0f} KB/s — allocation churn "
            f"causing GC pressure (GC counts: {gc_counts})"
        )
    return False, ""


def _check_gil_contention(data: dict[str, Any]) -> tuple[bool, str]:
    process = data.get("process", {})
    thread_cpu = process.get("per_thread_cpu", {})
    threads = thread_cpu.get("active_threads", [])
    total_cpu = thread_cpu.get("total_cpu_pct", 0)

    system = data.get("system", {}).get("cpu", {})
    ctx_switches = system.get("context_switches_per_s", 0)

    # If we have many active Python threads but total CPU doesn't scale
    if len(threads) > 4 and total_cpu < len(threads) * 80 and ctx_switches > 50000:
        return True, (
            f"{len(threads)} active threads but only {total_cpu:.0f}% "
            f"CPU used with {ctx_switches:.0f} ctx switches/s — "
            f"GIL contention limiting parallelism"
        )
    return False, ""


def _check_gpu_underutilized(data: dict[str, Any]) -> tuple[bool, str]:
    gpu = data.get("gpu", {})
    sm_util = gpu.get("nvidia_smi", {}).get("gpu_utilization_pct", -1)
    if sm_util < 0:
        sm_util = gpu.get("dcgm", {}).get("sm_active_pct_mean", -1)

    if 0 <= sm_util < 50:
        tensor = gpu.get("dcgm", {}).get("sm_tensor_active_pct_mean", -1)
        detail = f", tensor active {tensor:.0f}%" if tensor >= 0 else ""
        return True, f"GPU SM utilization {sm_util:.0f}% (<50%){detail}"
    return False, ""


def _check_memory_bound(data: dict[str, Any]) -> tuple[bool, str]:
    dcgm = data.get("gpu", {}).get("dcgm", {})
    dram = dcgm.get("dram_active_pct_mean", -1)
    sm = dcgm.get("sm_active_pct_mean", -1)

    if dram > 80 and 0 <= sm < 80:
        return True, (
            f"GPU memory bandwidth {dram:.0f}% (>80%) while SM only "
            f"{sm:.0f}% — memory-bound inference"
        )
    return False, ""


def _check_thermal(data: dict[str, Any]) -> tuple[bool, str]:
    smi = data.get("gpu", {}).get("nvidia_smi", {})
    throttled = smi.get("is_throttled", False)
    throttle_detail = smi.get("throttle_detail", [])

    if throttled:
        return True, f"GPU throttling active: {throttle_detail}"
    return False, ""


def _check_thread_thrashing(data: dict[str, Any]) -> tuple[bool, str]:
    system = data.get("system", {}).get("cpu", {})
    ctx_switches = system.get("context_switches_per_s", 0)
    cores = system.get("cores", 1)

    if ctx_switches > 100000 and cores > 0:
        return True, (
            f"{ctx_switches:.0f} context switches/s across {cores} "
            f"cores ({ctx_switches / cores:.0f}/core/s) — thread "
            f"thrashing"
        )
    return False, ""


def _check_fd_exhaustion(data: dict[str, Any]) -> tuple[bool, str]:
    fd = data.get("process", {}).get("fd_count", {})
    usage_pct = fd.get("fd_usage_pct", 0)
    count = fd.get("fd_count", 0)
    limit = fd.get("fd_limit", 0)

    if usage_pct > 80:
        return True, (
            f"File descriptors at {usage_pct:.0f}% of limit " f"({count}/{limit})"
        )
    return False, ""


def _check_offcpu(data: dict[str, Any]) -> tuple[bool, str]:
    offcpu = data.get("process", {}).get("offcpu_snapshot", {})
    blocked = offcpu.get("blocked_count", 0)
    total_threads = (
        data.get("process", {}).get("thread_states", {}).get("thread_count", 0)
    )

    if total_threads > 0 and blocked > total_threads * 0.3:
        return True, (
            f"{blocked}/{total_threads} threads blocked (off-CPU) — "
            f"significant waiting on locks or I/O"
        )
    return False, ""


def _check_swap(data: dict[str, Any]) -> tuple[bool, str]:
    memory = data.get("system", {}).get("memory", {})
    swap_pct = memory.get("swap_used_pct", 0)

    if swap_pct > 10:
        return True, f"Swap usage at {swap_pct:.0f}% — memory pressure"
    return False, ""


def _check_iowait(data: dict[str, Any]) -> tuple[bool, str]:
    cpu = data.get("system", {}).get("cpu", {})
    iowait = cpu.get("cpu_iowait_pct", 0)

    if iowait > 10:
        return True, f"CPU iowait at {iowait:.1f}% (>10%) — disk blocking"
    return False, ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    """Run all rules against the collected diagnostic data."""
    findings: list[dict[str, Any]] = []
    for rule in RULES:
        try:
            triggered, evidence = rule["check"](data)
        except Exception:
            triggered, evidence = False, ""
        if triggered:
            findings.append(
                {
                    "finding": rule["name"],
                    "description": rule["description"],
                    "severity": rule["severity"],
                    "evidence": evidence,
                    "recommended_action": rule["action"],
                }
            )

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 99))

    return {
        "findings": findings,
        "total": len(findings),
        "by_severity": {
            level: sum(1 for f in findings if f["severity"] == level)
            for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        },
    }
