"""Layer 4: Memory profiling (memray).

Allocation sites, peak memory timeline, allocation churn rates,
and leak detection via snapshot diffing.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _memray_available(container: str | None = None) -> bool:
    cmd = (["docker", "exec", container] if container else []) + ["memray", "--version"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_memray(container: str | None = None) -> bool:
    if _memray_available(container):
        return True
    cmd = (["docker", "exec", container] if container else []) + [
        "pip",
        "install",
        "--quiet",
        "memray",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def attach_and_record(
    pid: int,
    container: str | None = None,
    duration_s: float = 30.0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach memray to a running process and record allocations.

    Note: memray attach requires the process to have been started
    with memray (or uses the live mode via `memray attach` which
    uses the BPF/perf_events backend). For containers, we exec
    memray in run mode alongside the target.
    """
    if not install_memray(container):
        return {
            "layer": "memory",
            "error": "memray not available and could not be installed",
        }

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="memray-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = ["docker", "exec", container] if container else []
    output_file = output_dir / f"memray_{pid}.bin"

    # memray attach mode (for running processes)
    cmd = prefix + [
        "timeout",
        str(duration_s),
        "memray",
        "attach",
        "--pid",
        str(pid),
        "--output",
        str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=duration_s + 60
        )
        if result.returncode not in (0, 124):  # 124 = timeout (expected)
            return {
                "layer": "memory",
                "error": f"memray attach failed: {result.stderr[:300]}",
                "pid": pid,
            }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {
            "layer": "memory",
            "error": "memray attach timed out",
            "pid": pid,
        }

    return {
        "layer": "memory",
        "pid": pid,
        "duration_s": duration_s,
        "output_file": str(output_file),
        "note": "run 'memray flamegraph <output_file>' to visualize",
    }


def snapshot_rss(
    container: str | None = None,
    pid: int | None = None,
    interval_s: float = 1.0,
    count: int = 30,
) -> dict[str, Any]:
    """Sample RSS over time — a lightweight memory timeline without
    memray overhead. Uses /proc/PID/status."""
    prefix = ["docker", "exec", container] if container else []
    samples: list[dict[str, Any]] = []

    for _ in range(count):
        if pid:
            cmd = prefix + ["cat", f"/proc/{pid}/status"]
        else:
            cmd = prefix + [
                "sh",
                "-c",
                "cat /proc/$(pgrep -f mlserver | head -1)/status",
            ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                rss_kb = 0
                for line in result.stdout.splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
                samples.append(
                    {
                        "timestamp": time.time(),
                        "rss_kb": rss_kb,
                    }
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(interval_s)

    if not samples:
        return {"layer": "memory", "error": "could not sample RSS"}

    rss_values = [s["rss_kb"] for s in samples]
    first_rss = rss_values[0]
    last_rss = rss_values[-1]
    min_rss = min(rss_values)
    max_rss = max(rss_values)

    return {
        "layer": "memory",
        "pid": pid,
        "samples": len(samples),
        "interval_s": interval_s,
        "rss_first_kb": first_rss,
        "rss_last_kb": last_rss,
        "rss_min_kb": min_rss,
        "rss_max_kb": max_rss,
        "rss_delta_kb": last_rss - first_rss,
        "rss_growth_rate_kb_per_s": (
            (last_rss - first_rss) / (len(samples) * interval_s)
            if len(samples) > 1
            else 0
        ),
        "timeline": [
            {
                "t": round(s["timestamp"] - samples[0]["timestamp"], 1),
                "rss_kb": s["rss_kb"],
            }
            for s in samples
        ],
    }


def collect_gc_stats(container: str | None = None) -> dict[str, Any]:
    """Python garbage collection statistics (if the server exposes them)."""
    prefix = ["docker", "exec", container] if container else []
    cmd = prefix + [
        "python3",
        "-c",
        "import gc; import json; "
        "print(json.dumps({"
        "'counts': list(gc.get_count()),"
        "'thresholds': list(gc.get_thresholds()),"
        "'stats': [dict(s) for s in gc.get_stats()]"
        "}))",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return {
                "layer": "memory",
                "gc_stats": json.loads(result.stdout.strip()),
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return {"layer": "memory", "gc_stats": None}


def collect_all(
    container: str | None = None,
    pid: int | None = None,
    duration_s: float = 30.0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Full memory profiling suite."""
    results: dict[str, Any] = {"layer": "memory", "timestamp": time.time()}

    # RSS timeline (always collect — no overhead)
    results["rss_timeline"] = snapshot_rss(
        container, pid, interval_s=1.0, count=min(duration_s, 60)
    )

    # GC stats (cheap)
    results["gc"] = collect_gc_stats(container)

    # Full memray profile (higher overhead — opt-in via duration > 0)
    if duration_s > 0 and pid:
        results["memray"] = attach_and_record(pid, container, duration_s, output_dir)

    return results
