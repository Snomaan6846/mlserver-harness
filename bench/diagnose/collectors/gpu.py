"""Layer 6: GPU analysis — DCGM telemetry, SM efficiency, throttling.

Collects GPU metrics via DCGM (dcgmi dmon), nvidia-smi, and
/proc/driver/nvidia. Works on GPU nodes with the NVIDIA driver.
"""

import subprocess
import time
from typing import Any


def _run(command: list[str], timeout: float = 15) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def gpu_available() -> bool:
    """Check if a GPU is present and nvidia-smi works."""
    return bool(_run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]))


def collect_nvidia_smi() -> dict[str, Any]:
    """Full nvidia-smi snapshot: GPU name, driver, utilization, memory,
    temperature, power, clocks, throttle reasons."""
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,driver_version,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
            "clocks.sm,clocks.mem,clocks_throttle_reasons.active,"
            "compute_mode,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {"layer": "gpu", "error": "nvidia-smi unavailable"}

    values = [v.strip() for v in output.split(",")]
    if len(values) < 15:
        return {"layer": "gpu", "error": "unexpected nvidia-smi output"}

    try:
        result: dict[str, Any] = {
            "layer": "gpu",
            "gpu_name": values[0],
            "driver_version": values[1],
            "gpu_utilization_pct": float(values[2]),
            "memory_utilization_pct": float(values[3]),
            "memory_used_mb": float(values[4]),
            "memory_total_mb": float(values[5]),
            "temperature_c": float(values[6]),
            "power_draw_w": float(values[7]),
            "power_limit_w": float(values[8]),
            "sm_clock_mhz": float(values[9]),
            "mem_clock_mhz": float(values[10]),
            "throttle_reasons": values[11],
            "compute_mode": values[12],
            "pstate": values[13],
        }
    except (ValueError, IndexError):
        return {"layer": "gpu", "error": "nvidia-smi parse failed"}

    # Throttle detection
    throttle = values[11]
    if throttle and throttle not in ("0x0", "None", ""):
        result["is_throttled"] = True
        result["throttle_detail"] = _decode_throttle(throttle)
    else:
        result["is_throttled"] = False

    return result


def collect_dcgm(duration_s: float = 10.0, interval_s: float = 1.0) -> dict[str, Any]:
    """DCGM dmon collection: SM activity, memory bandwidth, tensor
    activity, temperature, power — sampled at `interval_s`."""
    output = _run(
        [
            "timeout",
            str(duration_s),
            "dcgmi",
            "dmon",
            "-e",
            "100,101,102,103,104,105,150,155,203,204,205,206,207",
            "-d",
            str(int(interval_s * 1000)),
        ]
    )

    if not output:
        # Fallback: sample nvidia-smi at intervals
        return _sample_nvidia_smi_series(duration_s, interval_s)

    # Parse dcgmi dmon output (CSV-like with headers)
    samples: list[dict[str, float]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("ID") or line.startswith("="):
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                sample: dict[str, float] = {}
                # DCGM field IDs map to names
                field_map = {
                    100: "sm_active_pct",
                    101: "sm_tensor_active_pct",
                    102: "dram_active_pct",
                    103: "fp32_active_pct",
                    104: "fp16_active_pct",
                    105: "pcie_rx_mb",
                    150: "temp_c",
                    155: "power_w",
                    203: "memory_used_mb",
                    204: "memory_free_mb",
                    205: "sm_clock_mhz",
                    206: "mem_clock_mhz",
                    207: "video_clock_mhz",
                }
                for i, val in enumerate(parts):
                    try:
                        # Skip non-numeric (GPU ID, device name)
                        num = float(val)
                        # Assign to field if we have a mapping
                        # (simplified: position-based after ID)
                        if i >= 2 and (i - 2) < len(field_map):
                            field_id = list(field_map.keys())[i - 2]
                            sample[field_map[field_id]] = num
                    except ValueError:
                        continue
                if sample:
                    samples.append(sample)
            except (ValueError, IndexError):
                continue

    if not samples:
        return _sample_nvidia_smi_series(duration_s, interval_s)

    # Compute summary statistics
    summary: dict[str, Any] = {
        "layer": "gpu",
        "source": "dcgm",
        "samples": len(samples),
        "duration_s": duration_s,
    }
    for key in samples[0]:
        values = [s[key] for s in samples if key in s]
        if values:
            summary[f"{key}_mean"] = round(sum(values) / len(values), 2)
            summary[f"{key}_max"] = round(max(values), 2)
            summary[f"{key}_min"] = round(min(values), 2)

    # Detect kernel launch gaps (SM active but no tensor work)
    sm_means = [s.get("sm_active_pct", 0) for s in samples]
    tensor_means = [s.get("sm_tensor_active_pct", 0) for s in samples]
    if sm_means and tensor_means:
        avg_sm = sum(sm_means) / len(sm_means)
        avg_tensor = sum(tensor_means) / len(tensor_means)
        if avg_sm > 50 and avg_tensor < avg_sm * 0.3:
            summary["kernel_gap_detected"] = True
            summary["kernel_gap_note"] = (
                f"SM active {avg_sm:.0f}% but tensor only "
                f"{avg_tensor:.0f}% — possible memory transfer or "
                "kernel launch overhead dominating"
            )
        else:
            summary["kernel_gap_detected"] = False

    return summary


def _sample_nvidia_smi_series(duration_s: float, interval_s: float) -> dict[str, Any]:
    """Fallback: sample nvidia-smi at intervals (no DCGM)."""
    samples: list[dict[str, float]] = []
    elapsed = 0.0
    while elapsed < duration_s:
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu="
                "utilization.gpu,utilization.memory,memory.used,"
                "temperature.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ]
        )
        if output:
            values = [v.strip() for v in output.split(",")]
            try:
                samples.append(
                    {
                        "sm_active_pct": float(values[0]),
                        "dram_active_pct": float(values[1]),
                        "memory_used_mb": float(values[2]),
                        "temp_c": float(values[3]),
                        "power_w": float(values[4]),
                        "sm_clock_mhz": float(values[5]),
                    }
                )
            except (ValueError, IndexError):
                pass
        time.sleep(interval_s)
        elapsed += interval_s

    if not samples:
        return {"layer": "gpu", "error": "no GPU samples collected"}

    summary: dict[str, Any] = {
        "layer": "gpu",
        "source": "nvidia-smi",
        "samples": len(samples),
        "duration_s": duration_s,
    }
    for key in samples[0]:
        values = [s[key] for s in samples if key in s]
        if values:
            summary[f"{key}_mean"] = round(sum(values) / len(values), 2)
            summary[f"{key}_max"] = round(max(values), 2)

    return summary


def _decode_throttle(code: str) -> list[str]:
    """Decode NVIDIA throttle reason bitmask."""
    reasons = []
    try:
        mask = int(code, 16) if code.startswith("0x") else int(code)
    except ValueError:
        return [f"unknown_throttle_code:{code}"]

    throttle_map = {
        0x1: "GPU idle",
        0x2: "Application clock setting",
        0x4: "SW power cap",
        0x8: "HW slowdown",
        0x10: "Sync boost",
        0x20: "SW thermal cap",
        0x40: "HW thermal cap",
        0x80: "HW power brake",
    }
    for bit, reason in throttle_map.items():
        if mask & bit:
            reasons.append(reason)
    return reasons if reasons else [f"unknown_mask:0x{mask:x}"]


def collect_gpu_processes() -> dict[str, Any]:
    """Which processes are using the GPU and how much memory."""
    output = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    processes: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                processes.append(
                    {
                        "pid": int(parts[0]),
                        "process_name": parts[1],
                        "gpu_memory_mb": int(parts[2].rstrip(" MiB")),
                    }
                )
            except (ValueError, IndexError):
                pass
    return {
        "layer": "gpu",
        "gpu_processes": processes,
        "total_gpu_processes": len(processes),
    }


def collect_all(duration_s: float = 10.0) -> dict[str, Any]:
    """Full GPU analysis suite."""
    if not gpu_available():
        return {"layer": "gpu", "error": "no GPU available on this host"}

    results: dict[str, Any] = {"layer": "gpu", "timestamp": time.time()}
    results["nvidia_smi"] = collect_nvidia_smi()
    results["dcgm"] = collect_dcgm(duration_s)
    results["gpu_processes"] = collect_gpu_processes()
    return results
