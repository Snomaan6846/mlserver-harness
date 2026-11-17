"""Layer 3: On-CPU code profiling (py-spy).

Sampling profiles with native + Python frames, flame graphs,
top-N attribution. Attaches to the running MLServer process.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _pyspy_available(container: str | None = None) -> bool:
    """Check if py-spy is available in the target."""
    cmd = (["docker", "exec", container] if container else []) + ["py-spy", "--version"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_pyspy(container: str | None = None) -> bool:
    """Install py-spy into the target container if not present."""
    if _pyspy_available(container):
        return True
    cmd = (["docker", "exec", container] if container else []) + [
        "pip",
        "install",
        "--quiet",
        "py-spy",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def record_profile(
    pid: int,
    container: str | None = None,
    duration_s: float = 30.0,
    rate: int = 100,
    native: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach py-spy and record a sampling profile.

    Returns flame graph file path + top-N function attribution.
    """
    if not install_pyspy(container):
        return {
            "layer": "oncpu",
            "error": "py-spy not available and could not be installed",
        }

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="pyspy-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = ["docker", "exec", container] if container else []
    svg_path = output_dir / f"flame_{pid}.svg"
    raw_path = output_dir / f"flame_{pid}.txt"

    # Record with native frames for the flame graph
    cmd = prefix + [
        "py-spy",
        "record",
        "--pid",
        str(pid),
        "--duration",
        str(duration_s),
        "--rate",
        str(rate),
        "--format",
        "flamegraph",
        "--output",
        str(svg_path),
    ]
    if native:
        cmd.insert(-4, "--native")

    try:
        subprocess.run(cmd, capture_output=True, timeout=duration_s + 60)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"layer": "oncpu", "error": "py-spy record timed out"}

    # Also record collapsed stacks for programmatic analysis
    cmd = prefix + [
        "py-spy",
        "record",
        "--pid",
        str(pid),
        "--duration",
        str(duration_s),
        "--rate",
        str(rate),
        "--format",
        "raw",
        "--output",
        str(raw_path),
    ]
    if native:
        cmd.insert(-4, "--native")

    try:
        subprocess.run(cmd, capture_output=True, timeout=duration_s + 60)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Parse the raw output for top-N attribution
    top_functions = _parse_collapsed(raw_path)

    return {
        "layer": "oncpu",
        "pid": pid,
        "duration_s": duration_s,
        "rate_hz": rate,
        "native": native,
        "flame_graph": str(svg_path),
        "raw_stacks": str(raw_path),
        "top_functions": top_functions,
        "total_samples": sum(f["samples"] for f in top_functions),
    }


def dump_stacks(pid: int, container: str | None = None) -> dict[str, Any]:
    """Point-in-time stack dump — 'what is it doing RIGHT NOW'."""
    if not install_pyspy(container):
        return {"layer": "oncpu", "error": "py-spy not available"}

    prefix = ["docker", "exec", container] if container else []
    cmd = prefix + ["py-spy", "dump", "--pid", str(pid)]
    if True:  # always include native frames for dump
        cmd.insert(-2, "--native")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {
                "layer": "oncpu",
                "pid": pid,
                "stacks": result.stdout[:10000],  # cap for report
            }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return {"layer": "oncpu", "error": "py-spy dump failed"}


def _parse_collapsed(raw_path: Path) -> list[dict[str, Any]]:
    """Parse py-spy's raw/collapsed stack format into top-N functions.

    Format: 'frame1;frame2;frame3 count'
    """
    if not raw_path.is_file():
        return []

    function_totals: dict[str, int] = {}
    total_samples = 0

    try:
        text = raw_path.read_text()
    except OSError:
        return []

    for line in text.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            count = int(parts[1])
        except ValueError:
            continue
        total_samples += count
        frames = parts[0].split(";")
        # Track each unique frame's cumulative contribution
        seen = set()
        for frame in frames:
            # Simplify: function name (strip file:line)
            func = frame.split("(")[0].split(" ")[0] if "(" in frame else frame
            if func not in seen:  # avoid double-counting leaf vs total
                function_totals[func] = function_totals.get(func, 0) + count
                seen.add(func)

    if total_samples == 0:
        return []

    # Sort by sample count, compute percentage
    ranked = sorted(function_totals.items(), key=lambda x: x[1], reverse=True)
    top_n: list[dict[str, Any]] = []
    for func, count in ranked[:50]:  # top 50
        top_n.append(
            {
                "function": func[:120],
                "samples": count,
                "cpu_pct": round(count / total_samples * 100, 2),
            }
        )
    return top_n


def collect_all(
    pid: int,
    container: str | None = None,
    duration_s: float = 30.0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Full on-CPU profiling suite."""
    result = record_profile(pid, container, duration_s, output_dir=output_dir)
    dump = dump_stacks(pid, container)
    result["point_in_time_dump"] = dump
    return result
