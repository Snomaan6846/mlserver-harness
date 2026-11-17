"""Layer 2: Process and thread analysis.

Per-thread CPU, thread states, off-CPU analysis, file descriptors,
memory maps, and GIL contention — for the MLServer process (and its
worker processes) running inside a container or on the host.
"""

import os
import subprocess
import time
from typing import Any


def _exec_in_container(container: str | None, command: list[str]) -> str:
    """Run a command inside the target container (or locally if None).
    Returns stdout even on non-zero exit (pgrep returns 1 when no match)."""
    if container:
        full = ["docker", "exec", container] + command
    else:
        full = command
    try:
        result = subprocess.run(full, capture_output=True, text=True, timeout=30)
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def find_server_pids(container: str | None = None) -> list[int]:
    """Find MLServer process PIDs (frontend + workers)."""
    if container:
        # Inside a container: scan /proc for mlserver in cmdline
        # (pgrep may not be available in minimal containers)
        output = _exec_in_container(
            container,
            [
                "sh",
                "-c",
                "for p in $(ls /proc/ 2>/dev/null | grep -E '^[0-9]+$'); "
                "do if grep -q mlserver /proc/$p/cmdline 2>/dev/null; "
                "then echo $p; fi; done",
            ],
        )
    else:
        output = _exec_in_container(
            None,
            ["pgrep", "-f", "mlserver|mlserver_start"],
        )
    return [int(pid) for pid in output.split() if pid.isdigit()]


def collect_process_tree(
    container: str | None = None,
) -> dict[str, Any]:
    """Process hierarchy: frontend + workers with parent/child."""
    output = _exec_in_container(
        container,
        ["ps", "eo", "pid,ppid,pcpu,pmem,nlwp,comm,args", "--no-headers"],
    )
    processes: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(None, 5)
        if len(parts) >= 6:
            try:
                processes.append(
                    {
                        "pid": int(parts[0]),
                        "ppid": int(parts[1]),
                        "cpu_pct": float(parts[2]),
                        "mem_pct": float(parts[3]),
                        "threads": int(parts[4]),
                        "command": parts[5].strip(),
                    }
                )
            except ValueError:
                pass
    return {"layer": "process", "processes": processes}


def collect_thread_states(pid: int, container: str | None = None) -> dict[str, Any]:
    """Per-thread states: running (R), sleeping (S), disk-wait (D),
    zombie (Z), stopped (T). D-state threads indicate I/O blocking."""
    output = _exec_in_container(container, ["cat", f"/proc/{pid}/task/{pid}/status"])
    main_status: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            main_status[key.strip()] = val.strip()

    # Per-thread states
    task_dir = f"/proc/{pid}/task"
    threads: list[dict[str, Any]] = []
    listing = _exec_in_container(container, ["ls", task_dir])
    for tid_str in listing.split():
        if not tid_str.isdigit():
            continue
        tid = int(tid_str)
        stat = _exec_in_container(container, ["cat", f"{task_dir}/{tid_str}/stat"])
        if not stat:
            continue
        parts = stat.split()
        if len(parts) >= 3:
            # State is the third field (after pid and comm in parens)
            state = parts[2]
            threads.append(
                {
                    "tid": tid,
                    "state": state,
                    "comm": parts[1].strip("()") if len(parts) > 1 else "?",
                }
            )

    state_counts: dict[str, int] = {}
    for thread in threads:
        state_counts[thread["state"]] = state_counts.get(thread["state"], 0) + 1

    return {
        "layer": "process",
        "pid": pid,
        "main_thread_status": main_status,
        "thread_count": len(threads),
        "thread_states": state_counts,
        "threads_detail": threads[:100],  # cap for report size
    }


def collect_per_thread_cpu(
    pid: int, container: str | None = None, duration_s: float = 3.0
) -> dict[str, Any]:
    """Per-thread CPU utilization over a sampling window."""
    # First snapshot
    listing = _exec_in_container(container, ["ls", f"/proc/{pid}/task"])
    tids = [int(t) for t in listing.split() if t.isdigit()]
    snapshot1: dict[int, int] = {}
    for tid in tids:
        stat = _exec_in_container(container, ["cat", f"/proc/{pid}/task/{tid}/stat"])
        if stat:
            parts = stat.split()
            if len(parts) > 13:
                try:
                    utime = int(parts[13])
                    stime = int(parts[14])
                    snapshot1[tid] = utime + stime
                except (ValueError, IndexError):
                    pass

    time.sleep(duration_s)

    # Second snapshot
    snapshot2: dict[int, int] = {}
    for tid in tids:
        stat = _exec_in_container(container, ["cat", f"/proc/{pid}/task/{tid}/stat"])
        if stat:
            parts = stat.split()
            if len(parts) > 13:
                try:
                    utime = int(parts[13])
                    stime = int(parts[14])
                    snapshot2[tid] = utime + stime
                except (ValueError, IndexError):
                    pass

    ticks_per_s = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    threads: list[dict[str, Any]] = []
    for tid in sorted(snapshot2):
        delta = snapshot2.get(tid, 0) - snapshot1.get(tid, 0)
        cpu_pct = (delta / ticks_per_s / duration_s) * 100
        if cpu_pct > 0.1:  # only report active threads
            threads.append(
                {
                    "tid": tid,
                    "cpu_pct": round(cpu_pct, 1),
                    "cpu_ticks_delta": delta,
                }
            )

    threads.sort(key=lambda t: t["cpu_pct"], reverse=True)
    return {
        "layer": "process",
        "pid": pid,
        "duration_s": duration_s,
        "active_threads": threads,
        "total_cpu_pct": round(sum(t["cpu_pct"] for t in threads), 1),
    }


def collect_fd_count(pid: int, container: str | None = None) -> dict[str, Any]:
    """File descriptor count vs limits."""
    output = _exec_in_container(
        container, ["sh", "-c", f"ls /proc/{pid}/fd 2>/dev/null | wc -l"]
    )
    fd_count = int(output.strip()) if output.strip().isdigit() else 0

    limits = _exec_in_container(container, ["cat", f"/proc/{pid}/limits"])
    fd_limit = 0
    for line in (limits or "").splitlines():
        if "open files" in line.lower():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    fd_limit = int(parts[3])
                except ValueError:
                    pass

    return {
        "layer": "process",
        "pid": pid,
        "fd_count": fd_count,
        "fd_limit": fd_limit,
        "fd_usage_pct": round(fd_count / fd_limit * 100, 1) if fd_limit else 0,
    }


def collect_memory_map(pid: int, container: str | None = None) -> dict[str, Any]:
    """Memory breakdown from /proc/pid/smaps (RSS/PSS per mapping)."""
    output = _exec_in_container(container, ["cat", f"/proc/{pid}/smaps"])
    if not output:
        return {"layer": "process", "pid": pid, "error": "smaps unreadable"}

    # Parse the summary section (first mapping's header has totals)
    total_rss = 0
    total_pss = 0
    total_shared = 0
    total_private = 0
    mappings: list[dict[str, Any]] = []

    current: dict[str, Any] = {}
    for line in output.splitlines():
        if "-" in line.split()[0] if line.split() else False:
            if current:
                mappings.append(current)
            current = {"mapping": line.strip()[:120]}
        elif ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip().split()[0] if val.strip() else "0"
            try:
                kb = int(val)
            except ValueError:
                continue
            if key == "rss":
                current["rss_kb"] = current.get("rss_kb", 0) + kb
                total_rss += kb
            elif key == "pss":
                current["pss_kb"] = current.get("pss_kb", 0) + kb
                total_pss += kb
            elif "shared" in key:
                total_shared += kb
            elif "private" in key:
                total_private += kb
    if current:
        mappings.append(current)

    # Top mappings by RSS
    mappings.sort(key=lambda m: m.get("rss_kb", 0), reverse=True)

    return {
        "layer": "process",
        "pid": pid,
        "total_rss_kb": total_rss,
        "total_pss_kb": total_pss,
        "total_shared_kb": total_shared,
        "total_private_kb": total_private,
        "top_mappings": [m for m in mappings[:20] if m.get("rss_kb", 0) > 1024],
    }


def collect_offcpu_snapshot(pid: int, container: str | None = None) -> dict[str, Any]:
    """Point-in-time stack dump of all threads — reveals what blocked
    threads are waiting on (locks, I/O, condition variables)."""
    output = _exec_in_container(
        container,
        [
            "sh",
            "-c",
            f"for t in /proc/{pid}/task/*; do "
            f"tid=${{t##*/}}; "
            f"echo === TID $tid ===; "
            f"cat $t/stack 2>/dev/null || cat $t/wchan 2>/dev/null; "
            f"echo; done",
        ],
    )
    if not output:
        return {"layer": "process", "pid": pid, "error": "stack/wchan unreadable"}

    threads: list[dict[str, Any]] = []
    current_tid = None
    current_stack: list[str] = []
    for line in output.splitlines():
        if line.startswith("=== TID "):
            if current_tid is not None:
                threads.append(
                    {
                        "tid": current_tid,
                        "stack": current_stack[:20],
                    }
                )
            parts = line.split()
            try:
                current_tid = int(parts[2]) if len(parts) > 2 else 0
            except (ValueError, IndexError):
                current_tid = 0
            current_stack = []
        elif line.strip():
            current_stack.append(line.strip())
    if current_tid is not None:
        threads.append({"tid": current_tid, "stack": current_stack[:20]})

    return {
        "layer": "process",
        "pid": pid,
        "thread_stacks": threads,
        "blocked_count": sum(
            1
            for t in threads
            if any(
                w in " ".join(t["stack"]).lower()
                for w in ("futex", "wait", "poll", "schedule", "sleep")
            )
        ),
    }


def collect_all(
    container: str | None = None,
    duration_s: float = 3.0,
) -> dict[str, Any]:
    """Collect all process-layer metrics for the MLServer process."""
    pids = find_server_pids(container)
    if not pids:
        return {"layer": "process", "error": "no MLServer processes found"}

    results: dict[str, Any] = {
        "layer": "process",
        "pids": pids,
        "timestamp": time.time(),
    }

    # Main process (first PID = frontend)
    main_pid = pids[0]
    results["process_tree"] = collect_process_tree(container)
    results["thread_states"] = collect_thread_states(main_pid, container)
    results["per_thread_cpu"] = collect_per_thread_cpu(main_pid, container, duration_s)
    results["fd_count"] = collect_fd_count(main_pid, container)
    results["memory_map"] = collect_memory_map(main_pid, container)
    results["offcpu_snapshot"] = collect_offcpu_snapshot(main_pid, container)

    return results
