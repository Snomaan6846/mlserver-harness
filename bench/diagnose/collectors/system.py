"""Layer 1: System-level resource analysis (USE Method — Brendan Gregg).

For every resource (CPU, memory, network, storage, GPU): check
Utilization, Saturation, and Errors. Runs OUTSIDE the target container
or inside it via exec — collects from /proc, /sys, and standard tools.
"""

import subprocess
import time
from pathlib import Path
from typing import Any


def _run(command: list[str], timeout: float = 10) -> str:
    """Run a command and return stdout; empty string on failure."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _read(path: str) -> str:
    """Read a /proc or /sys file; empty string on failure."""
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return ""


# ---------------------------------------------------------------------------
# Collectors — each returns a dict of parsed metrics
# ---------------------------------------------------------------------------


def collect_cpu(duration_s: float = 5.0) -> dict[str, Any]:
    """CPU utilization, saturation (load vs cores, PSI), and per-core."""
    out: dict[str, Any] = {"layer": "system", "resource": "cpu"}

    # Load average vs core count
    loadavg = _read("/proc/loadavg")
    if loadavg:
        parts = loadavg.split()
        out["load_1m"] = float(parts[0])
        out["load_5m"] = float(parts[1])
        out["load_15m"] = float(parts[2])

    cores = _read("/proc/cpuinfo")
    core_count = cores.count("processor") if cores else 0
    out["cores"] = core_count

    # CPU time breakdown (vmstat-style from /proc/stat)
    stat = _read("/proc/stat")
    if stat:
        first = stat.splitlines()[0].split()
        if len(first) >= 5:
            values = [int(v) for v in first[1:]]
            total = sum(values)
            out["cpu_user_pct"] = round(values[0] / total * 100, 1)
            out["cpu_system_pct"] = round(values[2] / total * 100, 1)
            out["cpu_idle_pct"] = round(values[3] / total * 100, 1)
            out["cpu_iowait_pct"] = round(values[4] / total * 100, 1)
            if len(values) > 7:
                out["cpu_steal_pct"] = round(values[7] / total * 100, 1)

    # Context switches (from /proc/stat second line)
    for line in (stat or "").splitlines():
        if line.startswith("ctxt "):
            out["context_switches_total"] = int(line.split()[1])
        if line.startswith("procs_running "):
            out["procs_running"] = int(line.split()[1])
        if line.startswith("procs_blocked "):
            out["procs_blocked"] = int(line.split()[1])

    # PSI (Pressure Stall Info) — the modern saturation signal
    psi_cpu = _read("/proc/pressure/cpu")
    if psi_cpu:
        out["psi_cpu"] = _parse_psi(psi_cpu)

    # Delta over `duration_s` for rate metrics
    if duration_s > 0:
        ctxt1 = out.get("context_switches_total", 0)
        time.sleep(duration_s)
        stat2 = _read("/proc/stat")
        for line in (stat2 or "").splitlines():
            if line.startswith("ctxt "):
                ctxt2 = int(line.split()[1])
                out["context_switches_per_s"] = int((ctxt2 - ctxt1) / duration_s)

    return out


def collect_memory() -> dict[str, Any]:
    """Memory utilization, saturation (PSI, swap), and breakdown."""
    out: dict[str, Any] = {"layer": "system", "resource": "memory"}

    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":").lower()
                try:
                    out[key + "_kb"] = int(parts[1])
                except ValueError:
                    pass

    # Derived percentages
    total = out.get("memtotal_kb", 0)
    available = out.get("memavailable_kb", 0)
    if total > 0:
        out["memory_used_pct"] = round((total - available) / total * 100, 1)

    swap_total = out.get("swaptotal_kb", 0)
    swap_free = out.get("swapfree_kb", 0)
    if swap_total > 0:
        out["swap_used_pct"] = round((swap_total - swap_free) / swap_total * 100, 1)

    psi_mem = _read("/proc/pressure/memory")
    if psi_mem:
        out["psi_memory"] = _parse_psi(psi_mem)

    return out


def collect_io() -> dict[str, Any]:
    """Storage I/O utilization and saturation."""
    out: dict[str, Any] = {"layer": "system", "resource": "io"}

    psi_io = _read("/proc/pressure/io")
    if psi_io:
        out["psi_io"] = _parse_psi(psi_io)

    # /proc/diskstats for aggregate I/O
    diskstats = _read("/proc/diskstats")
    if diskstats:
        total_reads = 0
        total_writes = 0
        total_io_ms = 0
        for line in diskstats.splitlines():
            parts = line.split()
            if len(parts) >= 13:
                # Skip loop/ram devices
                device = parts[2]
                if device.startswith(("loop", "ram", "sr")):
                    continue
                total_reads += int(parts[3])
                total_writes += int(parts[7])
                total_io_ms += int(parts[12])
        out["disk_reads_total"] = total_reads
        out["disk_writes_total"] = total_writes
        out["disk_io_ms_total"] = total_io_ms

    return out


def collect_network() -> dict[str, Any]:
    """Network utilization and errors."""
    out: dict[str, Any] = {"layer": "system", "resource": "network"}

    netstat = _read("/proc/net/netstat")
    if netstat:
        lines = netstat.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("TcpExt:"):
                header = line.split()[1:]
                if i + 1 < len(lines):
                    values = lines[i + 1].split()[1:]
                    for key, val in zip(header, values):
                        try:
                            out[f"tcp_{key.lower()}"] = int(val)
                        except ValueError:
                            pass

    # /proc/net/snmp for retransmits
    snmp = _read("/proc/net/snmp")
    if snmp:
        lines = snmp.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Tcp:") and i + 1 < len(lines):
                header = line.split()[1:]
                values = lines[i + 1].split()[1:]
                for key, val in zip(header, values):
                    if key in ("RetransSegs", "OutSegs", "InErrs", "OutRsts"):
                        try:
                            out[f"tcp_{key.lower()}"] = int(val)
                        except ValueError:
                            pass

    return out


def collect_interrupts() -> dict[str, Any]:
    """Interrupt and softirq activity."""
    out: dict[str, Any] = {"layer": "system", "resource": "interrupts"}

    interrupts = _read("/proc/interrupts")
    if interrupts:
        lines = interrupts.splitlines()
        if lines:
            # First line has the CPU column headers
            cpu_count = len(lines[0].split())
            out["cpu_count_irq"] = cpu_count
            total_irq = 0
            for line in lines[1:]:
                parts = line.split()
                if parts and parts[0].rstrip(":").isdigit():
                    for val in parts[1 : 1 + cpu_count]:
                        try:
                            total_irq += int(val)
                        except ValueError:
                            pass
            out["total_interrupts"] = total_irq

    softirq = _read("/proc/softirqs")
    if softirq:
        lines = softirq.splitlines()
        totals: dict[str, int] = {}
        for line in lines[1:]:
            parts = line.split()
            if parts and parts[0].rstrip(":").isdigit():
                name = parts[1].rstrip(":")
                try:
                    totals[name] = sum(int(p) for p in parts[2:] if p.isdigit())
                except ValueError:
                    pass
        out["softirq_totals"] = totals

    return out


def collect_thermal() -> dict[str, Any]:
    """CPU thermal zones (if available)."""
    out: dict[str, Any] = {"layer": "system", "resource": "thermal"}
    zones: list[dict[str, Any]] = []
    thermal_dir = Path("/sys/class/thermal")
    if thermal_dir.is_dir():
        for zone in sorted(thermal_dir.glob("thermal_zone*")):
            temp_file = zone / "temp"
            type_file = zone / "type"
            if temp_file.is_file():
                try:
                    temp = int(temp_file.read_text().strip()) / 1000.0
                    zone_type = (
                        type_file.read_text().strip()
                        if type_file.is_file()
                        else "unknown"
                    )
                    zones.append({"zone": zone.name, "type": zone_type, "temp_c": temp})
                except (OSError, ValueError):
                    pass
    out["thermal_zones"] = zones
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_psi(text: str) -> dict[str, float]:
    """Parse PSI output: 'some avg10=0.00 avg60=0.00 avg300=0.00 total=0'"""
    result: dict[str, float] = {}
    for line in text.splitlines():
        kind = line.split()[0] if line.split() else ""
        for part in line.split():
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    result[f"{kind}_{key}"] = float(val)
                except ValueError:
                    pass
    return result


def collect_all(duration_s: float = 5.0) -> dict[str, Any]:
    """Collect all system-layer metrics."""
    return {
        "cpu": collect_cpu(duration_s),
        "memory": collect_memory(),
        "io": collect_io(),
        "network": collect_network(),
        "interrupts": collect_interrupts(),
        "thermal": collect_thermal(),
        "timestamp": time.time(),
    }
