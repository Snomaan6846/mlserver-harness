"""DIAGNOSIS.md report generator — the reviewable engineering report."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_report(
    data: dict[str, Any],
    findings: dict[str, Any],
    output_dir: Path,
    reference: dict[str, Any] | None = None,
) -> Path:
    """Generate DIAGNOSIS.md from collected data + findings."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Header
    lines.append("# MLServer Diagnostic Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(
        f"**Layers collected:** "
        f"{', '.join(sorted(set(k for k in data if k != 'timestamp')))}"
    )
    lines.append(
        f"**Findings:** {findings['total']} "
        f"({findings['by_severity'].get('CRITICAL', 0)} critical, "
        f"{findings['by_severity'].get('HIGH', 0)} high, "
        f"{findings['by_severity'].get('MEDIUM', 0)} medium)"
    )
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    summary = _executive_summary(data, findings, reference)
    lines.append(summary)
    lines.append("")

    # Findings
    if findings["findings"]:
        lines.append("## Findings")
        lines.append("")
        for i, f in enumerate(findings["findings"], 1):
            severity_icon = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🟢",
            }.get(f["severity"], "⚪")
            lines.append(
                f"### {i}. {severity_icon} [{f['severity']}] " f"{f['finding']}"
            )
            lines.append("")
            lines.append(f"**What:** {f['description']}")
            lines.append("")
            lines.append(f"**Evidence:** {f['evidence']}")
            lines.append("")
            lines.append(f"**Recommended action:** {f['recommended_action']}")
            lines.append("")
    else:
        lines.append("## Findings")
        lines.append("")
        lines.append(
            "No automated findings triggered. " "Review the raw layer data below."
        )
        lines.append("")

    # Latency attribution
    latency = data.get("latency", {})
    if latency.get("server_infer_ms_per_req"):
        lines.append("## Latency Attribution")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(
            f"| Server infer (per request) | "
            f"{latency['server_infer_ms_per_req']:.1f} ms |"
        )
        if latency.get("client_p50_ms"):
            lines.append(f"| Client p50 | " f"{latency['client_p50_ms']:.1f} ms |")
        if latency.get("client_p99_ms"):
            lines.append(f"| Client p99 | " f"{latency['client_p99_ms']:.1f} ms |")
        if latency.get("serving_overhead_ms"):
            lines.append(
                f"| Serving overhead (p50 - infer) | "
                f"{latency['serving_overhead_ms']:.1f} ms "
                f"({latency.get('serving_overhead_pct', 0):.0f}%) |"
            )
        lines.append("")

    # System layer summary
    system = data.get("system", {})
    if system:
        lines.append("## System Resources (USE Method)")
        lines.append("")
        cpu = system.get("cpu", {})
        mem = system.get("memory", {})
        lines.append("| Resource | Utilization | Saturation | Errors |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| CPU | {cpu.get('cpu_user_pct', '?')}% user, "
            f"{cpu.get('cpu_system_pct', '?')}% sys | "
            f"load {cpu.get('load_1m', '?')}/{cpu.get('cores', '?')} cores | "
            f"iowait {cpu.get('cpu_iowait_pct', '?')}% |"
        )
        lines.append(
            f"| Memory | {mem.get('memory_used_pct', '?')}% used | "
            f"swap {mem.get('swap_used_pct', '?')}% | "
            f"PSI {mem.get('psi_memory', {}).get('some_avg60', '?')} |"
        )
        io = system.get("io", {})
        lines.append(
            f"| I/O | {io.get('disk_reads_total', '?')} reads, "
            f"{io.get('disk_writes_total', '?')} writes | "
            f"PSI {io.get('psi_io', {}).get('some_avg60', '?')} | — |"
        )
        lines.append("")

    # Process layer summary
    process = data.get("process", {})
    if process:
        lines.append("## Process Analysis")
        lines.append("")
        thread_cpu = process.get("per_thread_cpu", {})
        if thread_cpu.get("active_threads"):
            lines.append("### Active threads (by CPU)")
            lines.append("")
            lines.append("| TID | CPU % |")
            lines.append("|---|---|")
            for t in thread_cpu["active_threads"][:10]:
                lines.append(f"| {t['tid']} | {t['cpu_pct']:.1f}% |")
            lines.append("")

        fd = process.get("fd_count", {})
        if fd.get("fd_count"):
            lines.append(
                f"- File descriptors: {fd['fd_count']}/"
                f"{fd['fd_limit']} ({fd['fd_usage_pct']:.0f}%)"
            )
        lines.append("")

    # On-CPU profile
    oncpu = data.get("oncpu", {})
    if oncpu.get("top_functions"):
        lines.append("## On-CPU Profile (Top Functions)")
        lines.append("")
        lines.append("| Function | CPU % |")
        lines.append("|---|---|")
        for f in oncpu["top_functions"][:20]:
            lines.append(f"| `{f['function']}` | {f['cpu_pct']:.1f}% |")
        if oncpu.get("flame_graph"):
            lines.append("")
            lines.append(f"*Flame graph: `{oncpu['flame_graph']}`*")
        lines.append("")

    # Memory profile
    memory = data.get("memory", {})
    if memory.get("rss_timeline", {}).get("samples"):
        rss = memory["rss_timeline"]
        lines.append("## Memory Timeline")
        lines.append("")
        lines.append(
            f"- RSS range: {rss['rss_min_kb'] / 1024:.0f} – "
            f"{rss['rss_max_kb'] / 1024:.0f} MB"
        )
        lines.append(f"- Growth rate: " f"{rss['rss_growth_rate_kb_per_s']:.0f} KB/s")
        lines.append("")

    # GPU layer
    gpu = data.get("gpu", {})
    if gpu and not gpu.get("error"):
        smi = gpu.get("nvidia_smi", {})
        if smi:
            lines.append("## GPU Analysis")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| GPU | {smi.get('gpu_name', '?')} |")
            lines.append(
                f"| SM utilization | " f"{smi.get('gpu_utilization_pct', '?')}% |"
            )
            lines.append(
                f"| Memory used | "
                f"{smi.get('memory_used_mb', '?')}/"
                f"{smi.get('memory_total_mb', '?')} MB |"
            )
            lines.append(f"| Temperature | " f"{smi.get('temperature_c', '?')}°C |")
            lines.append(
                f"| Power | "
                f"{smi.get('power_draw_w', '?')}/"
                f"{smi.get('power_limit_w', '?')} W |"
            )
            if smi.get("is_throttled"):
                lines.append(
                    f"| **Throttling** | " f"**{smi.get('throttle_detail', '?')}** |"
                )
            lines.append("")

        dcgm = gpu.get("dcgm", {})
        if dcgm.get("kernel_gap_detected"):
            lines.append(
                f"⚠️ **Kernel gap detected:** " f"{dcgm.get('kernel_gap_note', '')}"
            )
            lines.append("")

    # Reference comparison
    if reference:
        lines.append("## Reference Comparison")
        lines.append("")
        lines.append(_reference_table(reference))
        lines.append("")

    # Configuration audit
    lines.append("## Configuration Audit")
    lines.append("")
    lines.append(
        "Review the environment manifest in "
        "`manifest.json` for the full configuration."
    )
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report was generated by `make diagnose`. "
        "Raw layer data is in `layers/` and findings in "
        "`findings/findings.json`.*"
    )

    # Write
    report_path = output_dir / "DIAGNOSIS.md"
    report_path.write_text("\n".join(lines) + "\n")

    # Also write diagnosis.json
    diagnosis = {
        "timestamp": now,
        "layers": sorted(k for k in data if k != "timestamp"),
        "findings": findings,
        "latency_attribution": latency,
        "reference": reference,
    }
    (output_dir / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, default=str) + "\n"
    )

    return report_path


def _executive_summary(
    data: dict[str, Any],
    findings: dict[str, Any],
    reference: dict[str, Any] | None,
) -> str:
    """Compose a 3-5 sentence executive summary from the findings."""
    parts: list[str] = []

    high = [f for f in findings["findings"] if f["severity"] in ("CRITICAL", "HIGH")]

    if high:
        top = high[0]
        parts.append(
            f"The primary bottleneck is **{top['finding']}**: " f"{top['evidence']}."
        )
        if len(high) > 1:
            others = ", ".join(f["finding"] for f in high[1:3])
            parts.append(f"Additional high-severity findings: {others}.")
    else:
        medium = [f for f in findings["findings"] if f["severity"] == "MEDIUM"]
        if medium:
            parts.append(
                f"No critical issues found; {len(medium)} medium-severity "
                f"findings, primarily: {medium[0]['finding']}."
            )
        else:
            parts.append(
                "No automated findings triggered — the serving path "
                "appears healthy at the current load level."
            )

    # GPU context
    gpu = data.get("gpu", {})
    smi = gpu.get("nvidia_smi", {})
    if smi and not gpu.get("error"):
        util = smi.get("gpu_utilization_pct", 0)
        if util < 50:
            parts.append(
                f"GPU utilization is only {util:.0f}% — significant "
                f"headroom remains on the device."
            )
        else:
            parts.append(
                f"GPU utilization is {util:.0f}% — the device is " f"well-utilized."
            )

    # Reference context
    if reference and reference.get("mlserver_efficiency_pct"):
        eff = reference["mlserver_efficiency_pct"]
        parts.append(
            f"MLServer achieves {eff:.0f}% of the Triton reference " f"ceiling."
        )

    return " ".join(parts)


def _reference_table(ref: dict[str, Any]) -> str:
    """Generate the 3-point reference comparison table."""
    lines = [
        "| Reference | Throughput (RPS) | p50 (ms) | p99 (ms) |",
        "|---|---|---|---|",
    ]
    for label in ("floor", "mlserver", "ceiling"):
        entry = ref.get(label, {})
        if entry:
            lines.append(
                f"| **{label.title()}** ({entry.get('name', '?')}) | "
                f"{entry.get('throughput_rps', '?')} | "
                f"{entry.get('p50_ms', '?')} | "
                f"{entry.get('p99_ms', '?')} |"
            )
    if ref.get("mlserver_efficiency_pct"):
        lines.append("")
        lines.append(
            f"**MLServer efficiency:** "
            f"{ref['mlserver_efficiency_pct']:.0f}% of ceiling"
        )
    if ref.get("serving_overhead_ms"):
        lines.append(
            f"**Serving overhead:** "
            f"{ref['serving_overhead_ms']:.1f} ms/request above floor"
        )
    return "\n".join(lines)
