#!/usr/bin/env python3
"""Generate a markdown report from collected session metrics.

Reads each session's metrics-summary.csv (produced by tools/summarize.py)
and renders a per-domain markdown report with median tables, plus the
invalid-trial tally from SUMMARY.json.

CLI:
    tools/report.py                       # all of results/ -> reports/
    tools/report.py results/wire-format   # one domain
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def render_session(session: Path) -> str:
    summary = json.loads((session / "SUMMARY.json").read_text())
    plan = summary.get("plan", {})
    rows = summary.get("results", [])
    invalid = [row for row in rows if not row.get("valid")]

    lines: list[str] = []
    lines.append(f"## {session.parent.name}/{session.name}")
    lines.append("")
    lines.append(
        f"- Trials: {len(rows)} "
        f"(plan: {len(plan.get('variants', []))} variants x "
        f"{plan.get('concurrency')} x {plan.get('rounds')} rounds)"
    )
    lines.append(f"- Invalid: {len(invalid)}")
    if invalid:
        for row in invalid[:10]:
            lines.append(
                f"  - {row.get('case')}: {row.get('variant')} "
                f"C{row.get('concurrency')} — {row.get('error', 'gate failure')}"
            )
    lines.append("")

    csv_path = session / "metrics-summary.csv"
    if not csv_path.is_file():
        lines.append("*(no metrics-summary.csv — run tools/summarize.py)*")
        return "\n".join(lines)

    with csv_path.open() as handle:
        table = list(csv.DictReader(handle))
    if not table:
        lines.append("*(empty metrics summary)*")
        return "\n".join(lines)

    columns = [
        ("variant", "Variant"),
        ("concurrency", "C"),
        ("trials", "n"),
        ("rps_median", "RPS (median)"),
        ("rps_spread_pct", "Spread %"),
        ("p50_ms", "p50 ms"),
        ("p99_ms", "p99 ms"),
        ("cpu_ms_per_req", "CPU-ms/req"),
        ("gpu_util_mean", "GPU %"),
        ("mem_ws_max_mb", "Mem MB"),
    ]
    header = [label for key, label in columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in table:
        cells = [row.get(key, "") for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=REPO / "results",
        help="results root or a domain directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output markdown path (default: reports/benchmark-report.md)",
    )
    args = parser.parse_args()

    root = args.path.resolve()
    sessions = sorted(
        child for child in root.glob("**/*") if (child / "SUMMARY.json").is_file()
    )
    if not sessions:
        raise SystemExit(f"no collected sessions found under {root}")

    output = args.output or REPO / "reports" / "benchmark-report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "# MLServer Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Sessions: {len(sessions)}",
        "",
    ]
    for session in sessions:
        parts.append(render_session(session))
    output.write_text("\n".join(parts) + "\n")
    print(f"wrote {output} ({len(sessions)} sessions)")


if __name__ == "__main__":
    main()
