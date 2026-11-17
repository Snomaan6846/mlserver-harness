"""Reports: JSON (already written by the runner) + CSV + Markdown + HTML."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    return json.loads((path / "results.json").read_text())


def to_csv(results: dict[str, Any]) -> str:
    import csv
    import io

    rows = []
    for case in results["cases"]:
        if case.get("suite") == "abort":
            continue
        row = {
            "suite": case.get("suite"),
            "framework": case.get("framework"),
            "tier": case.get("tier"),
            "model": case.get("model"),
        }
        row.update({f"p_{k}": v for k, v in (case.get("params") or {}).items()})
        outcome = case.get("result") or {}
        row.update(
            {
                k: outcome.get(k)
                for k in (
                    "throughput_rps",
                    "latency_ms_p50",
                    "latency_ms_p95",
                    "latency_ms_p99",
                    "error_rate",
                    "requests",
                )
            }
        )
        row["status"] = outcome.get("status", "")
        rows.append(row)
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(
            buffer, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def to_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# MLServer Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Status: **{results.get('status', '?')}**",
        "",
    ]
    manifest = results.get("manifest", {})
    image = manifest.get("image", {})
    lines += [
        "## Environment",
        "",
        f"- Image: `{image.get('reference', '?')}`",
        f"- Target: {image.get('target', '?')}",
        f"- MLServer git: `{manifest.get('git', {}).get('mlserver-source', '?')[:12]}`",
        f"- Host: {manifest.get('host', {}).get('cpu_model', '?')} "
        f"({manifest.get('host', {}).get('cores_logical', '?')} cores)",
        "",
    ]
    by_suite: dict[str, list[dict]] = {}
    for case in results["cases"]:
        if case.get("suite") == "abort":
            lines += [
                "## Aborted",
                "",
                f"`{case.get('status')}` — " f"{case.get('detail', '')[:500]}",
                "",
            ]
            continue
        by_suite.setdefault(case["suite"], []).append(case)
    for suite, cases in sorted(by_suite.items()):
        lines.append(f"## {suite} ({len(cases)} cases)")
        lines.append("")
        lines.append("| model | params | RPS | p50 ms | p99 ms | err |")
        lines.append("|---|---|---|---|---|---|")
        for case in cases:
            outcome = case.get("result") or {}
            if outcome.get("status", "").startswith("SKIPPED"):
                lines.append(f"| {case.get('model')} | - | {outcome['status']} | | | |")
                continue
            params = ", ".join(
                f"{k}={v}"
                for k, v in (case.get("params") or {}).items()
                if not isinstance(v, (dict, list))
            )
            lines.append(
                f"| {case.get('model')} | {params} "
                f"| {outcome.get('throughput_rps', 0):.2f} "
                f"| {outcome.get('latency_ms_p50', 0):.2f} "
                f"| {outcome.get('latency_ms_p99', 0):.2f} "
                f"| {outcome.get('error_rate', 0):.2%} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def to_html(markdown: str, results: dict[str, Any]) -> str:
    """Self-contained HTML (embedded CSS; charts as inline SVG when
    matplotlib is available)."""
    charts = _pareto_svg(results)
    body = _minimal_markdown_to_html(markdown)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>MLServer Benchmark Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem auto;
       max-width: 60rem; color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem;
          text-align: left; font-size: 0.9rem; }}
th {{ background: #16213e; color: white; }}
tr:nth-child(even) {{ background: #f4f6fb; }}
h1, h2 {{ color: #16213e; }}
code {{ background: #eef; padding: 0.1rem 0.3rem; }}
</style></head><body>
{charts}
{body}
</body></html>"""


def _pareto_svg(results: dict[str, Any]) -> str:
    """Throughput vs p99 Pareto scatter (inline SVG, no JS deps)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    points = [
        (
            case["result"].get("throughput_rps", 0),
            case["result"].get("latency_ms_p99", 0),
            case.get("model", "?"),
        )
        for case in results["cases"]
        if case.get("result")
        and not (case["result"].get("status", "").startswith("SKIPPED"))
    ]
    if not points:
        return ""
    figure, axis = plt.subplots(figsize=(7, 4))
    for rps, p99, _ in points:
        axis.scatter(rps, p99, s=28, color="#16213e", alpha=0.6)
    axis.set_xlabel("Throughput (req/s)")
    axis.set_ylabel("p99 latency (ms)")
    axis.set_title("Throughput vs p99 (Pareto)")
    axis.grid(alpha=0.25)
    import io

    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue().decode()


def _minimal_markdown_to_html(markdown: str) -> str:
    """Tiny markdown renderer for our own report structure (headings,
    tables, bold, code)."""
    import html as html_module

    out: list[str] = []
    lines = markdown.splitlines()
    in_table = False
    for line in lines:
        escaped = html_module.escape(line)
        if line.startswith("# "):
            out.append(f"<h1>{escaped[2:]}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{escaped[3:]}</h2>")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            row = "".join(f"<{tag}>{html_module.escape(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            if line.strip() == "":
                out.append("")
            elif line.startswith("- "):
                out.append(f"<li>{escaped[2:]}</li>")
            else:
                out.append(f"<p>{escaped}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def write_reports(result_dir: Path) -> dict[str, Path]:
    results = load_results(result_dir)
    csv_path = result_dir / "results.csv"
    csv_path.write_text(to_csv(results))
    markdown_path = result_dir / "report.md"
    markdown = to_markdown(results)
    markdown_path.write_text(markdown)
    html_path = result_dir / "report.html"
    html_path.write_text(to_html(markdown, results))
    return {"csv": csv_path, "markdown": markdown_path, "html": html_path}
