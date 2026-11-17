#!/usr/bin/env python3
"""Result compliance checker: verify a collected session meets the gates.

Reads a session's SUMMARY.json and per-case receipts and checks:

  1. Every trial is valid (client validity, start-gate, exit codes)
  2. Every trial produced a receipts.json provenance record
  3. Every receipt proves parity ("verified") for its gates
  4. Trial count matches the plan (variants x concurrency x rounds)
  5. Round-to-round spread is within threshold (default 15%)

Exit code 0 = compliant, 1 = violations found.

CLI:
    tools/validate.py results/wire-format/stock-vs-binary
    tools/validate.py results/            # every session under results/
"""

import argparse
import json
import sys
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def validate_session(session_dir: Path, spread_threshold: float) -> list[str]:
    violations: list[str] = []
    summary_path = session_dir / "SUMMARY.json"
    if not summary_path.is_file():
        return [f"{session_dir}: no SUMMARY.json (session not collected?)"]

    summary = json.loads(summary_path.read_text())
    plan = summary.get("plan", {})
    rows = summary.get("results", [])
    expected = (
        len(plan.get("variants", []))
        * len(plan.get("concurrency", []))
        * plan.get("rounds", 0)
    )
    name = plan.get("session", session_dir.name)

    if expected and len(rows) != expected:
        violations.append(f"{name}: trial count {len(rows)} != expected {expected}")

    for row in rows:
        case = row.get("case", "?")
        if not row.get("valid"):
            violations.append(
                f"{name}/{case}: invalid trial " f"({row.get('error', 'gate failure')})"
            )
            continue
        receipts_path = session_dir / case / "receipts.json"
        if not receipts_path.is_file():
            violations.append(f"{name}/{case}: missing receipts.json")
            continue
        receipts = json.loads(receipts_path.read_text())
        correctness = receipts.get("correctness", {})
        for gate, result in correctness.items():
            if result is None:
                continue
            if isinstance(result, dict) and "verified" not in str(
                result.get("rest_reference", result.get("grpc_reference", ""))
            ):
                violations.append(f"{name}/{case}: gate '{gate}' not verified")

    # Round-spread check per (variant, concurrency) cell.
    cells: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row.get("valid") and "rps" in row:
            cells.setdefault((row["variant"], row["concurrency"]), []).append(
                row["rps"]
            )
    for (variant, concurrency), rates in sorted(cells.items()):
        if len(rates) < 2:
            continue
        low, high = min(rates), max(rates)
        spread = (high - low) / percentile(rates, 0.5)
        if spread > spread_threshold:
            violations.append(
                f"{name}: {variant} C{concurrency} spread {spread:.1%} "
                f"exceeds {spread_threshold:.0%} ({rates})"
            )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="session directory or results/ root")
    parser.add_argument("--spread-threshold", type=float, default=0.15)
    args = parser.parse_args()

    if (args.path / "SUMMARY.json").is_file():
        sessions = [args.path]
    else:
        sessions = sorted(
            child
            for child in args.path.glob("*/*")
            if (child / "SUMMARY.json").is_file()
        )
    if not sessions:
        raise SystemExit(f"no sessions found under {args.path}")

    all_violations: list[str] = []
    for session in sessions:
        violations = validate_session(session, args.spread_threshold)
        status = "PASS" if not violations else f"FAIL ({len(violations)})"
        label = session.name if session == args.path else session.relative_to(args.path)
        print(f"{label}: {status}")
        all_violations.extend(violations)

    if all_violations:
        print(f"\n{len(all_violations)} violation(s):")
        for violation in all_violations:
            print(f"  - {violation}")
        sys.exit(1)
    print(f"\nall {len(sessions)} session(s) compliant")


if __name__ == "__main__":
    main()
