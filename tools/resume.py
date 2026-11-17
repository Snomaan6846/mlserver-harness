#!/usr/bin/env python3
"""Resume a benchmark campaign: re-run every plan not marked complete.

Reads the checkpoint (results/STATE.json), finds every plan in
benchmarks/*/plans/*.json whose checkpoint status is not "complete",
and runs them through tools/run.py's orchestration in dependency-free
order. Sessions are atomic: an interrupted session re-runs from scratch
(the fresh-pod-per-trial discipline makes that scientifically sound).

Usage:
    tools/resume.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from checkpoint import Checkpoint  # noqa: E402


def discover_plans() -> list[tuple[str, str]]:
    plans: list[tuple[str, str]] = []
    for domain_dir in sorted((REPO / "benchmarks").iterdir()):
        plans_dir = domain_dir / "plans"
        if not plans_dir.is_dir():
            continue
        for plan_file in sorted(plans_dir.glob("*.json")):
            plans.append((domain_dir.name, plan_file.stem))
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would run; execute nothing"
    )
    args = parser.parse_args()

    checkpoint = Checkpoint(REPO / "results" / "STATE.json")
    pending = [
        (domain, plan)
        for domain, plan in discover_plans()
        if not checkpoint.is_complete(domain, plan)
    ]

    if not pending:
        print("nothing to resume — every plan is complete")
        return

    print(f"{len(pending)} plan(s) to run:")
    for domain, plan in pending:
        entry = checkpoint.entry(domain, plan)
        note = f" (previous: {entry['status']})" if entry else ""
        print(f"  {domain}/{plan}{note}")

    if args.dry_run:
        return

    from run import run_plan  # imported late: heavy module

    failures = []
    for domain, plan in pending:
        print(f"\n=== {domain}/{plan} ===")
        code = run_plan(
            domain,
            plan,
            rounds_override=None,
            concurrency_override=None,
            image_override=None,
            force=False,
            dry_run=False,
            context=None,
            namespace="mlserver-perf",
            client_pod=None,
        )
        if code != 0:
            failures.append(f"{domain}/{plan}")
    if failures:
        print(f"\n{len(failures)} plan(s) did not complete cleanly:")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)
    print("\ncampaign complete")


if __name__ == "__main__":
    main()
