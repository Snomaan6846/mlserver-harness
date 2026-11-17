#!/usr/bin/env python3
"""Checkpoint state for benchmark campaigns: what ran, what is left.

The state file (results/STATE.json) records one entry per (domain, plan):

    {"wire-format/stock-vs-binary": {
         "status": "complete",       # pending | running | complete | failed
         "trials": 30,
         "invalid": 0,
         "cases": 30,
         "updated": "2026-11-14T19:00:00+00:00"
     }}

Sessions are atomic: an interrupted session is re-run from scratch on
resume (the fresh-pod-per-trial discipline makes partial sessions
scientifically safe to redo). Checkpointing therefore operates at plan
granularity — which is what campaign-level resumability needs.

CLI:
    tools/checkpoint.py            # print status table
    tools/checkpoint.py --reset DOMAIN/PLAN
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self.state: dict[str, Any] = json.loads(self.path.read_text())
        else:
            self.state = {}

    # -- queries -----------------------------------------------------------

    def entry(self, domain: str, plan: str) -> dict[str, Any]:
        return self.state.get(f"{domain}/{plan}", {})

    def is_complete(self, domain: str, plan: str) -> bool:
        return self.entry(domain, plan).get("status") == "complete"

    def pending_plans(self) -> list[str]:
        return sorted(
            key
            for key, value in self.state.items()
            if value.get("status") != "complete"
        )

    # -- mutations ---------------------------------------------------------

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2) + "\n")

    def mark_running(self, domain: str, plan: str, cases: int) -> None:
        entry = self.entry(domain, plan)
        entry.update(
            status="running",
            cases=cases,
            updated=datetime.now(timezone.utc).isoformat(),
        )
        self.state[f"{domain}/{plan}"] = entry
        self._save()

    def record(
        self,
        domain: str,
        plan: str,
        trials: int,
        invalid: int,
        complete: bool,
    ) -> None:
        entry = self.entry(domain, plan)
        entry.update(
            status="complete" if complete else "failed",
            trials=trials,
            invalid=invalid,
            updated=datetime.now(timezone.utc).isoformat(),
        )
        self.state[f"{domain}/{plan}"] = entry
        self._save()

    def reset(self, domain: str, plan: str) -> None:
        self.state.pop(f"{domain}/{plan}", None)
        self._save()


def print_status(state_path: Path) -> None:
    if not state_path.is_file():
        print("no checkpoint state yet — nothing has run")
        return
    state = json.loads(state_path.read_text())
    if not state:
        print("checkpoint state is empty")
        return
    width = max(len(key) for key in state)
    print(f"{'plan'.ljust(width)}  {'status':9s}  {'trials':>6s}  {'invalid':>7s}")
    for key in sorted(state):
        entry = state[key]
        print(
            f"{key.ljust(width)}  {entry.get('status', '?'):9s}  "
            f"{str(entry.get('trials', '-')):>6s}  "
            f"{str(entry.get('invalid', '-')):>7s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "STATE.json",
    )
    parser.add_argument(
        "--reset", metavar="DOMAIN/PLAN", help="drop a plan's checkpoint entry"
    )
    args = parser.parse_args()
    if args.reset:
        domain, _, plan = args.reset.partition("/")
        Checkpoint(args.state).reset(domain, plan)
        print(f"reset {args.reset}")
    else:
        print_status(args.state)


if __name__ == "__main__":
    main()
