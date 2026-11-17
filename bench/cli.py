"""CLI entry points for the Docker-mode benchmark engine.

Interactive:   python3 -m bench.cli benchmark
Direct:        python3 -m bench.cli run --target cpu --framework sklearn
Dry run:       python3 -m bench.cli dry-run --target cpu --framework all
Compare:       python3 -m bench.cli compare BASELINE CANDIDATE
"""

import argparse
import json
import sys
from pathlib import Path

from . import compare as compare_mod
from . import report as report_mod
from .config import FRAMEWORKS_CPU, FRAMEWORKS_GPU, SUITES, TIERS, RunConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _exit_code_for(output: Path) -> tuple[int, set[str]]:
    """(exit code, failed suite names) for a completed run.

    The CLI must not exit 0 on a failed suite: the paired driver (and
    any caller) treats rc=0 as a healthy side. A failed suite used to
    exit 0 and the driver only noticed at the outcome check after
    merge — about 7 wasted minutes in the reviewer's mini A/A.

    main() must use ONLY this helper for the check — an inline
    json.loads in main() crashed with UnboundLocalError (a local
    `import json` elsewhere in main() shadows the module-level import
    for the whole function scope), making every clean run exit 1."""
    results_file = output / "results.json"
    if not results_file.is_file():
        return 0, set()
    payload = json.loads(results_file.read_text())
    failed = {
        c.get("suite", "?")
        for c in payload.get("cases", [])
        if c.get("status") == "SUITE_FAILED"
    }
    return (1 if failed else 0), failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    wizard = sub.add_parser("benchmark", help="interactive wizard")
    wizard.add_argument(
        "--non-interactive", action="store_true", help="use defaults without prompts"
    )

    run = sub.add_parser("run", help="run benchmarks (non-interactive)")
    _run_args(run)

    dry = sub.add_parser("dry-run", help="validate images + zoo + parity, no load")
    _run_args(dry)

    cmp_parser = sub.add_parser("compare", help="statistical regression")
    cmp_parser.add_argument("baseline", type=Path)
    cmp_parser.add_argument("candidate", type=Path)
    cmp_parser.add_argument("--throughput-sla", type=float, default=-0.05)
    cmp_parser.add_argument("--p99-sla", type=float, default=0.10)

    sub.add_parser("report", help="render reports for a results dir")

    diagnose = sub.add_parser("diagnose", help="run 6-layer diagnostic")
    diagnose.add_argument(
        "--engine",
        choices=["local", "docker", "podman", "container", "cluster", "isvc"],
        default="local",
        help="execution engine",
    )
    diagnose.add_argument(
        "--container", default=None, help="target container name (docker engine only)"
    )
    diagnose.add_argument("--output-dir", default=None)
    diagnose.add_argument("--metrics-port", type=int, default=8082)
    diagnose.add_argument("--profile-duration", type=float, default=30)
    diagnose.add_argument(
        "--skip-profile",
        action="store_true",
        help="skip py-spy/memray (faster, less detail)",
    )

    reference = sub.add_parser("reference", help="3-point reference")
    reference.add_argument(
        "--engine",
        choices=["local", "docker", "podman", "container", "cluster", "isvc"],
        default="local",
        help="execution engine",
    )
    reference.add_argument("--tier", default="small")
    reference.add_argument("--gpu", action="store_true")
    reference.add_argument("--duration", type=float, default=30)
    reference.add_argument("--output-dir", default=None)
    reference.add_argument(
        "--skip-triton", action="store_true", help="skip Triton ceiling measurement"
    )
    reference.add_argument(
        "--mlserver-results",
        default=None,
        help="path to existing MLServer results JSON",
    )
    return parser


def _run_args(run: argparse.ArgumentParser) -> None:
    run.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="replicates per cell (multi-round statistics; summary "
        "reports mean/std across rounds)",
    )
    run.add_argument(
        "--engine",
        choices=["local", "docker", "podman", "container", "cluster", "isvc"],
        default="local",
        help="execution engine: local (container) or cluster (ISVC/RHOAI)",
    )
    run.add_argument("--target", choices=["cpu", "gpu"], default="cpu")
    run.add_argument(
        "--framework",
        default=None,
        help="single framework, 'all' (per target), or comma-separated",
    )
    run.add_argument("--suites", default="all", help="comma-separated or 'all'")
    run.add_argument("--tiers", default="all")
    run.add_argument(
        "--profile",
        default="standard",
        choices=["quick", "standard", "exhaustive", "custom"],
    )
    run.add_argument("--image", default="", help="override the MLServer docker image")
    run.add_argument(
        "--real-models",
        action="store_true",
        help="include pinned downloadable reference models",
    )
    run.add_argument("--cache", default="false", choices=["false", "true", "both"])
    run.add_argument("--output-dir", default="")
    run.add_argument(
        "--run-id",
        default="",
        help="unique run identity (lowercase/dashes; auto-generated "
        "when empty). Names every container and result artifact so "
        "runs are never ambiguous in `docker ps` or on disk.",
    )
    run.add_argument(
        "--concurrency", default="", help="custom profile: comma-separated levels"
    )
    run.add_argument("--warmup-seconds", type=int, default=15)
    run.add_argument("--window-seconds", type=int, default=30)


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    allowed = FRAMEWORKS_GPU if args.target == "gpu" else FRAMEWORKS_CPU
    if not args.framework or args.framework == "all":
        frameworks = list(allowed)
    else:
        frameworks = [f.strip() for f in args.framework.split(",")]
    suites = (
        list(SUITES)
        if args.suites == "all"
        else [s.strip() for s in args.suites.split(",")]
    )
    tiers = (
        list(TIERS)
        if args.tiers == "all"
        else [t.strip() for t in args.tiers.split(",")]
    )
    custom: dict = {}
    if args.profile == "custom":
        if args.concurrency:
            custom["concurrency"] = [int(v) for v in args.concurrency.split(",")]
        custom["warmup_seconds"] = args.warmup_seconds
        custom["window_seconds"] = args.window_seconds
    return RunConfig(
        target=args.target,
        frameworks=frameworks,
        suites=suites,
        tiers=tiers,
        load_profile=args.profile,
        custom=custom,
        image=args.image,
        cache_enabled=args.cache,
        include_real=getattr(args, "real_models", False),
        output_dir=args.output_dir,
        rounds=getattr(args, "rounds", 1),
        run_id=getattr(args, "run_id", "") or "",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "benchmark":
        from .wizard import run_wizard

        if args.non_interactive:
            config = RunConfig()
        else:
            config = run_wizard()
        if config is None:
            sys.exit(0)
        from .runner import execute

        output = execute(config, REPO_ROOT)
        report_mod.write_reports(output)
        return

    if args.command in ("run", "dry-run"):
        config = _config_from_args(args)
        problems = config.validate()
        if problems:
            sys.exit(f"invalid: {problems}")
        from .runner import dry_run, execute

        if args.command == "dry-run":
            output = dry_run(config, REPO_ROOT)
        else:
            output = execute(config, REPO_ROOT)
            report_mod.write_reports(output)
            # Fail fast at the process boundary: a run that recorded
            # SUITE_FAILED cases exits non-zero so callers (paired
            # driver, pilot, orchestrator) see the failure immediately
            code, failed_suites = _exit_code_for(output)
            if failed_suites:
                print(
                    f"[cli] run complete with SUITE_FAILED cases in: "
                    f"{', '.join(sorted(failed_suites))} — see results.json"
                )
                sys.exit(code)
        status = "ok" if output else "?"
        print(f"[cli] {args.command} complete ({status})")
        return

    if args.command == "compare":
        result = compare_mod.compare(
            args.baseline,
            args.candidate,
            {"throughput_delta": args.throughput_sla, "p99_delta": args.p99_sla},
        )

        if result["status"] == "MANIFEST_MISMATCH":
            # The mismatch result has no verdicts — refuse with a clear
            # message and a distinct exit code (the old handler crashed
            # with KeyError: 'regressions' on this path).
            print("COMPARISON REFUSED — environment mismatch:")
            for mismatch in result["mismatches"]:
                print(f"  - {mismatch}")
            print(result["note"])
            (args.candidate / "comparison.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            sys.exit(2)
        print(json.dumps(result, indent=2)[:8000])
        (args.candidate / "comparison.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        sys.exit(1 if result["regressions"] else 0)

    if args.command == "diagnose":
        from bench.diagnose.cli import diagnose_command

        diagnose_command(args)
        return

    if args.command == "reference":
        from bench.diagnose.cli import reference_command

        reference_command(args)
        return

    if args.command == "report":
        import sys as _sys

        if len(_sys.argv) != 3:
            sys.exit("usage: python3 -m bench.cli report <results-dir>")
        paths = report_mod.write_reports(Path(_sys.argv[2]))
        print(f"wrote: {', '.join(str(p) for p in paths.values())}")


if __name__ == "__main__":
    main()
