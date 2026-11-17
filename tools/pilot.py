"""Scoped baseline PILOT: one round, timing + valid outcomes only.

Reviewer conditions:
- Clean tree (docs-only commit allowed before launch — done: 33f8951)
- Record idle load1 and control-plane processes
- Protocol, workers, middleware, saturation on sklearn and onnx
- Quick profile, one round — NOT a performance claim
- Separate output directory (results/val-pilot)
- Crosscheck + mutation-check before and after
- Read controls-findings.json before any numbers
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "val-pilot"
LOG_LINES = []
SUMMARY = {"phases": {}}

SUITES = "protocol,workers,middleware,saturation"
FRAMEWORKS = ["sklearn", "onnx"]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)
    _write()


def _write():
    (OUT / "SUMMARY.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    (OUT / "LOG.txt").write_text("\n".join(LOG_LINES) + "\n")


def record(phase, status, detail, seconds):
    SUMMARY[phase] = {"status": status, "detail": detail, "seconds": round(seconds, 1)}
    log(f"PHASE {phase}: {status} — {detail} ({seconds:.0f}s)")
    _write()


def run_cmd(phase, cmd, timeout_s, expect_rc=0):
    started = time.time()
    result = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout_s
    )
    elapsed = time.time() - started
    output = result.stdout + result.stderr
    phase_log = OUT / "phase-logs" / f"{phase}.log"
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    phase_log.write_text(output if output.strip() else "(no output)")
    tail = output.strip().splitlines()
    tail_line = tail[-1][:200] if tail else ""
    ok = result.returncode == expect_rc
    record(phase, "PASS" if ok else f"FAIL(rc={result.returncode})", tail_line, elapsed)
    return result.returncode, output


def _validate_out_dir(out: Path, force: bool) -> None:
    """Refuse to write into a non-empty directory without --force
    (same guard as the orchestrator — the smoke run overwrote the
    committed pilot evidence because the pilot had no such guard)."""
    if out.exists() and any(out.iterdir()) and not force:
        existing = sorted(p.name for p in out.iterdir())[:5]
        raise SystemExit(
            f"refusing to write into non-empty {out} "
            f"(contains: {', '.join(existing)}...) — pass --force to "
            "overwrite"
        )


def _record_crosscheck_numbers(phase, output):
    """Extract and record the actual ours-vs-ab numbers — the claim
    must state them, not just 'PASS' (a +13% gap is 'inconclusive',
    not validation)."""
    import re

    ours = re.search(r"ours mean = ([\d.]+)", output)
    ab = re.search(r"ab mean = ([\d.]+)", output)
    if ours and ab:
        ours_v, ab_v = float(ours.group(1)), float(ab.group(1))
        gap = (ours_v / ab_v - 1) * 100 if ab_v else 0
        if abs(gap) <= 10:
            verdict = "PASS (within ±10%)"
        elif gap > 0:
            verdict = (
                "INCONCLUSIVE (ours faster — ab single-threaded; "
                "not independent validation)"
            )
        else:
            verdict = "FAIL (ours SLOWER than ab — generator bottleneck)"
        record(phase, verdict, f"ours={ours_v:.0f} ab={ab_v:.0f} gap={gap:+.1f}%", 0.0)
        return "FAIL" in verdict
    else:
        record(phase, "INFO", "numbers not parsed from crosscheck output", 0.0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one tiny suite, TWO rounds — every code path including "
        "multi-round; run before the real launch",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="rounds per cell (baseline uses 5)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output directory (pilot: results/val-pilot; baseline: "
        "results/val-baseline; smoke: throwaway like results/smoke-tmp)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow writing into a non-empty output directory",
    )
    args = parser.parse_args()
    global OUT, ROUNDS
    _validate_out_dir(Path(args.out), args.force)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        ROUNDS = 2  # cover the multi-round path
        return smoke()
    ROUNDS = args.rounds
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY["started"] = datetime.now(timezone.utc).isoformat()
    SUMMARY["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    SUMMARY["purpose"] = "pilot: timing + valid outcomes, NOT a performance claim"

    # Host condition (reviewer: record idle load + control plane)
    load1 = os.getloadavg()[0]
    cp = subprocess.run(
        ["ps", "-eo", "comm,pcpu", "--no-headers"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    control_plane = [
        line.strip()
        for line in cp.stdout.splitlines()
        if any(
            k in line
            for k in (
                "kube-apiserver",
                "etcd",
                "kubelet",
                "containerd",
                "kube-controller",
                "kube-scheduler",
            )
        )
    ]
    SUMMARY["host_condition"] = {
        "idle_load1": round(load1, 2),
        "control_plane_processes": control_plane,
        "note": "root-owned K8s control plane, ~20-25% of one core, "
        "present during all runs on this workstation",
    }
    log(f"host: load1={load1:.2f}, control-plane procs: {len(control_plane)}")
    record(
        "host-check", "PASS", f"load1={load1:.2f}, {len(control_plane)} cp procs", 1.0
    )

    # Pre-flight: mutation + crosscheck (crosscheck needs models —
    # on a fresh checkout it can't run until a run generates the repo)
    run_cmd("preflight-mutation", ["make", "mutation-check"], 2400)
    if (REPO / "models-zoo" / "sklearn-grad-boost-small").is_dir():
        _, xout = run_cmd("preflight-crosscheck", ["make", "crosscheck"], 1200)
        if _record_crosscheck_numbers("preflight-crosscheck-numbers", xout):
            log("ABORT: generator slower than ab — numbers cannot be trusted")
            sys.exit(1)
    else:
        record(
            "preflight-crosscheck",
            "NOT_RUN",
            "models-zoo not present — crosscheck requires a prior run",
            0.0,
        )

    # Pilot matrix
    total_cases = 0
    for fw in FRAMEWORKS:
        out_dir = OUT / f"pilot-{fw}"
        cmd = [
            sys.executable,
            "-m",
            "bench.cli",
            "run",
            "--suites",
            SUITES,
            "--framework",
            fw,
            "--tiers",
            "small",
            "--profile",
            "quick",
            "--rounds",
            str(ROUNDS),
            "--output-dir",
            str(out_dir),
        ]
        started = time.time()
        run_cmd(f"pilot-{fw}", cmd, 7200)
        elapsed = time.time() - started
        # Record the outcome verdict IN THIS SUMMARY (not the
        # orchestrator's log which gets overwritten); abort on failure
        import local_validation as lv

        lv.OUT = OUT
        lv.SUMMARY = SUMMARY
        lv.LOG_LINES = LOG_LINES
        # Ensure the phases dict exists (local_validation.record
        # writes to SUMMARY["phases"][phase] — the original wiring
        # crashed with KeyError: 'phases' because the pilot's summary
        # had no phases key; the whole matrix would run and then die)
        SUMMARY.setdefault("phases", {})
        ok = lv.check_run_outcomes(f"pilot-{fw}-outcomes", out_dir, elapsed)
        if not ok:
            log(f"ABORT: outcome check failed for {fw}")
            SUMMARY["finished"] = datetime.now(timezone.utc).isoformat()
            _write()
            sys.exit(1)
        if (out_dir / "results.json").is_file():
            results = json.loads((out_dir / "results.json").read_text())
            cases = len(results.get("cases", []))
            total_cases += cases
            log(f"  {fw}: {cases} cases in {elapsed:.0f}s ({elapsed/cases:.1f}s/case)")
            # Round-count verification on the REAL path (the unwired
            # --rounds was the previous silent failure — smoke-only
            # verification would not catch it here)
            round_values = {c.get("round", 0) for c in results.get("cases", [])}
            if ROUNDS > 1 and len(round_values) < ROUNDS:
                log(
                    f"ABORT: expected {ROUNDS} distinct rounds for {fw}, "
                    f"got {sorted(round_values)}"
                )
                SUMMARY["finished"] = datetime.now(timezone.utc).isoformat()
                _write()
                sys.exit(1)
            log(f"  {fw}: rounds verified: {sorted(round_values)}")

    # Post-run: crosscheck + mutation + controls scan
    run_cmd("post-mutation", ["make", "mutation-check"], 2400)
    _, xout = run_cmd("post-crosscheck", ["make", "crosscheck"], 1200)
    if _record_crosscheck_numbers("post-crosscheck-numbers", xout):
        log(
            "WARNING: post-run crosscheck FAILED — treat all numbers "
            "from this run as generator-limited"
        )

    # Controls scan
    sys.path.insert(0, str(REPO / "tools"))
    import local_validation as lv

    sys.path.pop(0)
    findings = lv._scan_positive_controls(OUT)
    record(
        "controls-scan",
        "INFO",
        f"{len(findings)} findings: " + "; ".join(findings),
        0.5,
    )

    # Timing: ACTUAL elapsed for the rounds that ran (the previous
    # version multiplied one round by 5 and called it "measured")
    # record() writes to the TOP LEVEL of SUMMARY (not SUMMARY["phases"]
    # — that dict holds only the outcome-check entries from
    # local_validation.check_run_outcomes). Sum the top-level pilot-*
    # entries, excluding the outcome checks by name.
    matrix_seconds = sum(
        v["seconds"]
        for k, v in SUMMARY.items()
        if isinstance(v, dict)
        and k.startswith("pilot-")
        and "outcomes" not in k
        and "seconds" in v
    )
    SUMMARY["timing"] = {
        "rounds_executed": ROUNDS,
        "matrix_seconds_actual": round(matrix_seconds),
        "matrix_hours_actual": round(matrix_seconds / 3600, 2),
        "total_cases": total_cases,
        "note": (
            f"ACTUAL elapsed for {ROUNDS} round(s)"
            if ROUNDS > 1
            else "one round; 5-round figure requires running 5 rounds"
        ),
    }
    log(f"TIMING: {ROUNDS} round(s) = {matrix_seconds:.0f}s actual")

    SUMMARY["finished"] = datetime.now(timezone.utc).isoformat()
    _write()
    log(f"PILOT COMPLETE: {total_cases} cases, {ROUNDS} round(s)")


def smoke():
    """Every code path in ~4 minutes: one suite, one framework,
    TWO ROUNDS (covers the multi-round path), post-run mutation,
    crosscheck, controls scan, outcome check, round-count verification,
    timing summary. Catches wiring crashes and unwired flags before
    they kill a baseline."""
    SUMMARY["started"] = datetime.now(timezone.utc).isoformat()
    SUMMARY["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    SUMMARY["purpose"] = f"smoke: every code path, {ROUNDS} rounds"

    _host_check()
    run_cmd("smoke-mutation", ["make", "mutation-check"], 2400)

    out_dir = OUT / "smoke-caching"
    cmd = [
        sys.executable,
        "-m",
        "bench.cli",
        "run",
        "--suites",
        "caching",
        "--framework",
        "sklearn",
        "--tiers",
        "small",
        "--profile",
        "quick",
        "--rounds",
        str(ROUNDS),
        "--output-dir",
        str(out_dir),
    ]
    started = time.time()
    run_cmd("smoke-caching", cmd, 600)
    elapsed = time.time() - started

    import local_validation as lv

    lv.OUT = OUT
    lv.SUMMARY = SUMMARY
    lv.LOG_LINES = LOG_LINES
    SUMMARY.setdefault("phases", {})
    ok = lv.check_run_outcomes("smoke-caching-outcomes", out_dir, elapsed)
    if not ok:
        log("SMOKE FAIL: outcome check failed")
        sys.exit(1)

    # Verify the rounds actually landed in the results
    results = json.loads((out_dir / "results.json").read_text())
    round_values = {c.get("round", 0) for c in results.get("cases", [])}
    if len(round_values) < ROUNDS:
        log(
            f"SMOKE FAIL: expected {ROUNDS} distinct rounds, "
            f"got {sorted(round_values)}"
        )
        sys.exit(1)
    log(f"  rounds verified: {sorted(round_values)}")

    # Post-run phases (previously missing from smoke)
    run_cmd("smoke-post-mutation", ["make", "mutation-check"], 2400)
    _, xout = run_cmd("smoke-crosscheck", ["make", "crosscheck"], 1200)
    if _record_crosscheck_numbers("smoke-crosscheck-numbers", xout):
        log("SMOKE FAIL: generator slower than ab")
        sys.exit(1)

    findings = lv._scan_positive_controls(OUT)
    record(
        "smoke-controls",
        "INFO",
        f"{len(findings)} findings: " + "; ".join(findings),
        0.5,
    )

    # Timing from ACTUAL elapsed (not extrapolated)
    SUMMARY["timing"] = {
        "smoke_rounds": ROUNDS,
        "smoke_seconds": round(elapsed),
        "note": "smoke timing — actual elapsed, not extrapolated",
    }

    SUMMARY["finished"] = datetime.now(timezone.utc).isoformat()
    _write()
    log("SMOKE PASS: every code path executed")
    return 0


def _host_check():
    load1 = os.getloadavg()[0]
    cp = subprocess.run(
        ["ps", "-eo", "comm,pcpu", "--no-headers"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    control_plane = [
        line.strip()
        for line in cp.stdout.splitlines()
        if any(
            k in line
            for k in (
                "kube-apiserver",
                "etcd",
                "kubelet",
                "containerd",
                "kube-controller",
                "kube-scheduler",
            )
        )
    ]
    SUMMARY["host_condition"] = {
        "idle_load1": round(load1, 2),
        "control_plane_processes": control_plane,
        "note": "root-owned K8s control plane, present during all runs",
    }
    record(
        "host-check", "PASS", f"load1={load1:.2f}, {len(control_plane)} cp procs", 1.0
    )


if __name__ == "__main__":
    main()
