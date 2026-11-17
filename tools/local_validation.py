"""Local validation orchestrator (MINIMAL plan — docs/VALIDATION_PLAN.md).

Runs every phase sequentially (the suites share ports), records a
machine-readable summary per step, and never edits the repo working
tree (frozen for the run's duration). CPU hogs for V2 are cleaned up
in a finally block. Per-phase timeouts are 4x the expected duration
so a hang cannot block the whole run.

Usage: python3 tools/local_validation.py
Output: results/val-minimal/ (LOG.txt, per-phase dirs, SUMMARY.json)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "val-minimal"
"""Output directory — parameterized via --out (THOROUGH writes to
results/val-thorough/ so it never overwrites the minimal evidence;
the original hardcoding is what let test runs erase the committed
26-phase record)."""
LOG_LINES: list[str] = []
SUMMARY: dict = {"started": datetime.now(timezone.utc).isoformat(), "phases": {}}


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG_LINES.append(line)


def record(phase: str, status: str, detail: str, seconds: float) -> None:
    SUMMARY["phases"][phase] = {
        "status": status,
        "detail": detail,
        "seconds": round(seconds, 1),
    }
    log(f"PHASE {phase}: {status} — {detail} ({seconds:.0f}s)")
    (OUT / "SUMMARY.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    (OUT / "LOG.txt").write_text("\n".join(LOG_LINES) + "\n")


def run_cmd(
    phase: str,
    cmd: list[str],
    timeout_s: float,
    cwd: Path | None = None,
    env: dict | None = None,
    expect_rc: int = 0,
) -> tuple[int, str]:
    """Run a command with a timeout; record pass/fail by EXPECTED exit
    code (V2-detection expects rc=1 — a regression IS the pass
    criterion; the original rc==0-only judging recorded it as FAIL)."""
    started = time.time()
    merged_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=merged_env,
        )
        elapsed = time.time() - started
        output = result.stdout + result.stderr
        # FULL OUTPUT SAVED: a failed phase must leave evidence. The
        # pre-flight mutation failure was undiagnosable because
        # run_cmd kept only make's last line.
        phase_log = OUT / "phase-logs" / f"{phase}.log"
        phase_log.parent.mkdir(parents=True, exist_ok=True)
        phase_log.write_text(output if output.strip() else "(no output)")
        tail = output.strip().splitlines()
        tail_line = tail[-1][:200] if tail else ""
        ok = result.returncode == expect_rc
        record(
            phase,
            "PASS" if ok else f"FAIL(rc={result.returncode}, want {expect_rc})",
            tail_line,
            elapsed,
        )
        return result.returncode, output
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        record(phase, "TIMEOUT", f"exceeded {timeout_s:.0f}s", elapsed)
        return 124, ""


def _is_f01_cell(case: dict) -> bool:
    """Known product-bug (F-01) cells — EXPECTED parity failures, not
    harness failures. Matched on CELL PARAMS, not suite name (a
    suite-wide allow-list lets unrelated parity failures in the
    batching/faults suites pass too):
      - batching: max_batch_size > 1 AND max_batch_time > 0
      - faults: declared-shape axis, dynamic shape, batching on
    """
    suite = case.get("suite")
    params = case.get("params") or {}
    if suite == "batching":
        return (
            params.get("max_batch_size", 0) > 1 and params.get("max_batch_time", 0) > 0
        )
    if suite == "faults":
        return (
            params.get("axis") == "declared-shape"
            and params.get("shape") == "dynamic"
            and bool(params.get("batching"))
        )
    return False


def check_run_outcomes(phase: str, out_dir: Path, elapsed: float) -> bool:
    """Judge a bench run by its OUTCOMES, not just its exit code: a
    run that exits 0 with SUITE_FAILED cases or UNEXPECTED
    PARITY_FAILED cells is a FAIL.

    Status reads BOTH levels: a SUITE_FAILED case carries its status
    at the CASE level with result == {} (the first version of this
    check read only result.status — a no-op for the exact defect it
    was written for, verified by the reviewer feeding the original
    artifacts back through it)."""
    results_file = out_dir / "results.json"
    if not results_file.is_file():
        record(phase, "FAIL", "no results.json produced", elapsed)
        return False
    payload = json.loads(results_file.read_text())
    problems = []
    parity_failed_expected = 0
    for case in payload.get("cases", []):
        result = case.get("result") or {}
        status = case.get("status") or result.get("status", "")
        if status == "SUITE_FAILED":
            problems.append(
                f"SUITE_FAILED in {case.get('suite')}: "
                f"{str(case.get('detail', result.get('detail', '')))[:120]}"
            )
        elif status == "PARITY_FAILED":
            if _is_f01_cell(case):
                parity_failed_expected += 1
            else:
                problems.append(
                    f"unexpected PARITY_FAILED in {case.get('suite')} "
                    f"(params: {case.get('params')})"
                )
    # Generator-bound policy (settled pre-THOROUGH per review):
    # - LADDER suites (saturation, protocol) sweep concurrency until
    #   throughput plateaus — hitting the generator ceiling at the top
    #   rungs is EXPECTED behavior; the data point "ceiling is the
    #   generator at C<N>" is a finding, not a failure. Phase stays
    #   green with the count recorded.
    # - Everywhere else, a generator-bound case is a client-limited
    #   measurement masquerading as a server result: FAIL.
    # Note: the cap is whole-process CPU averaged over warmup+window
    # together — conservative (includes warmup).
    LADDER_SUITES = ("saturation", "protocol")
    generator_bound_expected = 0
    generator_bound_unexpected = 0
    for case in payload.get("cases", []):
        if not (case.get("result") or {}).get("generator_bound"):
            continue
        if case.get("suite") in LADDER_SUITES:
            generator_bound_expected += 1
        else:
            generator_bound_unexpected += 1
    if generator_bound_unexpected:
        problems.append(
            f"{generator_bound_unexpected} GENERATOR_BOUND case(s) — "
            "client-limited measurement, not a server result"
        )
    detail = (
        f"{payload.get('status')} {len(payload.get('cases', []))} cases; "
        f"F-01 parity cells (expected): {parity_failed_expected}; "
        f"generator-ceiling ladder cells: {generator_bound_expected}"
    )
    if problems:
        record(phase, "FAIL", "; ".join(problems[:3]), elapsed)
        return False
    record(phase, "PASS", detail, elapsed)
    return True


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== PHASE 0: pre-flight (reviewer launch conditions) ===")
    # Host cleanliness
    load1 = os.getloadavg()[0]
    import psutil

    procs = psutil.cpu_percent(interval=2)
    log(f"host: load1={load1:.2f}, cpu={procs:.0f}%, cores={os.cpu_count()}")
    record(
        "preflight-host",
        "PASS" if load1 < os.cpu_count() * 0.5 else "WARN",
        f"load1={load1:.2f} cpu={procs:.0f}%",
        2.0,
    )
    # Frozen repo
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO
    )
    clean = not status.stdout.strip()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    record(
        "preflight-repo",
        "PASS" if clean else "FAIL",
        f"clean={clean} sha={sha[:12]}",
        1.0,
    )
    SUMMARY["commit"] = sha

    # STANDING RULE (reviewer): every gate is checked against a
    # RECORDED FAILING RUN before its PASS is relied on — three fixes
    # landed without evidence during this validation, each caught only
    # by feeding real failing artifacts back through the gate. Made
    # mechanical: the committed SUITE_FAILED fixture must FAIL the
    # outcome checker.
    #
    # NOTE THE POLARITY: check_run_outcomes returns False when it
    # DETECTS problems — for the fixture, False is the CORRECT answer
    # (the checker rejecting a bad artifact is the gate working). The
    # first version of this gate aborted when the checker correctly
    # failed the fixture — inverted logic that killed the THOROUGH
    # launch 3 seconds in (the fifth new-code-without-a-real-test
    # defect of this validation, caught by the launch itself).
    fixture_dir = REPO / "bench" / "tests" / "fixtures" / "suite_failed"
    # Record under a name that does NOT count as non-pass: the
    # fixture rejection IS the expected result, but
    # check_run_outcomes records it as FAIL — the bare
    # "preflight-gate-selfcheck: FAIL" was a false red (reviewer).
    checker_rejected_fixture = not check_run_outcomes(
        "preflight-gate-fixture-rejection-expected", fixture_dir, 0.0
    )
    if checker_rejected_fixture:
        record(
            "preflight-gate-selfcheck-verdict",
            "PASS",
            "outcome checker correctly REJECTED the recorded "
            "SUITE_FAILED fixture — gate verified",
            0.0,
        )
    else:
        raise SystemExit(
            "GATE SELF-CHECK FAILED: the outcome checker PASSED a "
            "recorded SUITE_FAILED artifact — do not trust any PASS "
            "from this run"
        )

    # Verdict checks 5/6 "before" (reviewer)
    run_cmd("preflight-mutation", ["make", "mutation-check"], 2400)
    run_cmd("preflight-crosscheck", ["make", "crosscheck"], 1200)


def phase1_code() -> None:
    log("=== PHASE 1: code-level ===")
    run_cmd("test", ["make", "test"], 900)
    run_cmd("lint", ["make", "lint"], 300)


def phase2_cli() -> None:
    log("=== PHASE 2: CLI + zoo regeneration ===")
    run_cmd("zoo-clean", ["make", "models-clean"], 120)
    run_cmd("zoo-regen", ["make", "models"], 900)
    run_cmd("dry-run", ["make", "dry-run", "FRAMEWORK=sklearn"], 1800)


def matrix_run(
    phase: str,
    suites: str,
    framework: str,
    tier: str,
    extra: list[str] | None = None,
    expect: float = 3600,
) -> Path:
    out_dir = OUT / phase
    cmd = [
        sys.executable,
        "-m",
        "bench.cli",
        "run",
        "--suites",
        suites,
        "--framework",
        framework,
        "--tiers",
        tier,
        "--profile",
        "quick",
        "--output-dir",
        str(out_dir),
        *(extra or []),
    ]
    started = time.time()
    run_cmd(phase, cmd, expect * 4)
    check_run_outcomes(phase + "-outcomes", out_dir, time.time() - started)
    return out_dir


def phase3_matrix() -> None:
    log("=== PHASE 3: MINIMAL suite matrix ===")
    # M1: sklearn/small, ALL suites
    matrix_run("m1-sklearn-small", "all", "sklearn", "small", expect=13000)
    # M2/M3: xgboost + lightgbm representative suites (parity gates run
    # automatically at each run's start — all their models verify)
    rep = "batching,middleware,codec,grpc,faults,processing"
    matrix_run("m2-xgboost-small", rep, "xgboost", "small", expect=3000)
    matrix_run("m3-lightgbm-small", rep, "lightgbm", "small", expect=3000)
    # M4: onnx — the 7 core suites not yet run on onnx
    core7 = "protocol,batching,workers,caching,multimodel,lifecycle,saturation"
    matrix_run("m4-onnx-small", core7, "onnx", "small", expect=2500)
    # M5: sklearn/medium tier-sensitive subset
    tier5 = "middleware,batching,codec,processing,saturation"
    matrix_run("m5-sklearn-medium", tier5, "sklearn", "medium", expect=4500)


def phase3_m6_overlay() -> None:
    """smoke-wire on the default image + binary-tensor check on the
    fork-overlay image (skip-gated measurement paths)."""
    log("=== PHASE 3/M6: smoke-wire + fork-overlay ===")
    run_cmd("m6-smoke-wire", ["make", "smoke-wire"], 900)
    # Fork-overlay binary-tensor parity check (inline, like the session
    # validation): start the overlay, send binary + JSON, compare.
    started = time.time()
    try:
        code = subprocess.run(
            [sys.executable, "-c", OVERLAY_CHECK_SCRIPT],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        ok = code.returncode == 0
        record(
            "m6-overlay-binary",
            "PASS" if ok else "FAIL",
            code.stdout.strip().splitlines()[-1][:200] if code.stdout.strip() else "",
            time.time() - started,
        )
    except subprocess.TimeoutExpired:
        record("m6-overlay-binary", "TIMEOUT", "600s", time.time() - started)


OVERLAY_CHECK_SCRIPT = r"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from bench import wire
from bench.docker_server import MLServerContainer, detect_default_image, resolve_digest

REPO = Path(".").resolve()
src = REPO / "models-zoo" / "onnx-mlp-small"
image = resolve_digest(detect_default_image("cpu", ""))
site = "/usr/local/lib/python3.12/site-packages"
copies = [
    "MLServer/mlserver/",
    "MLServer/runtimes/onnx/mlserver_onnx/",
    "MLServer/runtimes/sklearn/mlserver_sklearn/",
    "MLServer/runtimes/xgboost/mlserver_xgboost/",
    "MLServer/runtimes/lightgbm/mlserver_lightgbm/",
]
pkg = {
    "MLServer/mlserver/": "mlserver",
    "MLServer/runtimes/onnx/mlserver_onnx/": "mlserver_onnx",
    "MLServer/runtimes/sklearn/mlserver_sklearn/": "mlserver_sklearn",
    "MLServer/runtimes/xgboost/mlserver_xgboost/": "mlserver_xgboost",
    "MLServer/runtimes/lightgbm/mlserver_lightgbm/": "mlserver_lightgbm",
}
dockerfile = f"FROM {image}\n" + "".join(
    f"COPY {c} {site}/{pkg[c]}/\n" for c in copies
)
build = subprocess.run(
    [
        "podman", "build", "-t", "mlserver-bench-smoke:fork-overlay",
        "-f", "-", str(REPO.parent),
    ],
    input=dockerfile, capture_output=True, text=True, timeout=600)
assert build.returncode == 0, build.stderr[-300:]

settings = json.loads((src / "model-settings.json").read_text())
batch = dict(np.load(src / "validation.npz"))
body, hl = wire.build_binary_request(settings, batch)

with MLServerContainer(
    name="m6-overlay", image="localhost/mlserver-bench-smoke:fork-overlay",
    model_repo=src.parent, target="cpu",
    env={"MLSERVER_REST_BINARY_TENSOR": "true"},
):
    import urllib.request
    json_req = {"inputs": [{"name": "input", "shape": list(batch["input"].shape),
                            "datatype": "FP32", "data": batch["input"].tolist()}],
                "outputs": [{"name": "output"}]}
    r = urllib.request.Request(
        "http://127.0.0.1:18080/v2/models/onnx-mlp-small/infer",
        data=json.dumps(json_req).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as resp:
        json_out = np.asarray(json.loads(resp.read())["outputs"][0]["data"])
    r = urllib.request.Request(
        "http://127.0.0.1:18080/v2/models/onnx-mlp-small/infer", data=body,
        headers={"Content-Type": "application/octet-stream",
                 "Inference-Header-Content-Length": str(hl)})
    with urllib.request.urlopen(r) as resp:
        data = resp.read()
        resp_hl = int(resp.headers.get("Inference-Header-Content-Length", "0"))
        bin_out = wire.parse_binary_response(data, resp_hl)["output"]
    np.testing.assert_allclose(json_out, bin_out, rtol=1e-6)
    print("overlay binary-tensor == JSON: PARITY OK")
"""


def _scan_positive_controls(out_root: Path) -> list[str]:
    """V4-style scan across every run under out_root: INVALID_KILL,
    TELEMETRY_EMPTY, and gzip_engaged findings — read BEFORE the
    numbers. gzip_engaged is REPORTED (not just flagged): T4 exists
    partly to settle whether large-tier responses cross starlette's
    500-byte minimum, and the answer lives in the middleware cells —
    without this scan it is only available by reading them by hand.

    NO TRUNCATION: the full findings list is returned AND written to
    <out_root>/controls-findings.json. The original recorded only the
    first 6-8 findings — with 19 gzip_engaged=false entries from the
    earlier runs, T4's gzip answer could be silently dropped from the
    log. Evidence must not be sliced."""
    findings = []
    for results_file in sorted(out_root.glob("*/results.json")):
        payload = json.loads(results_file.read_text())
        for case in payload.get("cases", []):
            result = case.get("result") or {}
            label = f"{results_file.parent.name}/{case.get('suite')}"
            if result.get("status") == "INVALID_KILL":
                findings.append(f"{label}: INVALID_KILL")
            if case.get("telemetry_status") == "TELEMETRY_EMPTY":
                findings.append(f"{label}: TELEMETRY_EMPTY")
            if "gzip_engaged" in result:
                findings.append(f"{label}: gzip_engaged={result['gzip_engaged']}")
    # Full evidence to a dedicated file — never sliced
    findings_file = out_root / "controls-findings.json"
    findings_file.parent.mkdir(parents=True, exist_ok=True)
    findings_file.write_text(json.dumps(findings, indent=2) + "\n")
    return findings


def phase_thorough_after() -> None:
    """Post-run checks for THOROUGH (the 'before and after' launch
    conditions): a multi-hour run is where host drift matters most —
    without these, the thorough claim could not say the host was
    unchanged across the run."""
    log("=== THOROUGH after-checks ===")
    started = time.time()
    findings = _scan_positive_controls(OUT)
    record(
        "thorough-v4-controls",
        "INFO",
        f"{len(findings)} findings (full list in "
        f"{OUT / 'controls-findings.json'}): " + "; ".join(findings),
        time.time() - started,
    )
    run_cmd("thorough-crosscheck-after", ["make", "crosscheck"], 1200)
    run_cmd("thorough-mutation-after", ["make", "mutation-check"], 2400)


def phase4_verdicts() -> None:
    log("=== PHASE 4: verdict validation (V1-V6) ===")
    subset = "protocol,middleware"
    # V1a/V1b: two identical 5-round runs, same commit
    v1a = matrix_run(
        "v1a-self", subset, "sklearn", "small", extra=["--rounds", "5"], expect=4500
    )
    v1b = matrix_run(
        "v1b-self", subset, "sklearn", "small", extra=["--rounds", "5"], expect=4500
    )
    # compare A vs B — expect OK (exit 0)
    run_cmd(
        "v1-self-comparison",
        [sys.executable, "-m", "bench.cli", "compare", str(v1a), str(v1b)],
        300,
    )
    # V2: degraded candidate under CPU contention
    log("V2: starting CPU hogs for degraded run...")
    hogs = []
    hog_count = max(1, os.cpu_count() - 2)
    try:
        for _ in range(hog_count):
            hogs.append(
                subprocess.Popen(
                    [sys.executable, "-c", "while True: pass"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        time.sleep(2)
        load = os.getloadavg()[0]
        log(f"V2: {hog_count} hogs running, load1={load:.1f}")
        v2 = matrix_run(
            "v2-degraded",
            subset,
            "sklearn",
            "small",
            extra=["--rounds", "5"],
            expect=7000,
        )
        # compare v1a vs degraded — expect REGRESSION (exit 1)
        run_cmd(
            "v2-detection",
            [sys.executable, "-m", "bench.cli", "compare", str(v1a), str(v2)],
            300,
            expect_rc=1,  # REGRESSION DETECTED is the pass criterion
        )
    finally:
        for hog in hogs:
            hog.terminate()
        for hog in hogs:
            try:
                hog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                hog.kill()
        log(f"V2: hogs stopped ({len(hogs)})")

    # V3a: manifest fields present in every results.json
    started = time.time()
    problems = []
    for results_file in sorted(OUT.glob("*/results.json")):
        payload = json.loads(results_file.read_text())
        manifest = payload.get("manifest") or {}
        for field in ("git", "image.digest", "host.cpu_model"):
            value = manifest
            for part in field.split("."):
                value = (value or {}).get(part) if isinstance(value, dict) else None
            if not value:
                problems.append(f"{results_file.parent.name}: missing {field}")
    record(
        "v3a-manifest-fields",
        "PASS" if not problems else "FAIL",
        "; ".join(problems[:3])
        or f"all {len(list(OUT.glob('*/results.json')))} runs carry manifests",
        time.time() - started,
    )
    # V3b: tampered manifest -> exit 2
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tampered = Path(tmp) / "tampered"
        shutil.copytree(v1a, tampered)
        payload = json.loads((tampered / "results.json").read_text())
        payload["manifest"]["image"]["digest"] = "quay.io/fake@sha256:deadbeef"
        (tampered / "results.json").write_text(json.dumps(payload))
        proc = subprocess.run(
            [sys.executable, "-m", "bench.cli", "compare", str(v1a), str(tampered)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=REPO,
        )
        record(
            "v3b-manifest-gate",
            "PASS" if proc.returncode == 2 else f"FAIL(rc={proc.returncode})",
            (
                (proc.stdout + proc.stderr).strip().splitlines()[-1][:150]
                if (proc.stdout + proc.stderr).strip()
                else ""
            ),
            1.0,
        )

    # V4: positive-controls scan (read BEFORE the numbers) — shared
    # with the thorough after-checks
    started = time.time()
    findings = _scan_positive_controls(OUT)
    record(
        "v4-positive-controls",
        "INFO",
        f"{len(findings)} findings (full list in "
        f"{OUT / 'controls-findings.json'}): " + "; ".join(findings),
        time.time() - started,
    )

    # V5/V6: cross-check + mutation after
    run_cmd("v5-crosscheck-after", ["make", "crosscheck"], 1200)
    run_cmd("v6-mutation-after", ["make", "mutation-check"], 2400)


def phase5_wrapup() -> None:
    log("=== PHASE 5: wrap-up ===")
    run_cmd("make-clean", ["make", "clean"], 120)
    SUMMARY["finished"] = datetime.now(timezone.utc).isoformat()
    (OUT / "SUMMARY.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    (OUT / "LOG.txt").write_text("\n".join(LOG_LINES) + "\n")
    fails = [
        k
        for k, v in SUMMARY["phases"].items()
        if v["status"] not in ("PASS", "INFO", "WARN")
    ]
    log(
        f"ORCHESTRATOR DONE: {len(SUMMARY['phases'])} phases, "
        f"{len(fails)} non-pass: {fails}"
    )


def _validate_out_dir(out: Path, force: bool) -> None:
    """Refuse to write into a non-empty directory without --force.

    The original --out DEFAULT was results/val-minimal — running with
    no flag rewrote the committed evidence (the hardcoding moved from
    the code into the default; reviewer catch). Now --out is REQUIRED
    and a non-empty target needs an explicit --force."""
    if out.exists() and any(out.iterdir()) and not force:
        existing = sorted(p.name for p in out.iterdir())[:5]
        raise SystemExit(
            f"refusing to write into non-empty {out} "
            f"(contains: {', '.join(existing)}...) — pass --force to "
            "overwrite"
        )


def phase3_thorough() -> None:
    """THOROUGH matrix (VALIDATION_PLAN.md T1-T4): the framework x
    suite cross-product that MINIMAL skipped, plus the large-tier
    check. Judged by the same outcome checker as the minimal phases.

    Planning estimate caveat: the VALIDATION_PLAN priced THOROUGH at
    ~7.5h using the case-count estimator that overcounted ~3x — the
    measured minimal pace suggests ~1.5-2h. Neither figure is a
    measurement; this note exists so nobody quotes the plan number as
    one."""
    log("=== PHASE 3 (THOROUGH): T1-T4 ===")
    # T1: xgboost/small, ALL 14 suites
    matrix_run("t1-xgboost-small-all", "all", "xgboost", "small", expect=2000)
    # T2: lightgbm/small, ALL 14 suites
    matrix_run("t2-lightgbm-small-all", "all", "lightgbm", "small", expect=2000)
    # T3: onnx — the non-core suites fresh on the final commit
    matrix_run(
        "t3-onnx-small-rest",
        "framework,codec,grpc,processing,faults",
        "onnx",
        "small",
        expect=1500,
    )
    # T4: large-tier spot check (tier-sensitive subset; also settles
    # whether large-tier responses exceed 500B and gzip finally engages)
    matrix_run(
        "t4-sklearn-large",
        "middleware,batching,codec,processing,saturation",
        "sklearn",
        "large",
        expect=2500,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        help="output directory (minimal evidence lives in "
        "results/val-minimal; THOROUGH uses results/val-thorough)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow writing into a non-empty output directory",
    )
    parser.add_argument(
        "--plan",
        choices=["minimal", "thorough"],
        default="minimal",
        help="minimal: M1-M6 + V1-V6 verdicts; thorough: the T1-T4 "
        "framework x suite cross-product delta",
    )
    args = parser.parse_args()
    global OUT
    OUT = REPO / args.out
    _validate_out_dir(OUT, args.force)
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    if args.plan == "thorough":
        phase3_thorough()
        phase_thorough_after()
    else:
        phase1_code()
        phase2_cli()
        phase3_matrix()
        phase3_m6_overlay()
        phase4_verdicts()
    phase5_wrapup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
