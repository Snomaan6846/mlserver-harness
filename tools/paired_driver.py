"""Paired driver: alternating baseline/candidate rounds in one session.

The drift analysis showed ~12% host-level shifts between rounds
(inter-cell correlation 0.55-0.58). An unpaired comparison has a
~10% noise floor from drift alone. This driver cancels the drift by
running baseline-image and candidate-image rounds back-to-back,
alternating the order on odd rounds.

Design (per reviewer):
- Alternates order on odd rounds (baseline-first on even, candidate-
  first on odd) to cancel position-dependent bias — except the
  saturation suite, where the baseline ALWAYS runs first so this
  round's open-loop rates exist before the candidate side needs them
- Alternates at SUITE granularity (not whole-round) to keep the two
  sides close in time
- Drives open-loop at the BASELINE's measured rate for that round,
  keyed per framework+model+fraction+arrival (each model has its own
  peak), so both sides see the same load
- Every side runs under its own --run-id: containers are named
  mlserver-bench-{token}-{side}-r{round}-{suite}-r{n}-{i} and each
  suite writes its own round-N/{side}/{suite}/{framework}/results.json
  — no name collisions, no result overwrites, no ambiguity about
  which run/side/round/suite a container or artifact belongs to
- Records per-side manifests; comparison uses --allow-image-change
- Uses PAIRED statistics: per-round log-ratios per cell, paired
  t-test + Holm-Bonferroni (NOT Welch on pooled replicates — that
  leaves common-mode drift in the variance)

Controls before real use:
- A/A: same image on both sides → deltas scatter around 0
- Graded A/B: calibrated container CPU limit (--cpus) on the
  candidate side only — host hogs trigger connection resets (F-06) —
  with the realized degradation measured from the paired result

Usage:
  python3 tools/paired_driver.py --baseline-image IMG \
      --candidate-image IMG --rounds 5 --out results/paired-run
  python3 tools/paired_driver.py --aa --rounds 5 \
      --out results/paired-aa
  python3 tools/paired_driver.py --graded --degradation 0.10 \
      --rounds 5 --out results/paired-graded  # --cpus on candidate
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SUITES = "protocol,workers,middleware,saturation"
FRAMEWORKS = ["sklearn", "onnx"]

sys.path.insert(0, str(REPO))
from bench.config import generate_run_id  # noqa: E402

# Unique per driver invocation: stamped into every container name and
# result artifact this driver produces, so two invocations (or a
# retry after a crash) can never collide or be confused.
RUN_TOKEN = generate_run_id("pd")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------


def calibrate_cpu_limit(
    target_degradation: float,
    baseline_image: str,
    framework: str = "sklearn",
    repeats: int = 3,
) -> dict[str, Any]:
    """Calibrate a CFS quota (--cpus) against MEASURED degradation.

    Host CPU hogs — the previous graded mechanism — trigger connection
    resets on the contended side (F-06: 2/2 graded attempts aborted
    with Errno 104 over the 0.10% error guard; every un-hogged run
    clean). A container CPU limit degrades the candidate
    deterministically WITHOUT loading the host: the kernel, the docker
    proxy and the client stay unthrottled — only the server's own
    processes run slower.

    Every step is the MEDIAN of `repeats` quick-suite runs. The stop
    condition carries a 0.5% tolerance: a 9.96% measured degradation
    IS a ~10% degradation within noise (a strict >= once skipped a
    10.0% step and ran 6 hogs = 19% — contention deep enough to reset
    connections immediately). A failed bench.cli invocation RAISES.

    Returns {"cpu_limit": float, "steps": [{"cpu_limit", "degradation"}]}.
    The calibration is a STARTING POINT — the control's actual size is
    the realized degradation measured from the paired result itself.
    """
    log(f"  calibrating cpu limit for ~{target_degradation:.0%} degradation...")
    # Smaller limit = more degradation; sweep from unthrottled down.
    # The single-worker server saturates near ~1.2 cores, so limits
    # above ~2 are near-no-ops and the useful range is below that.
    candidates = [2.0, 1.6, 1.4, 1.25, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    steps: list[dict[str, Any]] = []
    baseline_rate: float | None = None
    for limit in candidates:
        if baseline_rate is None:
            baseline_rate = float(
                np.median(
                    [
                        _quick_throughput(framework, baseline_image, cpu_limit=None)
                        for _ in range(repeats)
                    ]
                )
            )
            if baseline_rate <= 0:
                raise RuntimeError(
                    "cpu-limit calibration failed: baseline quick-runs "
                    "produced no throughput"
                )
        with_limit = float(
            np.median(
                [
                    _quick_throughput(framework, baseline_image, cpu_limit=limit)
                    for _ in range(repeats)
                ]
            )
        )
        degradation = 1 - with_limit / baseline_rate
        steps.append({"cpu_limit": limit, "degradation": round(degradation, 4)})
        log(
            f"    --cpus {limit}: {degradation:.1%} degradation "
            f"(median of {repeats})"
        )
        # 0.5% tolerance: within measurement noise of the target
        if degradation >= target_degradation - 0.005:
            return {"cpu_limit": limit, "steps": steps}
    return {"cpu_limit": candidates[-1], "steps": steps}


def _quick_throughput(
    framework: str, image: str, cpu_limit: float | None = None
) -> float:
    """Run one quick protocol suite and return mean throughput.

    cpu_limit applies a CFS quota to the run's containers (the graded
    mechanism). Raises on a failed run — a silent 0.0 used to make the
    calibration fall through to the maximum degradation."""
    import tempfile

    env = {**os.environ}
    if cpu_limit is not None:
        env["PAIRED_CPU_LIMIT"] = str(cpu_limit)

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "bench.cli",
                "run",
                "--suites",
                "protocol",
                "--framework",
                framework,
                "--tiers",
                "small",
                "--profile",
                "quick",
                "--rounds",
                "1",
                "--image",
                image,
                "--output-dir",
                tmp,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=REPO,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"quick calibration run failed (rc={result.returncode}): "
                f"{result.stderr[-300:]}"
            )
        rf = Path(tmp) / "results.json"
        if rf.is_file():
            data = json.loads(rf.read_text())
            rates = [
                c["result"]["throughput_rps"]
                for c in data.get("cases", [])
                if "throughput_rps" in c.get("result", {})
            ]
            return float(np.mean(rates)) if rates else 0.0
        return 0.0


# ---------------------------------------------------------------------------
# Gates (pre/post-run checks) — output ALWAYS captured, retried once
# ---------------------------------------------------------------------------


def _extract_failures(output: str) -> list[str]:
    """Failing test / mutation names from gate output, for abort and
    INVALID messages. The FULL output lives in gates/*.log — messages
    must name what failed, not just that something did (the reviewer
    got an INVALID with the cause discarded)."""
    failures: list[str] = []
    not_caught = "NOT caught" in output
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("FAILED "):
            failures.append(s[len("FAILED ") :].split(" ")[0])
        elif not_caught and s.startswith("- "):
            failures.append(s[2:].split(" ")[0])
    return failures[:10]


def _run_gate(
    name: str, cmd: list[str], out_dir: Path, timeout: int
) -> tuple[bool, list[str]]:
    """Run one gate, capturing ALL output to {out_dir}/gates/{name}.log.

    Returns (passed, failing_names)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO, timeout=timeout
    )
    gates_dir = out_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    log_text = (
        f"$ {' '.join(cmd)}\n"
        f"exit: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )
    (gates_dir / f"{name}.log").write_text(log_text)
    return result.returncode == 0, _extract_failures(result.stdout + result.stderr)


def _run_gate_with_retry(
    name: str, cmd: list[str], out_dir: Path, timeout: int
) -> tuple[bool, bool, list[str]]:
    """Run a gate, retrying ONCE on failure.

    Gates must be reliable under host contention: a post-run mutation
    check once failed at 94 s (vs ~35 s idle) and INVALIDated a good
    run, with the cause discarded. A first-fail-retry-pass is recorded
    as a FLAKE (contention evidence, both logs kept) and does NOT
    fail the gate; only a double failure does. The full output of
    every attempt is captured in gates/."""
    passed, failures = _run_gate(name, cmd, out_dir, timeout)
    if passed:
        return True, False, failures
    log(f"  gate {name} FAILED — retrying once (contention?)")
    retry_passed, retry_failures = _run_gate(f"{name}-retry", cmd, out_dir, timeout)
    if retry_passed:
        log(f"  gate {name} retry PASSED — recorded as a flake")
        return True, True, failures or retry_failures
    return False, False, retry_failures or failures


# ---------------------------------------------------------------------------
# Side execution: fail fast, retry once, keep evidence
# ---------------------------------------------------------------------------


def _verify_completeness(
    side_name: str,
    side_dir: Path,
    rounds: int,
    suites: list[str],
    frameworks: list[str],
) -> None:
    """Belt-and-braces structural check at merge time: every (round,
    suite, framework) cell the driver ran must have contributed at
    least one case.

    Fail-fast (see _run_side_with_retry) intercepts the reachable
    scenarios within the side; this catches anything that slips past
    it between the side check and the merge — the round check alone
    counted rounds, not suites (reviewer defect 4)."""
    data = json.loads((side_dir / "results.json").read_text())
    expected = {(r, s, fw) for r in range(rounds) for s in suites for fw in frameworks}
    seen = {
        (c.get("round"), c.get("suite"), c.get("framework"))
        for c in data.get("cases", [])
    }
    missing = expected - seen
    if missing:
        raise RuntimeError(
            f"{side_name} missing suite cells (a suite produced no "
            f"cases): {sorted(missing)[:5]} of {len(missing)}"
        )


def _side_failed(side_dir: Path) -> tuple[bool, list[str]]:
    """Check a side's results.json IMMEDIATELY after run_side.

    A SUITE_FAILED suite (or a crash that produced no results) must
    stop the run within that side, not at the outcome check after
    merge — the reviewer's mini A/A wasted ~7 minutes continuing after
    a failed candidate saturation side."""
    rf = side_dir / "results.json"
    if not rf.is_file():
        return True, ["no results.json produced"]
    try:
        cases = json.loads(rf.read_text()).get("cases", [])
    except ValueError:
        return True, ["results.json unparseable"]
    if not cases:
        return True, ["suite produced no cases"]
    failures = [
        f"{c.get('suite')}/{c.get('framework', '?')}: "
        f"{str(c.get('detail', c.get('error', '')))[:100]}"
        for c in cases
        if c.get("status") == "SUITE_FAILED"
    ]
    return bool(failures), failures[:5]


def _run_side_with_retry(
    label: str,
    image: str,
    fw: str,
    round_index: int,
    out_dir: Path,
    target_qps: dict[str, float] | None,
    cpu_limit: float | None,
    suite: str,
    retries: list[dict[str, Any]],
    retried_cells: set[tuple[int, str, str, str]],
) -> Path:
    """Run one side, checking its results immediately (fail fast) and
    retrying ONCE on failure — the same contention policy as the
    gates.

    A first-fail-retry-pass is a recorded flake: the retried side is a
    measurement from a DIFFERENT MOMENT than its pair, so the pair is
    MARKED, not hidden — case['retried_side'] in the merged data, a
    retried_side flag on the comparison verdicts, and a suite_retries
    entry in the result. Both attempts' result files are kept under
    retries/. A double failure aborts."""
    run_id = f"{RUN_TOKEN}-{label}-r{round_index}"
    side_dir = out_dir / f"round-{round_index}" / label / suite / fw
    first_error = ""
    try:
        run_side(
            label,
            image,
            fw,
            round_index,
            out_dir,
            target_qps=target_qps,
            cpu_limit=cpu_limit,
            suite=suite,
            run_id=run_id,
        )
    except RuntimeError as error:
        # The CLI exits non-zero on SUITE_FAILED — the side's
        # results.json still carries the recorded failure (and the
        # container logs, if the server was at fault, are in the
        # side run's container-logs/). Keep the rc evidence and let
        # the results check below decide.
        first_error = str(error)[:200]
    failed, failures = _side_failed(side_dir)
    if first_error and not failed:
        failed, failures = True, [first_error]
    if not failed:
        return side_dir

    # Keep the failed attempt's evidence BEFORE the retry overwrites
    # results.json
    evidence = (
        out_dir / "retries" / f"r{round_index}-{label}-{suite}-{fw}-attempt1.json"
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    if (side_dir / "results.json").is_file():
        shutil.copy2(side_dir / "results.json", evidence)
    log(f"  {label}/{suite}/{fw} r{round_index}: FAILED — {failures[:2]}")
    log(f"  retrying once (contention?) — evidence kept: {evidence.name}")

    retry_error = ""
    try:
        run_side(
            label,
            image,
            fw,
            round_index,
            out_dir,
            target_qps=target_qps,
            cpu_limit=cpu_limit,
            suite=suite,
            run_id=f"{run_id}-retry2",
        )
    except RuntimeError as error:
        retry_error = str(error)[:200]
    failed2, failures2 = _side_failed(side_dir)
    if retry_error and not failed2:
        failed2, failures2 = True, [retry_error]
    if failed2:
        raise RuntimeError(
            f"side {label}/{suite}/{fw} round {round_index} failed BOTH "
            f"attempts — first: {failures[:2]}; retry: {failures2[:2]}; "
            f"attempt-1 evidence in retries/"
        )
    retries.append(
        {
            "round": round_index,
            "side": label,
            "suite": suite,
            "framework": fw,
            "first_attempt_failures": failures[:3],
            "evidence": str(evidence),
        }
    )
    retried_cells.add((round_index, label, suite, fw))
    log(
        f"  {label}/{suite}/{fw} r{round_index}: retry PASSED — "
        "recorded as a flake; the pair is marked (measurement from a "
        "different moment)"
    )
    return side_dir


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------


def merge_sides(
    out_dir: Path,
    num_rounds: int,
    suites: list[str] | None = None,
    frameworks: list[str] | None = None,
    retried_cells: set[tuple[int, str, str, str]] | None = None,
) -> tuple[Path, Path]:
    """Merge per-round per-side results into two comparable directories.

    baseline_dir/ contains all baseline rounds; candidate_dir/ contains
    all candidate rounds. Both carry the full manifest.

    Layout: round-{n}/{side}/{suite}/{framework}/results.json — every
    suite writes its OWN directory. The previous layout
    (round-{n}/{side}/{framework}) made all four suites of a side
    write the same results.json, atomically overwriting each other:
    only the last suite's data survived, silently.
    """
    baseline_dir = out_dir / "baseline"
    candidate_dir = out_dir / "candidate"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    suite_list = suites or SUITES.split(",")
    framework_list = frameworks or list(FRAMEWORKS)

    for side, target in (("baseline", baseline_dir), ("candidate", candidate_dir)):
        all_cases = []
        manifest = None
        for round_index in range(num_rounds):
            for suite in suite_list:
                for fw in framework_list:
                    side_dir = out_dir / f"round-{round_index}" / side / suite / fw
                    rf = side_dir / "results.json"
                    if not rf.is_file():
                        continue
                    data = json.loads(rf.read_text())
                    if manifest is None:
                        manifest = data.get("manifest")
                    for case in data.get("cases", []):
                        case["round"] = round_index
                        case["framework"] = fw
                        # A retried side is a measurement from a
                        # different moment than its pair — MARK it,
                        # never hide it (reviewer item 2)
                        if (
                            retried_cells
                            and (
                                round_index,
                                side,
                                suite,
                                fw,
                            )
                            in retried_cells
                        ):
                            case["retried_side"] = True
                        all_cases.append(case)

        payload = {
            "status": "COMPLETE",
            "manifest": manifest,
            "config": {"load_profile": "quick", "paired": True},
            "paired_run": {
                "token": RUN_TOKEN,
                "rounds": num_rounds,
                "suites": suite_list,
                "frameworks": framework_list,
            },
            "cases": all_cases,
        }
        (target / "results.json").write_text(json.dumps(payload, indent=2))
        log(f"  merged {side}: {len(all_cases)} cases")

    return baseline_dir, candidate_dir


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def paired_compare(
    baseline_dir: Path,
    candidate_dir: Path,
    sla_throughput: float = -0.10,
    sla_p99: float = 0.10,
    p99_informational: bool = True,
) -> dict[str, Any]:
    """Paired comparison: per-round log-ratios per cell.

    Method: paired t-test on log(candidate/baseline) per round, per
    cell. Log-ratios are symmetric around 0 (no effect), which the
    t-test handles at ANY n >= 2 (Wilcoxon needs n >= 6 for
    two-sided p < 0.05 — the original was structurally unable to
    flag anything at 3 or 5 rounds).

    Holm-Bonferroni across cells controls the family-wise error.
    Both throughput AND p99 are computed. A throughput flag requires
    BOTH the effect threshold (SLA, -10% — consistent with
    compare.py's committed default) AND Holm-adjusted significance.

    p99 is INFORMATIONAL by default: the 2-round mini A/A measured
    per-test p99 deltas from -48.7% to +114.4% under identical
    images — at that noise, a p99 gate cannot detect a realistic
    regression at 5 rounds without false positives. Every run
    records the p99 A/A scatter (p05/p50/p95/max of per-cell mean
    ratios) so a threshold can be set from MEASURED full-A/A scatter
    later; pass p99_informational=False to re-enable gating.

    Minimum rounds (reviewer's 10-seed validation of this statistic,
    ±12% shared round drift, 2% noise, 20 cells):
      -15% effect: 3 rounds flags 8/10 seeds; 5 rounds 10/10
      -10% effect: 3 rounds flags 5/10 seeds; 5 rounds 10/10
      -5% effect:  5 rounds flags 5/10 seeds; 8 rounds 10/10
    """
    from scipy import stats

    baseline = json.loads((baseline_dir / "results.json").read_text())
    candidate = json.loads((candidate_dir / "results.json").read_text())

    def group_by_cell(payload, metric):
        cells = {}
        for case in payload.get("cases", []):
            result = case.get("result") or {}
            value = result.get(metric)
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            key = (
                case.get("suite"),
                case.get("framework"),
                case.get("model"),
                json.dumps(case.get("params", {}), sort_keys=True),
            )
            rnd = case.get("round", 0)
            cells.setdefault(key, {})[rnd] = float(value)
        return cells

    verdicts = []
    raw_pairs = []
    p99_ratios: list[float] = []

    # Cells whose side was RETRIED are measurements from a different
    # moment than their pair — flag them on the verdicts (reported,
    # never hidden; see _run_side_with_retry)
    retried_keys = {
        (
            case.get("suite"),
            case.get("framework"),
            case.get("model"),
            json.dumps(case.get("params", {}), sort_keys=True),
        )
        for case in baseline.get("cases", []) + candidate.get("cases", [])
        if case.get("retried_side")
    }

    for metric, sla_delta, direction, informational in (
        ("throughput_rps", sla_throughput, "lower", False),
        ("latency_ms_p99", sla_p99, "higher", p99_informational),
    ):
        base_cells = group_by_cell(baseline, metric)
        cand_cells = group_by_cell(candidate, metric)
        matched_keys = set(base_cells) & set(cand_cells)

        for key in sorted(matched_keys):
            base_rounds = base_cells[key]
            cand_rounds = cand_cells[key]
            common = sorted(set(base_rounds) & set(cand_rounds))
            if len(common) < 2:
                continue

            log_ratios = [
                np.log(cand_rounds[r] / base_rounds[r])
                for r in common
                if base_rounds[r] > 0 and cand_rounds[r] > 0
            ]
            if len(log_ratios) < 2:
                continue

            mean_log_ratio = float(np.mean(log_ratios))
            mean_ratio = float(np.exp(mean_log_ratio))
            delta = mean_ratio - 1.0

            if np.std(log_ratios) > 0:
                _, p = stats.ttest_1samp(log_ratios, 0)
                p = float(p)
            else:
                p = 0.0 if abs(mean_log_ratio) > 1e-10 else 1.0

            if metric == "latency_ms_p99":
                p99_ratios.append(mean_ratio)
            if informational:
                # Reported, never flagged (see docstring: mini-A/A p99
                # scatter was -48.7%..+114.4% under identical images)
                effect = False
            elif direction == "lower":
                effect = delta < sla_delta
            else:
                effect = delta > sla_delta

            # Include params in the label (two cells with the same
            # suite/framework/model but different params were
            # indistinguishable — the reviewer found ambiguous labels)
            params_str = key[3].strip("{}").replace('"', "")[:60]
            cell_label = f"{key[0]}|{key[1]}|{key[2]}|{params_str}"
            metric_label = "throughput" if "throughput" in metric else "p99"
            verdict = {
                "case": cell_label,
                "metric": metric_label,
                "rounds": len(log_ratios),
                "mean_ratio": round(mean_ratio, 4),
                "delta_pct": round(delta * 100, 2),
                "p_raw": round(p, 6),
                "regression": False,
                "informational": informational,
                "retried_side": key in retried_keys,
            }
            raw_pairs.append((len(verdicts), p, effect, verdict))
            verdicts.append(verdict)

    # Holm-Bonferroni across ALL tests
    indexed = sorted(raw_pairs, key=lambda x: x[1])
    m = len(raw_pairs)
    running = 0.0
    regressions = 0
    for rank, (idx, p, effect, verdict) in enumerate(indexed):
        running = max(running, (m - rank) * p)
        adj = min(1.0, running)
        verdicts[idx]["p_adjusted"] = round(adj, 6)
        verdicts[idx]["regression"] = effect and adj < 0.05
        if verdicts[idx]["regression"]:
            regressions += 1

    p99_scatter = None
    if p99_ratios:
        p99_scatter = {
            "n": len(p99_ratios),
            "p05": round(float(np.percentile(p99_ratios, 5)), 4),
            "p50": round(float(np.percentile(p99_ratios, 50)), 4),
            "p95": round(float(np.percentile(p99_ratios, 95)), 4),
            "min": round(float(min(p99_ratios)), 4),
            "max": round(float(max(p99_ratios)), 4),
        }

    return {
        "matched_cells": len({v["case"] for v in verdicts}),
        "compared_cells": len(verdicts),
        "regressions": regressions,
        "verdicts": verdicts,
        "status": "REGRESSION" if regressions else "OK",
        "method": (
            "paired: per-round log-ratios + paired t-test + "
            f"Holm-Bonferroni across {m} tests"
        ),
        "p99_policy": (
            "informational — threshold to be set from measured "
            "full-A/A scatter (mini-A/A scatter: -48.7%..+114.4%)"
            if p99_informational
            else f"gating at +{sla_p99:.0%}"
        ),
        "p99_scatter": p99_scatter,
        "minimum_rounds_guide": {
            # Reviewer's 10-seed validation of this statistic
            # (±12% shared round drift, 2% noise, 20 cells):
            "-15% effect": "3 rounds: 8/10 seeds; 5 rounds: 10/10",
            "-10% effect": "3 rounds: 5/10 seeds; 5 rounds: 10/10",
            "-5% effect": "5 rounds: 5/10 seeds; 8 rounds: 10/10",
        },
    }


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def run_aa(
    out_dir: Path,
    image: str,
    rounds: int,
    suites: list[str] | None = None,
    frameworks: list[str] | None = None,
) -> dict:
    """A/A control: same image on both sides."""
    log(f"=== A/A control: {rounds} rounds, same image ===")
    return _run_paired(
        out_dir, image, image, rounds, suites=suites, frameworks=frameworks
    )


def run_graded(
    out_dir: Path,
    image: str,
    rounds: int,
    target_degradation: float,
    suites: list[str] | None = None,
    frameworks: list[str] | None = None,
) -> dict:
    """Graded A/B: CALIBRATED container CPU limit on the candidate side
    only (--cpus, no host load — hogs triggered F-06 in 2/2 attempts).

    The cpu limit is a calibrated STARTING POINT only — the control's
    actual size is the realized degradation measured from the paired
    result itself (mean candidate/baseline throughput ratio over
    closed-loop cells), recorded in the output."""
    log(f"=== Graded A/B: {rounds} rounds, ~{target_degradation:.0%} ===")
    calibration = calibrate_cpu_limit(target_degradation, image)
    cpu_limit = calibration["cpu_limit"]
    log(
        f"  calibrated starting point: --cpus {cpu_limit} "
        f"(steps: {calibration['steps']})"
    )
    result = _run_paired(
        out_dir,
        image,
        image,
        rounds,
        cpu_limit=cpu_limit,
        suites=suites,
        frameworks=frameworks,
    )
    result["calibrated_cpu_limit"] = cpu_limit
    result["calibration_steps"] = calibration["steps"]
    result["target_degradation"] = target_degradation
    # The control's ACTUAL size, measured (not assumed)
    result["realized_degradation"] = _realized_degradation(result)
    log(
        f"  realized degradation: "
        f"{result['realized_degradation']}"
        f" (target was {target_degradation:.0%})"
    )
    return result


def _realized_degradation(result: dict) -> float | None:
    """The graded control's measured size: mean candidate/baseline
    throughput ratio over CLOSED-LOOP cells (open-loop cells are
    rate-matched by design, so their throughput reflects rate
    adherence, not capacity)."""
    closed_ratios = [
        v["mean_ratio"]
        for v in result.get("verdicts", [])
        if v.get("metric") == "throughput" and "mode: closed" in v.get("case", "")
    ]
    if not closed_ratios:
        return None
    return round(1 - float(np.mean(closed_ratios)), 4)


def _run_paired(
    out_dir: Path,
    baseline_image: str,
    candidate_image: str,
    rounds: int,
    cpu_limit: float | None = None,
    suites: list[str] | None = None,
    frameworks: list[str] | None = None,
) -> dict:
    """Run the paired comparison and return the paired verdict.

    Pre-flight: clean tree, mutation check, crosscheck (the load
    generator is validated against ApacheBench — a generator-side
    bias would poison every paired delta).
    Alternates at SUITE level (one suite at a time per side).
    Aborts on any failed side. Verifies rounds AND per-suite
    completeness, outcome checks (SUITE_FAILED / parity /
    generator-bound), and digest-pinned manifests on both sides.
    Post-run mutation + crosscheck failures invalidate the verdict.
    """
    suites = suites or SUITES.split(",")
    frameworks = frameworks or list(FRAMEWORKS)

    # Refuse non-empty output directory (same guard as the
    # orchestrator and pilot)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to write into non-empty {out_dir} — "
            "use a fresh directory or clear it first"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep stale containers from previous CRASHED driver runs (the
    # pd- namespace only — never touches containers of other runs).
    # One paired driver at a time per host (serial discipline).
    from bench.docker_server import clean_stale_containers

    stale = clean_stale_containers(run_id="pd")
    if stale:
        log(f"  removed stale paired-driver containers: {', '.join(stale)}")
    log(f"run token: {RUN_TOKEN}")

    # Pre-flight: clean tree + mutation check
    tree = subprocess.run(
        ["git", "status", "--porcelain", "--", "bench", "tools"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if tree.stdout.strip():
        raise SystemExit(
            "refusing: bench/ or tools/ dirty. Commit first.\n" f"{tree.stdout[:200]}"
        )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    log(f"commit: {sha[:12]}")

    mut_ok, mut_flaked, mut_failures = _run_gate_with_retry(
        "preflight-mutation-check", ["make", "mutation-check"], out_dir, 2400
    )
    if not mut_ok:
        raise SystemExit(
            "pre-flight mutation check failed (both attempts) — "
            f"failing: {mut_failures or 'unknown — see gates/'}; "
            f"full output: {out_dir / 'gates' / 'preflight-mutation-check.log'}"
        )
    if mut_flaked:
        log(
            "  WARNING: pre-flight mutation check flaked once "
            "(passed on retry — contention?). Both logs in gates/"
        )

    # Pre-flight crosscheck: our load generator vs ApacheBench on the
    # same server. A generator-side bias would poison every paired
    # delta, so it is a hard gate (the docstring used to CLAIM this
    # check without running it — reviewer defect 4).
    xcheck_ok, _, xcheck_failures = _run_gate_with_retry(
        "preflight-crosscheck", ["make", "crosscheck"], out_dir, 1800
    )
    if not xcheck_ok:
        raise SystemExit(
            "pre-flight crosscheck failed (both attempts) — the load "
            "generator disagrees with ApacheBench; refusing to run. "
            f"Failing: {xcheck_failures or 'unknown — see gates/'}; "
            f"full output: {out_dir / 'gates' / 'preflight-crosscheck.log'}"
        )

    # Rate propagation: keyed by framework, model, fraction, arrival —
    # PER ROUND (the old single "{frac}:{arrival}" key meant the last
    # model's rate overwrote the others, giving every model one rate;
    # on odd rounds the candidate ran first and used stale round-0
    # rates). The dict resets each round.
    # NOTE: `suites` is the PARAMETER (normalized at the top of this
    # function). A leftover `suites = SUITES.split(",")` here used to
    # clobber it after the parameterization — --suites did nothing,
    # and the scoped mini A/A silently ran all four suites.
    baseline_qps: dict[str, float] = {}
    suite_retries: list[dict[str, Any]] = []
    retried_cells: set[tuple[int, str, str, str]] = set()

    for round_index in range(rounds):
        # Reset rates each round (stale round-0 rates on round 1
        # produced fake open-loop deltas of -6% and +8% in the A/A)
        baseline_qps = {}

        for suite in suites:
            # The saturation suite ALWAYS runs the baseline first (on
            # every round) so this round's open-loop rates exist
            # before the candidate side needs them. Other suites
            # alternate order on odd rounds for drift balance.
            if suite == "saturation" or round_index % 2 == 0:
                order = [("baseline", baseline_image), ("candidate", candidate_image)]
            else:
                order = [("candidate", candidate_image), ("baseline", baseline_image)]

            for label, image in order:
                for fw in frameworks:
                    # Graded control: the CANDIDATE side only runs
                    # under the calibrated CPU limit (baseline stays
                    # clean — the degradation is the thing measured)
                    side_limit = cpu_limit if label == "candidate" else None
                    # Fail fast + retry once: the side's results are
                    # checked IMMEDIATELY (a failed suite stops the
                    # run within that side, not at the outcome check
                    # after merge), and a single failure retries once
                    # with both attempts' evidence kept.
                    _run_side_with_retry(
                        label,
                        image,
                        fw,
                        round_index,
                        out_dir,
                        baseline_qps if label == "candidate" else None,
                        side_limit,
                        suite,
                        suite_retries,
                        retried_cells,
                    )

                # After the baseline side, record open-loop peaks
                # keyed by framework+model+fraction+arrival
                if label == "baseline":
                    for fw in frameworks:
                        side_dir = (
                            out_dir / f"round-{round_index}" / "baseline" / suite / fw
                        )
                        rf = side_dir / "results.json"
                        if not rf.is_file():
                            continue
                        data = json.loads(rf.read_text())
                        for case in data.get("cases", []):
                            if case.get("params", {}).get("mode") == "open":
                                frac = case.get("params", {}).get("fraction_of_peak", 0)
                                arrival = case.get("params", {}).get("arrival", "")
                                qps = case.get("result", {}).get("target_qps")
                                model = case.get("model", "")
                                if qps:
                                    key = f"{fw}:{model}:{frac}:{arrival}"
                                    baseline_qps[key] = qps

        log(f"  round {round_index + 1}/{rounds} complete")

    # Merge and verify round count
    baseline_dir, candidate_dir = merge_sides(
        out_dir, rounds, suites, frameworks, retried_cells
    )

    # Verify both sides have all rounds AND pass outcome checks
    sys.path.insert(0, str(REPO / "tools"))
    import local_validation as lv

    sys.path.pop(0)
    lv.OUT = out_dir

    # Every (round, suite, framework) cell the driver ran must have
    # contributed at least one case. The round check alone counted
    # rounds, not suites: a side missing one suite's data (rc=0 but
    # no cases) passed unnoticed (reviewer defect 4).
    for side_name, side_dir in (
        ("baseline", baseline_dir),
        ("candidate", candidate_dir),
    ):
        data = json.loads((side_dir / "results.json").read_text())
        round_set = {c.get("round", -1) for c in data.get("cases", [])}
        if len(round_set) < rounds:
            raise RuntimeError(
                f"{side_name} missing rounds: expected {rounds}, "
                f"got {sorted(round_set)}"
            )
        _verify_completeness(side_name, side_dir, rounds, suites, frameworks)
        # Pinned image-digest assertion: no floating tags in benchmark
        # runs — a tag that moved mid-run would compare two different
        # builds. Both sides' manifests must carry a digest.
        digest = ((data.get("manifest") or {}).get("image") or {}).get("digest", "")
        if not digest:
            raise RuntimeError(
                f"{side_name} manifest has no image digest — floating "
                "tags are not comparable (resolve the image to a digest)"
            )
        # Outcome check: SUITE_FAILED, unexpected PARITY_FAILED,
        # generator-bound (reuses the orchestrator's checker)
        ok = lv.check_run_outcomes(f"paired-{side_name}-outcomes", side_dir, 0.0)
        if not ok:
            raise RuntimeError(
                f"{side_name} failed outcome checks — see " f"{out_dir}/SUMMARY.json"
            )

    # Positive-controls scan (gzip engagement, INVALID_KILL,
    # TELEMETRY_EMPTY) across both merged sides — reported in the
    # result and written to controls-findings.json, never sliced
    # (same semantics as the orchestrator's scanner).
    result_controls = lv._scan_positive_controls(out_dir)

    # Paired comparison
    result = paired_compare(baseline_dir, candidate_dir)
    result["baseline_image"] = baseline_image
    result["candidate_image"] = candidate_image
    result["rounds"] = rounds
    result["cpu_limit"] = cpu_limit
    result["commit"] = sha
    result["controls_findings"] = result_controls
    # Retried sides are reported, never hidden: each entry names the
    # failed first attempt (with its evidence file) and the pair is
    # marked in the merged data + verdicts
    result["suite_retries"] = suite_retries
    # Under matched open-loop rates (the candidate is driven at the
    # baseline's rate), throughput mostly reflects rate adherence on
    # both sides — p99 and failures are the informative metrics for
    # those cells; throughput regressions there mean the candidate
    # could not SUSTAIN the baseline's rate (real, but a different
    # question than closed-loop capacity).
    result["open_loop_note"] = (
        "open-loop cells run at the baseline's rate on both sides: "
        "p99 and failures are the informative metrics there; a "
        "throughput regression means the rate was not sustained"
    )

    # Post-run checks INVALIDATE the verdict on DOUBLE failure
    # (recording them without affecting the verdict left
    # silently-untrustworthy results looking healthy — reviewer
    # defect 4). A single failure that passes on retry is recorded as
    # a flake (contention evidence) and does NOT invalidate — the
    # reviewer's mini A/A was INVALIDated by a contention flake with
    # the cause discarded. Full output of every attempt: gates/.
    mut_ok, mut_flaked, mut_failures = _run_gate_with_retry(
        "postrun-mutation-check", ["make", "mutation-check"], out_dir, 2400
    )
    result["post_mutation_pass"] = mut_ok
    result["post_mutation_flaked"] = mut_flaked
    if mut_failures:
        result["post_mutation_failing"] = mut_failures
    xcheck_ok, xcheck_flaked, xcheck_failures = _run_gate_with_retry(
        "postrun-crosscheck", ["make", "crosscheck"], out_dir, 1800
    )
    result["post_crosscheck_pass"] = xcheck_ok
    result["post_crosscheck_flaked"] = xcheck_flaked
    if xcheck_failures:
        result["post_crosscheck_failing"] = xcheck_failures
    if mut_flaked or xcheck_flaked:
        log(
            "  WARNING: post-run gate flake(s) — mutation: "
            f"{'FLAKED' if mut_flaked else 'ok'}, crosscheck: "
            f"{'FLAKED' if xcheck_flaked else 'ok'} — both logs in gates/"
        )
    if not (mut_ok and xcheck_ok):
        result["status"] = "INVALID"
        result["invalid_reason"] = (
            "post-run mutation check: "
            f"{'PASS' if mut_ok else 'FAIL'}"
            f" (failing: {mut_failures or 'unknown'}); "
            "post-run crosscheck: "
            f"{'PASS' if xcheck_ok else 'FAIL'}"
            f" (failing: {xcheck_failures or 'unknown'}) — "
            "the harness changed under the run; the comparison is "
            "not trustworthy. Full gate output in gates/"
        )
        log(f"  WARNING: {result['invalid_reason']}")

    (out_dir / "paired-result.json").write_text(json.dumps(result, indent=2))
    log(
        f"=== PAIRED RESULT: {result['status']} "
        f"({result['regressions']} regressions, "
        f"{result['compared_cells']} cells) ==="
    )
    return result


def run_side(
    label: str,
    image: str,
    framework: str,
    round_index: int,
    out_dir: Path,
    target_qps: dict[str, float] | None = None,
    cpu_limit: float | None = None,
    suite: str = "",
    run_id: str = "",
) -> Path:
    """Run one side of one round for one framework, ONE SUITE at a time.

    Every side runs under its own --run-id ({token}-{side}-r{round}),
    so its containers are named
    mlserver-bench-{token}-{side}-r{round}-{suite}-r{n}-{i} and its
    results land in round-{n}/{side}/{suite}/{framework}/ — unique,
    self-describing, never overwritten by the next suite.

    cpu_limit: CFS quota applied to every container this side starts
    (the graded control's degradation). Host CPU hogs — the previous
    mechanism — loaded the whole host and triggered connection resets
    on the contended side (F-06); a container-local quota degrades the
    candidate deterministically while the host, the docker proxy and
    the client stay unthrottled."""
    if cpu_limit is not None:
        log(f"  {label}/{suite}/{framework}: --cpus {cpu_limit}")

    # Suite-scoped side dir: each suite writes its OWN
    # results.json. The old layout (no suite segment) made all
    # suites of a side atomically overwrite the same file — only
    # the last suite's data survived, silently.
    side_dir = out_dir / f"round-{round_index}" / label / suite / framework
    side_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "bench.cli",
        "run",
        "--suites",
        suite,
        "--framework",
        framework,
        "--tiers",
        "small",
        "--profile",
        "quick",
        "--rounds",
        "1",
        "--image",
        image,
        "--run-id",
        run_id or f"{RUN_TOKEN}-{label}-r{round_index}",
        "--output-dir",
        str(side_dir),
    ]

    env = {**os.environ}
    if target_qps:
        env["PAIRED_TARGET_QPS"] = json.dumps(target_qps)
    if cpu_limit is not None:
        env["PAIRED_CPU_LIMIT"] = str(cpu_limit)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=REPO,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"side {label}/{suite}/{framework} round {round_index} "
            f"failed (rc={result.returncode}): {result.stderr[-300:]}"
        )

    return side_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-image", required=True)
    parser.add_argument(
        "--candidate-image", default="", help="candidate image (defaults to baseline)"
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--aa", action="store_true")
    parser.add_argument("--graded", action="store_true")
    parser.add_argument(
        "--degradation",
        type=float,
        default=0.05,
        help="target degradation for graded (0.05 = 5%%)",
    )
    parser.add_argument(
        "--suites",
        default=SUITES,
        help=f"comma-separated suites (default: {SUITES})",
    )
    parser.add_argument(
        "--frameworks",
        default=",".join(FRAMEWORKS),
        help=f"comma-separated frameworks (default: {','.join(FRAMEWORKS)})",
    )
    args = parser.parse_args()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]

    if args.aa:
        result = run_aa(
            Path(args.out),
            args.baseline_image,
            args.rounds,
            suites=suites,
            frameworks=frameworks,
        )
    elif args.graded:
        result = run_graded(
            Path(args.out),
            args.baseline_image,
            args.rounds,
            args.degradation,
            suites=suites,
            frameworks=frameworks,
        )
    else:
        if not args.candidate_image:
            parser.error("--candidate-image required unless --aa or --graded")
        result = _run_paired(
            Path(args.out),
            args.baseline_image,
            args.candidate_image,
            args.rounds,
            suites=suites,
            frameworks=frameworks,
        )

    if result["status"] == "INVALID":
        return 1
    return 1 if result["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
