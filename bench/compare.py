"""Statistical regression comparison: baseline vs candidate.

Matches cases by (suite, framework, tier, model, params) and
AGGREGATES replicate rounds per key — per-round values are the
replicates. (The original version collapsed rounds to the last one,
so --rounds improved the run summary but not this comparison.)

Statistics (per external review):
- Welch's t-test with a proper t-distribution (scipy) on RUN-LEVEL
  replicates. Within-run stability windows are NOT used as
  replicates: they capture within-run noise and understate
  run-to-run variance. The old normal approximation was
  anti-conservative at small n (3v3, ~2.4 sigma gap -> p=0.014 where
  a t-distribution gives ~0.07).
- A minimum replicate count (MIN_REPLICATES) is required for any
  p-value; below it the comparison is labeled
  "insufficient_replicates" and is informational only — never a
  regression verdict from a single point estimate.
- Holm-Bonferroni correction across the family of throughput tests:
  dozens of cells at alpha=0.05 would otherwise produce expected
  false regressions.
- p99 uses the same replicate machinery; single-round p99 deltas are
  informational (p99 is the noisiest metric here).
- Correctness cells get explicit pass/fail comparisons: a NEW
  PARITY_FAILED / SUITE_FAILED, or worsened fault row-count outcomes
  (exact down; truncated / overflow / empty / failed up) is a hard
  regression. The original version treated these cells as
  zero-throughput and never flagged them.
- Manifests are compared first (image digest, host, load profile);
  a mismatch refuses the comparison — a regression verdict across
  different environments is not a verdict.

Non-zero exit on any regression.
"""

import json
from pathlib import Path
from typing import Any

from scipy import stats

SLA_DEFAULTS = {"throughput_delta": -0.10, "p99_delta": 0.10}
"""Default SLA raised from -5% to -10%: the demonstrated detection
floor is ~-10% (graded validation detected -9.8% to -19%; nothing
below -9.8% was tested; run-drift reaches ±6%). A -5% SLA would
report changes indistinguishable from noise as verdicts. Raise
further or build the paired driver if smaller effects matter."""

P_VALUE = 0.05
MIN_REPLICATES = 3
"""Rounds per cell required for a p-value. Below this, comparisons
are informational (the review's validation procedure recommends at
least 5)."""


def load(path: Path) -> dict[str, Any]:
    return json.loads((path / "results.json").read_text())


def case_key(case: dict[str, Any]) -> str:
    params = case.get("params") or {}
    stable = sorted(
        (k, v) for k, v in params.items() if isinstance(v, (str, int, float, bool))
    )
    return "|".join(
        [
            case.get("suite", "?"),
            case.get("framework", "?"),
            case.get("tier", "?"),
            case.get("model", "?"),
            ";".join(f"{k}={v}" for k, v in stable),
        ]
    )


def group_cases(
    cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group ALL cases per key — replicate rounds stay in the list
    (sorted by round index) instead of collapsing to the last one."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        result = case.get("result")
        if not result or result.get("status", "").startswith("SKIPPED"):
            continue
        grouped.setdefault(case_key(case), []).append(case)
    for group in grouped.values():
        group.sort(key=lambda c: c.get("round") or 0)
    return grouped


def _zero_variance(values: list[float]) -> bool:
    """True when the replicates are identical (or a single value) —
    the t-test is meaningless there and scipy warns about
    catastrophic cancellation."""
    return len(set(values)) <= 1


def replicate_values(group: list[dict[str, Any]], metric: str) -> list[float]:
    """Per-round values of a metric (one per round; rounds missing the
    metric are skipped)."""
    values = []
    for case in group:
        value = case.get("result", {}).get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


# ---------------------------------------------------------------------------
# Statistics (scipy-backed; the t-distribution replaces the old normal
# approximation, which was anti-conservative at small n)
# ---------------------------------------------------------------------------


def welchs_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t statistic and two-sided p-value (t-distribution)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    import warnings

    with warnings.catch_warnings():
        # Near-identical samples trigger scipy's catastrophic-
        # cancellation RuntimeWarning — benign for the p-value.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.ttest_ind(a, b, equal_var=False)
    t = float(result.statistic)
    p = float(result.pvalue)
    if p != p:  # NaN (e.g. zero variance both sides)
        return t, 1.0
    return t, p


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U (exact for small samples — the old normal
    approximation claimed p=0.0495 for complete 3v3 separation, whose
    exact minimum two-sided p is 0.1: never significant at 0.05)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    try:
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down). Controls the
    family-wise error rate across the compared cells."""
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (original_index, p) in enumerate(indexed):
        running = max(running, (m - rank) * p)
        adjusted[original_index] = min(1.0, running)
    return adjusted


# ---------------------------------------------------------------------------
# Environment gate
# ---------------------------------------------------------------------------


def manifest_mismatches(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    allow_image_change: bool = False,
) -> list[str]:
    """Fields that must match for a comparison to be a verdict.

    allow_image_change: permits the image DIGEST to differ (baseline
    vs optimized-build comparisons — the entire point of a baseline).
    Host, target and load-profile are still gated."""
    problems = []
    for payload, label in ((baseline, "baseline"), (candidate, "candidate")):
        if not payload.get("manifest"):
            problems.append(f"{label}: no manifest recorded")
    if problems:
        return problems

    def field(payload: dict, path: str) -> Any:
        value: Any = payload["manifest"]
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    gated_fields = (
        "image.target",
        "host.cpu_model",
        "host.cores_logical",
    )
    if not allow_image_change:
        gated_fields = ("image.digest",) + gated_fields

    for path in gated_fields:
        base_value = field(baseline, path)
        cand_value = field(candidate, path)
        # Fields must be PRESENT: a field missing on both sides used
        # to pass silently (None == None) — an ungated comparison.
        if base_value is None:
            problems.append(f"{path}: missing in baseline manifest")
        elif cand_value is None:
            problems.append(f"{path}: missing in candidate manifest")
        elif base_value != cand_value:
            problems.append(f"{path}: baseline={base_value!r} candidate={cand_value!r}")
    base_profile = (baseline.get("config") or {}).get("load_profile")
    cand_profile = (candidate.get("config") or {}).get("load_profile")
    if base_profile != cand_profile:
        problems.append(
            f"load_profile: baseline={base_profile!r} candidate={cand_profile!r}"
        )
    return problems


# ---------------------------------------------------------------------------
# Correctness cells (explicit pass/fail — never zero-throughput)
# ---------------------------------------------------------------------------

FAULT_OUTCOME_KEYS = ("exact", "truncated", "overflow_leakage", "empty")
"""Row-count outcomes, faults suite only. NOTE: "failed" is NOT here —
every load result carries a failed count, so including it made the
correctness comparison fire on EVERY throughput cell (a single
transient error inside the 0.1% guard became a hard regression, and
summed totals made 3-vs-5 rounds with identical per-round failures
flag too). Load errors are the error guard's job."""
FAULT_WORKER_KILL_KEYS = ("load_failed", "outage_duration_ms")
"""Worker-kill outcomes, faults suite only (lower is better)."""
WORKER_KILL_TOLERANCE = 0.5
"""Relative worsening required before a worker-kill outcome flags —
these are single noisy measurements per round, so a modest change
(e.g. a 4.8s -> 5.5s outage) must not read as a regression."""
FAILURE_STATUSES = ("PARITY_FAILED", "SUITE_FAILED", "FAILED_CORRECTNESS")


def correctness_comparison(
    base_group: list[dict[str, Any]], cand_group: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Compare correctness-bearing cells. Returns a verdict dict when
    the cell carries correctness signal, else None.

    Scope (per review — the original fired on EVERY throughput cell):
    - failure STATUSES (PARITY_FAILED / SUITE_FAILED / ...): any cell
    - row-count and worker-kill outcomes: the FAULTS SUITE ONLY —
      every load result carries a failed count, so outcome comparison
      without the suite gate turned transient load errors into hard
      regressions
    - outcomes compare PER-ROUND MEANS, not summed totals (3-vs-5
      rounds with identical per-round failures must not flag)
    """
    base_failures = {
        status
        for case in base_group
        if (status := case.get("result", {}).get("status")) in FAILURE_STATUSES
    }
    cand_failures = {
        status
        for case in cand_group
        if (status := case.get("result", {}).get("status")) in FAILURE_STATUSES
    }
    new_failures = sorted(cand_failures - base_failures)

    is_faults_cell = any(
        case.get("suite") == "faults" for case in base_group + cand_group
    )

    def outcome_means(
        group: list[dict[str, Any]], keys: tuple[str, ...]
    ) -> dict[str, float] | None:
        per_round: list[dict[str, int]] = []
        for case in group:
            result = case.get("result", {})
            if not any(k in result for k in keys):
                continue
            round_outcomes = {}
            for key in keys:
                value = result.get(key)
                if isinstance(value, int):
                    round_outcomes[key] = value
            if round_outcomes:
                per_round.append(round_outcomes)
        if not per_round:
            return None
        return {
            key: sum(r.get(key, 0) for r in per_round) / len(per_round) for key in keys
        }

    outcome_keys = FAULT_OUTCOME_KEYS + FAULT_WORKER_KILL_KEYS
    base_outcomes = outcome_means(base_group, outcome_keys) if is_faults_cell else None
    cand_outcomes = outcome_means(cand_group, outcome_keys) if is_faults_cell else None

    if not new_failures and base_outcomes is None and cand_outcomes is None:
        return None

    regression = bool(new_failures)
    outcome_changes: dict[str, dict[str, float]] = {}
    if base_outcomes is not None and cand_outcomes is not None:
        for key in outcome_keys:
            base_value = base_outcomes.get(key, 0)
            cand_value = cand_outcomes.get(key, 0)
            if base_value != cand_value:
                outcome_changes[key] = {
                    "baseline": round(base_value, 2),
                    "candidate": round(cand_value, 2),
                }
                if key == "exact":
                    regression = regression or cand_value < base_value
                elif key in FAULT_WORKER_KILL_KEYS:
                    # Worker-kill outcomes are single noisy measurements
                    # per round — a modest change (4.8s -> 5.5s outage)
                    # must not flag. Only large worsenings do.
                    if base_value > 0:
                        regression = regression or cand_value > base_value * (
                            1 + WORKER_KILL_TOLERANCE
                        )
                    else:
                        regression = regression or cand_value > 0
                else:
                    # row-count outcomes: lower is better, any worsening
                    regression = regression or cand_value > base_value

    return {
        "new_failure_statuses": new_failures,
        "outcome_changes": outcome_changes or None,
        "regression": regression,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    baseline_dir: Path,
    candidate_dir: Path,
    sla: dict[str, float] | None = None,
    allow_image_change: bool = False,
) -> dict[str, Any]:
    sla = {**SLA_DEFAULTS, **(sla or {})}
    baseline_payload = load(baseline_dir)
    candidate_payload = load(candidate_dir)

    mismatches = manifest_mismatches(
        baseline_payload, candidate_payload, allow_image_change
    )
    if mismatches:
        return {
            "status": "MANIFEST_MISMATCH",
            "mismatches": mismatches,
            "note": "refusing to compare: a regression verdict across "
            "different environments (image, host or load shape) is not "
            "a verdict",
        }

    baseline = group_cases(baseline_payload["cases"])
    candidate = group_cases(candidate_payload["cases"])
    matched = sorted(set(baseline) & set(candidate))
    verdicts: list[dict[str, Any]] = []
    regressions = 0

    # Pass 1: build verdicts; collect p-values for Holm-Bonferroni
    pending_p: list[tuple[int, float]] = []
    for index, key in enumerate(matched):
        base_group, cand_group = baseline[key], candidate[key]

        # gzip-invalid guard: cells where the gzip axis never engaged
        # measured nothing — mark invalid, don't let noise read as a
        # regression.
        gzip_invalid = any(
            (case.get("result") or {}).get("gzip_engaged") is False
            for case in base_group + cand_group
        )

        verdict: dict[str, Any] = {"case": key, "rounds": len(cand_group)}

        if gzip_invalid:
            verdict["gzip_invalid"] = (
                "the gzip axis never engaged in this cell — the "
                "measurement is a no-op configuration, not a server "
                "result (invalid, not compared)"
            )

        correctness = correctness_comparison(base_group, cand_group)
        if correctness is not None:
            verdict["correctness"] = correctness
            if correctness["regression"]:
                regressions += 1

        base_tp = replicate_values(base_group, "throughput_rps")
        cand_tp = replicate_values(cand_group, "throughput_rps")
        base_p99 = replicate_values(base_group, "latency_ms_p99")
        cand_p99 = replicate_values(cand_group, "latency_ms_p99")

        if base_tp and cand_tp and not gzip_invalid:
            base_mean = sum(base_tp) / len(base_tp)
            cand_mean = sum(cand_tp) / len(cand_tp)
            tp_delta = (cand_mean - base_mean) / base_mean if base_mean else 0.0
            sufficient = min(len(base_tp), len(cand_tp)) >= MIN_REPLICATES
            tp_entry: dict[str, Any] = {
                "baseline_mean": round(base_mean, 3),
                "candidate_mean": round(cand_mean, 3),
                "delta": round(tp_delta, 4),
                "replicates": [len(base_tp), len(cand_tp)],
            }
            if sufficient:
                if _zero_variance(base_tp) and _zero_variance(cand_tp):
                    # Near-zero variance on BOTH sides: the t-test is
                    # meaningless (and scipy warns about catastrophic
                    # cancellation). Identical-per-round data is
                    # deterministic — judge on delta only, labeled.
                    tp_entry["no_variance"] = True
                    tp_entry["note"] = (
                        "zero variance on both sides — delta only "
                        "(deterministic per-round measurements)"
                    )
                    tp_entry["regression"] = tp_delta < sla["throughput_delta"]
                    if tp_entry["regression"]:
                        regressions += 1
                else:
                    _, p = welchs_t(cand_tp, base_tp)
                    tp_entry["welch_p_raw"] = round(p, 5)
                    pending_p.append((index, p))
                    tp_entry["_pending_index"] = index
                    tp_entry["regression"] = False  # finalized later
            else:
                tp_entry["insufficient_replicates"] = True
                tp_entry["note"] = (
                    f"fewer than {MIN_REPLICATES} rounds per side — "
                    "informational only, no significance verdict"
                )
                tp_entry["regression"] = False
            verdict["throughput"] = tp_entry

        if base_p99 and cand_p99:
            base_mean = sum(base_p99) / len(base_p99)
            cand_mean = sum(cand_p99) / len(cand_p99)
            p99_delta = (cand_mean - base_mean) / base_mean if base_mean else 0.0
            sufficient = min(len(base_p99), len(cand_p99)) >= MIN_REPLICATES
            p99_entry: dict[str, Any] = {
                "baseline_mean": round(base_mean, 3),
                "candidate_mean": round(cand_mean, 3),
                "delta": round(p99_delta, 4),
            }
            if sufficient:
                if _zero_variance(base_p99) and _zero_variance(cand_p99):
                    p99_entry["no_variance"] = True
                    p99_entry["note"] = (
                        "zero variance on both sides — delta only "
                        "(deterministic per-round measurements)"
                    )
                    p99_entry["regression"] = p99_delta > sla["p99_delta"]
                    if p99_entry["regression"]:
                        regressions += 1
                else:
                    _, p = welchs_t(cand_p99, base_p99)
                    p99_entry["welch_p_raw"] = round(p, 5)
                    p99_entry["_pending_index"] = len(pending_p)
                    pending_p.append((-1 - index, p))  # negative marks p99
                    p99_entry["regression"] = False
            else:
                p99_entry["insufficient_replicates"] = True
                p99_entry["note"] = (
                    "p99 on fewer than "
                    f"{MIN_REPLICATES} rounds — informational "
                    "(p99 is the noisiest metric)"
                )
                p99_entry["regression"] = False
            verdict["p99_ms"] = p99_entry

        verdicts.append(verdict)

    # Pass 2: Holm-Bonferroni across the family of tests, then verdicts
    adjusted = holm_bonferroni([p for _, p in pending_p])
    for (index, _), adj in zip(pending_p, adjusted):
        verdict = verdicts[index if index >= 0 else -1 - index]
        section = "throughput" if index >= 0 else "p99_ms"
        entry = verdict[section]
        entry["welch_p_adjusted"] = round(adj, 5)
        if section == "throughput":
            entry["regression"] = (
                entry["delta"] < sla["throughput_delta"] and adj < P_VALUE
            )
        else:
            entry["regression"] = entry["delta"] > sla["p99_delta"] and adj < P_VALUE
        if entry["regression"]:
            regressions += 1
        entry.pop("_pending_index", None)

    result = {
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
        "matched_cases": len(matched),
        "baseline_only": sorted(set(baseline) - set(candidate)),
        "candidate_only": sorted(set(candidate) - set(baseline)),
        "regressions": regressions,
        "sla": sla,
        "minimum_replicates": MIN_REPLICATES,
        "multiple_comparison": "Holm-Bonferroni across "
        f"{len(pending_p)} significance tests",
        "verdicts": verdicts,
        "status": "REGRESSION" if regressions else "OK",
    }
    if allow_image_change:
        result["image_change_allowed"] = {
            "baseline_digest": _field(baseline_payload, "image.digest"),
            "candidate_digest": _field(candidate_payload, "image.digest"),
            "note": "image digest intentionally differs (baseline vs "
            "optimized build); host and load-profile still gated",
        }
    return result


def _field(payload: dict, path: str) -> Any:
    value: Any = payload.get("manifest", {})
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--throughput-sla", type=float, default=-0.05)
    parser.add_argument("--p99-sla", type=float, default=0.10)
    parser.add_argument(
        "--allow-image-change",
        action="store_true",
        help="permit the image digest to differ (baseline vs "
        "optimized-build comparison); host and load-profile still gated",
    )
    args = parser.parse_args()
    result = compare(
        args.baseline,
        args.candidate,
        {"throughput_delta": args.throughput_sla, "p99_delta": args.p99_sla},
        allow_image_change=args.allow_image_change,
    )
    print(json.dumps(result, indent=2, default=str)[:8000])
    (args.candidate / "comparison.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    raise SystemExit(1 if result["regressions"] else 0)
