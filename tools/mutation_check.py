"""Mutation checks: prove the regression tests have teeth.

For each harness bug fixed during the review cycle, reintroduce the
ORIGINAL bug via a targeted mutation, run the corresponding
regression test, and confirm it FAILS. A test that never fails on the
bug it claims to catch is decoration (the original 170 tests passed
while every bug existed; 211 passed while F-04 was wrong).

Safety: every mutation runs inside a DISPOSABLE GIT WORKTREE checked
out from HEAD (created in the repo's parent directory, so
sibling-relative paths like ../MLServer still resolve). The real
working tree is never touched — a SIGKILL mid-run can only orphan
the worktree directory (removable with `git worktree prune`), never
leave a mutated file where it could be committed. A startup check
refuses to run when the real tree is dirty: the worktree reflects
HEAD, so uncommitted changes would not be what gets checked.

A mutation counts as CAUGHT only when the unmutated baseline passes
AND the mutated run fails with real test failures (pytest exit 1, no
import/syntax/name errors — a broken mutation is not a catch; the
marker list is string-based and errs toward a visible NOT-CAUGHT).
Exit code 0 only when every mutation is CAUGHT.

Usage: python3 tools/mutation_check.py [--filter SUBSTRING]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIN_DECLARED_MUTATIONS = 37
"""Pinned minimum: a splice once silently swallowed a mutation and
the loss was invisible because every remaining mutation passed
(22/22 looked healthy). If a refactor drops the declared count below
this, fail loudly instead of shrinking the coverage unnoticed."""


class Mutation:
    def __init__(
        self,
        name,
        file,
        find,
        replace,
        test,
        description="",
        extra: list[tuple[str, str]] | None = None,
    ):
        self.name = name
        self.file = file
        self.find = find
        self.replace = replace
        self.test = test
        self.description = description
        # Layered fixes need multiple simultaneous regressions to
        # reproduce the original bug — extra (find, replace) pairs
        # applied alongside the primary one.
        self.extra = extra or []


MUTATIONS = [
    Mutation(
        name="bug1-summary-window-counts-all",
        file="bench/loadgen.py",
        find="""        if window is not None:
            samples = [s for s in self.samples if _in_window(s, window)]
        else:
            samples = self.samples""",
        replace="""        samples = self.samples""",
        test="bench/tests/test_regression_review.py::TestSummaryWindow",
        description="summary(window) counts warmup samples but divides "
        "by window length (420 rps for a true 100 rps)",
    ),
    Mutation(
        name="bug1b-with-ref-seconds",
        file="bench/loadgen.py",
        find="def _with_ref(samples: list[Sample], t0_ns: int) -> list[Sample]:",
        replace=(
            "def _with_ref(samples: list[Sample], t0_ns: float) "
            "-> list[Sample]:\n"
            "    t0_ns = t0_ns / 1e9  # BUG: seconds, breaks window filters"
        ),
        test=(
            "bench/tests/test_regression_review.py::TestSummaryWindow"
            "::test_window_filter_actually_filters"
        ),
        description="_with_ref stores seconds while end_ns is ns — window "
        "filters silently match nothing (vacuous gate)",
    ),
    Mutation(
        name="bug2-class-level-session",
        file="bench/wire.py",
        find="""    def __init__(
        self,
        model: str,
        body: bytes,
        header_length: int,""",
        replace="""    _session: Any = None

    def __init__(
        self,
        model: str,
        body: bytes,
        header_length: int,""",
        test="bench/tests/test_regression_review.py::TestBinarySenderSessions",
        description="BinaryTensorSender class-level session shared across "
        "event loops ('Event loop is closed')",
    ),
    Mutation(
        name="bug4-workers-or-none",
        file="bench/suites.py",
        find="env=env, parallel_workers=workers):",
        replace="env=env, parallel_workers=workers or None):",
        test="bench/tests/test_regression_review.py::TestWorkersZero",
        description="parallel_workers=workers or None turns rung 0 into "
        "rung 1 (server default 1)",
    ),
    Mutation(
        name="bug5-thread-env-dead-onnx",
        file="bench/suites.py",
        find='    if implementation.startswith("mlserver_onnx"):',
        replace=(
            "    if any(\n"
            '        m.startswith("mlserver_onnx") for m in [config.target]\n'
            "    ):"
        ),
        test="bench/tests/test_regression_review.py::TestThreadEnv",
        description="_thread_env matched mlserver_onnx against "
        "config.target (cpu/gpu): dead code, ORT options never applied",
    ),
    Mutation(
        name="bug6-probe-only-session-options",
        file="bench/suites.py",
        find="""                if not onnx_support:
                    onnx_support.update(
                        probe_onnx_settings(server.name, *ONNX_PROBE_KEYS)
                    )""",
        replace="""                if not onnx_support:
                    onnx_support.update(
                        probe_onnx_settings(server.name, "session_options")
                    )""",
        test="bench/tests/test_regression_review.py::TestFrameworkProbeCoverage",
        description="framework suite probes only session_options first; "
        "later re-probe guards never run — every other axis wrongly "
        "recorded SKIPPED_UNSUPPORTED_IMAGE",
    ),
    Mutation(
        name="bug14-grpc-chunk-count",
        file="bench/wire.py",
        find="""        if response.raw_output_contents:
            bytes_out = sum(len(chunk) for chunk in response.raw_output_contents)
        else:
            bytes_out = response.ByteSize()
        return True, bytes_out, len(self._request_bytes)""",
        replace="""        return True, len(response.raw_output_contents or []), len(
            self._request_bytes
        )""",
        test="bench/tests/test_regression_review.py::TestGrpcBytes",
        description="GrpcSender reported a chunk COUNT as bytes_out",
    ),
    Mutation(
        name="bug15-only-servererror-contained",
        file="bench/runner.py",
        find="""    except (ParityError, RunAborted):
        raise
    except Exception as error:""",
        replace="""    except (ParityError, RunAborted):
        raise
    except KeyError as error:""",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestSuiteFailureContainment::test_runtimeerror_recorded_not_fatal"
        ),
        description="runner contained only one exception type per suite — "
        "a single RuntimeError (error guard) killed the whole run. "
        "KeyError keeps the mutation faithful (always defined): the "
        "RuntimeError escapes exactly as the original bug did",
    ),
    Mutation(
        name="f04-kill-target-mlserver-grep",
        file="bench/suites.py",
        find="*spawn_main*) [ $p -ne $$ ] && echo $p;; esac; done",
        replace=(
            "grep -q mlserver /proc/$p/cmdline 2>/dev/null "
            "&& [ $p -ne 1 ] && echo $p; done"
        ),
        test=(
            "bench/tests/test_regression_review.py::TestWorkerKillLogic"
            "::test_worker_kill_targeting_and_controls"
        ),
        description="worker discovery grepped cmdline for 'mlserver' — "
        "matched only the discovery shell; the kill never landed",
    ),
    Mutation(
        name="f04-readiness-flag-tuple",
        file="bench/suites.py",
        find="and all(code == 200 for _, code in during)",
        replace="and all(code == 200 for code in during)",
        test=(
            "bench/tests/test_regression_review.py::TestWorkerKillLogic"
            "::test_readiness_flag_unpacks_tuples"
        ),
        description="silent-outage flag compared (ts, code) tuples to 200 "
        "— always False",
    ),
    Mutation(
        name="json-numpy-bool",
        file="bench/runner.py",
        find="json.dumps(payload, indent=2, default=_json_default)",
        replace="json.dumps(payload, indent=2)",
        test=(
            "bench/tests/test_regression_review.py::TestJsonSafety"
            "::test_atomic_write_numpy_fallback"
        ),
        description="np.bool_ from the stability gate crashed results.json " "writing",
    ),
    Mutation(
        name="bug8-caching-unique-deterministic",
        file="bench/suites.py",
        find="    factor = 1.0 + (seq % 977) / 31.0",
        replace="    factor = 1.0 + (42 % 977) / 31.0  # BUG: constant",
        test="bench/tests/test_regression_review.py::TestUniquePayload",
        description="caching 'unique' payload used one deterministic "
        "value — a second repeat, not a unique stream",
    ),
    Mutation(
        name="bug7-multimodel-duplicate-names",
        file="bench/suites.py",
        find="""    for i in range(count):
        source = model_dirs[i % len(model_dirs)]
        dest_name = f"{source.name}-{i}\"""",
        replace="""    for i, source in enumerate((model_dirs * 10)[:count]):
        dest_name = source.name""",
        test="bench/tests/test_regression_review.py::TestMultimodelRepo",
        description="multimodel (model_dirs*10)[:count] repeated directory "
        "names — copytree FileExistsError",
    ),
    Mutation(
        name="compare-rounds-collapse-to-last",
        file="bench/compare.py",
        find="""    for group in grouped.values():
        group.sort(key=lambda c: c.get("round") or 0)
    return grouped""",
        replace="""    return {k: [g[-1]] for k, g in grouped.items()}""",
        test=(
            "bench/tests/test_bench.py::TestCompare"
            "::test_rounds_aggregate_not_collapse"
        ),
        description="group_cases collapses rounds to the last one — "
        "replication invisible to the comparison",
    ),
    Mutation(
        name="compare-min-replicates-1",
        file="bench/compare.py",
        find="MIN_REPLICATES = 3",
        replace="MIN_REPLICATES = 1",
        test=(
            "bench/tests/test_bench.py::TestCompare"
            "::test_single_round_is_informational_not_regression"
        ),
        description="single-round cells get significance verdicts "
        "again — a 20% drop from one point estimate flags",
    ),
    Mutation(
        name="compare-no-holm-correction",
        file="bench/compare.py",
        find="""    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (original_index, p) in enumerate(indexed):
        running = max(running, (m - rank) * p)
        adjusted[original_index] = min(1.0, running)
    return adjusted""",
        replace="""    return list(p_values)""",
        test="bench/tests/test_bench.py::TestCompare::test_holm_bonferroni_monotone",
        description="Holm-Bonferroni removed — raw p-values used, "
        "expected false regressions uncontrolled",
    ),
    Mutation(
        name="compare-no-manifest-gate",
        file="bench/compare.py",
        find=(
            "mismatches = manifest_mismatches(\n"
            "        baseline_payload, candidate_payload, allow_image_change\n"
            "    )"
        ),
        replace="mismatches = []  # gate removed",
        test="bench/tests/test_bench.py::TestCompare::test_manifest_mismatch_refuses",
        description="environment gate removed — regressions judged "
        "across different images/hosts",
    ),
    Mutation(
        name="compare-correctness-fires-on-every-cell",
        file="bench/compare.py",
        find='FAULT_OUTCOME_KEYS = ("exact", "truncated", "overflow_leakage", "empty")',
        replace=(
            "FAULT_OUTCOME_KEYS = "
            '("exact", "truncated", "overflow_leakage", "empty", "failed")'
        ),
        test=(
            "bench/tests/test_bench.py::TestCompare"
            "::test_transient_load_error_is_not_correctness_regression"
        ),
        description="outcome comparison fires on every throughput cell — "
        "'failed' back in the outcome keys AND the faults-suite gate "
        "forced on (the fix is layered: either regression alone is "
        "harmless, so the mutation reintroduces both)",
        extra=[("is_faults_cell = any(", "is_faults_cell = True or any(")],
    ),
    Mutation(
        name="varying-sender-missing-attribute",
        file="bench/suites.py",
        find="""        # Must be set BEFORE _ensure_session reads it (the caching-suite
        # defect: this attribute was missing, every request raised
        # AttributeError and the error guard aborted the suite — caught
        # by the reviewer reading the committed evidence, missed by the
        # exit-code-judging orchestrator)
        self._accept_encoding = accept_encoding
        self._session: Any = None""",
        replace="""        # (mutation: the _accept_encoding assignment removed)
        self._session: Any = None""",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestEverySenderAgainstStubServer::test_varying_json_sender"
        ),
        description="VaryingJsonSender constructor missing _accept_encoding "
        "— every caching-suite request raised AttributeError and the "
        "error guard aborted the suite (rewritten against the current "
        "pool-based class after the original find-text was orphaned by "
        "the 64-body pool change — the strict rule correctly reported "
        "the stale mutation as NOT CAUGHT rather than silently passing)",
    ),
    Mutation(
        name="outcome-check-result-level-only",
        file="tools/local_validation.py",
        find='        status = case.get("status") or result.get("status", "")',
        replace='        status = result.get("status", "")',
        test=(
            "bench/tests/test_regression_review.py::TestOutcomeJudging"
            "::test_suite_failed_case_level_detected"
        ),
        description="outcome checker reads only result.status — "
        "SUITE_FAILED cases (case-level status, empty result) pass "
        "silently: the exact no-op the reviewer proved by feeding the "
        "original artifacts back through the first fix",
    ),
    Mutation(
        name="compare-worker-kill-no-tolerance",
        file="bench/compare.py",
        find="WORKER_KILL_TOLERANCE = 0.5",
        replace="WORKER_KILL_TOLERANCE = 0.0",
        test=(
            "bench/tests/test_bench.py::TestCompare"
            "::test_worker_kill_modest_change_not_regression"
        ),
        description="worker-kill outcomes flag on ANY worsening — a "
        "modest noisy change (4.8s -> 5.5s outage) reads as a "
        "regression",
    ),
    Mutation(
        name="compare-zero-variance-p-value",
        file="bench/compare.py",
        find="    return len(set(values)) <= 1",
        replace="    return False",
        test=(
            "bench/tests/test_bench.py::TestCompare" "::test_zero_variance_delta_only"
        ),
        description="zero-variance handling removed — identical-per-round "
        "data goes to the t-test (NaN p-value, suppressed warning, no "
        "regression verdict) instead of delta-only judgment",
    ),
    Mutation(
        name="compare-gzip-invalid-guard-removed",
        file="bench/compare.py",
        find="if base_tp and cand_tp and not gzip_invalid:",
        replace="if base_tp and cand_tp:",
        test=(
            "bench/tests/test_bench.py::TestWizard"
            "::test_gzip_invalid_cell_not_compared"
        ),
        description="gzip-invalid guard removed — no-op cells "
        "(gzip_engaged=False) are compared and their noise can read "
        "as a regression",
    ),
    Mutation(
        name="compare-digest-gate-always-off",
        file="bench/compare.py",
        find='gated_fields = ("image.digest",) + gated_fields',
        replace="pass  # digest gate removed",
        test=(
            "bench/tests/test_bench.py::TestCompare" "::test_manifest_mismatch_refuses"
        ),
        description="image digest never gated — cross-image "
        "comparisons pass silently",
    ),
    Mutation(
        name="open-loop-keyed-by-target-qps",
        file="bench/suites.py",
        find="""                    case = base_case(
                        "saturation",
                        model,
                        mode="open",
                        fraction_of_peak=fraction,
                        arrival=arrival,
                    )
                    case["result"] = result.summary()
                    case["result"]["target_qps"] = rate""",
        replace="""                    case = base_case(
                        "saturation",
                        model,
                        mode="open",
                        target_qps=rate,
                        fraction_of_peak=fraction,
                        arrival=arrival,
                    )
                    case["result"] = result.summary()""",
        test=(
            "bench/tests/test_regression_review.py::TestOpenLoopCellKeying"
            "::test_saturation_cases_keyed_correctly"
        ),
        description="open-loop cells keyed by target_qps (derived from "
        "each round's peak — differs every round, so compare.py can "
        "never pair them across runs; 30 unpaired cells per framework "
        "in the baseline)",
    ),
    Mutation(
        name="paired-driver-always-flag",
        file="tools/paired_driver.py",
        find='        verdicts[idx]["regression"] = effect and adj < 0.05',
        replace='        verdicts[idx]["regression"] = True',
        test=(
            "bench/tests/test_regression_review.py::TestPairedStatistics"
            "::test_null_case_no_false_positive"
        ),
        description="regression flag set to unconditionally True — "
        "every cell flags; the null case (no effect) produces "
        "false positives on every comparison",
    ),
    Mutation(
        name="paired-driver-single-key-rates",
        file="tools/paired_driver.py",
        find='key = f"{fw}:{model}:{frac}:{arrival}"',
        replace='key = f"{frac}:{arrival}"  # single key: last model wins',
        test=(
            "bench/tests/test_regression_review.py::TestPairedDriverSmoke"
            "::test_driver_loop_passes_correct_args"
        ),
        description="rate-propagation key loses framework+model — the "
        "last model's rate overwrites the others and every model is "
        "driven at one rate; open-loop deltas appear from nothing "
        "(the reviewer measured fake -6%/+8% deltas in a real A/A)",
    ),
    Mutation(
        name="paired-driver-side-dir-loses-suite",
        file="tools/paired_driver.py",
        find=(
            'side_dir = out_dir / f"round-{round_index}" / label / suite ' "/ framework"
        ),
        replace=('side_dir = out_dir / f"round-{round_index}" / label / framework'),
        test=(
            "bench/tests/test_regression_review.py::TestPairedDriverSmoke"
            "::test_run_side_suite_scoped_dir_and_run_id"
        ),
        description="side dir loses the suite segment — all four suites "
        "of a side atomically overwrite the same results.json and only "
        "the last suite's data survives, silently (merge then sees one "
        "suite instead of four)",
    ),
    Mutation(
        name="paired-driver-gate-output-discarded",
        file="tools/paired_driver.py",
        find='    (gates_dir / f"{name}.log").write_text(log_text)',
        replace="    pass  # gate output discarded",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage::test_gate_output_captured_to_files"
        ),
        description="gate output no longer written to gates/ — a failed "
        "post-run mutation check INVALIDates the run with the CAUSE "
        "discarded (the reviewer could not establish why their mini "
        "A/A came back INVALID; 94 s vs ~35 s suggested contention, "
        "but nothing was recorded)",
    ),
    Mutation(
        name="paired-driver-gate-no-retry",
        file="tools/paired_driver.py",
        find="""    passed, failures = _run_gate(name, cmd, out_dir, timeout)
    if passed:
        return True, False, failures
    log(f"  gate {name} FAILED — retrying once (contention?)")""",
        replace="""    passed, failures = _run_gate(name, cmd, out_dir, timeout)
    return passed, False, failures
    log(f"  gate {name} FAILED — retrying once (contention?)")""",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage"
            "::test_flaked_gate_retries_and_keeps_verdict"
        ),
        description="gate retry removed — a single contention flake "
        "(mutation check at 94 s vs ~35 s idle) invalidates an "
        "otherwise good multi-hour run; gates are not reliable under "
        "host contention without a retry",
    ),
    Mutation(
        name="paired-driver-fail-fast-removed",
        file="tools/paired_driver.py",
        find="""    failed, failures = _side_failed(side_dir)
    if first_error and not failed:
        failed, failures = True, [first_error]
    if not failed:
        return side_dir""",
        replace="""    failed, failures = _side_failed(side_dir)
    if first_error and not failed:
        failed, failures = True, [first_error]
    if True:
        return side_dir""",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage"
            "::test_fail_fast_suite_failed_aborts_within_side"
        ),
        description="side results no longer checked after run_side — a "
        "SUITE_FAILED suite is only noticed at the post-merge outcome "
        "check, ~7 wasted minutes in the reviewer's mini A/A (fail "
        "fast, reviewer item 1)",
    ),
    Mutation(
        name="paired-driver-retry-removed",
        file="tools/paired_driver.py",
        find="""    retry_error = ""
    try:
        run_side(""",
        replace="""    retry_error = ""
    raise RuntimeError("no retry")
    try:
        run_side(""",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage"
            "::test_suite_retry_flake_completes_and_marks_pair"
        ),
        description="SUITE_FAILED retry removed — a single transient "
        "failure (connection reset on the first case after server "
        "start) aborts the whole multi-hour run instead of being "
        "recorded as a flake (reviewer item 2)",
    ),
    Mutation(
        name="paired-driver-cpu-limit-ignored",
        file="bench/suites.py",
        find=(
            "        self._paired_cpu_limit: float | None = None\n"
            '        raw_limit = os.environ.get("PAIRED_CPU_LIMIT", "")'
        ),
        replace=(
            "        self._paired_cpu_limit: float | None = None\n"
            '        raw_limit = ""  # env ignored'
        ),
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage::test_paired_cpu_limit_env_plumbing"
        ),
        description="PAIRED_CPU_LIMIT env ignored — the graded control's "
        "candidate runs unthrottled, degrades nothing, and the control "
        "reports a false no-regression",
    ),
    Mutation(
        name="paired-driver-suites-flag-ignored",
        file="tools/paired_driver.py",
        find='    suites = suites or SUITES.split(",")',
        replace='    suites = SUITES.split(",")',
        test=(
            "bench/tests/test_regression_review.py::TestPairedDriverSmoke"
            "::test_scoped_suites_and_frameworks_respected"
        ),
        description="--suites parameter clobbered by the module "
        "constant — the flag did nothing and the scoped mini A/A "
        "silently ran all four suites (~45 min instead of ~20); the "
        "reviewer reproduced it with a stubbed side runner",
    ),
    Mutation(
        name="paired-driver-suite-completeness-removed",
        file="tools/paired_driver.py",
        find="    missing = expected - seen\n    if missing:",
        replace="    missing = expected - seen\n    if False:",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage"
            "::test_completeness_backstop_direct"
        ),
        description="per-suite completeness check disabled — a suite "
        "that returns rc=0 but produces no cases passes unnoticed "
        "(the round check counted rounds, not suites; reviewer "
        "defect 4)",
    ),
    Mutation(
        name="paired-driver-post-check-ignored",
        file="tools/paired_driver.py",
        find="    if not (mut_ok and xcheck_ok):",
        replace="    if False:",
        test=(
            "bench/tests/test_regression_review.py"
            "::TestPairedDriverReviewCoverage"
            "::test_post_run_check_failure_invalidates"
        ),
        description="post-run mutation/crosscheck failures no longer "
        "invalidate the verdict — a failed post-run check left "
        "silently-untrustworthy results looking healthy (reviewer "
        "defect 4)",
    ),
    Mutation(
        name="compare-outcome-sums-not-means",
        file="bench/compare.py",
        find=(
            "key: sum(r.get(key, 0) for r in per_round) / len(per_round)"
            " for key in keys"
        ),
        replace="key: sum(r.get(key, 0) for r in per_round) for key in keys",
        test=(
            "bench/tests/test_bench.py::TestCompare"
            "::test_round_count_imbalance_with_equal_rates_not_regression"
        ),
        description="outcomes summed across rounds — 3-vs-5 rounds with "
        "identical per-round rates flag as a regression",
    ),
]


def _tail(result: subprocess.CompletedProcess) -> str:
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


def _run_pytest(test: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            test,
            "-x",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd or REPO,
        timeout=120,
    )


def _prune_stale_worktrees(max_age_s: float = 3600.0) -> None:
    """Remove .mutation-worktree-* orphans left by hard-killed runs
    (SIGKILL cannot reach the finally block). Harmless but untidy.

    AGE CHECK: only remove worktrees older than max_age_s (1h). A
    concurrent mutation check's ACTIVE worktree must never be pruned
    — the original deleted every match unconditionally (a real flaw
    even though it did not cause the pre-flight failure)."""
    import shutil
    import time as _time

    _git("worktree", "prune")
    cutoff = _time.time() - max_age_s
    for stale in REPO.parent.glob(".mutation-worktree-*"):
        try:
            if stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            continue


def _assert_clean_tree() -> None:
    """The worktree is checked out from HEAD — uncommitted changes in
    the real tree would not be what gets checked. Refuse to run."""
    status = _git("status", "--porcelain", "--", "bench", "tools")
    if status.stdout.strip():
        raise SystemExit(
            "refusing to run: bench/ or tools/ has uncommitted changes.\n"
            "Mutations run against HEAD in a disposable worktree; commit "
            "or stash first so the checked code is the code you have.\n"
            f"git status:\n{status.stdout}"
        )


def _make_worktree() -> Path:
    """Disposable worktree in the repo's PARENT directory (so
    sibling-relative paths like ../MLServer resolve for tests that
    need them). The real tree is never mutated."""
    worktree = REPO.parent / f".mutation-worktree-{os.getpid()}"
    result = _git("worktree", "add", "--detach", str(worktree), "HEAD")
    if result.returncode != 0:
        raise SystemExit(f"git worktree add failed: {result.stderr.strip()[:300]}")
    return worktree


def _remove_worktree(worktree: Path) -> None:
    _git("worktree", "remove", "--force", str(worktree))


def run_mutation(mutation: Mutation, worktree: Path) -> tuple[bool, str]:
    """Apply mutation to the WORKTREE, run the test. Returns (caught, detail).

    STRICT catch rule (per review): the UNMUTATED test must pass
    first (a broken baseline proves nothing), and the mutated run
    must be a REAL test failure — pytest exit code 1, with no import,
    syntax or name errors in the output. Exit codes 2-5 (interrupted,
    internal, usage, no-tests) and mutated files that fail to import
    would otherwise masquerade as catches — the same class of mistake
    this exercise exists to remove. (The marker list is string-based
    and errs toward a visible NOT-CAUGHT.)
    """
    # 1. Baseline: the unmutated test must pass
    baseline = _run_pytest(mutation.test, worktree)
    if baseline.returncode != 0:
        return False, (
            f"BASELINE FAILED (exit {baseline.returncode}): " f"{_tail(baseline)}"
        )

    # 2. Mutated run (worktree only; restored immediately after)
    path = worktree / mutation.file
    original = path.read_text()
    if mutation.find not in original:
        return False, "MUTATION DID NOT APPLY (find-text not in file)"
    for extra_find, _ in mutation.extra:
        if extra_find not in original:
            return False, "MUTATION DID NOT APPLY (extra find-text missing)"
    try:
        mutated = original.replace(mutation.find, mutation.replace)
        for extra_find, extra_replace in mutation.extra:
            mutated = mutated.replace(extra_find, extra_replace)
        path.write_text(mutated)
        result = _run_pytest(mutation.test, worktree)
    finally:
        path.write_text(original)
    detail = _tail(result)
    if result.returncode != 1:
        return False, f"exit {result.returncode} is not a test failure: {detail}"
    combined = result.stdout + result.stderr
    for marker in (
        "ImportError",
        "ModuleNotFoundError",
        "SyntaxError",
        "NameError",
        "collection error",
    ):
        if marker in combined:
            return False, f"mutation broke the file ({marker}): {detail}"
    return True, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="")
    args = parser.parse_args()

    _prune_stale_worktrees()
    _assert_clean_tree()
    selected = [m for m in MUTATIONS if args.filter in m.name]
    if not args.filter and len(selected) < MIN_DECLARED_MUTATIONS:
        print(
            f"FAIL: only {len(selected)} mutations declared — the "
            f"pinned minimum is {MIN_DECLARED_MUTATIONS}. A refactor "
            "probably dropped one; shrink the pin DELIBERATELY, not "
            "by accident."
        )
        return 1
    worktree = _make_worktree()
    print(f"mutation checks: {len(selected)} selected")
    print(f"disposable worktree: {worktree} (real tree untouched)\n")
    uncaught = []
    try:
        for mutation in selected:
            caught, detail = run_mutation(mutation, worktree)
            status = "CAUGHT" if caught else "NOT CAUGHT"
            print(f"[{status:10s}] {mutation.name}")
            print(f"             {mutation.description}")
            print(f"             test: {mutation.test.split('::')[-1]}")
            if caught:
                # The detail line is the pytest tail of the MUTATED
                # run — "1 failed in 0.69s" here is the EVIDENCE of
                # the catch (the guard test failed against the broken
                # code, as intended), not a test-suite failure. This
                # output is kept verbatim in the paired driver's
                # gates/*.log, so it must be self-describing.
                print(
                    f"             evidence (mutated run fails the "
                    f"guard test): {detail}\n"
                )
            else:
                print(f"             {detail}\n")
            if not caught:
                uncaught.append(mutation.name)
    finally:
        _remove_worktree(worktree)

    if uncaught:
        print(f"RESULT: {len(uncaught)}/{len(selected)} mutations NOT caught:")
        for name in uncaught:
            print(f"  - {name}")
        return 1
    print(
        f"RESULT: all {len(selected)} mutations caught — the regression "
        "tests have teeth"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
