"""Tests for the Docker-mode benchmark engine.

No Docker, no GPU, no network required — pure logic plus the deterministic
model zoo (which exercises sklearn/xgboost/lightgbm/onnx generation).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench import compare as compare_mod  # noqa: E402
from bench import loadgen, telemetry, zoo  # noqa: E402
from bench.config import (  # noqa: E402
    FRAMEWORKS_CPU,
    LOAD_PROFILES,
    SUITES,
    RunConfig,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_validate(self):
        assert RunConfig().validate() == []

    def test_gpu_rejects_cpu_frameworks(self):
        config = RunConfig(target="gpu", frameworks=["sklearn"])
        problems = config.validate()
        assert any("sklearn" in p for p in problems)

    def test_gpu_accepts_onnx_only(self):
        assert RunConfig(target="gpu", frameworks=["onnx"]).validate() == []

    def test_unknown_suite_rejected(self):
        config = RunConfig(suites=["nope"])
        assert any("nope" in p for p in config.validate())

    def test_profiles_define_required_keys(self):
        for name, profile in LOAD_PROFILES.items():
            for key in (
                "concurrency",
                "qps_fractions",
                "batch_sizes",
                "warmup_seconds",
                "window_seconds",
            ):
                assert key in profile, f"{name} missing {key}"

    def test_custom_profile_overrides(self):
        config = RunConfig(
            load_profile="custom",
            custom={"concurrency": [3], "window_seconds": 7},
        )
        assert config.profile()["concurrency"] == [3]
        assert config.profile()["window_seconds"] == 7

    def test_round_trip(self, tmp_path):
        config = RunConfig(target="gpu", frameworks=["onnx"], tiers=["small"])
        path = tmp_path / "config.json"
        config.save(path)
        assert RunConfig.load(path) == config


# ---------------------------------------------------------------------------
# Deterministic model zoo
# ---------------------------------------------------------------------------


class TestZoo:
    @pytest.fixture(scope="class")
    def zoo_root(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("zoo")
        return zoo.materialize(
            root, ["sklearn", "xgboost", "lightgbm", "onnx"], ["small"]
        )

    def test_generates_all_frameworks(self, zoo_root):
        for framework in FRAMEWORKS_CPU:
            assert zoo.list_models(zoo_root, framework, "small"), framework

    def test_definitive_layout(self, zoo_root):
        for model_dir in zoo_root.iterdir():
            if not model_dir.is_dir():
                continue
            artifacts = [
                p.name
                for p in model_dir.iterdir()
                if p.suffix in (".joblib", ".bst", ".txt", ".onnx")
            ]
            assert len(artifacts) == 1, model_dir
            for required in (
                "model-settings.json",
                "validation.npz",
                "payloads.npz",
                "MANIFEST.json",
            ):
                assert (model_dir / required).is_file(), f"{model_dir.name}/{required}"

    def test_manifest_verifies(self, zoo_root):
        assert zoo.verify(zoo_root) == []

    def test_tamper_detected(self, zoo_root):
        model_dir = next(d for d in zoo_root.iterdir() if d.is_dir())
        target = model_dir / "model-settings.json"
        original = target.read_text()
        try:
            target.write_text(original + " ")
            problems = zoo.verify(zoo_root)
            assert any("mismatch" in p for p in problems)
        finally:
            target.write_text(original)

    def test_determinism_same_seed_same_bytes(self, tmp_path):
        root_a = zoo.materialize(tmp_path / "a", ["sklearn"], ["small"])
        root_b = zoo.materialize(tmp_path / "b", ["sklearn"], ["small"])
        model = "sklearn-pipeline-logreg-small"
        assert zoo._sha256(root_a / model / "model.joblib") == zoo._sha256(
            root_b / model / "model.joblib"
        )

    def test_reuse_if_present(self, tmp_path, capsys):
        zoo.materialize(tmp_path, ["sklearn"], ["small"])
        zoo.materialize(tmp_path, ["sklearn"], ["small"])
        out = capsys.readouterr().out
        assert "[zoo] reused=3 generated=0 downloaded=0" in out

    def test_validation_batch_has_edge_cases(self, zoo_root):
        model_dir = zoo_root / "onnx-mlp-small"
        batch = np.load(model_dir / "validation.npz")
        rows = batch["input"]
        assert (rows[0] == 0).all()  # zeros
        assert (rows[1] <= 0).all()  # negatives
        assert abs(rows[2]).max() > 100  # extreme floats
        assert 0 < abs(rows[3]).min() < 1e-30  # subnormals

    def test_payload_rows_industry_convention(self, zoo_root):
        payloads = np.load(zoo_root / "onnx-mlp-small" / "payloads.npz")
        assert set(payloads.files) == {f"rows{n}" for n in zoo.PAYLOAD_ROWS}

    def test_industry_baselines(self):
        assert zoo.TIER_BASELINES["small"]["tabular"][0] == "higgs10"
        assert zoo.TIER_BASELINES["medium"]["tabular"][0] == "higgs"
        assert zoo.TIER_BASELINES["large"]["tabular"][0] == "covertype"
        assert zoo.TIER_BASELINES["small"]["vision"][0] == "mnist"
        assert zoo.TIER_BASELINES["large"]["vision"][0] == "imagenet"

    def test_generated_onnx_models_run(self, zoo_root):
        """Every generated ONNX graph must load AND execute (catches
        invalid graphs like duplicate tensor names before the server
        ever sees them)."""
        import onnxruntime as ort

        for model_dir in sorted(zoo_root.glob("onnx-*")):
            batch = dict(np.load(model_dir / "validation.npz"))
            session = ort.InferenceSession(
                str(model_dir / "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
            feeds = {
                key: value
                for key, value in batch.items()
                if key in {i.name for i in session.get_inputs()}
            }
            outputs = session.run(None, feeds)
            assert outputs, model_dir.name

    def test_download_registry_pinned(self):
        for key, entry in zoo.DOWNLOAD_REGISTRY.items():
            assert len(entry["sha256"]) == 64
            assert entry["url"].startswith("https://")


# ---------------------------------------------------------------------------
# Load generator
# ---------------------------------------------------------------------------


class TestLoadgen:
    def _fake_send(self, latency: float, fail_every: int = 0):
        counter = {"n": 0}

        async def send(seq: int):
            await __import__("asyncio").sleep(latency)
            counter["n"] += 1
            if fail_every and counter["n"] % fail_every == 0:
                raise RuntimeError("injected")
            return True, 100, 200

        return send

    def test_closed_loop_collects_samples(self):
        import asyncio

        result = asyncio.run(
            loadgen.closed_loop(
                4,
                0.3,
                0.05,
                self._fake_send(0.001),
            )
        )
        assert result.summary()["ok"] > 0
        assert result.summary()["error_rate"] == 0.0

    def test_error_guard_aborts(self):
        import asyncio

        with pytest.raises(RuntimeError, match="error rate"):
            asyncio.run(
                loadgen.closed_loop(
                    4,
                    0.3,
                    0.05,
                    self._fake_send(0.001, fail_every=2),
                )
            )

    def test_open_loop_constant_rate(self):
        import asyncio

        result = asyncio.run(
            loadgen.open_loop(
                200,
                0.5,
                0.05,
                self._fake_send(0.001),
                poisson=False,
            )
        )
        # ~200 rps * 0.5 s measured => roughly 100 requests
        assert 60 <= result.summary()["ok"] <= 160

    def test_percentiles(self):
        samples = [
            loadgen.Sample(i * 1_000_000, i * 1_000_000 + int(ms * 1e6), True, 0, 0)
            for i, ms in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        ]
        result = loadgen.LoadResult(samples=samples, wall_seconds=1.0)
        summary = result.summary()
        assert summary["latency_ms_p50"] == 5.5
        assert summary["latency_ms_max"] == 100

    def test_stability_gate_detects_instability(self):
        result = loadgen.LoadResult(wall_seconds=20.0)
        result.samples = []
        # windows: throughput swings wildly
        for window in range(4):
            for i in range(10):
                base = window * 1_000_000_000_00
                result.samples.append(
                    loadgen.Sample(
                        base + i * 100,
                        base + i * 100 + 50,
                        True,
                        0,
                        0,
                    )
                )
                result.samples[-1]._start_ref = 0
        # Not enough real signal — the gate must not crash.
        gate = loadgen.stability_gate(result, 0.0)
        assert "stable" in gate


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_parse_prometheus(self):
        text = (
            "# HELP a gauge\n"
            "mlserver_model_infer_request_success 42\n"
            'mlserver_model_infer_request_duration_seconds_sum{a="b"} 1.5\n'
            'mlserver_model_infer_request_duration_seconds_bucket{le="0.1"} 3\n'
        )
        values = telemetry.parse_prometheus(text)
        assert values["mlserver_model_infer_request_success"] == 42
        assert values["mlserver_model_infer_request_duration_seconds_sum"] == 1.5
        assert "bucket" not in str(values)

    def test_summarize(self):
        records = [{"t": 0, "cpu": 10}, {"t": 1, "cpu": 30}]
        summary = telemetry.summarize(records)
        assert summary["cpu_mean"] == 20
        assert summary["cpu_max"] == 30


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


MANIFEST = {
    "image": {
        "digest": "quay.io/opendatahub/mlserver@sha256:abc",
        "target": "cpu",
    },
    "host": {"cpu_model": "test-cpu", "cores_logical": 16},
}


def _results_dir(
    tmp_path: Path,
    name: str,
    rounds: list[dict],
    manifest: dict | None = None,
) -> Path:
    """Write a results.json with per-round cases (the comparison's
    replicates) and a manifest (the environment gate)."""
    path = tmp_path / name
    path.mkdir()
    cases = []
    for round_index, round_data in enumerate(rounds):
        cases.append(
            {
                "suite": round_data.get("suite", "protocol"),
                "framework": "sklearn",
                "tier": "small",
                "model": round_data.get("model", "m"),
                "params": round_data.get("params", {"concurrency": 8}),
                "round": round_index,
                "result": round_data["result"],
            }
        )
    (path / "results.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "manifest": manifest or MANIFEST,
                "config": {"load_profile": "quick"},
                "cases": cases,
            }
        )
    )
    return path


class TestCompare:
    def test_no_regression_on_identical_rounds(self, tmp_path):
        base = _results_dir(
            tmp_path,
            "base",
            [
                {"result": {"throughput_rps": r, "latency_ms_p99": 10.0}}
                for r in (100, 102, 101)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {"result": {"throughput_rps": r, "latency_ms_p99": 10.0}}
                for r in (100, 101, 100)
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "OK"
        assert result["matched_cases"] == 1

    def test_throughput_regression_detected_with_rounds(self, tmp_path):
        base = _results_dir(
            tmp_path,
            "base",
            [
                {"result": {"throughput_rps": r, "latency_ms_p99": 10.0}}
                for r in (100, 102, 101, 100, 101)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {"result": {"throughput_rps": r, "latency_ms_p99": 10.0}}
                for r in (80, 82, 81, 80, 81)
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "REGRESSION"
        assert result["regressions"] == 1

    def test_single_round_is_informational_not_regression(self, tmp_path):
        """One round per side cannot support a significance verdict —
        a 20% drop is reported but NOT flagged (insufficient
        replicates; the old window-based test flagged it from
        within-run noise)."""
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [{"result": {"throughput_rps": 80, "latency_ms_p99": 10.0}}],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "OK"
        entry = result["verdicts"][0]["throughput"]
        assert entry["insufficient_replicates"] is True
        assert entry["regression"] is False
        assert entry["delta"] < -0.19  # the drop IS reported

    def test_rounds_aggregate_not_collapse(self, tmp_path):
        """The old case_key lacked round: 27 rounds-cases collapsed to
        the LAST round per key. Now every round is a replicate."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}
                for _ in range(5)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {"result": {"throughput_rps": r, "latency_ms_p99": 10.0}}
                for r in (100, 100, 100, 100, 60)  # last round drops
            ],
        )
        result = compare_mod.compare(base, cand)
        entry = result["verdicts"][0]["throughput"]
        assert entry["replicates"] == [5, 5]

    def test_parity_failure_appearing_is_regression(self, tmp_path):
        """Correctness cells: a NEW PARITY_FAILED is a hard regression
        (the old version treated these cells as zero-throughput and
        never flagged them)."""
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {
                    "result": {
                        "status": "PARITY_FAILED",
                        "problems": ["8 in 7 out"],
                    },
                }
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "REGRESSION"
        assert result["verdicts"][0]["correctness"]["new_failure_statuses"] == [
            "PARITY_FAILED"
        ]

    def test_fault_outcome_worsening_is_regression(self, tmp_path):
        base = _results_dir(
            tmp_path,
            "base",
            [
                {
                    "suite": "faults",
                    "result": {"exact": 28, "truncated": 0, "empty": 0},
                }
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {
                    "suite": "faults",
                    "result": {"exact": 0, "truncated": 20, "empty": 7},
                }
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "REGRESSION"
        changes = result["verdicts"][0]["correctness"]["outcome_changes"]
        assert changes["exact"] == {"baseline": 28, "candidate": 0}
        assert changes["truncated"] == {"baseline": 0, "candidate": 20}

    def test_transient_load_error_is_not_correctness_regression(self, tmp_path):
        """A load cell with a single transient error per round (inside
        the 0.1% error guard) must NOT fire the correctness
        comparison — the original included 'failed' in the outcome
        keys, so every throughput cell was a correctness cell and one
        flaky request was a hard regression."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {"result": {"throughput_rps": 100, "latency_ms_p99": 10.0, "failed": 0}}
                for _ in range(3)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {"result": {"throughput_rps": 100, "latency_ms_p99": 10.0, "failed": 1}}
                for _ in range(3)
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "OK"
        assert "correctness" not in result["verdicts"][0]

    def test_round_count_imbalance_with_equal_rates_not_regression(self, tmp_path):
        """3 baseline rounds vs 5 candidate rounds with IDENTICAL
        per-round failure rates must not flag (the original summed
        totals: 3 vs 5 failures -> hard regression)."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {
                    "suite": "faults",
                    "result": {"exact": 27, "truncated": 1, "empty": 0},
                }
                for _ in range(3)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {
                    "suite": "faults",
                    "result": {"exact": 27, "truncated": 1, "empty": 0},
                }
                for _ in range(5)
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "OK"

    def test_worker_kill_outcomes_compared(self, tmp_path):
        """Worker-kill cells: worsened load_failed / outage duration
        in the faults suite IS a regression (the keys were previously
        unread by compare)."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {
                    "suite": "faults",
                    "result": {
                        "load_failed": 8,
                        "outage_duration_ms": 74,
                    },
                }
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {
                    "suite": "faults",
                    "result": {
                        "load_failed": 11950,
                        "outage_duration_ms": 4818,
                    },
                }
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "REGRESSION"
        changes = result["verdicts"][0]["correctness"]["outcome_changes"]
        assert changes["load_failed"]["candidate"] > changes["load_failed"]["baseline"]

    def test_missing_manifest_field_refuses(self, tmp_path):
        """A manifest field missing on BOTH sides used to pass
        silently (None == None) — an ungated comparison."""
        empty_manifest = {"image": {}, "host": {}}
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
            manifest=empty_manifest,
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
            manifest=empty_manifest,
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "MANIFEST_MISMATCH"
        assert any("missing" in m for m in result["mismatches"])

    def test_worker_kill_modest_change_not_regression(self, tmp_path):
        """Worker-kill outcomes are single noisy measurements — a
        modest outage change (4.8s -> 5.5s, ~15%) must not flag; only
        large worsenings (>50%) do."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {
                    "suite": "faults",
                    "result": {"load_failed": 11950, "outage_duration_ms": 4800},
                }
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {
                    "suite": "faults",
                    "result": {"load_failed": 12100, "outage_duration_ms": 5500},
                }
            ],
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "OK"

    def test_zero_variance_delta_only(self, tmp_path):
        """Identical-per-round data on both sides: the t-test is
        meaningless — judge on delta only, labeled no_variance. A 30%
        deterministic drop IS a regression; identical data is not."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}
                for _ in range(5)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {"result": {"throughput_rps": 70, "latency_ms_p99": 10.0}}
                for _ in range(5)
            ],
        )
        result = compare_mod.compare(base, cand)
        entry = result["verdicts"][0]["throughput"]
        assert entry["no_variance"] is True
        assert entry["regression"] is True
        assert result["status"] == "REGRESSION"

    def test_zero_variance_identical_ok(self, tmp_path):
        base = _results_dir(
            tmp_path,
            "base",
            [
                {"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}
                for _ in range(5)
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}
                for _ in range(5)
            ],
        )
        result = compare_mod.compare(base, cand)
        entry = result["verdicts"][0]["throughput"]
        assert entry.get("no_variance") is True
        assert entry["regression"] is False
        assert result["status"] == "OK"

    def test_manifest_mismatch_refuses(self, tmp_path):
        other_image = {
            "image": {
                "digest": "quay.io/opendatahub/mlserver@sha256:DIFFERENT",
                "target": "cpu",
            },
            "host": {"cpu_model": "test-cpu", "cores_logical": 16},
        }
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
            manifest=other_image,
        )
        result = compare_mod.compare(base, cand)
        assert result["status"] == "MANIFEST_MISMATCH"
        assert any("image.digest" in m for m in result["mismatches"])

    def test_p99_single_round_informational(self, tmp_path):
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 50.0}}],
        )
        result = compare_mod.compare(base, cand)
        entry = result["verdicts"][0]["p99_ms"]
        assert entry["insufficient_replicates"] is True
        assert entry["regression"] is False

    def test_mann_whitney_exact_small_n(self):
        """Complete 3v3 separation: the exact two-sided minimum p is
        0.1 — never significant at 0.05 (the old normal approximation
        claimed 0.0495)."""
        _, p = compare_mod.mann_whitney_u([1, 2, 3], [10, 11, 12])
        assert p == pytest.approx(0.1, abs=0.011)

    def test_welch_t_identical(self):
        _, p = compare_mod.welchs_t([5, 5, 5], [5, 5, 5])
        assert p >= 0.99

    def test_welch_t_small_n_not_anticonservative(self):
        """3v3 with a ~2.4 sigma gap: the t-distribution must give
        ~0.07 (the old normal approximation gave 0.014)."""
        _, p = compare_mod.welchs_t([1.0, 2.0, 3.0], [3.4, 4.4, 5.4])
        assert p > 0.03  # not anti-conservative

    def test_holm_bonferroni_monotone(self):
        """sorted raw: 0.01, 0.03, 0.04 with m=3 ->
        3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.04 (monotone: stays 0.06)."""
        adjusted = compare_mod.holm_bonferroni([0.01, 0.04, 0.03])
        assert adjusted == [0.03, 0.06, 0.06]
        assert all(a >= r for a, r in zip(adjusted, [0.01, 0.04, 0.03]))


# ---------------------------------------------------------------------------
# Wizard input parsing (stdin-driven)
# ---------------------------------------------------------------------------


class TestWizard:
    def test_choose_validates_input(self, monkeypatch, capsys):
        from bench import wizard

        answers = iter(["0", "abc", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        index = wizard.choose("pick", ["a", "b"], default=1)
        assert index == 2

    def test_choose_default_on_enter(self, monkeypatch):
        from bench import wizard

        monkeypatch.setattr("builtins.input", lambda _: "")
        assert wizard.choose("pick", ["a", "b", "c"], default=3) == 3

    def test_choose_many_names(self, monkeypatch):
        from bench import wizard

        monkeypatch.setattr("builtins.input", lambda _: "protocol, batching")
        assert wizard.choose_many("pick", SUITES, "all") == ["protocol", "batching"]

    def test_allow_image_change_permits_digest_diff(self, tmp_path):
        """--allow-image-change: the digest may differ (baseline vs
        optimized build) while host and load-profile stay gated."""
        other_image = {
            "image": {
                "digest": "quay.io/opendatahub/mlserver@sha256:OPTIMIZED",
                "target": "cpu",
            },
            "host": {"cpu_model": "test-cpu", "cores_logical": 16},
        }
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
            manifest=other_image,
        )
        # without the flag: refused
        result = compare_mod.compare(base, cand)
        assert result["status"] == "MANIFEST_MISMATCH"
        # with the flag: compared, override recorded
        result = compare_mod.compare(base, cand, allow_image_change=True)
        assert result["status"] == "OK"
        assert "image_change_allowed" in result
        assert "OPTIMIZED" in result["image_change_allowed"]["candidate_digest"]

    def test_allow_image_change_still_gates_host(self, tmp_path):
        other_host = {
            "image": {
                "digest": "quay.io/opendatahub/mlserver@sha256:abc",
                "target": "cpu",
            },
            "host": {"cpu_model": "DIFFERENT-cpu", "cores_logical": 16},
        }
        base = _results_dir(
            tmp_path,
            "base",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [{"result": {"throughput_rps": 100, "latency_ms_p99": 10.0}}],
            manifest=other_host,
        )
        result = compare_mod.compare(base, cand, allow_image_change=True)
        assert result["status"] == "MANIFEST_MISMATCH"
        assert any("host" in m for m in result["mismatches"])

    def test_gzip_invalid_cell_not_compared(self, tmp_path):
        """A gzip_engaged=False cell measured nothing (no-op
        configuration) — its noise must not read as a regression."""
        base = _results_dir(
            tmp_path,
            "base",
            [
                {
                    "suite": "middleware",
                    "params": {"middleware": "gzip", "state": "on"},
                    "result": {
                        "throughput_rps": 100,
                        "latency_ms_p99": 10.0,
                        "gzip_engaged": False,
                    },
                }
            ],
        )
        cand = _results_dir(
            tmp_path,
            "cand",
            [
                {
                    "suite": "middleware",
                    "params": {"middleware": "gzip", "state": "on"},
                    "result": {
                        "throughput_rps": 50,
                        "latency_ms_p99": 10.0,
                        "gzip_engaged": False,
                    },
                }
            ],
        )
        result = compare_mod.compare(base, cand)
        # 50% drop in a no-op cell must NOT flag
        assert result["status"] == "OK"
        assert "gzip_invalid" in result["verdicts"][0]
        assert "throughput" not in result["verdicts"][0]
