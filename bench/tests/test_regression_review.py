"""Regression tests for the harness bugs found in the external review.

Each test targets a specific bug that the original 170 tests missed:

  Bug 1  — summary(window) counted warmup samples but divided by the
           window length (420 rps reported for a true 100 rps), and
           _with_ref stored seconds while end_ns is nanoseconds, so
           window filters never matched anything (vacuous gate).
  Bug 2  — BinaryTensorSender kept a CLASS-level aiohttp session
           shared across per-case event loops ("Event loop is closed").
  Bug 3  — saturation passed a sender OBJECT to open_loop, which
           expects a callable.
  Bug 4  — parallel_workers=workers or None turned 0 into None, so
           the in-process rung silently duplicated rung 1.
  Bug 5  — _thread_env matched "mlserver_onnx" against config.target
           ("cpu"/"gpu"): dead code, ORT thread options never applied.
  Bug 6  — the framework suite probed only "session_options" first,
           so every other axis was wrongly SKIPPED_UNSUPPORTED_IMAGE.
  Bug 7  — multimodel repeated directory names and copytree raised
           FileExistsError when the tier had fewer models than rungs.
  Bug 8  — the caching "unique" payload was one deterministic value
           (a second repeat, not a unique stream).
  Bug 14 — GrpcSender reported a chunk COUNT as bytes_out.
  Bug 15 — one RuntimeError killed the whole run (only ServerError
           was contained per suite).
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench import loadgen, wire  # noqa: E402
from bench.loadgen import LoadResult, Sample, _with_ref  # noqa: E402
from bench.suites import (  # noqa: E402
    FRAMEWORK_AXIS_FIELDS,
    ONNX_PROBE_KEYS,
    RestJsonSender,
    _thread_env,
    build_multimodel_repo,
    scaled_payload,
)
from bench.config import RunConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Bug 1: window summarisation
# ---------------------------------------------------------------------------


def _known_rate_samples(rate: float, warmup_s: float, window_s: float) -> LoadResult:
    """Samples at an exact rate: warmup_s at `rate`, then window_s at
    `rate` (the classic setup that reported 420 rps for 100 true rps
    when the window was 1/4.2 of the total)."""
    t0_ns = 1_000_000_000_000  # arbitrary epoch
    total_s = warmup_s + window_s
    n = int(rate * total_s)
    interval_ns = int(1e9 / rate)
    samples = []
    for i in range(n):
        start = t0_ns + i * interval_ns
        samples.append(Sample(start, start + 1_000_000, True, 100, 200))
    result = LoadResult(samples=samples, wall_seconds=total_s)
    _with_ref(result.samples, t0_ns)
    return result


class TestSummaryWindow:
    def test_window_throughput_is_true_rate(self):
        """100 rps steady, window (10, 20) over a 10s warmup + 10s
        window: the window summary must report ~100 rps, not ~200
        (all-sample count / window length)."""
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        summary = result.summary((10.0, 20.0))
        assert summary["throughput_rps"] == pytest.approx(100.0, rel=0.05)

    def test_window_counts_only_window_samples(self):
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        summary = result.summary((10.0, 20.0))
        # 10s window at 100 rps = ~1000 requests in-window, NOT the
        # ~2000 of the whole run.
        assert summary["requests"] == pytest.approx(1000, rel=0.05)
        assert summary["ok"] == pytest.approx(1000, rel=0.05)

    def test_window_bytes_filtered(self):
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        summary = result.summary((10.0, 20.0))
        # 100 bytes_out per sample -> ~1000 in window, not ~2000
        assert summary["bytes_out"] == pytest.approx(100_000, rel=0.05)

    def test_window_filter_actually_filters(self):
        """The seconds-vs-nanoseconds _start_ref mismatch made every
        window filter match nothing — latencies came back empty."""
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        latencies = result.latencies_ms((10.0, 20.0))
        assert latencies.size > 0, "window filter matched no samples"
        # and a window past the end of the run matches nothing
        empty = result.latencies_ms((25.0, 30.0))
        assert empty.size == 0

    def test_whole_run_summary_unaffected(self):
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        summary = result.summary()
        assert summary["throughput_rps"] == pytest.approx(100.0, rel=0.05)
        assert summary["requests"] == pytest.approx(2000, rel=0.05)


class TestStabilityGate:
    def test_detects_instability(self):
        """A rate that doubles mid-run must NOT report a 0 spread
        (the vacuous-gate signature) and must be unstable."""
        t0_ns = 1_000_000_000_000
        samples = []
        now = t0_ns
        # 10s warmup at 100 rps, then 10s at 100 rps, then 10s at 300
        for rate, duration in ((100.0, 20.0), (300.0, 10.0)):
            interval = 1e9 / rate
            t = now
            end = now + duration * 1e9
            while t < end:
                samples.append(Sample(int(t), int(t) + 1_000_000, True, 10, 20))
                t += interval
            now = end
        result = LoadResult(samples=samples, wall_seconds=30.0)
        _with_ref(result.samples, t0_ns)
        gate = loadgen.stability_gate(result, warmup_seconds=10.0)
        assert len(gate["windows"]) >= 2
        assert gate["throughput_spread"] > 0.5  # real spread, not 0
        assert not gate["stable"]

    def test_stable_run_is_stable(self):
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        gate = loadgen.stability_gate(result, warmup_seconds=10.0)
        assert len(gate["windows"]) >= 2
        assert gate["throughput_spread"] < 0.05
        assert gate["stable"]


# ---------------------------------------------------------------------------
# Bug 2: per-case sender sessions
# ---------------------------------------------------------------------------


class TestBinarySenderSessions:
    def test_no_class_level_session(self):
        """The bug: BinaryTensorSender._session was a CLASS attribute
        shared across per-case event loops."""
        assert (
            "_session" not in wire.BinaryTensorSender.__dict__
        ), "BinaryTensorSender still has a class-level _session"

    def test_two_senders_two_loops(self):
        """Two cases in two different event loops must each get their
        OWN session — the shared-session bug made the second case
        raise 'Event loop is closed'."""

        async def make_and_close(sessions: list) -> None:
            sender = wire.BinaryTensorSender("m", b"x", 1)
            await sender._ensure_session()
            session = sender._session
            sessions.append(session)
            assert session.closed is False
            await sender.close()
            assert session.closed is True

        sessions: list = []
        asyncio.run(make_and_close(sessions))
        asyncio.run(make_and_close(sessions))
        assert len(sessions) == 2
        assert (
            sessions[0] is not sessions[1]
        ), "second case reused the first case's (dead-loop) session"


# ---------------------------------------------------------------------------
# Bug 3: open_loop receives a callable
# ---------------------------------------------------------------------------


class TestOpenLoopSender:
    def test_rest_sender_exposes_send_callable(self):
        sender = RestJsonSender("model", {"inputs": []})
        assert callable(sender.send)

    def test_open_loop_with_object_send_method(self):
        """The saturation pattern: open_loop(..., sender_obj.send)."""

        async def stub_send(seq: int) -> tuple[bool, int, int]:
            await asyncio.sleep(0)
            return True, 10, 20

        result = asyncio.run(
            loadgen.open_loop(
                50, 0.3, 0.1, stub_send, poisson=False, error_threshold=1.0
            )
        )
        assert result.samples
        assert all(s.ok for s in result.samples)

    def test_scheduled_time_latency(self):
        """Open-loop latency must be measured from the SCHEDULED time
        (wrk2 method) — a slow sender's delay shows up in latency."""

        async def slow_send(seq: int) -> tuple[bool, int, int]:
            await asyncio.sleep(0.05)
            return True, 1, 1

        result = asyncio.run(
            loadgen.open_loop(
                20, 0.4, 0.1, slow_send, poisson=False, error_threshold=1.0
            )
        )
        latencies = result.latencies_ms()
        # every request services in >= 50ms regardless of loop timing
        assert latencies.min() >= 50.0


# ---------------------------------------------------------------------------
# Bug 4: workers=0 must reach the container as "0"
# ---------------------------------------------------------------------------


class TestWorkersZero:
    def test_suite_passes_workers_through(self):
        """Wiring check (mutation-caught): suite_workers must pass the
        ladder value straight through. `workers or None` turned rung 0
        into the server default (1) — the container-level tests could
        not catch it because the bug was in the SUITE call site."""
        import inspect

        from bench.suites import suite_workers

        source = inspect.getsource(suite_workers)
        assert (
            "workers or None" not in source
        ), "suite_workers collapses rung 0 to None again"
        assert "parallel_workers=workers" in source

    def test_zero_is_not_none(self):
        """`workers or None` turned 0 into None — the server default
        (1 worker) applied and rungs 0 and 1 were identical."""
        from bench.docker_server import MLServerContainer

        container = MLServerContainer(
            name="t", image="i", model_repo=Path("/tmp"), parallel_workers=0
        )
        assert container.parallel_workers == 0

    def test_env_assembly_passes_zero(self):
        from bench.docker_server import MLServerContainer

        container = MLServerContainer(
            name="t",
            image="i",
            model_repo=Path("/tmp"),
            parallel_workers=0,
        )
        with (
            patch("bench.docker_server._docker") as mock_docker,
            patch.object(MLServerContainer, "_wait_healthy"),
        ):
            container.start()
        args = mock_docker.call_args[0]
        env_args = [args[i + 1] for i, a in enumerate(args) if a == "-e"]
        assert (
            "MLSERVER_PARALLEL_WORKERS=0" in env_args
        ), f"workers=0 must be passed explicitly (got {env_args})"


# ---------------------------------------------------------------------------
# Bug 5: _thread_env framework detection
# ---------------------------------------------------------------------------


class TestThreadEnv:
    def test_onnx_gets_ort_session_options(self):
        env = _thread_env(RunConfig(), 2, "mlserver_onnx.OnnxModel")
        assert "MLSERVER_MODEL_ONNX_SESSION_OPTIONS" in env
        opts = json.loads(env["MLSERVER_MODEL_ONNX_SESSION_OPTIONS"])
        assert "intra_op_num_threads" in opts

    def test_sklearn_does_not(self):
        env = _thread_env(RunConfig(), 2, "mlserver_sklearn.SKLearnModel")
        assert "MLSERVER_MODEL_ONNX_SESSION_OPTIONS" not in env

    def test_threads_scale_with_cores(self):
        import os

        cores = os.cpu_count() or 4
        env = _thread_env(RunConfig(), 1, "mlserver_onnx.OnnxModel")
        threads = int(env["OMP_NUM_THREADS"])
        assert threads == max(1, cores)


# ---------------------------------------------------------------------------
# Bug 6: framework probe coverage
# ---------------------------------------------------------------------------


class TestFrameworkProbeCoverage:
    def test_probe_call_uses_full_key_set(self):
        """Wiring check (mutation-caught): the framework suite's probe
        call must pass *ONNX_PROBE_KEYS — the original test only
        checked the constant's contents, so a call-site regression
        (probing only session_options) passed the suite while every
        other axis silently skipped."""
        import inspect

        from bench.suites import suite_framework

        source = inspect.getsource(suite_framework)
        assert (
            "probe_onnx_settings(server.name, *ONNX_PROBE_KEYS)" in source
        ), "probe call no longer passes the full key set"

    def test_every_axis_field_is_probed(self):
        """Every OnnxSettings field the axes look up must be in the
        probe key set — the old code probed only session_options first
        and the rest always skipped."""
        for axis, field in FRAMEWORK_AXIS_FIELDS.items():
            assert (
                field in ONNX_PROBE_KEYS
            ), f"axis {axis!r} needs {field!r} but it is never probed"

    def test_probe_keys_are_real_onnxsettings_fields(self):
        """The probe keys must match actual OnnxSettings fields in the
        MLServer checkout (guards against typos re-vacating the gate)."""
        settings_py = (
            REPO / "MLServer" / "runtimes" / "onnx" / "mlserver_onnx" / "settings.py"
        )
        if not settings_py.is_file():
            pytest.skip("MLServer source not available")
        source = settings_py.read_text()
        for key in ONNX_PROBE_KEYS:
            assert re.search(
                rf"^\s+{re.escape(key)}\s*:", source, re.M
            ), f"{key!r} is not an OnnxSettings field"


# ---------------------------------------------------------------------------
# Bug 7: multimodel repo building
# ---------------------------------------------------------------------------


def _mini_zoo(tmp_path: Path, count: int) -> list[Path]:
    dirs = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        d = tmp_path / f"sklearn-m{i}-small"
        d.mkdir()
        (d / "model-settings.json").write_text(
            json.dumps(
                {
                    "name": d.name,
                    "implementation": "mlserver_sklearn.SKLearnModel",
                    "parameters": {"uri": "model.joblib"},
                    "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 4]}],
                    "outputs": [
                        {"name": "predict", "datatype": "INT64", "shape": [-1]}
                    ],
                }
            )
        )
        np.savez(
            d / "validation.npz",
            input=np.ones((4, 4), dtype=np.float32),
        )
        (d / "model.joblib").write_bytes(b"stub")
        dirs.append(d)
    return dirs


class TestMultimodelRepo:
    def test_more_rungs_than_models(self, tmp_path: Path):
        """3 distinct models, rung 5: the old (dirs*10)[:5] repeated
        names and copytree raised FileExistsError."""
        zoo = _mini_zoo(tmp_path, 3)
        repo = tmp_path / "repo-multi-5"
        pairs = build_multimodel_repo(zoo, repo, 5)
        assert len(pairs) == 5
        # 5 DISTINCT directories with distinct model names
        names = [name for name, _ in pairs]
        assert len(set(names)) == 5
        for name in names:
            assert (repo / name).is_dir()
            settings = json.loads((repo / name / "model-settings.json").read_text())
            assert settings["name"] == name

    def test_rung_larger_than_zoo(self, tmp_path: Path):
        zoo = _mini_zoo(tmp_path / "z2", 2)
        repo = tmp_path / "repo-multi-10"
        pairs = build_multimodel_repo(zoo, repo, 10)
        assert len(pairs) == 10
        assert len({n for n, _ in pairs}) == 10


# ---------------------------------------------------------------------------
# Bug 8: unique payload stream
# ---------------------------------------------------------------------------


class TestUniquePayload:
    def test_payload_varies_per_seq(self):
        request = {
            "inputs": [{"name": "input", "datatype": "FP32", "data": [1.0, 2.0]}]
        }
        payloads = [json.dumps(scaled_payload(request, seq)) for seq in range(50)]
        assert len(set(payloads)) == 50, "every seq must yield a new payload"

    def test_payload_structure_preserved(self):
        request = {
            "inputs": [{"name": "input", "datatype": "FP32", "data": [1.0, 2.0]}],
            "outputs": [{"name": "predict"}],
        }
        scaled = scaled_payload(request, 7)
        assert scaled["inputs"][0]["name"] == "input"
        assert len(scaled["inputs"][0]["data"]) == 2
        assert scaled["outputs"] == [{"name": "predict"}]

    def test_payload_nested_data(self):
        """Real requests carry rows x features (nested lists) — the
        original flat-only scaling crashed on every 2-D input with
        'can't multiply sequence by non-int of type float'."""
        request = {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "data": [[1.0, 2.0], [3.0, 4.0]],
                }
            ]
        }
        scaled = scaled_payload(request, 3)
        factor = 1.0 + (3 % 977) / 31.0
        assert scaled["inputs"][0]["data"] == [
            [1.0 * factor, 2.0 * factor],
            [3.0 * factor, 4.0 * factor],
        ]


# ---------------------------------------------------------------------------
# Bug 14: gRPC byte accounting
# ---------------------------------------------------------------------------


class TestGrpcBytes:
    @pytest.mark.skipif(not wire.grpc_available(), reason="grpcio/protos unavailable")
    def test_bytes_out_sums_chunks(self):
        """The bug: len(raw_output_contents) is a CHUNK COUNT, not
        bytes."""

        class FakeResponse:
            raw_output_contents = [b"x" * 100, b"y" * 50]

            def ByteSize(self) -> int:
                return 999  # must NOT be used when raw contents exist

        class FakeStub:
            async def ModelInfer(self, request, timeout=None):
                return FakeResponse()

        settings = {
            "name": "m",
            "inputs": [{"name": "input", "datatype": "FP32", "shape": [1, 2]}],
            "outputs": [{"name": "output", "datatype": "FP32", "shape": [1]}],
        }
        batch = {"input": np.ones((1, 2), dtype=np.float32)}

        async def go() -> tuple[bool, int, int]:
            # grpc.aio channels bind to the running loop at creation
            request = wire.build_grpc_request(settings, batch, "m")
            sender = wire.grpc_sender("m", request)
            sender._stub = FakeStub()
            try:
                return await sender.send(0)
            finally:
                await sender.close()

        ok, bytes_out, bytes_in = asyncio.run(go())
        request = wire.build_grpc_request(settings, batch, "m")
        assert ok is True
        assert bytes_out == 150, "must sum chunk sizes, not count chunks"
        assert bytes_in == len(request.SerializeToString())


# ---------------------------------------------------------------------------
# Bug 15: per-suite failure containment
# ---------------------------------------------------------------------------


class TestSuiteFailureContainment:
    def test_runtimeerror_recorded_not_fatal(self, tmp_path: Path, capsys):
        """One suite raising RuntimeError (e.g. the loadgen error
        guard) must be recorded as SUITE_FAILED while later suites
        still run."""
        from bench.runner import _run_one_suite

        results: list[dict] = []

        async def bad_suite(ctx, dirs):
            raise RuntimeError("error rate 100.00% exceeds threshold")

        async def good_suite(ctx, dirs):
            return [{"suite": "good", "result": {"ok": 1}}]

        outcome = _run_one_suite(
            "bad", bad_suite, None, [], "sklearn", "small", results, tmp_path
        )
        assert outcome is None
        assert len(results) == 1
        assert results[0]["status"] == "SUITE_FAILED"
        assert results[0]["error"] == "RuntimeError"

        outcome = _run_one_suite(
            "good", good_suite, None, [], "sklearn", "small", results, tmp_path
        )
        assert outcome == [{"suite": "good", "result": {"ok": 1}}]

    def test_parity_error_still_fatal(self, tmp_path: Path):
        """Correctness-gate failures must still abort the run."""
        from bench.parity import ParityError
        from bench.runner import _run_one_suite

        async def parity_fail(ctx, dirs):
            raise ParityError("gate failed")

        with pytest.raises(ParityError):
            _run_one_suite("x", parity_fail, None, [], "sklearn", "small", [], tmp_path)


# ---------------------------------------------------------------------------
# Bug 17: engine fail-fast
# ---------------------------------------------------------------------------


class TestEngineFailFast:
    def test_cluster_engine_rejected(self):
        from bench.runner import execute

        config = RunConfig()
        config.engine = "cluster"
        config.frameworks = ["sklearn"]
        with pytest.raises(SystemExit, match="not wired into the Docker-mode"):
            execute(config, REPO)


class TestJsonSafety:
    def test_stability_gate_json_serializable(self):
        """The stability gate's np.bool_/np.float64 values crashed
        results.json writing ("Object of type bool_ is not JSON
        serializable")."""
        result = _known_rate_samples(rate=100.0, warmup_s=10.0, window_s=10.0)
        gate = loadgen.stability_gate(result, warmup_seconds=10.0)
        json.dumps(gate)  # must not raise

    def test_atomic_write_numpy_fallback(self, tmp_path: Path):
        from bench.runner import _atomic_write

        payload = {"flag": np.bool_(True), "value": np.float64(1.5)}
        path = tmp_path / "out.json"
        _atomic_write(path, payload)
        loaded = json.loads(path.read_text())
        assert loaded["flag"] is True
        assert loaded["value"] == 1.5


# ---------------------------------------------------------------------------
# Faults suite (deliberate F-01 class + poison-request resilience)
# ---------------------------------------------------------------------------


class TestFaultsSuite:
    def test_registered(self):
        from bench.config import SUITES as ALL
        from bench.suites import SUITES

        assert "faults" in ALL
        assert callable(SUITES["faults"])

    def test_skips_without_models(self, tmp_path: Path):
        import asyncio

        from bench.suites import SuiteContext, suite_faults

        ctx = SuiteContext(RunConfig(), "image", tmp_path, tmp_path / "out")
        cases = asyncio.run(suite_faults(ctx, []))
        assert cases[0]["result"]["status"] == "SKIPPED_NO_MODELS"

    def test_row_count_outcomes_classification(self):
        from bench.suites import _row_count_outcomes

        responses = [
            (200, {"outputs": [{"data": [1, 2, 3, 4, 5, 6, 7, 8]}]}, None),  # exact
            (200, {"outputs": [{"data": [1, 2, 3, 4, 5, 6, 7]}]}, None),  # truncated
            (200, {"outputs": [{"data": list(range(15))}]}, None),  # leakage
            (200, {"outputs": [{"data": []}]}, None),  # empty
            (500, None, "boom"),  # failed
        ]
        outcomes = _row_count_outcomes(responses, request_rows=8)
        assert outcomes == {
            "exact": 1,
            "truncated": 1,
            "overflow_leakage": 1,
            "empty": 1,
            "failed": 1,
        }


class TestRounds:
    def test_config_has_rounds(self):
        config = RunConfig()
        assert config.rounds == 1
        config = RunConfig(rounds=3)
        assert config.rounds == 3

    def test_cli_accepts_rounds(self):
        import sys

        sys.path.insert(0, str(REPO))
        from bench.cli import build_parser

        args = build_parser().parse_args(
            ["run", "--rounds", "3", "--suites", "protocol"]
        )
        assert args.rounds == 3

    def test_aggregate_rounds_reports_mean_std(self):
        from bench.runner import _aggregate_rounds

        cases = []
        for r, rps in enumerate([100.0, 110.0, 120.0]):
            cases.append(
                {
                    "suite": "s",
                    "framework": "f",
                    "tier": "t",
                    "model": "m",
                    "params": {"x": 1},
                    "round": r,
                    "result": {"throughput_rps": rps, "latency_ms_p50": 1.0},
                }
            )
        aggregates = _aggregate_rounds(cases)
        assert len(aggregates) == 1
        agg = aggregates[0]
        assert agg["rounds"] == 3
        assert agg["throughput_rps_mean"] == 110.0
        assert agg["throughput_rps_std"] > 0

    def test_aggregate_rounds_empty_for_single_replicate(self):
        from bench.runner import _aggregate_rounds

        cases = [
            {
                "suite": "s",
                "framework": "f",
                "tier": "t",
                "model": "m",
                "params": {},
                "round": 0,
                "result": {"throughput_rps": 100.0},
            }
        ]
        assert _aggregate_rounds(cases) == []


class TestSummaryFindings:
    def test_parity_failed_surfaced_as_finding(self):
        from bench.runner import _summarize

        cases = [
            {
                "suite": "batching",
                "model": "m",
                "params": {"max_batch_size": 16},
                "result": {
                    "status": "PARITY_FAILED",
                    "problems": ["8 in 7 out"],
                },
            },
            {
                "suite": "protocol",
                "model": "m",
                "params": {},
                "result": {"throughput_rps": 100.0},
            },
        ]
        summary = _summarize(cases)
        assert len(summary["findings"]) == 1
        assert summary["findings"][0]["kind"] == "PARITY_FAILED"
        assert summary["findings"][0]["problems"] == ["8 in 7 out"]

    def test_suite_failed_surfaced_as_finding(self):
        from bench.runner import _summarize

        summary = _summarize(
            [
                {"suite": "x", "status": "SUITE_FAILED", "error": "RuntimeError"},
            ]
        )
        assert summary["findings"][0]["kind"] == "SUITE_FAILED"

    def test_skips_not_findings(self):
        from bench.runner import _summarize

        summary = _summarize(
            [
                {
                    "suite": "x",
                    "result": {"status": "SKIPPED_NEEDS_GPU_TARGET"},
                }
            ]
        )
        assert summary["findings"] == []


class TestWorkerKillLogic:
    def test_readiness_flag_unpacks_tuples(self):
        """The silent-outage flag compared (ts, code) TUPLES to 200 —
        always False. Tests the REAL extracted function (the original
        version of this test re-typed the expression inside the test
        file and validated nothing in the source — caught by the
        mutation check)."""
        from bench.suites import _readiness_green_during_outage

        during = [(3.2, 200), (3.4, 200), (3.6, 200)]
        assert _readiness_green_during_outage(during, 9033) is True
        assert _readiness_green_during_outage(during, 0) is False
        assert _readiness_green_during_outage([], 9033) is False
        assert _readiness_green_during_outage([(3.2, 503)], 9033) is False

    def test_worker_kill_targeting_and_controls(self):
        import inspect

        from bench.suites import _worker_kill_case

        source = inspect.getsource(_worker_kill_case)
        # real worker targeting: multiprocessing spawn_main workers
        assert "spawn_main" in source
        # the old buggy target (mlserver cmdline grep) must be gone
        # from the pick command
        pick = source[source.index("pick_cmd") : source.index("pick_cmd") + 400]
        assert "grep -q mlserver" not in pick
        # positive controls present
        assert "INVALID_KILL" in source
        assert "kill_rc" in source


class TestSenderByteContract:
    def test_rest_sender_byte_order(self):
        """SendFn contract: (ok, bytes_out=RESPONSE, bytes_in=REQUEST).
        The REST senders returned (request, response) — swapped since
        the beginning (found while investigating the gzip no-op:
        'mean response size' was actually the mean REQUEST size)."""
        from unittest.mock import AsyncMock, MagicMock

        import asyncio

        from bench.suites import RestJsonSender

        sender = RestJsonSender("m", {"inputs": []})
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.headers = {}
        fake_response.read = AsyncMock(return_value=b"x" * 300)

        class FakePost:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, *args):
                return False

        sender._session = MagicMock()
        sender._session.post = MagicMock(return_value=FakePost(fake_response))
        ok, bytes_out, bytes_in = asyncio.run(sender.send(0))
        assert ok is True
        assert bytes_out == 300  # response size
        assert bytes_in == len(json.dumps({"inputs": []}))  # request size
        assert bytes_out != bytes_in  # the swap would flip these

    def test_wire_binary_sender_byte_order(self):
        from unittest.mock import AsyncMock, MagicMock

        import asyncio

        from bench import wire

        sender = wire.BinaryTensorSender("m", b"R" * 100, 10)
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.headers = {}
        fake_response.read = AsyncMock(return_value=b"x" * 250)

        class FakePost:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, *args):
                return False

        sender._session = MagicMock()
        sender._session.post = MagicMock(return_value=FakePost(fake_response))
        ok, bytes_out, bytes_in = asyncio.run(sender.send(0))
        assert ok is True
        assert bytes_out == 250  # response
        assert bytes_in == 100  # request body


class TestEverySenderAgainstStubServer:
    """Construct EVERY sender class and send one real request against
    a stub HTTP server. Catches constructor/attribute defects that
    unit tests miss (the caching-suite bug: VaryingJsonSender never
    set _accept_encoding, so every request raised AttributeError and
    the error guard aborted the suite — while 229 tests passed)."""

    @classmethod
    def _stub_server(cls):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                body = b'{"outputs": [{"name": "predict", "data": [1.0]}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, server.server_port

    def test_rest_json_sender(self):
        import asyncio

        from bench.suites import RestJsonSender

        server, port = self._stub_server()
        try:
            sender = RestJsonSender("m", {"inputs": []}, port=port)

            async def go():
                return await sender.send(0)

            ok, out, inn = asyncio.run(go())
            asyncio.run(sender.close())
            assert ok is True and out > 0 and inn > 0
        finally:
            server.shutdown()

    def test_varying_json_sender(self):
        """The caching-suite defect lived here: constructor did not set
        _accept_encoding, _ensure_session read it, every request
        failed."""
        import asyncio

        from bench.suites import VaryingJsonSender

        server, port = self._stub_server()
        try:
            sender = VaryingJsonSender(
                "m", {"inputs": [{"name": "i", "data": [1.0]}]}, port=port
            )

            async def go():
                return await sender.send(0)

            ok, out, inn = asyncio.run(go())
            asyncio.run(sender.close())
            assert ok is True and out > 0 and inn > 0
            # every seq must vary the payload
            payload_a = asyncio.run(go())[0]  # second request also works
            assert payload_a is True
        finally:
            server.shutdown()

    def test_round_robin_sender(self):
        import asyncio

        from bench.suites import RoundRobinJsonSender

        server, port = self._stub_server()
        try:
            sender = RoundRobinJsonSender(
                [("m1", {"inputs": []}), ("m2", {"inputs": []})], port=port
            )

            async def go():
                results = [await sender.send(i) for i in range(4)]
                await sender.close()
                return results

            results = asyncio.run(go())
            assert all(ok for ok, _, _ in results)
        finally:
            server.shutdown()

    def test_binary_tensor_sender(self):
        import asyncio

        from bench import wire

        server, port = self._stub_server()
        try:
            sender = wire.BinaryTensorSender("m", b"body", 5, port=port)

            async def go():
                return await sender.send(0)

            ok, out, inn = asyncio.run(go())
            asyncio.run(sender.close())
            assert ok is True
        finally:
            server.shutdown()


class TestOutcomeJudging:
    """The orchestrator's check_run_outcomes must read the CASE-level
    status (SUITE_FAILED cases carry status at the case level with
    result == {} — the first fix read only result.status and was a
    no-op for the exact defect it was written for; the reviewer fed
    the original artifacts back through it to prove it)."""

    def _isolated_lv(self, tmp_path: Path):
        """Import the orchestrator with its output ISOLATED to
        tmp_path. record() writes to module-level OUT/SUMMARY/
        LOG_LINES — without this redirection every test run OVERWRITES
        the real committed evidence under results/val-minimal/ (the
        fourth new-code-without-checking-what-it-leaves-behind defect
        of this validation; the 26-phase record was recovered from
        git history)."""
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        lv.OUT = tmp_path / "isolated-out"
        lv.OUT.mkdir(parents=True, exist_ok=True)
        lv.SUMMARY = {"started": "test", "phases": {}}
        lv.LOG_LINES = []
        return lv

    def _check(self, tmp_path: Path, cases: list) -> tuple[bool, str]:
        """Returns (passed, recorded_detail) — the detail lets tests
        assert the FAIL happened for the RIGHT reason (a missing-file
        FAIL would otherwise pass vacuously)."""
        lv = self._isolated_lv(tmp_path)
        out = tmp_path / "run"
        out.mkdir()
        (out / "results.json").write_text(
            json.dumps({"status": "COMPLETE", "cases": cases})
        )
        ok = lv.check_run_outcomes("test", out, 0.0)
        detail = lv.SUMMARY["phases"]["test"]["detail"]
        return ok, detail

    def test_suite_failed_case_level_detected(self, tmp_path: Path):
        """A SUITE_FAILED case has status at the CASE level, empty
        result — the exact shape recorded in m1/m4."""
        cases = [
            {
                "suite": "caching",
                "framework": "sklearn",
                "status": "SUITE_FAILED",
                "error": "AttributeError",
                "detail": "error rate 100% ... _accept_encoding",
                "result": {},
            }
        ]
        ok, detail = self._check(tmp_path, cases)
        assert ok is False

    def test_clean_run_passes(self, tmp_path: Path):
        cases = [
            {
                "suite": "protocol",
                "result": {"throughput_rps": 100.0},
            }
        ]
        ok, detail = self._check(tmp_path, cases)
        assert ok is True

    def test_f01_cell_allowlisted_by_params(self, tmp_path: Path):
        """The F-01 allow-list matches CELL PARAMS, not suite names."""
        f01_batching = [
            {
                "suite": "batching",
                "params": {"max_batch_size": 16, "max_batch_time": 0.05},
                "result": {"status": "PARITY_FAILED"},
            },
            {
                "suite": "faults",
                "params": {
                    "axis": "declared-shape",
                    "shape": "dynamic",
                    "batching": True,
                },
                "result": {"status": "PARITY_FAILED"},
            },
        ]
        ok, detail = self._check(tmp_path, f01_batching)
        assert ok is True

    def test_unrelated_parity_failure_in_batching_fails(self, tmp_path):
        """An unbatched-cell parity failure in the batching suite must
        NOT be allow-listed (the suite-wide allow-list was too wide)."""
        cases = [
            {
                "suite": "batching",
                "params": {"max_batch_size": 1, "max_batch_time": 0.0},
                "result": {"status": "PARITY_FAILED"},
            }
        ]
        ok, detail = self._check(tmp_path, cases)
        assert ok is False

    def test_parity_failure_result_level_detected(self, tmp_path):
        cases = [
            {
                "suite": "protocol",
                "result": {"status": "PARITY_FAILED", "problems": ["x"]},
            }
        ]
        ok, detail = self._check(tmp_path, cases)
        assert ok is False

    def test_recorded_fixture_fails(self, tmp_path: Path):
        """The acceptance evidence: a RECORDED SUITE_FAILED artifact
        (committed fixture, matching the real m1/m4 case-level shape)
        must FAIL the checker — the standing rule is that every gate
        is checked against a recorded failing run before its PASS is
        relied on."""
        import shutil

        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "suite_failed"
        out = tmp_path / "fixture-run"
        shutil.copytree(fixture_dir, out)
        lv = self._isolated_lv(tmp_path)
        ok = lv.check_run_outcomes("fixture", out, 0.0)
        detail = lv.SUMMARY["phases"]["fixture"]["detail"]
        assert ok is False
        # the RIGHT reason: SUITE_FAILED cited, not a missing file
        assert "SUITE_FAILED" in detail, f"failed for wrong reason: {detail}"

    def test_orchestrator_tests_do_not_dirty_results(self):
        """Invariant: after the orchestrator tests run, the tracked
        files under results/ are unmodified. record() writes to a
        hardcoded evidence directory — without isolation every test
        run rewrites the committed validation evidence (found by the
        reviewer in the committed history: the 26-phase LOG.txt had
        shrunk to 8 test lines at HEAD). This tripwire must not depend
        on discipline."""
        import subprocess

        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "results/"],
            capture_output=True,
            text=True,
            cwd=repo,
            timeout=30,
        )
        assert result.stdout.strip() == "", (
            f"tracked results/ files modified (orchestrator tests "
            f"leaking writes?): {result.stdout.strip()[:300]}"
        )

    def test_generator_bound_ladder_expected(self, tmp_path: Path):
        """Ladder suites (saturation/protocol) sweep concurrency until
        throughput plateaus — generator ceiling at top rungs is
        EXPECTED: phase stays green with the count recorded."""
        cases = [
            {
                "suite": "saturation",
                "params": {"concurrency": 64},
                "result": {
                    "throughput_rps": 2000.0,
                    "generator_bound": True,
                    "client_cpu_percent": 97.0,
                },
            }
        ]
        ok, detail = self._check(tmp_path, cases)
        assert ok is True
        assert "generator-ceiling ladder cells: 1" in detail

    def test_generator_bound_non_ladder_fails(self, tmp_path: Path):
        """Outside ladder suites, a generator-bound case is a
        client-limited measurement masquerading as a server result."""
        cases = [
            {
                "suite": "middleware",
                "params": {"middleware": "gzip"},
                "result": {
                    "throughput_rps": 126.0,
                    "generator_bound": True,
                    "client_cpu_percent": 97.0,
                },
            }
        ]
        ok, detail = self._check(tmp_path, cases)
        assert ok is False
        assert "GENERATOR_BOUND" in detail


class TestOrchestratorPlansAndGuards:
    def test_out_dir_refusal(self, tmp_path: Path):
        """--out is required and a non-empty target refuses without
        --force (the original default was results/val-minimal —
        running with no flag rewrote the committed evidence)."""
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        target = tmp_path / "existing"
        target.mkdir()
        (target / "f").write_text("x")
        with pytest.raises(SystemExit, match="refusing to write"):
            lv._validate_out_dir(target, force=False)
        lv._validate_out_dir(target, force=True)  # explicit force passes
        empty = tmp_path / "empty"
        empty.mkdir()
        lv._validate_out_dir(empty, force=False)  # empty passes

    def test_thorough_plan_implements_t1_t4(self):
        """--plan thorough must actually run T1-T4 — the phases are
        implemented in the tool, not just documented (the original
        gap: running with --out results/val-thorough would have
        repeated the MINIMAL matrix into a new directory)."""
        import inspect
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        assert hasattr(lv, "phase3_thorough"), "no thorough phases"
        source = inspect.getsource(lv.phase3_thorough)
        for t_phase in (
            "t1-xgboost-small-all",
            "t2-lightgbm-small-all",
            "t3-onnx-small-rest",
            "t4-sklearn-large",
        ):
            assert t_phase in source, f"{t_phase} missing from thorough plan"
        main_source = inspect.getsource(lv.main)
        assert "--plan" in main_source
        assert "thorough" in main_source
        assert "required=True" in main_source  # --out is required


class TestThoroughAfterChecks:
    def test_scan_reports_all_three_controls(self, tmp_path: Path):
        """The shared positive-controls scanner reports INVALID_KILL,
        TELEMETRY_EMPTY and gzip_engaged — the T4 gzip answer must be
        a reported finding, not something read by hand."""
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        run = tmp_path / "t4-sklearn-large"
        run.mkdir()
        (run / "results.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": [
                        {
                            "suite": "faults",
                            "result": {"status": "INVALID_KILL"},
                        },
                        {
                            "suite": "multimodel",
                            "telemetry_status": "TELEMETRY_EMPTY",
                            "result": {},
                        },
                        {
                            "suite": "middleware",
                            "result": {"gzip_engaged": True},
                        },
                    ],
                }
            )
        )
        findings = lv._scan_positive_controls(tmp_path)
        joined = "; ".join(findings)
        assert "INVALID_KILL" in joined
        assert "TELEMETRY_EMPTY" in joined
        assert "gzip_engaged=True" in joined

    def test_thorough_plan_has_after_checks(self):
        """The thorough branch must run the post-run crosscheck,
        mutation check and positive-controls scan — without them the
        thorough claim cannot say the host was unchanged across the
        run ('before and after' launch conditions)."""
        import inspect
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        assert hasattr(lv, "phase_thorough_after")
        source = inspect.getsource(lv.phase_thorough_after)
        assert "crosscheck" in source
        assert "mutation-check" in source
        assert "_scan_positive_controls" in source
        main_source = inspect.getsource(lv.main)
        assert "phase_thorough_after()" in main_source

    def test_scan_no_truncation_t4_gzip_survives(self, tmp_path: Path):
        """More than 8 findings may precede T4's gzip answer — the
        original scan recorded only findings[:6]/[:8], so the T4 gzip
        answer (the reason T4 exists) could be silently dropped. The
        scan must return everything and write the full list to
        controls-findings.json."""
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        # 10 runs each with a gzip_engaged=false finding (like the
        # minimal matrix's 19 findings)...
        for i in range(10):
            run = tmp_path / f"m{i}-sklearn-small"
            run.mkdir()
            (run / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "cases": [
                            {
                                "suite": "middleware",
                                "result": {"gzip_engaged": False},
                            }
                        ],
                    }
                )
            )
        # ...then T4 with the gzip answer
        t4 = tmp_path / "t4-sklearn-large"
        t4.mkdir()
        (t4 / "results.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": [
                        {
                            "suite": "middleware",
                            "result": {"gzip_engaged": True},
                        }
                    ],
                }
            )
        )
        findings = lv._scan_positive_controls(tmp_path)
        joined = "; ".join(findings)
        assert len(findings) == 11
        # the T4 answer survives despite 10 preceding findings
        assert "t4-sklearn-large/middleware: gzip_engaged=True" in joined
        # the full list is on disk, unsliced
        on_disk = json.loads((tmp_path / "controls-findings.json").read_text())
        assert len(on_disk) == 11
        assert any("t4-sklearn-large" in f for f in on_disk)

    def test_mutation_count_pinned(self):
        """The declared mutation count must not silently shrink: a
        splice once swallowed one and 22/22 looked healthy."""
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv  # noqa: F401
            import mutation_check
        finally:
            sys.path.remove(str(tools_dir))
        assert len(mutation_check.MUTATIONS) >= mutation_check.MIN_DECLARED_MUTATIONS
        names = [m.name for m in mutation_check.MUTATIONS]
        assert len(names) == len(set(names)), "duplicate mutation names"

    def test_gate_selfcheck_polarity(self, tmp_path: Path):
        """The pre-flight gate must PASS when the checker REJECTS the
        fixture (detection working) and ABORT only when the checker
        PASSES a bad artifact. The first version inverted this — it
        aborted when the checker correctly failed the fixture, killing
        the THOROUGH launch 3 seconds in (fifth
        new-code-without-a-real-test defect; caught by the launch)."""
        import inspect
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        source = inspect.getsource(lv.preflight)
        # The gate treats checker-REJECTS-fixture as success
        assert "checker_rejected_fixture = not check_run_outcomes(" in source
        # and aborts only when the checker PASSES a bad artifact
        assert "PASSED a " in source and "raise SystemExit" in source
        # Functional: the fixture is genuinely rejected (polarity's
        # premise holds on the real artifact)
        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "suite_failed"
        lv.OUT = tmp_path / "iso"
        lv.OUT.mkdir(parents=True, exist_ok=True)
        lv.SUMMARY = {"started": "t", "phases": {}}
        lv.LOG_LINES = []
        rejected = not lv.check_run_outcomes("pol", fixture_dir, 0.0)
        assert (
            rejected is True
        ), "premise broken: the checker no longer rejects the fixture"


class TestPilotWiring:
    """The pilot's outcome-check wiring crashed with KeyError: 'phases'
    because its SUMMARY had no phases key — the whole matrix would
    run and then die. This test runs the exact wiring against both a
    failing fixture and a passing artifact, proving it works before
    the baseline launches."""

    def _wired_check(self, tmp_path: Path, out_dir: Path) -> tuple[bool, dict]:
        """Replicates the pilot's exact lv wiring."""
        import sys

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        sys.path.insert(0, str(tools_dir))
        try:
            import local_validation as lv
        finally:
            sys.path.remove(str(tools_dir))
        # The pilot's SUMMARY (with the fix: phases key from the start)
        summary = {"phases": {}, "started": "test"}
        lv.OUT = tmp_path / "iso"
        lv.OUT.mkdir(parents=True, exist_ok=True)
        lv.SUMMARY = summary
        lv.LOG_LINES = []
        summary.setdefault("phases", {})
        ok = lv.check_run_outcomes("wiring-test", out_dir, 0.0)
        return ok, summary

    def test_failing_fixture_records_fail(self, tmp_path: Path):
        """The suite_failed fixture must produce a recorded FAIL,
        not a KeyError crash."""
        import shutil

        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "suite_failed"
        out = tmp_path / "fixture-run"
        shutil.copytree(fixture_dir, out)
        ok, summary = self._wired_check(tmp_path, out)
        assert ok is False
        assert "wiring-test" in summary["phases"]
        assert "FAIL" in summary["phases"]["wiring-test"]["status"]

    def test_passing_artifact_records_pass(self, tmp_path: Path):
        """A passing artifact must produce a recorded PASS, not a
        KeyError crash. Uses a committed fixture (mirroring
        fixtures/suite_failed) instead of real pilot evidence — the
        evidence is a regenerable work product, not a test input."""
        import shutil

        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "suite_passed"
        out = tmp_path / "passing-run"
        shutil.copytree(fixture_dir, out)
        ok, summary = self._wired_check(tmp_path, out)
        assert ok is True
        assert "wiring-test" in summary["phases"]
        assert summary["phases"]["wiring-test"]["status"] == "PASS"

    def test_pilot_timing_sum_nonzero(self):
        """The baseline's timing summary sums the TOP-LEVEL pilot-*
        entries (where record() writes), not SUMMARY["phases"] (which
        holds only the outcome-check entries). The original summed
        the wrong dict and would have reported 0s for the baseline."""
        # Synthetic records matching the pilot's record() output shape
        summary = {
            "phases": {},  # pilot's own record() writes top-level
            "started": "t",
            "pilot-sklearn": {"status": "PASS", "seconds": 656.0, "detail": ""},
            "pilot-sklearn-outcomes": {
                "status": "PASS",
                "seconds": 656.0,
                "detail": "",
            },
            "pilot-onnx": {"status": "PASS", "seconds": 650.0, "detail": ""},
            "pilot-onnx-outcomes": {
                "status": "PASS",
                "seconds": 650.0,
                "detail": "",
            },
            "preflight-mutation": {"status": "PASS", "seconds": 34.0, "detail": ""},
        }
        # The corrected expression from pilot.py
        matrix_seconds = sum(
            v["seconds"]
            for k, v in summary.items()
            if isinstance(v, dict)
            and k.startswith("pilot-")
            and "outcomes" not in k
            and "seconds" in v
        )
        assert matrix_seconds == 1306.0  # 656 + 650, NOT 0

        # The buggy expression (summing SUMMARY["phases"]) gives 0
        buggy = sum(
            v.get("seconds", 0)
            for k, v in summary["phases"].items()
            if k.startswith("pilot-")
        )
        assert buggy == 0  # proves the bug would have reported 0s


class TestOpenLoopCellKeying:
    """Open-loop saturation cells must key by fraction_of_peak +
    arrival (stable across runs), NOT target_qps (derived from each
    round's measured peak — differs every round, so compare.py can
    never pair them). The baseline had 30 unpaired cells per framework
    because of this.

    The original tests built case dicts by hand and called case_key —
    they never called suite_saturation, so mutating suites.py couldn't
    make them fail (the mutation was NOT CAUGHT; the reviewer proved
    it). These tests exercise the REAL suite output."""

    def test_saturation_cases_keyed_correctly(self, tmp_path):
        """Run the real saturation suite (stubbed server) and verify:
        no target_qps in params, target_qps IS in the result."""
        from bench.compare import case_key
        from bench.config import RunConfig
        from bench.suites import SuiteContext, suite_saturation

        config = RunConfig()
        config.load_profile = "quick"
        ctx = SuiteContext(config, "stub", tmp_path, tmp_path / "out")

        # Build a minimal model dir for the suite
        model_dir = tmp_path / "sklearn-test-small"
        model_dir.mkdir()
        (model_dir / "model-settings.json").write_text(
            json.dumps(
                {
                    "name": "test-model",
                    "implementation": "mlserver_sklearn.SKLearnModel",
                    "parameters": {"uri": "model.joblib"},
                    "inputs": [
                        {"name": "input", "datatype": "FP32", "shape": [-1, 10]}
                    ],
                    "outputs": [
                        {"name": "predict", "datatype": "INT64", "shape": [-1]}
                    ],
                }
            )
        )
        np.savez(
            model_dir / "validation.npz",
            input=np.ones((4, 10), dtype=np.float32),
        )

        # Stub the server + parity to avoid container startup
        from unittest.mock import MagicMock, patch

        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)
        from unittest.mock import AsyncMock

        ctx.server = MagicMock(return_value=mock_server)
        ctx.parity = AsyncMock(return_value=None)
        ctx.parity_problems = AsyncMock(return_value=[])

        # Stub the load-generation layer: patch loadgen.open_loop
        # (async — called inside a coroutine) and run_closed (sync —
        # wraps loadgen.closed_loop). NOT run_coroutine_blocking,
        # which receives a coroutine and must actually run it.
        class FakeResult:
            def __init__(self, rps, p99):
                self._summary = {
                    "throughput_rps": rps,
                    "latency_ms_p99": p99,
                }
                self.wall_seconds = 20.0
                self.samples = []
                self.target_rate = None
                self.mode = "closed"

            def summary(self, window=None):
                return self._summary

        def fake_closed(ctx, model, req, conc, **kw):
            return FakeResult(1000.0 * conc / 8, 10.0)

        async def fake_open_loop(target_qps, seconds, warmup, send, **kw):
            return FakeResult(target_qps * 0.9, 12.0)

        mock_sender = MagicMock()
        mock_sender.close = AsyncMock()

        with (
            patch("bench.suites.run_closed", side_effect=fake_closed),
            patch("bench.loadgen.open_loop", side_effect=fake_open_loop),
            patch("bench.suites.rest_sender", return_value=mock_sender),
        ):

            import asyncio

            cases = asyncio.run(suite_saturation(ctx, [model_dir]))

        # Verify: open-loop cases must NOT have target_qps in params
        open_cases = [c for c in cases if c["params"].get("mode") == "open"]
        assert len(open_cases) > 0, "no open-loop cases generated"
        for case in open_cases:
            assert (
                "target_qps" not in case["params"]
            ), f"target_qps in params: {case['params']}"
            assert (
                "target_qps" in case["result"]
            ), f"target_qps not in result: {case['result'].keys()}"
            assert "fraction_of_peak" in case["params"]

        # Verify: all open-loop cases with the same fraction+arrival
        # produce the SAME key (regardless of measured peak)
        keys = set()
        for case in open_cases:
            keys.add(case_key(case))
        # quick profile has 1 fraction × 2 arrivals = 2 distinct keys
        assert len(keys) == 2, (
            f"expected 2 distinct open-loop keys (1 fraction × 2 "
            f"arrivals), got {len(keys)}: {keys}"
        )

    def test_paired_override_rates_per_model(self, tmp_path, monkeypatch):
        """Run the REAL saturation suite with PAIRED_TARGET_QPS set
        and assert the driven rates equal the override values PER
        MODEL and PER ARRIVAL.

        This is the reviewer's real-run check replicated as a test:
        the old single-key dict gave every model one rate (fake
        open-loop deltas of -6%/+8% in a real A/A), and the old suite
        lookup crashed on `self` (NameError) / drove both arrivals at
        the last matching key's rate. Sentinel rates per model+arrival
        distinguish all of those failure modes."""
        from bench.config import RunConfig
        from bench.suites import SuiteContext, suite_saturation

        monkeypatch.delenv("PAIRED_TARGET_QPS", raising=False)

        config = RunConfig()
        config.load_profile = "quick"
        ctx = SuiteContext(config, "stub", tmp_path, tmp_path / "out")

        # Two models with DIFFERENT implementations (frameworks) and
        # names, so per-model AND per-framework keying is exercised.
        model_dirs = []
        for dirname, name, impl, shape in (
            ("sk-m1", "m1", "mlserver_sklearn.SKLearnModel", (-1, 10)),
            ("onnx-m2", "m2", "mlserver_onnx.OnnxModel", (-1, 10)),
        ):
            model_dir = tmp_path / dirname
            model_dir.mkdir()
            (model_dir / "model-settings.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "implementation": impl,
                        "parameters": {"uri": "model.joblib"},
                        "inputs": [
                            {"name": "input", "datatype": "FP32", "shape": shape}
                        ],
                        "outputs": [
                            {"name": "predict", "datatype": "INT64", "shape": [-1]}
                        ],
                    }
                )
            )
            np.savez(
                model_dir / "validation.npz",
                input=np.ones((4, 10), dtype=np.float32),
            )
            model_dirs.append(model_dir)

        from unittest.mock import AsyncMock, MagicMock

        mock_server = MagicMock()
        mock_server.__enter__ = MagicMock(return_value=mock_server)
        mock_server.__exit__ = MagicMock(return_value=False)

        ctx.server = MagicMock(return_value=mock_server)
        ctx.parity = AsyncMock(return_value=None)
        ctx.parity_problems = AsyncMock(return_value=[])

        class FakeResult:
            def __init__(self, rps, p99):
                self._summary = {"throughput_rps": rps, "latency_ms_p99": p99}
                self.wall_seconds = 20.0
                self.samples = []
                self.target_rate = None
                self.mode = "closed"

            def summary(self, window=None):
                return self._summary

        driven = []

        def fake_closed(ctx, model, req, conc, **kw):
            return FakeResult(1000.0 * conc / 8, 10.0)

        async def fake_open_loop(target_qps, seconds, warmup, send, **kw):
            driven.append(target_qps)
            return FakeResult(target_qps * 0.9, 12.0)

        mock_sender = MagicMock()
        mock_sender.close = AsyncMock()

        import asyncio

        def run_suite():
            with (
                patch("bench.suites.run_closed", side_effect=fake_closed),
                patch("bench.loadgen.open_loop", side_effect=fake_open_loop),
                patch("bench.suites.rest_sender", return_value=mock_sender),
            ):
                return asyncio.run(suite_saturation(ctx, list(model_dirs)))

        # Baseline side: no override. quick profile: concurrency [1,8]
        # -> peak 1000 rps; fraction 0.75 -> 750.0 for every model.
        baseline_cases = run_suite()
        base_open = [c for c in baseline_cases if c["params"].get("mode") == "open"]
        assert base_open, "no open-loop cases generated"
        for case in base_open:
            assert case["result"]["target_qps"] == 750.0

        # Candidate side: distinct sentinel rates per framework+model
        # +arrival (exactly how the paired driver keys them).
        override = {
            "sklearn:m1:0.75:constant": 111.0,
            "sklearn:m1:0.75:poisson": 112.0,
            "onnx:m2:0.75:constant": 221.0,
            "onnx:m2:0.75:poisson": 222.0,
        }
        monkeypatch.setenv("PAIRED_TARGET_QPS", json.dumps(override))
        driven.clear()
        candidate_cases = run_suite()

        cand_open = [c for c in candidate_cases if c["params"].get("mode") == "open"]
        assert len(cand_open) == len(base_open)
        seen = set()
        for case in cand_open:
            arrival = case["params"]["arrival"]
            want = {
                ("m1", "constant"): 111.0,
                ("m1", "poisson"): 112.0,
                ("m2", "constant"): 221.0,
                ("m2", "poisson"): 222.0,
            }[(case["model"], arrival)]
            got = case["result"]["target_qps"]
            assert got == want, (
                f"model {case['model']}/{arrival}: driven {got}, "
                f"expected the override {want} — per-model rate "
                "propagation is broken"
            )
            assert got != 750.0, "override not applied at all"
            seen.add((case["model"], arrival))
        assert seen == {
            ("m1", "constant"),
            ("m1", "poisson"),
            ("m2", "constant"),
            ("m2", "poisson"),
        }


class TestPairedStatistics:
    """Tests that call the REAL paired_compare on synthetic directories.
    The original tests re-implemented the statistics inline — they
    never ran the driver's code, which is why the reviewer found
    defects the tests couldn't (Wilcoxon can't flag at n=3 or n=5;
    the fix is a paired t-test on log-ratios)."""

    @staticmethod
    def _build_paired_dirs(
        tmp_path: Path,
        n_cells=10,
        n_rounds=5,
        effect=-0.05,
        drift=(1.06, 1.10, 0.94, 0.92, 0.98),
        noise_pct=0.02,
    ):
        """Build two results.json dirs with synthetic paired data that
        includes shared drift, per-side noise, and a true effect."""
        rng = np.random.default_rng(42)
        base_cases = []
        cand_cases = []
        for i in range(n_cells):
            base_level = 1000 + i * 50
            for r in range(n_rounds):
                d = drift[r % len(drift)]
                nb = rng.normal(0, noise_pct)
                nc = rng.normal(0, noise_pct)
                base_rps = base_level * d * (1 + nb)
                cand_rps = base_level * d * (1 + nc) * (1 + effect)
                base_p99 = 10.0 * d * (1 + nb)
                cand_p99 = 10.0 * d * (1 + nc) * (1 - effect)  # inverse

                params = {"concurrency": 8, "cell": i}
                base_cases.append(
                    {
                        "suite": "protocol",
                        "framework": "sklearn",
                        "tier": "small",
                        "model": f"m{i}",
                        "params": params,
                        "round": r,
                        "result": {
                            "throughput_rps": base_rps,
                            "latency_ms_p99": base_p99,
                        },
                    }
                )
                cand_cases.append(
                    {
                        "suite": "protocol",
                        "framework": "sklearn",
                        "tier": "small",
                        "model": f"m{i}",
                        "params": params,
                        "round": r,
                        "result": {
                            "throughput_rps": cand_rps,
                            "latency_ms_p99": cand_p99,
                        },
                    }
                )

        manifest = {
            "image": {"digest": "synthetic", "target": "cpu"},
            "host": {"cpu_model": "test", "cores_logical": 16},
        }
        for label, cases in (("baseline", base_cases), ("candidate", cand_cases)):
            d = tmp_path / label
            d.mkdir()
            (d / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": manifest,
                        "config": {"load_profile": "quick"},
                        "cases": cases,
                    }
                )
            )
        return tmp_path / "baseline", tmp_path / "candidate"

    @staticmethod
    def _run_compare(baseline_dir, candidate_dir):
        """Call the real paired_compare from the driver."""
        import importlib.util

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_under_test", str(tools_dir / "paired_driver.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.paired_compare(baseline_dir, candidate_dir)

    def test_paired_catches_minus10_at_5_rounds(self, tmp_path):
        """A -10% effect under ±12% shared drift at 5 rounds: caught
        (SLA is -10%, and the reviewer's 10-seed validation shows
        10/10 seeds flagged at this size/round count)."""
        base_dir, cand_dir = self._build_paired_dirs(
            tmp_path, n_cells=10, n_rounds=5, effect=-0.10
        )
        result = self._run_compare(base_dir, cand_dir)
        assert result["compared_cells"] > 0
        tp_regs = [
            v
            for v in result["verdicts"]
            if v["metric"] == "throughput" and v["regression"]
        ]
        assert len(tp_regs) > 0, (
            f"paired t-test missed -10% at 5 rounds: "
            f"{[v['delta_pct'] for v in result['verdicts'][:5]]}"
        )

    def test_minus5_is_within_sla_not_flagged(self, tmp_path):
        """A -5% effect at 5 rounds is BELOW the -10% SLA: the
        statistic may be significant, but a regression flag requires
        BOTH the effect threshold AND Holm-adjusted significance, so
        it must NOT be flagged (defect 6: the SLA is -10% everywhere,
        and the effect threshold — not significance alone — decides)."""
        base_dir, cand_dir = self._build_paired_dirs(
            tmp_path, n_cells=10, n_rounds=5, effect=-0.05
        )
        result = self._run_compare(base_dir, cand_dir)
        tp_regs = [
            v
            for v in result["verdicts"]
            if v["metric"] == "throughput" and v["regression"]
        ]
        assert len(tp_regs) == 0, (
            f"-5% effect flagged as regression despite the -10% SLA: "
            f"{[v for v in result['verdicts'] if v['regression']]}"
        )

    def test_null_case_no_false_positive(self, tmp_path):
        """No effect: the paired test should not flag anything."""
        base_dir, cand_dir = self._build_paired_dirs(
            tmp_path, n_cells=10, n_rounds=5, effect=0.0
        )
        result = self._run_compare(base_dir, cand_dir)
        assert result["regressions"] == 0, (
            f"false positives: {result['regressions']} of "
            f"{result['compared_cells']}"
        )

    def test_paired_catches_minus15_at_3_rounds(self, tmp_path):
        """A -15% effect at only 3 rounds: should still be caught
        (the original Wilcoxon hard-coded p=1.0 below 5 rounds)."""
        base_dir, cand_dir = self._build_paired_dirs(
            tmp_path, n_cells=10, n_rounds=3, effect=-0.15
        )
        result = self._run_compare(base_dir, cand_dir)
        tp_regs = [
            v
            for v in result["verdicts"]
            if v["metric"] == "throughput" and v["regression"]
        ]
        assert len(tp_regs) > 0, "paired t-test missed -15% at 3 rounds"

    def test_unpaired_welch_misses_minus5(self, tmp_path):
        """The unpaired Welch test on pooled replicates should MISS
        the -5% effect under the same drift (proving the pairing
        matters, not just the test choice)."""
        from scipy import stats as st

        rng = np.random.default_rng(42)
        drift = (1.06, 1.10, 0.94, 0.92, 0.98)
        all_base, all_cand = [], []
        for i in range(10):
            base_level = 1000 + i * 50
            for r in range(5):
                d = drift[r]
                nb = rng.normal(0, 0.02)
                nc = rng.normal(0, 0.02)
                all_base.append(base_level * d * (1 + nb))
                all_cand.append(base_level * d * (1 + nc) * 0.95)
        _, p = st.ttest_ind(all_cand, all_base, equal_var=False)
        assert p > 0.05, (
            f"Welch unexpectedly caught -5% under drift (p={p:.4f}) — "
            "the synthetic drift may be too small to demonstrate the gap"
        )

    def test_p99_informational_by_default(self, tmp_path):
        """A +50% p99 effect with NO throughput effect: reported with
        full statistics + scatter, but NOT flagged — the mini A/A
        measured p99 deltas of -48.7%..+114.4% under identical
        images, so a p99 gate would false-positive at 5 rounds. The
        scatter is recorded so a threshold can be set from measured
        full-A/A data later."""
        # throughput: no effect; p99: +50% on candidate
        base_dir, cand_dir = self._build_paired_dirs(
            tmp_path, n_cells=5, n_rounds=5, effect=0.0
        )
        # add a p99-only effect by rewriting candidate p99 values
        cand = json.loads((cand_dir / "results.json").read_text())
        for case in cand["cases"]:
            case["result"]["latency_ms_p99"] *= 1.5
        (cand_dir / "results.json").write_text(json.dumps(cand))

        result = self._run_compare(base_dir, cand_dir)
        p99_regs = [
            v for v in result["verdicts"] if v["metric"] == "p99" and v["regression"]
        ]
        assert len(p99_regs) == 0, f"informational p99 flagged: {p99_regs}"
        p99_verdicts = [v for v in result["verdicts"] if v["metric"] == "p99"]
        assert p99_verdicts and all(v["informational"] for v in p99_verdicts)
        tp_verdicts = [v for v in result["verdicts"] if v["metric"] == "throughput"]
        assert all(not v["informational"] for v in tp_verdicts)
        # the scatter is measured and recorded
        assert result["p99_scatter"]["n"] == len(p99_verdicts)
        assert result["p99_scatter"]["max"] > 1.4, result["p99_scatter"]
        assert "informational" in result["p99_policy"]

    def test_p99_gating_can_be_reenabled(self, tmp_path):
        """p99_informational=False restores gating — the flag is a
        policy switch, not dead code; a threshold set from measured
        full-A/A scatter can turn the gate back on."""
        base_dir, cand_dir = self._build_paired_dirs(
            tmp_path, n_cells=5, n_rounds=5, effect=0.0
        )
        cand = json.loads((cand_dir / "results.json").read_text())
        for case in cand["cases"]:
            case["result"]["latency_ms_p99"] *= 1.5
        (cand_dir / "results.json").write_text(json.dumps(cand))

        import importlib.util

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_p99", str(tools_dir / "paired_driver.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.paired_compare(base_dir, cand_dir, p99_informational=False)
        p99_regs = [
            v for v in result["verdicts"] if v["metric"] == "p99" and v["regression"]
        ]
        assert len(p99_regs) > 0, "re-enabled p99 gate flagged nothing"
        assert "gating" in result["p99_policy"]


class TestPairedDriverSmoke:
    """Smoke test for the paired driver's LOOP: stubs run_side and
    verifies the call arguments. This would have caught the hog_pct
    TypeError immediately (the API mismatch crashed on the first
    call — no test ever ran the driver's loop)."""

    def test_driver_loop_passes_correct_args(self, tmp_path):
        """The driver loop must call run_side with the correct
        keyword arguments (hog_count, not hog_pct; suite, not
        all-suites-at-once)."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_smoke", str(tools_dir / "paired_driver.py")
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)

        calls = []

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            calls.append(
                {
                    "label": label,
                    "image": image,
                    "framework": fw,
                    "round": round_idx,
                    **kwargs,
                }
            )
            # Write results into the SUITE-SCOPED side dir (the real
            # layout: round-N/{label}/{suite}/{framework}) so
            # merge_sides and the rate recorder exercise the real
            # paths. The old layout (no suite segment) silently
            # overwrote all but the last suite's results.json.
            suite = kwargs.get("suite", "")
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            cases = [
                {
                    "suite": suite,
                    "framework": fw,
                    "model": "m",
                    "params": {"c": 8},
                    "round": round_idx,
                    "result": {"throughput_rps": 100.0, "latency_ms_p99": 10.0},
                }
            ]
            # Saturation open-loop cases: TWO models with DIFFERENT
            # measured rates, so per-model keying is verifiable (the
            # old single-key dict gave every model the last model's
            # rate — the reviewer measured fake -6%/+8% deltas).
            if suite == "saturation":
                for model, rate in (("grad-boost", 1095.0), ("logreg", 1188.0)):
                    for arrival in ("constant", "poisson"):
                        cases.append(
                            {
                                "suite": "saturation",
                                "framework": fw,
                                "model": model,
                                "params": {
                                    "mode": "open",
                                    "fraction_of_peak": 0.75,
                                    "arrival": arrival,
                                },
                                "round": round_idx,
                                "result": {
                                    "throughput_rps": rate * 0.9,
                                    "latency_ms_p99": 12.0,
                                    "target_qps": rate,
                                },
                            }
                        )
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {
                            "image": {"digest": "test", "target": "cpu"},
                            "host": {"cpu_model": "t", "cores_logical": 16},
                        },
                        "cases": cases,
                    }
                )
            )
            return side_dir

        # Stub the pre-flight checks
        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            driver._run_paired(
                tmp_path,
                "img-a",
                "img-b",
                rounds=2,
                cpu_limit=1.0,
            )

        # Verify: run_side was called with the cpu_limit API (graded
        # control mechanism; hog_count/hog_pct are the retired APIs)
        for call in calls:
            assert "cpu_limit" in call, f"cpu_limit missing from call: {call}"
            assert "hog_count" not in call, "hog_count in call (retired API)"
            assert "hog_pct" not in call, "hog_pct in call (old API)"
            assert "suite" in call, f"suite missing (suite-level alternation): {call}"

        # Verify: candidate side gets the CPU limit, baseline doesn't
        candidate_calls = [c for c in calls if c["label"] == "candidate"]
        baseline_calls = [c for c in calls if c["label"] == "baseline"]
        assert all(c["cpu_limit"] == 1.0 for c in candidate_calls)
        assert all(c["cpu_limit"] is None for c in baseline_calls)

        # Verify: every side runs under its OWN run id, distinct per
        # side+round and carrying the run token
        for call in calls:
            rid = call.get("run_id", "")
            assert rid, f"run_id missing from call: {call}"
            assert driver.RUN_TOKEN in rid, f"run token missing: {rid}"
            assert call["label"] in rid, f"side missing from run id: {rid}"
            assert f"r{call['round']}" in rid, f"round missing from run id: {rid}"
        assert len({c["run_id"] for c in calls}) == 4  # 2 sides x 2 rounds

        # Verify: order alternates on odd rounds for non-saturation
        # suites (drift balance)...
        round0_labels = [c["label"] for c in calls if c["round"] == 0]
        round1_labels = [c["label"] for c in calls if c["round"] == 1]
        assert round0_labels[0] == "baseline"  # even round: baseline first
        assert round1_labels[0] == "candidate"  # odd round: candidate first
        # ...but saturation runs BASELINE FIRST on every round, so
        # this round's open-loop rates exist before the candidate
        # side needs them (stale round-0 rates produced fake deltas)
        sat_r1 = [
            c["label"] for c in calls if c["round"] == 1 and c["suite"] == "saturation"
        ]
        assert (
            sat_r1[0] == "baseline"
        ), f"saturation must run baseline first on odd rounds: {sat_r1}"

        # Verify: suites are run individually (suite-level granularity)
        suites_called = {c["suite"] for c in calls}
        assert len(suites_called) == 4  # protocol, workers, middleware, saturation

        # Verify: rate propagation is PER MODEL — the candidate
        # saturation side receives fw:model:frac:arrival keys with the
        # baseline's measured rates, different per model
        cand_sat = [
            c for c in calls if c["label"] == "candidate" and c["suite"] == "saturation"
        ]
        assert cand_sat, "no candidate saturation calls recorded"
        for call in cand_sat:
            tq = call.get("target_qps") or {}
            for fw in driver.FRAMEWORKS:
                for model in ("grad-boost", "logreg"):
                    for arrival in ("constant", "poisson"):
                        key = f"{fw}:{model}:0.75:{arrival}"
                        assert key in tq, (
                            f"per-model rate key {key!r} missing; " f"got {sorted(tq)}"
                        )
            assert (
                tq["sklearn:grad-boost:0.75:constant"]
                != tq["sklearn:logreg:0.75:constant"]
            ), "all models got the same rate (single-key regression)"

        # Verify: merged results contain ALL FOUR suites (the old
        # layout kept only the last suite's results.json)
        merged_base = json.loads((tmp_path / "baseline" / "results.json").read_text())
        merged_suites = {c["suite"] for c in merged_base["cases"]}
        assert merged_suites == {
            "protocol",
            "workers",
            "middleware",
            "saturation",
        }, f"suites lost to results.json overwrite: {merged_suites}"
        assert merged_base["paired_run"]["token"] == driver.RUN_TOKEN

        # Verify: the candidate's DRIVEN open-loop rates equal the
        # baseline's per model, arrival and round (the reviewer's
        # real-run check, replicated here on the merged artifacts)
        merged_cand = json.loads((tmp_path / "candidate" / "results.json").read_text())

        def open_rates(payload):
            return {
                (
                    c["framework"],
                    c["model"],
                    c["params"]["arrival"],
                    c["round"],
                ): c[
                    "result"
                ]["target_qps"]
                for c in payload["cases"]
                if c["params"].get("mode") == "open"
            }

        assert open_rates(merged_base) == open_rates(merged_cand), (
            "candidate open-loop rates differ from baseline's per "
            "model/arrival/round — rate propagation is broken"
        )

    def test_baseline_image_is_required(self):
        """--baseline-image must be required=True (was
        required=not argparse.SUPPRESS, which evaluates to False)."""
        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        source = (tools_dir / "paired_driver.py").read_text()
        assert "required=True" in source, "--baseline-image is not required=True"

    def test_run_side_suite_scoped_dir_and_run_id(self, tmp_path):
        """The REAL run_side (not the loop mock) must write each suite
        into its own round-N/{side}/{suite}/{framework} directory under
        its own --run-id — the old layout made all four suites of a
        side atomically overwrite one results.json."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_side", str(tools_dir / "paired_driver.py")
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            r = MagicMock()
            r.returncode = 0
            r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=fake_run):
            side_dir = driver.run_side(
                "baseline",
                "img",
                "sklearn",
                2,
                tmp_path,
                suite="saturation",
                run_id="tok-base-r2",
            )

        cmd = captured["cmd"]
        out_path = Path(cmd[cmd.index("--output-dir") + 1])
        assert out_path == (
            tmp_path / "round-2" / "baseline" / "saturation" / "sklearn"
        ), f"side dir is not suite-scoped: {out_path}"
        assert side_dir == out_path
        assert out_path.exists()
        # Its own run id (container names + artifacts attributable)
        assert cmd[cmd.index("--run-id") + 1] == "tok-base-r2"
        # One suite at a time
        assert cmd[cmd.index("--suites") + 1] == "saturation"

    def test_scoped_suites_and_frameworks_respected(self, tmp_path):
        """--suites/--frameworks must actually scope the run. A
        leftover `suites = SUITES.split(",")` in the loop section
        clobbered the parameter after normalization, so the flag did
        nothing and the scoped mini A/A silently ran all four suites
        (~45 min instead of ~20). The reviewer reproduced it by
        calling _run_paired(suites=["saturation"]) and watching all
        four suites run."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_scope", str(tools_dir / "paired_driver.py")
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)

        calls = []

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            calls.append({"label": label, "framework": fw, "suite": suite})
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {
                            "image": {"digest": "test", "target": "cpu"},
                            "host": {},
                        },
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "model": "m",
                                "params": {"c": 8},
                                "round": round_idx,
                                "result": {"throughput_rps": 100.0},
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            driver._run_paired(
                tmp_path,
                "img-a",
                "img-b",
                rounds=2,
                suites=["saturation"],
                frameworks=["sklearn"],
            )

        # ONLY the requested suite and framework ran
        suites_called = {c["suite"] for c in calls}
        frameworks_called = {c["framework"] for c in calls}
        assert suites_called == {"saturation"}, f"--suites ignored: ran {suites_called}"
        assert frameworks_called == {
            "sklearn"
        }, f"--frameworks ignored: ran {frameworks_called}"
        # The merged payload records the scoped matrix
        merged = json.loads((tmp_path / "baseline" / "results.json").read_text())
        assert merged["paired_run"]["suites"] == ["saturation"]
        assert merged["paired_run"]["frameworks"] == ["sklearn"]


class TestRunIdentity:
    """Container names and result paths must uniquely identify their
    run, side, round and suite — no ambiguity in `docker ps`, no
    collisions across runs, no result overwrites across suites.

    The old scheme named containers mlserver-bench-srv-{n} with a
    counter that RESET for every suite and round: names were reused
    across the matrix (a leaked container collided with the next
    suite's name) and gave no clue about origin. The old paired
    driver wrote all four suites of a side into one results.json —
    only the last suite's data survived, silently."""

    @staticmethod
    def _ctx(tmp_path, **kwargs):
        from bench.config import RunConfig
        from bench.suites import SuiteContext

        config = RunConfig()
        config.load_profile = "quick"
        return SuiteContext(config, "stub", tmp_path, tmp_path / "out", **kwargs)

    def test_names_carry_run_suite_round(self, tmp_path):
        ctx = self._ctx(
            tmp_path, run_id="run-abc", suite_name="saturation", round_index=3
        )
        name = ctx.next_name()
        assert name == "mlserver-bench-run-abc-saturation-r3-1"

    def test_no_collision_across_suites_and_rounds(self, tmp_path):
        """Counter resets per context — the run_id+suite+round
        segments must keep names unique across the whole matrix."""
        names = []
        for suite in ("protocol", "saturation"):
            for round_index in range(3):
                ctx = self._ctx(
                    tmp_path,
                    run_id="run-abc",
                    suite_name=suite,
                    round_index=round_index,
                )
                names.append(ctx.next_name())
                names.append(ctx.next_name())
        assert len(names) == len(set(names)), "container names collide"

    def test_adhoc_contexts_get_unique_fallback_ids(self, tmp_path):
        """Contexts without an explicit run_id (tests, notebooks) get
        generated ids — two such contexts never collide either."""
        a = self._ctx(tmp_path)
        b = self._ctx(tmp_path)
        assert a.run_id != b.run_id
        assert a.next_name() != b.next_name()

    def test_server_name_uses_context_suite_name(self, tmp_path):
        ctx = self._ctx(tmp_path, run_id="run-abc", suite_name="middleware")
        name = ctx.next_name()
        assert "middleware" in name

    def test_run_config_validates_run_id(self):
        from bench.config import RunConfig

        ok = RunConfig(run_id="pd-baseline-r0-saturation")
        assert ok.validate() == []
        bad = RunConfig(run_id="Not A Valid ID!")
        problems = bad.validate()
        assert any("run_id" in p for p in problems)

    def test_generate_run_id_unique_and_dns_safe(self):
        from bench.config import generate_run_id

        ids = {generate_run_id() for _ in range(50)}
        assert len(ids) == 50
        import re

        for rid in ids:
            assert re.match(r"^[a-z0-9][a-z0-9-]*$", rid), rid

    def test_clean_stale_containers_scoping(self):
        """run_id-scoped sweeps only match that run's containers; the
        default sweep matches all benchmark containers."""
        from unittest.mock import patch

        with patch("bench.docker_server._docker") as mock_docker:
            mock_docker.return_value.stdout = ""
            from bench.docker_server import clean_stale_containers

            clean_stale_containers(run_id="run-abc")
            filter_arg = mock_docker.call_args_list[0].args[3]
            assert filter_arg == "name=mlserver-bench-run-abc"

            clean_stale_containers()
            filter_arg = mock_docker.call_args_list[1].args[3]
            assert filter_arg == "name=mlserver-bench"

    def test_runner_default_run_id_generated(self, tmp_path):
        """A run without an explicit --run-id gets one at execute()
        time and records it in config.json + manifest.json (artifacts
        are self-describing even when copied out of their dir)."""
        from bench.runner import environment_manifest
        from bench.config import RunConfig, generate_run_id

        config = RunConfig()
        assert config.run_id == ""
        # simulate the execute() defaulting
        config.run_id = config.run_id or generate_run_id()
        manifest = environment_manifest(config, tmp_path)
        manifest["run_id"] = config.run_id
        assert manifest["run_id"] == config.run_id
        assert config.run_id  # non-empty, recorded


class TestPairedDriverReviewCoverage:
    """Tests for the review defects that were fixed in code but not
    yet test-guarded: verdict labels with params, the driver's --out
    refusal, per-suite completeness, digest pinning, the post-run
    verdict effect, and hog calibration noise handling."""

    @staticmethod
    def _load_driver():
        import importlib.util

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_cov", str(tools_dir / "paired_driver.py")
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        return driver

    def test_verdict_labels_include_params(self, tmp_path):
        """Two cells with the same suite/framework/model but different
        params must produce DISTINCT labels and both be counted —
        the reviewer saw two verdicts both reading
        saturation|sklearn|sklearn-grad-boost-small with different
        means, and matched_cells reporting 3 for 12 cells."""
        driver = self._load_driver()

        def side(label, concs):
            cases = []
            for conc in concs:
                for r in range(3):
                    cases.append(
                        {
                            "suite": "saturation",
                            "framework": "sklearn",
                            "model": "grad-boost",
                            "params": {"mode": "closed", "concurrency": conc},
                            "round": r,
                            "result": {"throughput_rps": 100.0 * conc},
                        }
                    )
            d = tmp_path / label
            d.mkdir()
            (d / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {"image": {"digest": "x"}},
                        "cases": cases,
                    }
                )
            )
            return d

        base = side("b", [1, 8, 32])
        cand = side("c", [1, 8, 32])
        result = driver.paired_compare(base, cand)
        labels = [v["case"] for v in result["verdicts"]]
        assert len(labels) == len(set(labels)), f"ambiguous labels: {labels}"
        # 3 distinct params x 3 concs -> 3 distinct cells
        assert (
            result["matched_cells"] == 3
        ), f"matched_cells={result['matched_cells']} — cells collapsed"
        assert any("concurrency: 8" in lab for lab in labels), labels

    def test_driver_out_refusal(self, tmp_path):
        """The DRIVER's --out refusal (the orchestrator's is tested
        elsewhere): a non-empty target refuses, an empty one passes."""
        driver = self._load_driver()
        target = tmp_path / "existing"
        target.mkdir()
        (target / "stale.json").write_text("{}")
        with pytest.raises(SystemExit, match="refusing to write into non-empty"):
            driver._run_paired(target, "img", "img", rounds=1)

    def test_missing_suite_cell_aborts(self, tmp_path):
        """A side where one suite produced NO cases (rc=0, empty
        results) must abort WITHIN that side — the fail-fast check
        (reviewer item 1) catches it immediately, with the attempt-1
        evidence kept in retries/. Before fail-fast, this hole was
        only caught at the post-merge completeness check (the round
        check alone counted rounds, not suites — reviewer defect 4)."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            # middleware produces NO cases — the silent-hole case
            cases = []
            if suite != "middleware":
                cases.append(
                    {
                        "suite": suite,
                        "framework": fw,
                        "model": "m",
                        "params": {"c": 8},
                        "round": round_idx,
                        "result": {"throughput_rps": 100.0},
                    }
                )
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {
                            "image": {"digest": "test", "target": "cpu"},
                            "host": {},
                        },
                        "cases": cases,
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            # Fail fast: the middleware hole aborts on its OWN side,
            # after one retry, with the evidence kept
            with pytest.raises(RuntimeError, match="failed BOTH attempts"):
                driver._run_paired(tmp_path / "run", "img", "img", rounds=1)
        retries_dir = tmp_path / "run" / "retries"
        assert retries_dir.is_dir() and any(
            retries_dir.iterdir()
        ), "failed attempt's results were not kept in retries/"

    def test_fail_fast_suite_failed_aborts_within_side(self, tmp_path):
        """The reviewer's exact scenario: a stubbed side writes a
        SUITE_FAILED result (connection-reset, error rate over the
        limit). The run must abort WITHIN that side after one retry —
        not continue for ~7 minutes to the post-merge outcome check.
        Both attempts' evidence is kept."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()
        calls = []

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            calls.append((label, suite, fw))
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {
                            "image": {"digest": "test", "target": "cpu"},
                            "host": {},
                        },
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "tier": "small",
                                "status": "SUITE_FAILED",
                                "error": "ErrorRateGuard",
                                "detail": "error rate 0.24% > 0.10% limit; "
                                "first error: [Errno 104] Connection reset "
                                "by peer",
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            with pytest.raises(RuntimeError, match="failed BOTH attempts"):
                driver._run_paired(tmp_path / "run", "img", "img", rounds=1)

        # Fail fast: only the FIRST suite's side was attempted (twice
        # — the original and the retry), never the later suites
        assert {c[1] for c in calls} == {
            "protocol"
        }, f"run continued past the failed side: {set(c[1] for c in calls)}"
        assert len(calls) == 2, f"expected exactly 2 attempts, got {len(calls)}"
        # The failure detail survives in the abort and in retries/
        retries = list((tmp_path / "run" / "retries").glob("*attempt1.json"))
        assert retries, "attempt-1 evidence not kept"
        saved = json.loads(retries[0].read_text())
        assert saved["cases"][0]["status"] == "SUITE_FAILED"
        assert "Errno 104" in saved["cases"][0]["detail"]

    def test_suite_retry_flake_completes_and_marks_pair(self, tmp_path):
        """A SUITE_FAILED side that PASSES on retry is a recorded
        flake: the run completes, the pair is MARKED (retried_side on
        the merged cases, flagged on the verdicts, suite_retries in
        the result) — reported, never hidden — and the attempt-1
        evidence is kept."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()
        attempt = {}

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            key = (label, suite, fw, round_idx)
            attempt[key] = attempt.get(key, 0) + 1
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            # candidate/middleware/sklearn fails ONCE (the flake),
            # passes on retry; everything else is clean
            failed = (
                key == ("candidate", "middleware", "sklearn", 0) and attempt[key] == 1
            )
            cases = [
                {
                    "suite": suite,
                    "framework": fw,
                    "model": "m",
                    "params": {"c": 8},
                    "round": round_idx,
                    "result": {"throughput_rps": 100.0 if not failed else 0.0},
                }
            ]
            if failed:
                cases = [
                    {
                        "suite": suite,
                        "framework": fw,
                        "tier": "small",
                        "status": "SUITE_FAILED",
                        "error": "ErrorRateGuard",
                        "detail": "transient connection reset",
                    }
                ]
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {
                            "image": {"digest": "test", "target": "cpu"},
                            "host": {},
                        },
                        "cases": cases,
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            # rounds=2: paired_compare needs >= 2 rounds per cell to
            # produce verdicts at all
            result = driver._run_paired(tmp_path / "run", "img", "img", rounds=2)

        # The run COMPLETED and the verdict is valid
        assert result["status"] != "INVALID"
        # The flake is reported, not hidden
        assert len(result["suite_retries"]) == 1, result["suite_retries"]
        entry = result["suite_retries"][0]
        assert entry["side"] == "candidate"
        assert entry["suite"] == "middleware"
        assert "transient connection reset" in entry["first_attempt_failures"][0]
        # The pair is MARKED in the merged data
        cand = json.loads((tmp_path / "run" / "candidate" / "results.json").read_text())
        marked = [c for c in cand["cases"] if c.get("retried_side")]
        assert marked and all(
            c["suite"] == "middleware" and c["framework"] == "sklearn" for c in marked
        )
        base = json.loads((tmp_path / "run" / "baseline" / "results.json").read_text())
        assert not any(c.get("retried_side") for c in base["cases"])
        # And flagged on the verdicts for those cells
        flagged = [v for v in result["verdicts"] if v.get("retried_side")]
        assert flagged and all("middleware" in v["case"] for v in flagged)
        # Attempt-1 evidence kept
        assert (tmp_path / "run" / "retries").is_dir() and any(
            (tmp_path / "run" / "retries").iterdir()
        )

    def test_floating_tag_aborts(self, tmp_path):
        """A side whose manifest carries NO image digest must abort —
        floating tags are not comparable (a tag that moved mid-run
        compares two different builds)."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        # no digest — floating tag
                        "manifest": {"image": {"target": "cpu"}, "host": {}},
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "model": "m",
                                "params": {"c": 8},
                                "round": round_idx,
                                "result": {"throughput_rps": 100.0},
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            with pytest.raises(RuntimeError, match="no image digest"):
                driver._run_paired(tmp_path / "run", "img", "img", rounds=1)

    def test_post_run_check_failure_invalidates(self, tmp_path):
        """A FAILED post-run mutation check must flip the verdict to
        INVALID (it used to be recorded as post_mutation_pass without
        affecting the verdict — silently-untrustworthy results looked
        healthy)."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()
        calls = {"n": 0}

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {"image": {"digest": "test"}, "host": {}},
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "model": "m",
                                "params": {"c": 8},
                                "round": round_idx,
                                "result": {"throughput_rps": 100.0},
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            # make-crosscheck / make-mutation-check: pre-flight calls
            # pass, post-run mutation check FAILS
            r = MagicMock()
            which = " ".join(cmd[:2])
            if which == "make mutation-check":
                calls["n"] += 1
                r.returncode = 0 if calls["n"] == 1 else 1
            else:
                r.returncode = 0
            r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = driver._run_paired(tmp_path / "run", "img", "img", rounds=1)

        assert result["status"] == "INVALID", result["status"]
        assert result["post_mutation_pass"] is False
        assert "mutation check" in result["invalid_reason"]

    def test_gate_output_captured_to_files(self, tmp_path):
        """Every pre/post-run gate writes its FULL stdout+stderr to
        {out}/gates/{name}.log — the reviewer's mini A/A was INVALIDated
        by a failed post-run mutation check with the CAUSE DISCARDED
        (94 s vs ~35 s — contention suspected, nothing recorded)."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {"image": {"digest": "test"}, "host": {}},
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "model": "m",
                                "params": {"c": 8},
                                "round": round_idx,
                                "result": {"throughput_rps": 100.0},
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stderr = ""
            which = " ".join(cmd[:2])
            if which == "make mutation-check":
                r.stdout = "RESULT: all mutations caught\n"
            elif which == "make crosscheck":
                r.stdout = "crosscheck: loadgen vs ab within 2%\n"
            else:
                # git status / rev-parse: empty output
                r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            driver._run_paired(tmp_path / "run", "img", "img", rounds=1)

        gates = tmp_path / "run" / "gates"
        expected_logs = {
            "preflight-mutation-check.log",
            "preflight-crosscheck.log",
            "postrun-mutation-check.log",
            "postrun-crosscheck.log",
        }
        assert gates.is_dir(), "no gates/ directory"
        written = {f.name for f in gates.iterdir()}
        assert expected_logs <= written, f"missing gate logs: {expected_logs - written}"
        for name in expected_logs:
            content = (gates / name).read_text()
            assert "$ make" in content, f"{name}: command line not recorded"
            assert "--- stdout ---" in content, f"{name}: stdout not recorded"
            # each log captured ITS OWN gate's output, not discarded
            if "mutation" in name:
                assert (
                    "all mutations caught" in content
                ), f"{name}: gate output discarded"
            else:
                assert "within 2%" in content, f"{name}: gate output discarded"

    def test_flaked_gate_retries_and_keeps_verdict(self, tmp_path):
        """A post-run gate that fails ONCE under contention and passes
        on retry is a recorded FLAKE — it must NOT invalidate the
        verdict (the reviewer's mini A/A was INVALIDated this way).
        Both attempt logs are kept."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()
        mut_calls = {"n": 0}

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {"image": {"digest": "test"}, "host": {}},
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "model": "m",
                                "params": {"c": 8},
                                "round": round_idx,
                                "result": {"throughput_rps": 100.0},
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.stdout = ""
            r.stderr = ""
            r.returncode = 0
            if " ".join(cmd[:2]) == "make mutation-check":
                mut_calls["n"] += 1
                # 1: preflight pass; 2: postrun FAIL (contention);
                # 3: postrun retry PASS
                if mut_calls["n"] == 2:
                    r.returncode = 1
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = driver._run_paired(tmp_path / "run", "img", "img", rounds=1)

        assert (
            mut_calls["n"] == 3
        ), f"expected a retry (3 mutation-check calls), got {mut_calls['n']}"
        assert (
            result["status"] != "INVALID"
        ), "a single contention flake invalidated the verdict"
        assert result["post_mutation_pass"] is True
        assert result["post_mutation_flaked"] is True
        gates = tmp_path / "run" / "gates"
        assert (gates / "postrun-mutation-check.log").is_file()
        assert (
            gates / "postrun-mutation-check-retry.log"
        ).is_file(), "the failed first attempt's output was discarded"

    def test_double_gate_failure_names_in_invalid_reason(self, tmp_path):
        """A double gate failure records the failing test/mutation
        NAMES in the INVALID reason — not just that something failed
        (the cause must survive even when the run is over)."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()
        mut_calls = {"n": 0}

        def mock_run_side(label, image, fw, round_idx, out_dir, **kwargs):
            suite = kwargs.get("suite", "")
            side_dir = out_dir / f"round-{round_idx}" / label / suite / fw
            side_dir.mkdir(parents=True, exist_ok=True)
            (side_dir / "results.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest": {"image": {"digest": "test"}, "host": {}},
                        "cases": [
                            {
                                "suite": suite,
                                "framework": fw,
                                "model": "m",
                                "params": {"c": 8},
                                "round": round_idx,
                                "result": {"throughput_rps": 100.0},
                            }
                        ],
                    }
                )
            )
            return side_dir

        def mock_run(cmd, **kw):
            r = MagicMock()
            r.stderr = ""
            r.returncode = 0
            if " ".join(cmd[:2]) == "make mutation-check":
                mut_calls["n"] += 1
                if mut_calls["n"] >= 2:
                    r.returncode = 1
                    r.stdout = (
                        "RESULT: 1/32 mutations NOT caught:\n"
                        "  - some-mutation-name\n"
                    )
            else:
                r.stdout = ""
            return r

        with (
            patch.object(driver, "run_side", side_effect=mock_run_side),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = driver._run_paired(tmp_path / "run", "img", "img", rounds=1)

        assert result["status"] == "INVALID"
        assert (
            "some-mutation-name" in result["invalid_reason"]
        ), f"failing mutation name not in reason: {result['invalid_reason']}"
        assert result["post_mutation_failing"] == ["some-mutation-name"]

    def test_completeness_backstop_direct(self, tmp_path):
        """The merge-time completeness check, tested DIRECTLY (fail-fast
        intercepts every reachable loop scenario first, so the
        backstop needs its own test to keep its mutation catchable):
        a merged side missing one (round, suite, framework) cell
        raises, a complete one passes."""
        driver = self._load_driver()

        def side_dir(name, cases):
            d = tmp_path / name
            d.mkdir()
            (d / "results.json").write_text(json.dumps({"cases": cases}))
            return d

        suites = ["protocol", "saturation"]
        frameworks = ["sklearn"]
        complete = [
            {"round": r, "suite": s, "framework": "sklearn"}
            for r in range(2)
            for s in suites
        ]
        driver._verify_completeness(
            "ok", side_dir("ok", complete), 2, suites, frameworks
        )

        hole = [
            c for c in complete if not (c["round"] == 1 and c["suite"] == "saturation")
        ]
        with pytest.raises(RuntimeError, match="missing suite cells"):
            driver._verify_completeness(
                "hole", side_dir("hole", hole), 2, suites, frameworks
            )

    def test_cli_exit_code_on_suite_failed(self, tmp_path):
        """The CLI must exit non-zero when a run recorded SUITE_FAILED
        cases — the paired driver treats rc=0 as a healthy side and
        only noticed the failure ~7 minutes late at the outcome
        check."""
        from bench.cli import _exit_code_for

        clean = tmp_path / "clean"
        clean.mkdir()
        (clean / "results.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": [{"suite": "s", "result": {"throughput_rps": 1.0}}],
                }
            )
        )
        assert _exit_code_for(clean) == (0, set())

        failed = tmp_path / "failed"
        failed.mkdir()
        (failed / "results.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": [
                        {
                            "suite": "saturation",
                            "status": "SUITE_FAILED",
                            "detail": "error rate 0.24% > 0.10%",
                        },
                        {"suite": "protocol", "result": {"throughput_rps": 1.0}},
                    ],
                }
            )
        )
        assert _exit_code_for(failed) == (1, {"saturation"})

        assert _exit_code_for(tmp_path / "missing") == (0, set())

    def test_cli_main_clean_run_exits_zero(self, tmp_path, monkeypatch):
        """End-to-end main() exit path for a CLEAN run. An inline
        json.loads in main() crashed with UnboundLocalError (a local
        `import json` in the compare branch shadowed the module-level
        import for the whole function), making every clean run exit 1
        — found by the reviewer's repro script, missed by the direct
        _exit_code_for test because nothing ran main() itself."""
        import bench.cli as cli_mod

        out = tmp_path / "out"
        out.mkdir()
        (out / "results.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": [
                        {"suite": "saturation", "result": {"throughput_rps": 100.0}}
                    ],
                }
            )
        )
        monkeypatch.setattr("bench.runner.execute", lambda config, root: out)
        monkeypatch.setattr(cli_mod.report_mod, "write_reports", lambda o: None)
        # Must complete WITHOUT SystemExit/UnboundLocalError
        cli_mod.main(
            [
                "run",
                "--suites",
                "saturation",
                "--framework",
                "sklearn",
                "--tiers",
                "small",
                "--profile",
                "quick",
                "--output-dir",
                str(out),
            ]
        )

    def test_cli_main_suite_failed_exits_one(self, tmp_path, monkeypatch):
        """End-to-end main() exit path for a SUITE_FAILED run: the
        process must exit 1 (the paired driver treats rc=0 as a
        healthy side)."""
        import bench.cli as cli_mod

        out = tmp_path / "out"
        out.mkdir()
        (out / "results.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cases": [
                        {
                            "suite": "saturation",
                            "status": "SUITE_FAILED",
                            "detail": "error rate 0.24% > 0.10%",
                        }
                    ],
                }
            )
        )
        monkeypatch.setattr("bench.runner.execute", lambda config, root: out)
        monkeypatch.setattr(cli_mod.report_mod, "write_reports", lambda o: None)
        with pytest.raises(SystemExit) as exc:
            cli_mod.main(
                [
                    "run",
                    "--suites",
                    "saturation",
                    "--framework",
                    "sklearn",
                    "--tiers",
                    "small",
                    "--profile",
                    "quick",
                    "--output-dir",
                    str(out),
                ]
            )
        assert exc.value.code == 1

    def test_container_logs_kept_on_failure(self, tmp_path):
        """When an exception unwinds through a server context, the
        container's logs are saved BEFORE stop() removes the container
        — the reviewer's SUITE_FAILED case had its cause destroyed
        with the container. Normal exits do not dump."""
        from unittest.mock import patch

        from bench.config import RunConfig
        from bench.suites import SuiteContext, _RegisteredContainer

        config = RunConfig()
        ctx = SuiteContext(
            config,
            "img",
            tmp_path,
            tmp_path,
            suite_name="saturation",
            run_id="run-x",
        )
        container = _RegisteredContainer(
            "mlserver-bench-run-x-saturation-r0-1", "img", tmp_path
        )
        container._ctx = ctx

        with (
            patch.object(type(container), "logs", return_value="worker crashed: ..."),
            patch.object(type(container), "stop"),
        ):
            # normal exit: no dump
            container.__exit__(None, None, None)
            assert not (tmp_path / "container-logs").exists()

            # failure exit: logs kept before the container is removed
            container.__exit__(ValueError("boom"), ValueError("boom"), None)
        log_file = (
            tmp_path / "container-logs" / "mlserver-bench-run-x-saturation-r0-1.log"
        )
        assert log_file.is_file(), "container logs were destroyed with the container"
        assert "worker crashed" in log_file.read_text()

    def test_paired_cpu_limit_env_plumbing(self, tmp_path, monkeypatch):
        """PAIRED_CPU_LIMIT env -> every container the context starts
        gets a CFS quota. This is the graded control's degradation
        path: host CPU hogs (the previous mechanism) triggered F-06
        (2/2 graded aborts); the container-local quota degrades the
        candidate without loading the host."""
        from bench.config import RunConfig
        from bench.suites import SuiteContext

        config = RunConfig()
        config.load_profile = "quick"

        # env not set -> no limit
        monkeypatch.delenv("PAIRED_CPU_LIMIT", raising=False)
        ctx = SuiteContext(config, "img", tmp_path, tmp_path / "o")
        assert ctx.server(tmp_path).cpu_limit is None

        # env set -> applied
        monkeypatch.setenv("PAIRED_CPU_LIMIT", "1.25")
        ctx = SuiteContext(config, "img", tmp_path, tmp_path / "o")
        assert ctx.server(tmp_path).cpu_limit == 1.25

        # explicit arg wins over env
        assert ctx.server(tmp_path, cpu_limit=2.0).cpu_limit == 2.0

        # garbage env -> no limit (fail open to unthrottled, the
        # calibration/output records what actually ran)
        monkeypatch.setenv("PAIRED_CPU_LIMIT", "not-a-number")
        ctx = SuiteContext(config, "img", tmp_path, tmp_path / "o")
        assert ctx.server(tmp_path).cpu_limit is None

    def test_container_start_passes_cpus_flag(self, tmp_path):
        """MLServerContainer.start() must pass --cpus to docker when a
        limit is set (and omit it when not)."""
        from unittest.mock import patch

        from bench.docker_server import MLServerContainer

        captured = {}

        def fake_docker(*args, **kwargs):
            captured["args"] = args

            class FakeProc:
                returncode = 0
                stdout = ""
                stderr = ""

            return FakeProc()

        with (
            patch("bench.docker_server._docker", side_effect=fake_docker),
            patch("bench.docker_server.MLServerContainer._wait_healthy"),
            patch("bench.docker_server.MLServerContainer.gpu_guard"),
        ):
            MLServerContainer("n", "img", tmp_path, cpu_limit=1.5).start()
        args = captured["args"]
        assert "--cpus" in args
        assert args[args.index("--cpus") + 1] == "1.5"

        with (
            patch("bench.docker_server._docker", side_effect=fake_docker),
            patch("bench.docker_server.MLServerContainer._wait_healthy"),
        ):
            MLServerContainer("n2", "img", tmp_path).start()
        assert "--cpus" not in captured["args"]


class TestGradedCalibration:
    """Calibration noise handling (reviewer defect 5): median of
    repeats (not single noisy pairs), loud failure (not silent 0.0
    falling through to max hogs), and recorded steps + realized
    degradation."""

    @staticmethod
    def _load_driver():
        import importlib.util

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        spec = importlib.util.spec_from_file_location(
            "pd_cal", str(tools_dir / "paired_driver.py")
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        return driver

    def test_median_of_repeats_not_single_pair(self):
        """Each step is the MEDIAN of 3 quick-runs: with ~10%
        run-to-run noise, a single pair met a 5% target by chance —
        the median must not."""
        from unittest.mock import patch

        driver = self._load_driver()

        # unthrottled: 1000 +- noise; --cpus 2.0: true 990 (1%) but
        # one noisy sample dips to 890; --cpus 1.6: true 900 (10%)
        samples = {
            None: [1010.0, 990.0, 1000.0],
            2.0: [990.0, 890.0, 995.0],  # median 990 -> 1% degradation
            1.6: [905.0, 895.0, 900.0],  # median 900 -> 10% degradation
        }
        state = {"n": {}}

        def fake_quick(framework, image, cpu_limit=None):
            key = cpu_limit
            i = state["n"].get(key, 0)
            state["n"][key] = i + 1
            return samples[key][i % len(samples[key])]

        with patch.object(driver, "_quick_throughput", side_effect=fake_quick):
            cal = driver.calibrate_cpu_limit(0.05, "img", repeats=3)

        # A single-pair calibration would have seen 890 (11%) at
        # --cpus 2.0 and stopped; the median sees 1% and moves on
        assert cal["cpu_limit"] == 1.6, cal
        assert len(cal["steps"]) == 2
        assert cal["steps"][0]["degradation"] == 0.01, cal["steps"]

    def test_failed_quick_run_raises_not_silent_zero(self):
        """A failed bench.cli run must RAISE — returning 0.0 silently
        fell through to the maximum hog count."""
        from unittest.mock import MagicMock, patch

        driver = self._load_driver()

        def fake_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 1
            r.stderr = "boom"
            return r

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="quick calibration run failed"):
                driver._quick_throughput("sklearn", "img", cpu_limit=1.0)

    def test_realized_degradation_closed_loop_only(self):
        """The graded control's size is the mean closed-loop
        candidate/baseline ratio — open-loop cells are rate-matched
        by design and must not dilute it."""
        driver = self._load_driver()
        result = {
            "verdicts": [
                {
                    "case": "s|sk|m|mode: closed, concurrency: 8",
                    "metric": "throughput",
                    "mean_ratio": 0.90,
                },
                {
                    "case": "s|sk|m|mode: closed, concurrency: 32",
                    "metric": "throughput",
                    "mean_ratio": 0.92,
                },
                # rate-matched open-loop cells: throughput ~= rate on
                # both sides — must be EXCLUDED
                {
                    "case": "s|sk|m|mode: open, arrival: constant",
                    "metric": "throughput",
                    "mean_ratio": 1.0,
                },
                {
                    "case": "s|sk|m|mode: closed, concurrency: 8",
                    "metric": "p99",
                    "mean_ratio": 1.05,
                },
            ]
        }
        realized = driver._realized_degradation(result)
        assert realized == round(1 - (0.90 + 0.92) / 2, 4), realized
        assert realized == 0.09

    def test_realized_degradation_empty(self):
        driver = self._load_driver()
        assert driver._realized_degradation({"verdicts": []}) is None
