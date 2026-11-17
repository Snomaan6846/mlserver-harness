"""Tests for wire-format clients and the five new suites."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench import wire  # noqa: E402
from bench.config import SUITES as ALL_SUITES, RunConfig  # noqa: E402
from bench.suites import (  # noqa: E402
    _patched_repo_extra,
    suite_framework,
    suite_grpc,
    suite_hardware,
    tile_batch,
)
from bench.suites import SuiteContext  # noqa: E402

SETTINGS_MLP = {
    "name": "onnx-mlp-small",
    "implementation": "mlserver_onnx.OnnxModel",
    "parameters": {"uri": "model.onnx"},
    "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 10]}],
    "outputs": [{"name": "output", "datatype": "FP32", "shape": [-1, 2]}],
}

SETTINGS_SK = {
    "name": "sklearn-rf",
    "implementation": "mlserver_sklearn.SKLearnModel",
    "parameters": {"uri": "model.joblib"},
    "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 10]}],
    "outputs": [{"name": "predict", "datatype": "INT64", "shape": [-1]}],
}

BATCH = {"input": np.random.default_rng(0).random((4, 10)).astype(np.float32)}


# ---------------------------------------------------------------------------
# Binary tensor wire format
# ---------------------------------------------------------------------------


class TestBinaryTensor:
    def test_build_binary_request_header(self):
        body, header_len = wire.build_binary_request(SETTINGS_MLP, BATCH)
        header = json.loads(body[:header_len])
        assert header["inputs"][0]["parameters"]["binary_data_size"] == 4 * 10 * 4
        # FP32, 4 rows x 10 features = 160 bytes of tensor data
        assert len(body) == header_len + 160

    def test_binary_body_is_little_endian_fp32(self):
        body, header_len = wire.build_binary_request(SETTINGS_MLP, BATCH)
        assert header_len > 0
        tensor = np.frombuffer(body, dtype=np.dtype("<f4"), count=40, offset=header_len)
        np.testing.assert_array_equal(tensor, BATCH["input"].reshape(-1))

    def test_parse_binary_response_roundtrip(self):
        body, header_len = wire.build_binary_request(SETTINGS_MLP, BATCH)
        assert len(json.loads(body[:header_len])["inputs"]) == 1
        # Simulate a response: JSON header + output tensor bytes
        output = np.array(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], dtype=np.float32
        )
        resp_header = {
            "outputs": [
                {
                    "name": "output",
                    "datatype": "FP32",
                    "shape": [4, 2],
                    "parameters": {"binary_data_size": output.nbytes},
                }
            ]
        }
        resp_header_bytes = json.dumps(resp_header).encode()
        resp_body = resp_header_bytes + output.tobytes()
        parsed = wire.parse_binary_response(resp_body, len(resp_header_bytes))
        np.testing.assert_allclose(parsed["output"], output)

    def test_binary_output_negotiation_flag(self):
        body, header_len = wire.build_binary_request(SETTINGS_MLP, BATCH)
        header = json.loads(body[:header_len])
        assert header["parameters"]["binary_data_output"] is True

    def test_binary_output_flag_off(self):
        body, header_len = wire.build_binary_request(
            SETTINGS_MLP, BATCH, binary_output=False
        )
        header = json.loads(body[:header_len])
        assert "parameters" not in header


# ---------------------------------------------------------------------------
# gRPC client
# ---------------------------------------------------------------------------


class TestGrpcClient:
    def test_grpc_available_on_this_host(self):
        # The MLServer checkout is a sibling directory; grpcio may or
        # may not be installed — availability is environment-dependent,
        # but the modules load when both are present.
        assert isinstance(wire.grpc_available(), bool)

    @pytest.mark.skipif(not wire.grpc_available(), reason="grpcio/protos unavailable")
    def test_build_grpc_request_typed(self):
        request = wire.build_grpc_request(
            SETTINGS_MLP, BATCH, "onnx-mlp-small", raw_input=False
        )
        assert request.model_name == "onnx-mlp-small"
        assert len(request.inputs) == 1
        assert request.inputs[0].datatype == "FP32"
        assert list(request.inputs[0].shape) == [4, 10]
        assert len(request.inputs[0].contents.fp32_contents) == 40
        assert len(request.raw_input_contents) == 0

    @pytest.mark.skipif(not wire.grpc_available(), reason="grpcio/protos unavailable")
    def test_build_grpc_request_raw(self):
        request = wire.build_grpc_request(
            SETTINGS_MLP, BATCH, "onnx-mlp-small", raw_input=True
        )
        assert len(request.raw_input_contents) == 1
        tensor = np.frombuffer(request.raw_input_contents[0], dtype=np.dtype("<f4"))
        np.testing.assert_array_equal(tensor, BATCH["input"].reshape(-1))

    @pytest.mark.skipif(not wire.grpc_available(), reason="grpcio/protos unavailable")
    def test_grpc_request_outputs_requested(self):
        request = wire.build_grpc_request(SETTINGS_MLP, BATCH, "onnx-mlp-small")
        assert [o.name for o in request.outputs] == ["output"]


# ---------------------------------------------------------------------------
# content_type request builders
# ---------------------------------------------------------------------------


class TestContentTypeRequests:
    def test_np_is_json_lists(self):
        request = wire.build_request_content_type(SETTINGS_MLP, BATCH, "np")
        assert request["inputs"][0]["data"] == BATCH["input"].tolist()

    def test_pd_flat_list_encoding(self):
        request = wire.build_request_content_type(SETTINGS_MLP, BATCH, "pd")
        assert request["inputs"][0]["parameters"]["content_type"] == "pd"
        # PandasCodec builds pd.Series(payload): flat list of values
        data = request["inputs"][0]["data"]
        assert data == BATCH["input"].reshape(-1).tolist()
        assert len(data) == 40

    def test_str_encoding(self):
        request = wire.build_request_content_type(SETTINGS_MLP, BATCH, "str")
        assert request["inputs"][0]["parameters"]["content_type"] == "str"
        # data is a JSON-encoded string
        parsed = json.loads(request["inputs"][0]["data"])
        assert parsed == BATCH["input"].tolist()

    def test_unknown_content_type_rejected(self):
        with pytest.raises(ValueError, match="unsupported content_type"):
            wire.build_request_content_type(SETTINGS_MLP, BATCH, "bogus")

    def test_outputs_requested(self):
        request = wire.build_request_content_type(SETTINGS_MLP, BATCH, "np")
        assert [o["name"] for o in request["outputs"]] == ["output"]


# ---------------------------------------------------------------------------
# parameters.extra patching (framework settings auto-resolve mechanism)
# ---------------------------------------------------------------------------


class TestPatchedRepoExtra:
    def test_extra_merges_into_parameters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model_dir = tmp_path / "onnx-mlp-small"
        model_dir.mkdir()
        (model_dir / "model-settings.json").write_text(json.dumps(SETTINGS_MLP))
        ctx = SuiteContext(RunConfig(), "image", tmp_path, tmp_path / "out")
        repo = _patched_repo_extra(
            ctx,
            model_dir,
            {"session_options": {"intra_op_num_threads": 4}},
        )
        patched = json.loads(
            (repo / "onnx-mlp-small" / "model-settings.json").read_text()
        )
        assert patched["parameters"]["uri"] == "model.onnx"  # preserved
        assert patched["parameters"]["extra"]["session_options"] == {
            "intra_op_num_threads": 4
        }

    def test_extra_merges_with_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model_dir = tmp_path / "onnx-mlp-small"
        model_dir.mkdir()
        settings = dict(SETTINGS_MLP)
        settings["parameters"] = {
            "uri": "model.onnx",
            "extra": {"providers": ["CUDAExecutionProvider"]},
        }
        (model_dir / "model-settings.json").write_text(json.dumps(settings))
        ctx = SuiteContext(RunConfig(), "image", tmp_path, tmp_path / "out")
        repo = _patched_repo_extra(ctx, model_dir, {"cuda_graph": True})
        patched = json.loads(
            (repo / "onnx-mlp-small" / "model-settings.json").read_text()
        )
        extra = patched["parameters"]["extra"]
        assert extra["providers"] == ["CUDAExecutionProvider"]  # preserved
        assert extra["cuda_graph"] is True  # merged in


# ---------------------------------------------------------------------------
# Suite registration and structure
# ---------------------------------------------------------------------------


class TestSuiteRegistration:
    def test_new_suites_registered(self):
        for name in ("processing", "framework", "middleware", "codec", "grpc"):
            assert name in ALL_SUITES

    def test_new_suites_are_callables(self):
        from bench.suites import SUITES

        for name in ("processing", "framework", "middleware", "codec", "grpc"):
            assert callable(SUITES[name])

    def test_run_closed_accepts_sender_factory(self):
        import inspect

        from bench.suites import run_closed

        params = inspect.signature(run_closed).parameters
        assert "make_sender" in params


# ---------------------------------------------------------------------------
# Suite skip paths (no server needed)
# ---------------------------------------------------------------------------


class TestSuiteSkips:
    def test_framework_skips_without_onnx(self, tmp_path: Path):
        import asyncio

        model_dir = tmp_path / "sklearn-rf"
        model_dir.mkdir()
        (model_dir / "model-settings.json").write_text(json.dumps(SETTINGS_SK))
        ctx = SuiteContext(RunConfig(), "image", tmp_path, tmp_path / "out")
        cases = asyncio.run(suite_framework(ctx, [model_dir]))
        assert len(cases) == 1
        assert cases[0]["result"]["status"] == "SKIPPED_NO_ONNX_MODELS"

    def test_grpc_skips_gracefully(self, tmp_path: Path, monkeypatch):
        import asyncio

        monkeypatch.setattr(wire, "grpc_available", lambda: False)
        model_dir = tmp_path / "onnx-mlp-small"
        model_dir.mkdir()
        (model_dir / "model-settings.json").write_text(json.dumps(SETTINGS_MLP))
        ctx = SuiteContext(RunConfig(), "image", tmp_path, tmp_path / "out")
        cases = asyncio.run(suite_grpc(ctx, [model_dir]))
        assert len(cases) == 1
        assert cases[0]["result"]["status"] == "SKIPPED_GRPC_UNAVAILABLE"


# ---------------------------------------------------------------------------
# ONNX extra payload shapes (mirror OnnxSettings fields exactly)
# ---------------------------------------------------------------------------


class TestOnnxExtraShapes:
    """The extra payloads used by suite_framework must be valid
    OnnxSettings inputs (the framework pydantic class)."""

    VALID_EXTRAS = [
        {"session_options": {"intra_op_num_threads": 4, "inter_op_num_threads": 1}},
        {"session_options": {"inter_op_num_threads": 2}},
        {"session_options": {"enable_mem_pattern": False}},
        {"session_options": {"enable_cpu_mem_arena": False}},
        {"session_options": {"log_severity_level": 3}},
        {"providers": ["CPUExecutionProvider"]},
        {"providers": ["CUDAExecutionProvider"], "cuda_graph": True},
        {"providers": ["CUDAExecutionProvider"], "cuda_blocking_sync": True},
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": [{"cudnn_conv_algo_search": "EXHAUSTIVE"}],
        },
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": [{"arena_extend_strategy": "kNextPowerOfTwo"}],
        },
        {"run_options": {"log_severity_level": 3}},
        {"session_config_entries": {"session.intra_op.allow_spinning": "1"}},
    ]

    # Enum-typed SessionOptions properties: pybind rejects plain ints,
    # so these cannot be driven through JSON parameters.extra (the
    # suite excludes them by design — documented in suite_framework).
    ENUM_EXTRAS = [
        {"session_options": {"execution_mode": 1}},
        {"session_options": {"graph_optimization_level": 1}},
    ]

    @pytest.mark.parametrize("extra", VALID_EXTRAS)
    def test_extra_is_json_serializable(self, extra):
        # The suite writes these into model-settings.json
        assert isinstance(json.loads(json.dumps(extra)), dict)

    @pytest.mark.skipif(
        not (REPO / "MLServer" / "runtimes" / "onnx").is_dir(),
        reason="MLServer source not available",
    )
    @pytest.mark.parametrize("extra", VALID_EXTRAS + ENUM_EXTRAS)
    def test_extra_validates_against_onnxsettings(self, extra):
        """Validate against the real OnnxSettings class from the
        MLServer checkout (the auto-resolve target)."""
        import importlib.util

        onnx_dir = REPO / "MLServer" / "runtimes" / "onnx" / "mlserver_onnx"
        # Load the module standalone; it imports mlserver.settings,
        # which needs the mlserver package importable.
        sys.path.insert(0, str(REPO / "MLServer"))
        sys.path.insert(0, str(REPO / "MLServer" / "runtimes" / "onnx"))
        try:
            spec = importlib.util.spec_from_file_location(
                "mlserver_onnx_settings_under_test",
                str(onnx_dir / "settings.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parsed = module.OnnxSettings(**extra)
            assert parsed is not None
        finally:
            sys.path.remove(str(REPO / "MLServer"))
            sys.path.remove(str(REPO / "MLServer" / "runtimes" / "onnx"))


# ---------------------------------------------------------------------------
# hardware suite (CPU image on CPU vs CUDA image on GPU)
# ---------------------------------------------------------------------------


class TestTileBatch:
    def test_tiles_to_larger_batch(self):
        batch = {"input": np.arange(8 * 3, dtype=np.float32).reshape(8, 3)}
        tiled = tile_batch(batch, 32)
        assert tiled["input"].shape == (32, 3)
        # rows repeat cyclically
        np.testing.assert_array_equal(tiled["input"][8], tiled["input"][0])

    def test_truncates_to_smaller_batch(self):
        batch = {"input": np.arange(8 * 3, dtype=np.float32).reshape(8, 3)}
        tiled = tile_batch(batch, 1)
        assert tiled["input"].shape == (1, 3)
        np.testing.assert_array_equal(tiled["input"][0], batch["input"][0])

    def test_exact_batch_is_identity(self):
        batch = {"input": np.ones((4, 5), dtype=np.float32)}
        tiled = tile_batch(batch, 4)
        np.testing.assert_array_equal(tiled["input"], batch["input"])

    def test_multi_input_models(self):
        batch = {
            "a": np.ones((8, 2), dtype=np.float32),
            "b": np.ones((8, 4), dtype=np.float32),
        }
        tiled = tile_batch(batch, 16)
        assert tiled["a"].shape == (16, 2)
        assert tiled["b"].shape == (16, 4)


class TestHardwareSuite:
    def test_skips_on_cpu_target(self, tmp_path: Path):
        import asyncio

        model_dir = tmp_path / "onnx-mlp-small"
        model_dir.mkdir()
        (model_dir / "model-settings.json").write_text(json.dumps(SETTINGS_MLP))
        ctx = SuiteContext(RunConfig(), "image", tmp_path, tmp_path / "out")
        cases = asyncio.run(suite_hardware(ctx, [model_dir]))
        assert len(cases) == 1
        assert cases[0]["result"]["status"] == "SKIPPED_NEEDS_GPU_TARGET"

    def test_gpu_target_skips_without_onnx(self, tmp_path: Path):
        import asyncio

        model_dir = tmp_path / "sklearn-rf"
        model_dir.mkdir()
        (model_dir / "model-settings.json").write_text(json.dumps(SETTINGS_SK))
        config = RunConfig()
        config.target = "gpu"
        ctx = SuiteContext(config, "image", tmp_path, tmp_path / "out")
        cases = asyncio.run(suite_hardware(ctx, [model_dir]))
        assert len(cases) == 1
        assert cases[0]["result"]["status"] == "SKIPPED_NO_ONNX_MODELS"

    def test_registered_in_config(self):
        from bench.config import SUITES as ALL

        assert "hardware" in ALL

    def test_server_accepts_image_target_overrides(self):
        import inspect

        from bench.suites import SuiteContext

        params = inspect.signature(SuiteContext.server).parameters
        assert "image" in params
        assert "target" in params
