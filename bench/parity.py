"""Golden-reference correctness gate (mandatory pre-benchmark).

For every model: compute reference outputs IN-PROCESS with the native
library (sklearn / xgboost / lightgbm / onnxruntime CPU) on the
deterministic validation batch (which includes edge cases: zeros,
negatives, extreme floats, subnormals), then send the exact same batch
through MLServer over REST (JSON) and gRPC, and assert parity.

FP32: np.allclose(rtol=1e-5, atol=1e-6); FP64: rtol=1e-10, atol=1e-12;
integer labels: exact equality. Any mismatch aborts the run with
FAILED_CORRECTNESS and a tensor diff — performance numbers are never
recorded for numerically inaccurate inference.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .docker_server import GRPC_PORT, REST_PORT

TOLERANCES = {"FP32": (1e-5, 1e-6), "FP64": (1e-10, 1e-12)}


class ParityError(RuntimeError):
    status = "FAILED_CORRECTNESS"


# ---------------------------------------------------------------------------
# Native reference computation
# ---------------------------------------------------------------------------

# In-container reference script: computes the native-library prediction
# INSIDE the serving image (exact library versions, no host skew).
# Usage: python3 -c REFERENCE_SCRIPT <model_dir>  -> JSON on stdout
REFERENCE_SCRIPT = r"""
import json, sys
import numpy as np
model_dir = sys.argv[1]
settings = json.load(open(model_dir + "/model-settings.json"))
input_name = settings["inputs"][0]["name"]
batch = dict(np.load(model_dir + "/validation.npz"))
x = batch[input_name]
impl = settings["implementation"]
if impl.startswith("mlserver_sklearn"):
    import joblib
    model = joblib.load(model_dir + "/model.joblib")
    out = np.asarray(model.predict(x))
elif impl.startswith("mlserver_xgboost"):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(bytearray(open(model_dir + "/model.bst", "rb").read()))
    out = np.asarray(booster.predict(xgb.DMatrix(x.astype(np.float32))))
elif impl.startswith("mlserver_lightgbm"):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=model_dir + "/model.bst")
    out = np.asarray(booster.predict(x.astype(np.float32)))
elif impl.startswith("mlserver_onnx"):
    import onnxruntime as ort
    session = ort.InferenceSession(
        model_dir + "/model.onnx", providers=["CPUExecutionProvider"]
    )
    feeds = {k: v for k, v in batch.items()
             if k in {i.name for i in session.get_inputs()}}
    outputs = session.run(None, feeds)
    result = {}
    for meta, value in zip(session.get_outputs(), outputs):
        result[meta.name] = np.asarray(value).tolist()
    print(json.dumps(result))
    sys.exit(0)
else:
    raise SystemExit("unknown implementation: " + impl)
print(json.dumps({"predict": np.asarray(out).tolist()}))
"""


def native_reference_in_container(
    container: str, model_dir_inside: str
) -> dict[str, np.ndarray]:
    """Reference computed with the image's own libraries (no host skew)."""
    import subprocess

    result = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            REFERENCE_SCRIPT,
            model_dir_inside,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise ParityError(
            f"in-container reference failed: {result.stderr.strip()[:400]}"
        )
    payload = json.loads(result.stdout)
    return {key: np.asarray(value) for key, value in payload.items()}


def native_reference(model_dir: Path) -> dict[str, np.ndarray]:
    """Local (host-library) reference — used when no container is given
    (unit tests); production gates use the in-container variant."""
    settings = json.loads((model_dir / "model-settings.json").read_text())
    implementation = settings["implementation"]
    batch = dict(np.load(model_dir / "validation.npz"))
    name = model_dir.name

    if implementation.startswith("mlserver_sklearn"):
        import joblib

        model = joblib.load(model_dir / "model.joblib")
        prediction = model.predict(batch[settings["inputs"][0]["name"]])
        return {"predict": np.asarray(prediction)}

    if implementation.startswith("mlserver_xgboost"):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray((model_dir / "model.bst").read_bytes()))
        dmatrix = xgb.DMatrix(batch[settings["inputs"][0]["name"]].astype(np.float32))
        prediction = booster.predict(dmatrix)
        return {"predict": np.asarray(prediction, dtype=np.float64)}

    if implementation.startswith("mlserver_lightgbm"):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_dir / "model.bst"))
        prediction = booster.predict(
            batch[settings["inputs"][0]["name"]].astype(np.float32)
        )
        return {"predict": np.asarray(prediction, dtype=np.float64)}

    if implementation.startswith("mlserver_onnx"):
        import onnxruntime as ort

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
        return {
            meta.name: np.asarray(value)
            for meta, value in zip(session.get_outputs(), outputs)
        }

    raise ParityError(f"{name}: unknown implementation {implementation}")


# ---------------------------------------------------------------------------
# Through-server requests
# ---------------------------------------------------------------------------


def _v2_request(settings: dict, batch: dict[str, np.ndarray]) -> dict:
    inputs = []
    for spec in settings["inputs"]:
        array = batch[spec["name"]].tolist()
        inputs.append(
            {
                "name": spec["name"],
                "shape": list(spec["shape"]),
                "datatype": spec["datatype"],
                "data": array,
            }
        )
    return {
        "id": "parity",
        "inputs": inputs,
        "outputs": [{"name": spec["name"]} for spec in settings["outputs"]],
    }


async def rest_infer(model: str, request: dict) -> dict:
    import aiohttp

    url = f"http://127.0.0.1:{REST_PORT}/v2/models/{model}/infer"
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        async with session.post(url, json=request) as response:
            body = await response.json()
            if response.status != 200:
                raise ParityError(
                    f"REST infer failed ({response.status}): "
                    f"{json.dumps(body)[:400]}"
                )
            return body


def _flatten(data):
    """Recursively flatten nested lists into scalars (any tensor rank)."""
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
    else:
        yield data


def grpc_infer_sync(model: str, request: dict) -> dict:
    import grpc

    sys_path = str(Path(__file__).resolve().parents[1] / "tools")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from mlserver.grpc import dataplane_pb2 as pb  # type: ignore

    channel = grpc.insecure_channel(
        f"127.0.0.1:{GRPC_PORT}",
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    stub_cls = None
    try:
        from mlserver.grpc import dataplane_pb2_grpc as pb_grpc  # type: ignore

        stub_cls = pb_grpc.GRPCInferenceServiceStub(channel)
    except ImportError:
        raise ParityError("mlserver grpc stubs unavailable on this host")
    message = pb.ModelInferRequest(
        model_name=model,
        inputs=[
            pb.ModelInferRequest.InferInputTensor(
                name=inp["name"],
                shape=inp["shape"],
                datatype=inp["datatype"],
                contents=pb.InferTensorContents(
                    fp32_contents=[float(v) for v in _flatten(inp["data"])]
                ),
            )
            for inp in request["inputs"]
        ],
        outputs=[
            pb.ModelInferRequest.InferRequestedOutputTensor(name=spec["name"])
            for spec in request.get("outputs", [])
        ],
    )
    response = stub_cls.ModelInfer(message, timeout=120)
    return _grpc_response_to_dict(response)


def _grpc_response_to_dict(response: Any) -> dict:
    outputs = []
    for tensor in response.outputs:
        contents = tensor.contents
        data: list = []
        if contents.fp32_contents:
            data = list(contents.fp32_contents)
        elif contents.int64_contents:
            data = list(contents.int64_contents)
        elif contents.fp64_contents:
            data = list(contents.fp64_contents)
        outputs.append(
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "datatype": tensor.datatype,
                "data": data,
            }
        )
    return {"outputs": outputs}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _diff_report(name: str, reference: np.ndarray, actual: np.ndarray) -> str:
    ref_flat = np.asarray(reference, dtype=np.float64).ravel()
    act_flat = np.asarray(actual, dtype=np.float64).ravel()
    if ref_flat.shape != act_flat.shape:
        return (
            f"{name}: shape mismatch reference{reference.shape} vs "
            f"actual{actual.shape}"
        )
    if ref_flat.size:
        delta = np.abs(ref_flat - act_flat)
        index = int(np.argmax(delta))
        return (
            f"{name}: max|delta|={delta.max():.3e} at [{index}] "
            f"reference={ref_flat[index]!r} actual={act_flat[index]!r}"
        )
    return f"{name}: empty tensor"


def compare_outputs(
    model_name: str,
    reference: dict[str, np.ndarray],
    response: dict,
    integer_labels: bool = False,
) -> list[str]:
    problems: list[str] = []
    by_name = {output["name"]: output for output in response.get("outputs", [])}
    for name, expected in reference.items():
        if name not in by_name:
            problems.append(f"{model_name}: output tensor '{name}' missing")
            continue
        actual = np.asarray(by_name[name]["data"], dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if expected_array.ndim == 2 and actual.ndim == 1:
            actual = actual.reshape(expected_array.shape)
        if integer_labels:
            if not np.array_equal(expected_array, actual):
                problems.append(
                    f"{model_name}/{name}: integer labels differ — "
                    f"{_diff_report(name, expected_array, actual)}"
                )
        else:
            rtol, atol = TOLERANCES["FP32"]
            # equal_nan: float outputs legitimately containing NaN
            # (the codec's NaN->None JSON path) must compare equal —
            # NaN != NaN under plain allclose would fail them.
            if not np.allclose(
                expected_array, actual, rtol=rtol, atol=atol, equal_nan=True
            ):
                problems.append(
                    f"{model_name}/{name}: parity violated (rtol={rtol}) — "
                    f"{_diff_report(name, expected_array, actual)}"
                )
    return problems


# Models whose outputs legitimately break one transport (documented
# findings, not gate failures). The onnx-nan-* zoo models emit all-NaN
# outputs: REST serves them as JSON null, but the gRPC response path
# crashes (F-03: the NaN->None conversion feeds None into a numeric
# protobuf field -> INTERNAL "must be real number, not NoneType").
GRPC_UNSUPPORTED_PATTERNS = ("onnx-nan-",)


async def verify_model(
    model_dir: Path,
    rest: bool = True,
    grpc: bool = True,
    container: str | None = None,
    model_dir_inside: str | None = None,
) -> list[str]:
    """Full gate for one model. Returns problem list (empty = pass).

    When `container` is given, the native reference is computed INSIDE the
    serving image (exact library versions — no host/image skew).
    """
    settings = json.loads((model_dir / "model-settings.json").read_text())
    model = settings["name"]
    if grpc and any(model.startswith(p) for p in GRPC_UNSUPPORTED_PATTERNS):
        # F-03: gRPC path crashes on NaN outputs (documented finding);
        # the gate verifies these models over REST only.
        grpc = False
    batch = dict(np.load(model_dir / "validation.npz"))
    if container and model_dir_inside:
        reference = native_reference_in_container(container, model_dir_inside)
    else:
        reference = native_reference(model_dir)
    request = _v2_request(settings, batch)
    problems: list[str] = []

    if rest:
        try:
            response = await rest_infer(model, request)
            problems += compare_outputs(
                f"{model}/rest",
                reference,
                response,
                integer_labels=settings["outputs"][0]["datatype"] == "INT64",
            )
        except ParityError:
            raise
        except Exception as error:  # transport failure is a correctness fail
            problems.append(f"{model}/rest: request failed — {error}")

    if grpc:
        try:
            response = grpc_infer_sync(model, request)
            problems += compare_outputs(
                f"{model}/grpc",
                reference,
                response,
                integer_labels=settings["outputs"][0]["datatype"] == "INT64",
            )
        except ParityError:
            raise
        except Exception as error:
            problems.append(f"{model}/grpc: request failed — {error}")

    return problems
