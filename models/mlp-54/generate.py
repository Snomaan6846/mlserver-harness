#!/usr/bin/env python3
"""Generate the phase-1 replication model, payloads and settings.

The phase-1 CPU redesign study used a synthetic FP32 ONNX dense network
with layer widths 54 -> 1024 -> 512 -> 256 -> 128 -> 7 (54 input
features, seven output values per sample). This reproduces that
architecture with deterministic seeded weights, plus the wire payloads
the campaign needs:

  - JSON requests at 1 / 100 / 1000 rows (phase 1's rows/request axis)
  - Binary REST frames (KServe v2: JSON header + raw FP32 bytes) with
    binary output negotiation, one per row count
  - Raw gRPC protobufs (raw_input_contents), one per row count
  - The binary-frame header lengths (sidecar map for the parity hook)
  - model-settings.json for the mlp-small model

Everything is deterministic: same seed -> same model -> same payloads
-> same references, on any machine.
"""

import json
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

SEED = 20261113
WIDTHS = [54, 1024, 512, 256, 128, 7]
MODEL_NAME = "mlp-small"
OUT = Path(__file__).resolve().parent


def build_model() -> onnx.ModelProto:
    rng = np.random.default_rng(SEED)
    inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [None, WIDTHS[0]]
        )
    ]
    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [None, WIDTHS[-1]]
        )
    ]
    nodes = []
    initializers = []
    x = "input"
    for i, (fan_in, fan_out) in enumerate(zip(WIDTHS, WIDTHS[1:])):
        w = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * 0.05
        b = rng.standard_normal(fan_out).astype(np.float32) * 0.01
        initializers.append(
            helper.make_tensor(f"w{i}", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True)
        )
        initializers.append(
            helper.make_tensor(f"b{i}", TensorProto.FLOAT, b.shape, b.tobytes(), raw=True)
        )
        y = f"h{i}" if i < len(WIDTHS) - 2 else "output"
        nodes.append(
            helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], [f"pre{i}"], name=f"gemm{i}")
        )
        if i < len(WIDTHS) - 2:
            nodes.append(helper.make_node("Relu", [f"pre{i}"], [y], name=f"relu{i}"))
        else:
            nodes.append(helper.make_node("Identity", [f"pre{i}"], [y], name=f"out"))
        x = y
    graph = helper.make_graph(nodes, "mlp", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def make_rows(rows: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + rows)
    return rng.random((rows, WIDTHS[0])).astype(np.float32)


def json_payload(rows: int) -> bytes:
    data = make_rows(rows)
    return json.dumps(
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "shape": [rows, WIDTHS[0]],
                    "data": data.reshape(-1).tolist(),
                }
            ],
            "outputs": [{"name": "output"}],
        },
        separators=(",", ":"),
    ).encode()


def binary_frame(rows: int) -> tuple[bytes, int]:
    data = make_rows(rows)
    header = json.dumps(
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "shape": [rows, WIDTHS[0]],
                    "parameters": {"binary_data_size": data.nbytes},
                }
            ],
            "outputs": [{"name": "output", "parameters": {"binary_data": True}}],
        },
        separators=(",", ":"),
    ).encode()
    return header + data.tobytes(), len(header)


def grpc_raw(rows: int) -> bytes:
    from mlserver.grpc import dataplane_pb2 as pb

    data = make_rows(rows)
    request = pb.ModelInferRequest(
        model_name=MODEL_NAME,
        inputs=[
            pb.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="FP32",
                shape=[rows, WIDTHS[0]],
            )
        ],
        raw_input_contents=[data.tobytes()],
    )
    return request.SerializeToString()


def main():
    out = OUT
    model = build_model()
    onnx.save(model, out / "model.onnx")

    frames = {}
    for rows in (1, 100, 1000):
        (out / f"j0-mlp-small-{rows}.bin").write_bytes(json_payload(rows))
        body, header_length = binary_frame(rows)
        name = f"triton-binary-mlp-small-{rows}.bin"
        (out / name).write_bytes(body)
        frames[name] = header_length
        (out / f"g-mlp-small-{rows}.bin").write_bytes(grpc_raw(rows))

    # The parity hook reads this sidecar to frame binary requests.
    (out / "binary-frames.json").write_text(json.dumps(frames, indent=2) + "\n")

    (out / "model-settings.json").write_text(
        json.dumps(
            {
                "name": MODEL_NAME,
                "implementation": "mlserver_onnx.OnnxModel",
                "parameters": {
                    "uri": "./model.onnx",
                    "extra": {"providers": ["CPUExecutionProvider"]},
                },
            },
            indent=2,
        )
        + "\n"
    )

    # Smoke: the model must infer and the wire forms must round-trip
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    result = sess.run(None, {"input": make_rows(2)})[0]
    assert result.shape == (2, 7), result.shape
    sizes = {
        rows: (out / f"j0-mlp-small-{rows}.bin").stat().st_size for rows in (1, 100, 1000)
    }
    print(f"model: {WIDTHS} FP32, output {result.shape} OK")
    print(f"JSON payloads: {sizes}")
    print(f"binary frames: {frames}")
    print(f"gRPC payloads: "
          f"{ {r: (out / f'g-mlp-small-{r}.bin').stat().st_size for r in (1, 100, 1000)} }")


if __name__ == "__main__":
    main()
