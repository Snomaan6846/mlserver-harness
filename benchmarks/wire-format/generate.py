#!/usr/bin/env python3
"""Generate the wire-format domain: variants and plans.

Wire-format benchmarks measure protocol and encoding efficiency — how the
request bytes become tensors and back. Workload: RF-DETR Small on the GPU
deployment (compute-bound; the frontend is the bottleneck in stock
configurations).

Plans:
  stock-vs-binary      REST JSON vs KServe v2 binary tensor (headline 4x)
  stock-vs-grpc        stock gRPC vs opt-in raw buffers + converter fix
  binary-fallback      binary ON serving JSON clients (no-regression gate)
  mixed-negotiation    alternating binary/JSON on the batched model
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUDA_PROVIDERS = '["CUDAExecutionProvider","CPUExecutionProvider"]'
REPO = HERE.parents[1]
IMAGES = json.loads((REPO / "infrastructure" / "images.json").read_text())

GPU = {
    "deployment": "mlserver-rfdetr",
    "service": "mlserver-rfdetr",
    "device": "cuda",
    "expect_cuda": True,
    "image": IMAGES["ship"],
}
GPU_BATCHED = {**GPU, "model": "rfdetr-small-batched"}


def env(
    binary="false",
    grpc_raw="false",
    transport="zmq",
    cuda_graph="false",
    workers="1",
):
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": CUDA_PROVIDERS,
        "MLSERVER_PARALLEL_WORKERS": workers,
        "MLSERVER_REST_WORKER_JSON": "false",
        "MLSERVER_REST_BINARY_TENSOR": binary,
        "MLSERVER_REST_WORKER_BINARY": "false",
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": grpc_raw,
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": transport,
        "MLSERVER_MODEL_ONNX_CUDA_GRAPH": cuda_graph,
        "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
        "MLSERVER_MODEL_ONNX_SESSION_OPTIONS": json.dumps(
            {"intra_op_num_threads": 2, "inter_op_num_threads": 1}
        ),
        "MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES": json.dumps(
            {
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            },
        ),
    }


variants = {
    # REST wire tracks
    "stock-json": {**GPU, "env": env(transport="queue")},
    "binary-rest": {**GPU, "env": env(binary="true")},
    # JSON payloads against the binary-enabled server (fallback compliance)
    "binary-rest-json-clients": {**GPU, "env": env(binary="true")},
    # gRPC wire tracks
    "grpc-stock": {**GPU, "env": env(transport="queue")},
    "grpc-raw": {
        **GPU,
        "env": env(binary="true", grpc_raw="true", workers="2", cuda_graph="true"),
    },
    # Batched model (mixed wire co-batching)
    "batched-stock": {**GPU_BATCHED, "env": env(transport="queue")},
    "batched-binary": {**GPU_BATCHED, "env": env(binary="true")},
    "batched-binary-json-clients": {**GPU_BATCHED, "env": env(binary="true")},
}

URL = "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer"
BATCHED_URL = (
    "http://mlserver-rfdetr.mlserver-perf.svc:8080"
    "/v2/models/rfdetr-small-batched/infer"
)
TARGET = "mlserver-rfdetr.mlserver-perf.svc:8081"

JSON_SPEC = {
    "protocol": "rest",
    "url": URL,
    "payloads": ["/store/payloads/j0-0.bin"],
    "headers": {"Content-Type": "application/json"},
    "min_response_bytes": 100000,
}
BINARY_SPEC = {
    "protocol": "rest",
    "url": URL,
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}
GRPC_SPEC = {
    "protocol": "grpc",
    "target": TARGET,
    "payloads": ["/store/payloads/g-0.bin"],
    "min_response_bytes": 100000,
}
BATCHED_JSON_SPEC = {**JSON_SPEC, "url": BATCHED_URL}
BATCHED_BINARY_SPEC = {**BINARY_SPEC, "url": BATCHED_URL}


def entry(name, hook_variant, spec):
    return {
        "name": name,
        "prepare_argv": ["python3", "/store/tools/hook.py", hook_variant],
        "spec": spec,
    }


def plan(session, entries, concurrency, rounds=3):
    return {
        "round_offset": 0,
        "rounds": rounds,
        "clients": 2,
        "seconds": 60,
        "warmup_seconds": 60,
        "warmup_requests": 8,
        "timeout_seconds": 120,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "concurrency": concurrency,
        "session": session,
        "variants": entries,
    }


CONC = [1, 2, 4, 8, 16]

plans = {
    "stock-vs-binary": plan(
        "stock-vs-binary",
        [
            entry("stock-json", "stock-json", JSON_SPEC),
            entry("binary-rest", "binary-rest", BINARY_SPEC),
        ],
        CONC,
    ),
    "stock-vs-grpc": plan(
        "stock-vs-grpc",
        [
            entry("grpc-stock", "grpc-stock", GRPC_SPEC),
            entry("grpc-raw", "grpc-raw", GRPC_SPEC),
        ],
        CONC,
    ),
    "binary-fallback": plan(
        "binary-fallback",
        [
            entry("stock-json", "stock-json", JSON_SPEC),
            entry("binary-rest-json-clients", "binary-rest-json-clients", JSON_SPEC),
        ],
        CONC,
    ),
    "mixed-negotiation": plan(
        "mixed-negotiation",
        [
            entry("batched-binary", "batched-binary", BATCHED_BINARY_SPEC),
            entry(
                "batched-binary-json-clients",
                "batched-binary-json-clients",
                BATCHED_JSON_SPEC,
            ),
            entry("batched-stock", "batched-stock", BATCHED_JSON_SPEC),
        ],
        [1, 2, 4],
    ),
}


def main():
    (HERE / "variants.json").write_text(json.dumps(variants, indent=2))
    plans_dir = HERE / "plans"
    plans_dir.mkdir(exist_ok=True)
    for name, p in plans.items():
        (plans_dir / f"{name}.json").write_text(json.dumps(p, indent=2))
    total = sum(
        len(p["variants"]) * len(p["concurrency"]) * p["rounds"] for p in plans.values()
    )
    print(
        f"wire-format: {len(variants)} variants, {len(plans)} plans, " f"{total} trials"
    )
    for name, p in plans.items():
        n = len(p["variants"]) * len(p["concurrency"]) * p["rounds"]
        print(f"  {name}: {n} trials")


if __name__ == "__main__":
    main()
