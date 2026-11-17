#!/usr/bin/env python3
"""Generate the stability domain: variants and plans.

Long-duration reliability under continuous load. The soak plan drives the
optimized serving path (binary tensor + worker processing) at device
saturation for a full hour: zero failed requests and bounded memory are
the pass criteria.

Plans:
  soak-60min    one variant, C8, 3600 s measured window, single round
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

SOAK_WORKER_ENV = {
    "MLSERVER_MODEL_ONNX_PROVIDERS": '["CUDAExecutionProvider","CPUExecutionProvider"]',
    "MLSERVER_PARALLEL_WORKERS": "1",
    "MLSERVER_REST_WORKER_JSON": "false",
    "MLSERVER_REST_BINARY_TENSOR": "true",
    "MLSERVER_REST_WORKER_BINARY": "true",
    "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
    "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
    "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
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
    "soak-binary-worker": {**GPU, "env": SOAK_WORKER_ENV},
    "soak-binary-frontend": {
        **GPU,
        "env": {**SOAK_WORKER_ENV, "MLSERVER_REST_WORKER_BINARY": "false"},
    },
}

BINARY_SPEC = {
    "protocol": "rest",
    "url": "http://mlserver-rfdetr.mlserver-perf.svc:8080"
    "/v2/models/rfdetr-small/infer",
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}


def entry(name, hook_variant, spec):
    return {
        "name": name,
        "prepare_argv": ["python3", "/store/tools/hook.py", hook_variant],
        "spec": spec,
    }


plans = {
    "soak-60min": {
        "round_offset": 0,
        "rounds": 1,
        "clients": 2,
        "seconds": 3600,
        "warmup_seconds": 60,
        "warmup_requests": 8,
        "timeout_seconds": 3700,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "concurrency": [8],
        "session": "soak-60min",
        "variants": [
            entry("soak-binary-worker", "soak-binary-worker", BINARY_SPEC),
        ],
    },
    "soak-60min-frontend": {
        "round_offset": 0,
        "rounds": 1,
        "clients": 2,
        "seconds": 3600,
        "warmup_seconds": 60,
        "warmup_requests": 8,
        "timeout_seconds": 3700,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "concurrency": [8],
        "session": "soak-60min-frontend",
        "variants": [
            entry("soak-binary-frontend", "soak-binary-frontend", BINARY_SPEC),
        ],
    },
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
        f"stability: {len(variants)} variants, {len(plans)} plans, " f"{total} trials"
    )


if __name__ == "__main__":
    main()
