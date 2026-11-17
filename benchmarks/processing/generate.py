#!/usr/bin/env python3
"""Generate the processing domain: variants and plans.

Processing benchmarks measure WHERE request parsing and response encoding
happen — the frontend asyncio event loop vs parallel inference workers
(rest_worker_json / rest_worker_binary). Two workload regimes:

  - GPU compute-bound (RF-DETR Small): throughput is device-limited; the
    payoff is CPU cost and single-request latency.
  - CPU frontend-bound (MLP-54): the frontend dispatch is the ceiling;
    the payoff is throughput for small payloads.

Plans:
  frontend-vs-worker-gpu    stock/binary-frontend/binary-worker (RF-DETR)
  frontend-vs-worker-cpu    6 tracks x 3 row counts (MLP-54)
  worker-scaling            W1-W12 flat-check (MLP-54)
  concurrency-saturation    C2-C128 (MLP-54)
  batched-fallback          batched model +/- worker-binary
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
CPU = {
    "deployment": "mlserver-rfdetr-cpu",
    "service": "mlserver-rfdetr-cpu",
    "device": "cpu",
    "expect_cuda": False,
    "image": IMAGES["ship"],
    "model": "mlp-small",
}


def gpu_env(
    binary="false", worker_json="false", worker_binary="false", cuda_graph="false"
):
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": CUDA_PROVIDERS,
        "MLSERVER_PARALLEL_WORKERS": "1",
        "MLSERVER_REST_WORKER_JSON": worker_json,
        "MLSERVER_REST_BINARY_TENSOR": binary,
        "MLSERVER_REST_WORKER_BINARY": worker_binary,
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
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


def cpu_env(
    workers=4,
    transport="zmq",
    worker_json="false",
    binary="false",
    worker_binary="false",
    threads=2,
):
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": '["CPUExecutionProvider"]',
        "MLSERVER_PARALLEL_WORKERS": str(workers),
        "MLSERVER_REST_WORKER_JSON": worker_json,
        "MLSERVER_REST_BINARY_TENSOR": binary,
        "MLSERVER_REST_WORKER_BINARY": worker_binary,
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": transport,
        "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
        "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
        "MLSERVER_MODEL_ONNX_SESSION_OPTIONS": json.dumps(
            {"intra_op_num_threads": threads, "inter_op_num_threads": 1}
        ),
        "MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES": json.dumps(
            {
                "session.intra_op.allow_spinning": "0",
                "session.inter_op.allow_spinning": "0",
            },
        ),
    }


def grpc_cpu_env(raw_buffers="false", **kw):
    e = cpu_env(**kw)
    e["MLSERVER_GRPC_RAW_INPUT_BUFFERS"] = raw_buffers
    e["MLSERVER_REST_WORKER_JSON"] = "false"
    e["MLSERVER_REST_BINARY_TENSOR"] = "false"
    e["MLSERVER_REST_WORKER_BINARY"] = "false"
    return e


GPU_BATCHED = {**GPU, "model": "rfdetr-small-batched"}

variants = {
    # --- GPU (RF-DETR) ---
    "gpu-stock-json": {**GPU, "env": gpu_env()},
    "gpu-binary-frontend": {**GPU, "env": gpu_env(binary="true")},
    "gpu-binary-worker": {**GPU, "env": gpu_env(binary="true", worker_binary="true")},
    "gpu-batched-frontend": {**GPU_BATCHED, "env": gpu_env(binary="true")},
    "gpu-batched-worker": {
        **GPU_BATCHED,
        "env": gpu_env(binary="true", worker_binary="true"),
    },
    # --- CPU (MLP-54) ---
    "mlp-json-frontend": {**CPU, "env": cpu_env()},
    "mlp-json-worker": {**CPU, "env": cpu_env(worker_json="true")},
    "mlp-binary-frontend": {**CPU, "env": cpu_env(binary="true")},
    "mlp-binary-worker": {**CPU, "env": cpu_env(binary="true", worker_binary="true")},
    "mlp-grpc-stock": {**CPU, "env": grpc_cpu_env()},
    "mlp-grpc-raw": {**CPU, "env": grpc_cpu_env(raw_buffers="true")},
}
# Worker-count axis (MLP)
for w in (1, 2, 8, 12):
    variants[f"mlp-json-frontend-w{w}"] = {**CPU, "env": cpu_env(workers=w)}
    variants[f"mlp-json-worker-w{w}"] = {
        **CPU,
        "env": cpu_env(workers=w, worker_json="true"),
    }
    variants[f"mlp-binary-worker-w{w}"] = {
        **CPU,
        "env": cpu_env(workers=w, binary="true", worker_binary="true"),
    }
    variants[f"mlp-grpc-raw-w{w}"] = {
        **CPU,
        "env": grpc_cpu_env(raw_buffers="true", workers=w),
    }

GPU_URL = "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer"
GPU_BATCHED_URL = (
    "http://mlserver-rfdetr.mlserver-perf.svc:8080"
    "/v2/models/rfdetr-small-batched/infer"
)
CPU_URL = "http://mlserver-rfdetr-cpu.mlserver-perf.svc:8080/v2/models/mlp-small/infer"
CPU_TARGET = "mlserver-rfdetr-cpu.mlserver-perf.svc:8081"

GPU_JSON_SPEC = {
    "protocol": "rest",
    "url": GPU_URL,
    "payloads": ["/store/payloads/j0-0.bin"],
    "headers": {"Content-Type": "application/json"},
    "min_response_bytes": 100000,
}
GPU_BINARY_SPEC = {
    "protocol": "rest",
    "url": GPU_URL,
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}
GPU_BATCHED_BINARY_SPEC = {**GPU_BINARY_SPEC, "url": GPU_BATCHED_URL}

ROWS = (1, 100, 1000)
FRAMES = {1: 164, 100: 168, 1000: 170}


def mlp_json_spec(rows):
    return {
        "protocol": "rest",
        "url": CPU_URL,
        "payloads": [f"/store/payloads/j0-mlp-small-{rows}.bin"],
        "headers": {"Content-Type": "application/json"},
        "min_response_bytes": 20,
    }


def mlp_binary_spec(rows):
    return {
        "protocol": "rest",
        "url": CPU_URL,
        "payloads": [f"/store/payloads/triton-binary-mlp-small-{rows}.bin"],
        "headers": {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(FRAMES[rows]),
        },
        "min_response_bytes": 20,
    }


def mlp_grpc_spec(rows):
    return {
        "protocol": "grpc",
        "target": CPU_TARGET,
        "payloads": [f"/store/payloads/g-mlp-small-{rows}.bin"],
        "min_response_bytes": 20,
    }


def mlp_spec(track, rows):
    if "grpc" in track:
        return mlp_grpc_spec(rows)
    if track.startswith("mlp-binary"):
        return mlp_binary_spec(rows)
    return mlp_json_spec(rows)


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
        "warmup_requests": 256,
        "timeout_seconds": 120,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "concurrency": concurrency,
        "session": session,
        "variants": entries,
    }


TRACKS = {
    "mlp-json-frontend": "mlp-json-frontend",
    "mlp-json-worker": "mlp-json-worker",
    "mlp-binary-frontend": "mlp-binary-frontend",
    "mlp-binary-worker": "mlp-binary-worker",
    "mlp-grpc-stock": "mlp-grpc-stock",
    "mlp-grpc-raw": "mlp-grpc-raw",
}

plans = {
    "frontend-vs-worker-gpu": plan(
        "frontend-vs-worker-gpu",
        [
            entry("gpu-stock-json", "gpu-stock-json", GPU_JSON_SPEC),
            entry("gpu-binary-frontend", "gpu-binary-frontend", GPU_BINARY_SPEC),
            entry("gpu-binary-worker", "gpu-binary-worker", GPU_BINARY_SPEC),
        ],
        [1, 4, 8, 16],
    ),
    "frontend-vs-worker-cpu": plan(
        "frontend-vs-worker-cpu",
        [
            entry(f"{track}@r{rows}", track, mlp_spec(track, rows))
            for track, hook in TRACKS.items()
            for rows in ROWS
        ],
        [64],
    ),
    "worker-scaling": plan(
        "worker-scaling",
        [
            entry(f"{track}-w{w}@r1", f"{track}-w{w}", mlp_spec(track, 1))
            for track in (
                "mlp-json-frontend",
                "mlp-json-worker",
                "mlp-binary-worker",
                "mlp-grpc-raw",
            )
            for w in (1, 2, 8, 12)
        ],
        [64],
    ),
    "concurrency-saturation": plan(
        "concurrency-saturation",
        [
            entry(f"{track}@r1", track, mlp_spec(track, 1))
            for track in (
                "mlp-json-frontend",
                "mlp-json-worker",
                "mlp-binary-worker",
                "mlp-grpc-raw",
            )
        ],
        [2, 16, 128],
    ),
    "batched-fallback": plan(
        "batched-fallback",
        [
            entry(
                "gpu-batched-frontend", "gpu-batched-frontend", GPU_BATCHED_BINARY_SPEC
            ),
            entry("gpu-batched-worker", "gpu-batched-worker", GPU_BATCHED_BINARY_SPEC),
        ],
        [4],
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
        f"processing: {len(variants)} variants, {len(plans)} plans, " f"{total} trials"
    )
    for name, p in plans.items():
        n = len(p["variants"]) * len(p["concurrency"]) * p["rounds"]
        print(f"  {name}: {n} trials")


if __name__ == "__main__":
    main()
