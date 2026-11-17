#!/usr/bin/env python3
"""Generate the transport domain: variants and plans.

IPC and network-path benchmarks:

  - queue-vs-zmq: the frontend<->worker transport (multiprocessing queue
    vs ZeroMQ) — measured ~10-11% in zmq's favour on every processing
    track. Workload: MLP-54 (fast model, dispatch-sensitive).
  - route-vs-direct: OpenShift edge-TLS route vs cluster-internal service
    — route overhead appears only at C1; both reach the device plateau
    at C8+. Qualifies the binary extension for external routes
    (KSERVE_QUALIFIED). Note: routed gRPC is unavailable on OpenShift
    edge routes (h2c backend) — REST only.

Plans:
  queue-vs-zmq      3 processing tracks on queue vs zmq (MLP-54)
  route-vs-direct   binary REST via route vs service (RF-DETR)
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUDA_PROVIDERS = '["CUDAExecutionProvider","CPUExecutionProvider"]'
REPO = HERE.parents[1]
IMAGES = json.loads((REPO / "infrastructure" / "images.json").read_text())

CPU = {
    "deployment": "mlserver-rfdetr-cpu",
    "service": "mlserver-rfdetr-cpu",
    "device": "cpu",
    "expect_cuda": False,
    "image": IMAGES["ship"],
    "model": "mlp-small",
}
GPU = {
    "deployment": "mlserver-rfdetr",
    "service": "mlserver-rfdetr",
    "device": "cuda",
    "expect_cuda": True,
    "image": IMAGES["ship"],
}


def cpu_env(
    transport="zmq", worker_json="false", binary="false", worker_binary="false"
):
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": '["CPUExecutionProvider"]',
        "MLSERVER_PARALLEL_WORKERS": "4",
        "MLSERVER_REST_WORKER_JSON": worker_json,
        "MLSERVER_REST_BINARY_TENSOR": binary,
        "MLSERVER_REST_WORKER_BINARY": worker_binary,
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": transport,
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


def gpu_binary_env():
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": CUDA_PROVIDERS,
        "MLSERVER_PARALLEL_WORKERS": "1",
        "MLSERVER_REST_WORKER_JSON": "false",
        "MLSERVER_REST_BINARY_TENSOR": "true",
        "MLSERVER_REST_WORKER_BINARY": "false",
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
    # queue vs zmq on three processing tracks (MLP-54, 1-row payloads)
    "mlp-json-frontend-queue": {**CPU, "env": cpu_env(transport="queue")},
    "mlp-json-frontend-zmq": {**CPU, "env": cpu_env(transport="zmq")},
    "mlp-json-worker-queue": {
        **CPU,
        "env": cpu_env(transport="queue", worker_json="true"),
    },
    "mlp-json-worker-zmq": {**CPU, "env": cpu_env(transport="zmq", worker_json="true")},
    "mlp-binary-worker-queue": {
        **CPU,
        "env": cpu_env(transport="queue", binary="true", worker_binary="true"),
    },
    "mlp-binary-worker-zmq": {
        **CPU,
        "env": cpu_env(transport="zmq", binary="true", worker_binary="true"),
    },
    # route qualification (RF-DETR, binary REST)
    "route-direct": {**GPU, "env": gpu_binary_env()},
    "route-edge-tls": {**GPU, "env": gpu_binary_env()},
}

CPU_URL = "http://mlserver-rfdetr-cpu.mlserver-perf.svc:8080/v2/models/mlp-small/infer"


def mlp_json_spec():
    return {
        "protocol": "rest",
        "url": CPU_URL,
        "payloads": ["/store/payloads/j0-mlp-small-1.bin"],
        "headers": {"Content-Type": "application/json"},
        "min_response_bytes": 20,
    }


def mlp_binary_spec():
    return {
        "protocol": "rest",
        "url": CPU_URL,
        "payloads": ["/store/payloads/triton-binary-mlp-small-1.bin"],
        "headers": {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": "164",
        },
        "min_response_bytes": 20,
    }


ROUTE_HOST = os.environ.get(
    "BENCHMARK_ROUTE_HOST",
    # Your cluster's external route hostname (assigned by OpenShift when
    # infrastructure/manifests/route.yaml is applied):
    #   oc -n mlserver-perf get route mlserver-rfdetr \
    #       -o jsonpath={.spec.host}
    "mlserver-rfdetr-mlserver-perf.apps.REPLACE-ME.example.com",
)


def route_spec(via_route: bool):
    host = (
        "mlserver-rfdetr.mlserver-perf.svc:8080"
        if not via_route
        else f"{ROUTE_HOST}:443"
    )
    scheme = "http" if not via_route else "https"
    return {
        "protocol": "rest",
        "url": f"{scheme}://{host}/v2/models/rfdetr-small/infer",
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


plans = {
    "queue-vs-zmq": plan(
        "queue-vs-zmq",
        [
            entry(
                "mlp-json-frontend-queue", "mlp-json-frontend-queue", mlp_json_spec()
            ),
            entry("mlp-json-frontend-zmq", "mlp-json-frontend-zmq", mlp_json_spec()),
            entry("mlp-json-worker-queue", "mlp-json-worker-queue", mlp_json_spec()),
            entry("mlp-json-worker-zmq", "mlp-json-worker-zmq", mlp_json_spec()),
            entry(
                "mlp-binary-worker-queue", "mlp-binary-worker-queue", mlp_binary_spec()
            ),
            entry("mlp-binary-worker-zmq", "mlp-binary-worker-zmq", mlp_binary_spec()),
        ],
        [64],
    ),
    # The route hostname comes from BENCHMARK_ROUTE_HOST (see
    # route_spec above; apply manifests/route.yaml first, then read
    # the assigned host with: oc -n mlserver-perf get route
    # mlserver-rfdetr -o jsonpath={.spec.host}).
    "route-vs-direct": plan(
        "route-vs-direct",
        [
            entry("route-direct", "route-direct", route_spec(False)),
            entry("route-edge-tls", "route-edge-tls", route_spec(True)),
        ],
        [1, 8, 16],
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
        f"transport: {len(variants)} variants, {len(plans)} plans, " f"{total} trials"
    )


if __name__ == "__main__":
    main()
