#!/usr/bin/env python3
"""Generate the threading domain: variants and plans.

Thread and runtime tuning for ONNX Runtime under MLServer. Measured
findings this domain reproduces (Tesla T4 / 28-vCPU node):

  - GPU serving is device-bound: intra-op threads (T1-T8) and spinning
    have no effect; keep defaults.
  - CPU serving is T-dominant: intra_op_num_threads is the only lever
    (T2 -> T8 = 2.7x); worker count is flat.
  - Spinning adds ~11% on CPU at T8; ORT's default spin parameters are
    already optimal.
  - OS-managed affinity matches manual pinning.
  - cuDNN EXHAUSTIVE search = HEURISTIC plateau (bit-exact); DEFAULT
    drifts numerically and fails the parity gate (keep HEURISTIC).

Known limitation: ORT enum-typed settings (inter-op PARALLEL execution
mode) cannot be expressed through MLServer's JSON settings — the worker
raises TypeError at session init. There is deliberately no plan for it
until that defect is fixed (see reports/PHASE3_ROADMAP.md, WI-10).

Plans:
  gpu-thread-sweep          T1-T8 on REST and gRPC (expect: flat)
  gpu-spinning              spin ON/OFF (expect: no effect)
  gpu-cudnn                 EXHAUSTIVE vs HEURISTIC (expect: parity)
  cpu-thread-factorial      W{1,2,4} x T{1,2,4,8} (expect: T-dominant)
  cpu-grpc-thread-scaling   stock vs opt-in gRPC at T2/T8
  cpu-spinning              spin ON/OFF at W4xT8 (expect: ~+11%)
  cpu-affinity              OS-managed vs pinned (expect: parity)
  cpu-spin-tuning           ORT defaults vs tuned (expect: parity)
  cpu-latency-c1            T2 vs T8 single-request latency
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
}


def gpu_env(
    threads=2,
    spin="0",
    cudnn=None,
    grpc_raw="false",
    binary_worker=False,
    cuda_graph=False,
):
    e = {
        "MLSERVER_MODEL_ONNX_PROVIDERS": CUDA_PROVIDERS,
        "MLSERVER_PARALLEL_WORKERS": "1",
        "MLSERVER_REST_WORKER_JSON": "false",
        "MLSERVER_REST_BINARY_TENSOR": "true",
        "MLSERVER_REST_WORKER_BINARY": str(binary_worker).lower(),
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": grpc_raw,
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
        "MLSERVER_MODEL_ONNX_CUDA_GRAPH": str(cuda_graph).lower(),
        "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
        "MLSERVER_MODEL_ONNX_SESSION_OPTIONS": json.dumps(
            {"intra_op_num_threads": threads, "inter_op_num_threads": 1}
        ),
        "MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES": json.dumps(
            {
                "session.intra_op.allow_spinning": spin,
                "session.inter_op.allow_spinning": "0",
            },
        ),
    }
    if cudnn:
        e["MLSERVER_MODEL_ONNX_PROVIDER_OPTIONS"] = json.dumps(
            [{"cudnn_conv_algo_search": cudnn}, {}]
        )
    return e


def cpu_env(workers=4, threads=8, spin="0", affinity=None, tuned_spin=None):
    config_entries = {
        "session.intra_op.allow_spinning": spin,
        "session.inter_op.allow_spinning": "0",
    }
    if affinity:
        config_entries["session.intra_op_thread_affinities"] = affinity
    if tuned_spin:
        config_entries["session.intra_op.spin_duration_us"] = "1000"
        config_entries["session.intra_op.spin_backoff_max"] = "8"
    return {
        "MLSERVER_MODEL_ONNX_PROVIDERS": '["CPUExecutionProvider"]',
        "MLSERVER_PARALLEL_WORKERS": str(workers),
        "MLSERVER_REST_WORKER_JSON": "false",
        "MLSERVER_REST_BINARY_TENSOR": "true",
        "MLSERVER_REST_WORKER_BINARY": "false",
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "false",
        "MLSERVER_PARALLEL_INFERENCE_TRANSPORT": "zmq",
        "MLSERVER_MODEL_ONNX_CUDA_GRAPH": "false",
        "MLSERVER_MODEL_ONNX_CUDA_BLOCKING_SYNC": "false",
        "MLSERVER_MODEL_ONNX_SESSION_OPTIONS": json.dumps(
            {"intra_op_num_threads": threads, "inter_op_num_threads": 1}
        ),
        "MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES": json.dumps(config_entries),
    }


def cpu_grpc_env(threads, raw_buffers):
    e = cpu_env(workers=1, threads=threads)
    e["MLSERVER_GRPC_RAW_INPUT_BUFFERS"] = raw_buffers
    return e


variants: dict[str, dict] = {}

# GPU thread axis (REST + gRPC)
for t in (1, 2, 4, 8):
    variants[f"gpu-t{t}"] = {**GPU, "env": gpu_env(threads=t)}
    variants[f"gpu-grpc-t{t}"] = {**GPU, "env": gpu_env(threads=t, grpc_raw="true")}
# GPU spinning
variants["gpu-spin-off"] = {**GPU, "env": gpu_env(spin="0")}
variants["gpu-spin-on"] = {**GPU, "env": gpu_env(spin="1")}
# GPU cuDNN algorithm search
variants["gpu-cudnn-heuristic"] = {**GPU, "env": gpu_env(cudnn="HEURISTIC")}
variants["gpu-cudnn-exhaustive"] = {**GPU, "env": gpu_env(cudnn="EXHAUSTIVE")}
# CPU W x T factorial
for w in (1, 2, 4):
    for t in (1, 2, 4, 8):
        variants[f"cpu-w{w}t{t}"] = {**CPU, "env": cpu_env(workers=w, threads=t)}
# CPU gRPC thread scaling
variants["cpu-grpc-stock-t2"] = {**CPU, "env": cpu_grpc_env(2, "false")}
variants["cpu-grpc-stock-t8"] = {**CPU, "env": cpu_grpc_env(8, "false")}
variants["cpu-grpc-raw-t2"] = {**CPU, "env": cpu_grpc_env(2, "true")}
variants["cpu-grpc-raw-t8"] = {**CPU, "env": cpu_grpc_env(8, "true")}
# CPU spinning
variants["cpu-spin-off"] = {**CPU, "env": cpu_env(workers=4, threads=8, spin="0")}
variants["cpu-spin-on"] = {**CPU, "env": cpu_env(workers=4, threads=8, spin="1")}
# CPU affinity (1-based IDs, count = T-1)
variants["cpu-affinity-os"] = {**CPU, "env": cpu_env(workers=1, threads=8)}
variants["cpu-affinity-pinned"] = {
    **CPU,
    "env": cpu_env(workers=1, threads=8, affinity="1;2;3;4;5;6;7"),
}
# CPU spin-parameter tuning
variants["cpu-spin-default"] = {**CPU, "env": cpu_env(workers=4, threads=8, spin="1")}
variants["cpu-spin-tuned"] = {
    **CPU,
    "env": cpu_env(workers=4, threads=8, spin="1", tuned_spin=True),
}
# CPU C1 latency
variants["cpu-latency-t2"] = {**CPU, "env": cpu_env(workers=1, threads=2)}
variants["cpu-latency-t8"] = {**CPU, "env": cpu_env(workers=1, threads=8)}
# CUDA graph replay A/B (campaign wb-A5 + gpu-binary-graph: parity)
variants["gpu-binary-worker"] = {**GPU, "env": gpu_env(binary_worker=True)}
variants["gpu-binary-worker-graph"] = {
    **GPU,
    "env": gpu_env(binary_worker=True, cuda_graph=True),
}

GPU_URL = "http://mlserver-rfdetr.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer"
GPU_TARGET = "mlserver-rfdetr.mlserver-perf.svc:8081"
CPU_URL = (
    "http://mlserver-rfdetr-cpu.mlserver-perf.svc:8080/v2/models/rfdetr-small/infer"
)
CPU_TARGET = "mlserver-rfdetr-cpu.mlserver-perf.svc:8081"

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
CPU_BINARY_SPEC = {
    "protocol": "rest",
    "url": CPU_URL,
    "payloads": ["/store/payloads/triton-binary-rest.bin"],
    "headers": {
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "225",
    },
    "min_response_bytes": 100000,
}


def grpc_spec(target):
    return {
        "protocol": "grpc",
        "target": target,
        "payloads": ["/store/payloads/g-0.bin"],
        "min_response_bytes": 100000,
    }


GPU_GRPC_SPEC = grpc_spec(GPU_TARGET)
CPU_GRPC_SPEC = grpc_spec(CPU_TARGET)


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


plans = {
    "gpu-thread-sweep": plan(
        "gpu-thread-sweep",
        [entry(f"gpu-t{t}", f"gpu-t{t}", GPU_BINARY_SPEC) for t in (1, 2, 4, 8)]
        + [
            entry(f"gpu-grpc-t{t}", f"gpu-grpc-t{t}", GPU_GRPC_SPEC)
            for t in (1, 2, 4, 8)
        ],
        [8],
    ),
    "gpu-spinning": plan(
        "gpu-spinning",
        [
            entry("gpu-spin-off", "gpu-spin-off", GPU_BINARY_SPEC),
            entry("gpu-spin-on", "gpu-spin-on", GPU_BINARY_SPEC),
        ],
        [1, 8],
    ),
    "gpu-cudnn": plan(
        "gpu-cudnn",
        [
            entry("gpu-cudnn-heuristic", "gpu-cudnn-heuristic", GPU_BINARY_SPEC),
            entry("gpu-cudnn-exhaustive", "gpu-cudnn-exhaustive", GPU_BINARY_SPEC),
        ],
        [8],
    ),
    "cpu-thread-factorial": plan(
        "cpu-thread-factorial",
        [
            entry(f"cpu-w{w}t{t}", f"cpu-w{w}t{t}", CPU_BINARY_SPEC)
            for w in (1, 2, 4)
            for t in (1, 2, 4, 8)
        ],
        [8],
    ),
    "cpu-grpc-thread-scaling": plan(
        "cpu-grpc-thread-scaling",
        [
            entry("cpu-grpc-stock-t2", "cpu-grpc-stock-t2", CPU_GRPC_SPEC),
            entry("cpu-grpc-stock-t8", "cpu-grpc-stock-t8", CPU_GRPC_SPEC),
            entry("cpu-grpc-raw-t2", "cpu-grpc-raw-t2", CPU_GRPC_SPEC),
            entry("cpu-grpc-raw-t8", "cpu-grpc-raw-t8", CPU_GRPC_SPEC),
        ],
        [8],
    ),
    "cpu-spinning": plan(
        "cpu-spinning",
        [
            entry("cpu-spin-off", "cpu-spin-off", CPU_BINARY_SPEC),
            entry("cpu-spin-on", "cpu-spin-on", CPU_BINARY_SPEC),
        ],
        [8],
    ),
    "cpu-affinity": plan(
        "cpu-affinity",
        [
            entry("cpu-affinity-os", "cpu-affinity-os", CPU_BINARY_SPEC),
            entry("cpu-affinity-pinned", "cpu-affinity-pinned", CPU_BINARY_SPEC),
        ],
        [8],
    ),
    "cpu-spin-tuning": plan(
        "cpu-spin-tuning",
        [
            entry("cpu-spin-default", "cpu-spin-default", CPU_BINARY_SPEC),
            entry("cpu-spin-tuned", "cpu-spin-tuned", CPU_BINARY_SPEC),
        ],
        [8],
    ),
    "gpu-cuda-graph": plan(
        "gpu-cuda-graph",
        [
            entry("gpu-binary-worker", "gpu-binary-worker", GPU_BINARY_SPEC),
            entry(
                "gpu-binary-worker-graph",
                "gpu-binary-worker-graph",
                GPU_BINARY_SPEC,
            ),
        ],
        [1, 8],
    ),
    "cpu-latency-c1": plan(
        "cpu-latency-c1",
        [
            entry("cpu-latency-t2", "cpu-latency-t2", CPU_BINARY_SPEC),
            entry("cpu-latency-t8", "cpu-latency-t8", CPU_BINARY_SPEC),
        ],
        [1],
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
        f"threading: {len(variants)} variants, {len(plans)} plans, " f"{total} trials"
    )
    for name, p in plans.items():
        n = len(p["variants"]) * len(p["concurrency"]) * p["rounds"]
        print(f"  {name}: {n} trials")


if __name__ == "__main__":
    main()
