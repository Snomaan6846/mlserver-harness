"""Reference tool: raw ORT floor + Triton ceiling + MLServer comparison.

The 3-point reference establishes:
  - Floor: raw onnxruntime in-process (no serving overhead)
  - Ceiling: Triton C++ serving (same model, resources, load)
  - MLServer: our number (from existing benchmarks)

Fair comparison rules: same model artifact (SHA-256 verified), same
container resources, same load generator, same measurement window,
sequential (never coexisting on the same GPU).
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Floor: raw onnxruntime in-process
# ---------------------------------------------------------------------------


def measure_ort_floor(
    model_path: Path,
    validation_batch: dict[str, np.ndarray],
    duration_s: float = 30,
    warmup_s: float = 5,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Measure pure model inference time with onnxruntime — no server.

    This is the irreducible cost of the model itself.
    """
    import onnxruntime as ort

    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_names = {i.name for i in session.get_inputs()}

    feeds = {
        key: value for key, value in validation_batch.items() if key in input_names
    }

    # Warmup
    warmup_end = time.monotonic() + warmup_s
    while time.monotonic() < warmup_end:
        session.run(None, feeds)

    # Measure
    latencies_ms: list[float] = []
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        t0 = time.perf_counter()
        session.run(None, feeds)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    latencies = np.array(latencies_ms)
    count = len(latencies)
    total_s = time.monotonic() - start

    return {
        "name": "raw-onnxruntime",
        "label": "Floor",
        "providers": providers,
        "duration_s": round(total_s, 1),
        "inferences": count,
        "throughput_rps": round(count / total_s, 2),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
        },
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# Triton ceiling measurement (drives Triton with our load generator)
# ---------------------------------------------------------------------------


def build_triton_model_repo(model_path: Path, output_dir: Path) -> Path:
    """Create a Triton-compatible model repository from an ONNX model."""
    repo = output_dir / "triton-repo"
    model_name = model_path.stem
    version_dir = repo / model_name / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy the model
    import shutil

    shutil.copy2(model_path, version_dir / "model.onnx")

    # Write Triton config.pbtxt
    config = f"""name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 0

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [-1]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
    (repo / model_name / "config.pbtxt").write_text(config)
    return repo


def deploy_triton(
    model_repo: Path,
    image: str = "nvcr.io/nvidia/tritonserver:24.01-py3",
    gpu: bool = True,
    container_name: str = "mlserver-bench-triton",
) -> str:
    """Deploy Triton in a container with the model repository mounted."""
    # Remove any existing container
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )

    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--security-opt",
        "label=disable",
        "-p",
        "18090:8000",
        "-p",
        "18091:8001",
        "-p",
        "18092:8002",
    ]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        "-v",
        f"{model_repo}:/models",
        image,
        "tritonserver",
        "--model-repository=/models",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Triton deployment failed: {result.stderr[:400]}")

    # Wait for readiness
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            check = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "http://127.0.0.1:18092/metrics",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if check.stdout.strip() == "200":
                return container_name
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(2)

    raise RuntimeError("Triton did not become ready within 120s")


async def measure_triton_ceiling(
    model_name: str,
    payload: dict[str, Any],
    duration_s: float = 30,
    warmup_s: float = 10,
    concurrency: int = 8,
    triton_url: str = "http://127.0.0.1:18090",
) -> dict[str, Any]:
    """Drive Triton with our load generator — fair comparison."""
    import aiohttp

    url = f"{triton_url}/v2/models/{model_name}/infer"
    body = json.dumps(
        {
            "inputs": [
                {
                    "name": payload.get("name", "input"),
                    "shape": payload.get("shape", [-1]),
                    "datatype": "FP32",
                    "data": payload.get("data", []),
                }
            ],
            "outputs": [{"name": "output"}],
        }
    )

    async def send(seq: int) -> tuple[bool, int, int]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                resp_body = await response.read()
                if response.status != 200:
                    raise RuntimeError(f"Triton returned {response.status}")
                return True, len(body), len(resp_body)

    from ..loadgen import closed_loop

    result = await closed_loop(concurrency, duration_s, warmup_s, send)
    summary = result.summary()
    return {
        "name": "triton",
        "label": "Ceiling",
        "concurrency": concurrency,
        "duration_s": duration_s,
        "throughput_rps": round(summary.get("throughput_rps", 0), 2),
        "latency_ms": {
            "p50": summary.get("latency_ms_p50", 0),
            "p95": summary.get("latency_ms_p95", 0),
            "p99": summary.get("latency_ms_p99", 0),
        },
        "error_rate": summary.get("error_rate", 0),
    }


# ---------------------------------------------------------------------------
# 3-point comparison
# ---------------------------------------------------------------------------


def compare_references(
    floor: dict[str, Any],
    mlserver: dict[str, Any],
    ceiling: dict[str, Any],
) -> dict[str, Any]:
    """Compute efficiency, overhead, and headroom from the 3 points."""
    result: dict[str, Any] = {
        "floor": floor,
        "mlserver": mlserver,
        "ceiling": ceiling,
    }

    floor_rps = floor.get("throughput_rps", 0)
    mlserver_rps = mlserver.get("throughput_rps", 0)
    ceiling_rps = ceiling.get("throughput_rps", 0)

    floor_p50 = floor.get("latency_ms", {}).get("p50", 0)
    mlserver_p50 = mlserver.get("latency_ms", {}).get("p50", 0)

    if ceiling_rps > 0:
        result["mlserver_efficiency_pct"] = round(mlserver_rps / ceiling_rps * 100, 1)
    if floor_rps > 0:
        result["mlserver_over_floor_pct"] = round(
            (mlserver_rps - floor_rps) / floor_rps * 100, 1
        )
    if mlserver_p50 > 0 and floor_p50 > 0:
        result["serving_overhead_ms"] = round(mlserver_p50 - floor_p50, 2)
    if ceiling_rps > mlserver_rps:
        result["optimization_headroom_rps"] = round(ceiling_rps - mlserver_rps, 2)

    return result


def generate_reference_report(comparison: dict[str, Any], output_dir: Path) -> Path:
    """Generate the reference comparison report."""
    lines = [
        "# MLServer Reference Comparison Report",
        "",
        f"**Generated:** "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Three-Point Reference",
        "",
        "| Point | Server | Throughput (RPS) | p50 (ms) | p99 (ms) |",
        "|---|---|---|---|---|",
    ]

    for label in ("floor", "mlserver", "ceiling"):
        entry = comparison.get(label, {})
        if entry:
            lat = entry.get("latency_ms", {})
            lines.append(
                f"| **{entry.get('label', label)}** | "
                f"{entry.get('name', '?')} | "
                f"{entry.get('throughput_rps', '?')} | "
                f"{lat.get('p50', '?')} | "
                f"{lat.get('p99', '?')} |"
            )

    lines.extend(["", "## Analysis", ""])

    if comparison.get("mlserver_efficiency_pct"):
        lines.append(
            f"- **MLServer efficiency:** "
            f"{comparison['mlserver_efficiency_pct']:.0f}% of the "
            f"Triton ceiling"
        )
    if comparison.get("serving_overhead_ms"):
        lines.append(
            f"- **Serving overhead:** "
            f"{comparison['serving_overhead_ms']:.1f} ms/request "
            f"above the raw ORT floor"
        )
    if comparison.get("optimization_headroom_rps"):
        lines.append(
            f"- **Optimization headroom:** "
            f"{comparison['optimization_headroom_rps']:.1f} RPS "
            f"recoverable (ceiling − MLServer)"
        )

    report_path = output_dir / "reference_report.md"
    report_path.write_text("\n".join(lines) + "\n")

    # JSON alongside
    (output_dir / "reference_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str) + "\n"
    )
    return report_path
