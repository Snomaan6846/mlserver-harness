"""Ceiling probe: F-02 confirm-or-kill measurements.

Measures the frontend/IPC ceiling hypothesis directly:

  1. workers (0/2/4/8) x concurrency (4/8/16/32) grid — does the
     4-worker plateau stay flat (~1.4k rps) as C grows, or track C?
  2. httptools vs h11 — the stock image lacks httptools (uvicorn
     installed without [standard]); an overlay image with httptools
     pip-installed measures the HTTP-parser axis at the 4-worker C8
     cell.

Writes results/ceiling-probe/ceiling.json and prints a table.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench import loadgen  # noqa: E402
from bench.docker_server import (  # noqa: E402
    MLServerContainer,
    detect_default_image,
    probe_image_import,
    resolve_digest,
)
from bench.suites import RestJsonSender, build_request  # noqa: E402

MODEL_DIR = REPO / "models-zoo" / "sklearn-grad-boost-small"
SECONDS = 8
WARMUP = 3


def measure(model: str, request: dict, concurrency: int) -> dict:
    """One closed-loop measurement + client CPU accounting.

    client_cpu_percent is the load-generator process's average CPU
    over the window: near 100% on a single-process asyncio client
    means the CLIENT may be the bottleneck and the server number is
    a lower bound.
    """
    import psutil

    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime the counter

    async def go():
        sender = RestJsonSender(model, request)
        try:
            return await loadgen.closed_loop(concurrency, SECONDS, WARMUP, sender.send)
        finally:
            await sender.close()

    summary = asyncio.run(go()).summary()
    summary["client_cpu_percent"] = round(proc.cpu_percent(interval=None), 1)
    return summary


def main() -> int:
    settings = json.loads((MODEL_DIR / "model-settings.json").read_text())
    model = settings["name"]
    batch = dict(np.load(MODEL_DIR / "validation.npz"))
    request = build_request(settings, batch)
    image = detect_default_image("cpu", "")

    results: dict[str, dict] = {"grid": {}, "httptools": {}}

    # ---- 1. workers x concurrency grid (stock image) ----------------
    print(f"[ceiling] grid: model={model} ({SECONDS}s windows)")
    print(f"{'':10s}", end="")
    for c in (4, 8, 16, 32):
        print(f"{'C' + str(c):>9s}", end="")
    print()
    for workers in (0, 2, 4, 8):
        print(f"workers={workers:<3d}", end="")
        for concurrency in (4, 8, 16, 32):
            with MLServerContainer(
                name=f"ceiling-w{workers}",
                image=image,
                model_repo=MODEL_DIR.parent,
                target="cpu",
                parallel_workers=workers if workers > 0 else 0,
                env={},
            ):
                summary = measure(model, request, concurrency)
            rps = summary["throughput_rps"]
            results["grid"][f"w{workers}_c{concurrency}"] = round(rps, 1)
            print(f"{rps:>9.0f}", end="")
        print()

    # ---- 2. httptools vs h11 at the 4-worker C8 cell -----------------
    print("\n[ceiling] httptools axis (4 workers, C8)")
    stock = resolve_digest(image)
    if not probe_image_import(stock, "httptools"):
        print(
            "  stock image: httptools NOT installed (h11 fallback) — building overlay"
        )
        import subprocess

        build = subprocess.run(
            [
                "podman",
                "build",
                "-t",
                "mlserver-bench-httptools:latest",
                "-f",
                "-",
                str(REPO.parent),
            ],
            input=f"FROM {stock}\nRUN pip install --no-cache-dir httptools\n",
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build.returncode != 0:
            print(f"  overlay build failed: {build.stderr[-300:]}")
            return 1
        overlay = "localhost/mlserver-bench-httptools:latest"
        if not probe_image_import(overlay, "httptools"):
            print("  overlay still lacks httptools?!")
            return 1
    else:
        overlay = stock

    # Interleaved A/B/A/B (3 pairs, alternating order) — the first
    # measurement was a single fixed-order sample. Also measures
    # IN-PROCESS with httptools so the residual-vs-in-process claim
    # has a like-for-like baseline (the original compared 4w+httptools
    # against 0w+h11, which is not a valid residual).
    httptools_rates: dict[str, list[float]] = {"h11-stock": [], "httptools-overlay": []}
    client_cpu: dict[str, list[float]] = {"h11-stock": [], "httptools-overlay": []}
    for repeat in range(3):
        order = (
            (
                ("h11-stock", stock, 4),
                ("httptools-overlay", overlay, 4),
            )
            if repeat % 2 == 0
            else (
                ("httptools-overlay", overlay, 4),
                ("h11-stock", stock, 4),
            )
        )
        for label, img, workers in order:
            with MLServerContainer(
                name=f"ceiling-http-{label[:4]}",
                image=img,
                model_repo=MODEL_DIR.parent,
                target="cpu",
                parallel_workers=workers,
            ):
                summary = measure(model, request, 8)
            httptools_rates[label].append(summary["throughput_rps"])
            client_cpu[label].append(summary["client_cpu_percent"])
            print(
                f"  [{repeat + 1}] {label:20s} (4w): "
                f"{summary['throughput_rps']:.0f} rps "
                f"(client cpu {summary['client_cpu_percent']:.0f}%)"
            )
    # In-process with httptools (like-for-like residual baseline),
    # 3 repeats (review: the IPC share rested on n=2)
    inproc_rates: list[float] = []
    inproc_cpu: list[float] = []
    for repeat in range(3):
        with MLServerContainer(
            name="ceiling-inproc-httptools",
            image=overlay,
            model_repo=MODEL_DIR.parent,
            target="cpu",
            parallel_workers=0,
        ):
            summary = measure(model, request, 8)
        inproc_rates.append(summary["throughput_rps"])
        inproc_cpu.append(summary["client_cpu_percent"])
        print(
            f"  [{repeat + 1}] httptools in-process (0w): "
            f"{summary['throughput_rps']:.0f} rps "
            f"(client cpu {summary['client_cpu_percent']:.0f}%)"
        )

    results["httptools"] = {
        label: {
            "rates": [round(r, 1) for r in rates],
            "mean": round(sum(rates) / len(rates), 1),
        }
        for label, rates in httptools_rates.items()
    }
    results["httptools"]["inprocess_httptools"] = {
        "rates": [round(r, 1) for r in inproc_rates],
        "mean": round(sum(inproc_rates) / len(inproc_rates), 1),
    }
    results["client_cpu_measurements"] = {
        "note": (
            "load-generator process CPU% per measurement; near 100% "
            "means the single-process asyncio client may be the "
            "bottleneck and server numbers are lower bounds"
        ),
        "worker_mode_4w": {label: values for label, values in client_cpu.items()},
        "inprocess_httptools": inproc_cpu,
    }
    results["note"] = (
        "client_cpu_percent is recorded per measurement in "
        "client_cpu_measurements; near-100% means the single-process "
        "asyncio client may be the bottleneck (server numbers are "
        "lower bounds)"
    )

    out = REPO / "results" / "ceiling-probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[ceiling] written: {out / 'ceiling.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
