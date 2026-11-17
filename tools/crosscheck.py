"""Cross-check the load generator against an external client.

Runs our asyncio loadgen and ApacheBench (ab) against the SAME server
and configuration, then compares throughputs. A large divergence
indicates a client-side bottleneck (our generator too slow / capped)
rather than server behaviour — the same class of issue the review
flagged for single-client methodology.

Usage:
    python3 tools/crosscheck.py [--concurrency 8] [--seconds 10]
                                [--framework sklearn] [--model-tier small]
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench import loadgen  # noqa: E402
from bench.docker_server import (  # noqa: E402
    MLServerContainer,
    detect_default_image,
)
from bench.suites import RestJsonSender, build_request  # noqa: E402

REST_PORT = 18080


def run_ours(model: str, request: dict, concurrency: int, seconds: int):
    async def go():
        sender = RestJsonSender(model, request)
        try:
            return await loadgen.closed_loop(
                concurrency, seconds, warmup_seconds=2, send=sender.send
            )
        finally:
            await sender.close()

    result = asyncio.run(go())
    return result.summary()


def run_ab(model: str, request: dict, concurrency: int, seconds: int):
    """ApacheBench against the same endpoint; ab drives its own
    request count from the time budget (n = rate estimate x seconds)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as body_file:
        json.dump(request, body_file)
        body_path = body_file.name

    # Estimate a request count well above expected throughput: ab
    # stops when -n is exhausted, so give it plenty.
    requests = max(20000, concurrency * 2000 * seconds)
    url = f"http://127.0.0.1:{REST_PORT}/v2/models/{model}/infer"
    result = subprocess.run(
        [
            "ab",
            "-n",
            str(requests),
            "-c",
            str(concurrency),
            "-k",  # keepalive (our client uses persistent connections)
            "-t",
            str(seconds),
            "-p",
            body_path,
            "-T",
            "application/json",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=seconds * 6 + 60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ab failed: {result.stderr[:300]}")

    match = re.search(r"Requests per second:\s+([\d.]+)", result.stdout)
    rps = float(match.group(1)) if match else 0.0
    failed_match = re.search(r"Non-2xx responses:\s+(\d+)", result.stdout)
    non_2xx = int(failed_match.group(1)) if failed_match else 0
    match_p50 = re.search(r"50%\s+([\d]+)", result.stdout)
    p50_ms = float(match_p50.group(1)) if match_p50 else 0.0
    return {"throughput_rps": rps, "non_2xx": non_2xx, "latency_ms_p50": p50_ms}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--framework", default="sklearn")
    parser.add_argument("--model-tier", default="small")
    args = parser.parse_args()

    if not subprocess.run(["which", "ab"], capture_output=True).stdout.strip():
        print("[crosscheck] ab not installed — SKIPPED")
        return 0

    from bench import zoo

    zoo_root = REPO / "models-zoo"
    model_dirs = zoo.list_models(zoo_root, args.framework, args.model_tier)
    if not model_dirs:
        print(f"[crosscheck] no {args.framework}/{args.model_tier} models")
        return 1
    model_dir = sorted(model_dirs)[0]
    settings = json.loads((model_dir / "model-settings.json").read_text())
    model = settings["name"]
    batch = dict(np.load(model_dir / "validation.npz"))
    request = build_request(settings, batch)

    image = detect_default_image("cpu", "")
    print(
        f"[crosscheck] model={model} concurrency={args.concurrency} "
        f"seconds={args.seconds} repeats={args.repeats}"
    )
    with MLServerContainer(
        name="crosscheck",
        image=image,
        model_repo=model_dir.parent,
        target="cpu",
    ):
        # Both orders (ours-first and ab-first), repeated, to remove
        # warmup-order drift. ab is single-threaded HTTP/1.0-with-
        # keepalive: it is a LOWER bound on client capability — this
        # checks our generator is not slower than a known client, NOT
        # independent validation of absolute accuracy.
        ours_rates: list[float] = []
        ab_rates: list[float] = []
        for repeat in range(args.repeats):
            order = ("ours", "ab") if repeat % 2 == 0 else ("ab", "ours")
            for phase in order:
                if phase == "ours":
                    ours = run_ours(model, request, args.concurrency, args.seconds)
                    ours_rates.append(ours["throughput_rps"])
                    print(
                        f"  [{repeat + 1}] ours: {ours['throughput_rps']:8.0f} rps "
                        f"p50={ours['latency_ms_p50']:.2f} ms "
                        f"errors={ours['failed']}"
                    )
                else:
                    theirs = run_ab(model, request, args.concurrency, args.seconds)
                    ab_rates.append(theirs["throughput_rps"])
                    print(
                        f"  [{repeat + 1}] ab:   {theirs['throughput_rps']:8.0f} rps "
                        f"non-2xx={theirs['non_2xx']}"
                    )

    ours_mean = sum(ours_rates) / len(ours_rates)
    ab_mean = sum(ab_rates) / len(ab_rates)
    ratio = ours_mean / ab_mean if ab_mean else float("inf")
    gap_pct = abs(ratio - 1) * 100
    print()
    print(f"[crosscheck] ours mean = {ours_mean:.0f} rps | ab mean = {ab_mean:.0f} rps")
    print(f"[crosscheck] ratio ours/ab = {ratio:.2f} (gap {gap_pct:.1f}%)")
    print(
        "[crosscheck] interpretation: ours is consistent with ab within "
        f"{gap_pct:.1f}%; ab is single-threaded HTTP/1.0, so ours >= ab "
        "shows our generator is not the slower client — NOT independent "
        "validation (a multi-threaded generator like oha/wrk2 would be "
        "needed for that)"
    )
    if ratio < 0.9:
        print(
            "[crosscheck] FAIL: ours is >10% SLOWER than ab — our "
            "generator is likely the bottleneck; do not trust its "
            "absolute numbers until fixed"
        )
        return 2
    if ratio > 1.11:
        print(
            "[crosscheck] ours is faster than ab by >10% — consistent "
            "with single-threaded ab being the slow side (expected); "
            "inconclusive about our absolute accuracy"
        )
        return 0
    print("[crosscheck] within ±10% tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
