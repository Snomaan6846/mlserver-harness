"""Count the FULL local-engine validation matrix: every case the repo
can generate for localhost, per profile, with time estimates.

Mirrors each suite's case-generation logic statically (no servers).
Counts assume the published odh-master image: binary-tensor variants
and fork-only framework axes record SKIPPED cases (still counted —
they are part of a complete run's output).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Zoo model counts per (framework, tier)
MODELS = {
    "sklearn": {"variants": 3},  # pipeline-logreg, random-forest, grad-boost
    "xgboost": {"variants": 2},  # classifier, regressor
    "lightgbm": {"variants": 2},  # classifier, regressor
    "onnx": {"variants": 3},  # mlp, cnn, nan
}
TIERS = 3
FRAMEWORKS_CPU = ["sklearn", "xgboost", "lightgbm", "onnx"]

PROFILES = {
    "quick": {
        "concurrency": [1, 8],
        "qps_fractions": [0.75],
        "batch_sizes": [1],
        "warmup_s": 5,
        "window_s": 10,
        "batching_sizes": [1, 16],
        "batching_times": [0.0, 0.01],
        "workers_ladder": [0, 4],
        "framework_intra": [1, 2, 4],
        "hw_batch_ladder": [1, 32],
    },
    "standard": {
        "concurrency": [1, 2, 4, 8, 16, 32, 64],
        "qps_fractions": [0.5, 0.9, 1.1],
        "batch_sizes": [1, 32],
        "warmup_s": 15,
        "window_s": 30,
        "batching_sizes": [1, 16, 64],
        "batching_times": [0.0, 0.005, 0.025],
        "workers_ladder": [0, 1, 2, 4, 8, 16],
        "framework_intra": [1, 2, 4, 8],
        "hw_batch_ladder": [1, 8, 32, 128],
    },
    "exhaustive": {
        "concurrency": [1, 2, 4, 8, 16, 32, 64, 128],
        "qps_fractions": [0.25, 0.5, 0.75, 0.9, 1.0, 1.1],
        "batch_sizes": [1, 8, 32, 128, 512],
        "warmup_s": 30,
        "window_s": 60,
        "batching_sizes": [1, 4, 16, 64, 256],
        "batching_times": [0.0, 0.001, 0.005, 0.01, 0.025],
        "workers_ladder": [0, 1, 2, 4, 8, 16],
        "framework_intra": [1, 2, 4, 8],
        "hw_batch_ladder": [1, 8, 32, 128],
    },
}

# Per-case wall time (measured from the validation runs):
#   server start + parity gate + warmup + window + overhead
CASE_SECONDS = {
    "quick": 40,
    "standard": 75,
    "exhaustive": 130,
}
# Cases that skip (no load): image-probe skips, documented skips
SKIP_CASE_SECONDS = 8


def suite_cases(suite, framework, m, profile):
    """Cases one suite generates for one (framework, tier) with m models."""
    p = PROFILES[profile]
    if suite == "protocol":
        return m * min(4, len(p["concurrency"]))
    if suite == "batching":
        return len(p["batching_sizes"]) * len(p["batching_times"])
    if suite == "workers":
        return m * (len(p["workers_ladder"]) + 1)  # + warm_workers note
    if suite == "caching":
        return 3  # skip-note + repeat + unique
    if suite == "multimodel":
        return 3  # counts 2/5/10
    if suite == "lifecycle":
        return 4  # classic + eager + lazy + comparison
    if suite == "saturation":
        return m * (len(p["concurrency"]) + len(p["qps_fractions"]) * 2)
    if suite == "processing":
        # 4 variants x JSON + 2 binary (skipped on this image) + 2 transports
        return m * (4 + 2 + 2)
    if suite == "framework":
        if framework != "onnx":
            return 1  # skip note
        return len(p["framework_intra"]) + 7  # axes (4 measured + 3 skipped)
    if suite == "middleware":
        return 10  # 5 axes x 2 states
    if suite == "codec":
        base = 2  # np + str-skip
        if framework == "sklearn":
            base += 1 + 3 + 2  # pd + dtype(2+cmp) + predict_fn(2)
        elif framework == "onnx":
            base += 3 + 2  # dtype(2+cmp)... nan needs nan+clean+cmp
        else:
            base += 3  # dtype pair + cmp
        return base
    if suite == "grpc":
        return 6  # rest, typed, raw x2, maxlen x2 (1 model)
    if suite == "faults":
        return 8  # 4 shape + parity + poison + 2 kill
    if suite == "hardware":
        return 1  # SKIPPED_NEEDS_GPU_TARGET (no GPU on this host)
    return 0


SUITES = [
    "protocol",
    "batching",
    "workers",
    "caching",
    "multimodel",
    "lifecycle",
    "saturation",
    "processing",
    "framework",
    "middleware",
    "codec",
    "grpc",
    "faults",
    "hardware",
]


def count(profile, rounds):
    total_cases = 0
    total_seconds = 0.0
    per_framework = {}
    for fw in FRAMEWORKS_CPU:
        m = MODELS[fw]["variants"]
        fw_cases = 0
        fw_seconds = 0.0
        for suite in SUITES:
            cases = suite_cases(suite, fw, m, profile)
            fw_cases += cases
            # estimate: skipped fractions are lighter
            load_frac = 0.85  # most cases run loads
            fw_seconds += cases * (
                load_frac * CASE_SECONDS[profile] + (1 - load_frac) * SKIP_CASE_SECONDS
            )
        per_framework[fw] = (fw_cases, fw_seconds)
        total_cases += fw_cases
        total_seconds += fw_seconds
    # x3 tiers
    total_cases *= TIERS
    total_seconds *= TIERS
    # x rounds
    total_cases *= rounds
    total_seconds *= rounds
    return total_cases, total_seconds, per_framework


def fmt_hours(seconds):
    hours = seconds / 3600
    if hours >= 24:
        return f"{hours:.0f}h ({hours/24:.1f} days)"
    return f"{hours:.0f}h"


AUX_TOOLS = [
    # (name, runs, minutes each)
    ("diagnose (4 frameworks x 3 tiers)", 12, 12),
    ("reference: ORT floor (onnx models)", 9, 4),
    ("reference: Triton ceiling (GPU)", 0, 0),  # NOT locally runnable
    ("crosscheck (30 models, both orders)", 30, 2),
    ("ceiling-probe (grid + httptools A/B)", 1, 40),
    ("smoke-wire", 1, 5),
    ("mutation-check (21 mutations)", 1, 12),
    ("zoo regenerate + verify (32 models)", 1, 8),
    ("make test (229 tests)", 1, 1),
    ("make dry-run", 1, 2),
    ("compare end-to-end on real results", 2, 1),
]


def main():
    print("=" * 74)
    print("FULL LOCAL-ENGINE VALIDATION MATRIX")
    print("=" * 74)
    for profile in ("quick", "standard", "exhaustive"):
        for rounds in (1, 5):
            cases, seconds, per_fw = count(profile, rounds)
            print(
                f"\n{profile:10s} x {rounds} round(s): {cases:,} cases, "
                f"{fmt_hours(seconds)}"
            )
            for fw, (fc, fs) in per_fw.items():
                print(f"    {fw:10s}: {fc * TIERS * rounds:,} cases")

    print("\n" + "=" * 74)
    print("AUXILIARY LOCAL TOOLS (once each)")
    print("=" * 74)
    aux_minutes = 0
    for name, runs, minutes in AUX_TOOLS:
        if runs == 0:
            print(f"  {name:48s} EXCLUDED (needs GPU -> cluster)")
            continue
        aux_minutes += runs * minutes
        print(f"  {name:48s} {runs} x {minutes}min = {runs*minutes}min")
    print(f"  {'TOTAL':48s} {aux_minutes}min " f"({fmt_hours(aux_minutes*60)})")

    print("\n" + "=" * 74)
    print("RECOMMENDED VALIDATION (reviewer conditions: standard, 5 rounds)")
    print("=" * 74)
    cases, seconds, _ = count("standard", 5)
    aux_s = aux_minutes * 60
    total_s = seconds + aux_s
    print(f"  benchmark suites: {cases:,} cases -> {fmt_hours(seconds)}")
    print(f"  auxiliary tools:                -> {fmt_hours(aux_s)}")
    print(f"  TOTAL:                          -> {fmt_hours(total_s)}")
    print(
        f"\n  (quick x 5 rounds alternative: "
        f"{count('quick', 5)[0]:,} cases, {fmt_hours(count('quick', 5)[1])})"
    )

    print("\n" + "=" * 74)
    print("NOT LOCALLY RUNNABLE (cluster-only; this host has no GPU)")
    print("=" * 74)
    for item in [
        "benchmark --target gpu (GPU image, all GPU suites)",
        "hardware suite GPU arm (CPU-vs-GPU matrix)",
        "reference Triton ceiling (needs GPU)",
        "diagnose GPU layer (py-spy/nvidia-smi on CUDA containers)",
        "framework suite GPU axes (cuda_graph, cudnn, blocking_sync)",
        "CUDA-marked tests (auto-skip on CPU-only hosts)",
        "ISVC/OpenShift engine (tools/) — cluster by definition",
        "crosscheck with GPU-tuned concurrency",
    ]:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
