"""Interactive benchmark wizard: clean prompts, numbered options,
sensible defaults on Enter, strict input validation, confirmation gate."""

import sys
from pathlib import Path

from .config import FRAMEWORKS_CPU, SUITES, TIERS, RunConfig


def prompt(message: str) -> str:
    return input(message).strip()


def choose(message: str, options: list[str], default: int = 1) -> int:
    """Numbered choice; Enter selects the default."""
    for index, option in enumerate(options, start=1):
        marker = "*" if index == default else " "
        print(f"  {marker} {index}) {option}")
    while True:
        answer = prompt(f"{message} [{default}]: ")
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        print(f"  ! enter a number between 1 and {len(options)}")


def choose_many(message: str, options: list[str], default: str) -> list[str]:
    """Comma-separated selection or a single number; Enter = default;
    'all' selects every option."""
    print(f"  options: {', '.join(options)}")
    while True:
        answer = prompt(f"{message} [{default}]: ")
        if not answer:
            answer = default
        if answer == "all":
            return list(options)
        if answer.isdigit():  # single index
            index = int(answer)
            if 1 <= index <= len(options):
                return [options[index - 1]]
        selected = [part.strip() for part in answer.split(",") if part.strip()]
        if selected and all(item in options for item in selected):
            return selected
        print("  ! use comma-separated names from the list, or 'all'")


def ask_image(target: str, detected: str) -> str:
    answer = prompt(f"  Docker image for {target} [{detected}]: ")
    return answer or detected


def run_wizard() -> RunConfig | None:
    print("=" * 64)
    print(" MLServer Benchmark Wizard")
    print("=" * 64)

    # 1. Target hardware
    target_index = choose(
        "Target hardware",
        ["CPU", "GPU", "Both (CPU then GPU)"],
    )
    targets = [["cpu"], ["gpu"], ["cpu", "gpu"]][target_index - 1]

    # 2. Frameworks (filtered by target)
    frameworks: list[str] = []
    for target in targets:
        if target == "cpu":
            if len(targets) == 1:
                index = choose(
                    "CPU frameworks",
                    [
                        "All CPU frameworks (sklearn, xgboost, lightgbm, onnx)",
                        "sklearn",
                        "xgboost",
                        "lightgbm",
                        "onnx",
                    ],
                )
                frameworks += (
                    list(FRAMEWORKS_CPU) if index == 1 else [FRAMEWORKS_CPU[index - 2]]
                )
            else:
                frameworks += list(FRAMEWORKS_CPU)
                print("  (both targets: all CPU frameworks selected)")
        else:
            frameworks.append("onnx")
    frameworks = list(dict.fromkeys(frameworks))  # dedupe, keep order

    # 3. Suites
    suites = choose_many(
        "Benchmark suites (comma-separated or 'all')",
        SUITES,
        default="all",
    )
    if suites == ["all"]:
        suites = list(SUITES)

    # 4. Model tiers
    tier_index = choose(
        "Model complexity tiers",
        ["All tiers (small, medium, large)", "small", "medium", "large"],
    )
    tiers = list(TIERS) if tier_index == 1 else [TIERS[tier_index - 2]]

    # 5. Load profile
    profile_index = choose(
        "Load profile",
        [
            "Quick Sanity (short warmup + fast sweep)",
            "Standard (balanced sweeps + stability windows)",
            "Exhaustive (full multi-window sweep, all matrices)",
            "Custom (specify concurrency, QPS, warmup, window)",
        ],
        default=2,
    )
    profiles = ["quick", "standard", "exhaustive", "custom"]
    load_profile = profiles[profile_index - 1]
    custom: dict = {}
    if load_profile == "custom":
        custom["concurrency"] = _int_list(
            prompt("  concurrency levels [1,8,32]: ") or "1,8,32"
        )
        custom["qps_fractions"] = [
            float(v)
            for v in (prompt("  QPS fractions of peak [0.5,1.1]: ") or "0.5,1.1").split(
                ","
            )
        ]
        custom["warmup_seconds"] = int(prompt("  warmup seconds [10]: ") or "10")
        custom["window_seconds"] = int(
            prompt("  measurement window seconds [30]: ") or "30"
        )

    # 6. Server & run options
    print("\nServer & run options:")
    images: dict[str, str] = {}
    from .docker_server import detect_default_image

    for target in dict.fromkeys(["cpu", "gpu"] if len(targets) > 1 else targets):
        detected = detect_default_image(target, "")
        images[target] = ask_image(target, detected)
    image = images[targets[0]]
    if len(targets) > 1:
        print(f"  (CPU image: {images['cpu']}; GPU image: {images['gpu']})")

    warm_index = choose(
        "warm_workers (deprecated no-op in MLServer — kept for schema)",
        ["true (default)", "false"],
    )
    warm_workers = warm_index == 1

    cache_index = choose(
        "cache_enabled (MLServer has no response cache yet — the suite "
        "probes and skips, measuring repeat-vs-unique distributions)",
        ["false (default)", "true", "both (A/B comparison)"],
    )
    cache_enabled = ["false", "true", "both"][cache_index - 1]

    real_index = choose(
        "Include pinned downloadable reference models (ONNX Model Zoo)",
        ["no (synthetic zoo only, default)", "yes (adds mnist-8 real model)"],
    )
    include_real = real_index == 2

    baseline = (
        prompt("Baseline result path for regression comparison (Enter = none): ")
        or None
    )

    config = RunConfig(
        target=targets[0],
        frameworks=frameworks,
        suites=suites,
        tiers=tiers,
        load_profile=load_profile,
        custom=custom,
        image=image,
        warm_workers=warm_workers,
        cache_enabled=cache_enabled,
        include_real=include_real,
        baseline=baseline,
    )
    problems = config.validate()
    if problems:
        print(f"\n! configuration invalid: {problems}")
        return None

    # 7. Confirmation gate
    cases = _estimate_matrix(config)
    print("\n" + "-" * 64)
    print(" EXECUTION PLAN SUMMARY")
    print("-" * 64)
    print(f"  Target hardware : {targets}")
    print(f"  Docker image(s) : {images}")
    print(f"  Frameworks      : {frameworks}")
    print(f"  Suites          : {suites}")
    print(f"  Model tiers     : {tiers}")
    print(f"  Load profile    : {load_profile}")
    print(
        f"  Server settings : warm_workers={warm_workers} "
        f"cache_enabled={cache_enabled} real_models={include_real}"
    )
    print("  Output          : results/<target>-<timestamp>/")
    print(f"  Estimated cases : ~{cases}")
    if baseline:
        print(f"  Baseline        : {baseline}")
    print("-" * 64)

    answer = prompt("Confirm and proceed with benchmark run? [y/N]: ")
    if answer.lower() != "y":
        print("Cancelled — nothing started.")
        return None
    return config


def _int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def _estimate_matrix(config: RunConfig) -> int:
    profile = config.profile()
    per_model = (
        len(profile["concurrency"])
        + len(profile["qps_fractions"]) * 2
        + 3 * len(config.suites)
    )
    models = len(config.frameworks) * len(config.tiers) * 2
    return per_model * models


if __name__ == "__main__":
    config = run_wizard()
    if config:
        from .runner import execute

        repo_root = Path(__file__).resolve().parents[1]
        output = execute(config, repo_root)
        print(f"\nresults: {output}")
        sys.exit(0)
    sys.exit(0)
