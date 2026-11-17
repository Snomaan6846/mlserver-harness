"""Run orchestration: manifest -> zoo -> parity -> suites -> results.

Determinism (MLPerf closed-division standard): pinned seeds, SHA-verified
models/payloads, and a complete environment manifest embedded in every
result — git SHAs, image digests, package versions, hardware.
"""

import asyncio
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import suites, zoo
from .config import RunConfig, generate_run_id
from .docker_server import clean_stale_containers, detect_default_image
from .parity import ParityError


def _git_sha(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module in (
        "mlserver",
        "sklearn",
        "xgboost",
        "lightgbm",
        "onnx",
        "onnxruntime",
        "numpy",
        "grpcio",
        "pandas",
    ):
        try:
            imported = __import__(module)
            versions[module] = getattr(imported, "__version__", "unknown")
        except ImportError:
            versions[module] = "not-installed"
    return versions


def _image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else image


def environment_manifest(config: RunConfig, repo_root: Path) -> dict[str, Any]:
    import psutil

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": {
            "mlserver-benchmarks": _git_sha(repo_root),
            "mlserver-source": _git_sha(repo_root.parent / "MLServer"),
        },
        "image": {
            "reference": config.image,
            "digest": _image_digest(config.image),
            "target": config.target,
        },
        "packages": _package_versions(),
        "python": platform.python_version(),
        "host": {
            "cpu_model": _cpu_model(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "platform": platform.platform(),
        },
        "seeds": {"zoo": zoo.SEED},
        "run_config": config.to_dict(),
    }
    gpu = _gpu_info()
    if gpu:
        manifest["gpu"] = gpu
    return manifest


def _cpu_model() -> str:
    try:
        return subprocess.run(
            ["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def _gpu_info() -> dict[str, Any] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "name": pynvml.nvmlDeviceGetName(handle),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "cuda": pynvml.nvmlSystemGetCudaDriverVersion(),
            "vram_gb": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1e9, 1),
        }
    except Exception:
        return None


class RunAborted(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status


async def _close_sender_sessions() -> None:
    """Close shared sender resources at end of run.

    Sender sessions are per-case (loop-bound, closed in run_closed);
    nothing shared remains, but the hook stays for future resources.
    """


def _run_one_suite(
    suite_name: str,
    suite_fn: Any,
    context: Any,
    model_dirs: Any,
    framework: str,
    tier: str,
    results: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]] | None:
    """Run one suite with failure containment.

    Returns the suite's cases, or None when the suite failed (the
    failure is recorded in `results` and results.json is updated —
    the run continues with the remaining suites).

    Correctness-gate failures (ParityError, RunAborted) propagate:
    they invalidate every result, so the whole run must abort. The
    old inline handler caught only ServerError — a single
    RuntimeError (e.g. the loadgen error guard on a 100%-error case)
    killed the entire run.
    """
    try:
        return asyncio.run(suite_fn(context, model_dirs))
    except (ParityError, RunAborted):
        raise
    except Exception as error:
        print(
            f"[run] suite {suite_name} [{framework}/{tier}] "
            f"FAILED: {type(error).__name__}: {str(error)[:300]}"
        )
        results.append(
            {
                "suite": suite_name,
                "framework": framework,
                "tier": tier,
                "status": "SUITE_FAILED",
                "error": type(error).__name__,
                "detail": str(error)[:2000],
            }
        )
        _atomic_write(
            output_dir / "results.json",
            {"status": "RUNNING", "cases": results},
        )
        return None


def execute(config: RunConfig, repo_root: Path) -> Path:
    """Run the full benchmark matrix for a validated config."""
    problems = config.validate()
    if problems:
        raise SystemExit("invalid config: " + "; ".join(problems))

    # The Docker-mode runner only executes locally. ENGINE=cluster
    # must fail fast instead of silently benchmarking local
    # containers (the ISVC path is the OpenShift harness in tools/
    # and the engine adapters in bench/engines/).
    if config.engine not in ("local", "docker", "podman", "container"):
        raise SystemExit(
            f"engine {config.engine!r} is not wired into the Docker-mode "
            "runner. For cluster execution use the OpenShift harness "
            "(tools/run-wire-format et al.) or the ISVC engine adapters "
            "in bench/engines/ — ENGINE=local is the only mode this "
            "runner executes."
        )

    # Image precedence policy (bench/images.py): override wins (tag ->
    # digest, digest -> as-is); otherwise the floating default resolves
    # to the immutable digest at run start. Never benchmark a floating tag.
    from .docker_server import resolve_digest
    from .images import resolve_image

    default = detect_default_image(config.target, "")
    pinned = resolve_image(config.image, default, resolver=resolve_digest)
    if pinned != config.image:
        print(f"[run] image pinned: {config.image} -> {pinned}")
        config.image = pinned
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else repo_root / "results" / f"{config.target}-{stamp}"
    )
    output_dir = Path(output_dir)
    output_dir = output_dir.resolve()  # absolute: -v mounts must not
    # be interpreted as named volumes by docker/podman
    output_dir.mkdir(parents=True, exist_ok=True)
    config.repo_root = str(repo_root)
    # Unique run identity: every container, Kubernetes resource and
    # result artifact from this run carries it — runs are never
    # ambiguous in `docker ps` or on disk, and leaked containers from
    # a crashed run can never collide with a fresh run's names.
    if not config.run_id:
        config.run_id = generate_run_id()
    config.save(output_dir / "config.json")

    print(f"[run] target={config.target} image={config.image}")
    print(f"[run] frameworks={config.frameworks} tiers={config.tiers}")
    print(f"[run] suites={config.suites} profile={config.load_profile}")
    print(f"[run] run-id={config.run_id}")
    print(f"[run] output={output_dir}")

    # 1. Deterministic model zoo: materialize on demand (reuse-if-present,
    #    generate-or-download otherwise) + checksum gate
    zoo_root = zoo.materialize(
        repo_root,
        config.frameworks,
        config.tiers,
        include_real=config.include_real,
    )
    checksum_problems = zoo.verify(zoo_root)
    if checksum_problems:
        raise RunAborted("FAILED_CHECKSUM", "; ".join(checksum_problems[:5]))
    print(f"[run] zoo verified: {len(list(zoo_root.iterdir()))} models")

    # 2. Environment manifest (carries the run id — artifacts are
    #    self-describing even when copied out of their output dir)
    manifest = environment_manifest(config, repo_root)
    manifest["run_id"] = config.run_id
    _atomic_write(output_dir / "manifest.json", manifest)

    # 3. Stale-container cleanup + signal-safe teardown
    stale = clean_stale_containers()
    if stale:
        print(f"[run] removed stale containers: {', '.join(stale)}")

    results: list[dict[str, Any]] = []
    status = "COMPLETE"
    try:
        for index, framework in enumerate(config.frameworks):
            all_models = sorted(
                model
                for tier in config.tiers
                for model in zoo.list_models(zoo_root, framework, tier)
            )
            if not all_models:
                continue
            # Parity gate before ANY measurement: one server per
            # framework, mounted with only that framework's models.
            framework_repo = output_dir / f"repo-{framework}"
            if framework_repo.exists():
                shutil.rmtree(framework_repo)
            framework_repo.mkdir(parents=True)
            for model_dir in all_models:
                shutil.copytree(model_dir, framework_repo / model_dir.name)
            context = suites.SuiteContext(
                config,
                config.image,
                zoo_root,
                output_dir,
                run_index=index,
                suite_name="parity",
                run_id=config.run_id,
            )
            with context.server(framework_repo):
                for model_dir in sorted(framework_repo.iterdir()):
                    if model_dir.is_dir():
                        asyncio.run(context.parity(framework_repo, model_dir))
            print(
                f"[run] parity gate passed: {framework} " f"({len(all_models)} models)"
            )
            for tier in config.tiers:
                model_dirs = zoo.list_models(zoo_root, framework, tier)
                if not model_dirs:
                    continue
                for round_index in range(max(1, config.rounds)):
                    if config.rounds > 1:
                        print(
                            f"[run] round {round_index + 1}/{config.rounds} "
                            f"[{framework}/{tier}]"
                        )
                    for suite_name in config.suites:
                        suite_fn = suites.SUITES[suite_name]
                        context = suites.SuiteContext(
                            config,
                            config.image,
                            zoo_root,
                            output_dir,
                            run_index=index,
                            round_index=round_index,
                            suite_name=suite_name,
                            run_id=config.run_id,
                        )
                        started = time.time()
                        cases = _run_one_suite(
                            suite_name,
                            suite_fn,
                            context,
                            model_dirs,
                            framework,
                            tier,
                            results,
                            output_dir,
                        )
                        if cases is None:
                            continue  # suite failed; recorded + written
                        for case in cases:
                            case["framework"] = framework
                            case["tier"] = tier
                            case["round"] = round_index
                            case["elapsed_s"] = round(time.time() - started, 1)
                        results.extend(cases)
                        _atomic_write(
                            output_dir / "results.json",
                            {"status": "RUNNING", "cases": results},
                        )
                        done = len(cases)
                        print(
                            f"[run] suite {suite_name} [{framework}/{tier}]: "
                            f"{done} cases"
                        )
    except (ParityError, RunAborted) as error:
        status = getattr(error, "status", "FAILED")
        print(f"[run] ABORTED ({status}): {error}")
        results.append(
            {"suite": "abort", "status": status, "detail": str(error)[:2000]}
        )
    finally:
        # Scoped teardown: only THIS run's containers (by run id), so
        # the sweep can never touch a concurrent run's servers. The
        # global sweep at run start handles stale containers from
        # crashed runs (serial-host discipline).
        clean_stale_containers(run_id=config.run_id)
        # Close shared sender sessions (REST + binary tensor).
        asyncio.run(_close_sender_sessions())

    payload = {
        "status": status,
        "manifest": manifest,
        "config": config.to_dict(),
        "cases": results,
        "summary": _summarize(results),
    }
    _atomic_write(output_dir / "results.json", payload)
    print(f"[run] {status}: {output_dir / 'results.json'} " f"({len(results)} cases)")
    return output_dir


def dry_run(config: RunConfig, repo_root: Path) -> Path:
    """Validate images, zoo checksums, boot MLServer, run the parity
    gate, and exit — no load testing."""
    config.suites = []
    config.load_profile = "quick"
    output = execute(config, repo_root)
    return output


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_suite: dict[str, int] = {}
    findings: list[dict[str, Any]] = []
    for case in cases:
        suite = case.get("suite", "?")
        by_suite[suite] = by_suite.get(suite, 0) + 1
        result = case.get("result", {})
        status = result.get("status", "")
        if status == "PARITY_FAILED":
            findings.append(
                {
                    "kind": "PARITY_FAILED",
                    "suite": suite,
                    "model": case.get("model"),
                    "params": case.get("params"),
                    "problems": result.get("problems"),
                }
            )
        elif status.startswith("SKIPPED_"):
            # Informational skips stay out of findings; they are
            # expected behaviour (missing GPU, unsupported image, ...)
            pass
        if result.get("generator_bound"):
            # Client-limited measurement: the number is the generator's
            # ceiling, not a server result — a visible finding, never a
            # silent field.
            findings.append(
                {
                    "kind": "GENERATOR_BOUND",
                    "suite": suite,
                    "model": case.get("model"),
                    "client_cpu_percent": result.get("client_cpu_percent"),
                }
            )
    for case in cases:
        if case.get("status") == "SUITE_FAILED":
            findings.append(
                {
                    "kind": "SUITE_FAILED",
                    "suite": case.get("suite"),
                    "framework": case.get("framework"),
                    "error": case.get("error"),
                    "detail": (case.get("detail") or "")[:300],
                }
            )
    return {
        "total_cases": len(cases),
        "by_suite": by_suite,
        "findings": findings,
        "rounds": _aggregate_rounds(cases),
        "abort": next((c for c in cases if c.get("suite") == "abort"), None),
    }


def _aggregate_rounds(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell aggregates across replicate rounds.

    Groups measured cases by (suite, framework, tier, model, params)
    and reports mean/std for the headline metrics — the honest noise
    estimate for single-vs-multi round comparisons. Returns [] when
    every cell has one replicate.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for case in cases:
        if case.get("round") is None:
            continue
        result = case.get("result", {})
        if result.get("status") or "throughput_rps" not in result:
            continue  # skips/findings have no replicates to aggregate
        key = (
            case.get("suite"),
            case.get("framework"),
            case.get("tier"),
            case.get("model"),
            json.dumps(case.get("params", {}), sort_keys=True),
        )
        groups.setdefault(key, []).append(case)
    aggregates = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        suite, framework, tier, model, params_json = key
        entry: dict[str, Any] = {
            "suite": suite,
            "framework": framework,
            "tier": tier,
            "model": model,
            "params": json.loads(params_json),
            "rounds": len(group),
        }
        for metric in ("throughput_rps", "latency_ms_p50", "latency_ms_p99"):
            values = [
                c["result"][metric]
                for c in group
                if c["result"].get(metric) is not None
            ]
            if len(values) < 2:
                continue
            entry[f"{metric}_mean"] = round(sum(values) / len(values), 3)
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                entry[f"{metric}_std"] = round(variance**0.5, 3)
        aggregates.append(entry)
    return aggregates


def _atomic_write(path: Path, payload: Any) -> None:
    import os

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    """JSON fallback: numpy scalars (bool_, float64, int64, ...) and
    other non-stdlib numerics serialize as their Python equivalents
    instead of crashing the run at write time."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)
