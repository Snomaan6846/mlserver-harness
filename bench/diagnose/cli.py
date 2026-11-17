"""Diagnose + Reference CLI entry points."""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def diagnose_command(args: argparse.Namespace) -> None:
    """Run the full diagnostic suite."""
    from bench.diagnose.collectors import (
        gpu,
        latency,
        memory,
        oncpu,
        process,
        system,
    )
    from bench.diagnose.findings import engine
    from bench.diagnose import report as report_mod

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            REPO_ROOT
            / "results"
            / f"diagnose-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(exist_ok=True)

    container = args.container
    print(f"[diagnose] output: {output_dir}")
    print(f"[diagnose] container: {container or 'local'}")

    data: dict = {}

    # Layer 1: System
    print("[diagnose] Layer 1: System (USE method)...")
    data["system"] = system.collect_all(duration_s=5)
    _save_layer(layers_dir, "01_system.json", data["system"])

    # Layer 2: Process
    print("[diagnose] Layer 2: Process/thread analysis...")
    pids = process.find_server_pids(container)
    if pids:
        data["process"] = process.collect_all(container, duration_s=3)
    else:
        data["process"] = {"error": "no MLServer processes found"}
        print("[diagnose]   WARNING: no MLServer processes found")
    _save_layer(layers_dir, "02_process.json", data["process"])

    # Layer 3: On-CPU profiling
    if pids and not args.skip_profile:
        print("[diagnose] Layer 3: On-CPU profiling (py-spy)...")
        data["oncpu"] = oncpu.collect_all(
            pids[0],
            args.container if args.engine == "docker" else None,
            duration_s=args.profile_duration,
            output_dir=layers_dir,
        )
    else:
        data["oncpu"] = {"error": "skipped or no PID"}
    _save_layer(layers_dir, "03_oncpu.json", data.get("oncpu", {}))

    # Layer 4: Memory
    print("[diagnose] Layer 4: Memory profiling...")
    data["memory"] = memory.collect_all(
        args.container if args.engine == "docker" else None,
        pid=pids[0] if pids else None,
        duration_s=0 if args.skip_profile else args.profile_duration,
    )
    _save_layer(layers_dir, "04_memory.json", data["memory"])

    # Layer 5: Latency decomposition
    print("[diagnose] Layer 5: Latency decomposition...")
    data["latency"] = latency.collect_all(
        metrics_port=args.metrics_port,
    )
    _save_layer(layers_dir, "05_latency.json", data["latency"])

    # Layer 6: GPU
    print("[diagnose] Layer 6: GPU analysis...")
    if gpu.gpu_available():
        data["gpu"] = gpu.collect_all(duration_s=10)
    else:
        data["gpu"] = {"error": "no GPU available"}
        print("[diagnose]   no GPU on this host")
    _save_layer(layers_dir, "06_gpu.json", data["gpu"])

    # Findings
    print("[diagnose] Running findings engine...")
    findings = engine.evaluate(data)
    findings_dir = output_dir / "findings"
    findings_dir.mkdir(exist_ok=True)
    (findings_dir / "findings.json").write_text(
        json.dumps(findings, indent=2, default=str) + "\n"
    )

    # Report
    print("[diagnose] Generating DIAGNOSIS.md...")
    report_path = report_mod.generate_report(data, findings, output_dir)

    print("\n[diagnose] COMPLETE")
    print(f"  report: {report_path}")
    print(
        f"  findings: {findings['total']} "
        f"({findings['by_severity'].get('CRITICAL', 0)} critical, "
        f"{findings['by_severity'].get('HIGH', 0)} high, "
        f"{findings['by_severity'].get('MEDIUM', 0)} medium)"
    )


def reference_command(args: argparse.Namespace) -> None:
    """Run the 3-point reference measurement."""
    from bench.reference.measure import (
        compare_references,
        generate_reference_report,
        measure_ort_floor,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            REPO_ROOT
            / "results"
            / f"reference-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model path from the zoo
    from bench import zoo

    zoo_root = zoo.materialize(REPO_ROOT, ["onnx"], [args.tier or "small"])
    model_dir = zoo_root / f"onnx-mlp-{args.tier or 'small'}"
    if not model_dir.is_dir():
        model_dir = zoo_root / f"onnx-cnn-{args.tier or 'small'}"

    model_path = model_dir / "model.onnx"
    if not model_path.is_file():
        sys.exit(f"model not found: {model_path}")

    batch = dict(__import__("numpy").load(model_dir / "validation.npz"))

    # 1. Floor: raw ORT
    print("[reference] Measuring floor (raw onnxruntime)...")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.gpu
        else ["CPUExecutionProvider"]
    )
    floor = measure_ort_floor(
        model_path,
        batch,
        duration_s=args.duration,
        providers=providers,
    )
    print(
        f"  floor: {floor['throughput_rps']} RPS, "
        f"p50 {floor['latency_ms']['p50']}ms"
    )

    # 2. MLServer (from existing results or live measurement)
    mlserver = args.mlserver_results
    if not mlserver:
        print(
            "[reference] MLServer results not provided — " "using floor as placeholder"
        )
        mlserver = floor

    # 3. Ceiling: Triton (optional — requires Triton image)
    ceiling = None
    if not args.skip_triton:
        print("[reference] Triton ceiling measurement...")
        try:
            from bench.reference.measure import (
                build_triton_model_repo,
                deploy_triton,
                measure_triton_ceiling,
            )

            repo = build_triton_model_repo(model_path, output_dir)
            container = deploy_triton(repo, gpu=args.gpu)
            try:
                payload_data = batch[list(batch.keys())[0]].tolist()
                ceiling = asyncio.run(
                    measure_triton_ceiling(
                        model_path.stem,
                        {
                            "name": "input",
                            "shape": [-1, batch[list(batch.keys())[0]].shape[-1]],
                            "data": payload_data[:1],
                        },  # single row
                        duration_s=args.duration,
                        concurrency=8,
                    )
                )
                print(f"  ceiling: {ceiling['throughput_rps']} RPS")
            finally:
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    capture_output=True,
                )
        except Exception as error:
            print(f"  WARNING: Triton measurement failed: {error}")

    if ceiling is None:
        ceiling = floor  # no ceiling available

    # 4. Compare
    comparison = compare_references(floor, mlserver, ceiling)
    report = generate_reference_report(comparison, output_dir)
    print("\n[reference] COMPLETE")
    print(f"  report: {report}")


def _save_layer(layers_dir: Path, name: str, data: dict) -> None:
    """Save a layer's data as JSON."""
    (layers_dir / name).write_text(json.dumps(data, indent=2, default=str) + "\n")
