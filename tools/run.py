#!/usr/bin/env python3
"""Universal benchmark runner: plan -> gated session -> collected results.

This is the workstation-side orchestrator. Given a domain and a plan, it:

  1. Validates the plan locally (schema, variants, payload files, images)
  2. Checks cluster prerequisites (oc login, client pod)   [runtime only]
  3. Clears stale sweeps from previous interrupted runs
  4. Uploads tooling, payloads, references, variants and the plan to the
     client pod's /store
  5. Runs the sweep inside the pod (tools/sweep.py drives every trial
     through tools/hook.py: fresh pod, parity gates, receipts)
  6. Collects the session back to results/<domain>/<plan>/
  7. Summarizes (merges cluster telemetry) and records checkpoint state

Every trial remains independently gated: bit-exact REST / byte-exact gRPC
parity against frozen references, fresh pod per trial (UID proof), and a
synchronized <=20 ms client start gate.

Usage:
    tools/run.py --domain wire-format --plan stock-vs-binary
    tools/run.py --domain processing --plan frontend-vs-worker-cpu --rounds 1
    tools/run.py --domain stability --plan soak-60min --dry-run

Environment:
    BENCHMARK_CONTEXT    oc context (default: current)
    BENCHMARK_NAMESPACE  namespace (default: mlserver-perf)
    BENCHMARK_CLIENT_POD client pod name (default: auto-discover)
    BENCHMARK_IMAGE      override every variant's image
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import Checkpoint  # noqa: E402


# --------------------------------------------------------------------------
# Local (no cluster) helpers
# --------------------------------------------------------------------------


def log(message: str, sink: Path | None = None) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if sink is not None:
        with sink.open("a") as handle:
            handle.write(line + "\n")


def load_plan(domain: str, plan_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    domain_dir = REPO / "benchmarks" / domain
    plan_path = domain_dir / "plans" / f"{plan_name}.json"
    variants_path = domain_dir / "variants.json"
    if not plan_path.is_file():
        raise SystemExit(f"Plan not found: {plan_path}")
    if not variants_path.is_file():
        raise SystemExit(
            f"Variants not found: {variants_path} — run "
            f"'python3 benchmarks/{domain}/generate.py' first"
        )
    return json.loads(plan_path.read_text()), json.loads(variants_path.read_text())


def validate_plan(
    plan: dict[str, Any],
    variants: dict[str, Any],
    image_override: str | None,
) -> tuple[list[str], list[str]]:
    """Static validation: returns (payload files to upload, problems)."""
    problems: list[str] = []
    required_keys = (
        "session",
        "rounds",
        "concurrency",
        "clients",
        "seconds",
        "warmup_seconds",
        "timeout_seconds",
        "ready_timeout_seconds",
        "prepare_timeout_seconds",
        "max_start_lateness_ms",
        "variants",
    )
    for key in required_keys:
        if key not in plan:
            problems.append(f"plan missing key: {key}")
    if problems:
        return [], problems

    names = [entry["name"] for entry in plan["variants"]]
    if len(set(names)) != len(names):
        problems.append("duplicate variant entry names in plan")
    for entry in plan["variants"]:
        hook_variant = entry.get("prepare_argv", [""])[-1]
        if hook_variant not in variants:
            problems.append(
                f"plan entry '{entry['name']}' references unknown variant "
                f"'{hook_variant}'"
            )
        if not entry.get("spec", {}).get("payloads"):
            problems.append(f"plan entry '{entry['name']}' has no payloads")

    # Payload files must exist locally (they are uploaded to the pod).
    uploads: set[str] = set()
    for entry in plan["variants"]:
        for payload in entry.get("spec", {}).get("payloads", []):
            name = payload.rsplit("/", 1)[-1]
            matches = list(REPO.glob(f"payloads/*/{name}"))
            if not matches:
                problems.append(f"payload file not found in payloads/: {name}")
            else:
                uploads.add(str(matches[0]))
    if not uploads:
        problems.append("no payload files resolved")

    # Every variant needs a resolvable image.
    for name, config in variants.items():
        if image_override or config.get("image"):
            continue
        problems.append(f"variant '{name}' has no image (set 'image' or --image)")

    if problems:
        return [], problems
    return sorted(uploads), []


def referenced_models(plan: dict[str, Any], variants: dict[str, Any]) -> list[str]:
    """Models whose references/payloads the plan's variants touch."""
    models: set[str] = set()
    for entry in plan["variants"]:
        hook_variant = entry.get("prepare_argv", [""])[-1]
        config = variants.get(hook_variant, {})
        models.add(config.get("model", "rfdetr-small"))
    return sorted(models)


def estimate_duration_seconds(plan: dict[str, Any]) -> int:
    cases = len(plan["variants"]) * len(plan["concurrency"]) * plan["rounds"]
    per_case = plan["warmup_seconds"] + plan["seconds"] + 180
    return max(600, cases * per_case)


def resolve_digest(image: str) -> str:
    """Resolve a floating tag to its immutable digest (never benchmark a
    floating tag). Uses docker pull + inspect; falls back to the tag only
    if resolution is impossible."""
    if "@sha256:" in image:
        return image
    import subprocess

    def inspect(fmt: str) -> str:
        result = subprocess.run(
            ["docker", "inspect", "--format", fmt, image],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    subprocess.run(["docker", "pull", image], capture_output=True, timeout=600)
    digest = inspect("{{index .RepoDigests 0}}")
    if "@sha256:" not in digest:
        ident = inspect("{{.Id}}")
        if ident.startswith("sha256:"):
            repo = image.rsplit(":", 1)[0] if ":" in image else image
            return f"{repo}@{ident}"
        return image
    return digest


# --------------------------------------------------------------------------
# Cluster helpers (runtime only — never called during --dry-run)
# --------------------------------------------------------------------------


class Cluster:
    def __init__(self, context: str | None, namespace: str, client_pod: str | None):
        self.context = context
        self.namespace = namespace
        self.client_pod = client_pod

    def oc(self, *args: str, check: bool = True, capture: bool = True) -> str:
        command = ["oc"]
        if self.context:
            command += ["--context", self.context]
        command += ["-n", self.namespace] + list(args)
        result = subprocess.run(
            command,
            capture_output=capture,
            text=True,
            timeout=600,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"oc {' '.join(shlex.quote(a) for a in args)} failed: "
                f"{result.stderr.strip()[:400]}"
            )
        return result.stdout.strip() if capture else ""

    def retry(self, description: str, args: tuple[str, ...], attempts: int = 3) -> str:
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.oc(*args)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                last = error
                log(f"  > {description}: attempt {attempt} failed — retrying")
                time.sleep(10 * attempt)
        raise RuntimeError(f"{description} failed after {attempts} attempts: {last}")

    def ensure_login(self) -> None:
        self.oc("whoami")

    def discover_client_pod(self) -> str:
        if self.client_pod:
            return self.client_pod
        pod = self.oc(
            "get",
            "pods",
            "-l",
            "app=mlserver-client",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        )
        if not pod:
            raise SystemExit("No mlserver-client pod found — run 'make setup' first")
        self.client_pod = pod
        return pod

    def exec_pod(self, command: str, timeout: int = 0) -> subprocess.CompletedProcess:
        args = ["exec", self.client_pod, "--", command]
        full = ["oc"]
        if self.context:
            full += ["--context", self.context]
        full += ["-n", self.namespace] + args
        return subprocess.run(
            full, shell=False, capture_output=True, text=True, timeout=timeout or None
        )

    def upload(self, local: Path, remote: str) -> None:
        self.retry(
            f"upload {local.name}",
            ("cp", str(local), f"{self.client_pod}:{remote}"),
        )


def kill_stale_sweeps(cluster: Cluster) -> int:
    """Kill any sweep left behind by a previously interrupted run."""
    script = (
        "import os, signal\n"
        "killed = 0\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit():\n"
        "        continue\n"
        "    try:\n"
        "        cmd = open(f'/proc/{pid}/cmdline', 'rb').read()"
        ".replace(b'\\x00', b' ').decode('utf-8', 'replace')\n"
        "        if '/store/tools/sweep.py' in cmd:\n"
        "            os.kill(int(pid), signal.SIGKILL)\n"
        "            killed += 1\n"
        "    except Exception:\n"
        "        pass\n"
        "print(killed)\n"
    )
    result = cluster.exec_pod(f"python3 -c {shlex.quote(script)}")
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_plan(
    domain: str,
    plan_name: str,
    rounds_override: int | None,
    concurrency_override: list[int] | None,
    image_override: str | None,
    force: bool,
    dry_run: bool,
    context: str | None,
    namespace: str,
    client_pod: str | None,
) -> int:
    plan, variants = load_plan(domain, plan_name)
    if rounds_override:
        plan["rounds"] = rounds_override
    if concurrency_override:
        plan["concurrency"] = concurrency_override
    if image_override:
        for config in variants.values():
            config["image"] = image_override

    # Image precedence policy (bench/images.py):
    #   - variant override image (tag)  -> resolved to digest
    #   - variant override image (sha)  -> used as-is
    #   - empty image (ISVC mode)       -> inherited from the cluster's
    #     serving runtime template, used as-is (already digest-pinned);
    #     the template has higher precedence than any engine default
    seen: dict[str, str] = {}
    for config in variants.values():
        tag = config.get("image", "")
        if not tag:
            continue  # ISVC inherit — leave empty, template image applies
        if "@sha256:" in tag:
            continue  # already immutable
        if tag not in seen:
            seen[tag] = resolve_digest(tag)
            if seen[tag] != tag:
                log(f"image pinned: {tag} -> {seen[tag]}")
        config["image"] = seen[tag]

    session = plan["session"]
    result_root = REPO / "results" / domain / plan_name
    cases = len(plan["variants"]) * len(plan["concurrency"]) * plan["rounds"]

    log(
        f"plan {domain}/{plan_name}: {cases} trials "
        f"({len(plan['variants'])} variants x {plan['concurrency']} x "
        f"{plan['rounds']} rounds)"
    )

    uploads, problems = validate_plan(plan, variants, image_override)
    if problems:
        for problem in problems:
            print(f"  problem: {problem}", file=sys.stderr)
        raise SystemExit(f"plan validation failed ({len(problems)} problems)")

    plan_path = REPO / "benchmarks" / domain / "plans" / f"{plan_name}.json"
    models = referenced_models(plan, variants)
    log(f"models touched: {', '.join(models)}")
    log(f"payloads to upload: {len(uploads)}")

    if dry_run:
        log("DRY RUN OK — plan validates locally; no cluster commands executed")
        return 0

    # ---- cluster phase ----------------------------------------------------
    cluster = Cluster(
        context or os.environ.get("BENCHMARK_CONTEXT"),
        namespace,
        client_pod or os.environ.get("BENCHMARK_CLIENT_POD"),
    )
    cluster.ensure_login()
    cluster.discover_client_pod()
    log(f"client pod: {cluster.client_pod}")

    checkpoint = Checkpoint(REPO / "results" / "STATE.json")
    if checkpoint.is_complete(domain, plan_name) and not force:
        log(
            f"checkpoint says {domain}/{plan_name} is already complete "
            f"— use --force to re-run"
        )
        return 0
    checkpoint.mark_running(domain, plan_name, cases)

    if result_root.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result_root.rename(result_root.with_name(f"{plan_name}-partial-{stamp}"))
        log(f"moved existing results to {plan_name}-partial-{stamp}")
    logs = result_root / "logs"
    logs.mkdir(parents=True)

    stale = kill_stale_sweeps(cluster)
    log(f"stale sweeps killed: {stale}")

    # Upload tooling, variants, plan, payloads, references.
    cluster.exec_pod(
        "mkdir -p /store/tools /store/plans /store/sessions "
        "/store/payloads /store/references"
    )
    for tool in ("sweep.py", "client.py", "hook.py"):
        cluster.upload(REPO / "tools" / tool, f"/store/tools/{tool}")
    variants_tmp = result_root / "variants.json"
    variants_tmp.write_text(json.dumps(variants, indent=2))
    cluster.upload(variants_tmp, "/store/variants.json")
    cluster.upload(plan_path, f"/store/plans/{session}.json")
    for local in uploads:
        cluster.upload(Path(local), f"/store/payloads/{Path(local).name}")
    for model in models:
        model_refs = None
        for candidate in (REPO / "references").iterdir():
            if not candidate.is_dir():
                continue
            stem = "mlp-small" if model == "mlp-54" else model
            if stem.startswith(candidate.name) or candidate.name in stem:
                model_refs = candidate
                break
        if model_refs is None:
            log(
                f"  > WARNING: no local references for {model} — run "
                f"'infrastructure/capture-references.sh' once before relying "
                f"on parity gates"
            )
            continue
        for ref in sorted(model_refs.iterdir()):
            cluster.upload(ref, f"/store/references/{ref.name}")
    log("uploads complete")

    # Sampler (workstation side): cluster telemetry aligned to the session.
    duration = estimate_duration_seconds(plan)
    samples = result_root / "samples.jsonl"
    sampler = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "tools" / "sampler.py"),
            "--output",
            str(samples),
            "--interval",
            "3",
            "--duration",
            str(duration),
        ],
        stdout=(logs / "sampler.log").open("w"),
        stderr=subprocess.STDOUT,
    )

    # Sweep inside the pod (retry the whole session on infrastructure
    # failures — the sweep itself is all-or-nothing per session).
    sweep_log = logs / "sweep.log"
    attempt = 0
    while True:
        attempt += 1
        cluster.exec_pod(f"rm -rf /store/sessions/{session}")
        result = cluster.exec_pod(
            f"python3 /store/tools/sweep.py --plan /store/plans/{session}.json "
            f"--output /store/sessions/{session}",
            timeout=duration + 1800,
        )
        sweep_log.write_text(result.stdout + "\n" + result.stderr)
        if result.returncode == 0:
            break
        if attempt >= 3:
            log(
                f"  > sweep failed after {attempt} attempts — collecting "
                f"what exists"
            )
            break
        log(f"  > sweep attempt {attempt} failed — retrying")
        time.sleep(30)

    sampler.terminate()
    sampler.wait(timeout=30)

    # Collect the session back.
    cluster.retry(
        "collect session",
        ("cp", f"{cluster.client_pod}:/store/sessions/{session}", str(result_root)),
    )

    # Summarize (merges sampler telemetry into metrics CSVs).
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "summarize.py"),
            "--session",
            str(result_root),
            "--sampler",
            str(samples),
        ],
        capture_output=True,
    )

    summary_path = result_root / "SUMMARY.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        rows = summary.get("results", [])
        invalid = [row for row in rows if not row.get("valid")]
        complete = len(rows) >= cases and not invalid
        checkpoint.record(
            domain,
            plan_name,
            trials=len(rows),
            invalid=len(invalid),
            complete=complete,
        )
        log(f"trials: {len(rows)}/{cases}, invalid: {len(invalid)}")
        if complete:
            log(f"COMPLETE: {result_root}")
            return 0
        return 1
    log(f"WARNING: no SUMMARY.json collected — see {sweep_log}")
    checkpoint.record(domain, plan_name, trials=0, invalid=0, complete=False)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="benchmark domain (wire-format, processing, ...)",
    )
    parser.add_argument(
        "--plan", required=True, help="plan name under benchmarks/<domain>/plans/"
    )
    parser.add_argument("--rounds", type=int, help="override the plan's round count")
    parser.add_argument(
        "--concurrency", help="override concurrency list, comma-separated"
    )
    parser.add_argument("--image", help="override every variant's image")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even if the checkpoint says complete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the plan locally; touch nothing",
    )
    parser.add_argument(
        "--namespace", default=os.environ.get("BENCHMARK_NAMESPACE", "mlserver-perf")
    )
    parser.add_argument("--client-pod", default=os.environ.get("BENCHMARK_CLIENT_POD"))
    parser.add_argument("--context", default=os.environ.get("BENCHMARK_CONTEXT"))
    args = parser.parse_args()

    concurrency = None
    if args.concurrency:
        concurrency = [int(value) for value in args.concurrency.split(",")]

    raise SystemExit(
        run_plan(
            args.domain,
            args.plan,
            args.rounds,
            concurrency,
            args.image,
            args.force,
            args.dry_run,
            args.context,
            args.namespace,
            args.client_pod,
        )
    )


if __name__ == "__main__":
    main()
