# Quickstart

From zero to a gated benchmark result in ~20 minutes (given cluster access).

## 1. Prerequisites

```bash
# Workstation
python3 -c "import aiohttp, grpc, numpy"   # or: pip install -e ".[dev]"
oc whoami                                  # logged into your OpenShift cluster

# Cluster: a GPU node (Tesla-T4 class) for the GPU domains. The CPU
# domains run on the same node. Nothing else is needed — setup creates
# the namespace, RBAC, PVCs, client pod and serving deployments.
```

## 2. Bootstrap the environment (once)

```bash
make setup
```

This creates everything: namespace `mlserver-perf`, RBAC, the client pod
(with tools, payloads and references synced to its `/store`), models on
the models PVC, the GPU serving deployment, frozen parity references,
and finally a smoke test (one gated trial). Re-running `make setup` is
safe — completed steps are skipped.

## 3. Verify

```bash
make verify
```

Read-only readiness check: client pod, deployments, tools, references.

## 4. Run a benchmark

```bash
# The headline comparison: binary tensor REST vs stock JSON
make run-plan DOMAIN=wire-format PLAN=stock-vs-binary

# Fast version while learning (1 round, single concurrency)
make run-plan DOMAIN=wire-format PLAN=stock-vs-binary --dry-run   # validate only
python3 tools/run.py --domain wire-format --plan stock-vs-binary \
    --rounds 1 --concurrency 8
```

Watch the log: each trial prints the hook's parity verification and the
load result. When it finishes:

```bash
make validate     # compliance gates: valid trials, receipts, spread
make report       # reports/benchmark-report.md with median tables
```

Results land in `results/<domain>/<plan>/` — one directory per session
with `SUMMARY.json`, per-case receipts, telemetry and metrics CSVs.

## 5. Run a whole domain (or everything)

```bash
make run-wire-format    # ~2 h
make run-processing     # ~4 h
make run-threading      # ~4 h
make run-transport      # ~30 min
make run-stability      # ~1 h (60-minute soak)
make run-all            # everything (~12 h)
```

Interrupted? The checkpoint tracks completed plans:

```bash
make checkpoint         # what is complete / pending
make resume             # run everything not yet complete
```

## 6. Next steps

- `docs/METHODOLOGY.md` — how the gates work and why
- `docs/NAMING.md` — reading variant names
- `benchmarks/<domain>/README.md` — what each domain measures
- `examples/` — end-to-end runnable scripts
