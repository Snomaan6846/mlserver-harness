# Tools

How to measure. All tools are model-agnostic: models, variants and
plans are data (`benchmarks/*/variants.json`, `plans/*.json`).

## Workstation-side

| Tool | Role |
|---|---|
| `run.py` | **Universal runner.** Validates a plan locally, syncs everything to the client pod, clears stale state, executes the sweep, collects, summarizes, checkpoints. `--dry-run` validates without touching the cluster. |
| `resume.py` | Re-runs every plan the checkpoint does not mark complete. |
| `checkpoint.py` | Campaign state (`results/STATE.json`): status, trial/invalid counts per plan. |
| `cleanup.py` | Stale-sweep kill, pod `/store` storage report/prune, local sampler kill. Invoked automatically by the runner; `--prune` to act. |
| `sampler.py` | Cluster telemetry (pod CPU/memory, GPU via DCGM) sampled every 3 s during a session. |
| `summarize.py` | Merges a session's trials with aligned telemetry into `metrics-summary.csv` / `metrics-cases.csv`. |
| `analyze.py` | Paired statistical analysis: variant-vs-variant ratios with bootstrap CI95. |
| `validate.py` | Compliance checker: trial validity, receipts, parity gates, counts, round spread. Exit 1 on violation. |
| `report.py` | Markdown report from collected sessions. |

## Pod-side (executed inside the client pod by the runner)

| Tool | Role |
|---|---|
| `sweep.py` | Session engine: walks (round × concurrency × variant), alternating variant order per round; writes `SUMMARY.json` incrementally. |
| `client.py` | Load generator: REST (aiohttp) / gRPC, synchronized start gate, warm-up + measured window, per-request sampling to `samples.jsonl`. |
| `hook.py` | Per-trial prepare/verify: patch deployment, fresh pod (UID proof), readiness, provider proof, **parity gates against frozen references**, provenance receipts. `--capture-refs` saves references instead of comparing (deliberate, one-time). |

## Contracts

- **Plan JSON** (consumed by `sweep.py`): `session`, `rounds`,
  `concurrency[]`, `clients`, `seconds`, `warmup_seconds`,
  `timeout_seconds`, `ready_timeout_seconds`, `prepare_timeout_seconds`,
  `max_start_lateness_ms`, `variants[{name, prepare_argv, spec}]`.
- **Variant JSON** (consumed by `hook.py` via `/store/variants.json`):
  `deployment`, `service`, `device`, `expect_cuda`, optional `model`,
  `image`, `command`, and `env` (MLSERVER_* map).
- **Receipt JSON** (per case): variant, deployment, image digest, pod
  UID, env applied, providers line, model-ready seconds, correctness
  gate results.

## Tests

`tools/tests/` covers the pure logic (plan validation, checkpoint
state machine). `make test`. No cluster required.
