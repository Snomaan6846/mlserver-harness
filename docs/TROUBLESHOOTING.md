# Troubleshooting

Every failure mode below was hit at least once during the original
campaigns. Symptoms first, fixes second.

## "no mlserver-client pod found"

The client deployment is down or was never created.
`make setup` (idempotent). If the image build fails, check the
internal registry and retry: `oc -n mlserver-perf logs
bc/mlserver-client`.

## Sweep fails on every trial with "hook exited nonzero"

Read the case's `prepare.log` — the hook prints its stage. Common
causes:

- **Parity mismatch**: responses differ from the frozen references.
  Either the image changed behavior (expected after a code change —
  re-capture references deliberately with
  `infrastructure/capture-references.sh` and review why), or you are
  serving a different model than the references were captured for.
- **"No header length recorded for binary payload"**: the binary-frame
  sidecar is missing for a payload — regenerate the domain plans
  (`make generate`) and let the runner re-upload.
- **Providers mismatch**: the pod reports CPUExecutionProvider on a
  `cuda` variant — the GPU was unavailable or the node selector
  matched a CPU-only node.
- **TypeError about incompatible function arguments** (ORT enum):
  you set an ORT enum-typed option (e.g. inter-op PARALLEL) — not
  expressible in JSON settings (known defect, roadmap WI-10).

## "sweep failed after 3 attempts — collecting what exists"

Infrastructure instability (login expiry, node churn). The session is
retried whole; if it still fails, results so far are collected and the
plan is marked failed in the checkpoint. `make resume` re-runs it.

## oc cp fails with "no space left on device"

The client pod's `/store` filled with old sessions:
`tools/cleanup.py --pod-storage --prune` (keeps tools, payloads and
references). The runner also clears stale sweeps before every plan.

## Results look wrong: throughput far below expectation

1. Check `providers` in the receipts — CPU fallback is the classic
   silent killer (the gate catches it, but only if you read failures).
2. Check the GPU utilization column in `metrics-summary.csv` — if the
   GPU is idle while RPS is low, the frontend is the bottleneck (that
   is a finding, not a bug).
3. Check start lateness in `SUMMARY.json` — clients that missed the
   ≤20 ms gate are marked invalid, and their trials do not count.

## Two campaigns ran concurrently

Stop one. The hook enforces single-tenancy per trial (it scales the
sibling deployment to zero), but two concurrent sweeps will thrash
deployments against each other and produce garbage. Kill stale sweeps:
`tools/cleanup.py --stale-sweeps --prune`.

## Re-running a plan does nothing

The checkpoint marks it complete. `make run-plan ... ` with
`--force`, or reset the entry:
`tools/checkpoint.py --reset wire-format/stock-vs-binary`.

## References missing after setup

`infrastructure/capture-references.sh` runs during setup; if it was
skipped (verify with `make verify`), run it directly. MLP references
require the CPU deployment — the script scales it up and back down
automatically.

## Where to look, in order

1. `results/<domain>/<plan>/logs/sweep.log` — the sweep's own output
2. `results/<domain>/<plan>/case-NNNN/prepare.log` — the hook's stage
3. `results/<domain>/<plan>/case-NNNN/receipts.json` — provenance
4. `results/<domain>/<plan>/SUMMARY.json` — per-trial validity and
   error strings
