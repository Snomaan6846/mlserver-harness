# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-11-14

Initial release — restructured from the RHOAIENG-102092 campaign
tooling (MLServer-resources) into a standalone, domain-organized
benchmark suite.

### Added
- Domain-organized benchmark plans (22 plans / 5 domains): wire-format,
  processing, threading, stability, transport — each generated from a
  checked-in `generate.py` with a matching `variants.json`
- `tools/run.py` — universal runner: local plan validation, pod sync,
  stale-sweep cleanup, sweep execution, collection, summarization,
  checkpointing
- `tools/checkpoint.py` + `tools/resume.py` — campaign state and resume
- `tools/cleanup.py` — stale sweep / pod storage / local sampler guards
- `tools/validate.py` — result compliance checker (gates, receipts,
  trial counts, round spread)
- `tools/report.py` — markdown report generation
- `infrastructure/` — setup / verify / teardown / capture-references
  scripts plus all cluster manifests (RBAC, client build/deploy, GPU and
  CPU serving deployments, model loader)
- SHA-pinned models (`models/*/MANIFEST.json`) with the deterministic
  MLP-54 generator
- Documentation set: QUICKSTART, METHODOLOGY, CONFIGURATION, MODELS,
  NAMING, ENVIRONMENT, TROUBLESHOOTING

### Ported from the campaign tooling (battle-tested, 1,214 gated trials)
- `tools/sweep.py` — plan-driven session engine with alternating rounds
  and synchronized client start gate
- `tools/client.py` — REST/gRPC load generator with per-request sampling
- `tools/hook.py` — per-trial prepare/verify hook (now fully
  data-driven: all variant definitions live in `variants.json`)
- `tools/sampler.py`, `tools/summarize.py`, `tools/analyze.py` —
  telemetry collection, session summarization, paired bootstrap analysis

### Historical
- `reports/` carries the RHOAIENG-102092 engineering record: final
  report, benchmarking ledger, phase-3 work plan, and the phase-1
  studies (archived)
