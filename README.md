# MLServer Benchmarks

A reproducible, correctness-gated benchmark suite for
[MLServer](https://github.com/opendatahub-io/MLServer) serving performance —
REST/gRPC wire formats, frontend-vs-worker processing placement, and ONNX
Runtime tuning — on OpenShift with GPU and CPU serving profiles.

**Engine-first architecture** — pick WHERE to run, then WHAT to do:

```bash
# All activities work on both engines:
make benchmark ENGINE=local      # or ENGINE=cluster (ISVC on RHOAI)
make diagnose ENGINE=local      # or ENGINE=cluster
make reference ENGINE=local     # or ENGINE=cluster
```

The local engine auto-detects your container runtime:
**Podman** (preferred, RHEL/Fedora default) → **Docker** → **nerdctl**.
If both Podman and Docker are installed, Podman is used.

- **Docker engine** (`make benchmark`, `bench/`) — local, all-framework
  (sklearn / xgboost / lightgbm / onnx), Triton/MLPerf-style methodology,
  interactive wizard, statistical regression comparison.
- **OpenShift engine** (`make run-plan`, `tools/`) — the battle-tested
  campaign tool from RHOAIENG-102092 (wire-format, processing, threading,
  stability, transport domains; frozen-reference parity gates).

Every trial in both engines is **gated**: responses must match native-library
references, and invalid trials are marked, never silently dropped.

## Local engine (Podman/Docker) — complete `make` workflow

```bash
make benchmark                  # interactive wizard (alias: make run)
make benchmark-cpu              # full suite, all 4 CPU frameworks
make benchmark-gpu              # full suite, GPU onnx (provider guard on)
make benchmark-one TARGET=cpu FRAMEWORK=sklearn
make dry-run TARGET=cpu FRAMEWORK=all
    # validates images + model checksums, boots MLServer, runs the
    # correctness parity gate, exits without load testing
make compare BASELINE=<dir> CANDIDATE=<dir>
    # Mann-Whitney U + Welch's t regression check (non-zero exit on regression)
make models                     # materialize the model zoo on demand
make models-clean              # drop the zoo (regenerates deterministically)
make test                       # unit + lightweight integration tests
make clean                      # stop containers, remove temp files + zoo
```

### What the wizard collects

`make benchmark` prompts for: target hardware (CPU / GPU / both) →
frameworks (filtered by target) → suites (`protocol`, `batching`, `workers`,
`caching`, `multimodel`, `lifecycle`, `saturation`, `processing`,
`framework`, `middleware`, `codec`, `grpc`, `hardware`) → model tiers
(small / medium / large) → load profile (quick / standard / exhaustive /
custom) → server options (image with override, `warm_workers`, cache mode,
optional pinned reference models, baseline for comparison) → a formatted
**Execution Plan Summary** → `Confirm and proceed? [y/N]`. Anything but
`y`/`Y` exits cleanly.

### Coverage (Docker engine)

| Suite | What it measures |
|---|---|
| `protocol` | REST JSON baseline curve across the concurrency ladder (gRPC → `grpc` suite, binary → `processing`, codecs → `codec`) |
| `batching` | `max_batch_size` × `max_batch_time` sweep: fill efficiency, queueing, p99 |
| `workers` | `parallel_workers` ladder (0=in-process … cores), thread-env contention |
| `caching` | capability probe (MLServer has no response cache — roadmap WI-04) + repeat-vs-unique distributions |
| `multimodel` | 2/5/10 models per instance: noisy-neighbor p99, RSS scaling |
| `lifecycle` | load/unload latency, ready time, cold-start vs warm inference |
| `saturation` | closed-loop ladder to peak, then open-loop (constant + Poisson) at fractions of peak |
| `processing` | wire format × processing path: `rest_worker_json`, `rest_binary_tensor`/`rest_worker_binary` (binary tensor extension, JSON + binary request encodings), `parallel_inference_transport` queue/zmq |
| `framework` | model framework-level pydantic settings via `parameters.extra` (OnnxSettings): `session_options` sweeps (intra/inter-op threads, mem pattern, CPU arena), `providers`, `provider_options` (cudnn algo search, arena strategy), `run_options`, `session_config_entries`, `cuda_graph`, `cuda_blocking_sync` (GPU-gated) |
| `middleware` | server-level overhead on/off deltas: `gzip_enabled`, `access_log`, `metrics_endpoint`, `use_structured_logging`, `tracing_server` (OpenTelemetry) |
| `codec` | input codec variants (`parameters.content_type`: np / pd), sklearn `predict_fn` (predict / predict_proba) |
| `grpc` | gRPC dataplane: typed contents vs `raw_input_contents`, `grpc_raw_input_buffers` zero-copy path, `grpc_max_message_length` |
| `hardware` | deployment matrix: **CPU image on CPU vs CUDA image on GPU** (stock provider behaviour, GPU arm anti-silent-fallback guarded), request-batch sweep for the crossover curve + speedup ratios |
| `faults` | fault injection: declared dynamic `[-1,N]` shapes × batching (F-01 class — truncation/leakage/empty measured per response), poison-request resilience among valid traffic |

**Coverage ledger** — `docs/COVERAGE.md` maps every MLServer
setting to its suite or a documented exclusion reason.
**Findings** — `docs/FINDINGS.md` records correctness/performance
bugs the gates catch (F-01: batcher mis-splits `-1`-shaped requests —
truncation, cross-request leakage, empty responses, all silent; F-02:
in-process vs 4-worker serving flips per model).

**Replication** — `--rounds N` repeats the suite matrix N times; the
summary reports per-cell mean ± std (`results/*/results.json` →
`summary.rounds`).

**External cross-check** — `make crosscheck` runs ApacheBench against
the same server/config as the load generator and flags divergence
(measured agreement: 14%).

**Image feature probing** — fork features (binary tensor paths, zmq
transport, newer OnnxSettings fields) may postdate the published
`odh-master` image. Every suite probes the running image first and
records unsupported variants as `SKIPPED_UNSUPPORTED_IMAGE` instead of
silently measuring an unapplied setting. Use `--image-source source` to
benchmark a fresh build from the MLServer checkout.

Load profiles follow Triton `perf_analyzer` / MLPerf conventions: mandatory
warmup, sliding-window stability gates (±5%), anti-coordinated-omission
open-loop arrival, full percentile distributions (p50…p99.9), Prometheus
`/metrics` scraping for latency decomposition, host + GPU (pynvml) telemetry.

### Determinism (MLPerf closed-division standard)

- Fixed seeds everywhere (`PYTHONHASHSEED`, numpy, model generation)
- SHA-256 manifests for every model and payload, verified before every run
- Environment manifest in every result: git SHAs (both repos), image digest,
  package versions, CPU/GPU hardware, exact config
- **Golden-reference parity gate before any measurement**: native-library
  reference computed inside the serving image (exact library versions — no
  host/image skew) vs through-server REST + gRPC. FP32 rtol=1e-5, INT64
  exact. Mismatch ⇒ `FAILED_CORRECTNESS`, no numbers recorded.
- **GPU anti-silent-fallback guard**: CUDAExecutionProvider must be actively
  executing (log verification + GPU utilization check) or the run aborts
  with `FAILED_GPU_PROVIDER_FALLBACK` — CPU-fallback numbers are never
  recorded as GPU results.

### Model zoo (industry baselines, auto-materialized)

| Framework | Baseline schemas | Canonical models |
|---|---|---|
| sklearn | higgs10 / higgs / covertype | LogisticRegression pipeline, RandomForest, GradientBoosting |
| xgboost | higgs10 / higgs / covertype | XGBClassifier, XGBRegressor (hist) |
| lightgbm | higgs10 / higgs / covertype | LGBMClassifier, LGBMRegressor |
| onnx | MNIST / CIFAR / ImageNet shapes | dense MLP + CNN (CPU + CUDA providers) |

Tiers: small (10 feat / MNIST), medium (28 feat / CIFAR), large (54 feat /
ImageNet). Payloads: rows per request ∈ {1, 8, 32, 128, 512}. The zoo is
**gitignored** and materialized on demand (`make models`) — reuse-if-present,
generate-or-download otherwise; a pinned registry adds real reference models
(e.g., ONNX Model Zoo mnist-8, SHA-256 verified).

Default images (floating, always latest code — override with `IMAGE=` or
`--image` for R&D): CPU `quay.io/opendatahub/mlserver:odh-master`, GPU
`quay.io/opendatahub/odh-mlserver-cuda:odh-master`.

## Diagnose — 6-layer performance diagnostic

```bash
make diagnose              # full 6-layer diagnostic
make diagnose-quick        # skip py-spy/memray (faster)
make reference             # 3-point: raw ORT floor → MLServer → Triton ceiling
make reference-gpu         # with CUDA provider
```

Six layers: system (USE method), process/thread, on-CPU (py-spy),
memory (memray + RSS), latency decomposition (Prometheus), GPU (DCGM).
Twelve automated findings rules (frontend-bound, swap, GIL contention,
thermal throttling, etc.) produce a reviewable `DIAGNOSIS.md` with
severity-ranked findings and recommended actions.

See `docs/DIAGNOSE.md` for the full guide.

## ISVC mode

For cluster deployments via RHOAI's native serving path
(InferenceService + ServingRuntime + PVC):
`bench/isvc.py` provides the ISVCDeployment class with fresh-pod
discipline, route opt-in, and prerequisites verification.
See `docs/ISVC_DEPLOYMENT_PLAN.md` for the research-backed design.

## OpenShift engine

```
make setup        # bootstrap the cluster environment (once)
make smoke-test   # one gated trial to prove the setup
make run-plan DOMAIN=wire-format PLAN=stock-vs-binary
make report       # markdown report from collected results
```

## What the OpenShift engine measures

| Domain | Question | Headline result (Tesla T4, RF-DETR) |
|---|---|---|
| `wire-format` | How do request bytes become tensors? | Binary tensor REST: **4.0×** vs JSON (7.9 → 31.5 RPS) |
| `processing` | Frontend or workers for parse/encode? | Worker binary: **−24.5% CPU-ms/req**; +37% for fast models |
| `threading` | Which ORT runtime knobs matter? | CPU threads T2→T8: **2.7×**; GPU: no effect (device-bound) |
| `stability` | Does it hold up under sustained load? | 60-min soak: **110,917/110,917** requests, zero failures |
| `transport` | Queue vs ZeroMQ; route vs direct? | zmq: **+10–11%**; route overhead only at C1 |

Full findings and methodology: `reports/FINAL_REPORT.md`.

## Repository layout

```
bench/            Docker engine: wizard, suites, parity gates, loadgen, telemetry, compare
benchmarks/       OpenShift engine: domains, variants.json + plans/
models/           OpenShift-engine models (SHA-pinned)
models-zoo/       Docker-engine zoo (gitignored, auto-materialized)
payloads/         Wire-format request files (OpenShift engine)
references/       Frozen parity references (OpenShift engine)
infrastructure/   Cluster manifests + setup scripts; images.json
tools/            OpenShift engine: runner, sweep, hook, sampler, analysis
docs/             Guides: quickstart, methodology, configuration, naming
examples/         Runnable end-to-end examples
results/          Collected evidence (generated; gitignored)
reports/          Historical engineering reports from the RHOAIENG-102092 work
```

## Requirements

- Docker engine: Docker (or podman) + Python ≥ 3.10 with `aiohttp`,
  `grpcio`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `psutil`
  (`pip install -e .`); a GPU host for GPU targets
- OpenShift engine: an OpenShift cluster with `oc` access (Tesla-T4-class
  GPU node for GPU domains)

## Status

Built from the tooling, models and methodology of the RHOAIENG-102092
campaigns (1,214 gated trials; PR #378 merged). See `CHANGELOG.md`,
`docs/METHODOLOGY.md` and `reports/` for the full history.
