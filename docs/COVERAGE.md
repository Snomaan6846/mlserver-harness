# Coverage Ledger

The definitive mapping of **every benchmarkable MLServer configuration**
(from the source audit: all `Settings` / `ModelSettings` / `ModelParameters`
fields, all `MLSERVER_*` env vars, codecs, middleware, gRPC dataplane,
framework-level pydantic settings) to the suite that measures it — or the
documented reason it is consciously excluded.

## Covered by suites (13)

| MLServer surface | Suite | Axis |
|---|---|---|
| REST vs gRPC, JSON vs binary payloads, payload sizes | `protocol` | protocol × encoding × concurrency |
| `max_batch_size`, `max_batch_time` (ModelSettings) | `batching` | full cross product |
| `parallel_workers` (server + model level), thread env (OMP/MKL/BLAS) | `workers` | ladder 0…cores |
| `warm_workers` | `workers` | documented deprecated no-op (A/B recorded as skipped) |
| response caching | `caching` | capability probe (no cache exists — roadmap WI-04) + repeat/unique distributions |
| multi-model serving | `multimodel` | 2/5/10 models, noisy-neighbor p99, RSS |
| load/unload latency, ready time, cold vs warm inference | `lifecycle` | repository API timings |
| `load_models_at_startup` | `lifecycle` | lazy vs eager: startup-to-ready + deferred load API |
| closed/open loop saturation | `saturation` | concurrency ladder + constant/Poisson at fractions of peak |
| `rest_worker_json`, `rest_binary_tensor`, `rest_worker_binary` | `processing` | wire-format × processing-path matrix, JSON + binary request encodings |
| `parallel_inference_transport` (queue/zmq) | `processing` | IPC transport (zmq pre-start image probe) |
| OnnxSettings `session_options` (intra/inter-op threads, mem pattern, CPU arena, log severity) | `framework` | `parameters.extra` sweeps |
| OnnxSettings `providers`, `provider_options` (cudnn algo search, arena strategy) | `framework` | `parameters.extra` (GPU-gated axes) |
| OnnxSettings `run_options`, `session_config_entries` | `framework` | `parameters.extra` |
| OnnxSettings `cuda_graph`, `cuda_blocking_sync` | `framework` | GPU-gated |
| `gzip_enabled` | `middleware` | on/off delta |
| `access_log` | `middleware` | on/off delta |
| `metrics_endpoint` | `middleware` | set/unset delta |
| `use_structured_logging` | `middleware` | on/off delta |
| `tracing_server` (OpenTelemetry) | `middleware` | on/off delta (no-collector cost) |
| `parameters.content_type` (np/pd codecs) | `codec` | codec variants |
| sklearn `predict_fn` (predict/predict_proba) | `codec` | `parameters.extra` |
| gRPC typed contents vs `raw_input_contents` | `grpc` | wire format |
| `grpc_raw_input_buffers` | `grpc` | zero-copy server path |
| `grpc_max_message_length` | `grpc` | accept envelope limits |
| CPU image on CPU vs CUDA image on GPU | `hardware` | deployment matrix + request-batch crossover |
| Malformed/dynamic-shape request resilience | `faults` | poison-among-valid, declared-shape × batching with per-response row-count validation |

## Consciously excluded (with reasons)

| Surface | Reason |
|---|---|
| `cors_settings` | Response-header middleware; the fork blocks wildcard origins by security policy; inference-throughput impact is within noise of the covered middleware axes |
| `debug` | Enables the logging interceptor — a logging axis already measured (`access_log`, `use_structured_logging`) |
| `_custom_grpc_server_settings` / `_custom_rest_server_settings` / `_custom_metrics_server_settings` | Escape hatches for arbitrary uvicorn/grpc server options — no stable benchmarkable axis; future axis if a specific option shows production impact |
| `inference_pool_gid` / `autogenerate_inference_pool_gid` | Shared inference pools across models; requires gid coordination — niche advanced config; future `multimodel` axis |
| `environment_path` / `environment_tarball` / `environments_dir` | Custom runtime environments — **blocked in PRODUCTION mode** by the trusted-runtimes security model (deliberate) |
| `empty_registry_readiness` | Readiness semantics flag, not a perf axis (its default `True` is what makes the lazy-load axis measurable) |
| `strict_readiness` | Per-model readiness semantics, not perf |
| `model_operation_timeout`, `parallel_workers_timeout`, `worker_start_timeout` | Timeout/robustness behaviour, not throughput axes |
| `model_repository_implementation` (+args) | Pluggable repository (loading-time infra); future `lifecycle` axis if custom repos are used in production |
| `host`, ports, `root_path`, `server_name`, `server_version`, `metrics_dir`, `metrics_rest_server_prefix` | Infrastructure identity — no perf axis |
| `log_level` / `logging_settings` | Server log verbosity — logging overhead is covered by `access_log`/`use_structured_logging`; ORT-side severity is swept (`framework` log_severity_3) |
| ModelSettings `versions` (multi-version serving) | Repository/lifecycle complexity, not a per-request axis; future `lifecycle` axis |
| ModelSettings `platform` | Metadata only |
| `parameters.version` / `format` / `uri` | Metadata / loading infra |
| Codecs `str` / `base64` / `datetime` | Apply to BYTES/string payloads — the zoo is deliberately numeric-only (deterministic industry baselines); recorded as documented skips in `codec` |
| sklearn `predict_fn=transform` | Requires transformer models; `predict_proba` already exercises the mechanism |
| ORT `execution_mode`, `graph_optimization_level` | **pybind enum properties reject plain ints** — cannot be driven through JSON `parameters.extra` (verified; excluded in `framework` by design) |
| `rest_worker_binary` without `rest_binary_tensor` | Invalid config (server rejects at startup — validated constraint) |
| mlflow / huggingface / seldon runtimes | Not in the ODH fork's runtime set (sklearn, xgboost, lightgbm, onnx) — out of scope |

## Image-dependent variants (skip until image catches up)

These record `SKIPPED_UNSUPPORTED_IMAGE` on the published `odh-master`
image because the fork features postdate it (verified by in-container
probes). They measure normally on a source build
(`--image-source source`) or once a newer image is published:

- `processing`: `rest_binary_tensor` / `rest_worker_binary` variants
- `processing`: zmq transport (zmq extra missing from image)
- `framework`: `providers`, `run_options`, `session_config_entries`,
  `cuda_graph`, `cuda_blocking_sync`, `provider_options` axes

## Payload axis (honest boundary)

The zoo generates `payloads.npz` (row ladders 1/8/32/128/512) and the
load profiles declare `batch_sizes`, but **no suite currently reads
them** — every suite except `hardware` sends the 8-row validation
batch. The hardware suite covers the row ladder through its own
`tile_batch` mechanism. Wiring the payload ladder into the
protocol/processing suites remains open work; until then, wire-format
deltas are bounded to the 8-row request size.

## Parity gate scope (honest boundaries)

The run-start parity gate verifies each model over **REST JSON and
gRPC typed contents** with the validation batch, against an
in-container native reference (plus `equal_nan` for float outputs).
It does NOT itself cover the binary-tensor path, gRPC raw contents,
the `pd` content type, or batched merges — those are covered by
per-suite checks where they matter (`faults` validates response row
counts per request; `processing`/`codec` cases pass through the same
gate on their patched configs). Widening the gate to those encodings
remains open work.

## Methodology notes

- **Rounds** (`--rounds N`): per-cell mean ± std across replicates.
  Single-replicate runs report only the stability-gate spread as a
  noise estimate; multi-round runs add between-round variance.
- **External cross-check** (`make crosscheck`): ApacheBench vs the
  load generator on the same server/config — guards against
  client-side bottlenecks masquerading as server results.

## Remaining work (execution, not coverage)

1. **Full local benchmark run** — `make benchmark-cpu` (all frameworks,
   all 13 suites, standard profile) has never run to completion;
   individual suites are live-validated (processing, framework, codec,
   grpc, middleware, lifecycle, hardware skip-path)
2. **GPU/cluster validation on T4** — hardware suite GPU arm, diagnose
   GPU layers, Triton ceiling, full ISVC flow (blocked on go-ahead)
3. **Image catch-up** — publish an `odh-master` image containing the
   fork's binary-tensor/zmq/OnnxSettings features to un-skip the
   image-dependent variants
