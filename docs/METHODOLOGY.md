# Methodology

How a trial is measured, what makes it valid, and why the design is the
way it is. This discipline was forged across 1,214 gated trials on the
RHOAIENG-102092 campaigns; every rule below exists because its absence
once produced a wrong number.

## The trial lifecycle

Every trial (one variant at one concurrency in one round) follows the
same protocol, enforced by `tools/sweep.py` + `tools/hook.py`:

1. **Prepare** (`hook.py`): the serving deployment is patched to the
   variant's configuration (image, env) and restarted. The previous
   pod's UID is captured first and proven gone after — a trial never
   measures a warm process left over from a previous configuration.
2. **Readiness**: the model must report ready through the Service
   (not just the pod).
3. **Provider proof**: the active ONNX Runtime execution providers are
   read from the server logs and must match the variant's device
   (a CPU-fallback silently invalidates GPU comparisons).
4. **Parity gate**: the hook sends one REST and one gRPC request (and a
   binary-frame REST request when the extension is enabled) and compares
   the responses against the **frozen references**:
   - REST: SHA-256 of the parsed output arrays (bit-exact)
   - gRPC: SHA-256 of the raw response bytes (byte-exact)
   - Binary REST: SHA-256 of the raw tensor payload after the JSON
     header (per-request generated IDs exempted)
   Any mismatch fails the trial. References are captured once,
   deliberately (`infrastructure/capture-references.sh`), never during
   a trial.
5. **Load** (`client.py` ×2): two load-generator clients synchronize
   through a start file and must begin within **≤20 ms** of the
   requested instant; 60 s warm-up, then a 60 s measured window.
   Every request is checked (minimum response bytes; per-request
   success sampling).
6. **Receipt**: each case directory records the image digest, pod UID,
   applied env, providers line, parity hashes, model-ready timing and
   the full result. `SUMMARY.json` is written incrementally per case.

## What makes a trial valid

A trial is valid only if: every client reports valid, every client
process exits 0, and the maximum start lateness is within the gate.
The parity gate failing (step 4) makes `hook.py` exit nonzero, which
marks the trial invalid — **incorrect results are never fast results**.

## Paired design

Control and candidate variants alternate within each round: even rounds
run the plan's variant order forward, odd rounds reversed. This cancels
temporal drift (cluster noise, node temperature, background activity).
Three rounds per cell; **medians** are reported, never means. Paired
bootstrap confidence intervals (20,000 resamples) come from
`tools/analyze.py` when comparing two variants.

## Fresh-pod discipline

Serving state is not trustable across configurations. Every trial gets a
new pod, and the old pod's UID must be gone before measurement. This is
why an interrupted session re-runs from scratch on resume rather than
continuing: partial sessions are scientifically safe to redo, never to
append to.

## Isolation

- Load generators run in a **separate pod** from the server (dedicated
  CPU reservations) so client CPU cannot steal serving cores.
- The CPU serving deployment never coexists with the GPU deployment
  (single GPU; the hook enforces mutual exclusion).
- GPU telemetry (utilization, framebuffer, power, temperature) is
  sampled every 3 s from the workstation and aligned to each trial's
  measurement window by `tools/summarize.py`.

## Trial accounting

Counts come from the per-session `SUMMARY.json` (the authoritative log
of every trial that produced a measurement). Aborted attempts that
produced no result are not trials. Case directories pruned during
storage cleanup remain counted through their SUMMARY entries.

## What the gates caught in practice

- cuDNN `DEFAULT` algorithm search silently changing outputs by 9.0e-05
  (three trials rejected — the gate was right)
- A review-introduced JSON-delegation regression (16% on the
  worker-JSON path) caught by the paired benchmark before ship
- Operator interference and rollout transients, correctly attributed
  because every trial carried its provenance receipt

See `reports/BENCHMARKING_HISTORY.md` (failure register) for the
complete list with root causes.

## Wire-format and processing-path measurement (processing suite)

The processing suite measures WHERE request bytes get parsed and how
they travel to inference workers:

- `rest_worker_json` — JSON frames forwarded to workers (frontend
  skips re-serialization)
- `rest_binary_tensor` + `rest_worker_binary` — the KServe v2 Binary
  Tensor Data extension (JSON header + raw little-endian tensor
  bytes; zero-copy `np.frombuffer` views on the server)
- `parallel_inference_transport` — multiprocessing queue vs ZeroMQ

Client-side request encodings are exercised in matching pairs: a JSON
request and a binary tensor request through each enabled server
configuration, so the wire-format delta is measured independently of
the server processing path. Binary responses are parsed and
shape/dtype-verified (not just status-checked).

## Framework-level settings via parameters.extra (framework suite)

Model framework runtimes resolve their own pydantic settings from
`parameters.extra` in `model-settings.json`, with
`MLSERVER_MODEL_<RUNTIME>_*` env vars merged underneath (file wins on
conflict). For ONNX (`OnnxSettings`), that exposes the full ORT tuning
surface: `providers`, `provider_options`, `session_options`,
`session_config_entries`, `run_options`, `cuda_graph`,
`cuda_blocking_sync`.

The suite patches `parameters.extra` per variant (deep-merged,
preserving existing keys) and measures each axis at fixed load.
**Pybind enum limitation:** `execution_mode` and
`graph_optimization_level` are ORT enum properties that reject plain
ints via `setattr` — JSON cannot carry enum objects, so these axes are
excluded by design (documented in the suite source).

## Middleware overhead deltas (middleware suite)

Each server-level middleware axis (`gzip_enabled`, `access_log`,
`metrics_endpoint`, `use_structured_logging`, `tracing_server`) is
measured on and off against the same model and load; the delta is the
per-request middleware cost. Tracing is measured with an unreachable
collector endpoint (`tracing_no_collector`) — the realistic
misconfiguration cost: span creation plus batched export retries.

## Codec variants (codec suite)

Input `parameters.content_type` selects the decoding codec. The suite
measures `np` (JSON lists, the default) on any model and `pd` (pandas
Series construction) on pandas-native runtimes (sklearn).
`str`/`base64`/`datetime` content types apply to BYTES/string payloads;
with a numeric-only zoo they are recorded as documented skips rather
than error cases. sklearn's `predict_fn` selection
(`predict`/`predict_proba`/`transform`) is swept through the same
`parameters.extra` mechanism.

## gRPC dataplane (grpc suite)

The gRPC suite drives the real gRPC dataplane (proto modules loaded
standalone from the MLServer checkout — no host/server library skew):
typed `contents` fields vs `raw_input_contents` bytes, crossed with
the server's `grpc_raw_input_buffers` zero-copy parsing path, plus
`grpc_max_message_length` limits. A REST JSON baseline is measured at
identical concurrency for the protocol delta.

## CPU vs GPU deployment matrix (hardware suite)

The hardware suite compares what a customer actually chooses between:
the **CPU image on CPU** and the **CUDA image on GPU** — two arms of
the same model, same request-batch ladder, same load, each on its
intended hardware:

- **CPU arm** — the CPU image (digest-pinned default), no `--gpus`,
  no GPU guard: CPU is the intended configuration, not a fallback.
- **GPU arm** — the run's GPU image, `--gpus all`, with the
  anti-silent-fallback guard ENFORCED
  (`FAILED_GPU_PROVIDER_FALLBACK` if CUDAExecutionProvider is not
  actively executing).

Both arms run the images' **stock provider behaviour** — no
`parameters.extra` pinning — because the images' defaults are the
thing being compared. The request-batch-size sweep produces the
crossover curve: small batches are latency-bound (where CPU can win),
large batches are throughput-bound (where GPU dominates). Comparison
cases carry `speedup_throughput` and `p50_delta_ms` per batch size.

## Round interleaving and remaining sequential axes

`--rounds N` reverses paired-arm order on odd rounds (middleware
states, processing variants): A,B,A,B across rounds, so slow drift
affects both arms equally. Axes that STILL run sequentially within a
round (drift-sensitive until interleaved):

- `framework` axes (each is a separate patched-model server)
- `codec` dtype/NaN/predict_fn comparisons (separate models)
- `hardware` arms (CPU image vs GPU image)
- `batching` cells (separate patched servers)
- `workers` ladder (separate servers)
- `faults` axes

The ceiling/httptools A/B in `tools/ceiling_probe.py` interleaves
explicitly (alternating order, 3 repeats).

## Output-size bounds on codec measurements

Codec-axis deltas (NaN-to-None loop, int-vs-float encoding) are
bounded by the measured models' output size — the small tier emits
80 elements per response (8 rows × 10 features). The NaN loop cost
scales with output elements; larger tiers amplify it. State the
output size when quoting codec numbers.
