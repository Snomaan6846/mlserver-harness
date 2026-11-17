# Benchmarks

What to measure, organized by domain. Each domain has:

- `generate.py` — the source of truth: defines variants and plans,
  writes `variants.json` + `plans/*.json` (all checked in; regenerate
  with `make generate`)
- `variants.json` — server configurations (deployment, model, image, env)
- `plans/<name>.json` — one benchmark: variant entries + concurrency +
  rounds + timing

Run any plan: `make run-plan DOMAIN=<domain> PLAN=<name>`.

## Domain index

| Domain | Measures | Plans |
|---|---|---|
| [wire-format](wire-format/) | Protocol & encoding efficiency | stock-vs-binary, stock-vs-grpc, binary-fallback, mixed-negotiation |
| [processing](processing/) | Frontend vs worker parse/encode | frontend-vs-worker-gpu, frontend-vs-worker-cpu, worker-scaling, concurrency-saturation, batched-fallback |
| [threading](threading/) | ORT thread/runtime tuning | gpu-thread-sweep, gpu-spinning, gpu-cudnn, cpu-thread-factorial, cpu-grpc-thread-scaling, cpu-spinning, cpu-affinity, cpu-spin-tuning, cpu-latency-c1 |
| [stability](stability/) | Sustained-load reliability | soak-60min, soak-60min-frontend |
| [transport](transport/) | IPC and network path | queue-vs-zmq, route-vs-direct |

## Expected results (Tesla T4, 28-vCPU node — from the RHOAIENG-102092 campaigns)

| Plan | Expectation |
|---|---|
| stock-vs-binary | ~4.0× at C8 (7.9 → 31.5 RPS); GPU 24% → 80% |
| stock-vs-grpc | ~3.9× at C8; C1 4.5× |
| binary-fallback | parity (1.00–1.06×) — JSON clients unaffected |
| frontend-vs-worker-gpu | C1 p50 −6.3%; C8 CPU-ms −24.5%; RPS parity (device-bound) |
| frontend-vs-worker-cpu | 1-row: worker +37%; 100-row: frontend +12%; 1000-row: compute-bound |
| worker-scaling | flat W1–W12 (frontend dispatch ceiling) |
| concurrency-saturation | saturates by C16 |
| gpu-thread-sweep | flat T1–T8 (~31.5 RPS) |
| cpu-thread-factorial | T-dominant (T2→T8 2.7×); W flat within ~3% |
| cpu-spinning | +11% at W4×T8 |
| cpu-affinity / cpu-spin-tuning / gpu-cudnn | parity |
| queue-vs-zmq | zmq +10–11% |
| soak-60min | zero failures; max working set ~1.06 GB |

Numbers differing wildly from these usually mean an environment problem
(see `docs/TROUBLESHOOTING.md`) — or a genuine regression, which is the
point of running this suite.

## Adding a benchmark

Add variants and a plan to the domain's `generate.py`, run it, commit
the generated JSON together with the generator change. No tool changes
are needed.
