# Diagnose: 6-Layer Performance Diagnostic

`make diagnose` runs a comprehensive 6-layer analysis of a running MLServer
instance and produces a reviewable `DIAGNOSIS.md` report with
severity-ranked findings and recommended actions.

## Engine support

Diagnose runs on both engines:

```bash
make diagnose ENGINE=local      # local container (Podman/Docker, default)
make diagnose ENGINE=cluster    # ISVC predictor pod on RHOAI
```

The local engine auto-detects Podman → Docker → nerdctl (Podman preferred).

When `ENGINE=cluster`, the diagnose tool connects to the ISVC's
predictor pod via `oc exec` and runs all 6 layers inside it.
Profiling tools (py-spy, memray) are installed into the pod via the
derived image (see `bench/image_manager.py`).

## Quick start

```bash
# Against a running container
make diagnose CONTAINER=<container-name>

# Quick mode (skip py-spy/memray — faster, less detail)
make diagnose-quick

# Against a local process (no container)
make diagnose
```

## The six layers

| Layer | Question it answers | What it collects |
|---|---|---|
| 1. System | Is any resource saturated? | CPU/memory/IO/network utilization, PSI, context switches, interrupts, thermal |
| 2. Process | What is each thread doing? | Per-thread CPU, thread states, off-CPU analysis, fd count, memory maps |
| 3. On-CPU | Which functions burn CPU? | py-spy flame graphs (native + Python), top-N function attribution |
| 4. Memory | Where do allocations go? | RSS timeline, GC stats, memray allocation profiles |
| 5. Latency | Where does each request's time go? | Prometheus stage timing, queue depth, client/server percentile splits |
| 6. GPU | Is the GPU actually busy? | DCGM SM/memory/thermal, nvidia-smi, kernel gap detection |

## Automated findings

The findings engine evaluates all layers against 12 rules and emits
severity-ranked results:

| Finding | Severity | What it means |
|---|---|---|
| `swap_active` | CRITICAL | Memory swapping — all measurements unreliable |
| `frontend_bound_serving` | HIGH | GPU idle + CPU saturated → serialization is the bottleneck |
| `queue_dominated_latency` | HIGH | >50% of latency outside the model |
| `gil_contention` | HIGH | Multi-threaded Python fighting for the GIL |
| `thermal_throttling` | HIGH | GPU performance limited by temperature/power |
| `fd_exhaustion_risk` | HIGH | File descriptors near the limit |
| `gpu_underutilized` | MEDIUM | SM < 50% — model too small or serving starved |
| `memory_bound_inference` | MEDIUM | Memory bandwidth > 80% — quantization candidate |
| `memory_churn` | MEDIUM | High allocation rate causing GC pressure |
| `thread_thrashing` | MEDIUM | >100K context switches/s |
| `offcpu_blocked` | MEDIUM | Significant threads blocked on locks/IO |
| `iowait_heavy` | MEDIUM | >10% CPU time in iowait |

## Output structure

```
results/diagnose-<timestamp>/
├── DIAGNOSIS.md              ← the reviewable report
├── diagnosis.json            ← machine-readable findings
├── layers/
│   ├── 01_system.json        ← USE method data
│   ├── 02_process.json       ← thread/process analysis
│   ├── 03_oncpu.json         ← py-spy profile + top-N
│   ├── 04_memory.json        ← RSS timeline + GC + memray
│   ├── 05_latency.json       ← Prometheus snapshot
│   └── 06_gpu.json           ← DCGM/nvidia-smi data
└── findings/
    └── findings.json         ← rule engine output
```

## Reference comparison (`make reference`)

Run alongside diagnose for the complete picture:

```bash
make reference           # floor + ceiling + MLServer
make reference-gpu       # with CUDA provider
```

Produces a 3-point comparison:

| Point | What | Significance |
|---|---|---|
| **Floor** | Raw onnxruntime in-process | The irreducible model cost |
| **MLServer** | Our measurement | Where we are |
| **Ceiling** | Triton C++ serving | What a C++ stack achieves |

Report includes: MLServer efficiency (% of ceiling), serving overhead
(ms above floor), optimization headroom (RPS recoverable).

## ISVC mode

For cluster deployments, use `bench/isvc.py` — creates InferenceServices
with PVC model storage. See `docs/ISVC_DEPLOYMENT_PLAN.md` for the full
architecture and design decisions.
