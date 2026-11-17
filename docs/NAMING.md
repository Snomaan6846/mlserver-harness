# Naming conventions

Variant and plan names are structured, not decorative — you should be
able to read any result row and know exactly what ran.

## Variant names

```
[model-]track[-modifiers]
```

**Tracks** (the wire path and processing location):

| Track | Meaning | Server setting |
|---|---|---|
| `stock-json` | Plain JSON REST, frontend processing | — (all opt-ins off) |
| `binary-rest` | KServe v2 binary tensor REST, frontend | `rest_binary_tensor=true` |
| `binary-worker` | Binary REST, worker processing | + `rest_worker_binary=true` |
| `json-worker` | JSON REST, worker processing | `rest_worker_json=true` |
| `grpc-stock` | Stock gRPC (protobuf) | — |
| `grpc-raw` | gRPC with raw input buffers | `grpc_raw_input_buffers=true` |

**Modifiers** (appended after the track):

| Suffix | Meaning | Example |
|---|---|---|
| `-wN` | N parallel workers | `mlp-binary-worker-w8` |
| `-tN` | ONNX intra-op threads = N | `cpu-w4t8` |
| `-queue` | multiprocessing-queue transport (vs zmq) | `mlp-json-worker-queue` |
| `-graph` | CUDA graph replay | `gpu-binary-worker-graph` |
| `-spin-on`/`-spin-off` | ORT thread spinning | `cpu-spin-on` |
| `-binon`/`-binoff` | binary tensor on/off (fallback tests) | `json-worker-binon` |

**Model prefixes** (when not obvious from context): variants touching
the MLP workload carry `mlp-`; GPU RF-DETR variants carry `gpu-`;
CPU RF-DETR variants carry `cpu-`.

**Example decomposition:**

```
mlp-binary-worker-w8-queue
 │    │      │     │   │
 │    │      │     │   └── multiprocessing queue transport
 │    │      │     └────── 8 parallel workers
 │    │      └──────────── processing in workers
 │    └─────────────────── binary tensor wire format
 └──────────────────────── MLP-54 workload (CPU, frontend-bound)
```

## Plan names

`<what-is-being-compared>.json` — e.g. `stock-vs-binary`,
`frontend-vs-worker-cpu`, `queue-vs-zmq`, `cpu-thread-factorial`.

## Result layout

```
results/<domain>/<plan>/case-NNNN/          one trial
results/<domain>/<plan>/SUMMARY.json        all trials + the frozen plan
results/<domain>/<plan>/metrics-summary.csv medians per (variant, C)
```

Re-runs of a completed plan move the old directory to
`<plan>-partial-<timestamp>/` — nothing is ever overwritten.
