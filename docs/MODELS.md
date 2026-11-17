# Models

Two complementary workloads bracket the practical serving spectrum.
Both are SHA-pinned (`models/<name>/MANIFEST.json`) — a mismatch halts
trust in the references, so verify after any copy:

```bash
python3 - << 'EOF'
import hashlib, json
from pathlib import Path
for manifest_path in sorted(Path("models").glob("*/MANIFEST.json")):
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["files"].items():
        file = manifest_path.parent / name
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        status = "ok" if actual == digest else "MISMATCH"
        print(f"{manifest['model']}/{name}: {status}")
EOF
```

## rfdetr-small — compute-bound regime

| Property | Value |
|---|---|
| Architecture | DINO-DETR object detector (COCO) |
| File | `model.onnx` — 114 MB, ONNX FP32, opset 17, batch 1 |
| Input | `input` [1,3,512,512] FP32 |
| Outputs | `dets` [1,300,4], `labels` [1,300,91] |
| Inference | ~250 ms (Tesla T4) / ~195 ms (28-vCPU CPU, T8) |
| Role | GPU domains; CPU threading domains |

Serving name: `rfdetr-small`. The `rfdetr-small-batched` variant adds
dynamic batching (`max_batch_size=4`) for the batched-model plans.

## mlp-54 — frontend-bound regime

| Property | Value |
|---|---|
| Architecture | Dense network 54→1024→512→256→128→7 |
| File | `model.onnx` — 3.0 MB, ONNX FP32, dynamic batch |
| Input | `input` [N,54] FP32 (N = 1/100/1000 rows per request) |
| Output | `output` [N,7] |
| Inference | ~1.2 ms (CPU) |
| Role | Processing and transport domains (the dispatch ceiling regime) |

Serving name: `mlp-small`. Fully deterministic: `models/mlp-54/generate.py`
rebuilds the model **and all its wire payloads** from a fixed seed —
same seed, same bytes, on any machine.

## Payloads

`payloads/<model>/` carries the prebuilt wire files (uploading payloads,
not generating them on the pod, keeps trials free of generator noise):

- `j0-*.bin` — JSON request bodies (15.26 MB for RF-DETR; 1.1 KB–1.04 MB
  for MLP per row count)
- `triton-binary-*.bin` — KServe v2 binary frames (3.15 MB RF-DETR;
  216 B–212 KB MLP) with the JSON-header length in the
  `Inference-Header-Content-Length` header
- `g-*.bin` — gRPC protobufs (raw input contents)

## References

`references/<model>/` holds the frozen parity outputs (the correctness
contract — see `docs/METHODOLOGY.md`). Missing references for a model
are captured on first setup via `infrastructure/capture-references.sh`.

## Adding a model

1. Create `models/<name>/` with the ONNX file, `model-settings.json`
   and a `MANIFEST.json` (SHA-256 of every file)
2. Add payloads under `payloads/<name>/` (and a generator if the model
   is synthetic)
3. Capture references once: `infrastructure/capture-references.sh`
4. Add variants + plans in the relevant domain's `generate.py`

No tool changes are needed — variants, plans and references are all
data.
