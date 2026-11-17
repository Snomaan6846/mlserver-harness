# Models

SHA-pinned benchmark models (see `MANIFEST.json` in each directory and
`docs/MODELS.md` for the full guide).

| Model | Regime | Serving name | Size |
|---|---|---|---|
| `rfdetr-small/` | Compute-bound (~250 ms GPU) | `rfdetr-small` | 114 MB |
| `rfdetr-small-batched/` | Same + dynamic batching | `rfdetr-small-batched` | 114 MB |
| `mlp-54/` | Frontend-bound (~1.2 ms CPU) | `mlp-small` | 3.0 MB |

`mlp-54/generate.py` rebuilds the model and all its payloads
deterministically from a fixed seed.
