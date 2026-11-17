# Infrastructure

Where to run: cluster environment management.

| File | Role |
|---|---|
| `setup.sh` | Full bootstrap (idempotent): namespace, RBAC, PVCs, client build+deploy, model upload, serving deployments, reference capture, smoke test |
| `verify.sh` | Read-only readiness check |
| `teardown.sh` | Scale down (data preserved); `TEARDOWN_FULL=1` removes everything |
| `capture-references.sh` | Capture/verify the frozen parity references |
| `manifests/` | RBAC, client build/deploy, GPU + CPU serving deployments, model loader |
| `images/` | Dockerfiles (client image, mlserver-zmq variant) |
| `images.json` | Known image digests (baseline, ship) |

Requirements and layout: `docs/ENVIRONMENT.md`.
