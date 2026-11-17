# Environment

## Workstation

- Python ≥ 3.10 with `aiohttp`, `grpcio`, `numpy` (`pip install -e .`)
- `oc` (OpenShift CLI), logged in
- The runner shells out to `oc` for everything cluster-side; no other
  cluster tooling is needed

## Cluster

| Requirement | Why |
|---|---|
| OpenShift 4.x with `oc` admin in the target namespace | Deployments, pods, image builds |
| One GPU node (Tesla-T4 class, CC 7.5, ≥16 GB) | GPU domains; also hosts the CPU serving deployment (identical CPU type for controlled comparison) |
| ~205 GiB node RAM / 28+ allocatable CPU | Serving deployment requests the full node |
| Internal image registry | Client image build (`oc start-build`) |

Created by `make setup` (idempotent):

- Namespace `mlserver-perf`
- ServiceAccount `sweep-hook` + Role (patch deployments, read pods/logs
  — the least the hook needs; see `infrastructure/manifests/rbac.yaml`)
- PVC `client-store` (3 Gi) mounted at `/store` in the client pod —
  tools, payloads, references, session outputs
- PVC `models` mounted at `/mnt/models` on serving pods
- Client pod `mlserver-client` (4 CPU / 8 Gi reserved, on a node
  separate from serving) — runs the sweep and load clients
- Serving deployments `mlserver-rfdetr` (GPU, 1× T4) and
  `mlserver-rfdetr-cpu` (same node, providers=CPU, parked at 0)

## DNS names used by plans

Plans address the server through cluster-internal Service DNS:

- `mlserver-rfdetr.mlserver-perf.svc:8080` (REST) / `:8081` (gRPC)
- `mlserver-rfdetr-cpu.mlserver-perf.svc:8080` / `:8081`

The transport domain's `route-vs-direct` plan needs your cluster's
external route hostname. `make setup` applies the Route manifest; read
the assigned host and regenerate the plan:

    oc -n mlserver-perf get route mlserver-rfdetr \
        -o jsonpath={.spec.host}
    BENCHMARK_ROUTE_HOST=<that-host> make generate

## Operating rules

1. **One campaign at a time on the cluster.** The hook scales the
   sibling deployment and Triton to zero on every trial (single GPU);
   concurrent campaigns fight over it.
2. **Never hand-edit `/store` on the pod** — the runner owns it.
3. **Never edit a collected session** — re-run the plan instead
   (old results move to `<plan>-partial-<timestamp>/`).
4. Disk pressure: `make cleanup` reports pod `/store` usage;
   `tools/cleanup.py --pod-storage --prune` clears old sessions.
