# ISVC-Based Deployment Plan for the OpenShift Engine

**Status:** Proposed (research complete, evidence-backed)
**Approach:** Replace the hand-rolled Deployment+Service+Route manifests with
RHOAI's native serving stack: process the cluster's mlserver ServingRuntime
templates, create InferenceServices (ISVCs) in RawDeployment mode with PVC
model storage, and benchmark through the real RHOAI serving path.

---

## 1. Why ISVC instead of raw Deployments

| Hand-rolled manifests (today) | ISVC way |
|---|---|
| We own Deployment + Service + Route YAML, nodeSelectors, probes | RHOAI owns the serving path — exactly what users run in production |
| Benchmarks validate MLServer in a synthetic deployment | Benchmarks validate **the real RHOAI serving path** (odh-model-controller + KServe controller + ServingRuntime + ISVC) |
| Probes, ports, security contexts drift from product defaults | Inherit the product's probes (updated by product PRs like #970) and defaults automatically |
| Route created manually | odh-model-controller creates the Route for exposed ISVCs |
| New RHOAI features (auth, metrics dashboards) invisible to benchmarks | Measured as users get them |

## 2. Research evidence (all verified)

### 2.1 The cluster has exactly two mlserver templates

`oc get templates -n redhat-ods-applications` (verified live on
`syedali-cluster-pool-q7c2q`):

| Template | ServingRuntime created | Image (already digest-pinned) | Frameworks |
|---|---|---|---|
| `mlserver-runtime-template` | `mlserver-runtime` | `registry.redhat.io/rhoai/odh-mlserver-rhel9@sha256:cba2c008…` | sklearn, xgboost, lightgbm, onnx |
| `mlserver-cuda-runtime-template` | `mlserver-cuda-runtime` | `registry.redhat.io/rhoai/odh-mlserver-cuda-rhel9@sha256:a5ce74ee…` | onnx only |

The templates are OpenShift Templates in `redhat-ods-applications`
(installed by odh-model-controller from `config/runtimes/mlserver-template.yaml`
+ `mlserver-cuda-template.yaml`, incubating branch). The image values flow
from the `odh-model-controller-parameters` ConfigMap via kustomize
replacement — **the product images arrive digest-pinned, never floating**.

### 2.2 ISVC CRD supports everything the benchmark matrix needs

From the installed CRD (`serving.kserve.io/v1beta1`), `spec.predictor.model`
accepts the **full Container spec inline**: `env`, `envFrom`, `image`,
`resources`, `args`, `ports`, `volumeMounts` — plus `runtime` (ServingRuntime
reference) and `storage`/`storageUri`. Verified against the CRD schema in the
KServe source. This means **the entire per-trial env matrix
(MLSERVER_PARALLEL_WORKERS, MLSERVER_REST_BINARY_TENSOR, ORT thread settings,
…) is expressed directly on the ISVC**, no Deployment patching.

### 2.3 PVC storage is a DIRECT mount (critical for benchmarks)

From the KServe PVC provider documentation and source:

- `storageUri: "pvc://<pvc-name>/<model-dir>/"` references the PVC directly
- **"KServe uses direct PVC volume mounting, where the PVC volume is directly
  mounted to `/mnt/models` in the user container rather than creating a
  symlink"** — the storage-initializer init container (used for s3://, gs://,
  oci://) is **NOT invoked for pvc://**. Per-trial pod restarts do not
  re-copy models: startup stays fast and copy-free, which is exactly what a
  fresh-pod-per-trial benchmark needs.
- Mounted **read-only by default** (`storage.kserve.io/readonly` annotation
  to change) — read-only is what we want for model serving.
- The model lives at `<pvc>/<model-dir>/model-settings.json` — matches our
  zoo layout one model dir per ISVC.

### 2.4 RawDeployment mode (stable pods, predictable resources)

The cluster's kserve default is **Serverless** (verified in
`inferenceservice-config`), but per-ISVC the annotation
`serving.kserve.io/deploymentMode: RawDeployment` selects raw mode — the same
pattern your model-validation ISVCs use. Verified from KServe source:

- Raw mode creates a plain `Deployment` named `<isvc>-predictor` and a
  `Service` of the same name (constants.PredictorServiceName)
- Predictor pods carry the label `serving.kserve.io/inferenceservice: <isvc>`
  — the hook's fresh-pod UID proof and pod-log access target this label
- No Knative revision churn; pods are stable, resources are honored
- `networking.kserve.io/visibility` label controls exposure;
  `cluster-local` keeps an ISVC internal-only

### 2.5 odh-model-controller adds the OpenShift path — routes are OPT-IN

From the controller source (incubating branch,
`kserve_raw_route_reconciler.go` + `internal/controller/constants`):

- A Route is created **only** when the ISVC carries the label
  `networking.kserve.io/visibility: exposed`. Any other value — or no
  label at all — logs "Skipping route creation" and returns nil.
- `DeleteOwnedRouteIfNotExposed`: removing the label deletes the
  controller-owned Route on the next reconcile.
- Raw ISVCs are **private by default** (network policy reconciler:
  "absent visibility and explicit ClusterLocal" both mean private).
- The Route is edge-TLS; `enable-auth-oauth` on the ISVC switches it to a
  secured route (passthrough with OAuth proxy).

Design consequence: benchmark ISVCs carry **no visibility label by
default** — no Route, pure in-cluster service access, no cluster
pollution. The transport domain's route-vs-direct plan is the only place
that labels its ISVC `exposed`, measures through the controller-created
Route, and removes the label after (the controller cleans up the Route
itself).

### 2.6 The reconciler gives us fresh pods for free

Patching `spec.predictor.model.env` on the ISVC triggers the KServe
controller to roll a new `Deployment` generation → a new pod with a new UID.
The old pod terminates. The hook's discipline (capture old UID → patch →
wait for new ready pod → prove old UID gone) maps 1:1 onto ISVC env patching.

## 3. Proposed architecture — minimal ownership

**We own exactly two things: the PVC and the ISVC.** Everything else —
Deployment, Service, rollout, probes, Route lifecycle, pod cleanup,
network posture — is handled by KServe and odh-model-controller. This is
the entire point of the ISVC way: the benchmark exercises the real
serving path and owns none of it.

| Concern | Owner |
|---|---|
| Models PVC (create once, populate via model-loader) | **us** |
| InferenceService per trial (create, patch env/image, label) | **us** |
| Deployment + Service creation, rollout, replica management | KServe controller |
| Probes (startup/readiness/liveness — product-managed) | ServingRuntime template |
| Route creation/deletion (opt-in via visibility label) | odh-model-controller |
| Pod lifecycle, ownerRefs, garbage collection on ISVC delete | KServe controller |
| Network posture (private by default) | odh-model-controller |
| Reference ServingRuntime (`mlserver-runtime` / `mlserver-cuda-runtime`) | cluster (from the templates — used AS-IS, not copied) |

### What we DROP from the current infrastructure in ISVC mode

- `server-gpu.yaml` / `server-cpu.yaml` (hand-rolled Deployments) — gone
- Our Route manifest — gone (opt-in label instead)
- RBAC shrinks: ISVC CRUD + PVC + pod read/logs only — **no Deployment
  patching permissions at all**

### Flow

```
make setup (ISVC mode)
  1. Prerequisites: RHOAI installed, ISVC CRD + the two mlserver
     templates present — ABORT if missing (per requirement)
  2. Create the models PVC (once); populate models (model-loader pattern)
  3. RBAC for the sweep-hook ServiceAccount: ISVC + PVC + pods
  4. Capture parity references once (--capture-refs via a bootstrap ISVC)
  5. Network path declared up front (incluster -> port-forward from the
     runner host; route -> label exposed, benchmark the route URL from
     the runner host). NO Route unless the path (or the route-vs-direct
     suite) requires one

Per trial (tools/hook.py, ISVC mode):
  1. Capture current predictor pod UID (label
     serving.kserve.io/inferenceservice=<isvc>)
  2. Patch the ISVC:
       spec.predictor.model.env   = <variant's MLSERVER_* env>
       spec.predictor.model.image = <override, digest-pinned>   (only when
                                    an override was provided; absent =
                                    inherit the runtime's baked-in image)
  3. Wait for the KServe reconciler: new Deployment generation -> new
     ready pod (probes are the product's own)
  4. Prove the old pod UID is gone
  5. Provider proof + parity gates — unchanged
  6. Receipts record: image digest (or "inherited from runtime"),
     ISVC generation, pod UID, env

Benchmark ISVC shape (the only object we create per session):

    apiVersion: serving.kserve.io/v1beta1
    kind: InferenceService
    metadata:
      name: <session>
      annotations:
        serving.kserve.io/deploymentMode: RawDeployment
      # no visibility label -> private, no Route
    spec:
      predictor:
        model:
          modelFormat:
            name: <sklearn|xgboost|lightgbm|onnx>
          runtime: mlserver-runtime          # or mlserver-cuda-runtime
          storageUri: pvc://<models-pvc>/<model-dir>/
          env: [...]                          # the trial matrix
          # image: <override only>            # digest-pinned

Teardown: delete the ISVC — KServe garbage-collects everything it owns.
```

### Network path x load-generator location: two orthogonal options

Every cluster benchmark run declares TWO settings up front — they are
the baseline for the whole run, not per-suite options:

    network_path:   incluster | route
    load_generator: host | client-pod

The four combinations each answer a different question:

| | in-cluster path (service DNS) | route path (edge TLS) |
|---|---|---|
| host runner | port-forward to svc; tunnel in path. Correctness, parity, smoke, CI, dev. | Route URL from the host. User experience including the runner's own network. |
| client pod | Pod-to-pod direct, no tunnel, no route. Server performance isolated — the campaign classic. | Route from inside the cluster. Route overhead isolated from the runner's network — the gate8 pattern (route vs direct from the same client). |

- incluster (default path): no Route is created (no visibility label).
  With `load_generator: host`, the runner establishes `oc port-forward`
  to the ISVC's predictor Service and benchmarks against the forwarded
  local ports:

      oc port-forward svc/<isvc>-predictor 18080:8080 18081:8081 18082:8082
      # REST    -> http://127.0.0.1:18080
      # gRPC    -> 127.0.0.1:18081
      # metrics -> http://127.0.0.1:18082 (telemetry scraping)

  With `load_generator: client-pod`, the sweep runs inside the client
  pod (deployed on demand) driving the service DNS directly — the
  tunnel-free, route-free server-isolation mode.

- route: the ISVC is labeled `networking.kserve.io/visibility:
  exposed`; the route host comes from the ISVC status. From the host,
  this is the full user path; from the client pod, it isolates the
  route's proxy cost (the runner's network is excluded).

The client pod is OPTIONAL infrastructure — deployed only when
`load_generator: client-pod`. Host mode needs nothing but `oc` (the
port-forward is a background process the runner manages). The load
generator is location-agnostic: localhost-forwarded ports and route
URLs look identical to it.

**Protocol constraint — the route path is REST-only.** gRPC does not
work through OpenShift edge routes with MLServer's cleartext h2c
backend (campaign finding ROUTED_G_UNAVAILABLE: passthrough fails the
TLS handshake, edge forwards HTTP/1.1 only, h2c preface is rejected —
verified in results/phase2/gate8-completion/grpc-route-result.md).
Therefore:

| Path | REST | gRPC |
|---|---|---|
| incluster + host (port-forward) | ✓ | ✓ |
| incluster + client-pod | ✓ | ✓ |
| route (host or client-pod) | ✓ | ✗ ROUTED_G_UNAVAILABLE |

The runner validates protocol/path compatibility at plan-validation
time and fails fast (`FAILED_PATH_PROTOCOL`) — a gRPC plan with
`network_path: route` is rejected before any cluster interaction, not
discovered at runtime through connection errors. The parity gate keeps
its dual-protocol check (REST + gRPC) only on the in-cluster paths; on
the route path it gates REST only and records the gRPC exclusion in
the receipt.

- **incluster** (default): no Route is created (no visibility label).
  The runner establishes `oc port-forward` to the ISVC's predictor
  Service and benchmarks **from the runner host machine** against the
  forwarded local ports:

      oc port-forward svc/<isvc>-predictor 18080:8080 18081:8081 18082:8082
      # REST    -> http://127.0.0.1:18080
      # gRPC    -> 127.0.0.1:18081
      # metrics -> http://127.0.0.1:18082 (telemetry scraping)

- **route**: the ISVC is labeled `networking.kserve.io/visibility:
  exposed`; the runner reads the route host from the ISVC status /
  `oc get route` and benchmarks **from the runner host machine**
  through the route URL (https, edge TLS — the exact path users hit).

Both paths run the load generator on the runner host — the same
execution model as the Docker engine. The client-pod pattern is not
used in ISVC mode.

**Comparability rule**: results are only comparable within the same
(network_path, load_generator) pair. The manifest records both;
`make compare` warns across either differing axis.

**route-vs-direct** remains as the ONE suite that measures both paths
A/B (it is the qualification comparison); every other suite runs on
the declared path.

**Measurement semantics — what runner-host latency includes**:
benchmarking from the runner host captures the FULL path (host network +
tunnel or route + server). This is the desired semantics for
route-mode (user-experience benchmark: a real client sees exactly this).
It is distortion for server-isolation questions in port-forward mode —
and dangerously so: when the tunnel saturates before the server, both
variants of a paired A/B are equalized by the common bottleneck and
real server-side differences become invisible. Mitigations:

1. Network baseline calibration: measure a minimal endpoint
   (GET /v2/health/ready) through the same path before each run;
   record the floor (median/p99) in the manifest; report server-side
   latency as measured - baseline (approximate, clearly labeled)
2. Port-forward mode is scoped to correctness gates, parity, smoke,
   CI, dev iteration — never saturation/throughput claims
3. Route mode reports full user-experience numbers + the baseline
4. Server-isolation benchmarks (paired A/B on serving changes) use the
   in-cluster client pod — the campaign methodology's reason for
   putting the load generator inside the cluster in the first place

**Fidelity caveat (documented, not hidden)**: `oc port-forward`
tunnels through the Kubernetes API server — it is a development and
sanity path, not a high-throughput data path. At high RPS or with
multi-MB payloads (e.g. RF-DETR's 15 MB requests) the tunnel itself
becomes the bottleneck and distorts measurements. Therefore:

- host + port-forward (in-cluster path): right for correctness gates,
  parity checks, smoke tests, CI, and low-rate development iteration
- host + route: a real network path — user-experience numbers from
  the runner's location
- client-pod + in-cluster path: server-isolation benchmarks (the
  campaign mode — paired A/B on serving changes)
- client-pod + route: route-overhead isolation (the gate8 mode)

### Open item: the gRPC port (to verify, not to own)

The product mlserver template declares only REST 8080. MLServer serves
gRPC on 8081 by default. The ISVC's `predictor.model` accepts the full
Container spec (including `ports`), so adding
`ports: [{containerPort: 8081}]` + `MLSERVER_GRPC_PORT=8081` env on the
ISVC may propagate to the merged container and Service. Verification
step 5 proves or disproves this. Fallback (only if disproved): one
materialized ServingRuntime copy that adds the port — still owned by the
platform's controller; we would merely provide the runtime reference.

## 4. Trial matrix mapping (what changes, what doesn't)

| Concern | Today (raw Deployment) | ISVC mode |
|---|---|---|
| Env matrix (MLSERVER_*) | Patch Deployment env | Patch ISVC `predictor.model.env` |
| Image | Patch Deployment image | Patch ServingRuntime image (once) or ISVC `predictor.model.image` per-variant |
| Model | PVC mounted in Deployment | `storageUri: pvc://<pvc>/<model-dir>/` — direct RO mount |
| Fresh pod | Scale/rollout trick | Env patch → reconciler rollout (natural) |
| Pod targeting | `app=mlserver-rfdetr` label | `serving.kserve.io/inferenceservice=<isvc>` label |
| Route | Our manifest | Controller-created; host from ISVC status |
| GPU | `nvidia.com/gpu` request on Deployment | `predictor.model.resources` on ISVC |
| Node placement | nodeSelector (none, post-fix) | Same via `predictor.affinity`/nodeSelector if needed |
| Parity gates, receipts, sweep engine | Unchanged | Unchanged |

## 5. Image handling: precedence matrix

| Mode | Image provided | Resolution |
|---|---|---|
| Local (Docker) | none | Default floating tag (`odh-master`) → resolved to the latest sha256 at run start |
| Local (Docker) | tag | Resolved to its sha256 at run start |
| Local (Docker) | `@sha256:` | Used as-is |
| Cluster (ISVC) | none | **The cluster's serving runtime template baked-in image, used as-is** — the template has higher precedence than any engine default |
| Cluster (ISVC) | tag | That image, resolved to its sha256 |
| Cluster (ISVC) | `@sha256:` | Used as-is |

Rule: a benchmark never runs a floating tag; every tag resolves to its
immutable digest at run start. The only un-resolved case is the ISVC
inherit — the product templates arrive digest-pinned already.

Implemented (both engines, `bench/images.py` as the single policy):

- `bench/docker_server.resolve_digest()`: at run start, `docker pull` the
  tag if needed, resolve `RepoDigests[0]` → `repo@sha256:…`, pin it for
  every container run. Locally-built images fall back to the image ID.
  Verified live: `quay.io/opendatahub/mlserver:odh-master` →
  `…@sha256:c65f80da1fc97b95c1f3aaad32ba2a9803650ffde618720614a871312691a3ff`
- `tools/run.py` (OpenShift engine): resolves every variant image to its
  digest at session start and uploads digest-pinned variants.json
- ISVC mode follows the same rule: the template's image (or the
  `--image`/`IMAGE=` override) is resolved to a digest once at setup and
  patched into the materialized ServingRuntime; the ISVC never references a
  floating tag.

## 6. Verification plan (cluster proof, run when approved)

Each step is read-then-write, independently checkable:

1. **Templates present** (read-only): `oc get templates -n
   redhat-ods-applications | grep mlserver` — expect the two templates
2. **Materialize a ServingRuntime**: process the CPU template, override the
   image to the digest-pinned benchmark image, add gRPC port, apply as
   `mlserver-bench-cpu`; verify `oc get servingruntime mlserver-bench-cpu`
   and that the KServe controller accepts it
3. **Create one ISVC** (RawDeployment annotation, `pvc://` storageUri, a
   small zoo model): verify the reconciler creates
   `<isvc>-predictor` Deployment + Service; pod becomes ready; **verify
   /mnt/models is the direct PVC mount** (no storage-initializer init
   container in the pod spec — the proof for §2.3)
4. **Env patch = fresh pod**: patch `predictor.model.env` with a trivial
   change; verify a new pod UID replaces the old one (the fresh-pod
   discipline proof for §2.6)
5. **Parity gate through the ISVC path**: run the hook's REST + gRPC parity
   against the ISVC service (both ports)
6. **Route (opt-in)**: verify NO route exists for the default ISVC;
   label it `networking.kserve.io/visibility: exposed`; verify the
   controller-created Route and its host; run one request through it;
   remove the label and verify the controller deletes the Route
7. **Trial sweep**: `make run-plan DOMAIN=wire-format PLAN=stock-vs-binary`
   in ISVC mode — full gated session, receipts carry ISVC generation + pod
   UID + image digest

## 7. Risks and open considerations

| Risk | Mitigation |
|---|---|
| Serverless is the cluster default | Every benchmark ISVC carries `serving.kserve.io/deploymentMode: RawDeployment` (same as the model-validation ISVCs) |
| Product mlserver template exposes REST 8080 only | Our materialized ServingRuntime copy adds the gRPC 8081 port (gRPC benchmarks need it) |
| PVC topology (RWO + WaitForFirstConsumer) | All benchmark ISVCs + the client pod schedule against the same PVC; node pinning only if the storage class requires it |
| ISVC webhook restrictions on env | Verified: `predictor.model.env` is a plain Container field in the CRD; if a stricter webhook ships, fall back to envFrom a ConfigMap the hook patches |
| Reconciler adds latency to per-trial restarts | Measured in verification step 4; the rollout wait replaces our current patch+wait (expected comparable, a few seconds) |
| odh-model-controller version skew | Setup verifies template presence and aborts (per requirement); template processing reads whatever the cluster actually has |

## 8. What this does NOT change

- The Docker engine (`bench/`) — untouched
- The gates: bit-exact REST / byte-exact gRPC parity, fresh-pod UID proof,
  provider proof, receipts, paired alternating rounds
- The campaign tooling: sweep.py, client.py, sampler, summarize, analyze,
  validate, report, checkpoint/resume
- The plans and variant definitions (only the service/DNS names and the
  deployment mode field)

## 9. Implementation outline

1. `infrastructure/isvc.py` (or extend setup.sh): template discovery +
   ServingRuntime materialization + image digest override + gRPC port
2. `tools/hook.py`: ISVC deployment mode (env patch + reconciler wait +
   UID proof; the parity/receipt core is shared)
3. `tools/run.py`: ISVC-mode service-name resolution (`<session>-predictor`)
4. `benchmarks/*/generate.py`: `deployment: isvc` variants (service names
   from the ISVC convention)
5. RBAC additions: ISVC/ServingRuntime/Route access for the sweep-hook
   ServiceAccount
