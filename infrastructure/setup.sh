#!/usr/bin/env bash
# Full environment bootstrap: cluster -> ready-to-benchmark.
#
# Steps:
#   1. Prerequisites (oc, python3, numpy/aiohttp/grpcio for the client)
#   2. Namespace, RBAC, PVCs
#   3. Build and deploy the client pod (tools + payloads + references)
#   4. Upload models to the models PVC
#   5. Scale the GPU serving deployment
#   6. Capture frozen references (--capture-refs) and cross-check
#   7. Smoke test (one gated trial)
#
# Idempotent: re-running skips completed steps.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
NAMESPACE="${BENCHMARK_NAMESPACE:-mlserver-perf}"
OC="oc -n ${NAMESPACE}"

log() { echo "[setup] $(date -u +%FT%TZ) $*"; }
die() { echo "[setup] FAILED: $*" >&2; exit 1; }

# ------------------------------------------------------------------ step 1
log "step 1: prerequisites"
command -v oc >/dev/null || die "oc not found in PATH"
command -v python3 >/dev/null || die "python3 not found"
$OC whoami >/dev/null 2>&1 || die "not logged in (oc login first)"
python3 -c "import aiohttp, grpc, numpy" 2>/dev/null \
    || die "workstation python needs: aiohttp grpcio numpy (see pyproject.toml)"
log "  prerequisites OK"

# ------------------------------------------------------------------ step 2
log "step 2: namespace, RBAC, PVCs"
$OC get namespace "$NAMESPACE" >/dev/null 2>&1 || $OC create namespace "$NAMESPACE"
$OC apply -f "$REPO/infrastructure/manifests/rbac.yaml" >/dev/null
$OC apply -f "$REPO/infrastructure/manifests/client-build.yaml" >/dev/null
log "  namespace + RBAC + PVCs OK"

# ------------------------------------------------------------------ step 3
log "step 3: build + deploy the client pod"
if ! $OC get deploy/mlserver-client >/dev/null 2>&1; then
    tar -C "$REPO/tools" -cf - sweep.py client.py hook.py \
        | $OC start-build mlserver-client --from-dir=- --wait >/dev/null \
        || die "client image build failed"
    $OC apply -f "$REPO/infrastructure/manifests/client-deploy.yaml" >/dev/null
fi
$OC rollout status deploy/mlserver-client --timeout=300s >/dev/null
CLIENT_POD=$($OC get pods -l app=mlserver-client \
    -o jsonpath='{.items[0].metadata.name}')
[[ -n "$CLIENT_POD" ]] || die "no client pod"
log "  client pod: ${CLIENT_POD}"

# Upload tooling, payloads and references to /store.
$OC exec "$CLIENT_POD" -- mkdir -p /store/tools /store/plans /store/sessions \
    /store/payloads /store/references
for tool in sweep.py client.py hook.py; do
    $OC cp "$REPO/tools/$tool" "$CLIENT_POD:/store/tools/$tool"
done
for dir in payloads/rfdetr-small payloads/mlp-54; do
    for file in "$REPO/$dir"/*; do
        $OC cp "$file" "$CLIENT_POD:/store/payloads/$(basename "$file")"
    done
done
for dir in references/rfdetr-small references/mlp-54; do
    [[ -d "$REPO/$dir" ]] || continue
    for file in "$REPO/$dir"/*; do
        $OC cp "$file" "$CLIENT_POD:/store/references/$(basename "$file")"
    done
done
$OC cp "$REPO/infrastructure/images.json" "$CLIENT_POD:/store/images.json"
log "  /store populated (tools, payloads, references)"

# ------------------------------------------------------------------ step 4
log "step 4: upload models to the models PVC"
$OC apply -f "$REPO/infrastructure/manifests/model-loader.yaml" >/dev/null
$OC wait --for=condition=Ready pod/model-loader --timeout=300s >/dev/null
upload_model() {
    local name="$1"
    $OC exec model-loader -- mkdir -p "/mnt/models/${name}"
    # The loader container has no tar (oc cp needs it) — stream via stdin.
    $OC exec -i model-loader -- \
        python3 -c "import sys; open('/mnt/models/${name}/model.onnx','wb').write(sys.stdin.buffer.read())" \
        < "$REPO/models/${name}/model.onnx"
    $OC exec -i model-loader -- \
        python3 -c "import sys; open('/mnt/models/${name}/model-settings.json','wb').write(sys.stdin.buffer.read())" \
        < "$REPO/models/${name}/model-settings.json"
    log "  model ${name} uploaded"
}
upload_model rfdetr-small
upload_model rfdetr-small-batched
# mlp-54 deploys under the serving name "mlp-small" (model-settings.json)
$OC exec model-loader -- mkdir -p /mnt/models/mlp-small
$OC exec -i model-loader -- \
    python3 -c "import sys; open('/mnt/models/mlp-small/model.onnx','wb').write(sys.stdin.buffer.read())" \
    < "$REPO/models/mlp-54/model.onnx"
$OC exec -i model-loader -- \
    python3 -c "import sys; open('/mnt/models/mlp-small/model-settings.json','wb').write(sys.stdin.buffer.read())" \
    < "$REPO/models/mlp-54/model-settings.json"
log "  model mlp-small uploaded"
$OC delete pod model-loader --grace-period=5 >/dev/null

# ------------------------------------------------------------------ step 5
log "step 5: serving deployments"
$OC apply -f "$REPO/infrastructure/manifests/server-gpu.yaml" >/dev/null
$OC apply -f "$REPO/infrastructure/manifests/server-cpu.yaml" >/dev/null
$OC apply -f "$REPO/infrastructure/manifests/route.yaml" >/dev/null
$OC scale deploy/mlserver-rfdetr-cpu --replicas=0 >/dev/null 2>&1 || true
$OC rollout status deploy/mlserver-rfdetr --timeout=600s >/dev/null
log "  GPU deployment ready (CPU deployment parked at 0)"

# ------------------------------------------------------------------ step 6
log "step 6: verify references"
bash "$HERE/capture-references.sh" --verify-only

# ------------------------------------------------------------------ step 7
log "step 7: smoke test (one gated trial)"
make -C "$REPO" smoke-test NAMESPACE="$NAMESPACE"
log ""
log "SETUP COMPLETE — ready to benchmark"
log "  try: make run-plan DOMAIN=wire-format PLAN=stock-vs-binary"
