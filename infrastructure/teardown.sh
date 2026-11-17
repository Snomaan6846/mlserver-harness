#!/usr/bin/env bash
# Tear the benchmark environment down. Scales deployments to zero and
# deletes the client pod; keeps the namespace, PVCs (models, /store) and
# RBAC so a subsequent `make setup` is fast. Use TEARDOWN_FULL=1 to
# remove everything including PVC data.
set -euo pipefail

NAMESPACE="${BENCHMARK_NAMESPACE:-mlserver-perf}"
OC="oc -n ${NAMESPACE}"

log() { echo "[teardown] $(date -u +%FT%TZ) $*"; }

log "scaling deployments to zero"
$OC scale deploy/mlserver-rfdetr --replicas=0 >/dev/null 2>&1 || true
$OC scale deploy/mlserver-rfdetr-cpu --replicas=0 >/dev/null 2>&1 || true
$OC delete deploy/mlserver-client --ignore-not-found >/dev/null

if [[ "${TEARDOWN_FULL:-0}" == "1" ]]; then
    log "TEARDOWN_FULL: deleting PVCs (models and /store data)"
    $OC delete pvc client-store models --ignore-not-found >/dev/null
    log "TEARDOWN_FULL: deleting RBAC and namespace"
    $OC delete -f "$(dirname "${BASH_SOURCE[0]}")/manifests/rbac.yaml" \
        --ignore-not-found >/dev/null
    $OC delete namespace "$NAMESPACE" --ignore-not-found >/dev/null
fi

log "teardown complete (data preserved; TEARDOWN_FULL=1 to remove)"
