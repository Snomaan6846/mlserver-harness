#!/usr/bin/env bash
# Capture (or verify) the frozen parity references for every model.
#
# References are the correctness contract: every benchmark trial compares
# its responses byte-for-byte against them. They are captured ONCE,
# deliberately, from a known-good image — never during a normal trial.
#
# Usage:
#   capture-references.sh               # capture missing references
#   capture-references.sh --verify-only # report which exist; capture none
#
# The capture uses a dedicated stock-serving variant per model; the hook's
# --capture-refs mode saves the parity responses as the references.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
NAMESPACE="${BENCHMARK_NAMESPACE:-mlserver-perf}"
VERIFY_ONLY="${1:-}"
OC="oc -n ${NAMESPACE}"

log() { echo "[refs] $(date -u +%FT%TZ) $*"; }
die() { echo "[refs] FAILED: $*" >&2; exit 1; }

CLIENT_POD=$($OC get pods -l app=mlserver-client \
    -o jsonpath='{.items[0].metadata.name}') || die "no client pod"
[[ -n "$CLIENT_POD" ]] || die "no client pod"

# Reference file names the hook expects on /store/references, per model
# and device (see tools/hook.py: {device}{model_stem}-{path}-response.bin).
expect_rfdetr="cuda-rest-response.bin cuda-grpc-response.bin"
expect_batched="cuda-batched-rest-response.bin cuda-batched-grpc-response.bin"
expect_mlp="cpu-mlp-small-rest-response.bin cpu-mlp-small-grpc-response.bin"

existing() { $OC exec "$CLIENT_POD" -- ls /store/references 2>/dev/null || true; }

have() { existing | grep -qx "$1"; }

report() {
    local missing=0
    for ref in $expect_rfdetr $expect_batched $expect_mlp; do
        if have "$ref"; then
            log "  ok: $ref"
        else
            log "  MISSING: $ref"
            missing=$((missing + 1))
        fi
    done
    return $missing
}

# --- GPU references (rfdetr-small + batched) -------------------------------
# The capture variant must serve the stock configuration on the ship
# image so the references reflect the code under test.
capture_gpu() {
    log "capturing GPU references (rfdetr-small, rfdetr-small-batched)"
    $OC exec "$CLIENT_POD" -- rm -rf /store/capture-refs 2>/dev/null || true
    $OC exec "$CLIENT_POD" -- mkdir -p /store/capture-refs

    # Minimal variant table for the capture: stock env on the ship image.
    $OC exec -i "$CLIENT_POD" -- python3 -c "
import json
json.dump({
    'capture-rfdetr': {
        'deployment': 'mlserver-rfdetr',
        'service': 'mlserver-rfdetr',
        'device': 'cuda',
        'expect_cuda': True,
        'model': 'rfdetr-small',
    },
    'capture-rfdetr-batched': {
        'deployment': 'mlserver-rfdetr',
        'service': 'mlserver-rfdetr',
        'device': 'cuda',
        'expect_cuda': True,
        'model': 'rfdetr-small-batched',
    },
}, open('/store/variants.json', 'w'), indent=2)
"
    for variant in capture-rfdetr capture-rfdetr-batched; do
        $OC exec "$CLIENT_POD" -- env BENCHMARK_TRIAL_DIR=/store/capture-refs \
            python3 /store/tools/hook.py "$variant" --capture-refs \
            || die "reference capture failed for $variant"
    done
    log "  GPU references captured"
}

# --- CPU references (mlp-small) ---------------------------------------------
capture_mlp() {
    log "capturing MLP references (mlp-small, CPU)"
    $OC scale deploy/mlserver-rfdetr --replicas=0 >/dev/null
    $OC scale deploy/mlserver-rfdetr-cpu --replicas=1 >/dev/null
    $OC rollout status deploy/mlserver-rfdetr-cpu --timeout=600s >/dev/null
    $OC exec -i "$CLIENT_POD" -- python3 -c "
import json
json.dump({
    'capture-mlp': {
        'deployment': 'mlserver-rfdetr-cpu',
        'service': 'mlserver-rfdetr-cpu',
        'device': 'cpu',
        'expect_cuda': False,
        'model': 'mlp-small',
    },
}, open('/store/variants.json', 'w'), indent=2)
"
    $OC exec "$CLIENT_POD" -- env BENCHMARK_TRIAL_DIR=/store/capture-refs \
        python3 /store/tools/hook.py capture-mlp --capture-refs \
        || die "MLP reference capture failed"
    $OC scale deploy/mlserver-rfdetr-cpu --replicas=0 >/dev/null
    $OC scale deploy/mlserver-rfdetr --replicas=1 >/dev/null
    $OC rollout status deploy/mlserver-rfdetr --timeout=600s >/dev/null
    log "  MLP references captured"
}

report && { log "all references present"; exit 0; }
if [[ "$VERIFY_ONLY" == "--verify-only" ]]; then
    log "missing references (capture with: infrastructure/capture-references.sh)"
    exit 1
fi
capture_gpu
capture_mlp
report || die "references still missing after capture"
log "reference capture COMPLETE"
