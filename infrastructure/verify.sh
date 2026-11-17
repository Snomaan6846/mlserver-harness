#!/usr/bin/env bash
# Verify the environment is ready to benchmark. Read-only: no changes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
NAMESPACE="${BENCHMARK_NAMESPACE:-mlserver-perf}"
OC="oc -n ${NAMESPACE}"

pass=0; fail=0
check() {  # check <description> <command...>
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  ok:   $desc"; pass=$((pass + 1))
    else
        echo "  FAIL: $desc"; fail=$((fail + 1))
    fi
}

echo "[verify] environment readiness"

check "oc logged in"            $OC whoami
check "namespace exists"        $OC get namespace "$NAMESPACE"
check "client pod running"      $OC get pods -l app=mlserver-client
check "GPU deployment exists"   $OC get deploy/mlserver-rfdetr
check "CPU deployment exists"   $OC get deploy/mlserver-rfdetr-cpu
check "models on PVC"           $OC exec "$($OC get pods -l app=mlserver-client -o jsonpath='{.items[0].metadata.name}')" -- ls /store/payloads/j0-0.bin

CLIENT_POD=$($OC get pods -l app=mlserver-client \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$CLIENT_POD" ]]; then
    check "tools uploaded"      $OC exec "$CLIENT_POD" -- ls /store/tools/sweep.py
    check "variants table"      $OC exec "$CLIENT_POD" -- ls /store/variants.json
    check "references present"  bash "$HERE/capture-references.sh" --verify-only
fi

echo ""
echo "[verify] ${pass} passed, ${fail} failed"
[[ $fail -eq 0 ]] || exit 1
