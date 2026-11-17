#!/usr/bin/env bash
# Full domain sweep with checkpointing: run every plan in a domain,
# resuming cleanly if interrupted. Publishable results (3 rounds,
# full concurrency ladders).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOMAIN="${1:-wire-format}"
echo "=== MLServer Benchmarks — full sweep: ${DOMAIN} ==="
make verify

case "$DOMAIN" in
    wire-format|processing|threading|stability|transport)
        make "run-${DOMAIN}" ;;
    *) echo "unknown domain: $DOMAIN (see benchmarks/README.md)"; exit 2 ;;
esac

make validate
make report
echo ""
echo "report: reports/benchmark-report.md"
make checkpoint
