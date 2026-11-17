#!/usr/bin/env bash
# Quick benchmark: prove the environment end-to-end in ~5 minutes.
# One plan, one round, one concurrency — a real gated trial (fresh pod,
# parity gates, receipts) at minimum cost. NOT a publishable result.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== MLServer Benchmarks — quick benchmark ==="
make verify
python3 tools/run.py --domain wire-format --plan stock-vs-binary \
    --rounds 1 --concurrency 1,8
python3 tools/validate.py results/wire-format/stock-vs-binary
python3 tools/report.py results/wire-format/stock-vs-binary
echo ""
echo "report: reports/benchmark-report.md"
echo "next:   make run-plan DOMAIN=wire-format PLAN=stock-vs-binary   # full run"
