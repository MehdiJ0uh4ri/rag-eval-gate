#!/usr/bin/env bash
# The perf gate, kept side by side with the quality gate on purpose: both are
# "measure, compare to a committed threshold, exit non-zero". The only
# difference is that one measures milliseconds and the other measures meaning.
set -euo pipefail

cd "$(dirname "$0")/.."
URL="${1:-http://localhost:8080}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
mkdir -p "$ARTIFACTS"

echo "--- waiting for $URL/healthz"
for _ in $(seq 1 30); do
  if curl -fsS "$URL/healthz" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "$URL/healthz" >/dev/null || { echo "::error::service never became healthy"; exit 2; }

k6 run \
  --env BASE_URL="$URL" \
  --summary-export "$ARTIFACTS/k6-summary.json" \
  perf/k6-rag.js
