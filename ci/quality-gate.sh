#!/usr/bin/env bash
# The LLM quality gate. Same contract as the k6 perf gate: run the measurement,
# compare against thresholds, exit non-zero to block the deploy.
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source ci/models.env; set +a

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
ARTIFACTS="${ARTIFACTS:-artifacts}"
BASELINE="${BASELINE:-eval/baseline/main.json}"
REPEATS="${EVAL_REPEATS:-3}"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "::error::ANTHROPIC_API_KEY is not set -- the quality gate cannot run." >&2
  echo "A missing key must fail the build, not skip the gate." >&2
  exit 2
fi

mkdir -p "$ARTIFACTS"

echo "--- validating golden dataset"
"$PY" -m golden.loader

echo "--- generating + judging (${REPEATS} judge passes)"
"$PY" -m eval.run_eval \
  --out "$ARTIFACTS/results.json" \
  --save-answers "$ARTIFACTS/answers.json" \
  --repeats "$REPEATS"

echo "--- applying thresholds"
set +e
"$PY" -m eval.gate \
  --results "$ARTIFACTS/results.json" \
  --baseline "$BASELINE" \
  --markdown "$ARTIFACTS/report.md" \
  --json-out "$ARTIFACTS/verdict.json"
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  echo "quality gate: PASS"
else
  echo "::error::quality gate BLOCKED this change -- see artifacts/report.md"
fi
exit $rc
