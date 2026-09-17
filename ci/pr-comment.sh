#!/usr/bin/env bash
# Upsert the gate report as a single PR comment (never a new one per push).
set -euo pipefail

REPORT="${1:-artifacts/report.md}"
MARKER="<!-- rag-eval-gate -->"

[[ -f "$REPORT" ]] || { echo "no report at $REPORT"; exit 0; }
[[ -n "${PR_NUMBER:-}" ]] || { echo "not a PR context; skipping comment"; exit 0; }

existing=$(gh pr view "$PR_NUMBER" --json comments \
  --jq ".comments[] | select(.body | contains(\"$MARKER\")) | .id" | head -1)

if [[ -n "$existing" ]]; then
  gh api -X PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$existing" \
    -f body="$(cat "$REPORT")" >/dev/null
  echo "updated existing gate comment"
else
  gh pr comment "$PR_NUMBER" --body-file "$REPORT"
  echo "posted new gate comment"
fi
