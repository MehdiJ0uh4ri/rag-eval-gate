#!/usr/bin/env bash
# Deterministic environment setup for every CI job. Idempotent; safe locally.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${PYTHON:=python3}"
VENV="${VENV:-.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements-eval.txt

# Pre-download the embedding model so it is not fetched inside the timed
# section of the eval (and so a HuggingFace outage fails fast and obviously
# rather than as a mysterious judge timeout 20 minutes in).
set -a; source ci/models.env; set +a
"$VENV/bin/python" - <<'PY'
import os
from sentence_transformers import SentenceTransformer
name = os.environ["RAG_EMBEDDING_MODEL"]
SentenceTransformer(name)
print(f"embedding model cached: {name}")
PY

mkdir -p artifacts
echo "bootstrap ok"
