#!/usr/bin/env bash
# Research units, API contracts, and TypeScript/build; no datasets, weights, or deployment.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-train/bin/python}
uv pip check --python "$PY"
"$PY" -m pytest -q
npm --prefix app/web ci
npm --prefix app/web audit --audit-level=high
npm --prefix app/web run build
git diff --check
