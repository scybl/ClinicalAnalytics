#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/show_results.py
