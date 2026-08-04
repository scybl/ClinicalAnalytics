#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

if ! "$PYTHON_BIN" -c "import cv2, matplotlib, numpy, pandas, scipy, skimage, sklearn" >/dev/null 2>&1; then
  echo "Missing Python dependencies. Run: bash scripts/setup_env.sh"
  exit 1
fi

for script in q1.py q2.py q3.py q4.py; do
  echo
  echo "== Running $script =="
  "$PYTHON_BIN" "$script"
done
