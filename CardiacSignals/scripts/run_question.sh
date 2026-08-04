#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUESTION="${1:-q1}"

cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" && -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

case "$QUESTION" in
  q1|q2|q3)
    REQUIRED="ecg_signals_preprocessed.csv"
    ;;
  q4)
    REQUIRED="single_ecg_signal.csv"
    ;;
  q5)
    REQUIRED="heart-statlog.csv"
    ;;
  help|-h|--help)
    echo "Usage: bash scripts/run_question.sh [q1|q2|q3|q4|q5]"
    exit 0
    ;;
  *)
    echo "Unknown question: $QUESTION"
    echo "Usage: bash scripts/run_question.sh [q1|q2|q3|q4|q5]"
    exit 1
    ;;
esac

if [[ ! -f "$REQUIRED" ]]; then
  echo "Missing dataset: $REQUIRED"
  echo "Place the dataset in $ROOT, then rerun this command."
  exit 1
fi

"$PYTHON_BIN" "${QUESTION}.py"
