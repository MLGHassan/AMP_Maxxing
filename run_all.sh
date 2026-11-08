#!/usr/bin/env bash
# Runs the complete DeepAMP pipeline on macOS / Linux (and Git Bash on Windows).
#
#   bash run_all.sh                 full run
#   bash run_all.sh --quick         ~15 min reduced run, to check the install
#   bash run_all.sh --force         redo every step
#   PY=python3.13 bash run_all.sh   pick a different interpreter
#
# This is a thin wrapper: all the logic lives in run_all.py, so Windows, macOS and
# Linux run exactly the same steps.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${PY:-}" ]; then
  :
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1 && python3 -c '' >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "No usable python found. Set PY=/path/to/python and rerun." >&2
  exit 1
fi
echo "Using interpreter: $PY"
exec "$PY" run_all.py "$@"
