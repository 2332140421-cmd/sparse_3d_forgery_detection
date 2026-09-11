#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/root/autodl-tmp/projects/sparse_3d_forgery_detection"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
exec "$PYTHON_BIN" -m research_tools.v7.dense_local_structure_pilot.run_pipeline all --resume "$@"
