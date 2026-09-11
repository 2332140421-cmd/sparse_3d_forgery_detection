#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/projects/sparse_3d_forgery_detection"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m research_tools.v7.boundary_pooling_probe.pipeline \
  --phase all --resume --output-root "$OUTPUT_ROOT"
