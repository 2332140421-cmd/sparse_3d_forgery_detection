#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/root/autodl-tmp/projects/sparse_3d_forgery_detection"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_dense_local_structure_pilot_chunkfix_v1"
mkdir -p "$OUTPUT_ROOT"
LOG_PATH="${DENSE_PILOT_LOG:-$OUTPUT_ROOT/pipeline.log}"
exec >>"$LOG_PATH" 2>&1
echo "$$" > "$OUTPUT_ROOT/pipeline.pid"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] dense pilot starting pid=$$ resume=$*"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -u -m research_tools.v7.dense_local_structure_pilot.run_pipeline all --resume --expand-full "$@"
