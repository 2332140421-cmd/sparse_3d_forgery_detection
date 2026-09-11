#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/root/autodl-tmp/projects/sparse_3d_forgery_detection"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_dense_local_structure_pilot_gpu_resume_v1"
mkdir -p "$OUTPUT_ROOT"
PID_PATH="$OUTPUT_ROOT/pipeline.pid"
if [[ -s "$PID_PATH" ]]; then
  EXISTING_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    CMDLINE="$(tr '\0' ' ' < "/proc/$EXISTING_PID/cmdline" 2>/dev/null || true)"
    if [[ "$CMDLINE" == *"research_tools.v7.dense_local_structure_pilot.run_pipeline"* ]]; then
      echo "dense pilot already running pid=$EXISTING_PID" >&2
      exit 17
    fi
  fi
fi
LOG_PATH="${DENSE_PILOT_LOG:-$OUTPUT_ROOT/pipeline.log}"
exec >>"$LOG_PATH" 2>&1
echo "$$" > "$PID_PATH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] dense pilot starting pid=$$ resume=$*"
cd "$REPO_ROOT"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] repo=$REPO_ROOT python=$PYTHON_BIN output=$OUTPUT_ROOT"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase=prepare,benchmark,frontend,features,train,export"
exec "$PYTHON_BIN" -u -m research_tools.v7.dense_local_structure_pilot.run_pipeline all --resume --expand-full "$@"
