#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo"
py="${V8_PYTHON:-$repo/.venv/bin/python}"
out="${V8_OUTPUT:-/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_v1}"
moge="${V8_MOGE_CHECKPOINT:-/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/checkpoints/moge-2-vits-normal-model.pt}"
tracker="${V8_TRACKER_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/scaled_online.pth}"
mkdir -p "$out/logs"
export PYTHONPATH="${V8_MOGE_SOURCE:-/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe}:${V8_MOGE_PKGS:-/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs}:${V8_COTRACKER_SOURCE:-/root/.cache/torch/hub/facebookresearch_co-tracker_main}:$repo:${PYTHONPATH:-}"
exec stdbuf -oL -eL "$py" -u -m research_tools.v8.full_coverage_observation_field.runner \
  --output "$out" --moge-checkpoint "$moge" --tracker-checkpoint "$tracker" "$@" \
  2>&1 | tee -a "$out/logs/run_all.log"
