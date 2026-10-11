#!/usr/bin/env bash
set -euo pipefail
V8_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$V8_REPO"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$V8_REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$V8_REPO/.venv/bin/python" -u -m research_tools.v8.full_video_global_context.runner "$@"
