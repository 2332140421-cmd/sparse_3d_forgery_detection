#!/usr/bin/env bash
set -euo pipefail
STREAM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$STREAM_REPO"
export PYTHONUNBUFFERED=1
export PYTHONPATH="/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$STREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$STREAM_REPO/.venv/bin/python" -u -m research_tools.v8.full_video_streaming.runner "$@"
