#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
exec /root/autodl-tmp/projects/sparse_3d_forgery_detection/.venv/bin/python -u \
  -m research_tools.v8.full_video_training.run \
  --cache /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_global_context_v1 \
  --pretrained /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_component_global_context_pilot_v1 \
  --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_v1 \
  --resume "$@"
