#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/projects/sparse_3d_forgery_detection
cd "$repo"
export PYTHONPATH="$repo/src:$repo"
exec "$repo/.venv/bin/python" -u -m research_tools.v7.visual_structure_pilot.runner --resume "$@"
