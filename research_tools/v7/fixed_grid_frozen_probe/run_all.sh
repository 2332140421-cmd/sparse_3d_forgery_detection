#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
exec env PYTHONUNBUFFERED=1 .venv/bin/python -u -m research_tools.v7.fixed_grid_frozen_probe.runner --resume "$@"
