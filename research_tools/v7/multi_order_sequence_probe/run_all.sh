#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
exec ./.venv/bin/python -m research_tools.v7.multi_order_sequence_probe.runner all --resume --device cuda --budget-s 3600 "$@"
