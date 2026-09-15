#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
export PYTHONPATH=src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec .venv/bin/python -u -m research_tools.v7.periodic_requery_probe.runner all --resume --device cuda --frontend-budget-s 3600 --train-budget-s 900 "$@"
