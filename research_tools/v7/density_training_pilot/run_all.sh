#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
exec .venv/bin/python -u -m research_tools.v7.density_training_pilot.runner --stage all --resume "$@"
