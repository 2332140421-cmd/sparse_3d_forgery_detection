#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$repo/src:$repo:${PYTHONPATH:-}"
exec env PYTHONUNBUFFERED=1 "$repo/.venv/bin/python" -u -m research_tools.v7.component_geometry_time_order_pilot.runner --resume "$@"
