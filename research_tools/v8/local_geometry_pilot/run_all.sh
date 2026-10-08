#!/usr/bin/env bash
set -euo pipefail
PILOT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PILOT_REPO"
export PYTHONUNBUFFERED=1
exec "$PILOT_REPO/.venv/bin/python" -u -m research_tools.v8.local_geometry_pilot.runner "$@"
