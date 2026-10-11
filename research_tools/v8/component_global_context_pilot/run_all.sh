#!/usr/bin/env bash
set -euo pipefail
CONTEXT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$CONTEXT_REPO"
export PYTHONUNBUFFERED=1
exec "$CONTEXT_REPO/.venv/bin/python" -u -m research_tools.v8.component_global_context_pilot.run "$@"
