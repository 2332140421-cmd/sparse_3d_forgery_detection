#!/usr/bin/env bash
# User-owned terminal/tmux job. No GPU polling, retries or protocol fallback.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python
OUT=/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1
# --output is optional for an explicitly new attempt; old failures are preserved.
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[i]}" == --output ]]; then OUT="${args[i+1]}"; fi
done
mkdir -p "$OUT/invocations"
stem="$OUT/invocations/$(date +%Y%m%dT%H%M%S)_$$"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
printf 'Starting bounded resource preflight. Log: %s.log\n' "$stem"
"$PY" -u -m research_tools.xyz_pilot.fullvideo_direct_probe "${args[@]}" 2>&1 | tee "$stem.log"
statuses=("${PIPESTATUS[@]}")
python_status="${statuses[0]}"
printf '%s\n' "$python_status" > "$stem.exit_code"
printf 'Python exit=%s; tee exit=%s. No formal training started.\n' "$python_status" "${statuses[1]}"
exit "$python_status"
