#!/usr/bin/env bash
# Default full workflow is launched by the user, never a monitoring daemon.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python
OUT=/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_context_diagnosis_v1
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[i]}" == --output ]]; then OUT="${args[i+1]}"; fi
  if [[ "${args[i]}" == --output=* ]]; then OUT="${args[i]#--output=}"; fi
done
mkdir -p "$OUT/invocations"
stem="$OUT/invocations/$(date +%Y%m%dT%H%M%S)_$$"
# Keep a distinct launcher status if a signal prevents PIPESTATUS collection.
trap 'rc=$?; printf "%s\n" "$rc" > "$stem.launcher_exit_code"; if [[ ! -f "$stem.exit_code" ]]; then printf "%s\n" "UNKNOWN_INTERRUPTED" > "$stem.exit_code"; fi' EXIT
cd "$REPO" || exit 2
export PYTHONPATH="/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages:/media/njupt/8FCD3869E2FAB06F/chenyh/envs/rgbd-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 XFORMERS_DISABLED=1 MPLCONFIGDIR=/tmp/pvst_fullvideo_context_mpl
printf 'Log: %s.log\n' "$stem"
"$PY" -u -m research_tools.xyz_pilot.fullvideo_context_diagnosis "${args[@]}" 2>&1 | tee "$stem.log"
codes=("${PIPESTATUS[@]}")
printf '%s\n' "${codes[0]}" > "$stem.exit_code"
printf 'Python exit=%s, tee exit=%s\n' "${codes[0]}" "${codes[1]}"
if ((codes[0] != 0)); then exit "${codes[0]}"; fi
exit "${codes[1]}"
