#!/bin/bash
# Manual tmux launcher for this frozen entry. Not executed during closeout.
set -uo pipefail
repo=/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen
result=/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_expanded_spatial_training_v1
if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'Manual terminal/tmux launcher; never runs a monitoring loop.' 'This completed experiment is protected from duplicate execution.' 'For a newly authorized run, prepare its own frozen output before using its launcher.'
    exit 0
fi
cd "$repo" || exit 2
if [[ -e "$result/identity.json" || -e "$result/exit_code.txt" || -e "$result/run.log" ]]; then
    printf '%s\n' "Existing run preserved: $result; refusing duplicate training." >&2
    exit 2
fi
[[ -f "$result/protocol.json" && -f "$result/preparation.json" ]] || exit 2
export PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages:/media/njupt/8FCD3869E2FAB06F/chenyh/envs/rgbd-packages:.
export OMP_NUM_THREADS=4 XFORMERS_DISABLED=1 MPLCONFIGDIR=/tmp/pvst_expanded_spatial_mpl
trap 'rc=$?; printf "%s\n" "$rc" > "$result/launcher_exit_code.txt"; if [[ ! -f "$result/exit_code.txt" ]]; then printf "%s\n" "$rc" > "$result/exit_code.txt"; fi' EXIT
printf '%s\n' "$$" > "$result/runner.pid"
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.xyz_pilot.expanded_spatial run 2>&1 | tee "$result/run.log"
codes=("${PIPESTATUS[@]}")
printf '%s\n' "${codes[0]}" > "$result/exit_code.txt"
if (( codes[0] != 0 )); then exit "${codes[0]}"; fi
exit "${codes[1]}"
