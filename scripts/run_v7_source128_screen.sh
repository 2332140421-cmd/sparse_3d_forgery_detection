#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/projects/sparse_3d_forgery_detection"
OUTPUT_ROOT="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_source128_extension_v1"
PYTHON="$REPO_ROOT/.venv/bin/python"
SESSION="v7-source128-extension"
SCRIPT="$(readlink -f "$0")"

if [[ "${1:-start}" != "--worker" ]]; then
    if screen -ls 2>/dev/null | grep -Eq "[0-9]+\.${SESSION}([[:space:]]|$)"; then
        echo "SOURCE128_SCREEN_SESSION_ALREADY_EXISTS: $SESSION" >&2
        exit 2
    fi
    if [[ ! -x "$PYTHON" ]]; then
        echo "PROJECT_PYTHON_NOT_EXECUTABLE: $PYTHON" >&2
        exit 2
    fi
    run_id="$(date -u +%Y%m%dT%H%M%SZ)"
    log_path="$OUTPUT_ROOT/logs/source128_${run_id}.log"
    mkdir -p "$OUTPUT_ROOT/logs"
    screen -dmS "$SESSION" /bin/bash "$SCRIPT" --worker "$log_path"
    printf 'screen_session=%s\nlog=%s\nprogress=%s/progress.json\n' "$SESSION" "$log_path" "$OUTPUT_ROOT"
    exit 0
fi

log_path="${2:?worker mode requires the preselected log path}"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/state"
exec >>"$log_path" 2>&1
cd "$REPO_ROOT" || exit 2
started_unix="$("$PYTHON" -c 'import time; print(time.time())')"
command=("$PYTHON" -u -m research_tools.v7.source128_extension.runner all
    --resume
    --output-root "$OUTPUT_ROOT"
    --device cuda
    --frontend-budget-s 7200
    --train-budget-s 900
    --log-path "$log_path"
    --wrapper-pid "$$"
    --screen-session "$SESSION")
command_line="$(printf '%q ' "${command[@]}")"
printf '[supervisor] started_unix=%s wrapper_pid=%s\n[supervisor] command=%s\n[supervisor] log=%s\n' \
    "$started_unix" "$$" "$command_line" "$log_path"

"${command[@]}" &
runner_pid=$!
if wait "$runner_pid"; then
    exit_code=0
else
    exit_code=$?
fi

if "$PYTHON" -u -m research_tools.v7.source128_extension.runner record-wrapper-exit \
    --output-root "$OUTPUT_ROOT" \
    --runner-pid "$runner_pid" \
    --wrapper-pid "$$" \
    --exit-code "$exit_code" \
    --started-unix "$started_unix" \
    --log-path "$log_path" \
    --screen-session "$SESSION" \
    --command-line "$command_line"; then
    record_status=0
else
    record_status=$?
fi
if [[ "$record_status" -ne 0 ]]; then
    printf '[supervisor] FAILED_TO_RECORD_EXIT status=%s runner_exit=%s\n' "$record_status" "$exit_code"
fi
exit "$exit_code"
