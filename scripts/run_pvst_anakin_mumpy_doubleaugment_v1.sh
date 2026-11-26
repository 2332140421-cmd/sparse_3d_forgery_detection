#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived
exec bash "$REPO/scripts/run_pvst_anakin_mumpy_baseline_v1.sh" --output "$DATA/pvst_anakin_mumpy_doubleaugment_v1" --config "$REPO/docs/pvst/anakin_mumpy_doubleaugment_config_v1.json" --cache-root "$DATA/pvst_anakin_mumpy_baseline_v1" "$@"
