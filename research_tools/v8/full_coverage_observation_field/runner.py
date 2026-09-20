from __future__ import annotations

import argparse
from .pipeline import run_all


def main() -> None:
    p = argparse.ArgumentParser(description="V8 Full-Coverage Observation Field")
    p.add_argument("--output", default="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_v1/")
    p.add_argument("--identity-root", default="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_component_geometry_pilot_v1/")
    p.add_argument("--source-root", default="/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_source128_extension_v1/")
    p.add_argument("--moge-checkpoint", required=True)
    p.add_argument("--tracker-checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", choices=["all", "frontend", "features", "train", "evaluate", "report"], default="all")
    p.add_argument("--limit-windows", type=int, default=None, help="explicit smoke/debug limit; never used for formal all-stage runs")
    args = p.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
