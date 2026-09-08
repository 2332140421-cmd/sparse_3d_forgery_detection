"""Run frozen-artifact reconstruction followed by the single-condition diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from .analyze_conditional_nsi import OUTPUT_ROOT, analyze
from .artifact_reconstruction import reconstruct


def run(output: Path = OUTPUT_ROOT) -> dict[str, object]:
    summary_path = output / "reconstruction/reconstruction_summary.json"
    if not summary_path.exists():
        reconstruct(output)
    elif json.loads(summary_path.read_text(encoding="utf-8")).get("status") != "NSI_RECONSTRUCTION_VERIFIED":
        raise RuntimeError("NSI_RECONSTRUCTION_MISMATCH")
    return analyze(output)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
