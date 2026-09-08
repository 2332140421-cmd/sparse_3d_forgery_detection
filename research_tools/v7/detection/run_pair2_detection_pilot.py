"""Materialize-independent final evaluation/report for the V7 Pair2 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_pair2_detection import evaluate_from_frontend
from .protocol import GENERATORS, MODEL_NAMES, read_json, write_json


def _conclusion(metrics: dict[str, object], media_rows: list[dict[str, object]]) -> str:
    real = {str(row["source_id"]) for row in media_rows if row.get("status") == "MEDIA_VALID" and row.get("role") == "real"}
    fake = [row for row in media_rows if row.get("status") == "MEDIA_VALID" and row.get("role") == "fake"]
    generators = {str(row.get("generator")) for row in fake}
    if len(real) < 4 or len(fake) < 8 or len(generators) < 2:
        return "PAIR2_MEDIA_ACCESS_BLOCKED"
    model_rows = metrics["models"]
    aucs = [row.get("auroc") for row in model_rows.values() if row.get("auroc") is not None]
    paired = [row["paired"].get("positive_fraction") for row in model_rows.values() if row["paired"].get("positive_fraction") is not None]
    generator_positive = [
        result["paired"].get("positive_fraction")
        for row in model_rows.values()
        for generator, result in row["per_generator"].items()
        if result.get("paired", {}).get("positive_fraction") is not None
    ]
    if any(value is not None and value >= 0.60 for value in aucs) and sum(value >= 0.60 for value in paired) >= 1 and sum(value >= 0.50 for value in generator_positive) >= 2:
        return "H3_PILOT_SUPPORTED"
    if any(value is not None and value > 0.50 for value in aucs) or any(value > 0.50 for value in paired):
        return "H3_PILOT_INCONCLUSIVE"
    return "H3_CURRENT_REPRESENTATION_NOT_SUPPORTED"


def _fmt(value: object) -> str:
    if value is None:
        return "未计算"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(
    *,
    output: Path,
    metrics: dict[str, object],
    evaluation: dict[str, object],
    media_rows: list[dict[str, object]],
    frontend_meta: dict[str, object],
) -> str:
    conclusion = _conclusion(metrics, media_rows)
    counts = {}
    for row in media_rows:
        counts[row.get("role", "unknown")] = counts.get(row.get("role", "unknown"), 0) + int(row.get("status") == "MEDIA_VALID")
    lines = [
        "# V7 Pair2 real/fake detection pilot",
        "",
        "## 1. Research question",
        "",
        "测试冻结 real-only 三维结构演化正常性在严格 source-paired AI-generated videos 上的 video-level detection signal。",
        "",
        "## 2. Frozen training protocol",
        "",
        "M1=ΔS、M2=Δ²S、M3=[ΔS,Δ²S]；1.0 s true-PTS windows；25/50/75% anchors；组件、S_t、导数、p95 aggregation 和 Gaussian parameters 均未改变。fake 从未进入 fitting，normality model 未重新训练。",
        "",
        "## 3. Pair2 population and lineage",
        "",
        f"MEDIA_VALID real={counts.get('real', 0)}, fake={counts.get('fake', 0)}; generators={', '.join(GENERATORS)}。仅使用冻结 Pair2 source order 与 official pair lineage。",
        "",
        "## 4. Representation operation",
        "",
        f"real summary: `{json.dumps(frontend_meta.get('media_summary', {}).get('real', {}), ensure_ascii=False, sort_keys=True)}`",
        f"fake summary: `{json.dumps(frontend_meta.get('media_summary', {}).get('fake', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 5. Overall detection results",
        "",
        "| model | AUROC | AUPRC | real median/IQR | fake median/IQR | bootstrap |",
        "|---|---:|---:|---|---|---|",
    ]
    for model in MODEL_NAMES:
        row = metrics["models"][model]
        real = row["real_distribution"]
        fake = row["fake_distribution"]
        boot = row["bootstrap_95"]
        bootstrap = f"{_fmt(boot.get('lower_95'))}–{_fmt(boot.get('upper_95'))}" if boot.get("status") == "OK" else str(boot.get("status"))
        lines.append(f"| {model} | {_fmt(row.get('auroc'))} | {_fmt(row.get('auprc'))} | {_fmt(real.get('median'))}/{_fmt(real.get('iqr'))} | {_fmt(fake.get('median'))}/{_fmt(fake.get('iqr'))} | {bootstrap} |")
    lines += ["", "## 6. Paired source results", "", "| model | ΔA median | ΔA IQR | positive fraction |", "|---|---:|---:|---:|"]
    for model in MODEL_NAMES:
        row = metrics["models"][model]["paired"]
        lines.append(f"| {model} | {_fmt(row.get('median'))} | {_fmt(row.get('iqr'))} | {_fmt(row.get('positive_fraction'))} |")
    lines += ["", "## 7. Cross-generator results", ""]
    for generator in GENERATORS:
        lines.append(f"### {generator}")
        for model in MODEL_NAMES:
            row = metrics["models"][model]["per_generator"][generator]
            paired = row["paired"]
            lines.append(f"- {model}: N={row['source_count']}, AUROC={_fmt(row.get('auroc'))}, AUPRC={_fmt(row.get('auprc'))}, paired ΔA={_fmt(paired.get('median'))}, positive={_fmt(paired.get('positive_fraction'))}, status={row['status']}")
        lines.append("")
    lines += ["## 8. Macro cross-generator average", ""]
    for model in MODEL_NAMES:
        macro = metrics["models"][model]["macro_average"]
        lines.append(f"- {model}: AUROC={_fmt(macro.get('auroc'))}, AUPRC={_fmt(macro.get('auprc'))}, generator_count={macro.get('generator_count')}")
    lines += ["", "## 9. Real-only calibrated threshold", "", ""]
    for model in MODEL_NAMES:
        threshold = metrics["thresholds"]["models"][model]
        op = metrics["models"][model]["operating_point"]
        lines.append(f"- {model}: threshold={_fmt(threshold.get('threshold'))}, test-real FPR={_fmt(op.get('test_real_fpr'))}, fake TPR={_fmt(op.get('fake_tpr'))}, per-generator TPR={json.dumps(op.get('per_generator_tpr', {}), sort_keys=True)}")
    lines += ["", "## 9. Limitations", "", "`SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE`; no spatial localization accuracy is claimed. This is a small pilot with the existing frontend/component baseline and M3 covariance conditioning; no synthetic perturbation, localization proxy, or fake-aware tuning was performed.", "", "## 10. Conclusion", "", f"`{conclusion}`", ""]
    report = "\n".join(lines)
    (output / "docs").mkdir(parents=True, exist_ok=True)
    (output / "docs" / "pair2_detection_report.md").write_text(report, encoding="utf-8")
    write_json(output / "run_summary.json", {
        "status": conclusion,
        "h3_conclusion": conclusion,
        "media_valid_real": counts.get("real", 0),
        "media_valid_fake": counts.get("fake", 0),
        "generator_count": len({str(row.get("generator")) for row in media_rows if row.get("status") == "MEDIA_VALID" and row.get("role") == "fake"}),
        "fake_used_for_fitting": False,
        "normality_refit": False,
        "synthetic_perturbation": False,
        "spatial_localization_ground_truth": "SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE",
    })
    return report


def run_pilot(output: Path) -> dict[str, object]:
    media_rows = read_json(output / "manifests" / "pair2_media_manifest.json")
    frontend_meta = read_json(output / "frontend" / "run_meta.json")
    if not isinstance(media_rows, list) or not isinstance(frontend_meta, dict):
        raise ValueError("Pair2 media/frontend artifacts have invalid shape")
    evaluation = evaluate_from_frontend(output / "frontend" / "window_results.json", output)
    report = render_report(output=output, metrics=evaluation["metrics"], evaluation=evaluation, media_rows=media_rows, frontend_meta=frontend_meta)
    return {"conclusion": read_json(output / "run_summary.json")["status"], "report": str(output / "docs" / "pair2_detection_report.md"), "report_bytes": len(report.encode("utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
