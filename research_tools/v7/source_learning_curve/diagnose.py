"""Finite diagnostics for the completed source-learning-curve pilot.

The diagnostic deliberately does not train or re-run the frontend.  It replays
the saved models in ``eval`` mode on the saved training features, reads the
frozen validation scores, and writes small summaries outside the repository's
Git history.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source_learning_curve import runner as curve


DEFAULT_ROOT = curve.OUTPUT_ROOT
DEFAULT_DIAGNOSTIC_ROOT = DEFAULT_ROOT / "diagnostics_v1"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    # curve.atomic_json rejects NaN/Inf and keeps the diagnostic machine-readable.
    curve.atomic_json(path, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _load_scores(root: Path) -> list[dict[str, Any]]:
    path = root / "scores/validation_window_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["valid_unit_count"] = int(float(row.get("valid_unit_count", 0) or 0))
        row["offset_s"] = float(row["offset_s"])
    return rows


def _source_auroc(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, float]:
    values: dict[str, float] = {}
    by_source: dict[str, list[int]] = {}
    by_score: dict[str, list[float]] = {}
    for row, score in zip(rows, scores):
        source = str(row["source_id"])
        by_source.setdefault(source, []).append(int(row["label"]))
        by_score.setdefault(source, []).append(float(score))
    for source in sorted(by_source):
        value = periodic._auroc(by_source[source], by_score[source])
        if value is not None:
            values[source] = float(value)
    return values


def _source_bootstrap(values: Mapping[str, float]) -> dict[str, Any]:
    keys = sorted(values)
    if not keys:
        return {"source_count": 0, "mean": None, "ci95": [None, None], "sources": []}
    raw = np.asarray([values[key] for key in keys], dtype=np.float64)
    indices = np.random.default_rng(curve.BOOTSTRAP_SEED).integers(
        0, len(raw), size=(curve.BOOTSTRAP_REPLICATES, len(raw))
    )
    draws = raw[indices].mean(axis=1)
    return {
        "source_count": len(keys),
        "sources": keys,
        "mean": float(raw.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "seed": curve.BOOTSTRAP_SEED,
        "replicates": curve.BOOTSTRAP_REPLICATES,
    }


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError(f"row/score length mismatch: {len(rows)} != {len(scores)}")
    values = np.asarray(scores, dtype=np.float64)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("non-finite model score in diagnostic input")
    labels = [int(row["label"]) for row in rows]
    classes = {int(label) for label in labels}
    classification = periodic._classification(labels, values.tolist())
    source_values = _source_auroc(rows, values.tolist())
    result: dict[str, Any] = {
        "window_count": len(rows),
        "real_count": labels.count(0),
        "fake_count": labels.count(1),
        "source_count": len({str(row["source_id"]) for row in rows}),
        "dual_role_source_count": len(source_values),
        "pooled_auroc": periodic._auroc(labels, values.tolist()),
        "pooled_ap": periodic._ap(labels, values.tolist()),
        "classification": classification,
        "source_macro": _source_bootstrap(source_values),
        "single_class": classes != {0, 1},
    }
    return result


def _row_identity_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["window_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["source_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row["label"])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _feature_identity_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["window_id"]).encode("utf-8"))
        values = np.asarray(row["features"]["SET_A"], dtype=np.float64)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _score_column(ordering: int, size: int, seed: int) -> str:
    return f"ordering_{ordering}_size_{size}_seed_{seed}"


def _mean_validation_scores(rows: Sequence[Mapping[str, Any]], ordering: int, size: int, seeds: Sequence[int]) -> tuple[list[float], float]:
    values: list[float] = []
    max_existing_difference = 0.0
    mean_column = f"ordering_{ordering}_size_{size}_logit"
    for row in rows:
        seed_values = [float(row[_score_column(ordering, size, seed)]) for seed in seeds]
        if not all(math.isfinite(value) for value in seed_values):
            raise ValueError(f"non-finite validation score: {row['window_id']} {ordering}/{size}")
        mean_value = float(np.mean(np.asarray(seed_values, dtype=np.float64)))
        existing = _float(row.get(mean_column))
        if existing is not None:
            max_existing_difference = max(max_existing_difference, abs(existing - mean_value))
        values.append(mean_value)
    return values, max_existing_difference


def _metric_csv_row(split: str, ordering: int, size: int, seed: str | int, metrics: Mapping[str, Any], *, loss: Mapping[str, Any] | None = None) -> dict[str, Any]:
    macro = metrics.get("source_macro", {})
    cls = metrics.get("classification", {})
    row: dict[str, Any] = {
        "split": split,
        "ordering_seed": ordering,
        "source_count": size,
        "seed": seed,
        "window_count": metrics.get("window_count"),
        "real_count": metrics.get("real_count"),
        "fake_count": metrics.get("fake_count"),
        "source_count_scored": metrics.get("source_count"),
        "dual_role_source_count": metrics.get("dual_role_source_count"),
        "source_macro_auroc": macro.get("mean"),
        "source_macro_ci_low": (macro.get("ci95") or [None, None])[0],
        "source_macro_ci_high": (macro.get("ci95") or [None, None])[1],
        "pooled_auroc": metrics.get("pooled_auroc"),
        "pooled_ap": metrics.get("pooled_ap"),
        "precision": cls.get("precision"),
        "recall": cls.get("recall"),
        "f1": cls.get("f1"),
        "accuracy": cls.get("accuracy"),
        "tn": cls.get("tn"),
        "fp": cls.get("fp"),
        "fn": cls.get("fn"),
        "tp": cls.get("tp"),
    }
    if loss is not None:
        row.update(loss)
    return row


def _loss_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    history = list(record.get("fit", {}).get("loss_history", []))
    losses = np.asarray([float(item["loss"]) for item in history], dtype=np.float64)
    if losses.size == 0:
        return {"epochs_recorded": 0, "loss_nonfinite_count": None}
    if not np.all(np.isfinite(losses)):
        finite = losses[np.isfinite(losses)]
    else:
        finite = losses
    tail_start_index = max(0, len(losses) - 5)
    tail = losses[tail_start_index:]
    return {
        "epochs_recorded": int(len(losses)),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "minimum_loss": float(np.min(finite)) if finite.size else None,
        "loss_nonfinite_count": int(np.sum(~np.isfinite(losses))),
        "last_five_start_loss": float(losses[tail_start_index]),
        "last_five_delta": float(losses[-1] - losses[tail_start_index]),
        "last_five_range": float(np.max(tail) - np.min(tail)),
        "last_ten_monotone_decrease_steps": int(np.sum(np.diff(losses[-10:]) < 0)) if len(losses) >= 10 else None,
        "recorded_device": record.get("fit", {}).get("device"),
        "elapsed_s": record.get("elapsed_s"),
    }


def _forward_training_record(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = [
        {**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}}
        for row in rows
    ]
    standardizer = periodic._standardizer(record)
    batch = periodic.make_batch("SET_A", values, standardizer, require_labels=False)
    model = curve._model_from_record(record, "cpu")
    try:
        scores = periodic.score_batch("SET_A", model, batch, "cpu")
    finally:
        del model
    if list(batch["window_ids"]) != [str(row["window_id"]) for row in rows]:
        raise ValueError("training forward changed window order")
    return np.asarray(scores, dtype=np.float64)


def _consistency_rows(root: Path, rows: Sequence[Mapping[str, Any]], subsets: Mapping[str, Any], records: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records_by_key = {(int(item["ordering_seed"]), int(item["source_count"]), int(item["seed"])): item for item in records if item.get("status") == "TRAIN_COMPLETE"}
    output: list[dict[str, Any]] = []
    for seed in curve.MODEL_SEEDS:
        first = records_by_key[(curve.ORDER_SEEDS[0], 64, seed)]
        second = records_by_key[(curve.ORDER_SEEDS[1], 64, seed)]
        first_sources = [str(item) for item in subsets["subsets"][str(curve.ORDER_SEEDS[0])]["64"]]
        second_sources = [str(item) for item in subsets["subsets"][str(curve.ORDER_SEEDS[1])]["64"]]
        first_rows = [row for row in rows if str(row["source_id"]) in set(first_sources)]
        second_rows = [row for row in rows if str(row["source_id"]) in set(second_sources)]
        first_std = first["standardization"]; second_std = second["standardization"]
        std_mean_diff = float(np.max(np.abs(np.asarray(first_std["mean"]) - np.asarray(second_std["mean"]))))
        std_scale_diff = float(np.max(np.abs(np.asarray(first_std["scale"]) - np.asarray(second_std["scale"]))))
        state_differences = []
        for name, value in first["state_dict"].items():
            state_differences.append(float(np.max(np.abs(np.asarray(value, dtype=np.float64) - np.asarray(second["state_dict"][name], dtype=np.float64)))))
        first_losses = first["fit"]["loss_history"]
        second_losses = second["fit"]["loss_history"]
        loss_difference = max(abs(float(a["loss"]) - float(b["loss"])) for a, b in zip(first_losses, second_losses))
        first_scores, _ = _mean_validation_scores(validation_rows, curve.ORDER_SEEDS[0], 64, curve.MODEL_SEEDS)
        second_scores, _ = _mean_validation_scores(validation_rows, curve.ORDER_SEEDS[1], 64, curve.MODEL_SEEDS)
        output.append({
            "seed": seed,
            "source_set_equal": set(first_sources) == set(second_sources),
            "source_sequence_equal": first_sources == second_sources,
            "source_sequence_hash_20260909": hashlib.sha256("\n".join(first_sources).encode()).hexdigest(),
            "source_sequence_hash_20260910": hashlib.sha256("\n".join(second_sources).encode()).hexdigest(),
            "effective_row_order_equal": _row_identity_hash(first_rows) == _row_identity_hash(second_rows),
            "feature_identity_equal": _feature_identity_hash(first_rows) == _feature_identity_hash(second_rows),
            "training_window_count_first": len(first_rows),
            "training_window_count_second": len(second_rows),
            "training_real_count_first": sum(int(row["label"]) == 0 for row in first_rows),
            "training_fake_count_first": sum(int(row["label"]) == 1 for row in first_rows),
            "standardization_mean_max_abs_diff": std_mean_diff,
            "standardization_scale_max_abs_diff": std_scale_diff,
            "epoch_config_equal": first["fit"].get("epochs") == second["fit"].get("epochs") == curve.EPOCHS,
            "parameter_count_equal": first.get("parameter_count") == second.get("parameter_count"),
            "recorded_device_first": first["fit"].get("device"),
            "recorded_device_second": second["fit"].get("device"),
            "state_dict_max_abs_diff": max(state_differences),
            "loss_history_max_abs_diff": float(loss_difference),
            "validation_mean_logit_max_abs_diff": float(np.max(np.abs(np.asarray(first_scores) - np.asarray(second_scores)))),
            "interpretation": "same effective source/features/standardization; saved CUDA runs differ numerically, while deterministic CUDA settings are not recorded",
        })
    return output


def _source64_rows(validation_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ordering in curve.ORDER_SEEDS:
        scores, _ = _mean_validation_scores(validation_rows, ordering, 64, curve.MODEL_SEEDS)
        by_source: dict[str, list[int]] = {}
        by_scores: dict[str, list[float]] = {}
        by_support: dict[str, list[int]] = {}
        for row, score in zip(validation_rows, scores):
            source = str(row["source_id"])
            by_source.setdefault(source, []).append(int(row["label"]))
            by_scores.setdefault(source, []).append(float(score))
            by_support.setdefault(source, []).append(int(row["valid_unit_count"]))
        for source in sorted(by_source):
            labels = by_source[source]; values = by_scores[source]
            metrics = periodic._classification(labels, values)
            output.append({
                "ordering_seed": ordering,
                "source_id": source,
                "window_count": len(labels),
                "real_count": labels.count(0),
                "fake_count": labels.count(1),
                "source_auroc": periodic._auroc(labels, values),
                "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],
                "support_valid_windows": len(labels),
                "support_valid_unit_count": int(sum(by_support[source])),
                "logit_min": float(min(values)), "logit_max": float(max(values)),
            })
    return output


def _scale_rows(validation_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ordering in curve.ORDER_SEEDS:
        source_maps: dict[int, dict[str, float]] = {}
        source_counts: dict[int, dict[str, tuple[int, int]]] = {}
        for size in curve.TRAINING_SIZES:
            scores, _ = _mean_validation_scores(validation_rows, ordering, size, curve.MODEL_SEEDS)
            source_maps[size] = _source_auroc(validation_rows, scores)
            source_counts[size] = {}
            for source in sorted({str(row["source_id"]) for row in validation_rows}):
                values = [row for row in validation_rows if str(row["source_id"]) == source]
                source_counts[size][source] = (sum(int(row["label"]) == 0 for row in values), sum(int(row["label"]) == 1 for row in values))
        for left, right in ((16, 32), (32, 64)):
            common = sorted(set(source_maps[left]) & set(source_maps[right]))
            diffs = {source: source_maps[right][source] - source_maps[left][source] for source in common}
            summary = _source_bootstrap(diffs)
            output.append({
                "row_type": "summary", "ordering_seed": ordering, "comparison": f"{right}-{left}", "source_id": "",
                "source_count": summary.get("source_count"), "mean_diff": summary.get("mean"),
                "ci_low": (summary.get("ci95") or [None, None])[0], "ci_high": (summary.get("ci95") or [None, None])[1],
                "bootstrap_seed": curve.BOOTSTRAP_SEED, "bootstrap_replicates": curve.BOOTSTRAP_REPLICATES,
            })
            for source in common:
                left_rows = source_counts[left][source]; right_rows = source_counts[right][source]
                output.append({
                    "row_type": "source", "ordering_seed": ordering, "comparison": f"{right}-{left}", "source_id": source,
                    "auroc_left": source_maps[left][source], "auroc_right": source_maps[right][source],
                    "diff": diffs[source], "n_real": right_rows[0], "n_fake": right_rows[1],
                })
    return output


def _plot_learning_curve(path: Path, metrics: Sequence[Mapping[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {type(exc).__name__}: {exc}"
    fig, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    colors = {curve.ORDER_SEEDS[0]: "#1f77b4", curve.ORDER_SEEDS[1]: "#d62728"}
    styles = {"train": "-", "validation": "--"}
    for ordering in curve.ORDER_SEEDS:
        for split in ("train", "validation"):
            selected = [row for row in metrics if row["split"] == split and int(row["ordering_seed"]) == ordering and str(row["seed"]) == "MEAN_LOGIT"]
            selected.sort(key=lambda row: int(row["source_count"]))
            x = [int(row["source_count"]) for row in selected]
            y = [float(row["source_macro_auroc"]) if row["source_macro_auroc"] not in (None, "") else np.nan for row in selected]
            axis.plot(x, y, marker="o", color=colors[ordering], linestyle=styles[split], label=f"{split}, order {ordering}")
    axis.axhline(0.5, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xlabel("training source count")
    axis.set_ylabel("source-macro AUROC")
    axis.set_title("V7 source-learning curve: saved-model train replay vs validation")
    axis.set_xticks(list(curve.TRAINING_SIZES))
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return "written"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(diag: Path, root: Path, protocol: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]], consistency: Sequence[Mapping[str, Any]], source_errors: Sequence[Mapping[str, Any]], scale_rows: Sequence[Mapping[str, Any]], plot_status: str, input_checks: Mapping[str, Any]) -> Path:
    mean_rows = [row for row in metrics if str(row["seed"]) == "MEAN_LOGIT"]
    lines = [
        "# V7 source-learning-curve finite diagnostic",
        "",
        "本报告只复用已完成的模型、标准化、特征和验证分数。训练阶段为保存模型的 eval-mode 前向重放，不调用 optimizer；未重跑前端、tracking、depth、pose 或 segmentation。",
        "",
        "## 直接结论",
        "",
        "1. 两个 64-source ordering 使用相同的 64-source 集合、相同的有效训练窗口/特征、相同标准化和权重规则；ordering 只改变清单顺序。代码实际按 support 文件顺序筛选行，因此该顺序没有形成不同的训练 batch 顺序。保存的 CUDA 模型状态仍有非零差异，确定性 CUDA 设置未被旧记录保存，故只能把差异归因到已观察的数值/运行非确定性，不能定位到更细的 kernel 原因。",
        "2. 训练集 replay 与留出验证之间的差距见下表。训练指标高而验证指标低支持“拟合较好但跨 source 泛化不足”的解释；若训练和验证都低，则更接近表示/聚合能力有限。该诊断不证明唯一根因。",
        "3. 200 epochs 的 loss history 是否已基本平台化只作描述：所有正式模型是否记录完整、末五 epoch 变化和非有限值见 `train_validation_metrics.csv`。loss 下降本身不等于充分泛化。",
        "4. 规模配对差异按同一 source 进行 bootstrap；不把窗口或 seed 当独立 source。逐 source 变化和 64-source 错误分布用于判断是否由少数 source 主导。",
        "",
        "## 输入与评价口径",
        "",
        f"- 当前仓库 HEAD：`{input_checks['git_head']}`；协议：`{root / 'protocol.json'}`。",
        f"- validation 主集合：{input_checks['validation_source_count']} 个双类别 source、{input_checks['validation_window_count']} 个窗口（{input_checks['validation_real_count']} real / {input_checks['validation_fake_count']} fake）。macro 与 pooled 对每个条件使用同一窗口总体；单类别 source 不进入 source-macro。",
        f"- 正式模型记录：{input_checks['model_count']} 个；每个记录保存 {input_checks['epochs_min']}–{input_checks['epochs_max']} epochs；训练前向重放设备：CPU；原训练记录设备：`{input_checks['recorded_devices']}`。",
        "- 阈值固定为 logit >= 0；没有搜索验证集阈值、没有拟合校准器。source bootstrap 为 10,000 次，seed 20260909。",
        "",
        "## 训练/验证主表（seed 平均 logit）",
        "",
        "| split | ordering | train sources | windows (real/fake) | source-macro AUROC | pooled AUROC | AP | F1 | ACC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        lines.append("| {split} | {ordering_seed} | {source_count} | {window_count} ({real_count}/{fake_count}) | {macro} | {pooled} | {ap} | {f1} | {acc} |".format(
            split=row["split"], ordering_seed=row["ordering_seed"], source_count=row["source_count"], window_count=row["window_count"], real_count=row["real_count"], fake_count=row["fake_count"], macro=_fmt(row["source_macro_auroc"]), pooled=_fmt(row["pooled_auroc"]), ap=_fmt(row["pooled_ap"]), f1=_fmt(row["f1"]), acc=_fmt(row["accuracy"])))
    lines += [
        "",
        "训练 split 的 source-macro 是训练窗口所在 source 中具备 real/fake 两类者的等权均值；validation split 固定使用冻结的 14-source/83-window 主集合。训练指标不是泛化指标。每个 seed 和完整混淆计数均在 `train_validation_metrics.csv`。",
        "",
        "## 两个 64-source 条件",
        "",
        "| model seed | source set equal | effective row order equal | feature identity equal | std max diff | state max diff | loss-history max diff | validation mean-logit max diff |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in consistency:
        lines.append("| {seed} | {source_set_equal} | {effective_row_order_equal} | {feature_identity_equal} | {standardization_mean_max_abs_diff:.2g} | {state_dict_max_abs_diff:.4g} | {loss_history_max_abs_diff:.4g} | {validation_mean_logit_max_abs_diff:.4g} |".format(**row))
    lines += [
        "",
        "64-source records have identical standardization means/scales and the same effective rows. The `training_sources` lists are permutations of the same set, while the training implementation filters the already ordered support rows rather than iterating that list. Therefore the records do not establish an ordering-as-data effect. The nonzero saved-state differences occurred in models recorded as `cuda`; deterministic CUDA settings were not persisted, so a finer attribution is unavailable without the prohibited determinism audit/retraining.",
        "",
        "## 规模配对",
        "",
        "`scale_pair_differences.csv` contains source rows and summary rows for 32-16 and 64-32. Each interval resamples complete sources with the same source-level bootstrap rule; it is not a window-level significance test.",
        "",
        "| ordering | comparison | source count | mean AUROC difference | 95% CI |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in scale_rows:
        if row.get("row_type") == "summary":
            lines.append(f"| {row['ordering_seed']} | {row['comparison']} | {row['source_count']} | {_fmt(row.get('mean_diff'))} | [{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] |")
    lines += [
        "",
        "## 64-source source-level错误分布",
        "",
        "见 `source64_errors.csv`。它按 validation source 列出窗口数、source AUROC、固定零阈值 FN/FP 和有效 unit 支撑；不据此删除 source 或选择模型。",
        "",
        "## 收敛与四类解释",
        "",
        "- A（训练不充分）：只有在 loss 末段仍持续下降且训练指标尚未形成时才得到支持；本轮只报告已有 200-epoch history，不延长训练。",
        "- B（source 覆盖不足）：如果规模配对随 source 数增加而改善，且两个 ordering 在小规模差异大，这是当前数据支持最强的候选之一，但不是充分因果证据。",
        "- C（表示/聚合有限）：若 train replay 也低，或规模增加不带来稳定提升，更接近该解释。",
        "- D（拟合好、跨 source 弱）：若训练 macro/pooled 明显高于固定 validation，而 validation 仍受 source 影响，则支持该解释；这不等于统一部署模型已经泛化。",
        "",
        "本轮只在 A–D 之间做证据排序，不能从现有开发 source 区分 source 分布差异、模型随机/数值差异和表示失效的全部贡献。",
        "",
        "## 下一步（只推荐一个）",
        "",
        "优先扩大独立训练 source 覆盖，同时冻结当前表示、聚合、模型和验证集合。理由是现有学习曲线显示 source 数量变化与小规模 ordering 敏感性是可直接观测的变量；在没有先修复实质实现错误的证据时，不优先延长 200 epochs 或同时改编码器/聚合。该建议不在本轮执行。",
        "",
        "## 产物与边界",
        "",
        f"- 诊断目录：`{diag}`；图：`{diag / 'learning_curve.svg'}`（状态：{plot_status}）。",
        "- 原模型、原 validation 分数和原报告均未覆盖；本轮不提交数据、模型、视频或大数组。",
        "- 未访问旧 R7/V5；未修改正式 `src` 检测链；未运行新实验。",
        "",
    ]
    report = diag / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run(root: Path = DEFAULT_ROOT, diag: Path = DEFAULT_DIAGNOSTIC_ROOT) -> dict[str, Any]:
    started = time.time()
    diag.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    _, subsets = curve._load_split(root)
    support_rows = curve._load_r_rows(root)
    score_rows = _load_scores(root)
    models = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))
    records = [item for item in models.get("records", []) if item.get("status") == "TRAIN_COMPLETE"]
    if len(records) != len(curve.ORDER_SEEDS) * len(curve.TRAINING_SIZES) * len(curve.MODEL_SEEDS):
        raise ValueError(f"expected 24 complete model records, found {len(records)}")
    validation_sources = sorted({str(row["source_id"]) for row in score_rows})
    validation_real_count = sum(int(row["label"]) == 0 for row in score_rows)
    validation_fake_count = sum(int(row["label"]) == 1 for row in score_rows)

    metric_rows: list[dict[str, Any]] = []
    score_reproduction: list[dict[str, Any]] = []
    record_by_key = {(int(item["ordering_seed"]), int(item["source_count"]), int(item["seed"])): item for item in records}
    # Validation metrics are read from the existing saved seed logits.  This
    # does not invoke the model and therefore cannot alter the old scores.
    for ordering in curve.ORDER_SEEDS:
        for size in curve.TRAINING_SIZES:
            seed_scores = []
            for seed in curve.MODEL_SEEDS:
                values = [float(row[_score_column(ordering, size, seed)]) for row in score_rows]
                seed_scores.append(values)
                metric_rows.append(_metric_csv_row("validation", ordering, size, seed, _metrics(score_rows, values)))
            mean_values = np.mean(np.asarray(seed_scores, dtype=np.float64), axis=0).tolist()
            metrics = _metrics(score_rows, mean_values)
            _, max_difference = _mean_validation_scores(score_rows, ordering, size, curve.MODEL_SEEDS)
            score_reproduction.append({"ordering_seed": ordering, "source_count": size, "max_difference_from_saved_mean_logit": max_difference})
            metric_rows.append(_metric_csv_row("validation", ordering, size, "MEAN_LOGIT", metrics))

    # Replay each saved model on its own training rows.  No optimizer is called.
    train_seed_scores: dict[tuple[int, int, int], tuple[list[dict[str, Any]], np.ndarray]] = {}
    for ordering in curve.ORDER_SEEDS:
        for size in curve.TRAINING_SIZES:
            train_sources = set(str(item) for item in subsets["subsets"][str(ordering)][str(size)])
            train_rows = [row for row in support_rows if str(row["source_id"]) in train_sources]
            for seed in curve.MODEL_SEEDS:
                record = record_by_key[(ordering, size, seed)]
                scores = _forward_training_record(record, train_rows)
                train_seed_scores[(ordering, size, seed)] = (train_rows, scores)
                metric_rows.append(_metric_csv_row("train", ordering, size, seed, _metrics(train_rows, scores), loss=_loss_summary(record)))
            stacked = np.vstack([train_seed_scores[(ordering, size, seed)][1] for seed in curve.MODEL_SEEDS])
            mean_scores = stacked.mean(axis=0)
            mean_metrics = _metrics(train_rows, mean_scores)
            mean_loss = _loss_summary(record_by_key[(ordering, size, curve.MODEL_SEEDS[0])])
            metric_rows.append(_metric_csv_row("train", ordering, size, "MEAN_LOGIT", mean_metrics, loss=mean_loss))

    consistency = _consistency_rows(root, support_rows, subsets, records, score_rows)
    source_errors = _source64_rows(score_rows)
    scale_rows = _scale_rows(score_rows)
    _write_csv(diag / "train_validation_metrics.csv", metric_rows)
    _write_csv(diag / "ordering_64_consistency.csv", consistency)
    _write_csv(diag / "source64_errors.csv", source_errors)
    _write_csv(diag / "scale_pair_differences.csv", scale_rows)
    plot_status = _plot_learning_curve(diag / "learning_curve.svg", metric_rows)

    input_checks = {
        "git_head": curve.git_head(),
        "model_count": len(records),
        "validation_source_count": len(validation_sources),
        "validation_window_count": len(score_rows),
        "validation_real_count": validation_real_count,
        "validation_fake_count": validation_fake_count,
        "epochs_min": min(int(item["fit"]["epochs"]) for item in records),
        "epochs_max": max(int(item["fit"]["epochs"]) for item in records),
        "recorded_devices": sorted({str(item["fit"].get("device")) for item in records}),
        "score_reproduction_max_abs_diff": max(float(item["max_difference_from_saved_mean_logit"]) for item in score_reproduction),
        "support_valid_rows": len(support_rows),
    }
    report = _write_report(diag, root, protocol, metric_rows, consistency, source_errors, scale_rows, plot_status, input_checks)
    _write_json(diag / "final_status.json", {
        "status": "COMPLETE",
        "diagnostic": "source_learning_curve_finite_diagnostic_v1",
        "git_head": input_checks["git_head"],
        "model_forward_count": len(records),
        "validation_score_reproduction_max_abs_diff": input_checks["score_reproduction_max_abs_diff"],
        "output_report": str(report),
        "elapsed_s": time.time() - started,
        "boundaries": ["no retraining", "no frontend rerun", "no old R7/V5"],
    })
    return {"status": "COMPLETE", "report": str(report), "diagnostic_root": str(diag), "elapsed_s": time.time() - started, "input_checks": input_checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    args = parser.parse_args()
    result = run(args.root, args.diagnostic_root)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
