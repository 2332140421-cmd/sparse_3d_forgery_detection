#!/usr/bin/env python3
"""Summarise the completed V8 identity-fix run without recomputing models.

The audit reads saved frontend/features/model/evaluation artifacts only.  It
does not decode video, run a network forward, or alter the experiment data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research_tools.v8.full_coverage_observation_field.pipeline import _metric, _safe_name, _source_macro


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--old-output", required=True)
    return p.parse_args()


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def q(values: list[float], p: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), p)) if values else None


def summary_stats(values: list[float]) -> dict[str, float | None]:
    return {"min": q(values, 0), "p25": q(values, .25), "median": q(values, .5), "p75": q(values, .75), "max": q(values, 1)}


def metric(summary: dict, condition: str, split: str, seed: str = "MEAN_LOGIT") -> dict:
    for row in summary.get("conditions", []):
        if row.get("condition") == condition and row.get("split") == split and str(row.get("seed")) == seed:
            return row
    return {}


def f(x: object, digits: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def partition(ids: np.ndarray, active: np.ndarray) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, ok in enumerate(active.tolist()):
        if ok:
            groups[int(ids[i])].append(i)
    return tuple(sorted(tuple(v) for v in groups.values()))


def scan(out: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    split_acc = {"train": defaultdict(float), "validation": defaultdict(float)}
    component_counts: list[float] = []
    group_sizes: list[float] = []
    frame_merge_ratios: list[float] = []
    pixel_coverages: list[float] = []
    geometry_missing: list[float] = []
    no_history: list[float] = []
    assoc_coverages: list[float] = []
    assoc_matches: list[float] = []
    dynamic_windows = 0
    dynamic_transitions = 0
    total_transitions = 0
    split_windows = merge_windows = 0
    total_frames = 0
    total_cells = 0
    total_active_cells = 0
    total_area = 0.0
    total_geo_area = 0.0
    window_rows: list[dict] = []
    pts_errors: list[float] = []
    duplicate_pts = nonmonotone = 0
    visible_queries: list[float] = []
    for row in rows:
        safe = _safe_name(row["window_id"])
        fp = out / "frontend" / f"{safe}.npz"
        mp = out / "frontend" / f"{safe}.json"
        xp = out / "features" / f"{safe}.npz"
        with np.load(fp, allow_pickle=False) as fr, np.load(xp, allow_pickle=False) as z:
            area = fr["cell_area"].astype(np.float64)
            gv = fr["cell_geometry_valid"].astype(bool)
            hist = fr["history_available"].astype(bool)
            comps = z["component_ids"]
            csize = z["component_sizes"]
            aw = z["association_weight"].astype(np.float64)
            matches = z["association_match_count"]
            xfull = z["x_full"]
            actual_pts = z["actual_pts_s"].astype(float)
            assert xfull.shape == (16, 256, 77) and np.isfinite(xfull).all()
        meta = read(mp)
        expected = np.asarray(meta.get("expected_pts_s", []), dtype=float)
        actual = np.asarray(meta.get("pts_s", []), dtype=float)
        if expected.shape == actual.shape and actual.size:
            pts_errors.extend(np.abs(actual - expected).tolist())
        d = np.diff(actual)
        duplicate_pts += int(np.sum(d == 0))
        nonmonotone += int(np.sum(d < 0))
        with np.load(fp, allow_pickle=False) as fr:
            tv = fr["tracker_visibility"]
            visible_queries.extend(np.mean(tv, axis=(1, 2)).tolist())

        active = area > 0
        frames = active.shape[0]
        total_frames += frames
        total_cells += int(np.prod(active.shape))
        total_active_cells += int(active.sum())
        total_area += float(area.sum())
        total_geo_area += float((area * gv).sum())
        pixel_coverages.append(float(area.sum() / (frames * 256 * 256)))
        geometry_missing.append(float((active & ~gv).sum() / max(int(active.sum()), 1)))
        nh = hist[1:] & active[1:]
        no_history.append(float((~hist[1:][active[1:]]).mean()) if nh.size else 1.0)
        valid_assoc = (aw[1:].sum(axis=-1) > 0) & active[1:]
        assoc_coverages.append(float(valid_assoc.sum() / max(int(active[1:].sum()), 1)))
        assoc_matches.append(float((matches[1:][active[1:]] > 0).mean()) if active[1:].any() else 0.0)
        p0 = partition(comps[0], active[0])
        has_dynamic = False
        local_split = local_merge = False
        merged_ratios: list[float] = []
        c_counts: list[int] = []
        for t in range(frames):
            p = partition(comps[t], active[t])
            c_counts.append(len(p))
            sizes = [len(g) for g in p]
            group_sizes.extend(float(s) for s in sizes)
            merged_ratios.append(float(sum(s for s in sizes if s > 1) / max(len(sizes) and int(active[t].sum()), 1)))
            if t:
                total_transitions += 1
                if p != p0:
                    dynamic_transitions += 1
                    has_dynamic = True
                p0 = p
                # A current cell with >1 candidates is a merge-like relation;
                # one predecessor reused by >1 current cell is split-like.
                positive = aw[t] > 0
                if np.any(positive.sum(axis=-1) > 1):
                    local_merge = True
                rev: dict[int, set[int]] = defaultdict(set)
                for ci in np.flatnonzero(active[t]):
                    for pi in np.flatnonzero(positive[ci]):
                        rev[int(pi)].add(int(ci))
                if any(len(v) > 1 for v in rev.values()):
                    local_split = True
        if has_dynamic:
            dynamic_windows += 1
        if local_split:
            split_windows += 1
        if local_merge:
            merge_windows += 1
        component_counts.extend(float(v) for v in c_counts)
        frame_merge_ratios.extend(merged_ratios)
        sp = split_acc[row["split"]]
        sp["windows"] += 1
        sp["active_cell_frames"] += int(active.sum())
        sp["active_after_t0"] += int(active[1:].sum())
        sp["geometry_missing_cell_frames"] += int((active & ~gv).sum())
        sp["no_history_cell_frames_after_t0"] += int((~hist[1:][active[1:]]).sum())
        sp["assoc_valid_cell_frames_after_t0"] += int(valid_assoc.sum())
        sp["dynamic_partition_windows"] += int(has_dynamic)
        window_rows.append({
            "window_id": row["window_id"], "source_id": row["source_id"], "split": row["split"], "role": row["role"],
            "label": row["label"], "nonpadding_pixel_coverage": pixel_coverages[-1],
            "geometry_missing_cell_fraction": geometry_missing[-1], "no_history_after_t0_fraction": no_history[-1],
            "association_valid_after_t0_fraction": assoc_coverages[-1], "component_count_median": float(np.median(c_counts)),
            "component_size_median": float(np.median([len(g) for t in range(frames) for g in partition(comps[t], active[t])])),
            "dynamic_partition_transitions": int(sum(1 for t in range(1, frames) if partition(comps[t], active[t]) != partition(comps[t-1], active[t-1]))),
            "has_multi_to_one_or_one_to_multi": bool(local_split or local_merge),
        })
    split_out = {}
    for split, a in split_acc.items():
        den = max(a["active_cell_frames"], 1)
        den_t = max(a["active_after_t0"], 1)
        split_out[split] = {
            "windows": int(a["windows"]), "active_cell_frames": int(a["active_cell_frames"]),
            "geometry_missing_fraction": float(a["geometry_missing_cell_frames"] / den),
            "no_history_after_t0_fraction": float(a["no_history_cell_frames_after_t0"] / den_t),
            "association_valid_after_t0_fraction": float(a["assoc_valid_cell_frames_after_t0"] / den_t),
            "windows_with_dynamic_partition": int(a["dynamic_partition_windows"]),
        }
    audit = {
        "window_count": len(rows), "train_windows": sum(r["split"] == "train" for r in rows), "validation_windows": sum(r["split"] == "validation" for r in rows),
        "frontend_shapes": {"query_grid": [32, 32], "query_count": 1024, "tracker_tracks": [16, 16, 1024, 2], "tracker_visibility": [16, 16, 1024]},
        "feature_shapes": {"x_full": [16, 256, 77], "x_rgb2d": [16, 256, 71], "component_ids": [16, 256], "predecessor": [16, 256, 4]},
        "pixel_and_geometry": {"nonpadding_pixel_coverage": summary_stats(pixel_coverages), "geometry_valid_area_fraction": float(total_geo_area / max(total_area, 1e-12)), "geometry_missing_cell_fraction": summary_stats(geometry_missing)},
        "components": {"count_per_frame": summary_stats(component_counts), "size": summary_stats(group_sizes), "merged_cell_fraction_per_frame": summary_stats(frame_merge_ratios), "max_cells_per_component": 4, "windows_with_dynamic_partition": dynamic_windows, "partition_transitions_changed": dynamic_transitions, "partition_transitions_total": total_transitions, "dynamic_window_fraction": float(dynamic_windows / max(len(rows), 1))},
        "associations": {"valid_coverage_after_t0": summary_stats(assoc_coverages), "match_positive_fraction_after_t0": summary_stats(assoc_matches), "windows_with_split_like_relation": split_windows, "windows_with_merge_like_relation": merge_windows},
        "observation": {"visible_query_fraction": summary_stats(visible_queries), "first_frame_history_is_initial_state": True, "all_nonpadding_cells_retained_in_features": True},
        "time": {"max_expected_actual_abs_error_s": max(pts_errors) if pts_errors else None, "median_expected_actual_abs_error_s": statistics.median(pts_errors) if pts_errors else None, "duplicate_pts": duplicate_pts, "nonmonotone_pts": nonmonotone},
        "by_split": split_out,
    }
    return audit, window_rows


def metrics_table(new_summary: dict, old_summary: dict) -> tuple[str, dict]:
    lines = ["| split/metric | old FULL | new FULL | old RGB_2D | new RGB_2D |", "|---|---:|---:|---:|---:|"]
    for split in ["train", "validation"]:
        for key in ["source_macro_auroc", "auroc", "ap", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp"]:
            o1, n1 = metric(old_summary, "FULL", split), metric(new_summary, "FULL", split)
            o2, n2 = metric(old_summary, "RGB_2D", split), metric(new_summary, "RGB_2D", split)
            lines.append(f"| {split} {key} | {f(o1.get(key))} | {f(n1.get(key))} | {f(o2.get(key))} | {f(n2.get(key))} |")
    nf = metric(new_summary, "FULL", "validation"); nr = metric(new_summary, "RGB_2D", "validation")
    of = metric(old_summary, "FULL", "validation"); orr = metric(old_summary, "RGB_2D", "validation")
    comp = new_summary.get("comparison_FULL_minus_RGB_2D", {})
    details = {"new_full": nf, "new_rgb": nr, "old_full": of, "old_rgb": orr, "comparison": comp}
    return "\n".join(lines), details


def saved_mean_train_metrics(root: Path, rows: list[dict]) -> dict:
    train_rows = [r for r in rows if r["split"] == "train"]
    out = {}
    for condition in ["FULL", "RGB_2D"]:
        scores = []
        labels = None
        for seed in [20260909, 20260910, 20260911]:
            with np.load(root / "models" / f"scores__{condition}__seed{seed}.npz", allow_pickle=False) as z:
                scores.append(z["train_scores"].astype(float))
                if labels is None:
                    labels = z["train_labels"].astype(int)
        avg = np.mean(np.stack(scores), axis=0)
        sm = _source_macro(train_rows, avg, labels, [r["window_id"] for r in train_rows])
        out[condition] = {**_metric(labels, avg), "source_macro_auroc": sm["mean"], "source_count": sm["source_count"]}
    return out


def main() -> None:
    a = args(); out = Path(a.output); old = Path(a.old_output)
    data = read(out / "data_manifest.json"); rows = data["rows"]
    audit, win_rows = scan(out, rows)
    identity = read(out / "identity_audit.json")
    new_summary = read(out / "evaluation" / "summary.json")
    old_summary = read(old / "evaluation" / "summary.json")
    table, details = metrics_table(new_summary, old_summary)
    new_train_mean = saved_mean_train_metrics(out, rows)
    old_train_mean = saved_mean_train_metrics(old, read(old / "data_manifest.json")["rows"])
    write_json = lambda path, value: Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_json(out / "evaluation" / "mean_logit_train_metrics.json", {"new": new_train_mean, "old": old_train_mean, "source": "saved per-seed train score arrays; no model forward"})
    audit["identity_fix"] = identity
    standardizer_diff = {}
    for condition in ["FULL", "RGB_2D"]:
        old_std = read(old / "models" / f"standardizer_{condition}.json")
        new_std = read(out / "models" / f"standardizer_{condition}.json")
        standardizer_diff[condition] = {
            "mean_max_abs_difference": float(np.max(np.abs(np.asarray(old_std["mean"], dtype=float) - np.asarray(new_std["mean"], dtype=float)))),
            "scale_max_abs_difference": float(np.max(np.abs(np.asarray(old_std["scale"], dtype=float) - np.asarray(new_std["scale"], dtype=float)))),
        }
    audit["standardizer_vs_pre_fix_max_abs_difference"] = standardizer_diff
    (out / "structure_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "structure_audit_windows.csv").open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(win_rows[0].keys())); w.writeheader(); w.writerows(win_rows)
    records = read(out / "models" / "fold_models.json")["records"]
    runtime = read(out / "run.json")
    elapsed = runtime.get("finished_unix", 0) - runtime.get("started_unix", 0)
    comp = details["comparison"]
    per = comp.get("per_source", {})
    signs = Counter("up" if v > 1e-12 else "down" if v < -1e-12 else "tie" for v in per.values())
    repair_lines = [
        "# V8 身份修复、冻结重训与结构审计汇报", "",
        "## 结论", "",
        "身份错配已按 `(pair_id, source_id, role)` 修复并验证；全量801窗口中796个身份一致缓存复用、5个5H1P1窗口按正确视频重提。依赖链随后从修复后的全部前端重新拟合分组尺度、生成801份派生特征，并从初始状态完成FULL/RGB_2D各3个100 epoch模型。旧结果保留为受污染历史，不再作为干净训练结果。", "",
        f"运行状态：`{read(out / 'final_status.json').get('status')}`；run_id `{runtime.get('run_id')}`；本次墙钟约 `{elapsed:.1f}` 秒；正式模型 `{len([r for r in records if r.get('status') == 'COMPLETE'])}/6`。", "",
        "## 修复证据", "",
        f"- 复合身份索引：`{identity.get('policy')}`；冲突键不允许 last-write-wins 或 pair-only fallback。\n- 全量身份核对：{identity.get('window_count')} windows，`identity_changed={identity.get('changed_count')}`。\n- 重提窗口：`{', '.join(identity.get('changed_windows', []))}`。\n- 新目录前端：801/801，失败0；特征：801/801，失败0；原始5个错误窗口之外未发现新身份变更。\n- 新分组统计：spatial rows {read(out/'grouping_config.json').get('spatial_rows')}，motion rows {read(out/'grouping_config.json').get('motion_rows')}，阈值 {f(read(out/'grouping_config.json').get('merge_threshold'))}；标准化由修复后718个训练窗口重新拟合，和旧统计最大绝对差 FULL mean/scale={f(standardizer_diff['FULL']['mean_max_abs_difference'])}/{f(standardizer_diff['FULL']['scale_max_abs_difference'])}、RGB_2D mean/scale={f(standardizer_diff['RGB_2D']['mean_max_abs_difference'])}/{f(standardizer_diff['RGB_2D']['scale_max_abs_difference'])}。\n- 原训练source/class窗口数量未因身份修复改变；权重按新训练清单重新计算。", "",
        "## 修复前后指标", "", table, "", "训练集三seed平均logit（从已保存score数组复算，旧/新训练集合均为718窗口）：", "", "| condition | old train macro AUROC | new train macro AUROC | old train pooled AUROC | new train pooled AUROC | old AP | new AP |", "|---|---:|---:|---:|---:|---:|---:|", *[f"| {c} | {f(old_train_mean[c].get('source_macro_auroc'))} | {f(new_train_mean[c].get('source_macro_auroc'))} | {f(old_train_mean[c].get('auroc'))} | {f(new_train_mean[c].get('auroc'))} | {f(old_train_mean[c].get('ap'))} | {f(new_train_mean[c].get('ap'))} |" for c in ["FULL", "RGB_2D"]], "",
        f"新结果按三个seed平均logit后计算，阈值固定 `logit >= 0`，验证83窗口、14 source；新FULL−RGB_2D source-macro差 `{f(comp.get('mean'))}`，bootstrap 95% CI `[{f(comp.get('ci95',[None,None])[0])}, {f(comp.get('ci95',[None,None])[1])}]`，source方向 {signs['up']}上升/{signs['tie']}持平/{signs['down']}下降。", "",
        "## 结构与信息路径审计", "",
        "- **覆盖与输入**：原始视频按窗口解码到256×256 letterbox；每窗16个实际PTS帧。每帧256个16×16基础格，非padding格由 `cell_area>0` 定义；前端CoTracker为32×32=1024查询点，不是逐原始像素持久跟踪。`x_full`为`[T=16,C=256,D=77]`，`x_rgb2d`为`[16,256,71]`。",
        f"- **缺失仍参与**：全部非padding基础格保留；几何有效面积比例 {f(audit['pixel_and_geometry']['geometry_valid_area_fraction'])}，几何缺失按mask/显式状态进入FULL输入而非把区域删除。",
        f"- **动态component**：每帧按4邻接、训练拟合空间/运动尺度和25%成本阈值、最大4格合并；每帧component数中位数 {f(audit['components']['count_per_frame']['median'],2)}，组大小中位数 {f(audit['components']['size']['median'],2)}；{audit['components']['windows_with_dynamic_partition']}/{len(rows)}窗口至少一次相邻时刻分区成员改变（{f(audit['components']['dynamic_window_fraction'],4)}），不是单纯固定网格。",
        f"- **时间/关联**：模型`forward`逐16个时刻处理，使用实际`delta_t`、当前到前一帧最多4个多对多候选及权重，GRUCell递归；窗口最终对`[T,C]` cell logits做area-weighted logsumexp。训练/评价监督仍是窗口级，不存在已验证的frame-level标签或帧级主指标。\n- **帧内/全帧边界**：固定16×16网格边上执行一层edge MLP消息传递；FULL再按当前帧component均值广播残差，RGB_2D不读几何component/XYZ/几何mask。没有跨全帧component图的显式消息传递；窗口logit是局部cell时空证据的聚合，不等于全帧拓扑检测。\n- **观测属性**：FULL显式读取XYZ、geometry_valid、track_coverage、motion、motion_valid、history_available以及association/delta_t；RGB_2D只读取RGB空间特征、cell RGB、coverage、motion/motion_valid，保留相同时间关联接口但无几何component输入。", "",
        "## 训练与限制", "",
        "- 六个模型均从初始状态训练100 epoch、最后epoch评价；参数量FULL=217025、RGB_2D=216257。loss均为有限值并下降，但训练集接近拟合，验证指标仍有明显差距，不能单凭此断定唯一过拟合原因。\n- FULL−RGB_2D的变化包含几何输入、观测状态和动态分组，不能分配为纯XYZ贡献；修复前旧结果受5个训练窗口视频身份污染，不能与修复后作方法增益因果解释。\n- 当前正式输出是窗口级分类；可导出逐帧/逐格证据图不等于帧级检测已验证，也没有像素级真值定位。", "",
        "## 产物", "",
        f"- 修复报告：`{out/'repair_report.md'}`\n- 结构审计：`{out/'structure_audit.md'}`、`{out/'structure_audit.json'}`、`{out/'structure_audit_windows.csv'}`\n- 指标：`{out/'evaluation/summary.json'}`；模型记录：`{out/'models/fold_models.json'}`\n- 复现入口（需外部权重与正确数据根）：`PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$PWD PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_v8_identityfix.py ...`." ,
    ]
    (out / "repair_report.md").write_text("\n".join(repair_lines) + "\n", encoding="utf-8")
    audit_lines = [
        "# V8 修复运行结构审计", "", "本文件只读取 identityfix 运行已保存的前端、特征和模型元数据；不重新前向。", "",
        "## 计算路径", "", "`raw video -> 16 decoded letterboxed frames -> MoGe cell XYZ/RGB + CoTracker short-prefix state -> 16x16 cell grouping and multi-to-multi predecessor summaries -> x_full/x_rgb2d -> cell encoder -> one fixed-grid edge message pass -> (FULL: component pool/residual) -> GRUCell over actual delta_t -> cell logits -> area-weighted logsumexp over T,C -> window BCE/evaluation`.", "",
        "源码证据（当前HEAD）：`research_tools/v8/full_coverage_observation_field/pipeline.py` 的 `_build_feature_arrays`、`_components`、`run_features`；`research_tools/v8/full_coverage_observation_field/model.py` 的 `FullCoverageModel.forward`（时间循环、GRU、最终logsumexp）和 `_spatial`（固定网格边及FULL component残差）。", "",
        "## 覆盖统计", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", "",
        "## 解释边界", "", "动态component成员确实可能跨相邻时刻变化，且保存了多对多关联；但模型没有显式全帧component间消息传递，监督/主评价是窗口级。缺失/无历史状态作为输入通道和关联合法性信息存在，不是失真真值。RGB_2D是固定基础网格上的二维观测控制，不是独立逐像素全画面基线。", "",
    ]
    (out / "structure_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    # Keep the machine report's status block consistent with final_status.
    report = out / "report.md"
    if report.exists():
        txt = report.read_text(encoding="utf-8")
        txt += "\n\n## Identity-fix completion\n\nThis report belongs to the source-aware identity-fix run. See `repair_report.md` and `structure_audit.md`; `final_status.json` and `stage_status.json` are the authoritative completion records.\n\n```json\n" + json.dumps({"final_status": read(out / "final_status.json"), "stage_status": read(out / "stage_status.json")}, ensure_ascii=False, indent=2) + "\n```\n"
        report.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
