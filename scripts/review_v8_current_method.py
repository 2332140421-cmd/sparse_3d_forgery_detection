#!/usr/bin/env python3
"""Read-only audit of the frozen, identity-fixed V8 baseline.

The script reads manifests, saved frontend/features, and one formal checkpoint.
It never decodes frames, runs providers, trains, or writes into an experiment
directory other than the explicitly supplied review output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_tools.v8.full_coverage_observation_field.model import FullCoverageModel


SEEDS = (20260909, 20260910, 20260911)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--historical", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("min", "p25", "median", "p75", "max")}
    a = np.asarray(values, dtype=np.float64)
    return {"min": float(a.min()), "p25": float(np.quantile(a, .25)), "median": float(np.median(a)), "p75": float(np.quantile(a, .75)), "max": float(a.max())}


def video_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, dict[str, Any]] = {}
    occurrences: Counter[tuple[str, int]] = Counter()
    within_window_repeats = 0
    for row in rows:
        key = str(row.get("video_sha256") or row["video_path"])
        item = by_video.setdefault(key, {"path": row["video_path"], "frames": set(), "windows": 0})
        item["windows"] += 1
        selected = [int(x) for x in row["v8_frame_indices"]]
        within_window_repeats += len(selected) - len(set(selected))
        item["frames"].update(selected)
        for frame in selected:
            occurrences[(key, frame)] += 1

    total_video_frames = 0
    metadata_ok = 0
    ranges: list[dict[str, Any]] = []
    for key, item in by_video.items():
        cap = cv2.VideoCapture(item["path"])
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if cap.isOpened() else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        cap.release()
        good = frame_count > 0
        if good:
            metadata_ok += 1
            total_video_frames += frame_count
        selected = sorted(item["frames"])
        ranges.append({
            "video_identity": key,
            "window_count": item["windows"],
            "selected_unique_frames": len(selected),
            "selected_min_frame": selected[0] if selected else None,
            "selected_max_frame": selected[-1] if selected else None,
            "video_frame_count": frame_count if good else None,
            "fps_metadata": fps if fps > 0 else None,
        })
    selected_occurrences = sum(len(r["v8_frame_indices"]) for r in rows)
    unique_selected = len(occurrences)
    return {
        "unique_videos": len(by_video),
        "selected_frame_occurrences": selected_occurrences,
        "unique_video_frame_pairs": unique_selected,
        "repeated_occurrences_due_to_overlap": selected_occurrences - unique_selected,
        "video_frame_pairs_used_by_multiple_windows": sum(v > 1 for v in occurrences.values()),
        "maximum_window_reuse_of_one_video_frame": max(occurrences.values(), default=0),
        "within_window_repeated_frame_occurrences": within_window_repeats,
        "video_metadata_success": metadata_ok,
        "video_total_frames_sum": total_video_frames if metadata_ok == len(by_video) else None,
        "unique_selected_over_total_frames": unique_selected / total_video_frames if metadata_ok == len(by_video) and total_video_frames else None,
        "per_video_ranges": ranges,
        "interpretation": "Metadata-only frame counts; no video frame was decoded by this review.",
    }


def partitions(ids: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i in np.flatnonzero(mask):
        groups[int(ids[i])].append(int(i))
    return list(groups.values())


def scan_spatial(exp: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    content_canvas: list[float] = []
    all_counts: list[float] = []
    all_sizes: list[float] = []
    content_counts: list[float] = []
    content_sizes: list[float] = []
    padding_cells = content_cells = partial_cells = 0
    geometry_invalid_content_cells = 0
    total_content_area = final_content_area = 0.0
    partition_changes = 0
    partition_total = 0
    for row in rows:
        name = safe_name(row["window_id"])
        with np.load(exp / "features" / f"{name}.npz", allow_pickle=False) as z:
            area = z["area"].astype(np.float64)
            comp = z["component_ids"].astype(np.int64)
            xfull = z["x_full"]
        content_canvas.append(float(area.sum() / (area.shape[0] * 256 * 256)))
        total_content_area += float(area.sum())
        final_content_area += float(area[area > 0].sum())
        # FULL channel 70 (64 RGB + 3 cell RGB + 3 XYZ) is geometry validity.
        geometry_valid = xfull[..., 70] > .5
        geometry_invalid_content_cells += int(((area > 0) & ~geometry_valid).sum())
        for t in range(area.shape[0]):
            content = area[t] > 0
            full = np.ones(256, dtype=bool)
            ag = partitions(comp[t], full)
            cg = partitions(comp[t], content)
            all_counts.append(float(len(ag)))
            content_counts.append(float(len(cg)))
            all_sizes.extend(float(len(g)) for g in ag)
            content_sizes.extend(float(len(g)) for g in cg)
            padding_cells += int((area[t] == 0).sum())
            content_cells += int(content.sum())
            partial_cells += int(((area[t] > 0) & (area[t] < 256)).sum())
            if t:
                partition_total += 1
                prev = {tuple(g) for g in partitions(comp[t - 1], np.ones(256, bool))}
                now = {tuple(g) for g in ag}
                partition_changes += int(prev != now)
    return {
        "content_area_over_canvas_area": stats(content_canvas),
        "assigned_real_content_area_over_all_real_content": 1.0 if total_content_area else None,
        "final_aggregation_real_content_area_over_all_real_content": final_content_area / total_content_area if total_content_area else None,
        "content_cell_frames": content_cells,
        "padding_cell_frames": padding_cells,
        "partially_padded_border_cell_frames": partial_cells,
        "geometry_invalid_content_cell_frames": geometry_invalid_content_cells,
        "all_canvas_components_per_frame": stats(all_counts),
        "all_canvas_component_size_cells": stats(all_sizes),
        "content_intersecting_components_per_frame": stats(content_counts),
        "content_intersection_component_size_cells": stats(content_sizes),
        "partition_transitions_changed_all_canvas": partition_changes,
        "partition_transitions_total": partition_total,
    }


def sensitivity(exp: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = next(r for r in rows if r["split"] == "validation")
    path = exp / "features" / f"{safe_name(row['window_id'])}.npz"
    ckpt_path = exp / "models" / "FULL__seed20260909.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with np.load(path, allow_pickle=False) as z:
        x = (z["x_full"].astype(np.float32) - ckpt["mean"]) / ckpt["scale"]
        arrays = {k: z[k] for k in ("component_ids", "predecessor", "association_weight", "history_available", "delta_t", "area")}
    model = FullCoverageModel("FULL", 77)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    comp = arrays["component_ids"].copy()
    relabel = comp.copy()
    for t in range(comp.shape[0]):
        labels = np.unique(comp[t])
        mapping = dict(zip(labels.tolist(), labels[::-1].tolist()))
        relabel[t] = np.asarray([mapping[int(v)] for v in comp[t]])
    singleton = np.broadcast_to(np.arange(256, dtype=np.int64), comp.shape).copy()

    def forward(c: np.ndarray) -> float:
        with torch.inference_mode():
            return float(model(
                torch.from_numpy(x)[None], torch.from_numpy(c)[None],
                torch.from_numpy(arrays["predecessor"])[None],
                torch.from_numpy(arrays["association_weight"])[None],
                torch.from_numpy(arrays["history_available"])[None],
                torch.from_numpy(arrays["delta_t"])[None],
                torch.from_numpy(arrays["area"])[None],
            ).item())
    base, renamed, changed = forward(comp), forward(relabel), forward(singleton)
    return {
        "window_id": row["window_id"], "checkpoint": str(ckpt_path),
        "baseline_logit": base, "label_renamed_same_partition_logit": renamed,
        "singleton_partition_logit": changed,
        "same_partition_abs_difference": abs(base - renamed),
        "changed_partition_abs_difference": abs(base - changed),
        "scope": "One existing window, eval/no-grad, in memory only; proves path sensitivity, not generalization benefit.",
    }


def build_manifest(repo: Path, exp: Path, historical: Path, rows: list[dict[str, Any]], temporal: dict[str, Any], spatial: dict[str, Any]) -> dict[str, Any]:
    protocol = read_json(exp / "protocol.json")
    summary = read_json(exp / "evaluation" / "summary.json")
    fold = read_json(exp / "models" / "fold_models.json")
    models = []
    for rec in fold["records"]:
        cp = Path(rec["checkpoint"])
        standardizer = exp / "models" / f"standardizer_{rec['condition']}.json"
        models.append({
            "condition": rec["condition"], "seed": rec["seed"], "epoch": rec["epoch"],
            "checkpoint": str(cp), "checkpoint_sha256": sha256(cp),
            "standardizer": str(standardizer), "standardizer_sha256": sha256(standardizer),
            "parameter_count": rec["parameter_count"], "status": rec["status"],
        })
    def metric(condition: str) -> dict[str, Any]:
        return next(x for x in summary["conditions"] if x["condition"] == condition and str(x["seed"]) == "MEAN_LOGIT" and x["split"] == "validation")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return {
        "manifest_version": "v8-current-method-review-v1",
        "audit_start_head": head,
        "formal_experiment_code_head": protocol["git_head"],
        "formal_experiment": str(exp), "status": read_json(exp / "final_status.json"),
        "protocol": {"path": str(exp / "protocol.json"), "sha256": sha256(exp / "protocol.json"), "id": protocol["protocol_id"], "run_id": protocol["run_id"]},
        "data_manifest": {"path": str(exp / "data_manifest.json"), "sha256": sha256(exp / "data_manifest.json"), "windows": len(rows), "train": sum(r["split"] == "train" for r in rows), "validation": sum(r["split"] == "validation" for r in rows)},
        "feature_schema": {"path": str(exp / "feature_schema.json"), "sha256": sha256(exp / "feature_schema.json")},
        "grouping_config": {"path": str(exp / "grouping_config.json"), "sha256": sha256(exp / "grouping_config.json")},
        "models": models,
        "ensemble": "three saved seed logits averaged per window before metrics; final epoch 100; threshold logit >= 0",
        "formal_results": {"FULL": metric("FULL"), "RGB_2D": metric("RGB_2D"), "comparison": summary["comparison_FULL_minus_RGB_2D"]},
        "frontend": protocol["frontend"], "rgb_feature": protocol["rgb_feature"],
        "queue_inheritance": {"identity_source": protocol["input"]["identity_source"], "numeric_v7_features_read": False, "independent_full_video_queue": False, "temporal_coverage": {k: temporal[k] for k in temporal if k != "per_video_ranges"}},
        "spatial_coverage": spatial,
        "historical_pre_fix": {"path": str(historical), "status": "PRE_FIX_TRAIN_INPUT_IDENTITY_MISMATCH", "formal_use": False},
    }


def write_alignment(path: Path) -> None:
    rows = [
        ("全视频连续帧", "801个继承窗口，每窗16个采样时刻", "未实现", "pipeline.py:156-204; frontend.py:193-203", "未建立独立全视频帧清单与跨块hidden连续性"),
        ("真实内容区域全归属", "256×256画布中每个真实内容像素映射到一个16×16 cell，边界格保留面积", "已实现", "frontend.py:37-52,227-243", "不是逐像素token或逐像素跟踪"),
        ("缺失区域不删除", "有限占位和状态通道；最终按真实内容面积聚合", "已实现但有边界", "pipeline.py:457-475; model.py:128-131", "padding也进入标准化/消息/池化，直至最终面积权重归零"),
        ("动态components", "格级XYZ/运动摘要驱动逐帧union-find，最多4格", "已实现", "pipeline.py:297-356,404-476", "不是逐像素分组；没有全帧component图"),
        ("component内部结构学习", "隐藏cell特征mean pool后残差回传", "部分实现", "model.py:31-75", "稠密XYZ已在可学习编码前做格内mean"),
        ("整帧结构分布", "固定网格一层消息+所有时空cell logits聚合", "部分实现", "model.py:56-75,128-131", "无显式全帧component关系学习"),
        ("连续演化", "窗口内16时刻GRU，最多4个前驱，真实delta_t", "部分实现", "model.py:77-126", "hidden每窗重置；不跨窗口/全视频"),
        ("正式帧级输出", "内部cell_logits可按帧查看", "未验证", "model.py:126-139", "无定义的正式frame logit、帧级监督或帧级评价"),
        ("V7数值资产隔离", "V8从原视频生成前端和特征", "已实现", "frontend.py; pipeline.py:359-479", "窗口身份仍继承V7实验队列"),
        ("身份修复正式基准", "FULL/RGB_2D×3 seed，100 epoch", "已实现", "identityfix final_status/fold_models", "修复前结果仅历史记录"),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["用户目标", "实际实现", "状态", "证据", "缺口"]); w.writerows(rows)


def main() -> None:
    a = parse_args(); repo, exp, historical, out = map(Path, (a.repo, a.experiment, a.historical, a.output))
    out.mkdir(parents=True, exist_ok=True)
    rows = read_json(exp / "data_manifest.json")["rows"]
    temporal = video_metadata(rows)
    spatial = scan_spatial(exp, rows)
    diagnostic = sensitivity(exp, rows)
    manifest = build_manifest(repo, exp, historical, rows, temporal, spatial)
    write_json(out / "temporal_coverage.json", temporal)
    write_json(out / "spatial_component_audit.json", spatial)
    write_json(out / "component_forward_diagnostic.json", diagnostic)
    write_json(out / "current_v8_manifest.json", manifest)
    write_alignment(out / "method_alignment.csv")
    print(json.dumps({"temporal": {k: v for k, v in temporal.items() if k != "per_video_ranges"}, "spatial": spatial, "diagnostic": diagnostic}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
