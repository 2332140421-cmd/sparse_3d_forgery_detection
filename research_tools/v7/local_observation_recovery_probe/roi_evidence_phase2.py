"""Second-stage ROI evidence analysis for the existing 04LAX case.

The input is a user-exported annotation JSON. Only entries explicitly marked
CONFIRMED are used. This module reads saved O/R arrays and saved H structure;
it does not rerun any frontend or model.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from .roi_evidence import (
    DATA_ROOT, DEFAULT_OUTPUT, OLD_CASE, O_PARTICLE, OBS_SUPPORT, FAKE_VIDEO,
    read_json, write_json, write_csv, sha256, finite_visible, canonical_pairs,
    draw_overlay, save_png,
)

BOUNDARY_DETAILS = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1/review/details/0002_MANIP_25__fake.json"
R_GEOMETRY = OLD_CASE / "conditions/R_geometry.npz"
R_STRUCTURE = OLD_CASE / "conditions/R_structure.json"
CONFIRMED = "CONFIRMED"

def validate_annotations(path: Path, case_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    data = read_json(path)
    expected_video = case_manifest["video"]
    errors: list[str] = []
    if data.get("source") != case_manifest["source_id"]: errors.append("source_mismatch")
    if data.get("role") != case_manifest["role"]: errors.append("role_mismatch")
    if data.get("window_id") != case_manifest["window_id"]: errors.append("window_mismatch")
    video = data.get("video", {})
    for key in ("path", "sha256", "bytes"):
        if video.get(key) != expected_video.get(key): errors.append("video_%s_mismatch" % key)
    if not FAKE_VIDEO.is_file() or sha256(FAKE_VIDEO) != expected_video.get("sha256"): errors.append("current_video_hash_mismatch")
    if not FAKE_VIDEO.is_file() or FAKE_VIDEO.stat().st_size != int(expected_video.get("bytes", -1)): errors.append("current_video_size_mismatch")
    image_hw = tuple(int(x) for x in case_manifest["image_size_hw"])
    expected = {}
    with np.load(O_PARTICLE, allow_pickle=False) as o, np.load(R_GEOMETRY, allow_pickle=False) as r:
        for f, t in zip(o["frame_indices"].tolist(), o["timestamps_s"].tolist()): expected[int(f)] = float(t)
        for f, t in zip(r["frame_indices"].tolist(), r["timestamps_s"].tolist()): expected.setdefault(int(f), float(t))
    entries = data.get("entries", [])
    seen: set[int] = set(); confirmed: list[dict[str, Any]] = []
    for raw in entries:
        e = dict(raw); frame = int(e.get("source_frame_index", -1)); status = str(e.get("human_confirmation_status", "PENDING")).upper()
        if frame in seen: errors.append("duplicate_frame_%s" % frame)
        seen.add(frame)
        if frame not in expected: errors.append("frame_not_in_case_%s" % frame)
        if tuple(e.get("image_size_hw", [])) != image_hw: errors.append("image_size_mismatch_%s" % frame)
        if str(e.get("coordinate_convention", "")) != "xyxy_source_pixel": errors.append("coordinate_convention_mismatch_%s" % frame)
        if frame in expected and abs(float(e.get("pts_s", float("nan"))) - expected[frame]) > 1e-9: errors.append("pts_mismatch_%s" % frame)
        coords = [e.get(k) for k in ("x_min", "y_min", "x_max", "y_max")]
        if status == CONFIRMED:
            try:
                x0, y0, x1, y1 = [float(x) for x in coords]
                if not all(math.isfinite(x) for x in (x0, y0, x1, y1)) or not (0 <= x0 < x1 <= image_hw[1] and 0 <= y0 < y1 <= image_hw[0]):
                    errors.append("roi_out_of_bounds_%s" % frame)
            except (TypeError, ValueError):
                errors.append("roi_missing_%s" % frame)
            else:
                confirmed.append(e)
    if errors: raise ValueError("ROI_VALIDATION_FAILED:" + ",".join(errors))
    info = {"input_path": str(path), "entry_count": len(entries), "confirmed_count": len(confirmed),
            "excluded_count": len(entries) - len(confirmed), "excluded_status_counts": {
                s: sum(str(x.get("human_confirmation_status", "PENDING")).upper() == s for x in entries)
                for s in ("UNCERTAIN", "SKIP", "PENDING")}, "video_hash_verified": True, "image_size_hw": list(image_hw)}
    return data, confirmed, info

def load_condition_data() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with np.load(O_PARTICLE, allow_pickle=False) as z:
        out["O"] = {k: np.asarray(z[k]) for k in ("frame_indices","timestamps_s","uv","visibility","geometry_validity","xyz")}
    with np.load(R_GEOMETRY, allow_pickle=False) as z:
        out["R"] = {k: np.asarray(z[k]) for k in ("frame_indices","timestamps_s","uv","visibility","geometry_validity","xyz")}
    return out

def structure_sources() -> dict[str, dict[str, Any]]:
    boundary = read_json(BOUNDARY_DETAILS)
    r = read_json(R_STRUCTURE)
    out: dict[str, dict[str, Any]] = {}
    for condition, source in (("O", boundary.get("support", {}).get("H", {})), ("R", r.get("support", {}))):
        groups = list(source.get("groups", []))
        if condition == "R":
            groups = list(r.get("grouping", {}).get("groups", []))
        triplets = [x for x in source.get("triplets", []) if x.get("status", "VALID") == "VALID"]
        out[condition] = {"groups": groups, "triplets": triplets}
    return out

def frame_structure(condition: str, frame: int, source: Mapping[str, Any]) -> dict[str, Any]:
    groups = list(source["groups"]); triplets = list(source["triplets"])
    if condition == "R":
        relevant = [x for x in triplets if int(frame) in [int(y) for y in x.get("frame_indices", [])]]
        status = "VALID_R_H_TARGET_SUPPORT" if relevant else "NO_R_TRIPLET_AT_FRAME"
    else:
        relevant = triplets
        status = "SAVED_O_H_IDENTITY_SUPPORT_NOT_FRAME_SPECIFIC"
    pairs = canonical_pairs(pair for x in relevant for pair in x.get("pair_ids", x.get("pair_member_slots", x.get("pair_indices", []))))
    common = sorted({int(q) for pair in pairs for q in pair})
    historical = sorted({int(q) for g in groups for q in g.get("member_slots", g.get("track_ids", []))})
    retained = [g for g in groups if bool(g.get("retained", True))]
    owner_ids: set[int] = set()
    for x in relevant:
        for owner in x.get("owner_group_ids", []):
            # The saved boundary artifact uses integer group ids, while some
            # older structure artifacts used small mapping records.
            if isinstance(owner, Mapping):
                owner_ids.add(int(owner.get("local_group_id", owner.get("id", -1))))
            else:
                owner_ids.add(int(owner))
    owner_ids = sorted(owner_ids)
    if not owner_ids: owner_ids = sorted(int(g.get("local_group_id", -1)) for g in retained)
    return {"status": status, "triplet_count": len(relevant), "pairs": pairs, "common_members": common,
            "historical_members": historical, "groups": retained, "owner_group_ids": owner_ids}

def inside_rect(uv: np.ndarray, rect: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = [float(x) for x in rect]
    return np.isfinite(uv).all(axis=-1) & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)

def count_frame(condition: str, frame: int, entry: Mapping[str, Any], arrays: Mapping[str, Any], source: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = {int(x): i for i, x in enumerate(arrays["frame_indices"].tolist())}
    if frame not in indices:
        return {"condition": condition, "source_frame_index": frame, "roi_status": CONFIRMED, "data_status": "FRAME_NOT_IN_CONDITION_CACHE"}, []
    i = indices[frame]; uv = arrays["uv"][i]; vis = finite_visible(uv, arrays["visibility"][i])
    xyz_finite = np.isfinite(arrays["xyz"][i]).all(axis=-1)
    geo = vis & arrays["geometry_validity"][i].astype(bool) & xyz_finite
    rect = [entry["x_min"], entry["y_min"], entry["x_max"], entry["y_max"]]; in_roi = inside_rect(uv, rect)
    st = frame_structure(condition, frame, source)
    hist = np.zeros(len(uv), dtype=bool); hist[np.asarray(st["historical_members"], dtype=int)] = True
    common = np.zeros(len(uv), dtype=bool); common[np.asarray(st["common_members"], dtype=int)] = True
    pair_valid = [(a, b) for a, b in st["pairs"] if geo[a] and geo[b]]
    both = sum(bool(in_roi[a] and in_roi[b]) for a, b in pair_valid)
    one = sum(bool(in_roi[a] != in_roi[b]) for a, b in pair_valid)
    groups_intersect = 0
    for group in st["groups"]:
        members = [int(x) for x in group.get("member_slots", group.get("track_ids", [])) if 0 <= int(x) < len(uv)]
        if any(bool(in_roi[q] and vis[q]) for q in members): groups_intersect += 1
    records = []
    unknown = ~(np.isfinite(uv).all(axis=-1) & arrays["visibility"][i].astype(bool))
    for q in range(len(uv)):
        records.append({"condition": condition, "source_frame_index": frame, "pts_s": float(arrays["timestamps_s"][i]), "query_id": q,
                        "x": None if not np.isfinite(uv[q]).all() else float(uv[q,0]), "y": None if not np.isfinite(uv[q]).all() else float(uv[q,1]),
                        "visible": bool(vis[q]), "geometry_valid": bool(geo[q]), "roi_inside": "UNKNOWN" if unknown[q] else bool(in_roi[q]),
                        "historical_member": bool(hist[q]), "current_common_member": bool(common[q])})
    row = {"condition": condition, "source_frame_index": frame, "pts_s": float(arrays["timestamps_s"][i]), "roi_status": CONFIRMED,
           "data_status": "AVAILABLE", "structure_status": st["status"], "model_target": condition == "R" and frame in [503,506,509,512,515],
           "query_count": len(uv), "visible_uv_count": int(vis.sum()), "geometry_valid_count": int(geo.sum()),
           "roi_visible_uv_count": int((in_roi & vis).sum()), "roi_geometry_valid_count": int((in_roi & geo).sum()),
           "roi_historical_member_visible_count": int((in_roi & vis & hist).sum()), "roi_current_common_member_visible_count": int((in_roi & vis & common).sum()),
           "historical_member_total": int(hist.sum()), "current_common_member_total": int(common.sum()),
           "intersecting_local_group_count": int(groups_intersect), "valid_pair_count": len(pair_valid),
           "both_endpoints_pair_count": int(both), "one_endpoint_pair_count": int(one), "unknown_uv_count": int(unknown.sum()),
           "triplet_count": st["triplet_count"], "relation_identity_count": len(st["pairs"])}
    return row, records

def phase2(output: Path, annotation_path: Path) -> dict[str, Any]:
    case = read_json(output / "case_manifest.json")
    data, confirmed, validation = validate_annotations(annotation_path, case)
    write_json(output / "roi_annotations_input.json", data)
    write_json(output / "roi_annotations.json", {"status": "VALIDATED_USER_CONFIRMATION_PARTIAL", "validation": validation, "input": data})
    arrays = load_condition_data(); sources = structure_sources()
    rows: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []
    for entry in confirmed:
        frame = int(entry["source_frame_index"])
        for condition in ("O", "R"):
            row, rec = count_frame(condition, frame, entry, arrays[condition], sources[condition])
            row.update({"source": "04LAX", "role": "fake", "window_id": "0002_MANIP_25::fake"})
            rows.append(row); records.extend(rec)
    fields = sorted({k for row in rows for k in row})
    write_csv(output / "roi_support_counts.csv", rows, fields)
    write_csv(output / "roi_point_membership.csv", records, sorted({k for row in records for k in row}))
    feature_manifest_path = OBS_SUPPORT / "inputs/feature_manifest.json"
    feature_manifest = read_json(feature_manifest_path) if feature_manifest_path.is_file() else []
    source_rows = [x for x in feature_manifest if str(x.get("source_id", "")) == "04LAX" or "04LAX" in str(x.get("window_id", ""))]
    model_status = {"status": "NOT_AVAILABLE", "reason": "NO_CURRENT_FROZEN_FEATURE_ROW_FOR_04LAX", "frozen_model_root": str(OBS_SUPPORT),
                    "feature_manifest_path": str(feature_manifest_path), "feature_manifest_row_count": len(feature_manifest),
                    "feature_rows_for_source": len(source_rows),
                    "scores_written": False, "unit_scores": None, "window_scores": None,
                    "note": "Confirmed ROI was used only for observation-to-structure counts; no model input/output was fabricated."}
    write_json(output / "model_readout_status.json", model_status)
    write_json(output / "window_scores.json", model_status)
    make_roi_overlays(output, rows, confirmed, arrays, sources)
    report = build_report(case, validation, rows, records, model_status)
    (output / "report.md").write_text(report, encoding="utf-8")
    final = {"status": "COMPLETE_NO_MODEL_READOUT", "roi_status": "VALIDATED_USER_CONFIRMATION_PARTIAL",
             "confirmed_frame_count": len(confirmed), "excluded_frame_count": validation["excluded_count"],
             "model_readout_status": model_status["status"], "updated_unix": time.time()}
    write_json(output / "final_status.json", final)
    return final

def make_roi_overlays(output: Path, rows: Sequence[Mapping[str, Any]], confirmed: Sequence[Mapping[str, Any]], arrays: Mapping[str, Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]) -> None:
    import cv2
    frame_images = output / "frames"
    for e in confirmed:
        frame = int(e["source_frame_index"]); rect = [int(e[k]) for k in ("x_min","y_min","x_max","y_max")]
        base = frame_images / ("frame_%d.png" % frame)
        if not base.is_file(): continue
        image = cv2.imread(str(base))
        cv2.rectangle(image, (rect[0],rect[1]), (rect[2],rect[3]), (0,0,255), 2)
        cv2.putText(image, "CONFIRMED ROI frame %d" % frame, (8,22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,255), 1, cv2.LINE_AA)
        cv2.imwrite(str(frame_images / ("frame_%d_confirmed_roi.png" % frame)), image)

def build_report(case: Mapping[str, Any], validation: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], model_status: Mapping[str, Any]) -> str:
    lines = ["# 04LAX ROI evidence review（第二阶段）", "", "状态：COMPLETE_NO_MODEL_READOUT", "",
             "本阶段只使用人工标记为 CONFIRMED 的逐帧矩形；UNCERTAIN、SKIP、PENDING 均排除。ROI 仍是人工粗诊断区域，不是空间真值或分割 mask。", "",
             "## 输入核验", "",
             "- 视频 path、sha256、字节数与当前文件及旧案例 manifest 一致；原图尺寸为 480×360，坐标为 xyxy_source_pixel。",
             "- 人工输入条目 %s 个，其中 CONFIRMED=%s，排除=%s（%s）。" % (validation["entry_count"], validation["confirmed_count"], validation["excluded_count"], validation["excluded_status_counts"]),
             "- frame487 与 frame488 的 PTS、R query 起点身份保持不变；frame488 为 SKIP，因此不参与任何 ROI 统计。",
             "", "## ROI 到结构统计口径", "",
             "- roi_visible_uv_count：有限 UV 且保存 visibility 通过并落在矩形内的 query 数。",
             "- roi_geometry_valid_count：上述点再通过 geometry_validity 且 XYZ 有限。",
             "- historical_member_visible_count：点属于保存的历史 H group 成员集合；current_common_member_visible_count：点属于当前相关有效 triplet 的 pair 端点集合。",
             "- intersecting_local_group_count：至少一个可见 query 落在 ROI 内的保存局部组；不是像素 component。",
             "- pair 只来自保存的 H triplet canonical pair，且两端 geometry-valid 才进入 valid_pair；both/one 只按端点是否在 ROI 计数，关系线穿过 ROI 不计入。",
             "- masked/invisible UV 的 ROI 位置记 UNKNOWN，不计入 ROI 内或 ROI 外。", "",
             "## 当前模型读出", "",
             "- 当前 observation_support_pilot_v1 的 feature_manifest 共 %s 行，04LAX 行数为 %s，因此模型读出状态为 NOT_AVAILABLE。没有生成 unit logit、window logit 或窗口均值贡献。" % (model_status.get("feature_manifest_row_count", "NA"), model_status.get("feature_rows_for_source", "NA")),
             "- 已确认 ROI 仅用于解释观测与结构支撑，未修改分组、pair、历史尺度、S/Q、模型或训练数据。",
             "- R 缓存覆盖 frame 503/506/509/512/515 五个目标帧，并保存了跨这些帧的滑动三时刻 H triplet；这是旧案例结构证据，不能直接当作当前五时刻模型输入，也不等于冻结模型已经消费了该案例。",
             "", "## 结果表", "",
             "| condition | frame | pts(s) | model target | ROI visible | ROI geometry | ROI common members | groups | valid pairs | both endpoints | one endpoint | unknown UV |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        pts = r.get("pts_s")
        pts_text = "NA" if pts is None else "%.9f" % float(pts)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r.get("condition"), r.get("source_frame_index"), pts_text, r.get("model_target"), r.get("roi_visible_uv_count","NA"), r.get("roi_geometry_valid_count","NA"), r.get("roi_current_common_member_visible_count","NA"), r.get("intersecting_local_group_count","NA"), r.get("valid_pair_count","NA"), r.get("both_endpoints_pair_count","NA"), r.get("one_endpoint_pair_count","NA"), r.get("unknown_uv_count","NA")))
    lines += ["", "## 结论边界", "",
              "- 用户所指区域在 CONFIRMED 帧上是否有可见点、是否进入保存的结构成员及有效 pair，见 roi_support_counts.csv；这些是观测计数，不是检测准确率。",
              "- 对 R 目标帧，结构减少可能发生在 visibility、geometry、共同成员或 pair 支撑层；该案例没有当前冻结模型 logit，不能把计数变化解释成模型响应。",
              "- frame487 的 O 观察可用于历史阶段对照；R 在该帧未查询。frame488 被 SKIP，不能用于 ROI 连续性结论。",
              "- 没有兼容的 04LAX real 空间 ROI 对照；不能把 fake 框映射到 real，也不能据此说明伪造区域定位能力。",
              "- 本案例不能证明跨失踪事件的旧轨迹物理对应，也不能证明模型学会或未学会真实失真。"]
    return "\n".join(lines) + "\n"

def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); p.add_argument("--annotations",type=Path,default=None); args=p.parse_args(argv)
    if args.annotations is None: raise SystemExit("--annotations is required for phase 2")
    print(json.dumps(phase2(args.output,args.annotations),ensure_ascii=False,indent=2),flush=True); return 0

if __name__ == "__main__": raise SystemExit(main())
