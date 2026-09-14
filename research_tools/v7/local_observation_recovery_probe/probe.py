"""Bounded O/R/T diagnostic for the 04LAX fake observation case.

The module intentionally consumes an existing 289-point ParticleSequence for
the O condition.  R is an independent query with the already used
BootsTAPIR implementation, while T is a best-effort official CoTracker3
online check.  No detector score is computed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CASE_ID = "04LAX_fake_0002_MANIP_25"
SOURCE_ID = "04LAX"
WINDOW_ID = "0002_MANIP_25::fake"
TARGET_TIMES = (16.249583, 16.282950)
TARGET_FRAMES = (487, 488)
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DEFAULT_OUTPUT = DATA_ROOT / "derived/v7_activityforensics_local_observation_recovery_probe_v1"
FAKE_VIDEO = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1/source/activityforensics/raw/video/02_wan/04LAX+13.90=22.70=charades@train_delete@04LAX@365@wan.mp4"
REAL_VIDEO = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1/source/charades/videos/04LAX.mp4"
PARTICLE = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1/particles/0002_MANIP_25__fake__density289.npz"
PARTICLE_META = PARTICLE.with_suffix(".json")
DETAIL = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1/review/details/0002_MANIP_25__fake.json"
TAPNET_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/tapnet-c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNET_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/causal_bootstapir_checkpoint.pt"
TRACKER_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
COTRACKER_SHA = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"


def nearest_frame(frame_indices: Sequence[int], timestamps_s: Sequence[float], target_s: float) -> dict[str, Any]:
    """Map a user time to the nearest saved source frame without rounding FPS."""
    if len(frame_indices) != len(timestamps_s) or len(frame_indices) == 0:
        raise ValueError("frame_indices and timestamps_s must be non-empty and aligned")
    i = min(range(len(frame_indices)), key=lambda j: abs(float(timestamps_s[j]) - float(target_s)))
    return {
        "array_index": int(i),
        "source_frame_index": int(frame_indices[i]),
        "timestamp_s": float(timestamps_s[i]),
        "delta_s": float(timestamps_s[i] - target_s),
    }


def _finite_uv(uv: np.ndarray) -> np.ndarray:
    return np.asarray(np.isfinite(uv).all(axis=-1), dtype=bool)


def layer_counts(
    uv: np.ndarray,
    visibility: np.ndarray,
    geometry_validity: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
    *,
    group_slots: Iterable[int] = (),
    pair_slots: Iterable[Sequence[int]] = (),
) -> dict[str, Any]:
    """Count the actual observation layers for one frame.

    Missing UV/XYZ is never converted to a coordinate.  ``relation_support``
    counts only supplied pairs whose two endpoints are geometry-valid.
    """
    uv = np.asarray(uv)
    visibility = np.asarray(visibility, dtype=bool)
    finite = _finite_uv(uv)
    visible = visibility & finite
    if geometry_validity is None:
        geometry = None
    else:
        geometry = np.asarray(geometry_validity, dtype=bool) & visible
        if xyz is not None:
            geometry &= np.isfinite(np.asarray(xyz)).all(axis=-1)
    group = np.zeros(len(visible), dtype=bool)
    slots = np.asarray(list(group_slots), dtype=int)
    if slots.size:
        group[slots[(slots >= 0) & (slots < len(group))]] = True
    group_valid = int(np.count_nonzero(visible & group))
    relation_support = None
    if geometry is not None:
        relation_support = sum(bool(geometry[a] and geometry[b]) for a, b in pair_slots)
    return {
        "query_count": int(len(visible)),
        "uv_available_count": int(np.count_nonzero(finite)),
        "visibility_count": int(np.count_nonzero(visible)),
        "geometry_valid_count": None if geometry is None else int(np.count_nonzero(geometry)),
        "local_group_count": group_valid,
        "local_group_member_count": group_valid,
        "common_relation_support": relation_support,
        "final_display_count": int(np.count_nonzero(visible)),
        "display_reason": "VISIBLE_FINITE_UV" if np.any(visible) else "NO_VISIBLE_FINITE_UV",
    }


def fixed_members(visibility: np.ndarray, array_index: int) -> np.ndarray:
    """Return IDs visible at the fixed O reference time."""
    v = np.asarray(visibility, dtype=bool)
    if v.ndim != 2 or not 0 <= array_index < v.shape[0]:
        raise ValueError("visibility must be [time, query] and array_index must be valid")
    return np.flatnonzero(v[array_index]).astype(int)


def condition_frame_status(condition: str, source_frame_index: int, *, query_start_frame: int | None = None) -> str:
    """Return an explicit pre-query state instead of freezing or inventing points."""
    if condition == "R" and query_start_frame is not None and int(source_frame_index) < int(query_start_frame):
        return "NOT_QUERIED"
    return "AVAILABLE"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False, default=_jsonable)
        handle.write("\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _support(detail: Mapping[str, Any]) -> tuple[set[int], list[tuple[int, int]], dict[str, Any]]:
    support = detail.get("support", {}).get("H", {})
    groups = [g for g in support.get("groups", []) if g.get("retained", True)]
    slots = {int(s) for g in groups for s in g.get("member_slots", [])}
    pairs: set[tuple[int, int]] = set()
    for triplet in support.get("triplets", []):
        for pair in triplet.get("pair_member_slots", triplet.get("pair_ids", [])):
            if len(pair) == 2:
                pairs.add(tuple(sorted((int(pair[0]), int(pair[1])))))
    return slots, sorted(pairs), support


def _load_o(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrays = np.load(PARTICLE, allow_pickle=False)
    detail = _load_json(DETAIL)
    uv = arrays["uv"]
    vis = arrays["visibility"]
    geo = arrays["geometry_validity"]
    xyz = arrays["xyz"]
    group_slots, pairs, support = _support(detail)
    ref_index = int(np.argmin(np.abs(arrays["timestamps_s"] - TARGET_TIMES[0])))
    fixed = fixed_members(vis, ref_index)
    rows: list[dict[str, Any]] = []
    for i in range(len(arrays["frame_indices"])):
        c = layer_counts(uv[i], vis[i], geo[i], xyz[i], group_slots=group_slots, pair_slots=pairs)
        fc = layer_counts(uv[i, fixed], vis[i, fixed], geo[i, fixed], xyz[i, fixed], group_slots=range(len(fixed)), pair_slots=[])
        rows.append({
            "condition": "O",
            "array_index": i,
            "source_frame_index": int(arrays["frame_indices"][i]),
            "timestamp_s": float(arrays["timestamps_s"][i]),
            "fixed_query_count": int(len(fixed)),
            "fixed_uv_available_count": fc["uv_available_count"],
            "fixed_visibility_count": fc["visibility_count"],
            "fixed_geometry_valid_count": fc["geometry_valid_count"],
            "fixed_membership": "initial_visibility_at_16.249583s",
            **c,
            "local_group_definition": "retained H members from saved support",
            "relation_definition": "union of saved H triplet pair members",
        })
    manifest = {
        "case_id": CASE_ID,
        "condition": "O",
        "source_id": SOURCE_ID,
        "role": "fake",
        "window_id": WINDOW_ID,
        "video": {"path": str(FAKE_VIDEO), "status": "AVAILABLE" if FAKE_VIDEO.exists() else "SOURCE_MISSING", "sha256": _sha256(FAKE_VIDEO) if FAKE_VIDEO.exists() else None, "bytes": FAKE_VIDEO.stat().st_size if FAKE_VIDEO.exists() else None},
        "source_frames": arrays["frame_indices"].tolist(),
        "timestamps_s": arrays["timestamps_s"].tolist(),
        "image_size_hw": arrays["frame_sizes_hw"][0].tolist(),
        "query_count": int(len(arrays["track_ids"])),
        "query_initialization": {"source_frame_index": int(arrays["frame_indices"][0]), "timestamp_s": float(arrays["timestamps_s"][0]), "rule": "window first frame"},
        "verification_targets": [nearest_frame(arrays["frame_indices"], arrays["timestamps_s"], t) for t in TARGET_TIMES],
        "fixed_reference_members": fixed.tolist(),
        "raw_uv_status": "UNAVAILABLE_CANONICAL_ARTIFACT_MASKS_INVISIBLE_UV_TO_NAN",
        "support": {
            "h_group_slots": sorted(group_slots),
            "h_pair_count": len(pairs),
            "h_triplet_count": len(support.get("triplets", [])),
            "selected_triplet": ({
                "triplet_id": int(support["triplets"][0].get("triplet_id", 0)),
                "common_member_slots": [int(x) for x in support["triplets"][0].get("common_member_slots", [])],
                "pair_member_slots": [[int(x) for x in pair] for pair in support["triplets"][0].get("pair_member_slots", [])],
            } if support.get("triplets") else None),
        },
        "roi": {"status": "PENDING_SOURCE_PIXEL_MAPPING", "rectangle_xyxy": None, "reason": "conversation screenshots are browser captures without a source-pixel calibration file"},
        "provenance": _load_json(PARTICLE_META).get("provenance", {}) if PARTICLE_META.exists() else {},
        "artifact": str(PARTICLE),
    }
    return manifest, rows


def _decode_exact(video: Path, frame_indices: Sequence[int]):
    from sparse3d_forgery.video_input import VideoSource, decode_video
    return decode_video(VideoSource(sample_id=CASE_ID, source_video_id=SOURCE_ID, source_locator=video), frame_indices)


def _save_pngs(
    output: Path,
    decoded: Any,
    rows: Sequence[Mapping[str, Any]],
    uv: np.ndarray,
    vis: np.ndarray,
    prefix: str,
    *,
    geometry: np.ndarray | None = None,
    pair_slots: Sequence[Sequence[int]] = (),
) -> list[str]:
    try:
        import cv2
    except ImportError:
        return []
    shot_dir = output / "review" / "screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, frame in enumerate(decoded.frames):
        image = np.asarray(frame.rgb)[:, :, ::-1].copy()
        finite = np.isfinite(uv[i]).all(axis=-1) & vis[i]
        if geometry is not None and pair_slots:
            valid_geo = np.asarray(geometry[i], dtype=bool) & finite
            for a, b in pair_slots:
                if not (0 <= int(a) < len(valid_geo) and 0 <= int(b) < len(valid_geo)):
                    continue
                if not (valid_geo[int(a)] and valid_geo[int(b)]):
                    continue
                pa, pb = uv[i, int(a)], uv[i, int(b)]
                cv2.line(image, (int(round(float(pa[0]))), int(round(float(pa[1])))), (int(round(float(pb[0]))), int(round(float(pb[1])))), (255, 180, 40), 1, lineType=cv2.LINE_AA)
        for x, y in uv[i][finite]:
            cv2.circle(image, (int(round(float(x))), int(round(float(y)))), 2, (40, 220, 40), -1, lineType=cv2.LINE_AA)
        name = f"{prefix}_frame_{int(frame.source_frame_index)}.png"
        cv2.imwrite(str(shot_dir / name), image)
        names.append(str(Path("review/screenshots") / name))
    return names


def _run_requery(output: Path, *, max_extra_s: float = 1.0) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"condition": "R", "status": "NOT_RUN"}
    try:
        with np.load(PARTICLE, allow_pickle=False) as z:
            source_start = int(z["frame_indices"][0])
            target = int(z["frame_indices"][int(np.argmin(np.abs(z["timestamps_s"] - TARGET_TIMES[1])))])
            fps_delta = float(np.median(np.diff(z["timestamps_s"])))
            end = target + int(math.ceil(max_extra_s / fps_delta))
        frames = list(range(target, end + 1))
        decoded = _decode_exact(FAKE_VIDEO, frames)
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_v7_explicit_geometry_frontend import OnlineBootsTapir
        tracker = OnlineBootsTapir(TAPNET_SOURCE, TAPNET_CHECKPOINT, process_size=256, grid_size=17)
        uv, visible = tracker.track(decoded)
        geo = np.zeros_like(visible, dtype=bool)
        xyz = np.full((*uv.shape[:2], 3), np.nan, dtype=np.float32)
        arrays_path = output / "conditions" / "R_requery.npz"
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, frame_indices=np.asarray(frames, np.int64), timestamps_s=np.asarray([f.timestamp_s for f in decoded.frames], np.float64), frame_sizes_hw=np.asarray([[f.rgb.shape[0], f.rgb.shape[1]] for f in decoded.frames], np.int64), track_ids=np.arange(uv.shape[1], dtype=np.int64), uv=uv.astype(np.float32), visibility=visible, geometry_validity=geo, xyz=xyz)
        rows = []
        for i, frame in enumerate(decoded.frames):
            c = layer_counts(uv[i], visible[i], geo[i], xyz[i])
            rows.append({"condition": "R", "array_index": i, "source_frame_index": int(frame.source_frame_index), "timestamp_s": float(frame.timestamp_s), **c, "geometry_status": "NOT_COMPUTED_2D_REQUERY_ONLY", "support_status": "NOT_COMPUTED", "query_start_frame": target})
        _write_csv(output / "frame_layer_counts_R.csv", rows)
        result.update({"status": "COMPLETE_2D_ONLY", "query_start_frame": target, "query_start_pts_s": float(decoded.frames[0].timestamp_s), "frame_indices": frames, "timestamps_s": [float(f.timestamp_s) for f in decoded.frames], "query_count": int(uv.shape[1]), "raw_uv_status": "UNAVAILABLE_TRACKER_WRAPPER_MASKS_INVISIBLE_UV_TO_NAN", "artifact": str(arrays_path), "elapsed_s": time.perf_counter() - started})
        result["screenshots"] = _save_pngs(output, decoded, rows, uv, visible, "R")
        del tracker
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as exc:  # diagnostic boundary: preserve exact failure
        result.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started})
    return result


def _run_cotracker(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"condition": "T", "status": "NOT_RUN", "official_repo": "https://github.com/facebookresearch/co-tracker", "source_commit": COTRACKER_SHA}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the official online candidate")
        # The official hub entry is intentionally the only candidate.  If its
        # package/weights are unavailable, the report records the block rather
        # than substituting another tracker.
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", trust_repo=True).to("cuda").eval()
        result.update({"status": "BLOCKED_AFTER_LOAD_INTERFACE", "note": "official online model loaded but this bounded probe did not stitch its chunked online output into a 3D sequence; no T scores claimed"})
        del model
    except Exception as exc:
        result.update({"status": "BLOCKED_OFFICIAL_RUNTIME_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_s"] = time.perf_counter() - started
    _write_json(output / "conditions" / "T_status.json", result)
    return result


def _html_page(output: Path, manifest: Mapping[str, Any], statuses: Mapping[str, Any]) -> None:
    # The page is deliberately dependency-free and reads only case_data.json.
    page = """<!doctype html><meta charset='utf-8'><title>V7 local observation recovery</title>
<style>body{font:15px sans-serif;background:#111827;color:#e5e7eb;margin:20px}h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.panel{background:#1f2937;padding:12px;border-radius:8px}.view{position:relative;background:#000}.view img{width:100%;display:block}.view canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.muted{color:#9ca3af}.ok{color:#86efac}.warn{color:#fbbf24}button{margin:3px;padding:5px}table{border-collapse:collapse}td,th{padding:4px 8px;border-bottom:1px solid #374151;text-align:left}code{color:#bfdbfe}</style>
<h1>V7 04LAX fake 局部观测恢复</h1><p class='muted'>这是单案例观测诊断，不是检测器评价。O/R/T 的 query ID 不跨条件连接；不可见点不生成有效 XYZ。</p>
<div id='facts'></div><div><button id='prev'>上一帧</button><button id='next'>下一帧</button><span id='frame'></span></div>
<div class='grid'><section class='panel'><h2>O 原有结果</h2><div id='oStatus'></div><div class='view'><img id='oImg'><canvas id='oCanvas'></canvas></div></section>
<section class='panel'><h2>R 重新查询</h2><div id='rStatus'></div><div class='view'><img id='rImg'><canvas id='rCanvas'></canvas></div></section>
<section class='panel'><h2>T CoTracker3 online</h2><div id='tStatus'></div><div class='view'><img id='tImg'><canvas id='tCanvas'></canvas></div></section></div>
<p>图例：<span class='ok'>绿色=当前可见且有UV</span>；黄/灰状态文字表示未计算或阻塞。两个核验时刻是源帧 487/488（PTS 16.249583/16.282950 s）。截图仅显示保存的有效观测，canonical NPZ 中不可见 UV 已为 NaN，原始预测 UV 不可恢复。</p>
<script>
let d=null,i=0;const $=x=>document.getElementById(x);function esc(x){return String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function draw(panel){const p=d.conditions[panel],img=$(panel.toLowerCase()+'Img'),can=$(panel.toLowerCase()+'Canvas'); if(!p||!p.rows){img.removeAttribute('src');can.getContext('2d').clearRect(0,0,can.width,can.height);$(panel.toLowerCase()+'Status').textContent='NOT QUERIED';return}const sourceFrame=d.conditions.O.rows[i].source_frame_index;const j=p.rows.findIndex(r=>Number(r.source_frame_index)===Number(sourceFrame));if(j<0||!p.screenshots||!p.screenshots[j]){img.removeAttribute('src');can.getContext('2d').clearRect(0,0,can.width,can.height);$(panel.toLowerCase()+'Status').textContent=panel==='R'?'NOT QUERIED BEFORE R START / OUTSIDE R RANGE':(p.status||'NO RESULT');return}img.src='../'+p.screenshots[j];const row=p.rows[j];$(panel.toLowerCase()+'Status').innerHTML=`frame=${row.source_frame_index}, PTS=${Number(row.timestamp_s).toFixed(6)} s; query=${row.query_count}, UV=${row.uv_available_count}, visible=${row.visibility_count}, geometry=${row.geometry_valid_count??'UNKNOWN'}`;}
function render(){const o=d.conditions.O;if(!o)return;i=Math.max(0,Math.min(i,o.rows.length-1));$('frame').textContent=` O[${i}/${o.rows.length-1}] source_frame=${o.rows[i].source_frame_index} PTS=${Number(o.rows[i].timestamp_s).toFixed(6)} s`;draw('O');draw('R');draw('T')}
fetch('../case_data.json').then(x=>x.json()).then(x=>{d=x;const tr=x.manifest.support.selected_triplet;$('facts').innerHTML=`<p>source=${esc(x.manifest.source_id)} role=${esc(x.manifest.role)} window=${esc(x.manifest.window_id)}; query init=${Number(x.manifest.query_initialization.timestamp_s).toFixed(6)} s; ROI=${esc(x.manifest.roi.status)}; O selected H triplet=${tr?tr.triplet_id:'none'} with ${tr?tr.pair_member_slots.length:0} saved pair lines.</p>`;for(const k of ['R','T'])$(k.toLowerCase()+'Status').textContent=x.conditions[k]?.status||'NOT_RUN';$('oStatus').textContent='';render()});$('prev').onclick=()=>{i--;render()};$('next').onclick=()=>{i++;render()};
</script>"""
    review = output / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "index.html").write_text(page, encoding="utf-8")


def _write_report(output: Path, manifest: Mapping[str, Any], statuses: Mapping[str, Any]) -> None:
    rows = list(csv.DictReader((output / "frame_layer_counts_O.csv").open(encoding="utf-8")))
    target_rows = [row for row in rows if int(row["source_frame_index"]) in TARGET_FRAMES]
    lines = [
        "# V7 04LAX fake 局部观测恢复诊断结果",
        "",
        "这是一个单案例观测诊断，不是随机评价、检测器训练或 AUROC 实验。",
        "",
        "## 输入定位",
        "",
        f"- source/role/window：`{manifest['source_id']}` / `{manifest['role']}` / `{manifest['window_id']}`。",
        f"- 原视频：`{manifest['video']['path']}`；身份状态：`{manifest['video']['status']}`。",
        f"- 查询初始化：源帧 `{manifest['query_initialization']['source_frame_index']}`，PTS `{manifest['query_initialization']['timestamp_s']:.15f}` s。",
        "- 用户时刻按保存的真实 PTS 映射，而不是按 FPS 推算：",
        f"  - `{TARGET_TIMES[0]:.6f}` s → 源帧 `{TARGET_FRAMES[0]}`，PTS `{target_rows[0]['timestamp_s']}` s；",
        f"  - `{TARGET_TIMES[1]:.6f}` s → 源帧 `{TARGET_FRAMES[1]}`，PTS `{target_rows[1]['timestamp_s']}` s。",
        f"- 两帧相邻，PTS 间隔约 `{float(target_rows[1]['timestamp_s']) - float(target_rows[0]['timestamp_s']):.9f}` s；第一张不是查询初始化帧。",
        "",
        "## O 原有结果的逐层计数",
        "",
        "`uv_available` 是有限 UV 数，`visibility` 是 visibility 且 UV 有限，`geometry` 还要求 geometry_validity 与有限 XYZ。`local_group_member_count`（CSV 中兼容列名为 local_group_count）是落在已保存 H 保留组成员中的有效点数；`common_relation_support` 是已保存 H triplet pair 两端同时几何有效的关系数。",
        "",
        "| 源帧 | PTS(s) | query | UV | visibility | geometry | H成员有效点 | 关系支撑 | 绘图点 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        lines.append("| {source_frame_index} | {timestamp_s} | {query_count} | {uv_available_count} | {visibility_count} | {geometry_valid_count} | {local_group_count} | {common_relation_support} | {final_display_count} |".format(**row))
    lines += [
        "",
        "在源帧 487，289 个 query 全部可见、几何有效并显示；在源帧 488，三者均为 86。这个下降已经存在于 canonical NPZ 的 visibility/UV/geometry 层，不能归因于新页面的显示过滤。canonical NPZ 对不可见点将 UV 写为 NaN，所以未掩码 tracker 预测 UV 不可恢复；这不等于证明 tracker 没有预测输出。",
        "",
        "固定集合（源帧 487 首次核验时刻可见的 289 个 O query ID）在源帧 488 中仍有 86 个同时通过 visibility/geometry；这只是保存 ID 的状态统计，不证明物理对应必然正确。",
        "",
        "## R/T",
        "",
        f"- R：`{statuses.get('R', {}).get('status')}`。从源帧 `{statuses.get('R', {}).get('query_start_frame', 'UNKNOWN')}`（PTS `{statuses.get('R', {}).get('query_start_pts_s', 'UNKNOWN')}`）重新以 17×17/289 点查询；实际产物是独立 2D UV/visibility，未计算深度、pose、XYZ、H/B 支撑，不与 O ID 连接。耗时约 `{statuses.get('R', {}).get('elapsed_s', 'UNKNOWN')}` s。",
        f"- T：`{statuses.get('T', {}).get('status')}`。唯一候选为官方 CoTracker3 online（commit `{COTRACKER_SHA}`）；本次官方 hub 请求返回 `{statuses.get('T', {}).get('error', '未记录错误')}`，没有替换 tracker，也没有声称 T 已完成。",
        "",
        "## ROI 与显示边界",
        "",
        "用户截图是浏览器截图，当前没有可复核的浏览器到原视频像素变换文件，故 ROI 为 `PENDING_SOURCE_PIXEL_MAPPING`，没有编造 ROI 内外统计。页面逐帧显示源帧索引/PTS，并把 O、R、T 分栏；R 查询启动前没有轨迹，缺失不补零或 XYZ。",
        "",
        "页面：`review/index.html`；访问：",
        "```bash",
        f"python3 -m http.server 8765 --bind 127.0.0.1 --directory {output}",
        "# 浏览器打开 http://127.0.0.1:8765/review/",
        "```",
        "",
        "本结果不运行冻结检测器、不训练、不计算 AUROC，也不把单案例的点消失解释为伪造导致的跟踪失败。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, *, run_requery: bool = True, run_cotracker: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    manifest, o_rows = _load_o(output)
    (output / "review" / "screenshots").mkdir(parents=True, exist_ok=True)
    with np.load(PARTICLE, allow_pickle=False) as z:
        decoded = _decode_exact(FAKE_VIDEO, [int(x) for x in z["frame_indices"]])
        o_uv, o_vis, o_geo, o_xyz = z["uv"], z["visibility"], z["geometry_validity"], z["xyz"]
    selected_triplet = manifest.get("support", {}).get("selected_triplet") or {}
    selected_pairs = selected_triplet.get("pair_member_slots", [])
    manifest["screenshots"] = _save_pngs(output, decoded, o_rows, o_uv, o_vis, "O", geometry=o_geo, pair_slots=selected_pairs)
    media = output / "review" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for name, source in (("04LAX_fake.mp4", FAKE_VIDEO), ("04LAX_real_auxiliary.mp4", REAL_VIDEO)):
        target = media / name
        if source.exists() and not target.exists():
            target.symlink_to(source)
    manifest["media"] = {"fake": "review/media/04LAX_fake.mp4", "real_auxiliary": "review/media/04LAX_real_auxiliary.mp4" if REAL_VIDEO.exists() else None}
    _write_json(output / "roi_mapping.json", {
        "status": "PENDING_SOURCE_PIXEL_MAPPING",
        "source": "conversation screenshots",
        "source_image_size_hw": manifest["image_size_hw"],
        "rectangle_xyxy": None,
        "mapping_note": "Browser chrome/player controls make screenshot coordinates non-equivalent to source pixels; no calibrated source-pixel rectangle was supplied.",
        "no_roi_counts_emitted": True,
    })
    _write_csv(output / "frame_layer_counts_O.csv", o_rows)
    statuses: dict[str, Any] = {"O": {"status": "REUSED_CANONICAL_289", "rows": o_rows, "screenshots": manifest["screenshots"]}}
    if run_requery:
        statuses["R"] = _run_requery(output)
        if statuses["R"].get("artifact"):
            with np.load(statuses["R"]["artifact"], allow_pickle=False) as z:
                statuses["R"]["rows"] = list(csv.DictReader((output / "frame_layer_counts_R.csv").open(encoding="utf-8")))
    else:
        statuses["R"] = {"status": "NOT_RUN"}
    statuses["T"] = _run_cotracker(output) if run_cotracker else {"status": "NOT_RUN", "official_repo": "https://github.com/facebookresearch/co-tracker", "source_commit": COTRACKER_SHA}
    _write_json(output / "case_manifest.json", manifest)
    _write_json(output / "case_data.json", {"manifest": manifest, "conditions": statuses, "created_unix": time.time(), "elapsed_s": time.perf_counter() - started})
    _html_page(output, manifest, statuses)
    _write_report(output, manifest, statuses)
    rows = []
    for condition, status in statuses.items():
        if condition == "O":
            continue
        rows.append({"condition": condition, "status": status.get("status"), "elapsed_s": status.get("elapsed_s"), "error": status.get("error", "")})
    _write_csv(output / "condition_status.csv", rows)
    return {"manifest": manifest, "conditions": statuses, "elapsed_s": time.perf_counter() - started}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-requery", action="store_true")
    parser.add_argument("--run-cotracker", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.output, run_requery=not args.no_requery, run_cotracker=args.run_cotracker)
    print(json.dumps({"output": str(args.output), "elapsed_s": result["elapsed_s"], "R": result["conditions"]["R"].get("status"), "T": result["conditions"]["T"].get("status")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
