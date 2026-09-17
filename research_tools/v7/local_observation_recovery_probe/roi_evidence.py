"""First-stage manual ROI evidence for the 04LAX recovery case.

Only existing arrays and a fixed set of decoded source frames are read. No
tracking, depth, pose, segmentation, training, detector scoring, or AUROC is
run here. The second stage waits for user-confirmed frame-specific ROIs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np

from .probe import CASE_ID, FAKE_VIDEO, PARTICLE, SOURCE_ID, WINDOW_ID, _decode_exact
from research_tools.v7.local_structural_temporal_probe.representation import compute_local_derivatives

DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
OLD_CASE = DATA_ROOT / "derived/v7_activityforensics_local_observation_recovery_probe_v1"
DEFAULT_OUTPUT = OLD_CASE / "roi_evidence_review_v1"
OBS_SUPPORT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
O_PARTICLE = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1/particles/0002_MANIP_25__fake__density289.npz"
O_META = O_PARTICLE.with_suffix(".json")
FRAME_487, FRAME_488 = 487, 488
SUGGESTED_488 = [110, 0, 370, 359]

def sha256(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def finite_visible(uv: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    return np.isfinite(uv).all(axis=-1) & np.asarray(visibility, dtype=bool)

def frame_map(frames: np.ndarray) -> dict[int, int]:
    return {int(x): i for i, x in enumerate(np.asarray(frames).tolist())}

def point_payload(uv: np.ndarray, visibility: np.ndarray, geometry: np.ndarray | None) -> dict[str, Any]:
    uv = np.asarray(uv); visible = finite_visible(uv, visibility)
    geo = visible & np.asarray(geometry, dtype=bool) if geometry is not None else np.zeros(visible.shape, dtype=bool)
    return {"query_count": int(uv.shape[0]),
            "uv": [[float(x), float(y)] if ok else None for ok, (x, y) in zip(visible.tolist(), uv.tolist())],
            "visible": visible.tolist(), "geometry_valid": geo.tolist(),
            "visible_count": int(visible.sum()), "geometry_valid_count": int(geo.sum())}

def canonical_pairs(pairs: Iterable[Sequence[int]]) -> list[list[int]]:
    out: set[tuple[int, int]] = set()
    for pair in pairs:
        if len(pair) != 2: continue
        a, b = sorted((int(pair[0]), int(pair[1])))
        if a >= 0 and b >= 0 and a != b: out.add((a, b))
    return [[a, b] for a, b in sorted(out)]

def representation_audit(structure: Mapping[str, Any]) -> dict[str, Any]:
    triplets = list(structure.get("support", {}).get("triplets", []))
    valid = 0; errors = [0.0, 0.0, 0.0]; noncanonical = 0
    states_by_group: dict[int, dict[int, list[float]]] = {}
    members_by_group: dict[int, set[int]] = {}
    pairs_by_group: dict[int, set[tuple[int, int]]] = {}
    for item in triplets:
        states = np.asarray(item.get("states", []), dtype=float)
        times = np.asarray(item.get("timestamps_s", []), dtype=float)
        if states.shape != (3, 4) or times.shape != (3,) or not np.isfinite(states).all(): continue
        center, first, second = compute_local_derivatives(states, times)
        for j, name in enumerate(("center_state", "first_derivative", "second_derivative")):
            ref = np.asarray(item.get(name, []), dtype=float)
            if ref.shape == (4,): errors[j] = max(errors[j], float(np.max(np.abs((center, first, second)[j] - ref))))
        pairs = item.get("pair_ids", item.get("pair_indices", []))
        if any(int(a) >= int(b) for a, b in pairs if len((a, b)) == 2): noncanonical += 1
        gid = int(item.get("local_group_id", -1))
        members_by_group.setdefault(gid, set()).update(int(x) for x in item.get("members_considered", item.get("common_track_ids", [])))
        state_map = states_by_group.setdefault(gid, {})
        for slot, state in zip(item.get("target_slots", []), states.tolist()):
            slot = int(slot)
            if slot in state_map and state_map[slot] and not np.allclose(state_map[slot], state, rtol=0, atol=1e-10): state_map[slot] = []
            else: state_map[slot] = [float(x) for x in state]
        pairs_by_group.setdefault(gid, set()).update(tuple(sorted((int(a), int(b)))) for a, b in pairs if len((a, b)) == 2 and int(a) != int(b))
        valid += 1
    five = []
    for gid, slot_map in sorted(states_by_group.items()):
        if set(slot_map) >= set(range(5)) and all(slot_map.get(i) for i in range(5)):
            five.append({"local_group_id": gid, "member_slots": sorted(members_by_group.get(gid, set())), "pair_ids": [list(x) for x in sorted(pairs_by_group.get(gid, set()))], "states": [slot_map[i] for i in range(5)]})
    matches = structure.get("support", {}).get("target_matches", [])
    return {"saved_triplet_count": len(triplets), "finite_triplets_checked": valid,
            "derivative_recompute_max_abs": {"center": errors[0], "first": errors[1], "second": errors[2]},
            "noncanonical_pair_triplet_count": noncanonical,
            "target_frame_indices": [int(x["frame_index"]) for x in matches],
            "target_timestamps_s": [float(x["timestamp_s"]) for x in matches],
            "five_time_groups_reconstructed": len(five),
            "five_time_structure_status": "RECONSTRUCTABLE_FROM_SAVED_R_H_TRIPLETS" if five else "NOT_RECONSTRUCTABLE",
            "five_time_groups": five,
            "five_time_group_preview": five[:3],
            "q_status": "NOT_SAVED_IN_OLD_R_CASE; current model Q is not fabricated"}

def current_model_identity() -> dict[str, Any]:
    p = OBS_SUPPORT / "inputs/feature_manifest.json"; protocol = OBS_SUPPORT / "protocol.json"
    rows = read_json(p) if p.is_file() else []
    matches = [x for x in rows if str(x.get("source_id")) == SOURCE_ID or SOURCE_ID in str(x.get("window_id", ""))]
    d = read_json(protocol) if protocol.is_file() else {}; inp = d.get("input", {}) if isinstance(d, dict) else {}
    return {"experiment": str(OBS_SUPPORT), "feature_manifest": str(p), "feature_manifest_exists": p.is_file(),
            "matching_04LAX_feature_rows": len(matches), "matching_window_ids": [str(x.get("window_id")) for x in matches[:20]],
            "protocol_training_contains_04LAX": SOURCE_ID in list(inp.get("training_sources", [])),
            "protocol_validation_contains_04LAX": SOURCE_ID in list(inp.get("validation_sources", [])),
            "model_score_status": "NOT_AVAILABLE_NO_CURRENT_FROZEN_FEATURE_ROW_FOR_04LAX",
            "model_input_status": "NOT_CLAIMED_FROM_OLD_H_ONLY_CACHE"}

def draw_overlay(image_rgb: np.ndarray, payload: Mapping[str, Any], pairs: Sequence[Sequence[int]], label: str, color: tuple[int, int, int]) -> np.ndarray:
    import cv2
    image = np.asarray(image_rgb)[:, :, ::-1].copy(); uv = payload.get("uv", [])
    vis, geo = payload.get("visible", []), payload.get("geometry_valid", [])
    for a, b in pairs:
        if not (0 <= int(a) < len(uv) and 0 <= int(b) < len(uv)): continue
        if not (vis[int(a)] and vis[int(b)] and geo[int(a)] and geo[int(b)]): continue
        pa, pb = uv[int(a)], uv[int(b)]
        cv2.line(image, (round(pa[0]), round(pa[1])), (round(pb[0]), round(pb[1])), color, 1, lineType=cv2.LINE_AA)
    for i, p in enumerate(uv):
        if p is None or not vis[i]: continue
        cv2.circle(image, (round(p[0]), round(p[1])), 2, (70, 235, 70) if geo[i] else (0, 210, 255), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (4, 4), (330, 28), (15, 15, 15), -1)
    cv2.putText(image, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    return image[:, :, ::-1]

def save_png(path: Path, image_rgb: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), np.asarray(image_rgb)[:, :, ::-1])

def html_page() -> str:
    return r'''<!doctype html><meta charset="utf-8"><title>04LAX ROI evidence review</title>
<style>body{font-family:system-ui;background:#111827;color:#e5e7eb;margin:18px}h1{margin:0 0 8px}.card{background:#1f2937;border:1px solid #374151;border-radius:10px;padding:14px;margin:12px 0}button,select,input{font-size:15px;margin:3px;padding:5px}canvas{max-width:100%;border:1px solid #6b7280;background:#000;touch-action:none}.small{color:#cbd5e1;font-size:13px}.warn{color:#fbbf24}.coord{font-family:monospace}</style>
<h1>04LAX fake：人工 ROI → 结构证据准备</h1>
<div class="card"><div id="facts">读取中…</div><p class="warn">WAITING_FOR_ROI：建议框不是空间真值；点击不会写回服务器。frame487 的 R 状态固定显示为 NOT_QUERIED_BEFORE_R_START。</p></div>
<div class="card"><label>帧 <select id="frame"></select></label>
<label><input id="showO" type="checkbox" checked> O 点</label><label><input id="showOG" type="checkbox" checked> O geometry</label>
<label><input id="showR" type="checkbox" checked> R 点</label><label><input id="showRG" type="checkbox" checked> R geometry</label>
<label><input id="showPairs" type="checkbox" checked> 有效关系</label><p id="info" class="small"></p><canvas id="view" width="480" height="360"></canvas><p id="coord" class="coord">拖拽绘制原视频像素 ROI。</p></div>
<div class="card"><h2>逐帧 ROI</h2><label>状态 <select id="status"><option>PENDING</option><option>CONFIRMED</option><option>UNCERTAIN</option><option>SKIP</option></select></label>
<button id="clear">清除本帧</button><button id="export">下载 roi_annotations.json</button><button id="copy">复制 JSON</button>
<p class="small">每帧独立记录，不自动复制 frame488 框。CONFIRMED 只表示你确认粗矩形，不是 mask 或空间真值。</p><pre id="current" class="small"></pre></div>
<div class="card"><h2>五时刻与导出</h2><div id="targets" class="small"></div><p class="small">请至少查看 487、488 以及列出的五个 R 目标帧；不清楚的帧可选 UNCERTAIN 或 SKIP。下载文件后手动放回本目录或上传给我。</p></div>
<script>
let M,P,A={},drag=null;const $=x=>document.getElementById(x);
function f(){return P.frames.find(x=>x.source_frame_index===+$('frame').value)}
function ann(){let x=f(),a=A[x.source_frame_index];if(!a){let s=(x.source_frame_index===488&&M.old_roi&&M.old_roi.suggested_rectangle_xyxy)||null;A[x.source_frame_index]=a={source_frame_index:x.source_frame_index,pts_s:x.pts_s,x_min:s?s[0]:null,y_min:s?s[1]:null,x_max:s?s[2]:null,y_max:s?s[3]:null,coordinate_convention:'xyxy_source_pixel',human_confirmation_status:'PENDING'};}return a}
function draw(){let x=f();if(!x)return;let c=$('view'),ctx=c.getContext('2d'),im=new Image();im.onload=()=>{ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(im,0,0,c.width,c.height);let sx=c.width/x.width,sy=c.height/x.height;
function pts(r,show,onlyGeo,col){if(!show||!r)return;r.uv.forEach((p,i)=>{if(!p||!r.visible[i]||(onlyGeo&&!r.geometry_valid[i]))return;ctx.fillStyle=onlyGeo?'#4ade80':col;ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,2.2,0,Math.PI*2);ctx.fill()})}
pts(x.O,$('showO').checked,false,'#22d3ee');pts(x.O,$('showOG').checked,true,'#4ade80');pts(x.R,$('showR').checked,false,'#f472b6');pts(x.R,$('showRG').checked,true,'#facc15');
if($('showPairs').checked){function lines(r,pp,col){if(!r)return;ctx.strokeStyle=col;pp.forEach(ab=>{let a=r.uv[ab[0]],b=r.uv[ab[1]];if(!a||!b||!r.visible[ab[0]]||!r.visible[ab[1]]||!r.geometry_valid[ab[0]]||!r.geometry_valid[ab[1]])return;ctx.beginPath();ctx.moveTo(a[0]*sx,a[1]*sy);ctx.lineTo(b[0]*sx,b[1]*sy);ctx.stroke()})}lines(x.O,x.relations.O,'#fb923c');lines(x.R,x.relations.R,'#38bdf8')}
let a=ann();if(a.x_min!==null){ctx.strokeStyle=a.human_confirmation_status==='CONFIRMED'?'#ef4444':'#fbbf24';ctx.lineWidth=3;ctx.setLineDash(a.human_confirmation_status==='CONFIRMED'?[]:[7,5]);ctx.strokeRect(a.x_min*sx,a.y_min*sy,(a.x_max-a.x_min)*sx,(a.y_max-a.y_min)*sy);ctx.setLineDash([])}
$('info').textContent='frame '+x.source_frame_index+', PTS='+x.pts_s.toFixed(9)+' s; O='+(x.O?x.O.visible_count:'NA')+' visible/'+(x.O?x.O.geometry_valid_count:'NA')+' geometry; R='+(x.R?x.R.visible_count:'NOT_QUERIED')+' visible/'+(x.R?x.R.geometry_valid_count:'NA')+' geometry; R status='+x.R_status;$('current').textContent=JSON.stringify(a,null,2)};im.src='../frames/'+x.image}
function pos(e){let r=$('view').getBoundingClientRect(),x=f();return[Math.max(0,Math.min(x.width-1,(e.clientX-r.left)*x.width/r.width)),Math.max(0,Math.min(x.height-1,(e.clientY-r.top)*x.height/r.height))]}
$('view').onpointerdown=e=>{drag=pos(e);$('view').setPointerCapture(e.pointerId)};$('view').onpointermove=e=>{let p=pos(e);$('coord').textContent='原视频像素 x='+p[0].toFixed(1)+', y='+p[1].toFixed(1);if(!drag)return;let a=ann();a.x_min=Math.round(Math.min(drag[0],p[0]));a.x_max=Math.round(Math.max(drag[0],p[0]));a.y_min=Math.round(Math.min(drag[1],p[1]));a.y_max=Math.round(Math.max(drag[1],p[1]));draw()};$('view').onpointerup=()=>drag=null;
$('frame').onchange=draw;['showO','showOG','showR','showRG','showPairs'].forEach(x=>$(x).onchange=draw);$('status').onchange=()=>{ann().human_confirmation_status=$('status').value;draw()};$('clear').onclick=()=>{let a=ann();a.x_min=a.y_min=a.x_max=a.y_max=null;a.human_confirmation_status='PENDING';draw()};
function annFor(x){return A[x.source_frame_index]||{source_frame_index:x.source_frame_index,pts_s:x.pts_s,x_min:null,y_min:null,x_max:null,y_max:null,coordinate_convention:'xyxy_source_pixel',human_confirmation_status:'PENDING'}}
function data(){return{status:Object.values(A).some(x=>x.human_confirmation_status==='CONFIRMED')?'USER_CONFIRMATION_PARTIAL':'WAITING_FOR_ROI',source:M.source_id,role:M.role,window_id:M.window_id,video:M.video,not_ground_truth:true,entries:P.frames.map(x=>Object.assign({},annFor(x),{source:M.source_id,role:M.role,window_id:M.window_id,image_size_hw:[x.height,x.width]})),exported_at:new Date().toISOString()}}
$('export').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data(),null,2)],{type:'application/json'}));a.download='roi_annotations_04LAX.json';a.click()};$('copy').onclick=()=>navigator.clipboard&&navigator.clipboard.writeText(JSON.stringify(data(),null,2));
Promise.all([fetch('../case_manifest.json').then(r=>r.json()),fetch('frame_payload.json').then(r=>r.json())]).then(([m,p])=>{M=m;P=p;P.frames.forEach(x=>{let o=document.createElement('option');o.value=x.source_frame_index;o.textContent=x.source_frame_index+' ('+x.pts_s.toFixed(6)+' s)';$('frame').appendChild(o)});$('frame').value=488;$('facts').innerHTML='source='+M.source_id+' role='+M.role+' window='+M.window_id+'<br>原图='+M.image_size_hw[1]+'×'+M.image_size_hw[0]+'；UV='+M.coordinate_system.summary+'；<span class="warn">WAITING_FOR_ROI</span>';$('targets').textContent='R 五时刻目标： '+M.current_five_time.target_frame_indices.map((x,i)=>'frame '+x+' / '+M.current_five_time.target_timestamps_s[i].toFixed(9)+'s').join('；')+'。冻结模型：'+M.current_five_time.model_score_status;draw()}).catch(e=>$('facts').textContent='加载失败：'+e);
</script>'''

def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = read_json(OLD_CASE / "case_manifest.json"); structure = read_json(OLD_CASE / "conditions/R_structure.json"); old_roi = read_json(OLD_CASE / "roi_mapping.json")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(O_PARTICLE, allow_pickle=False) as oz, np.load(OLD_CASE / "conditions/R_geometry.npz", allow_pickle=False) as rz:
        of, ot, ouv, ovis, og = oz["frame_indices"], oz["timestamps_s"], oz["uv"], oz["visibility"], oz["geometry_validity"]
        rf, rt, ruv, rvis, rg = rz["frame_indices"], rz["timestamps_s"], rz["uv"], rz["visibility"], rz["geometry_validity"]
        if ouv.shape[1:] != (289, 2) or ruv.shape[1:] != (289, 2): raise ValueError("unexpected 289-query shape")
        image_hw = tuple(int(x) for x in oz["frame_sizes_hw"][0])
        if not np.all(np.asarray(oz["frame_sizes_hw"]) == np.asarray(rz["frame_sizes_hw"])[0]): raise ValueError("O/R frame-size mismatch")
    matches = structure.get("support", {}).get("target_matches", []); target_frames = [int(x["frame_index"]) for x in matches]
    frames = list(dict.fromkeys([FRAME_487, FRAME_488] + target_frames))
    decoded = _decode_exact(FAKE_VIDEO, frames); by_frame = {int(x.source_frame_index): x for x in decoded.frames}
    if set(by_frame) != set(frames): raise ValueError("exact source-frame decode incomplete")
    om, rm = frame_map(of), frame_map(rf)
    old_pairs = canonical_pairs((manifest.get("support", {}).get("selected_triplet") or {}).get("pair_member_slots", [])); rp: dict[int, list[list[int]]] = {}
    for t in structure.get("support", {}).get("triplets", []):
        if t.get("status") != "VALID": continue
        pairs = canonical_pairs(t.get("pair_ids", t.get("pair_indices", [])))
        for frame in t.get("frame_indices", []): rp.setdefault(int(frame), []).extend(pairs)
    rp = {k: canonical_pairs(v) for k, v in rp.items()}
    frame_rows = []; frame_dir = output / "frames"
    for frame in frames:
        src = by_frame[frame]; save_png(frame_dir / ("frame_%d.png" % frame), np.asarray(src.rgb))
        o = point_payload(ouv[om[frame]], ovis[om[frame]], og[om[frame]]) if frame in om else None
        r = point_payload(ruv[rm[frame]], rvis[rm[frame]], rg[rm[frame]]) if frame in rm else None
        r_status = "AVAILABLE" if r is not None else ("NOT_QUERIED_BEFORE_R_START" if frame < int(rf[0]) else "NO_SAVED_R_FRAME")
        rel_o = old_pairs if o is not None else []; rel_r = rp.get(frame, []) if r is not None else []
        save_png(frame_dir / ("frame_%d_O_overlay.png" % frame), draw_overlay(np.asarray(src.rgb), o or {"uv":[],"visible":[],"geometry_valid":[]}, rel_o, "O frame %d" % frame, (40,220,180)))
        save_png(frame_dir / ("frame_%d_R_overlay.png" % frame), draw_overlay(np.asarray(src.rgb), r or {"uv":[],"visible":[],"geometry_valid":[]}, rel_r, "R frame %d" % frame if r is not None else "R "+r_status, (220,180,40) if r is not None else (80,80,255)))
        frame_rows.append({"source_frame_index":frame,"pts_s":float(src.timestamp_s),"width":int(src.rgb.shape[1]),"height":int(src.rgb.shape[0]),"image":"frame_%d.png" % frame,"O":o,"R":r,"R_status":r_status,"relations":{"O":rel_o,"R":rel_r}})
    audit = representation_audit(structure); model = current_model_identity()
    write_json(output / "current_five_time_structure.json", {
        "status": audit["five_time_structure_status"],
        "target_frame_indices": audit["target_frame_indices"],
        "target_timestamps_s": audit["target_timestamps_s"],
        "groups": audit["five_time_groups"],
        "note": "Reconstructed from saved R H triplets with the existing representation arithmetic; Q and detector output are intentionally absent.",
    })
    with O_META.open(encoding="utf-8") as f: ometa = json.load(f)
    coord = {"summary":"source-video pixel XY (x=column, y=row), no resize transform in saved arrays","transform_to_original_xy":"identity","evidence":["O frame_sizes_hw is [360,480]","R frame_sizes_hw matches O and decoded source RGB","finite UV values lie within x<480,y<360"],"particle_provenance":ometa.get("provenance",{}),"warning":"Invisible UV is masked; ROI location is UNKNOWN for such points."}
    current = {"status":audit["five_time_structure_status"],"target_frame_indices":target_frames,"target_timestamps_s":[float(x["timestamp_s"]) for x in matches],"historical_frame_indices":[int(x) for x in structure.get("history_frame_indices",[])],"evaluation_frame_indices":[int(x) for x in structure.get("evaluation_frame_indices",[])],"canonical_pair_status":"CHECKED_FROM_SAVED_TRIPLETS","history_scale_status":"SAVED_PER_LOCAL_GROUP","q_status":audit["q_status"],"model_score_status":model["model_score_status"],"note":"Frame 487/488 are explanatory history/query frames; R targets are the five frames above."}
    out_manifest={"status":"WAITING_FOR_ROI","case_id":CASE_ID,"source_id":SOURCE_ID,"role":"fake","window_id":WINDOW_ID,"video":{"path":str(FAKE_VIDEO),"sha256":sha256(FAKE_VIDEO),"bytes":FAKE_VIDEO.stat().st_size,"status":"AVAILABLE"},"image_size_hw":list(image_hw),"coordinate_system":coord,"frame_contract":{"frame_487":{"source_frame_index":FRAME_487,"pts_s":float(by_frame[FRAME_487].timestamp_s),"purpose":"O history; R must show NOT_QUERIED"},"frame_488":{"source_frame_index":FRAME_488,"pts_s":float(by_frame[FRAME_488].timestamp_s),"purpose":"R query initialization"},"adjacent":True,"delta_s":float(by_frame[FRAME_488].timestamp_s-by_frame[FRAME_487].timestamp_s)},"query":{"condition":"R","source_frame_index":488,"pts_s":float(rt[0]),"query_count":289,"id_namespace":"R::independent_query_frame_488"},"current_five_time":current,"representation_audit":audit,"frozen_model_identity":model,"old_roi":{"status":old_roi.get("status"),"rectangle_xyxy":old_roi.get("rectangle_xyxy"),"suggested_rectangle_xyxy":old_roi.get("suggested_rectangle_xyxy"),"reusable_as_confirmed":False},"manual_review":{"annotation_file":"roi_annotations.json","no_server_write":True,"per_frame_rectangles":True,"not_spatial_ground_truth":True},"source_artifacts":{"old_case_manifest":str(OLD_CASE/"case_manifest.json"),"O_particle":str(O_PARTICLE),"R_geometry":str(OLD_CASE/"conditions/R_geometry.npz"),"R_structure":str(OLD_CASE/"conditions/R_structure.json")}}
    write_json(output/"case_manifest.json",out_manifest)
    entries=[{"source":SOURCE_ID,"role":"fake","window_id":WINDOW_ID,"source_frame_index":f,"pts_s":float(by_frame[f].timestamp_s),"image_size_hw":list(image_hw),"x_min":None,"y_min":None,"x_max":None,"y_max":None,"coordinate_convention":"xyxy_source_pixel; x=column, y=row","human_confirmation_status":"PENDING","suggested_rectangle_xyxy":SUGGESTED_488 if f==FRAME_488 else None,"not_ground_truth":True} for f in frames]
    write_json(output/"roi_annotations.json",{"status":"WAITING_FOR_ROI","not_ground_truth":True,"entries":entries})
    fields=["source_frame_index","pts_s","roi_status","visible_uv_count","geometry_valid_count","historical_member_count","current_common_member_count","intersecting_group_count","both_endpoint_pair_count","one_endpoint_pair_count","unknown_reason"]
    write_csv(output/"roi_support_counts.csv",[{"source_frame_index":f,"pts_s":float(by_frame[f].timestamp_s),"roi_status":"PENDING","visible_uv_count":"NA","geometry_valid_count":"NA","historical_member_count":"NA","current_common_member_count":"NA","intersecting_group_count":"NA","both_endpoint_pair_count":"NA","one_endpoint_pair_count":"NA","unknown_reason":"WAITING_FOR_ROI"} for f in frames],fields)
    write_json(output/"review/frame_payload.json",{"status":"WAITING_FOR_ROI","frames":frame_rows,"legend":{"visible":"finite UV and saved visibility","geometry":"saved geometry_validity and finite XYZ","relations":"saved pair identities only"}})
    (output/"review").mkdir(parents=True,exist_ok=True); (output/"review/index.html").write_text(html_page(),encoding="utf-8")
    write_json(output/"final_status.json",{"status":"WAITING_FOR_ROI","phase":"MANUAL_ROI_CONFIRMATION","output":str(output),"model_score_status":model["model_score_status"],"updated_unix":time.time()})
    report = report_text(out_manifest, audit, model); (output/"report.md").write_text(report,encoding="utf-8")
    return {"output":str(output),"status":"WAITING_FOR_ROI","frames":frames,"model_score_status":model["model_score_status"]}

def report_text(m: Mapping[str, Any], audit: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    t=m["current_five_time"]; fc=m["frame_contract"]
    lines=["# 04LAX ROI evidence review（第一阶段）","","状态：WAITING_FOR_ROI","",
    "本目录只准备人工逐帧 ROI 确认材料；没有运行 tracking、depth、pose、segmentation、训练或 detector score。没有生成 AUROC/AP 或空间定位指标。","",
    "## 直接结论","",
    "- 旧 ROI 仍为 PENDING_USER_CONFIRMATION；frame488 的建议框 [110, 0, 370, 359] 不是空间真值。",
    "- 原图尺寸 %sx%s（宽×高）；frame487 PTS %.15f，frame488 PTS %.15f，间隔 %.15f s。" % (m["image_size_hw"][1],m["image_size_hw"][0],fc["frame_487"]["pts_s"],fc["frame_488"]["pts_s"],fc["delta_s"]),
    "- R 在 frame488 才初始化；frame487 页面明确显示 NOT_QUERIED_BEFORE_R_START，不把 R 轨迹贴到该帧。",
    "- R H 缓存由保存的重叠三时刻 triplet 重建 %s 个五时刻组；目标帧 %s，不是 frame487/488。" % (audit["five_time_groups_reconstructed"],t["target_frame_indices"]),
    "- 当前冻结 observation_support_pilot_v1 没有 04LAX feature row（匹配行数 %s），冻结模型读出为 %s。" % (model["matching_04LAX_feature_rows"],model["model_score_status"]),
    "","## 坐标与身份核验","",
    "- O/R frame_sizes_hw 均为 [360,480]，与原视频解码尺寸一致；有限 UV 是原视频像素 XY（x=列、y=行），变换为 identity。canvas 缩放不改变导出的原始像素坐标。",
    "- 视频 sha256=%s；R query cohort=%s，query 起点 frame488。" % (m["video"]["sha256"],m["query"]["id_namespace"]),
    "- geometry_valid 是缓存掩码而非几何真值；被掩码 UV 第二阶段只能记 UNKNOWN。",
    "","## 五时刻表示审计","",
    "- 历史帧 %s；评估帧 %s；target_matches、共同成员、canonical pair、每组 history scale 已核对。" % (t["historical_frame_indices"],t["evaluation_frame_indices"]),
    "- 保存 triplet %s，有限并复算导数 %s；复算最大绝对误差 %s；非 canonical pair triplet %s。" % (audit["saved_triplet_count"],audit["finite_triplets_checked"],audit["derivative_recompute_max_abs"],audit["noncanonical_pair_triplet_count"]),
    "- Q 未保存在旧 R 案例中；不能把旧 H 结构直接冒充 STRUCTURE_SUPPORT 冻结模型输入。",
    "- 完整重建的 H 五时刻状态保存在 current_five_time_structure.json；它是审计材料，不是已评分模型输入。",
    "","## 请确认这些帧","",
    "1. frame487：只看 O 历史点覆盖与干净原图，R 应保持未查询。",
    "2. frame488：确认 R 查询起点附近的粗区域；建议框只作起点。",
    "3. R 五个目标帧：%s（PTS %s）。每帧独立画框，不复制 frame488 框。" % (t["target_frame_indices"],[round(x,9) for x in t["target_timestamps_s"]]),
    "","## 页面与导出","",
    "启动：python3 -m http.server 8765 --bind 127.0.0.1 --directory %s。" % str(DEFAULT_OUTPUT),
    "访问：http://127.0.0.1:8765/review/。选择帧、拖拽矩形，选择 CONFIRMED/UNCERTAIN/SKIP，点击下载 roi_annotations.json。",
    "浏览器下载不会自动写服务器文件；请将下载的 roi_annotations_04LAX.json 上传给我或手动保存为新目录的 roi_annotations.json。收到前不生成 ROI counts 或模型读出。",
    "","## 当前不输出","",
    "ROI 内点数、有效 pair、局部 logit、窗口 logit、空间覆盖率和正常对照暂不输出。"]
    return "\n".join(lines)+"\n"

def main(argv: Sequence[str] | None=None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=p.parse_args(argv)
    print(json.dumps(prepare(args.output),ensure_ascii=False,indent=2),flush=True); return 0

if __name__=="__main__": raise SystemExit(main())
