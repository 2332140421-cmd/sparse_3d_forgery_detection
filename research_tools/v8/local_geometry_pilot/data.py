"""Shared, content-corrected inputs derived only from frozen V8 caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import cell_neighbors, read_json, sha256_file, write_json_atomic, write_npz_atomic
from research_tools.v8.full_coverage_observation_field.pipeline import _UnionFind, _safe_name

SCHEMA = "v8-local-geometry-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def patches(a):
    if a.ndim == 4:
        return a.reshape(16, 16, 16, 16, 16, 3).transpose(0, 1, 3, 2, 4, 5).reshape(16, 256, 16, 16, 3)
    return a.reshape(16, 16, 16, 16, 16).transpose(0, 1, 3, 2, 4).reshape(16, 256, 16, 16)


def content_mask(frame_meta):
    result = np.zeros((16, 256, 256), bool)
    for ti, meta in enumerate(frame_meta):
        left, top = meta["mapping"]["pad_xy"]
        h, w = meta["mapping"]["content_hw"]
        result[ti, top:top + h, left:left + w] = True
    return result


def cell_mean(xyz, valid, content):
    mask = valid & content & np.isfinite(xyz).all(-1)
    count = mask.sum((2, 3))
    mu = np.where(mask[..., None], xyz, 0.).sum((2, 3), dtype=np.float64) / np.maximum(count[..., None], 1)
    return mu.astype(np.float32), count > 0, mask


def components(xyz, gv, motion, mv, active, scales):
    uf = _UnionFind(256)
    proposals = []
    for i, j in cell_neighbors():
        if not (active[i] and active[j]):
            continue
        terms = []
        if gv[i] and gv[j]: terms.append(float(np.linalg.norm(xyz[i] - xyz[j])) / scales["spatial_scale"])
        if mv[i] and mv[j]: terms.append(float(np.linalg.norm(motion[i] - motion[j])) / scales["motion_scale"])
        if terms: proposals.append((float(np.mean(terms)), int(i), int(j)))
    for cost, i, j in sorted(proposals):
        if cost <= scales["merge_threshold"]: uf.union(i, j, 4)
    roots = [uf.find(i) for i in range(256)]
    mapping = {root: k for k, root in enumerate(sorted(set(roots)))}
    return np.array([mapping[root] for root in roots], np.int64)


def atomic_npy(path, array):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def prepare(base: Path, out: Path, progress):
    rows = read_json(base / "data_manifest.json")["rows"]
    assert len(rows) == 801
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    assert len(train) == 718 and len(val) == 83
    assert len({r["source_id"] for r in train}) == 122 and len({r["source_id"] for r in val}) == 14
    assert not ({r["source_id"] for r in train} & {r["source_id"] for r in val})
    baseline_manifest_hash = sha256_file(base / "data_manifest.json")
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    identity = {"schema": SCHEMA, "baseline": str(base), "baseline_manifest_sha256": baseline_manifest_hash}
    complete = out / "inputs" / "manifest.json"
    if complete.exists():
        saved = read_json(complete)
        if saved["identity"] != identity: raise ValueError("INPUT_MANIFEST_IDENTITY_MISMATCH")
        for r in saved["rows"]:
            for kind in ("xyz", "valid", "content", "shared"):
                if sha256_file(Path(r[kind + "_path"])) != r[kind + "_sha256"]:
                    raise ValueError(f"INPUT_CACHE_CONTENT_MISMATCH:{r['window_id']}:{kind}")
        return rows, saved
    hash_cache = {}
    records = []
    for ix, row in enumerate(rows):
        name = _safe_name(row["window_id"])
        meta_path = base / "frontend" / f"{name}.json"
        front_path = base / "frontend" / f"{name}.npz"
        feature_path = base / "features" / f"{name}.npz"
        meta = read_json(meta_path)
        for key in ("window_id", "source_id", "role", "video_sha256"):
            if meta[key] != row[key]: raise ValueError(f"FRONTEND_IDENTITY:{row['window_id']}:{key}")
        assert meta["frame_indices"] == row["v8_frame_indices"]
        np.testing.assert_allclose(meta["pts_s"], row["v8_timestamps_s"], atol=1e-8, rtol=0)
        xyz, validity, sources = [], [], []
        for fm in meta["frame_meta"]:
            p = base / "raw_frontend" / "frames" / f"{fm['cache_key']}.npz"
            pm = read_json(p.with_suffix(".json"))
            if pm["video_sha256"] != row["video_sha256"] or pm["frame_index"] != fm["frame_index"]:
                raise ValueError("RAW_FRAME_IDENTITY")
            with np.load(p, allow_pickle=False) as z:
                xyz.append(z["points_raw"].astype(np.float32)); validity.append(z["validity"].astype(bool))
            if str(p) not in hash_cache: hash_cache[str(p)] = sha256_file(p)
            sources.append({"path": str(p), "sha256": hash_cache[str(p)], "frame_index": fm["frame_index"], "pts_s": fm["pts_s"]})
        xyz = patches(np.stack(xyz)); content = patches(content_mask(meta["frame_meta"]))
        valid = patches(np.stack(validity))
        mu, gv, valid = cell_mean(xyz, valid, content)
        finite_xyz = np.where(valid[..., None], xyz, 0.).astype(np.float32)
        with np.load(feature_path, allow_pickle=False) as z:
            x = z["x_full"].copy()
            shared = {k: z[k].copy() for k in ("predecessor", "association_weight", "history_available", "area", "delta_t")}
        with np.load(front_path, allow_pickle=False) as z:
            motion, mv = z["motion_xy"].copy(), z["motion_valid"].copy()
        area = content.sum((2, 3)).astype(np.float32)
        np.testing.assert_array_equal(area, shared["area"])
        x[..., 67:70] = mu; x[..., 70] = gv; x[..., 76] = ~gv
        active = area > 0
        x[~active] = 0.
        paths = {}
        for kind, array in (("xyz", finite_xyz), ("valid", valid), ("content", content)):
            p = out / "inputs" / f"{name}.{kind}.npy"
            atomic_npy(p, array); paths[kind + "_path"] = str(p); paths[kind + "_sha256"] = sha256_file(p)
        p = out / "inputs" / f"{name}.shared.npz"
        write_npz_atomic(p, x=x, mu=mu, gv=gv, motion=motion, mv=mv, **shared)
        paths.update(shared_path=str(p), shared_sha256=sha256_file(p))
        records.append({**row, **paths, "schema": SCHEMA, "sources": sources,
                        "source_feature_path": str(feature_path), "source_feature_sha256": sha256_file(feature_path),
                        "frontend_sha256": sha256_file(front_path), "metadata_sha256": sha256_file(meta_path)})
        progress("prepare", ix + 1, len(rows))
    # Fit common grouping statistics only on content-intersecting training edges.
    spatial, motion_dist, edge_rows = [], [], []
    edges = cell_neighbors(); ii, jj = edges.T
    x_sum = np.zeros(77, np.float64); x_sq = np.zeros(77, np.float64); x_n = 0
    abs_sum = np.zeros(3, np.float64); abs_sq = np.zeros(3, np.float64)
    rel_sq = np.zeros(3, np.float64); geo_n = 0
    for rec in records:
        if rec["split"] != "train": continue
        with np.load(rec["shared_path"], allow_pickle=False) as z:
            x, mu, gv, m, mv, area = [z[k] for k in ("x", "mu", "gv", "motion", "mv", "area")]
        active = area > 0
        a = x[active].astype(np.float64)
        x_sum += a.sum(0); x_sq += (a * a).sum(0); x_n += len(a)
        xyz = np.load(rec["xyz_path"], mmap_mode="r"); valid = np.load(rec["valid_path"], mmap_mode="r")
        values = xyz[valid].astype(np.float64)
        abs_sum += values.sum(0); abs_sq += (values * values).sum(0); geo_n += len(values)
        rel = (xyz - mu[..., None, None, :])[valid].astype(np.float64)
        rel_sq += (rel * rel).sum(0)
        both = active[:, ii] & active[:, jj]
        sg = both & gv[:, ii] & gv[:, jj]; sm = both & mv[:, ii] & mv[:, jj]
        sd = np.linalg.norm(mu[:, ii] - mu[:, jj], axis=-1)
        md = np.linalg.norm(m[:, ii] - m[:, jj], axis=-1)
        spatial.append(sd[sg]); motion_dist.append(md[sm]); edge_rows.append((sd, md, sg, sm))
    sv, mvv = np.concatenate(spatial), np.concatenate(motion_dist)
    ss = float(np.median(sv[sv > 1e-8])) if np.any(sv > 1e-8) else 1.
    ms = float(np.median(mvv[mvv > 1e-8])) if np.any(mvv > 1e-8) else 1.
    costs = []
    for sd, md, sg, sm in edge_rows:
        count = sg.astype(int) + sm.astype(int)
        costs.append(((sd / ss * sg + md / ms * sm) / np.maximum(count, 1))[count > 0])
    scales = {"spatial_scale": ss, "motion_scale": ms, "merge_threshold": float(np.quantile(np.concatenate(costs), .25)),
              "max_cells_per_component": 4, "fit_split": "train", "content_edges_only": True}
    x_mean = x_sum / x_n; x_scale = np.sqrt(np.maximum(x_sq / x_n - x_mean ** 2, 0))
    x_scale[x_scale < 1e-6] = 1.
    absolute_mean = abs_sum / geo_n
    absolute_scale = np.sqrt(np.maximum(abs_sq / geo_n - absolute_mean ** 2, 0)); absolute_scale[absolute_scale < 1e-6] = 1.
    relative_scale = np.sqrt(rel_sq / geo_n); relative_scale[relative_scale < 1e-6] = 1.
    statistics = {"cell_mean": x_mean.tolist(), "cell_scale": x_scale.tolist(), "fit_active_cell_times": x_n,
                  "absolute_mean": absolute_mean.tolist(), "absolute_scale": absolute_scale.tolist(),
                  "relative_scale": relative_scale.tolist(), "fit_valid_content_pixels": geo_n,
                  "formula": "cell=(x-train_active_mean)/train_active_std; absolute=(XYZ-train_valid_pixel_mean)/train_valid_pixel_std; relative=(XYZ-mu)/train_valid_pixel_RMS_relative; states unchanged; invalid continuous coordinates=0"}
    for ix, rec in enumerate(records):
        with np.load(rec["shared_path"], allow_pickle=False) as z: shared = {k: z[k] for k in z.files}
        active = shared["area"] > 0
        shared["component_ids"] = np.stack([components(shared["mu"][t], shared["gv"][t], shared["motion"][t], shared["mv"][t], active[t], scales) for t in range(16)])
        aw, pi = shared["association_weight"], shared["predecessor"]
        for t in range(1, 16):
            aw[t] = np.where(active[t - 1][pi[t]] & active[t, :, None], aw[t], 0.)
            aw[t] /= np.maximum(aw[t].sum(-1, keepdims=True), 1e-8)
        aw[0] = 0.
        write_npz_atomic(rec["shared_path"], **shared)
        rec["shared_sha256"] = sha256_file(rec["shared_path"])
        progress("grouping", ix + 1, len(records))
    write_json_atomic(out / "standardizer.json", statistics)
    write_json_atomic(out / "grouping_config.json", scales)
    manifest = {"identity": identity, "rows": records, "standardizer_sha256": sha256_file(out / "standardizer.json"),
                "grouping_sha256": sha256_file(out / "grouping_config.json")}
    manifest["input_id"] = digest(manifest)
    write_json_atomic(complete, manifest)
    write_json_atomic(out / "data_manifest.json", {"rows": rows, "baseline_manifest_sha256": baseline_manifest_hash})
    return rows, manifest


class Dataset(torch.utils.data.Dataset):
    def __init__(self, records, statistics):
        self.records = records
        self.mean = np.array(statistics["cell_mean"], np.float32)
        self.scale = np.array(statistics["cell_scale"], np.float32)
        self.shared = []
        for rec in records:
            with np.load(rec["shared_path"], allow_pickle=False) as z:
                a = {k: z[k].copy() for k in ("x", "mu", "component_ids", "predecessor", "association_weight", "history_available", "area", "delta_t")}
            a["x"] = ((a["x"] - self.mean) / self.scale).astype(np.float32)
            a["x"][a["area"] == 0] = 0.
            self.shared.append(a)

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        result = {k: torch.from_numpy(v) for k, v in self.shared[i].items()}
        for k in ("xyz", "valid", "content"):
            result[k] = torch.from_numpy(np.array(np.load(rec[k + "_path"], mmap_mode="r"), copy=True))
        result["label"] = torch.tensor(float(rec["label"]))
        result["index"] = i
        return result
