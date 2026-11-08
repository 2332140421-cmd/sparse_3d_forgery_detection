"""Small official-asset preview; no models, inferred masks or dataset split creation.

CLI caller prepares the requested review queue and emits CSV/HTML outside Git.
Existing ANAKIN tools assume aligned ZIP masks and cannot describe these missing
or unresolved annotations. This entry only selects, downloads, decodes and renders.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import av
import requests

SEED = 20260909
FP_REV = "2ee8f98a94ef41e25cf548b6763570d125ce0f5d"
HFV_REV = "7a567eef47636ed2d3729d20ad7551bc73b79616"
TT_REV = "6010d790fd6bb99992adb91ab9857f96d5253e05"
AF_REV = "a34d4b7b04b0f3f3e26ba900adc367218667c581"
LABELS = {0: "spatial anomaly", 1: "appearance anomaly", 2: "motion anomaly", 3: "real"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rows, count, key):
    rows = sorted(rows, key=key)
    random.Random(SEED).shuffle(rows)
    return rows[:count]


def fetch(root, relative, url):
    """Exact single file; completed files are reused, failed partials stay separate."""
    path = root / relative
    log = root / "downloads" / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    start, count = time.monotonic(), 0
    record = {"url": url, "path": relative, "status": "FAILED"}
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            with path.with_suffix(path.suffix + ".part").open("wb") as f:
                for chunk in r.iter_content(1024**2):
                    f.write(chunk)
                    count += len(chunk)
            if r.headers.get("Content-Length") and count != int(r.headers["Content-Length"]):
                raise ValueError("INCOMPLETE_RESPONSE")
        path.with_suffix(path.suffix + ".part").replace(path)
        record.update(status="COMPLETE", sha256=sha(path))
    except Exception as e:
        record["error"] = str(e)
    record.update(downloaded_bytes=count, seconds=time.monotonic() - start)
    save(log, record)
    print(record, flush=True)
    return path


def row(dataset, revision, sid, role, official, local, category, **more):
    return dict(dataset=dataset, revision=revision, sample_id=sid, role=role,
                source_id="", official_split="unspecified", category=category,
                official_path=official, local_path=local, original="", paired="no",
                time_annotation="not published", mask_type="无公开像素GT",
                mask_path="", alignment="not available", **more)


def fakeparts(root):
    folder = root / "fakeparts"
    for name in ("metadata.jsonl", "manifests/inpainting__propainter.jsonl", "manifests/interpolation__framer.jsonl"):
        fetch(root, "fakeparts/" + name,
              f"https://huggingface.co/datasets/hi-paris/FakeParts/resolve/{FP_REV}/{name}?download=true")
    if sha(folder / "metadata.jsonl") != "1d5195ced5ef8b30eaf94c2c72c0930f93c7f200c558af17df2a082d86b0189f":
        raise ValueError("FAKEPARTS_METADATA_SHA_MISMATCH")
    spatial = [json.loads(x) for x in (folder / "manifests/inpainting__propainter.jsonl").read_text().splitlines()]
    temporal = [json.loads(x) for x in (folder / "manifests/interpolation__framer.jsonl").read_text().splitlines()]
    selected = choose(spatial, 3, lambda x: x["sample_id"]) + choose(temporal, 1, lambda x: x["sample_id"])
    metadata_path = folder / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError("Official metadata.jsonl must be downloaded first")
    needed = {r["sample_id"] for r in selected}
    metadata = {}
    with metadata_path.open() as f:
        for line in f:
            item = json.loads(line)
            if item["sample_id"] in needed:
                metadata[item["sample_id"]] = item
    for item in selected:
        published = metadata[item["sample_id"]]
        for key in ("file_name", "frame_count", "fps"):
            if item[key] != published[key]:
                raise ValueError("METADATA_MANIFEST_CONFLICT:" + item["sample_id"] + ":" + key)
    save(folder / "selected_metadata.json", metadata)
    save(folder / "selection.json", {"seed": SEED, "rule": "sorted sample_id then seeded shuffle; 3 ProPainter + 1 Framer; no semantic category available; human activity unconfirmed before viewing", "rows": selected})
    result, tasks = [], []
    for item in selected:
        # Actual official manifests have null masks. Never reinterpret an upstream
        # multi-instance object label or generation mask as final edit-region GT.
        if item.get("mask") is not None:
            raise ValueError("NEW_MASK_REQUIRES_MANUAL_PROVENANCE_REVIEW")
        source = item.get("source_video")
        local = "fakeparts/" + item["file_name"]
        r = row("FakeParts", FP_REV, item["sample_id"], "fake", item["file_name"], local,
                item["task"] + "/" + item["method"])
        r.update(source_id=source or "missing", original="fakeparts/" + source if source else "",
                 paired="manifest basename link; exact generation mapping unresolved",
                 official_split="full (preview pool)",
                 time_annotation=json.dumps({"indexing": item.get("frame_indexing"), "real_frames": item.get("real_frames"), "fake_frames": item.get("fake_frames")}),
                 alignment="manifest mask=null; published assets/masks absent at pinned revision; target instance unknown",
                 provenance=item)
        result.append(r)
        tasks.append((local, item["file_name"]))
        if source:
            result.append(row("FakeParts", FP_REV, "source:" + source, "real", source,
                              "fakeparts/" + source, "retained source; reconstruction/selection caveat"))
            result[-1].update(source_id=source, official_split="full (preview pool)")
            tasks.append(("fakeparts/" + source, source))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda t: fetch(root, t[0], f"https://huggingface.co/datasets/hi-paris/FakeParts/resolve/{FP_REV}/{t[1]}?download=true"), tasks))
    return result


def extract_member(root, archive, member, destination):
    target = root / destination
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    # Full CRC read also verifies reused extracted members without trusting name.
    import zlib
    checksum = 0
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            checksum = zlib.crc32(chunk, checksum)
    entry = archive.getinfo(member)
    if target.stat().st_size != entry.file_size or checksum != entry.CRC:
        raise ValueError("ARCHIVE_MEMBER_MISMATCH:" + member)


def hfv(root):
    path = root / "hfv/cls_test_82_win_video.zip"
    if not path.exists():
        return []
    expected = "b63b1ef0e69cfe512e4f04ade59caa5bc800d070f5cba8134bf77eb3d88d74e4"
    if sha(path) != expected:
        raise ValueError("HFV_OFFICIAL_SHA_MISMATCH")
    result = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        save(root / "hfv/archive_inventory.json", [{"name": n, "bytes": z.getinfo(n).file_size} for n in names])
        train = [n for n in names if Path(n).name == "train_4.csv"]
        if len(train) != 1:
            raise ValueError("HFV_TRAIN_CSV_AMBIGUOUS")
        text = z.read(train[0]).decode("utf-8-sig")
        (root / "hfv/train_4.csv").write_text(text)
        # Author's loader uses whitespace-separated path, integer label, score.
        labels = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 2 or fields[1] not in {"0", "1", "2", "3"}:
                raise ValueError("HFV_CSV_FORMAT_NOT_VERIFIED:" + line[:100])
            labels.append((fields[0], int(fields[1])))
        selected = []
        for label in (0, 2, 3):
            selected += choose([x for x in labels if x[1] == label], 2, lambda x: x[0])
        save(root / "hfv/selection.json", {"seed": SEED, "class_map": LABELS, "rows": selected,
             "mapping_evidence": "HumanSAM ICCV2025 section3.1; author dataset loader; train_4.csv"})
        for filename, label in selected:
            matches = [n for n in names if Path(n).name == Path(filename).name and n.lower().endswith((".mp4", ".avi", ".mov"))]
            if len(matches) != 1:
                raise ValueError("HFV_MEMBER_AMBIGUOUS:" + filename)
            member = matches[0]
            local = "hfv/videos/" + Path(member).name
            extract_member(root, z, member, local)
            r = row("HFV", HFV_REV, filename, "real" if label == 3 else "fake", member, local, f"{label}: {LABELS[label]}")
            r.update(official_split="train_4.csv", source_id="unpaired", paired="no; independent real/fake videos", label_id=label)
            result.append(r)
    return result


def activityforensics(root, base):
    from scripts.prepare_v8_assets import identities, real_path, verify
    rows = identities(base)
    # Official AF train CSV membership, not the source Charades split in basename.
    candidates = [r for r in rows if r["official_split"] == "train" and r["scene_source"] == "charades"
                  and real_path(base, "charades", r["source_id"].split(":")[1]).is_file()]
    random.Random(SEED).shuffle(candidates := sorted(candidates, key=lambda r: r["video_id"]))
    selected, seen = [], set()
    for r in candidates:
        if r["source_id"] not in seen:
            selected.append(r)
            seen.add(r["source_id"])
        if len(selected) == 2:
            break
    save(root / "activityforensics/selection.json", {"seed": SEED, "rule": "AF official train, Charades, existing verified source; seeded sorted video_id; distinct sources", "rows": selected})
    with zipfile.ZipFile(base / "metadata/Charades.zip") as z:
        originals = {r["id"]: r for r in csv.DictReader(io.StringIO(z.read("Charades/Charades_v1_train.csv").decode()))}
    published = json.loads((base / "asset_preparation/official_files.json").read_text())
    nonmedia = [r["path"] for r in published["files"] if not r["path"].lower().endswith(".mp4")]
    with zipfile.ZipFile(base / "metadata/annot.zip") as z:
        annot_names = z.namelist()
    save(root / "activityforensics/annotation_evidence.json", {"revision": AF_REV, "published_non_mp4": nonmedia,
         "annotation_zip_members": annot_names, "note": "official time text annotations; no spatial annotation files found; feature archives not read"})
    result, checks = [], []
    entries = {r["path"]: r for r in published["files"]}
    for item in selected:
        source = item["source_id"].split(":")[1]
        target = base / "official" / AF_REV / item["media_path"]
        original = real_path(base, "charades", source)
        check = verify(target, entries[item["media_path"]])
        if check["status"] != "VERIFIED":
            raise ValueError("AF_FAKE_IDENTITY_MISMATCH:" + item["video_id"])
        import zlib
        crc = 0
        with original.open("rb") as f:
            for chunk in iter(lambda: f.read(1024**2), b""):
                crc = zlib.crc32(chunk, crc)
        with zipfile.ZipFile(base / "real_archives/Charades_v1_480.zip") as z:
            member = z.getinfo("Charades_v1_480/" + source + ".mp4")
        if crc != member.CRC or original.stat().st_size != member.file_size:
            raise ValueError("CHARADES_SOURCE_CRC_MISMATCH:" + source)
        checks.append({"fake": item["video_id"], "verification": check, "source": source, "source_archive_crc_verified": True})
        fake_local, real_local = "activityforensics/videos/" + item["video_id"] + ".mp4", "activityforensics/videos/" + source + ".mp4"
        for local, src in [(fake_local, target), (real_local, original)]:
            if not src.is_file():
                raise FileNotFoundError(src)
            p = root / local
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.symlink_to(src)
            if p.resolve() != src.resolve():
                raise ValueError("SYMLINK_IDENTITY_CONFLICT")
        info = originals.get(source, {})
        category = "activity deletion" if "_delete@" in item["media_path"] else "activity addition"
        r = row("ActivityForensics", AF_REV, item["video_id"], "fake", item["media_path"], fake_local, category)
        r.update(source_id=item["source_id"], original=real_local, paired="yes; exact Charades source ID",
                 official_split="train", time_annotation=json.dumps(item["intervals_s"]) + " seconds; endpoint convention unspecified",
                 description="原始动作: " + info.get("descriptions", info.get("script", "not published")) + "; 修改: " + category + " (official filename tag; specific edited action text not published in downloaded annotation)")
        result.append(r)
        result.append(row("ActivityForensics", AF_REV, item["source_id"], "real", str(original), real_local, info.get("descriptions", "Charades source")))
        result[-1].update(source_id=item["source_id"], official_split="source for AF train fake; Charades train")
    save(root / "activityforensics/selected_verification.json", checks)
    return result


def ttdf(root):
    path = root / "ttdf/c40.zip"
    if not path.exists():
        return []
    result, used, selected = [], set(), []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        save(root / "ttdf/archive_inventory.json", {"archive_sha256": sha(path),
             "drive_file_id": "1eI8iRCPQb0Z2jmnIMUi1qeBRi8ih1jAn",
             "members": [{"name": n, "bytes": z.getinfo(n).file_size} for n in names]})
        # Fixed Match-only review: same source supplies identity and motion.
        # Do not guess the order of two IDs in Mismatch filenames without a key.
        for method in ("magicDance", "magicAnimate", "animateAnyone"):
            candidates = [n for n in names if n.startswith(f"c40/match/{method}/") and n.endswith(".mp4")]
            shuffled = choose(candidates, len(candidates), lambda n: n)
            member = next(n for n in shuffled if Path(n).stem not in used)
            sid = Path(member).stem
            used.add(sid)
            original = "c40/original/" + sid + ".mp4"
            if original not in names:
                raise ValueError("TTDF_MATCH_ORIGINAL_MISSING:" + original)
            local, real_local = "ttdf/videos/" + method + "_match_" + sid + ".mp4", "ttdf/videos/original_" + sid + ".mp4"
            selected.append({"fake": member, "original": original, "configuration": "Match", "source": sid,
                             "identity_source": sid, "motion_source": sid})
            extract_member(root, z, member, local)
            extract_member(root, z, original, real_local)
            r = row("TT-DF", TT_REV, member, "fake", member, local, method + "/Match/c40")
            r.update(source_id=sid, original=real_local, paired="Match; same ID identity/reference and driving video",
                     identity_source=sid, motion_source=sid,
                     alignment="c40 archive has videos only; no pixel masks or independent edit time annotations")
            result.append(r)
            result.append(row("TT-DF", TT_REV, "original:" + sid, "real", original, real_local, "c40 original"))
            result[-1].update(source_id=sid)
        save(root / "ttdf/selection.json", {"seed": SEED, "rule": "one Match per method, sorted then seeded shuffle, distinct source IDs; no Mismatch ID-order inference", "rows": selected})
    return result


def decode(path, picture_dir):
    """Fully decode and retain actual PTS; second CPU pass stores only fixed frames."""
    start = time.monotonic()
    pts, ticks, dimensions = [], [], set()
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_count = 2
        rate, time_base = str(s.average_rate), str(s.time_base)
        for f in c.decode(s):
            if f.pts is None:
                raise ValueError("MISSING_PTS")
            ticks.append(f.pts)
            pts.append(float(f.pts * f.time_base))
            dimensions.add((f.width, f.height))
    if not pts or len(dimensions) != 1 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("INVALID_VIDEO_SEQUENCE")
    duration = pts[-1] - pts[0] + (pts[-1] - pts[-2] if len(pts) > 1 else 0)
    wanted = {min(range(len(pts)), key=lambda i: abs(pts[i] - (pts[0] + duration * fraction))): fraction for fraction in (.25, .5, .75)}
    pictures = []
    picture_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_count = 2
        for i, f in enumerate(c.decode(s)):
            if i in wanted:
                p = picture_dir / f"{int(100*wanted[i]):02d}.jpg"
                image = f.to_image()
                image.thumbnail((640, 640))
                image.save(p, quality=85)
                pictures.append({"fraction": wanted[i], "frame": i, "pts_s": pts[i], "path": str(p)})
            if i >= max(wanted):
                break
    width, height = dimensions.pop()
    return dict(frames=len(pts), width=width, height=height, fps=rate, time_base=time_base,
                duration_s=duration, pts_s=pts, pts_ticks=ticks, pictures=pictures,
                decode_seconds=time.monotonic()-start, bytes=path.stat().st_size, sha256=sha(path), decode="COMPLETE")


def render(root, rows):
    parts = ["<!doctype html><meta charset='utf-8'><title>候选伪造数据预览</title><style>body{font-family:sans-serif;max-width:1250px;margin:24px auto}video{width:46%;max-height:420px}img{max-width:31%}article{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap}</style><h1>候选数据人工预览</h1><p>按元数据预选；不按伪造明显程度取样。播放器 preload=none。没有确认对应的官方像素GT时不制作叠加。截图按实际PTS的25%、50%、75%，缩小仅用于显示。</p>"]
    for dataset in ("FakeParts", "HFV", "TT-DF", "ActivityForensics"):
        parts.append("<h2>" + dataset + "</h2>")
        group = [r for r in rows if r["dataset"] == dataset]
        if not group:
            parts.append("<p>未取得可解码样本；见review_summary.md的分发／下载限制。</p>")
        for i, r in enumerate(group, 1):
            esc = html.escape
            parts.append(f"<article><h3>{dataset}-{i:02d} {esc(r['role'])}: {esc(r['sample_id'])}</h3><p>{esc(r['category'])}</p>")
            parts.append("<pre>" + esc(str(r.get("description", "")) + "\n同源关系: " + r["paired"] + "\n时间: " + r["time_annotation"] + "\n空间标注: " + r["mask_type"] + "\n" + r["alignment"]) + "</pre>")
            if r.get("decode") == "COMPLETE":
                def player(local):
                    url = esc(os.path.relpath(root / local, root / "preview"), quote=True)
                    return f"<video controls preload='none' src='{url}'></video><a href='{url}'>原文件</a>"
                parts.append("<p>当前视频；右侧为官方指向的源视频（存在时，具体对应限制见上文）。</p>" + player(r["local_path"]))
                if r["original"] and (root / r["original"]).is_file():
                    parts.append(player(r["original"]))
                parts.append(f"<p>{r['frames']}帧；{r['width']}×{r['height']}；{r['fps']}fps；{r['duration_s']:.3f}s</p>")
                if r["role"] == "fake":
                    for p in r["pictures"]:
                        url = esc(os.path.relpath(p["path"], root / "preview"), quote=True)
                        parts.append(f"<span style='display:inline-block;width:32%'><img style='max-width:100%' src='{url}' loading='lazy'><br>{p['fraction']:.0%}; frame={p['frame']}; PTS={p['pts_s']:.6f}s<br>mask／叠加：无经核验的公开像素GT</span>")
                    parts.append("<p>无公开且经对应核验的像素GT；不显示mask或伪造叠加。</p>")
            else:
                parts.append("<p>无法取得／解码: " + esc(r.get("error", "missing")) + "</p>")
            parts.append("</article>")
    (root / "preview/index.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--stage", choices=("prepare", "render"), default="prepare")
    args = p.parse_args()
    root = args.root
    if args.stage == "prepare":
        rows = fakeparts(root) + hfv(root) + activityforensics(root, args.base) + ttdf(root)
        save(root / "selected_rows.json", rows)
    rows = json.loads((root / "selected_rows.json").read_text())
    for i, r in enumerate(rows):
        cache = root / "preview/stats" / f"{i:02d}.json"
        if cache.exists():
            old = json.loads(cache.read_text())
            if old["local_path"] != r["local_path"]:
                raise ValueError("PREVIEW_CACHE_IDENTITY_CONFLICT")
            # Reuse measurements, not old provenance/annotation descriptions.
            r.update({k: v for k, v in old.items() if k not in r})
            continue
        try:
            r.update(decode(root / r["local_path"], root / "preview/stills" / f"{i:02d}"))
        except Exception as e:
            r.update(decode="FAILED", error=str(e))
        save(cache, r)
        print(r["dataset"], r["sample_id"], r["decode"], flush=True)
    save(root / "results.json", rows)
    keys = ["dataset", "revision", "sample_id", "role", "source_id", "official_split", "category", "official_path", "local_path", "original", "paired", "frames", "width", "height", "fps", "duration_s", "time_base", "time_annotation", "mask_type", "mask_path", "alignment", "decode", "sha256", "bytes"]
    with (root / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    render(root, rows)
    print(Counter((r["dataset"], r["role"], r["decode"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
