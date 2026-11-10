"""Compare frozen GRU input and output with identical train-only linear probes."""

import csv
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONDITION = "RGB_DEPTH_PARTITION"
SEED = 20260909
COUNTS = {"train": (580, 572), "dev": (462, 462)}


def validate_features(x, width):
    expected = (sum(sum(counts) for counts in COUNTS.values()), 2, width)
    if x.shape != expected:
        raise ValueError(f"Expected GRU feature shape {expected}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("GRU features contain nonfinite values")


def fit_probe(x_train, y_train):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED))
    model.fit(x_train, y_train)
    if model[-1].n_iter_.max() >= model[-1].max_iter:
        raise RuntimeError("Linear probe reached the iteration limit")
    return model


def metrics(y, z):
    return {"auroc": float(roc_auc_score(y, z)), "ap": float(average_precision_score(y, z))}


def score(rows, z):
    video = {}
    for r, x in zip(rows, z, strict=True):
        v = video.setdefault(r["video_id"], {
            "pair_id": r["pair_id"], "role": r["role"], "scores": []})
        v["scores"].append(float(x))
    pairs = {}
    for v in video.values():
        x = v.pop("scores")
        v["logit"] = float(np.logaddexp.reduce(x) - np.log(len(x)))
        pairs.setdefault(v["pair_id"], {})[v["role"]] = v["logit"]
    return {
        "frame": metrics([r["label"] for r in rows], z),
        "video": metrics([v["role"] == "fake" for v in video.values()],
                         [v["logit"] for v in video.values()]),
        "pairs": {k: {**v, "fake_minus_real": v["fake"] - v["real"]} for k, v in pairs.items()},
    }


def align(selected, record, indices):
    rows = [selected[i] for i in indices]
    if (set(selected) != set(indices) or len(indices) != len(selected)
            or [r["label"] for r in rows] != record["labels"]
            or [r["head_logit"] for r in rows] != record["scores"]):
        raise ValueError("Frame IDs, labels or logits differ from original evaluation")
    return rows


def collect(model, rows, config, clips):
    import torch
    from .experiment import evaluate

    calls, captures = [], {}
    video = None

    def progress(stage, **data):
        nonlocal video
        video = data["video"]
        captures[video] = {}
        print(stage, data, flush=True)

    def capture_gru(_module, args, output):
        calls.append(torch.stack((args[0][0].detach(), output[0].detach())))

    def capture_model(_module, args, output):
        fs = args[0]
        if len(calls) != len(fs) or len(output) != len(fs):
            raise ValueError("Expected one GRU call per input frame")
        values = torch.stack(calls).float().cpu().numpy()
        calls.clear()
        z = torch.stack([p["frame"] for p in output]).float().cpu().tolist()
        for f, x, logit in zip(fs, values, z, strict=True):
            i = f["frame_index"]
            if i in captures[video]:
                # The original evaluator discards left context in the final dev window.
                if clips is not None:
                    raise ValueError("Repeated training frame")
                continue
            captures[video][i] = {"frame_index": i, "pts": float(f["pts"]),
                                  "label": bool(f["gt"].any()), "head_logit": logit,
                                  "features": x}

    handles = (model.sets.gru.register_forward_hook(capture_gru),
               model.register_forward_hook(capture_model))
    try:
        with torch.no_grad():
            evaluation = evaluate(model, rows, config, progress, clip_list=clips, pixel_ap=False)
    finally:
        for handle in handles:
            handle.remove()
    output = []
    by_video = {r["video_id"]: r for r in rows}
    for r in evaluation["records"]:
        key = r["video_id"]
        meta = by_video[key]
        indices = (range(meta["actual_frames"]) if clips is None else
                   [i for c in clips if c["video_id"] == key for i in range(c["start"], c["end"])])
        for frame in align(captures[key], r, indices):
            if frame["pts"] != meta["pts"][frame["frame_index"]]["pts_s"]:
                raise ValueError("Frame timestamp differs from frozen manifest")
            output.append({**{k: r[k] for k in ("video_id", "pair_id", "split", "role")}, **frame})
    return output


def main():
    import torch
    from .experiment import REPO, git
    from .rgb_partition_anakin_dev import OUT, fingerprint, read, validate_split
    from .set_fit_check import build
    from .data import save
    from scripts.prepare_v8_assets import digest

    started = time.monotonic()
    out = OUT.parent / "pvst_gru_readout_probe_v1_retry"
    c, m, identity = (read(OUT / f"{name}.json") for name in ("config", "manifest", "identity"))
    folder = OUT / CONDITION
    weight_path = folder / "last.pt"
    expected = {"condition": CONDITION, "epoch": 12, "updates": 576,
                "identity": identity["identity"],
                "initial_model_sha256": read(OUT / "preflight.json")["initial_model_sha256"]}
    if (fingerprint({k: v for k, v in identity.items() if k != "identity"}) != identity["identity"]
            or digest(OUT / "manifest.json") != identity["frozen_manifest_sha256"]
            or digest(OUT / "config.json") != identity["config_sha256"]
            or c["seed"] != SEED or c["D"] != 512
            or validate_split(m["rows"], m["clips"]) != identity["counts"]
            or read(folder / "result.json")["status"] != "COMPLETE"):
        raise ValueError("Frozen run identity mismatch")
    for rel, sha in identity["code"].items():
        if digest(REPO / rel) != sha:
            raise ValueError(f"Frozen model source differs: {rel}")
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if any(state[k] != v for k, v in expected.items()):
        raise ValueError("Expected the frozen RGB_DEPTH_PARTITION epoch-12 checkpoint")

    out.mkdir()
    model = build(c, out, rgb_only=True).eval().requires_grad_(False)
    if read(out / "initialization.json")["model_sha256"] != expected["initial_model_sha256"]:
        raise ValueError("Model initialization differs from the original run")
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    frames = {}
    for split, counts in COUNTS.items():
        rr = [r for r in m["rows"] if r["split"] == split]
        frames[split] = collect(model, rr, c, m["clips"] if split == "train" else None)
        if tuple(np.bincount([int(r["label"]) for r in frames[split]], minlength=2)) != counts:
            raise ValueError(f"Frame label counts differ: {split}")
    del model
    rows = frames["train"] + frames["dev"]
    x = np.stack([r.pop("features") for r in rows])
    validate_features(x, c["D"])
    original = {r["video_id"]: r for r in read(folder / "last_dev.json")["records"]}
    reference_logits = []
    for r in frames["dev"]:
        ref = original[r["video_id"]]
        i = r["frame_index"]
        if r["label"] != ref["labels"][i]:
            raise ValueError("Frozen development labels do not reproduce")
        reference_logits.append(ref["scores"][i])
    n = len(frames["train"])
    y_train = [r["label"] for r in rows[:n]]
    logits = {"head_logit": np.array([r["head_logit"] for r in rows])}
    for j, name in enumerate(("u_logit", "h_logit")):
        probe = fit_probe(x[:n, j], y_train)
        logits[name] = probe.decision_function(x[:, j])
        for row, z in zip(rows, logits[name], strict=True):
            row[name] = float(z)
    results = {s: {name: score(frames[s], z[a:b]) for name, z in logits.items()}
               for s, a, b in (("train", 0, n), ("dev", n, len(rows)))}
    replay_error = float(np.max(np.abs(logits["head_logit"][n:] - reference_logits)))
    reference = score(frames["dev"], reference_logits)
    np.savez(out / "features.npz", u=x[:, 0], h=x[:, 1])
    with (out / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": "COMPLETE", **expected, "head": git("rev-parse", "HEAD"),
               "checkpoint_sha256": digest(weight_path), "frame_counts": COUNTS,
               "dev_original_head_max_abs_logit_delta": replay_error,
               "saved_dev_head_metrics": reference, "metrics": results,
               "seconds": time.monotonic() - started}
    lines = ["# 冻结 GRU 输入与输出线性读出", "",
             "第12轮／576步；8对训练、4对开发。标准化及逻辑回归只用训练帧。", "",
             "| 范围 | 读出 | 帧 AUROC | 帧 AP | 视频 AUROC | 视频 AP |",
             "|---|---|---:|---:|---:|---:|"]
    for split, readings in results.items():
        for name, result in readings.items():
            values = [result[k][m] for k in ("frame", "video") for m in ("auroc", "ap")]
            lines.append(f"| {split} | {name} | " + " | ".join(f"{v:.5f}" for v in values) + " |")
    lines += ["", "开发逐对视频分差（fake−real）：", "",
              "| pair | 原帧头 | u | h |", "|---|---:|---:|---:|"]
    for pair in results["dev"]["head_logit"]["pairs"]:
        values = [v["pairs"][pair]["fake_minus_real"] for v in results["dev"].values()]
        lines.append(f"| {pair} | " + " | ".join(f"{v:.5f}" for v in values) + " |")
    lines += ["", f"原开发帧 logit 最大复现差：`{replay_error:.6g}`。"
              f"原保存帧 AUROC/AP：{reference['frame']['auroc']:.5f}/{reference['frame']['ap']:.5f}。",
              "逐对分差仅供诊断；线性读出结果不能证明信息完全丢失。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(out / "summary.json", summary)
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    main()
