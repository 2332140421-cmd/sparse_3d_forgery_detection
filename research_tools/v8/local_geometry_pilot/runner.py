"""Single recoverable execution path for the two-condition V8 cache pilot."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
import random
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file, write_json_atomic
from research_tools.v8.full_coverage_observation_field.pipeline import _bootstrap_diff, _metric, _source_class_weights, _source_macro
from .data import Dataset, SCHEMA, digest, prepare
from .model import LocalGeometryModel

CONDITIONS = ("DENSE_LOCAL", "MEAN_REPLICATED")
SEEDS = (20260909, 20260910, 20260911)
STOP = False


def atomic_torch(path, value):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def validate_checkpoint(payload, expected_identity):
    required = {"identity", "model", "optimizer", "epoch", "loss_history", "rng", "order_rng", "training_elapsed_s",
                "microbatch", "peak_allocated_gib", "initial_parameter_sha256"}
    if not required.issubset(payload) or payload["identity"] != expected_identity:
        raise ValueError("CHECKPOINT_IDENTITY_OR_FIELDS_MISMATCH")
    if not 0 < payload["epoch"] <= 100 or len(payload["loss_history"]) != payload["epoch"]:
        raise ValueError("CHECKPOINT_EPOCH_HISTORY_MISMATCH")
    if not np.isfinite(payload["loss_history"]).all() or not np.isfinite(payload["training_elapsed_s"]):
        raise ValueError("CHECKPOINT_NONFINITE_RECORD")


def move(batch):
    return {k: v.to("cuda", non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def loader(ds, order=None, microbatch=8):
    sampler = order if order is not None else range(len(ds))
    return torch.utils.data.DataLoader(ds, batch_size=microbatch, sampler=sampler, num_workers=2, pin_memory=True,
                                       prefetch_factor=2, persistent_workers=False)


def predict(model, ds):
    model.eval(); values = []
    with torch.inference_mode():
        for batch in loader(ds): values.append(model(move(batch)).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all(): raise FloatingPointError("NONFINITE_SCORES")
    return result


def train_epoch(model, ds, order, weights, opt, microbatch):
    model.train()
    losses = []; group_position = 0
    opt.zero_grad(set_to_none=True)
    for batch in loader(ds, order, microbatch):
        batch = move(batch)
        size = len(batch["label"])
        group_size = min(8, len(ds) - (group_position // 8) * 8)
        z = model(batch)
        w = weights[batch["index"]]
        loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(z, batch["label"], weight=w, reduction="sum")
        (loss_sum / group_size).backward()
        losses.append(loss_sum.detach())
        group_position += size
        if group_position % 8 == 0 or group_position == len(ds):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
    loss = torch.stack(losses).sum() / len(ds)
    if not torch.isfinite(loss): raise FloatingPointError("NONFINITE_LOSS")
    return float(loss.item())


def smoke(out, train_ds, statistics, input_id):
    indices = [next(i for i, r in enumerate(train_ds.records) if r["role"] == role) for role in ("real", "fake")]
    ds = torch.utils.data.Subset(train_ds, indices)
    batch = move(next(iter(loader(ds))))
    seed_all(SEEDS[0]); model = LocalGeometryModel("DENSE_LOCAL", statistics).cuda()
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
    z = model(batch); loss = torch.nn.functional.binary_cross_entropy_with_logits(z, batch["label"])
    loss.backward()
    gradients = [p.grad for p in model.patch_encoder.parameters()]
    if not all(g is not None and torch.isfinite(g).all() for g in gradients): raise ValueError("SMOKE_GEOMETRY_GRADIENT")
    if not any(g.abs().sum() > 0 for g in gradients): raise ValueError("SMOKE_ZERO_GEOMETRY_GRADIENT")
    opt.step(); torch.cuda.synchronize(); elapsed = time.monotonic() - start
    model.eval()
    with torch.inference_mode(): reference = model(batch)
    smoke_dir = out / "smoke"; smoke_dir.mkdir(exist_ok=True)
    atomic_torch(smoke_dir / "model.pt", {"model": model.state_dict(), "optimizer": opt.state_dict(), "input_id": input_id})
    other = LocalGeometryModel("DENSE_LOCAL", statistics).cuda()
    saved_smoke = torch.load(smoke_dir / "model.pt", weights_only=False)
    other.load_state_dict(saved_smoke["model"]); other.eval()
    restored_opt = torch.optim.AdamW(other.parameters(), lr=.001, weight_decay=.0001)
    restored_opt.load_state_dict(saved_smoke["optimizer"])
    with torch.inference_mode(): reload_scores = other(batch)
    torch.testing.assert_close(reference, reload_scores, atol=1e-6, rtol=1e-6)
    # A second identical step confirms optimizer state restoration as well.
    for net, optimizer in ((model, opt), (other, restored_opt)):
        net.train(); optimizer.zero_grad(set_to_none=True)
        second_loss = torch.nn.functional.binary_cross_entropy_with_logits(net(batch), batch["label"])
        second_loss.backward(); optimizer.step(); net.eval()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    evidence = {"status": "PASS", "input_id": input_id, "windows": [train_ds.records[i]["window_id"] for i in indices],
                "loss": float(loss), "save_reload_max_error": float((reference - reload_scores).abs().max()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "two_window_step_s": elapsed,
                "windows_per_second": 2 / elapsed, "parameter_count": sum(p.numel() for p in model.parameters()),
                "metric_format": _metric(batch["label"].cpu().numpy(), reload_scores.cpu().numpy()), "scope": "smoke only; models not formal"}
    write_json_atomic(smoke_dir / "summary.json", evidence)
    print("SMOKE", evidence, flush=True)
    del batch, model, other, opt, restored_opt
    return evidence


def train(out, train_ds, val_ds, statistics, identity, progress):
    model_dir = out / "models"; model_dir.mkdir(exist_ok=True)
    weight_map = _source_class_weights(train_ds.records)
    weights = torch.tensor([weight_map[r["window_id"]] for r in train_ds.records], dtype=torch.float32, device="cuda")
    write_json_atomic(out / "training_weights.json", weight_map)
    records = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            key = f"{condition}__seed{seed}"
            path = model_dir / f"{key}.pt"; last = model_dir / f"{key}.last.pt"
            model_identity = {**identity, "condition": condition, "seed": seed}
            seed_all(seed)
            model = LocalGeometryModel(condition, statistics).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            order_rng = torch.Generator().manual_seed(seed)
            initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
            epoch = 0; history = []; elapsed = 0.; microbatch = 8; peak = 0.
            if path.exists() or last.exists():
                payload = torch.load(path if path.exists() else last, map_location="cpu", weights_only=False)
                validate_checkpoint(payload, model_identity)
                model.load_state_dict(payload["model"]); opt.load_state_dict(payload["optimizer"])
                epoch, history, elapsed = payload["epoch"], payload["loss_history"], payload["training_elapsed_s"]
                order_rng.set_state(payload["order_rng"]); restore_rng(payload["rng"])
                microbatch = payload["microbatch"]; peak = payload["peak_allocated_gib"]
                if payload["initial_parameter_sha256"] != initial_hash: raise ValueError("INITIALIZATION_MISMATCH")
            print(f"MODEL {key} epoch={epoch} device={next(model.parameters()).device}", flush=True)
            while epoch < 100:
                epoch_start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                order = torch.randperm(len(train_ds), generator=order_rng).tolist()
                loss = train_epoch(model, train_ds, order, weights, opt, microbatch)
                torch.cuda.synchronize()
                elapsed += time.monotonic() - epoch_start; history.append(loss); epoch += 1
                peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
                payload = {"identity": model_identity, "condition": condition, "seed": seed, "epoch": epoch,
                           "model": model.state_dict(), "optimizer": opt.state_dict(), "loss_history": history,
                           "rng": rng_state(), "order_rng": order_rng.get_state(), "training_elapsed_s": elapsed,
                           "microbatch": microbatch, "peak_allocated_gib": peak, "parameter_count": sum(p.numel() for p in model.parameters()),
                           "initial_parameter_sha256": initial_hash}
                atomic_torch(last, payload)
                progress("train", len(records), 6, condition=condition, seed=seed, epoch=epoch, loss=loss, training_elapsed_s=elapsed)
                print(f"EPOCH {key} {epoch}/100 loss={loss:.8g} cumulative_s={elapsed:.2f}", flush=True)
                if STOP: raise InterruptedError("SAFE_STOP_AFTER_SAVED_EPOCH")
            if not path.exists(): atomic_torch(path, payload)
            score_path = model_dir / f"{key}.scores.npz"
            eval_start = time.monotonic()
            if not score_path.exists():
                tz, vz = predict(model, train_ds), predict(model, val_ds)
                tmp = score_path.with_name(score_path.name + ".tmp")
                with tmp.open("wb") as f: np.savez(f, train_scores=tz, val_scores=vz, identity=digest(model_identity))
                os.replace(tmp, score_path)
            else:
                with np.load(score_path, allow_pickle=False) as z:
                    if str(z["identity"]) != digest(model_identity): raise ValueError("SCORE_IDENTITY")
            record = {"condition": condition, "seed": seed, "epoch": epoch, "status": "COMPLETE", "checkpoint": str(path),
                      "checkpoint_sha256": sha256_file(path), "parameter_count": payload["parameter_count"], "loss_first": history[0],
                      "loss_last": history[-1], "loss_epochs": len(history), "training_elapsed_s": elapsed, "peak_allocated_gib": peak,
                      "evaluation_this_call_s": time.monotonic() - eval_start, "identity": model_identity,
                      "initial_parameter_sha256": initial_hash}
            csv_rows(model_dir / f"{key}.loss.csv", [{"epoch": i + 1, "loss": value} for i, value in enumerate(history)])
            records.append(record); write_json_atomic(model_dir / "fold_models.json", {"planned": 6, "completed": len(records), "failed": 0, "records": records})
            del model, opt
    return records


def csv_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate(out, rows, records):
    evaluation = out / "evaluation"; evaluation.mkdir(exist_ok=True)
    metrics = []; sources = []; windows = []; macro = {}; seed_macro = {}
    for condition in CONDITIONS:
        for split, field in (("train", "train_scores"), ("validation", "val_scores")):
            rr = [r for r in rows if r["split"] == split]; y = np.array([r["label"] for r in rr])
            zz = []
            for seed in SEEDS:
                with np.load(out / "models" / f"{condition}__seed{seed}.scores.npz", allow_pickle=False) as data: z = data[field].copy()
                zz.append(z)
            for seed, z in list(zip(SEEDS, zz)) + [("MEAN_LOGIT", np.mean(zz, axis=0))]:
                if len(z) != len(rr) or not np.isfinite(z).all(): raise ValueError("EVALUATION_SCORES")
                sm = _source_macro(rr, z, y, [r["window_id"] for r in rr])
                metrics.append({"condition": condition, "seed": seed, "split": split, **_metric(y, z), "source_macro_auroc": sm["mean"], "source_count": sm["source_count"]})
                for source, auc in sm["per_source"].items():
                    subset = [r for r in rr if r["source_id"] == source]
                    sources.append({"condition": condition, "seed": seed, "split": split, "source": source, "auroc": auc,
                                    "n_real": sum(r["label"] == 0 for r in subset), "n_fake": sum(r["label"] == 1 for r in subset)})
                if split == "validation":
                    if seed == "MEAN_LOGIT": macro[condition] = sm["per_source"]
                    else: seed_macro[(condition, seed)] = sm["mean"]
                for r, score in zip(rr, z):
                    windows.append({"condition": condition, "seed": seed, "split": split, "window_id": r["window_id"], "source": r["source_id"], "label": r["label"], "logit": float(score)})
    comparison = _bootstrap_diff(macro[CONDITIONS[0]], macro[CONDITIONS[1]], reps=10000, seed=20260909)
    differences = [{"source": s, "dense_auroc": macro[CONDITIONS[0]][s], "mean_auroc": macro[CONDITIONS[1]][s], "difference": d} for s, d in comparison["per_source"].items()]
    for row in differences:
        for seed in SEEDS:
            values = {r["condition"]: r["auroc"] for r in sources if r["split"] == "validation" and r["source"] == row["source"] and r["seed"] == seed}
            row[f"difference_seed_{seed}"] = values[CONDITIONS[0]] - values[CONDITIONS[1]]
    comparison["directions"] = {"positive": sum(r["difference"] > 0 for r in differences), "tied": sum(r["difference"] == 0 for r in differences), "negative": sum(r["difference"] < 0 for r in differences)}
    comparison["per_seed"] = {str(seed): seed_macro[(CONDITIONS[0], seed)] - seed_macro[(CONDITIONS[1], seed)] for seed in SEEDS}
    summary = {"metrics": metrics, "comparison": comparison, "models": records, "ensemble": "average three logits then metrics", "threshold": "logit >= 0"}
    write_json_atomic(evaluation / "summary.json", summary)
    csv_rows(evaluation / "metrics.csv", metrics); csv_rows(evaluation / "per_source_metrics.csv", sources)
    csv_rows(evaluation / "per_source_differences.csv", differences)
    (out / "scores").mkdir(exist_ok=True); csv_rows(out / "scores" / "window_scores.csv", windows)
    return summary


def report(out, summary, smoke_record, stage_time):
    # One small standalone loss plot from the saved complete histories.
    colors = ("#1565c0", "#d32f2f", "#388e3c", "#7b1fa2", "#ef6c00", "#00838f")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480" viewBox="0 0 900 480">', '<rect width="900" height="480" fill="white"/>', '<text x="40" y="24">Weighted window BCE, all 100 epochs</text>', '<path d="M50 40V390H720" fill="none" stroke="black"/>']
    for i, rec in enumerate(summary["models"]):
        key = f"{rec['condition']}__seed{rec['seed']}"
        with (out / "models" / f"{key}.loss.csv").open() as f:
            losses = [float(r["loss"]) for r in csv.DictReader(f)]
        points = " ".join(f"{50 + j / 99 * 670:.2f},{390 - min(l, 1.) * 330:.2f}" for j, l in enumerate(losses))
        svg += [f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="1.4"/>', f'<text x="730" y="{55+i*38}" font-size="11" fill="{colors[i]}">{rec["condition"]}<tspan x="730" dy="14">{rec["seed"]}</tspan></text>']
    for epoch in (1, 25, 50, 75, 100):
        svg.append(f'<text x="{50+(epoch-1)/99*670:.1f}" y="412" font-size="12">{epoch}</text>')
    for loss in (0., .2, .4, .6, .8, 1.):
        svg.append(f'<text x="10" y="{394-loss*330:.1f}" font-size="12">{loss:.1f}</text>')
    svg += ['<text x="360" y="450">Epoch</text>', '</svg>']
    (out / "evaluation" / "train_loss.svg").write_text("\n".join(svg), encoding="utf-8")
    lines = ["# V8 局部几何编码匹配对照 pilot", "", "工程状态：COMPLETE；DENSE_LOCAL / MEAN_REPLICATED 共6模型，每个100 epochs。",
             "", "数据：718训练窗口/122 source；83验证窗口/14 source（42 real/41 fake）；16个冻结时刻。",
             "", "两条件共享修正后的内容mask、77维输入、粗component、关联、RGB、标准化和初始化。唯一输入差异是几何卷积的真实XYZ与格均值复制。",
             "", "| split | condition | macro AUROC | pooled AUROC | AP | P | R | F1 | ACC | TN/FP/FN/TP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in summary["metrics"]:
        if m["seed"] != "MEAN_LOGIT": continue
        values = [m[k] for k in ("source_macro_auroc", "auroc", "ap", "precision", "recall", "f1", "accuracy")]
        lines.append(f"| {m['split']} | {m['condition']} | " + " | ".join("NA" if v is None else f"{v:.6f}" for v in values) + f" | {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']} |")
    c = summary["comparison"]
    lines += ["", f"主比较 DENSE_LOCAL−MEAN_REPLICATED：{c['mean']:.6f}；95% source paired bootstrap CI {c['ci95']}（10000次，seed=20260909）。",
              f"source方向：{c['directions']}；seed差值：{c['per_seed']}。", "", "## 成本与验证", "",
              f"GPU：{torch.cuda.get_device_name()}；两条件参数量相同：{summary['models'][0]['parameter_count']}。",
              f"正式训练累计秒：{sum(r['training_elapsed_s'] for r in summary['models']):.3f}；阶段耗时：{stage_time}。",
              f"峰值allocated显存GiB：{max(r['peak_allocated_gib'] for r in summary['models']):.3f}；smoke：{smoke_record}。",
              "", "## 结论与边界", "",
              "主点估计和seed/source方向见上表。区间跨0表示当前不确定性较大，不能认定等价或稳定有效。",
              "这是多轮开发使用的窗口级验证集；没有独立sealed test、帧级/像素级定位真值。本轮未单独估计padding修正收益。",
              "旧identityfix FULL仅是历史版本参考（macro=0.785714），新旧同时有padding与编码器差异，不能将其差值归因于局部形态。",
              "本轮未验证全视频连续帧、全帧component图、空间定位和未知source独立泛化；完成后不自动追加候选。"]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8"); (out / "review_summary.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--stage", choices=("prepare", "smoke", "all"), default="all")
    args = parser.parse_args(); base, out = Path(args.baseline).resolve(), Path(args.output).resolve()
    if base.name != "v8_full_coverage_observation_field_identityfix_v1": raise ValueError("ONLY_IDENTITYFIX_BASELINE_ALLOWED")
    if out == base or base in out.parents: raise ValueError("OUTPUT_MUST_BE_INDEPENDENT")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "protocol.json").exists() and not args.resume: raise ValueError("EXISTING_RUN_REQUIRES_RESUME")
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    def term(signum, frame):
        global STOP
        STOP = True
        print("TERM requested; finishing and saving current epoch", flush=True)
    signal.signal(signal.SIGTERM, term)
    start = time.monotonic(); phase = "prepare"
    costs = read_json(out / "stage_costs.json") if (out / "stage_costs.json").exists() else {}
    def progress(stage, completed, total, **extra):
        write_json_atomic(out / "progress.json", {"status": "RUNNING", "stage": stage, "completed": completed, "total": total, "pid": os.getpid(), "updated_unix": time.time(), **extra})
    try:
        write_json_atomic(out / "final_status.json", {"status": "RUNNING", "pid": os.getpid(), "started_unix": time.time()})
        ts = time.monotonic(); rows, manifest = prepare(base, out, progress)
        costs["prepare_cumulative_s"] = costs.get("prepare_cumulative_s", 0.) + time.monotonic() - ts
        statistics = read_json(out / "standardizer.json")
        config = {"schema": SCHEMA, "conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": 100,
                  "optimizer": "AdamW", "lr": .001, "weight_decay": .0001, "effective_batch": 8, "microbatch": 8,
                  "clip_norm": 1., "hidden": 128, "patch_chunk": 2048, "geometry_encoder": "Conv7-32(3x3)-ReLU-Conv32-32(3x3)-ReLU-content mean+max ->64",
                  "geometry_input": "absolute XYZ3+relative XYZ3+valid1; mean condition replaces absolute with cell mean and relative with zero; masks identical",
                  "standardization": statistics, "padding": "content-only statistics/means/grouping/messages/pooling/history; area-zero strictly excluded",
                  "loss": "source/class weighted window BCE; final epoch; threshold logit>=0", "bootstrap": {"replicates": 10000, "seed": 20260909, "unit": "source"},
                  "baseline": str(base), "input_id": manifest["input_id"], "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "code_sha256": {name: sha256_file(Path(__file__).parent / name) for name in ("model.py", "data.py", "runner.py")},
                  "torch": torch.__version__, "device": torch.cuda.get_device_name(), "randomness": {"cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}}
        identity = {"input_id": manifest["input_id"], "standardizer_sha256": sha256_file(out / "standardizer.json"), "config_id": digest({k: v for k, v in config.items() if k != "git_head"})}
        protocol = out / "protocol.json"
        if protocol.exists():
            old = read_json(protocol)
            if old["identity"] != identity: raise ValueError("PROTOCOL_RESUME_MISMATCH")
        else: write_json_atomic(protocol, {**config, "identity": identity})
        write_json_atomic(out / "stage_costs.json", costs)
        if args.stage == "prepare":
            write_json_atomic(out / "final_status.json", {"status": "PREPARED", "models": 0, "input_id": manifest["input_id"]})
            return
        train_ds = Dataset([r for r in manifest["rows"] if r["split"] == "train"], statistics)
        val_ds = Dataset([r for r in manifest["rows"] if r["split"] == "validation"], statistics)
        phase = "smoke"; ts = time.monotonic()
        smoke_path = out / "smoke" / "summary.json"
        smoke_record = read_json(smoke_path) if smoke_path.exists() else smoke(out, train_ds, statistics, manifest["input_id"])
        if smoke_record["input_id"] != manifest["input_id"]: raise ValueError("SMOKE_INPUT_IDENTITY")
        costs["smoke_cumulative_s"] = costs.get("smoke_cumulative_s", 0.) + time.monotonic() - ts
        if args.stage == "smoke":
            write_json_atomic(out / "stage_costs.json", costs)
            write_json_atomic(out / "final_status.json", {"status": "SMOKE_PASS", "models": 0, "input_id": manifest["input_id"]})
            return
        phase = "train"; records = train(out, train_ds, val_ds, statistics, identity, progress)
        phase = "evaluate"; ts = time.monotonic(); summary = evaluate(out, rows, records)
        costs["evaluate_cumulative_s"] = costs.get("evaluate_cumulative_s", 0.) + time.monotonic() - ts
        phase = "report"; ts = time.monotonic(); report(out, summary, smoke_record, costs)
        costs["report_cumulative_s"] = costs.get("report_cumulative_s", 0.) + time.monotonic() - ts
        costs["this_invocation_s"] = time.monotonic() - start
        write_json_atomic(out / "stage_costs.json", costs)
        write_json_atomic(out / "final_status.json", {"status": "COMPLETE", "models": 6, "epoch_each": 100, "report_complete": True, "finished_unix": time.time(), "input_id": manifest["input_id"]})
        write_json_atomic(out / "progress.json", {"status": "COMPLETE", "stage": "report", "completed": 1, "total": 1, "updated_unix": time.time()})
    except BaseException as exc:
        costs["last_failed_invocation_s"] = time.monotonic() - start
        write_json_atomic(out / "stage_costs.json", costs)
        write_json_atomic(out / "final_status.json", {"status": "STOPPED_SAFE" if isinstance(exc, InterruptedError) else "FAILED", "stage": phase,
                                                      "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "updated_unix": time.time()})
        raise


if __name__ == "__main__": main()
