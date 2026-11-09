"""Matched two-pair RGB-D versus RGB-only fit; reuse the ADR-0036 worker."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .data import save
from .experiment import REPO, git
from .set_fit_check import SOURCE, inputs, worker
from scripts.prepare_v8_assets import digest


CONDITIONS = ("RGBD", "RGB_DEPTH_PARTITION")
OUT = SOURCE.parent / "pvst_rgb_partition_compare_v1"


def read(path: Path):
    return json.loads(path.read_text())


def run_arm(condition):
    arm = OUT / condition
    arm.mkdir()
    start = time.monotonic()
    with (arm / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", "research_tools.rgbd_slice.rgb_partition_compare",
             "--worker", condition],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        timed_out = False
        try:
            code = child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    save(arm / "exit.json", {
        "exit_code": code, "timed_out": timed_out, "wall_s": time.monotonic() - start,
    })


def summarize(out):
    arms = [out / name for name in CONDITIONS]
    exits = [read(arm / "exit.json") for arm in arms]
    results = [read(arm / "result.json") if (arm / "result.json").exists() else {} for arm in arms]
    complete = all(
        e["exit_code"] == 0 and not e["timed_out"]
        and r.get("status") == "COMPLETE" and r.get("updates") == 80
        for e, r in zip(exits, results)
    )
    runs = dict(zip(CONDITIONS, [dict(exit=e, result=r) for e, r in zip(exits, results)]))
    summary = {"status": "COMPLETE" if complete else "INCOMPLETE", "runs": runs}
    lines = ["# 同深度分片下 RGB-D 与 RGB 的限定拟合对照",
             f"状态：{summary['status']}。同一两对训练片段；不作为泛化评价。"]
    if complete:
        init = [read(arm / "initialization.json") for arm in arms]
        ids = [read(arm / "selected_identity.json") for arm in arms]
        configs = [read(arm / "config.json") for arm in arms]
        for cfg in configs:
            cfg.pop("condition")
        if (init[0]["model_sha256"] != init[1]["model_sha256"]
                or ids[0] != ids[1] or configs[0] != configs[1]):
            summary["status"] = "UNMATCHED"
            lines[1] = "状态：UNMATCHED。初始化、样本或训练配置不一致，不计算对照差值。"
        else:
            lines += ["", "|条件|pair|fake均值logit|real均值logit|fake Dice|fake IoU|real像素FPR|",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for name, arm in zip(CONDITIONS, arms):
                evaluations = [read(arm / f"evaluation_{step:03d}.json")["records"]
                               for step in (0, 80)]
                runs[name]["frame_bce"] = {
                    str(step): sum(row["terms"][0] for row in records) / len(records)
                    for step, records in zip((0, 80), evaluations)
                }
                runs[name]["final_records"] = evaluations[-1]
                by_pair = {(r["pair_id"], r["role"]): r for r in evaluations[-1]}
                for (pair, role), fake in by_pair.items():
                    if role != "fake":
                        continue
                    real = by_pair[pair, "real"]
                    f, r = fake["frame_logits"], real["frame_logits"]
                    lines.append(f"|{name}|{pair}|{sum(f)/len(f):.5f}|{sum(r)/len(r):.5f}|"
                                 f"{fake['dice']:.5f}|{fake['iou']:.5f}|{real['real_FPR']:.5f}|")
            for name in CONDITIONS:
                bce = runs[name]["frame_bce"]
                lines += ["", f"{name} 帧BCE（FP32）：{bce['0']:.5f} → {bce['80']:.5f}。"]
    for name, e, r in zip(CONDITIONS, exits, results):
        lines += ["", f"{name}：更新 {r.get('updates', '未记录')}/80，"
                  f"退出码 {e['exit_code']}，超时 {e['timed_out']}，耗时 {e['wall_s']:.2f} 秒。"]
    save(out / "summary.json", summary)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=CONDITIONS)
    args = parser.parse_args()
    if args.worker:
        arm = OUT / args.worker
        if not arm.is_dir() or (arm / "config.json").exists():
            raise ValueError("RUN_VIA_COMPARISON_PARENT")
        return worker(arm, condition=args.worker)
    if git("status", "--porcelain", "--", "research_tools/rgbd_slice", "research_tools/depth_slice"):
        raise ValueError("UNCOMMITTED_EXECUTION_SOURCE")
    inputs()
    OUT.mkdir(parents=True)
    save(OUT / "identity.json", {"head": git("rev-parse", "HEAD"),
         "manifest_sha256": digest(SOURCE / "manifest.json"),
         "conditions": list(CONDITIONS), "updates_per_condition": 80,
         "seconds_per_condition": 1800})
    for condition in CONDITIONS:
        print(f"Starting {condition}; log: {OUT / condition / 'worker.log'}", flush=True)
        run_arm(condition)
    return 0 if summarize(OUT) == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
