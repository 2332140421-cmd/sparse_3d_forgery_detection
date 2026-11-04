# PVST 当前状态

2026-11-03；分支 `pvst-point-voxel-v03`；v0.3起点 `29205c78c85ad5a3e23237b7cf8aae4690fbbc70`。
状态RESOURCE_BLOCKED（v0.3三个候选尺度均超训练预算；未运行GPU探针，模型0/2、轮次0；见[结果](results_v03.md)）

本轮设计：[v0.3尺度与片段协议](design_v03.md)；旧v0.2规格与失败历史以下保留。

## v0.2历史记录（以下保留为背景，不是v0.3待运行流程）

完整规格：[design_v02.md](design_v02.md)；方法决策：[ADR-0029](../decisions/0029-pvst-point-voxel-v02.md)。
本轮限定资源验收：[最终资源与吞吐报告](compact_execution_results_v1.md)、[紧凑执行说明](compact_execution_check_v1.md)。31项测试通过；原失败窗口通过两次更新，最大节点窗口在fine第一层全量LayerNorm的FP32转换申请8.01GiB失败，卸载未解决。任务已退出，不放行4090正式训练。

最新历史运行报错：[node_chunks_v2报告](failure_report_node_chunks_v2.md)。该失败记录保持不变；本轮另用独立资源目录验收。

这是一套独立PVST实现，不覆盖旧V8/V9。实际运行状态以数据目录progress.json为准；本页记录新执行版本；旧失败原因见[报错报告](failure_report_v02.md)，修复范围与验收口径见[FFN执行修复](ffn_execution_fix_v02.md)。

固定数据：[身份清单](manifest_v02.json)、[场景预审](scene_review_v02.json)。
AF训练8 pair/16视频/13,213帧，开发4 pair/8视频/5,799帧；
ANAKIN训练8 pair/16视频/2,138帧，开发4 pair/8视频/924帧。
共32训练、16开发视频，22,074实际帧；保留原角色与全部PTS，不读测试集。
原片固定25/50/75%预审后仅将明显转向相机的AF BTW1H按原seed同角色顺序替换为QNUIU；
初版清单与修订历史保存在数据目录。预审不等于全视频逐帧场景审查。
训练尺度s=1.9888148307800293，仅训练集拟合，见[scale_v02.json](scale_v02.json)。

两条件均为17,821,672可训练参数，其中官方decoder 2,094,017；冻结DINOv2 304,372,736参数。
共同视觉448输入、32×32图像token，无CLS/register、无LoRA合并；既有518/LoRA缓存不兼容，重新提取共享冻结token。
MoGe沿用真实稠密256输出回映至原生分辨率，不重新运行，不以格均值替代点。
128/256/512、双尺度各三层、全部有效点/全部占据体素/81邻居上限保留。
BF16、target8+left16、空间tile256/halo1、attention query512、point32768、ffn_chunk8192；标准SDPA与显式注意力的输出及梯度等价。
仅复用当前窗口无参数路由，所有参数相关特征重算并完整反向。
旧版14项测试及720×1280资源证据仅是历史准备记录，不能用于新版本放行。
以下为上次node_chunks_v2失败执行的验收要求，并非本轮待执行任务。新版本必要测试、缓存复用和协议检查见FFN修复文档。原定重做XYZ、UV完整pair：每次有效更新仍先AF完整视频再ANAKIN完整视频；
随后仅对原高分辨率、失败样本和全48缓存fine/coarse最大节点窗口去重验收。全部通过才进入正式epoch0。

原计划后台流程如下（本次已退出，本轮不续跑）：复用缓存核验 → full-pair checks（XYZ先） → 全缓存坐标计数 → 代表窗口资源验收 → UV_PVST → XYZ_PVST → epoch0/best/last完整训练及开发评价 → 中文报告 → 普通Git发布。
每条件seed20260909，最多30轮/每轮16更新，patience8；仅AF开发视频AUROC选best（包括epoch0，平分取早）。
不增加条件、不用测试集；仅train/dev坐标用于资源计数，不用标签或分数选代表。检查失败保留错误并停止，不自动削减规格。

以下保留原正式训练命令用于历史复现，本轮不得启动；本轮限定入口见紧凑执行说明：

```bash
export PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages:.
export TORCH_HOME=/media/njupt/8FCD3869E2FAB06F/chenyh/cache/torch
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.pvst.train --config docs/pvst/config_v02.json
```

历史独立tmux会话（已退出）：`pvst_point_voxel_v02_nodes`，用 `tmux attach -t pvst_point_voxel_v02_nodes` 查看实时日志。
不要重复启动；恢复仅在核查实际退出原因与安全checkpoint后使用显式--resume。

绝对输出目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_point_voxel_v02/`

- `execution_revision.json`：本次`node_chunks_v2`登记；旧协议/状态/日志/退出码1归档在`execution_history/initial_8477da4/`，根exit_history继续追加。
- `protocol.json`：新代码/config/manifest/scale/commit身份，严格匹配，未关闭检查。
- `launch.json`：当前执行实际启动时间、源码commit、命令和PID入口。
- `run.log`：准备、每80目标帧及评价进度；`progress.json`：当前阶段，不要求Codex持续轮询。
- `full_pair_check.json`：新协议下两条件完整视频检查（另有逐条件摘要与逐块JSONL）；`resource_windows.json`：全缓存坐标计数及代表；`resource_*_node_chunks_v2.json`：新资源门槛；`UV_PVST/`、`XYZ_PVST/`：各自checkpoint/history/指标与固定预览。
- `report_zh.md`、`result_summary.json`：仅完成后生成的正式结论。
- `exit_code.txt`、`exit_history.jsonl`保留Python真实退出历史；tmux包装的`process_exit_code.txt`还覆盖系统终止情形。
- `publication.json`：发布状态与实验状态独立；Git网络失败不把已完成实验改成训练失败，也不触发重训。

大媒体、缓存、权重和逐样例图片全部留数据目录；Git仅源码、固定身份、小型证据和最终文档。
正式成本和分数尚未产生，不能用资源探针loss或历史V8指标代填。

2026-11-04单窗口独立资源检查：TIMEOUT；正式模型仍0/2，旧预算结论不变，见[900秒报告](max_window_check_v1.md)。

2026-11-04稀疏attention限定检查：小规模数值和中等规模反向改善，但N65,536候选SIGFPE/退出-8，验收FAILED，未进入完整窗口，正式0/2；见[结果](sparse_attention_check_v1.md)。

2026-11-04 SIGFPE定位：取得gdb原生栈及单算子复现，逻辑空列padding规避PyTorch2.5.1交集kernel整数零除；65,536两宽度及131,072/512单层前反向通过（FIX_VALIDATED），本轮无完整窗口/训练，正式0/2；见[报告](sparse_sigfpe_v1.md)。

2026-11-04修复后稀疏单窗口独立资源检查：OOM；正式模型仍0/2，旧预算结论不变，见[900秒报告](sparse_max_window_check_v1.md)。

2026-11-05 CSR行softmax检查：REPRESENTATION_BLOCKED；直接CSR构造反向在CUDA稠密化，未运行大单层/完整窗口；正式模型0/2。见[报告](csr_softmax_check_v1.md)。

2026-11-05 CSR values桥接限定验收：OOM；正式模型仍0/2。见[报告](csr_values_bridge_check_v1.md)。

逐head输出CPU暂存限定验收：OOM；正式模型仍0/2。见[报告](head_output_offload_check_v1.md)。
