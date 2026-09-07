# V7 GenVidBench Core-domain suitability pilot

Status: `GENVIDBENCH_DATA_ACCESS_BLOCKED`

## 1. Research question

本轮只回答：GenVidBench 的官方 metadata 是否能恢复与 V7
`Single Dominant Persistent Structured Motion` 研究域相匹配的 real/fake
source 关系，并为一个很小的真实视频 pilot 建立可审核材料。它不是
frontend、component、`S_t`、`ΔS_t`、`Δ²S_t` 或 detector 实验。

## 2. Motivation and domain boundary

既有 DeeptraceReward 视频级审核显示其样本中经常出现
`MULTI_SUBJECT_COMPLEX`、`ARTICULATED_COMPLEX`、`STATIC_VIDEO`、
`STRONG_OCCLUSION`、`FLUID_SMOKE_FIRE`、`TOPOLOGY_CHANGE`、
`CAMERA_MOTION_DOMINANT` 和 `SUBJECT_TOO_SMALL`。这表示它与当前
V7 Core Domain 的持续结构运动条件不总是匹配，而不是“数据集失败”；
DeeptraceReward 保留为未来 complex-domain external stress test。

Core candidate 只优先保留实际 metadata 中 `Object_dict == Vehicles` 且
`Action_dict != Static Postures` 的记录。该规则是 prefilter，不是自动
`IN_DOMAIN` 判定；其余语义必须人工审核。Core membership 只读官方
semantic metadata 和真实视频 RGB，不读 fake 内容、模型输出、XYZ、
tracking、depth、pose 或任何异常分数。

## 3. Official source and structure

- 官方代码仓库：`genvidbench/GenVidBench`；README 可见 main 提交为
  `de027df`（服务器上的 GitHub clone 受 GnuTLS 握手故障阻断，因此这里
  保留 web-visible revision，不伪造完整 SHA）。
- 官方数据入口：README 指向 HuggingFace `jian-0/GenVidBench`，
  API revision `701cafb6f999d7ea0cbf3c354df6177311a4d824`。
- 官方 README 声明总发布规模约 6.78M、11 个生成器、
  `CC BY-NC 4.0`；Vript、HD-VG-130M 及生成器上游许可细节尚未逐一
  确认，故记录 `LICENSE_DETAIL_UNRESOLVED`。
- 实际取得的 metadata 文件在
  `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_genvidbench_core_pilot_v1/metadata/`：
  `Pair1_labels.txt`、`Pair2_labels.txt`、`Pair1_verify.txt`、
  `HD_VG_130M_verify.txt`、`Vript_20k_classes.txt`、
  `HDVG_14k_classes.txt`、`VidProM_13k_classes.txt`、
  `classes_list.txt`。这些文件的 SHA-256 已写入
  `metadata/source_audit.json`。
- 标签格式实际为 `relative_path label`。Vript real 使用文件 stem 作为
  source identity；HD-VG-130M real 使用 `ordinal` 与 filename 中的
  source identity；Pair2 fake 文件包含相同 ordinal 和语义 caption。
  因而 Pair2 配对键为“ordinal + 共享 HDVG semantic record”，不是对
  文件名文字的猜配。
- 语义文件实际格式为
  `ordinal|||source_identity|||object_action_location|||caption`。
  `classes_list.txt` 的实际 object/action/location taxonomy 被解析并
  记录，没有新增自定义 taxonomy。

## 4. Metadata population

按官方标签文件解析得到：

| population | total | real | fake |
| --- | ---: | ---: | ---: |
| Pair1 | 74,135 | 20,131 | 54,004 |
| Pair2 | 67,860 | 13,416 | 54,444 |

Pair1 real source 为 Vript，Pair2 real source 为 HD-VG-130M。目标 paired
fake generators 为 MuseV、SVD、Mora、CogVideo。语义 metadata 行数为
Vript 20,131、HD-VG-130M 13,853、VidProM 13,501。

Metadata prefilter candidate 数：Vript 1,911；HD-VG-130M semantic rows
1,606，和实际 Pair2 real label 交集后 1,559。Train/test source IDs
没有重叠（Vript 与 HD-VG-130M source-level 分离）。

## 5. Deterministic pilot and lineage

按 `source_id` 升序、在 prefilter 后选择：

- train real pilot：32；
- test real source pilot：16；
- paired fake：64；
- 每个目标 generator：CogVideo 16、Mora 16、MuseV 16、SVD 16。

`manifests/train_real_pilot.json` 只含 real；
`manifests/test_real_source_pilot.json` 只含 real source；
`paired_test/paired_test_manifest.json` 每一条 fake 都保留 real source、
source identity/caption、generator、fake path 和 pair lineage。fake 从未
参与 Core candidate 选择。

## 6. Pilot review material

输出根目录：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_genvidbench_core_pilot_v1/`

已生成：

- `real_review/review_core.csv`：48 行 real，字段遵循现有 V7 审核契约；
- `metadata/source_audit.json`、`metadata/core_candidate_rules.json`；
- 三个 pilot/paired manifests；
- `summary/pilot_summary.json` 和 `summary/materialization_summary.json`。

CSV 中的 3 个 train 示例：

1. `Pair1/vript/-C_-HNTztXI-Scene-005.mp4`
2. `Pair1/vript/-Z3priQFMeE-Scene-021.mp4`
3. `Pair1/vript/-W954LYAU1k-Scene-006.mp4`

3 个 test source 示例：

1. `Pair2/hd_vg_130m/00015___3YhUv4cBoiA.100_1.mp4`
2. `Pair2/hd_vg_130m/00018___2qTZcCY8XHQ.15_1.mp4`
3. `Pair2/hd_vg_130m/00033___4Ndd6ph8Hlw.2_2.mp4`

官方 HuggingFace tree 当前提供的是大体积 archive parts，而不是这些
individual MP4；本轮没有下载或解压全量 archive。materializer 对 48 条
逻辑路径记录 `VIDEO_NOT_FOUND`，decode success=0、failure=48、
contact-sheet count=0。因此当前不能声称已经完成 RGB 人工审核，
也没有生成空的假 contact sheet。

## 7. Finalizer and leakage control

`research_tools/v7/datasets/genvidbench/finalize_core_pilot.py` 只接受
`video_decision == IN_DOMAIN`，保持 train real-only，并仅把 lineage 到
已通过的 test real source 的 fake 放入 test；`UNCERTAIN` 默认排除，
review CSV SHA-256 写入最终 manifest。由于 48 行当前仍为空决策，本轮未
执行 finalizer。

该工具和 parser 全部位于 `research_tools/v7/datasets/genvidbench/`；
没有新增或修改 `src/sparse3d_forgery/`，正式代码不依赖 research_tools。
未检查或使用 fake 结果来调整 Core selection，也未引入 model-output 字段。

## 8. Current conclusion and next step

Metadata 可用，实际 split/source/generator 结构和 Pair2 lineage 可恢复，
并已生成小规模 deterministic manifests；但官方媒体仍以大归档形式存在，
48 个真实视频尚未 materialize，故当前状态是
`GENVIDBENCH_DATA_ACCESS_BLOCKED`，而不是 Core Domain 已确认。下一步只需
在不改变 manifests 的前提下获得这 48 个 real MP4，运行现有 materializer，
由人工填写 `review_core.csv`，再执行 finalizer；只有明确 `IN_DOMAIN` 的
real source 才能进入后续 Explicit 3D、persistent particles、component、
`S_t/ΔS_t/Δ²S_t` 与 real-only normality 实验。

本轮没有运行 frontend、component、结构状态、normality、AUROC、训练或
fake detector，没有下载完整 GenVidBench，也没有删除 DeeptraceReward。
