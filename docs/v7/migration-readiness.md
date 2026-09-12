# V7 migration readiness

审计日期：2026-09-12
仓库：`/root/autodl-tmp/projects/sparse_3d_forgery_detection`
分支：`v7-dynamic-structure`
审计时 HEAD：`4823b2976f41a11286964506652113353af5d2be`
远端：`origin` → `git@github.com:2332140421-cmd/sparse_3d_forgery_detection.git`

本文件只记录迁移事实，不改变 V7 方法、数据划分、模型或实验结论。审计没有访问旧 R7/V5 仓库，也没有运行实验。

## 1. 代码完整性

审计时 `git status --short` 为空，`HEAD...origin/v7-dynamic-structure` 为 `0 0`，仓库共有 237 个已跟踪文件且没有未跟踪文件。当前 V7 的代码、测试和文档已经在本仓库；本轮没有发现需要从仓库外补入的自编 Python 或 shell 代码。

已跟踪的可恢复入口包括：

- `src/sparse3d_forgery/`：视频输入、ParticleSequence、VGGT 适配、几何/对齐和早期实验入口；
- `research_tools/v7/`：数据准备、观测密度、局部结构/时序、B0、boundary/pooling、normality、relation-first、NSI、YOLO26 对照、数据集 feasibility 及其分析代码；
- `research_tools/v7/boundary_pooling_probe/run_all.sh`：固定 `boundary_pooling_probe.pipeline --phase all --resume` 入口；
- `research_tools/v7/dense_local_structure_pilot/run_all.sh`：固定 dense pilot 的 `run_pipeline all --resume --expand-full` 入口；
- `scripts/`：显式几何前端、粒子提取、V7 core review、覆盖/前缀审计等脚本；
- `tests/research_tools/v7/`：对应研究工具测试；
- `docs/v7/design_contract.md` 及 `docs/experiments/v7-*.md`：V7 契约、协议和结果边界；
- `pyproject.toml`：项目环境的声明依赖；没有声明 YOLO26 依赖。

两个 shell 入口和所有被它们导入的 Python 模块均在 Git 中。代码仍包含当前服务器绝对路径（主要是 `/root/autodl-tmp/data/sparse_3d_forgery_detection` 和本地 `.venv`）；这不是遗漏代码，但在新服务器上必须恢复相同目录或按文件内参数调整。没有使用 `git add .` 或 `git add -A`。

### 仓库外代码边界

运行时通过 `sys.path.insert` 接入的只有第三方 TAPNet 源码，另有 Apple Depth Pro 源码目录；它们不是本项目自编代码，不复制进仓库。`tapnet-partial-gnutls` 是一个工作树不完整的第三方临时副本，当前入口不引用它；迁移时应使用完整的固定 commit 副本。没有发现仓库外的 V7 自编入口、补丁或任务调度器。

## 2. 当前服务器上的实验资产

以下路径来自当前代码、协议和已存在产物的实际引用。`ASSET_NOT_BACKED_UP` 表示当前服务器上存在，但审计没有第二份独立备份证据；它不等于不存在。Git 代码则用 `CODE_PUSHED`。未对大型视频、NPZ 或权重重新计算全量 SHA。

| 类别 | 当前路径 | 用途/恢复依赖 | Git 跟踪 | 可重建性与迁移方式 | 状态 |
|---|---|---|---|---|---|
| ActivityForensics/Charades 源视频与轻量元数据 | `/root/autodl-tmp/data/sparse_3d_forgery_detection/datasets/v7_core_candidates/activityforensics_charades_v1/source/`、`.../paired/`、`.../acquisition/` | 16 个冻结 source、real/fake、MANIP/CTRL 窗口；当前核心 V7 输入 | 否 | 可按 acquisition manifests 从固定 revision 重新取得；推荐保留源文件、manifest 和校验记录后用受控 `rsync` 迁移 | `ASSET_NOT_BACKED_UP` |
| DeeptraceReward 原始/提取媒体 | `/root/autodl-tmp/data/sparse_3d_forgery_detection/raw/deeptrace_reward/92e76e78e8c90a1ff7ec9354bee44eb024265e79/`、`extracted/deeptrace_reward/.../` | 早期显式几何/ParticleSequence 入口的输入；不是当前 ActivityForensics 核心 pilot 的替代数据 | 否 | 可由固定 revision 和下载记录重建；不要将 ZIP 或视频提交 Git | `ASSET_NOT_BACKED_UP` |
| 冻结 source/window manifests | `derived/v7_activityforensics_paired_second_order_pilot_v1/manifests/`、`derived/v7_activityforensics_density_matched_frontend_v1/window_manifest.json` | 16 source、192 windows、帧索引、PTS/时间窗口、role/kind/label | 否 | 可由现有 metadata 重新生成，但为保持实验身份应直接迁移并核对报告中记录的 SHA | `ASSET_NOT_BACKED_UP` |
| 289 点 ParticleSequence 与共享几何 | `derived/v7_activityforensics_density_matched_frontend_v1/particles/`（384 个窗口的 NPZ/JSON 与 `frontend_results.json`） | 后续 H/B 分组、关系状态和模型输入；依赖 Depth Pro/BootsTAPIR/位姿约定 | 否 | 理论上可重建，代价高且受 provider 版本、权重和路径影响；优先迁移原始 NPZ/JSON | `ASSET_NOT_BACKED_UP` |
| 64 点/观测密度基线 | `derived/v7_activityforensics_observation_density_diagnostic_v1/particles/`、`review/` | 嵌套密度对照、覆盖诊断和 review | 否 | 可从已保存 289 点/前端重新生成部分结果；为复现实验应迁移现有产物 | `ASSET_NOT_BACKED_UP` |
| H/B 分组、triplet 支撑和特征 | `derived/v7_activityforensics_local_organization_pilot_v1/manifests/`、`evaluation/`、`models/`、`scores/`；`derived/v7_activityforensics_boundary_pooling_pilot_v1/features/`、`scores/` | H 父组、B 子组、triplet/common support、A/C/D 输入和结果 | 否 | 可由 289 点与固定 manifests 重建，但模型/分组产物应作为研究资产迁移 | `ASSET_NOT_BACKED_UP` |
| 首帧 segmentation/mask 缓存 | `derived/v7_activityforensics_boundary_pooling_pilot_v1/masks/` 与对应 `manifests/segmentation.json` | B 边界划分；只作首帧边界参考，不是后续轨迹 mask | 否 | 可使用同一 `yolo26m-seg.pt` 重建；迁移缓存可避免 GPU 重跑 | `ASSET_NOT_BACKED_UP` |
| 模型、OOF 和指标 | `derived/v7_activityforensics_local_organization_pilot_v1/models/fold_models.json`、`scores/oof_window_scores.csv`、`evaluation/`；boundary `models/`、`evaluation/` | 已完成 source-disjoint pilot 的模型参数、OOF、per-source/summary | 否 | 可在输入和环境一致时重建；优先迁移小型 JSON/CSV，禁止提交模型大文件 | `ASSET_NOT_BACKED_UP` |
| Review 页面和媒体链接 | `derived/v7_activityforensics_boundary_pooling_pilot_v1/review/`、`derived/v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/review/` | `index.html`、details、screenshots、媒体链接，用于人工复核 | 否 | HTML/JSON/PNG 可迁移；媒体链接必须连同其真实目标迁移 | `ASSET_NOT_BACKED_UP` |
| Review symlink 目标 | 例：`boundary_pooling.../review/media/*.mp4` → `datasets/.../source/.../*.mp4` | 页面实际播放的原视频/短片 | 否 | 当前已核实 boundary review 有 192 个 symlink、0 个断链；symlink 本身不是备份，迁移时应恢复目标文件或重建链接 | `ASSET_NOT_BACKED_UP` |
| 外部 tracking/depth/segmentation/3D 权重 | `data/.../external/v7_explicit_geometry/checkpoints/`、`external/yolo26_depth/`、`external/vggt-weights/` | BootsTAPIR、Depth Pro、YOLO26 depth/seg、VGGT 等前端或对照 | 否 | 单独同步并按已记录版本/SHA 核验；不要进入普通 Git | `ASSET_NOT_BACKED_UP` |
| 第三方源码 | `external/v7_explicit_geometry/tapnet-c2cbab81.../`、`ml-depth-pro-9efe5c1.../`、`external/vggt/.../` | 运行时导入的 TAPNet/Depth Pro/VGGT 源码 | 否 | 按官方来源和固定 commit/archive 单独取得；不把第三方树复制为本项目代码 | `ASSET_NOT_BACKED_UP` |
| Python 环境 | 项目 `.venv/`；`/root/autodl-tmp/envs/v7-explicit-geometry/` | 项目 CPU/研究工具和显式几何前端运行时 | 否（`.venv/` 被忽略） | 不能复制环境目录；按版本和 `pyproject.toml` 重建，再单独安装 provider 依赖 | `UNKNOWN` |

当前可见的外部权重身份（来自现有代码/协议记录，未在本轮重算大文件 SHA）：

- BootsTAPIR：`causal_bootstapir_checkpoint.pt`，已有 SHA `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`；TAPNet source commit `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`。
- Apple Depth Pro：source commit `9efe5c1def37a26c5367a71df664b18e1306c708`，已有 checkpoint SHA `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`。
- YOLO26：`yolo26m-depth.pt` SHA `c0ff310571f0f159436e95a2a4fa3f8c40e1fbf36fac66a211347bd189990165`；`yolo26m-seg.pt` SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；运行协议记录 Ultralytics `8.4.146`。
- VGGT：source commit `a288dd0f14786c93483e45524328726ab7b1b4ce`；`facebook/VGGT-1B` revision `860abec7937da0a4c03c41d3c269c366e82abdf9`；仓库适配器记录的 `model.pt` SHA `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`。

这些 SHA 是身份记录，不是本轮对权重正确性的重新验收。`depth_pro.partial-222mb`、VRIPT/GenVidBench range probe 和 `.part` 文件属于不完整下载或缓存，不应作为可恢复正式资产。

## 3. 环境事实与缺口

- 项目 `.venv`：Python 3.12.3，torch `2.3.1+cu121`，CUDA runtime `12.1`，NumPy `1.26.1`，PyAV `18.1.0`，OpenCV `4.11.0.86`，Open3D `0.19.0`，scikit-learn `1.5.2`，Ultralytics `8.4.146`；`torch.cuda.is_available()` 当前为 true。
- 机器 GPU：RTX 4090 D，约 24 GiB，driver `595.71.05`。本轮只读取信息，没有运行 GPU 任务。
- 显式几何环境：`/root/autodl-tmp/envs/v7-explicit-geometry`，Python 3.12.3、torch `2.12.1+cu130`、CUDA `13.0`、Open3D `0.19.0`、PyAV `18.1.0`、OpenCV `4.11.0.86`、Ultralytics `8.4.146`；现有报告说明该环境使用 `--system-site-packages`，因此不是完全 hermetic。
- 当前没有单独声明的 YOLO 专用 lockfile 或独立 YOLO 环境记录；YOLO 包在项目 `.venv` 和显式几何环境均可见，但 `pyproject.toml` 没有把它列为项目依赖。这是迁移环境记录缺口，不能用“当前 import 成功”替代新服务器复核。
- `pyproject.toml` 声明 torch/torchvision、PyAV、NumPy、OpenCV、scikit-learn 及固定 VGGT git revision；它没有声明 Open3D、Ultralytics、TAPNet 或 Depth Pro。后者的源码、权重和运行时依赖仍需外部恢复。

## 4. 新服务器恢复顺序

1. Clone 新 GitHub 仓库，checkout `v7-dynamic-structure` 的明确提交（本次审计为 `4823b2976f41a11286964506652113353af5d2be`）。
2. 依据 `pyproject.toml` 重建项目环境；单独记录 torch/CUDA、PyAV、Open3D、Ultralytics 版本，不复制 `.venv`。
3. 使用独立传输通道同步 ActivityForensics/Charades 源视频、标注、固定 manifests、289 点/64 点缓存、H/B/segmentation、模型/OOF/metrics、review 资产及外部源码/权重。传输后按各 artifact 内已记录 SHA/identity 核对；视频和大权重不要重新加入 Git。
4. 恢复代码使用的目录布局，或在不改变协议的前提下显式传入等价路径；恢复 review symlink 的真实目标，不能只复制链接文件。
5. 做轻量恢复检查：导入核心模块；读取一个已有 ParticleSequence；读取一个 feature/model 记录；确认一个 review 媒体链接的真实目标可访问。无需训练即可判断基本链路是否连通。

未执行跨服务器恢复，因此不能声称完整实验已迁移成功。可在用户确认目标服务器后使用类似模板（先 dry-run，再执行）：

```bash
rsync -aH --info=progress2 \
  /root/autodl-tmp/data/sparse_3d_forgery_detection/ \
  <new-server>:/root/autodl-tmp/data/sparse_3d_forgery_detection/
```

应按资产优先级拆分传输并保留日志；不要把该模板当作已经执行的备份。

## 5. 结论

1. **代码可恢复：是。** V7 当前入口、分析实现、测试、配置和实验文档均已跟踪并推送到 `origin/v7-dynamic-structure`；本轮没有发现适合纳入 Git 的遗漏自编代码。
2. **完整实验可恢复：未确认。** 当前服务器的数据、NPZ、模型、媒体和权重均存在不同程度的产物，但没有第二份备份证据，统一按 `ASSET_NOT_BACKED_UP` 或 `UNKNOWN` 处理。
3. **必须单独同步：** ActivityForensics/Charades 与所需标注、冻结 manifests、289/64 ParticleSequence、H/B/segmentation/triplet/features、模型/OOF/metrics、review 真实媒体目标、TAPNet/Depth Pro/YOLO26/VGGT 源码与权重、以及可复现环境规格。
4. **已知缺口：** 绝对路径、非 hermetic provider 环境、部分 `.part`/partial cache、第三方临时 `tapnet-partial-gnutls` 工作树；这些不影响当前代码已在 Git，但会影响新服务器的直接复现。完整实验恢复前必须完成独立备份和轻量链路验证。
