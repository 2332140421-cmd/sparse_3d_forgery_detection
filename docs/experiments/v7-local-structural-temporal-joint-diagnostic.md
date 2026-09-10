# V7 局部结构时序联合诊断

状态：**JOINT_DIAGNOSTIC_COMPLETE（开发诊断，不是正式检测结果）**

数值诊断 artifact 的 `protocol.json` 记录生成时 `v7-dynamic-structure` HEAD 为
`2977e46adb95ead1e32307156835eb3fbf8170c7`；本轮在当前分支只扩展人工审查入口和标注清单，
仍只读取此前的 weightfix pilot、粒子序列和原始视频。没有调用 `train_model`、没有
`optimizer.step`、没有重跑 frontend/tracking/depth/pose，也没有运行 GPU。180 条
fold/seed 模型是重复拟合，不是 180 个独立样本。

## 1. 冻结输入与可复核产物

输入仍是此前冻结的 16 个 source pairs、192 个 real/fake MANIP/CTRL 窗口，以及原有
`S(t)=[mean,std,p25,p75]`、有符号真实时间差分和四个 arm：
`UNORDERED_STATE`、`ORDERED_FIRST`、`ORDERED_SECOND`、`PERMUTED_SECOND`。训练重建只
使用非 held-out source 的 real/fake MANIP 窗口及原 source/class 权重；CTRL 不进训练。
不改变有效 support，不增加 pair，不把缺失填零。

完整产物在数据盘（不进入 Git）：

```text
/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/
  v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/
```

主要文件：

```text
protocol.json
evaluation/oof_reproduction_summary.json
evaluation/model_reproduction.csv
evaluation/loss_curves.json
evaluation/relation_reconstruction_summary.json
evaluation/contribution_decomposition_summary.json
evaluation/contribution_responses.json
coverage/coverage_windows.json
coverage/structure_records.json
coverage/native_pair_curves.npz
review/index.html
review/data.json
review/screenshot_index.json
review/media/
review/screenshots/
```

`protocol.json` 记录输入文件 hash、窗口/arm/seed 总体、LOSO 重建规则、缺失处理、
聚合和显示规则。`native_pair_curves.npz` 只含数值数组，可用
`np.load(..., allow_pickle=False)` 读取，没有 object array。

启动审查页：

```bash
cd /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/review
python3 -m http.server 8765 --bind 127.0.0.1
```

在用户机器上通过 SSH 将 `127.0.0.1:8765` 转发到服务器后，用浏览器打开
`http://127.0.0.1:8765/`。页面是无 CDN 的单文件 HTML，支持 source、MANIP/CTRL、
real/fake 并排、播放/暂停/逐帧/定位首个目标、component/triplet/pair、覆盖/结构/模型
图层开关，以及开发标记的 JSON/CSV 导出和导入。播放器 seek 只作近似显示；源帧号、
PTS 和模型实际采样表是权威时间事实。

此前的固定案例物化了 8 个短片；本轮按每个 source 固定顺序补齐了最早的 MANIP 和
CTRL（16 个 source × 2 类 × real/fake 的默认覆盖为 64 个窗口），并保留四类额外
人工审查入口，去重后共 68 个可用窗口（real/fake 各 34 个）、68 条截图索引。当前
review 目录保留了此前案例别名，物理文件为 73 个 MP4、371 个 PNG 和 73 个 SVG；
这些别名不增加索引窗口。其余 124 个窗口仍在索引中并标记 `NOT_MATERIALIZED`；源文件
存在但尚未制作，不能写成源文件缺失。若源文件不存在则使用 `SOURCE_MISSING`，解码
异常使用 `DECODE_FAILED`。每个可用窗口还列出模型实际采样 frame index、array index
和 PTS 的精确截图。环境没有可用浏览器，因此完成了静态路径、媒体解码、脚本、无外部
CDN 和导入/导出逻辑检查，没有声称已完成浏览器视觉验收。截图仅作为人工核对材料，
不是数值输入。

## 2. 证据矩阵

| 问题 | 已有证据 | 本轮结果 | 可以推断什么 | 不能推断什么 |
| --- | --- | --- | --- | --- |
| checkpoint/评分是否复现 | 180 条保存模型、OOF 和各自 standardizer | 1,992 个 held-out seed logits；最大/均值/中位数/p99 绝对误差均为 `0.0`，0 个容差失败；标准化重算最大误差 `0.0` | 恢复架构、权重、标准化和 LOSO 前向路径与保存 OOF 一致 | 不证明模型学到可泛化伪造规律 |
| 训练拟合是否成立 | 每个 fold 的训练 real/fake MANIP、实际 source/class 权重、loss history | 训练 source 等权 AUROC 均值：A `0.691806`、B `0.681454`、C `0.700977`、D `0.661186`；训练加权 BCE：A `0.607231`、B `0.598488`、C `0.581322`、D `0.621708` | 模型在开发训练窗口上有有限拟合信号 | 有限 loss 或训练 AUROC 不等于收敛、机制正确或泛化 |
| 训练与 held-out 是否有落差 | 同一 checkpoint 的训练/held-out 重评分 | held-out MANIP AUROC 均值：A `0.542328`、B `0.470899`、C `0.510582`、D `0.502646`；对应未校准-logit BCE：A `0.727268`、B `0.775419`、C `0.738285`、D `0.750299` | 训练表现普遍高于 held-out，存在 source 泛化落差 | 不能把落差单独归因于数据量、表示或优化某一项 |
| 目标时间是否实际取样 | `coverage_windows.json` 中 source frame、array index、PTS、target match | 867 个有效 triplet；实际模型帧和目标匹配逐条记录，未把整个 MANIP 区间伪装成已评分连续曲线 | 可逐 triplet 核对真实帧与目标时间 | 不能从时间采样表证明目标内容确实被模型看见 |
| 目标空间是否被观测 | 每帧 UV、visibility、geometry validity、component 成员和无效原因 | 192 窗口中 166 有效、26 无有效 triplet；无效原因 `COMMON_VALID_MEMBERS_LT3=87`、`NO_HISTORY_COMPONENT=23`、`TARGET_FRAME_MISSING=18`（按窗口/原因记录）；共 867 triplet、324 个历史 component | 知道哪些稀疏点和关系真正进入计算 | “窗口有点”不等于“失真区域被覆盖”；需要页面上的人工画面/点覆盖核对 |
| raw relation 与 `S(t)` 是否对应 | 原 pair、历史尺度、同三时刻、saved state 和原生曲线 | 867 triplet 的状态重建最大/均值/中位数/p99 误差均 `0.0`；原生背景曲线 186,780 条；四 arm 输入重建最大误差 `0.0`（A 2,601 行、B 867 行、C 867 行、D 5,202 行） | 当前展示使用了与 pilot 相同的 pair/support、尺度和输入 | 不证明三维变化是真实物理失真，而非重建/provider 误差 |
| 局部响应与平均聚合 | 保存模型 encoder/head 和同一 batch 的逐 triplet 分解 | 1,992 次分解；窗口 logit 代数重建最大误差 `1.430511474609375e-06`，p99 `3.3097962503614235e-07` | q、component/triplet 权重和窗口平均关系可复核 | q 不是局部概率、贡献不是像素 GT；不能据此改成 top-k/max 评分 |
| source/seed 稳定性 | 15 个有模型记录的 held-out source、每臂 3 seed | source-level held-out AUROC 方向不一致；例如 A 在 `0FO58=1.000`，`04LAX=0.519`，`0ET8W=0.148`；`0CG15` 无有效 MANIP held-out AUROC。三 seed 不是独立样本 | 结果明显依赖 source，且应保留 seed 波动 | 不能宣称跨生成器泛化；不能选择“最好 source/seed”重报结果 |

`0HV07` 没有模型记录（无有效支撑）；`0CG15` 的模型记录存在但没有可计算的 held-out
MANIP 双类别 AUROC。两者都保留在覆盖/索引中，不填正常分数。

## 3. 拟合、覆盖、表示与聚合的联合解释

保存 checkpoint 的训练 AUROC 高于 held-out，说明“完全没有拟合”不是最直接解释，但
held-out 仍接近 chance，且 source 间方向不一致，支持“有限 source 泛化/样本规模与
观测异质性共同限制”的谨慎描述。它不能单独证明只要下载更多同源窗口就会解决问题。

26 个无有效支撑窗口和每帧/每个 triplet 的实际共同成员数量说明覆盖是现实限制；不过
没有人工标记，不能判断这些无效区域是否正好包含可见伪造。页面默认的四个案例是人工审查
入口：先隐藏 labels/scores 观察原视频，再逐帧核对 UV 覆盖，最后打开结构和模型层。任何
标记只是开发诊断，不进入训练或 population 筛选。

原始 pair 曲线和 `S(t)` 的精确重建说明当前代码没有在诊断中把关系错接或把缺失变成零。
它并不赋予 `S(t)` 一个“信息保留率”标量，也不说明 raw 关系的波动必然是伪造。贡献分解
证明的是现有 component/window mean 的代数稀释方式；它不是新评分规则，也没有据此重算
AUROC。

## 4. 下载与下一步

本轮没有下载任何数据、模型或视频，也没有重跑前端。当前不需要因为这次负面/接近 chance
结果就下载完整数据集。优先动作是用审查页核对四个固定案例，记录：失真是否可见、目标区
是否有有效 UV/geometry 点、raw relation 是否响应、`S(t)` 是否响应、模型是否响应。若
人工核对显示观测覆盖可靠而训练好/held-out 仍有落差，再以独立 source 扩充训练 population；
不要只增加同一 source 的窗口。若目标区域经常没有有效点，应先修正采样/测量覆盖，而不是
只扩大分类网络；若 raw relation 有变化而 `S(t)` 丢失对应变化，才有证据讨论 relation-level
表示。当前没有足够证据自动启动新模型、multi-order、完整数据下载或正式 detector 训练。

## 5. 本轮边界

- 未访问旧 R7/V5，未读取其他分支补全方案。
- 未修改正式 `src/` 检测链，未重跑 frontend、tracking、depth、pose 或 GPU。
- 未训练、未 fine-tune、未进行超参数搜索、未运行 full-video 或 sealed-test。
- 未把 CTRL 变成确定负类，未把 180 个模型当独立样本。
- 未下载数据、模型或视频；未提交媒体、NPZ、模型副本或大型缓存到 Git。
- 本轮新增的只有离线诊断工具、其 4 项契约测试和本报告；产物保留在数据盘。

## 6. 标注层级与测量覆盖核查（本轮）

标注证据清单位于数据盘：

```text
/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/
  v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/annotation_inventory.csv
```

清单逐项记录了路径、SHA-256、来源版本、字段、时间约定、pipeline 使用情况和证据状态。
当前可确认的层级如下：

| 层级 | 已核实内容 | 当前 pipeline/评价用途 | 边界 |
| --- | --- | --- | --- |
| 视频 | `activityforensics_source_mapping.json` 与 selected-pair manifest 给出 source、real/fake role、generator、operation 及同源工程配对 | 用于 source-disjoint 分组、窗口索引和视频/窗口标签 | 同源配对不等于像素相同；generator/operation 不进入数值模型输入 |
| 时间 | `annot.zip` 的 `duration` 和 `start=end` 秒区间；文件名编码 operation；window manifest 保存 `MANIP_25/50/75` 与 `CTRL` 区间、源 frame、PTS | `MANIP` 窗口用于 real=0/fake=1 的窗口级监督/审查，`CTRL` 只作描述性对照 | segment 的帧边界闭开约定仍未从本地文件核实；CTRL 不是确定的“未修改假视频”负类 |
| 空间 | 当前 ActivityForensics 物化目录内未找到可映射的 bbox、mask、polygon 或编辑区域文件；现有 `review_manifest.json` 的人工空间字段为空 | 不用于训练或正式空间评价 | 这是“本地未找到”；官方轻量资料没有为本快照确认逐帧空间 GT，不能据此断言数据集全无空间标注 |

因此当前训练标签充分到“窗口/时间区间的 real/fake 分类”层级，不能下放为所有
component、所有点或所有像素的正标签。fake 时间区间也不等于每个局部都有可见内部形变；
编辑输入 mask（即使将来找到）不能自动当作输出失真边界。官方 ActivityForensics 项目
把 `annot` 作为单独的注释资源发布，当前本地 `annot.zip` 的实际字段仍是时间段；官方
网页对另一数据集的 bbox 说明不与本项目混用。详见官方
[ActivityForensics repository](https://github.com/ActivityForensics/activityforensics) 和
[DeeptraceReward project page](https://deeptracereward.github.io/) 的来源说明。

## 7. 人工审查页面交付

页面仍是单文件、无外部 CDN 的离线审查页：

```text
/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/
  v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/review/index.html
```

启动：

```bash
cd /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/review
python3 -m http.server 8765 --bind 127.0.0.1
```

用户首次操作（不超过五步）：

1. 在该目录执行上面的本地 HTTP 服务命令，并在浏览器打开 `http://127.0.0.1:8765/`。
2. 先关闭 IDs、labels 和 scores，按 source 默认顺序查看最早 MANIP 与 CTRL。
3. 选择 real 或 fake 一侧，使用鼠标框选或点选原始画面中的局部区域。
4. 核对源 frame index、PTS、模型实际采样截图、曲线和覆盖计数，再填写观察类型与时间区间。
5. 点击“保存标记”后分别导出 JSON/CSV；导出的文件由浏览器下载到用户选择的下载目录，
   不会写回原始 artifact。

本轮页面行为包括：

- real/fake 的 component、triplet、pair 选择完全独立；相同编号不被解释为物理对应；
  切换局部图层不会重新加载同一视频，因此不重置当前时间。
- 选定 pair 直接显示原始距离、固定历史尺度归一化距离；选定 triplet 显示
  `S(t)=[mean,std,p25,p75]`、时间戳感知有符号一阶/二阶差分和真实 PTS。颜色文字严格为
  “相对当前 triplet 首时刻的归一化距离变化”。
- 模型表显示窗口 logit、选定 component/triplet 的 q、聚合权重及加权贡献；q 不是
  校准概率，贡献不是像素热力图。选定 triplet 的实际成员点会在画面上突出，普通可见点
  仍保留为背景。
- 每个可用媒体列出模型实际使用帧的精确 PNG 链接；播放器 seek 仅便捷浏览，源 frame
  index、array index 和 PTS 表是核对依据。坐标均按原始视频 UV 显示。
- 鼠标拖框或点选后，记录 source/window/role、原始帧号、PTS、原图坐标、可见形变、
  类型和备注，并自动给出选定时刻/区间的最大覆盖计数：可见查询点、geometry-valid 点、
  triplet 共同成员、两端均在区域内 pair、仅一端在区域内 pair 及是否处于模型采样帧。
  这些计数不是检测准确率；无区域时不输出覆盖率。
- 标记可导出为 JSON/CSV，也可重新导入；数值 `0` 不会被转换成 `null`。导入的标记只
  用于当前开发解释，不改变 population、训练或正式指标。

标注字段模板与清单分别位于数据盘的 `review/review_manifest.json` 和
`annotation_inventory.csv`；浏览器导出的 `v7_annotations.json` / `v7_annotations.csv`
是人工记录的独立文件，默认保存在浏览器下载目录。当前默认覆盖的 64 个窗口全部成功
生成；加上四个额外人工案例后索引中有 68 个 `AVAILABLE`，另有 124 个
`NOT_MATERIALIZED`，本次没有 `SOURCE_MISSING` 或 `DECODE_FAILED`。

当前 64 个初始查询不等于每帧 64 个有效三维点；component 是稀疏观测集合而不是完整
像素分割，`geometry_validity` 不是测量精确。近刚性允许真实形变和测量误差，component
连边阈值不是伪造判定阈值，稀疏空间提示也不是像素级定位结果。

## 8. 当前结论与唯一优先动作

当前可支持：视频/窗口级 real-fake 分类诊断、基于已给时间段的时间定位，以及“哪些稀疏
观测/关系在当前窗口被实际测量”的空间提示。当前不能支持：人工真值意义上的像素或
物体级失真边界、稀疏点覆盖率作为检测准确率、或将模型局部 q 解释为校准概率。

页面已经足以开始人工判断，但浏览器端到端视觉检查仍需用户在本地浏览器打开审查页完成。
本服务器未发现可用 Chromium/Firefox 等浏览器，因此本轮状态明确为
`BROWSER_ACCEPTANCE_PENDING`；媒体逐片解码、PNG/SVG 路径、无 CDN 静态检查和 358 项
自动化测试不能替代浏览器中的播放、坐标和导入导出验收。
唯一优先动作是按页面默认顺序审查每个 source 的最早 MANIP/CTRL（先关闭标签和分数），
记录是否看见局部形变及其真实测量覆盖；不在本轮自动改变采样、模型或数据筛选。

实际测量契约单独保存于：

```text
/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/
  v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/measurement_contract.json
```

该文件由现有 `coverage_windows.json`、`structure_records.json` 和冻结表示实现生成，明确
区分了窗口解码/前端处理帧（每窗口 6–31 个记录，因无效窗口可提前缺失）与模型实际使用帧
（每窗口 0–5 个，匹配成功的 target slot）。它记录了 64 个起始查询点、无中途新增点、
历史边界 `timestamp < window_start + 0.5s`、目标偏移 `[0.5,0.6,0.7,0.8,0.9]s`、
`0.05s` 匹配容差、共同成员至少 3、全组合 pair、历史距离中位数尺度、`S(t)` 四维统计量、
时间戳感知一阶/二阶差分及 triplet→component→window 聚合。它不把前端处理帧自动写成
分类器输入。
