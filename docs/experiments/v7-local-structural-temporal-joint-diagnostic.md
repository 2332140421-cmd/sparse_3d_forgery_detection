# V7 局部结构时序联合诊断

状态：**JOINT_DIAGNOSTIC_COMPLETE（开发诊断，不是正式检测结果）**

本诊断固定在 `v7-dynamic-structure` 的 `2977e46adb95ead1e32307156835eb3fbf8170c7`，只读取
此前的 weightfix pilot、粒子序列和原始视频。没有调用 `train_model`、没有
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

本轮物化了 8 个短片、8 张截图和 8 个覆盖 SVG：有效 MANIP、CTRL、无支撑和局部缺失
四类各一组 real/fake；其余 184 个窗口仍在索引中并标记 `MEDIA_UNAVAILABLE`，没有
伪造媒体。PyAV 可逐个解码这 8 个短片，PTS 单调；PNG 和覆盖文件路径均存在。环境没有
可用浏览器，因此完成了静态路径、媒体解码、脚本、无外部 CDN 和导入/导出逻辑检查，
没有声称已完成浏览器视觉验收。截图仅作为人工核对材料，不是数值输入。

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
- 本轮新增的只有离线诊断工具、其 3 项契约测试和本报告；产物保留在数据盘。
