# PVST完整视频匹配定位 v1：用户启动交付

**脚本已交付，全量任务尚未启动。** 本轮完成独立入口、固定配置、CPU契约检查和一个已有完整小病例的训练—保存/恢复—评价接口预检。没有全队列观测准备、队列极值测量、正式训练或开发效果比较；没有后台启动或持续轮询。

## 研究问题与固定范围

相同冻结视觉底座、XYZ联合块编码、定位修正和监督覆盖下，完整视频跨时刻连接是否帮助独立原视频定位？FRAME_ONLY完整视频输入、逐帧关系；FULL_VIDEO完整视频原始块直接双向关系。两组都用XYZ、输出全部帧。只比较跨时刻连接，不独立证明三维优于二维。
直接复用`FullVideoDirect`数学：epoch01冻结DINO/LoRA/官方头同次取得patch1024、predictor输入detail32、工作z0；新1024→128投影、原几何调制/逐点MLP/学习池化、两层D128/4头/FF256、dropout0，R回填32与detail32融合，零初始化末层delta叠加z0。无中继/筛块/帧分类/平滑导数损失/额外门控。
旧原型、旧实验与官方源码不改；本轮独立范围由ADR-0056、根契约17.13、AGENTS及`pvst_fullvideo_matched_config_v1.json`登记。

## 数据与预算已冻结

权威来源：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_expanded_spatial_training_v1/protocol.json`。直接复制其数据清单/角色，不调用旧freeze，不重新选来源。原protocol中的旧epoch/更新预算是历史字段；本轮预算由独立config明确为3轮，不改写旧文件。

|项目|实际核对值|
|---|---:|
|训练来源/视频|89 / 178|
|训练原始观测帧/合法监督帧每轮|22,882 / 22,878|
|开发来源/视频/唯一帧|19 / 38 / 5,152|
|总观测帧|28,034|
|保护测试来源（不参与）|94|
|每模式轮数/更新|3 / 534|
|每模式合法监督帧呈现|68,634|
|每模式原始输入帧呈现（含4帧无标签上下文）|68,646|
|额外BN统计前向/新增DFormer调用|0 / 0|

每条real/fake独立为一个逻辑视频，不拼配对真假C、不跨视频/集合注意力。source_group_id隔离、PTS、原生/工作尺寸、GT与角色沿用；官方标注技术核验后使用，不再要求逐条人工审批，不把自动/官方类别冒充人工语义确认。
40y1x9_fake[6,10)仍完整输入/输出，但不进入监督与计分；真实GT按原协议零，fake缺失/错位GT明确失败，不补零。
统一scale=12.880994319915771、h=.5、origin0，xyz_provider只除scale一次，FP32三轴路由全部有效点/非空块，无共有键前提。
两模式相同新模块初态、seed20260909，各新AdamW(lr1e-4/wd.01/clip1)，BF16前向/FP32池化及loss。178条视频每轮用seed+epoch无放回打乱，同一orders.json供两模式。每视频合法帧BCE均值+soft Dice(eps1)，再按合法帧平均，整片一次backward/step。无调度器/增强/类别权重/开发阈值搜索。

## 入口与一次启动的阶段

- `research_tools/xyz_pilot/fullvideo_matched_data.py`：专供本实验的两种工作缓存读取、身份兼容、几何/冻结视觉补齐、恒定路由和空间/队列统计。旧原型准备器硬编码三条fake及旧manifest，不能承担新队列，故新增薄适配器。
- `research_tools/xyz_pilot/fullvideo_matched.py`：复用原模型与空间指标，组织资源门、整视频训练、恢复、原生评价和报告。不是通用训练平台。
- `/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh`：现有v8 Python，原v8-ainpaint-packages/rgbd-packages、XFORMERS_DISABLED=1，python -u+tee；不安装依赖或下载。

默认无参数按以下顺序自动完成，不再等待Codex/人工批准：

1. 核对冻结protocol、保存配置、三轮视频顺序、观察清单、输入/源码身份和空间估算。
2. 共享观测准备。按cache_kind/cache_root读取旧混合PT或新工作RGB/GT mmap，每次只保留当前视频；不让全库常驻RAM。
3. 几何优先复用自身视频旧PT或匹配视频SHA、PTS、provider权重/revision及高分辨率的旧几何；缺失才从原生RGB按原geometry_shape长边≤768及num_tokens规则调用MoGe v2，force_projection=True/apply_mask=False/use_fp16=True，沿原remap_geometry回映，不用384工作RGB冒充provider输入。MoGe全队列最多加载一次；不顺带提DFormer。
4. 全队列冻结视觉最多加载一次：严格恢复扩充epoch01/官方权重/revision，全部eval/no_grad，原空间模型同次获取patch/detail/z0，不改官方头或缓存dtype。几何与视觉分阶段，结束释放MoGe/DINO。
5. 复用三病例观测按视频/帧/PTS/尺寸/视觉权重/provider/scale/h身份核对；忽略旧整体manifest摘要差异，不能只凭文件名复用。恒定points/keys/index可缓存，C/R每次由训练参数重算。
6. 全队列生成T、监督帧、N_t范围、M和缓存字节；保存train/dev各最大M、最大T、最大单帧N的元数据。指定资源门只测试训练最大M整片前后向（两模式各2次丢弃更新）及开发最大M整片推理；兼容且同T/M的既有PASS资源证据可复用，不重测全部老病例。开发GT不作更新。
7. 默认Q=0，高效SDPA+非重入checkpoint；仅明确OOM允许唯一备用Q=1024，先做CPU输出/输入/参数梯度等价检查，每块Q仍读完整K/V归一化，然后两组统一设置写入compute.json。非OOM错误直接停；备用仍失败则停，不跳来源、删块、降帧或改h。M极值也不代表T与像素量的所有联合最坏情况，其他病例之后失败同样记录并退出。
8. 读取冻结epoch01在同样5152目标上的已有评价作为二维参照，逐帧身份核对，不重载GPU参照。顺序FRAME_ONLY三轮、释放模型/optimizer，再FULL_VIDEO三轮。
9. 每轮训练观察及全部开发评价、保存权重和报告。第3轮主比较；每模式新增15开发AP从1/2/3轮选best（同分早轮），只作开发选择附表，不称最终测试。

训练观察固定seed按已有官方编辑动作规则从ID排序清单选择：添加/复制`an:4dpnv4`（床上添加玩具熊）、移除`an:4qw154`（移除球场上5名奔跑者），各real/fake，共4条完整训练视频。不是按模型成绩/GT面积/可见异常选取，不等同完整训练集拟合，不用于选权重。

## 本轮真实短预检

执行提交：`155a41b6f49f89d40000e32937e510c8944fc3a9`。复用已验收`4ym5n4_fake`的96帧完整冻结观测，经过新VideoCache接口。只在smoke子目录临时形成CPU路由，成功后删除临时路由与预检检查点，不生成全队列geometry/visual缓存。

|项目|实测|
|---|---|
|完整帧/M/合法监督|96 / 735 / 96|
|顶层实际前向|2（1次更新前向+1次恢复后完整上下文推理）|
|可丢弃更新|1|
|原生计分帧|1，仅frame0接口核验|
|严格模型/优化器/RNG恢复|PASS；GPU模型state hash精确一致；CPU下一步轨迹测试一致|
|loss / BCE / Dice loss|0.410608709 / 0.142669544 / 0.267939180|
|完整更新测时|1.964257秒，短预检非吞吐估算|
|峰值allocated/reserved|0.374966 / 0.644531 GiB|
|新增MoGe/冻结视觉/正式训练更新|0 / 0 / 0|
|Python/tee退出码|0 / 0|

预检结果仅确认接口，不报告AP收益；未做全开发评价。首步delta零初始化，上游梯度为零属于预期，本轮不拿它冒充特征已学会。先前资源原型的第二步梯度证据保留，不重复测三病例。
短预检后仅完善终端/失败记录、非等待文件锁、每轮汇总、报告与缓存占用记账，未改变模型/更新或恢复数学；CPU回归再验证，无额外GPU调用。其精确原identity保留在`preflight_evidence/identity_at_smoke_155a41b.json`；因尚无任何全量准备或正式状态，只刷新交付源码校验摘要，并在delivery_identity_refresh.json记录，不覆盖预检证据。

## 准备规模与空间：元数据估算，不是已完成计算

训练视频T范围60–240，开发72–240（冻结protocol元数据）。全队列M/N_t极值尚未计算；三个原型病例通过不保证89/19全队列可运行。

|项目|本入口登记范围|
|---|---:|
|已有混合观测包中的几何帧|3,397|
|尚未在这些包登记的几何帧|24,637；可能另有兼容旧高分辨率缓存，运行时核对|
|已可复用epoch01视觉帧|366|
|仍需准备的冻结视觉帧|27,668（按当前注册范围）|
|全部视觉payload保守上界|111.67 GiB|
|几何+路由payload保守上界|122.89 GiB|
|含15%余量与2GiB记录空间的阈值|271.75 GiB|
|本轮元数据估算时可用空间|10.36 TiB|

几何上界假设每个像素单独成块，刻意保守；视觉按当前实测dtype（patch FP32/detail BF16/z0 FP32）与pad14尺寸推导，旧三病例混合包密度约5,165,300字节/帧只作参照，不冒充新分离缓存实际大小。估算未扣除所有可复用空间；运行时不足明确退出，不删除旧数据或降精度。
本轮没有执行任何新增MoGe/DINO提取；不能把将来的准备成本写成0。运行时frontend_calls.jsonl逐次登记实际成功调用（包含显式恢复后必要重算），geometry/visual_cost、queue_stats和prepared.json区分新增/复用/已缓存路由及owned/referenced字节。不能由这次单例约2秒推算全队列耗时；终端训练ETA依据已完成视频实测平均估算，只指剩余训练部分，不保证整个含准备/评价的墙钟。

## 评价与自动交付

每轮同一38开发视频完整上下文、全部5152唯一目标，不回到8槽；FP32工作logit恢复原生后排序AP，只正GT fake帧→视频→原来源等权，real不混入AP，空GT AP不填0/1。logit≥0（概率.5）的Dice/IoU/召回、real FPR、fake背景FPR、预测面积与常数AP并列。
分类直接沿用已有分析`analysis/source_comparison.csv`整理后的19来源动作/机位/内容字段；不是重新粗关键词分组，不冒充人工。训练观察动作选择复用已有official_action规则，不看成绩。
自动输出：每轮逐帧/逐视频/逐来源JSON与CSV；全部19/原4/新增15/对象16/外观背景3及动作、机位分项；训练观察/在线loss分开；epoch3主表primary_epoch3.csv、FULL_VIDEO−FRAME_ONLY来源差值、best_development_supplement.csv、history.csv/curves.png和程序report.md。报告同时列AP/Dice/召回/误报取舍及多来源方向，不把梯度或loss当效果。
沿用既有54个固定开发目标（原4各首中末、其余各中帧，包含real/fake），末轮保存工作RGB、官方GT、缓存epoch01 z0与两组概率/0.5mask；统一概率0–1，不按效果挑图，不复用旧11或epoch10预测图。RGB缩略显示明确为模型工作RGB。
每组epoch01/02/03.pt、last.pt、final.pt和best_development.pt（链接到已有epoch权重）保存在数据目录；last含模型、optimizer、epoch/视频位置、有效更新、合法帧呈现、RNG及协议/源码身份。

## 从任意目录启动与恢复

普通完整启动（**本轮未执行**）：

```bash
/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh
```

可选tmux启动：

```bash
tmux new-session -s pvst-fullvideo-match 'bash /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh'
```

显式恢复：

```bash
/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh --resume
# 或在新tmux中启动恢复：
tmux new-session -s pvst-fullvideo-resume 'bash /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh --resume'
```

仅准备观测（仍可能很长，由用户启动）：

```bash
/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh --prepare-only
```

只读/CPU元数据计划：

```bash
/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh --plan-only
```

同一输出目录用非等待文件锁拒绝重复实例，不等GPU、不干预其他进程。无需Codex继续指挥。启动前保留已提交实现，不在正式任务中途修改相关源码或配置；不兼容resume明确报错，不强行加载。
每个完整视频后原子last提交，恢复以它为准，不重复计入有效曝光；若最后视频已完成但该轮评价未完成，恢复从该边界补评价再推进。准备阶段补缺项，完成标记可信后不重复全量审计。每次调用日志新文件，不覆盖失败日志；正常管道保存Python实际退出码，tee码另打印；launcher被中断无法捕获Python状态时标UNKNOWN_INTERRUPTED并保存launcher_exit_code，不伪称成功。
终端实时阶段/帧进度、新增复用数、模式、epoch/3、视频/178、T/M、loss分项、lr、耗时、ETA及每轮汇总；无持续监控进程。正式训练不逐步开profiler，仅资源门记录实际后端。

## 绝对产物路径与验证

- 结果根：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1`
- 日志：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1/invocations/<时间戳_PID>.log`；同前缀`.exit_code`与`.launcher_exit_code`。
- 状态：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1/status.json`；异常另存`failure_<时间>.json`，不覆盖旧日志。
- 短预检：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1/smoke/result.json`；实际日志`invocations/20261124T060052_1280058.log`。
- CPU计划：`identity.json/protocol.json/config.json/orders.json/monitor.json/categories.json/space_estimate.json`。
- 全量执行后产生：`queue_stats.json/csv`、`queue_extrema.json`、`prepared.json`、`compute.json`、资源尝试记录、两模式目录与`report.md/summary.json`。当前这些全量完成标记不存在。
- 正式预览（将来生成）：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1/preview/index.html`。
- 仓库固定配置：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/docs/pvst/pvst_fullvideo_matched_config_v1.json`。

CPU检查20项（原型8项＋本轮12项）：数据来源隔离/数量、相同无放回顺序与观察、缓存身份和两种格式、精确Q1024等价、宏平均、优化器/RNG下一步一致、完整视频边界曝光、磁盘先行保护、仅OOM备用、并发拒绝、自动末次/开发选择表与54固定预览身份。另bash -n、Python编译、git diff --check。
本轮无正式结果、无全量耗时承诺。发布代码/配置/说明，权重/缓存/预测不入Git；计算准备状态与Git状态分开。**脚本已交付，全量任务尚未启动。** 用户启动后自行运行到既定结束，之后再主动接续结果分析。

## 几何准备收尾恢复修复（2026-11-24）

本次只将`fullvideo_matched_data.py`的统计路径改成`out / f'geometry_cost_{time.time_ns()}.json'`。
原错误是Path与字符串相加，不涉及MoGe推断、FP32路由、模型、loss、顺序、预算或精度。检查本实验两个入口的同类构造，未发现其他同类错误；visual_cost原括号构造保持不变。

新增CPU回归直接调用prepare_geometry：临时目录中两个视频的全部5帧缓存已存在，mock权重/版本/旧索引、MoGe构建、媒体读取、torch.load、路由及CUDA清理；确认0推断、cached_route_frames=5、生成可解析geometry_cost_<时间戳>.json。不是只测试f-string。共21项CPU检查通过（明确屏蔽CUDA可用性/种子入口）。

原进程已结束，退出1；在prepare阶段完成216/216视频后保存统计时报错，尚无两模式正式检查点，也无prepared/compute完成标记。文件/元数据检查为216视频、28,034/28,034个预期PT文件存在且非空，缺项0、空文件0、临时文件0、异常索引0，全部视频/PTS/尺寸/provider权重/scale/h身份一致。本轮没有读取这些张量、重新路由或验收数值。

恢复证据目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1/recovery/geometry_finalize_20261124T162105_1795508465655269352`。

- 旧identity、原FAILED状态、failure JSON、geometry_progress和失败调用记录已先备份；coverage_audit.json与coverage_by_video.csv保存一次文件覆盖核对。
- 仅在准备阶段、无正式状态且protocol/config/orders/monitor/categories/provider/冻结权重以及其他计算源码不变的前提下，active identity只更新fullvideo_matched_data.py这一个摘要。精确旧/新SHA、实际修复提交和前提证据见identity_update.json；原校验保持严格，没有忽略变化选项或重建identity。
- 几何缓存和各自identity不改，生成来源仍为原失败提交d9998c3777b429d141031489a444b7fe0c6edee7，不能回填为修复提交。
- 原调用新增MoGe24,637、复用3,397、总耗时3,093.170476秒；原journal仍含24,637个唯一MoGe调用。原geometry_progress阶段耗时3,092.746621秒。cost_baseline.json保存此真实成本；未来恢复调用的new_MoGe=0只代表该次跳过推断，不是累计成本归零。
- 完整缓存情况下，恢复预期new_MoGe=0、cached_route_frames=28,034，补写几何收尾统计后进入冻结视觉准备；恢复实际耗时/调用数尚未测量。全实验实际调用继续以保留的frontend_calls.jsonl累计，各次失败/状态与geometry_cost/visual_cost日志分别保存耗时。
- 本轮只执行一次`--plan-only`确认身份/配置可通过，退出0；不执行prepare、--resume、视觉、资源门或训练。计划里的旧混合包缺口字段不代表本次几何缓存仍缺失，恢复采用实际文件缺项逻辑。没有手写prepared/COMPLETE标记。

用户从任意目录续跑（**本轮未执行**）：

```bash
/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh --resume --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1
```

如需tmux：

```bash
tmux new-session -s pvst-fullvideo-match-resume 'bash /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/scripts/run_pvst_fullvideo_matched_localization_v1.sh --resume --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1'
```

**修复已交付，恢复任务尚未启动。** 原失败与缓存保留，用户自行启动，Codex不轮询。
