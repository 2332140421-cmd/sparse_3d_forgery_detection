# V8空间定位监督扩量匹配对照

2026-10-30，COMPLETE；分支`v8-spatial-supervision-expansion`，起点`eedfd9d`。
仅FROZEN_VISUAL与LORA_ADAPT各一个seed；不预设扩量或视觉适配有效，不运行三维增强。
[唯一配置](../v8/spatial_supervision_expansion_config.json)、
[冻结manifest](../v8/evidence/spatial_supervision_expansion_manifest.json)、ADR-0027。

## 冻结规模与实际资格

固定ANAKIN revision `978fe26e78ecaaf6562aa249be43c2c690484d0a`，metadata1023条，
425条has-masks不是425个合格场景。沿用原人物/交互描述及排除词规则，以inpainting/splicing为主，
原32pair全部保留（含原swap1条），共109候选；准备前已有53套下载。
不读取模型分数，原片正常肢体运动不排除，也不以fake异常排除。

冻结seed20260909的候选顺序后，只为未有明确审阅结论的新候选生成原片25/50/75%缩略图，
实际一次查看66组：48初审通过、18排除；再合并既有审阅，109条中21条场景不符合/无法确认稳定相机，
88条进入空间技术核验，其中19条帧号/数量/原生尺寸对应未解决，最终69pair。
保留合法空GT帧，缺失不当空GT；ZIP按文件名数值排序，RGB转L/255>=0.5；未截断、复制或猜偏移修补。
稀疏预审不能证明全段无短暂切镜；逐项原片证据与原因留在输出scene_review/technical_review。

|集合|目标pair|实际pair/视频|独立source|实际帧|类型(inpainting/splicing/swap)|小/中/大目标|
|---|---:|---:|---:|---:|---:|---:|
|train|60|29/58|29|7282|19/9/1|8/19/2|
|dev|20|20/40|20|4946|10/10/0|7/12/1|
|test_reserved|20|20/40|20|5041|13/7/0|7/9/4|

train保留原24pair并新增5pair；dev保留原8pair并新增12pair，均扩充成功，达到本轮训练门槛。
训练距目标缺31pair，如实冻结，不放宽场景凑60。全部69source互不跨集合，每source最多一个编辑版本。
分配顺序在技术验收前写入candidate_order：先锁原角色，按类型×目标面积轮转取样，
优先保留20个未见source作test_reserved，再填dev至20，余下为train；不是看到性能后的样本选择。
面积为非空GT帧的原生mask面积比例均值：small<0.01、medium[0.01,0.10)、large>=0.10。
保留小目标，不按面积筛掉难例。

test_reserved保守排除所有准备前已下载及历史manifest涉及的source；优先官方test，
实际11pair来自官方test、9来自官方train。新dev含官方validation8/官方train12，train均官方train。
官方split字段原样保留；本轮是研究子集，不冒充完整官方benchmark划分。
test_reserved只有媒体/GT/PTS技术验收与清单，无RGB训练缓存、DINO特征或模型评分。

## 最小实现与只读复用

在现有包内新增expansion_data.py（固定候选、一次场景预审、增量纯视觉缓存）及expansion.py
（两个指定条件与吞吐验证）。前者直接被后者的VisualVideo调用，输出既有指标需要的chunk/native_gt/close；
旧prepare固定48/16且旧Video要求几何/statistics，无法仅改配置支持本轮。此处只补纯视觉边界，不造XYZ/track/grouping。
复用现有官方加载、patch_features、逐帧pixel_sums、完整pair累计梯度、原图指标、原子checkpoint、曲线和静态索引。
旧入口参数默认不变。64旧视频直接引用原缓存；只为34个新增train/dev视频准备RGB512/GT/content。
完整manifest含身份、URL/source分组、role/split、官方来源、路径、全部实际PTS/帧数/mapping及场景审阅；
仓库与数据目录manifest解析内容完全一致；Git按每视频一行紧凑存储全部PTS，不另建并行队列。

新增raw媒体138个文件、261516888字节（含一次原片预审；已有有效文件复用），使用固定官方大小/哈希核验。
增量edited/mask及缓存准备实测104.37秒；原片预审下载/解码未单独计时，保留未记录，不重测补数。
原权重、旧checkpoint/报告、原缓存只读；新样本不运行MoGe/CoTracker，不复制旧大型缓存，不拟合三维统计。

## 模型与冻结训练规则

两个条件从同一官方AINPAINT spatial初始化重新开始：权重SHA
`6f6210b10a9b09aabd2dbb139e8797cd68e786a7591440af6d7dca4ba9e58cc7`，
代码revision `753ae1b700b054c97e15569718690211e36cdf53`，DINO revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`；无下载或升级。
FROZEN_VISUAL冻结DINO及已有LoRA，只训练decoder；LORA_ADAPT冻结原DINO，训练原有rank4/alpha8 LoRA及decoder。
骨干/LoRA模块eval，decoder训练Dropout正常，5层BN官方初始buffers固定、仿射可训练。不合并LoRA，不从旧best微调。

完整RGB512/255→右下各补6至518→官方normalize→37×37特征→decoder518裁回512→原图回映。
FP32骨干、FP16再FP32量化（LoRA路径可微）、BF16 decoder。冻结条件仅为新增train/dev提取特征，旧缓存按
权重/视频SHA/PTS/预处理/shape/dtype核对复用；适配训练和评价始终在线，不保存逐轮特征。

每帧内容归一化BCE+非空GT Dice(eps1)，空GT只背景BCE。每视频所有帧平均，fake后real各0.5；
每chunk即时backward为`.5*chunk帧loss总和/视频总帧数`，每pair只zero/clip/step一次，尾块不偏权。
seed20260909、原确定性pair shuffle、AdamW wd1e-4、decoder lr1e-4、LoRA lr1e-5、全体clip1。
最多30轮，dev Dice patience8/min_delta.002；best严格最高含epoch0，delta只控制patience。
两条件同队列/初始化/顺序/监督/选模及冻结计算配置；保留完整分辨率、全部帧，无检测门控。

每轮原4pair monitor（95qlg9/76rjm7/7p8r14/40y1x9）与完整dev，monitor不冒充完整train。
dev AP每5轮；best/last完整train/dev最终精确pixel AP，未算为null。原图坐标、正GT帧先视频内后视频间均值，
空GT误报单列，阈值0。新增Dice均值/中位数/全列表/<0.05数量、类型与预先目标面积分组。
固定25/50/75%保存原始响应与索引。紧凑checkpoint保存LoRA/decoder/optimizer/RNG及pair边界，不保存3亿参数主干。

## 吞吐验证与冻结配置

同一原训练an:95qlg9完整pair180帧，每组合仅一次完整更新，四组合均成功，随后全部丢弃。

|检查点|chunk|更新(s)|帧/s|allocated/reserved(GiB)|
|---|---:|---:|---:|---:|
|全部24块|2|26.169|6.878|1.690/1.809|
|关闭|2|19.178|9.386|5.468/5.766|
|关闭|4|18.031|9.983|9.760/10.061|
|关闭|8|18.323|9.824|18.331/18.822|

选择关闭检查点+chunk4，已写入唯一正式配置，两条件共用。原探针配置SHA与选定后配置SHA分别保留，实测数据不改。相对原配置吞吐+45.1%、更新耗时−31.1%；chunk8没有更快且不满足
相对探针前可用显存的20%余量。冻结条件在同一选定配置额外验收一次完整pair，0.772秒，LoRA精确不变而decoder更新；
适配条件LoRA/decoder梯度有限非零并更新，BN buffers不变，compact恢复预测一致。
冻结条件可训练decoder2,094,017；适配另训练LoRA1,579,312。所有探针更新丢弃，正式初始化和RNG重置。

按实际帧数外推满30轮（非实测完成成本）：冻结训练937秒、常规监测540秒、周期AP额外194秒、最终评价236秒；
LoRA分别21883/7178/190/1168秒，总计约8.99小时。未含新增特征提取、checkpoint/样例I/O及共享服务器波动；早停可缩短。
正式开始后不依据开发分数调整计算配置。

38项相关测试通过，覆盖新manifest/source隔离、纯视觉读取/PTS/mapping、保留集拒绝入模型、
两条件参数更新、BN固定、尾块加权、compact恢复、分组报告及发布失败保留实验COMPLETE状态。

## 独立运行与交付

输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/`。
单入口`research_tools.v8.pretrained_spatial_context.expansion`，通过必要测试和smoke后先commit/push，再独立tmux执行。
自动复用/补齐冻结特征→两条件→旧24pair LoRA best/12在新dev一次冻结参考评分→最终报告。
旧开发Dice0.312192仅历史记录；主比较为新完整dev上的LORA_ADAPT−FROZEN_VISUAL。
原8pair仅附子集接续、不单独选模，test_reserved全程禁止进入模型。

实验报告及旧资产只读核对完成后先写COMPLETE/report_complete，再执行文档与Git交付。
发布或远端核验失败仅写publication.json，原COMPLETE状态不覆盖，实际退出码及错误日志保留。
必要报告、小型结果和曲线自动普通提交推送；只允许当前任务分支且工作区干净，不覆盖用户改动。
本轮最终指标与实际成本见下文完成结果。不追加新训练、额外seed、三维或test评分。

## 正式启动记录

代码`2436d58def77a4b53ae1bfe5ebcf21e7e2575b6e`先普通推送并核对远端后，于`2026-10-30T12:11:20.755132+08:00`
在独立tmux `v8_spatial_supervision_expansion_v1`启动单入口，无`--resume`，两条件均重新从官方初始化开始。
日志`run.log`、任务状态`progress.json`、实际退出码`exit_code.txt`、交付状态`publication.json`均在输出目录；
启动时RUNNING，指标当时尚未产生；最终成绩见完成结果，不逐轮查询或中途改设置。
一次启动核对：Python PID `2051085`、tmux pane存活，状态RUNNING/features，正在补齐新增train/dev特征；
保留测试集无RGB模型缓存或DINO特征。本段为启动时事实记录，不代替最终完成状态。

## 完成结果

# V8空间监督扩量：冻结视觉与LoRA匹配对照

研究子集划分，不冒充完整官方benchmark；test_reserved仅技术验收，从未评分或提取测试DINO。
仅检验视觉适配和当前扩量设置，不归因于三维或component。全部帧/完整分辨率/空间BCE+Dice不变。
实际队列：{"candidate_counts": {"metadata": 1023, "has_masks": 425, "eligible_including_retained": 109, "masked_types": {"splicing": 150, "swap": 161, "inpainting": 114}, "existing_triplets": 53}, "scene_passed": 88, "technical_passed": 69, "actual_pairs": {"train": 29, "dev": 20, "test_reserved": 20}, "independent_sources": {"train": 29, "dev": 20, "test_reserved": 20}, "frames": {"train": 7282, "dev": 4946, "test_reserved": 5041}, "types": {"train": {"inpainting": 19, "splicing": 9, "swap": 1}, "dev": {"inpainting": 10, "splicing": 10}, "test_reserved": {"splicing": 7, "inpainting": 13}}, "exclusions": [{"video_id": "4xded9", "reason": "scene_not_matching", "review": {"video_id": "4xded9", "scene_status": "not_matching", "reason": "冲浪画面大面积海浪主导，原片不满足场景限制", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4xded9.jpg"}}, {"video_id": "4wge37", "reason": "scene_not_matching", "review": {"video_id": "4wge37", "scene_status": "not_matching", "reason": "瀑布占据画面大面积，动态水体主导背景", "human_activity": "visible", "camera_motion": "fixed_or_mild", "shot_changes": "none_observed_in_sampled_frames_full_playback_pending", "pending": "人工播放全段确认相机和切镜；未冻结训练资格"}}, {"video_id": "4wdm37", "reason": "scene_not_matching", "review": {"video_id": "4wdm37", "scene_status": "not_matching", "reason": "河岸岩石人物活动，水体占据大面积背景/前景", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4wdm37.jpg"}}, {"video_id": "4ymrn4", "reason": "scene_not_matching", "review": {"video_id": "4ymrn4", "scene_status": "not_matching", "reason": "场馆取样视角明显平移，前景器材位置变化，缺稳定相机依据", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4ymrn4.jpg"}}, {"video_id": "9kg054", "reason": "scene_not_matching", "review": {"video_id": "9kg054", "scene_status": "not_matching", "reason": "林间人物活动中树干位置大幅改变，明显跟随平移", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/9kg054.jpg"}}, {"video_id": "4ym134", "reason": "scene_not_matching", "review": {"video_id": "4ym134", "scene_status": "not_matching", "reason": "攀岩人物旁大面积瀑布/流水主导", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4ym134.jpg"}}, {"video_id": "4ym8n4", "reason": "scene_not_matching", "review": {"video_id": "4ym8n4", "scene_status": "not_matching", "reason": "篮球活动镜头随球场明显平移，不足以确认仅轻微运动", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4ym8n4.jpg"}}, {"video_id": "9l8eg4", "reason": "scene_not_matching", "review": {"video_id": "9l8eg4", "scene_status": "not_matching", "reason": "广角草地运动镜头取样视角变化明显，无法确认稳定相机", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/9l8eg4.jpg"}}, {"video_id": "7ogk04", "reason": "scene_not_matching", "review": {"video_id": "7ogk04", "scene_status": "not_matching", "reason": "街角近距人物跟拍，明显倾斜与视角变化", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/7ogk04.jpg"}}, {"video_id": "4q26w4", "reason": "scene_not_matching", "review": {"video_id": "4q26w4", "scene_status": "not_matching", "reason": "体育场画面跨取样时刻明显平移，相机运动超轻微范围", "human_activity": "visible", "camera_motion": "large_pan", "shot_changes": "none_observed_in_sampled_frames_full_playback_pending", "pending": "人工播放全段确认相机和切镜；未冻结训练资格"}}, {"video_id": "4qwn54", "reason": "scene_not_matching", "review": {"video_id": "4qwn54", "scene_status": "not_matching", "reason": "滑板跟拍，街道建筑与取景方向明显改变", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4qwn54.jpg"}}, {"video_id": "4qwe54", "reason": "scene_not_matching", "review": {"video_id": "4qwe54", "scene_status": "not_matching", "reason": "竖屏街景取样视角及建筑位置明显变化", "human_activity": "visible", "camera_motion": "strong_handheld_or_pan", "shot_changes": "none_observed_in_sampled_frames_full_playback_pending", "pending": "人工播放全段确认相机和切镜；未冻结训练资格"}}, {"video_id": "7og204", "reason": "scene_not_matching", "review": {"video_id": "7og204", "scene_status": "not_matching", "reason": "沙滩奔跑跟拍，地平线/视角明显转动", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/7og204.jpg"}}, {"video_id": "4wg3e7", "reason": "scene_not_matching", "review": {"video_id": "4wg3e7", "scene_status": "not_matching", "reason": "球场跟拍，场线及远景位置明显改变", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4wg3e7.jpg"}}, {"video_id": "9my5y9", "reason": "scene_not_matching", "review": {"video_id": "9my5y9", "scene_status": "not_matching", "reason": "临水步道跟拍，背景视角随人物改变，不能确认稳定相机", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/9my5y9.jpg"}}, {"video_id": "71d2d7", "reason": "scene_not_matching", "review": {"video_id": "71d2d7", "scene_status": "not_matching", "reason": "俯拍行走跟随，地面纹理随取景改变，缺稳定相机依据", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/71d2d7.jpg"}}, {"video_id": "71dk67", "reason": "scene_not_matching", "review": {"video_id": "71dk67", "scene_status": "not_matching", "reason": "人物附近大面积瀑布急流主导", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/71dk67.jpg"}}, {"video_id": "71dvd7", "reason": "scene_not_matching", "review": {"video_id": "71dvd7", "scene_status": "not_matching", "reason": "桥上骑行跟拍，近处栏杆透视明显变化且水面占较大区域", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/71dvd7.jpg"}}, {"video_id": "71dz67", "reason": "scene_not_matching", "review": {"video_id": "71dz67", "scene_status": "not_matching", "reason": "自行车跳跃镜头明显上摇/平移跟随，背景楼宇取景改变", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/71dz67.jpg"}}, {"video_id": "9l8554", "reason": "scene_not_matching", "review": {"video_id": "9l8554", "scene_status": "not_matching", "reason": "草地人物滚动镜头房屋位置/取景朝向改变，不能确认相机稳定", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/9l8554.jpg"}}, {"video_id": "4qwm54", "reason": "scene_not_matching", "review": {"video_id": "4qwm54", "scene_status": "not_matching", "reason": "道路行走近景跟拍，透视与背景视角变化明显", "human_activity": "visible", "preview": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_supervision_expansion_v1/scene_preview/4qwm54.jpg"}}, {"video_id": "4dpv24", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "95qnq9", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "7e8jj7", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "9g2gk7", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4ymo34", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4q2gw4", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4vo387", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "9lpgg7", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "7e2k57", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "40ypx9", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "72yv89", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4wg1e7", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "76r0l7", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "71dpd7", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "9momg4", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4vg684", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "9l8654", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4xd2r9", "reason": "alignment_unresolved: frame numbering/count/native mask size"}, {"video_id": "4q2zw4", "reason": "alignment_unresolved: frame numbering/count/native mask size"}], "target_shortfall": {"train": 31, "dev": 0, "test_reserved": 0}, "training_allowed": 1, "seconds": 104.36731357499957, "new_media_files": 138, "new_media_bytes": 261516888, "original_preview_seconds": null, "official_splits": {"train": {"train": 29}, "dev": {"validation": 8, "train": 12}, "test_reserved": {"test": 11, "train": 9}}, "area_groups": {"train": {"medium": 19, "small": 8, "large": 2}, "dev": {"medium": 12, "small": 7, "large": 1}, "test_reserved": {"small": 7, "medium": 9, "large": 4}}}

|条件|checkpoint/轮次|完整训练Dice|开发Dice|IoU|pixel AP|GT召回|real空间FP|
|---|---|---:|---:|---:|---:|---:|---:|
|FROZEN_VISUAL|best/5|0.468311|0.204793|0.144827|0.291179|0.241096|0.028768|
|FROZEN_VISUAL|last/13|0.635555|0.200038|0.144206|0.377005|0.263752|0.011894|
|LORA_ADAPT|best/26|0.734976|0.268522|0.203035|0.433506|0.325815|0.011476|
|LORA_ADAPT|last/30|0.713135|0.196182|0.135897|0.407567|0.211789|0.004343|
|LORA_24_REFERENCE|旧best/12|不重新评分旧train|0.231016|0.170774|0.361594|0.330010|0.036878|
同队列LoRA−冻结视觉：{"spatial_dice": 0.06372897426346977, "spatial_iou": 0.058207690028011294, "pixel_ap": 0.142326961558367, "gt_recall": 0.0847186333439654, "real_spatial_fpr": -0.017292635819145967}
FROZEN_VISUAL分布/分组：{"mean": 0.204793288087986, "median": 0.11115658588249083, "dice_below_0_05": 10, "groups": {"type": {"inpainting": {"n": 10, "mean": 0.24765575420361552, "median": 0.2101597230356717}, "splicing": {"n": 10, "mean": 0.16193082197235648, "median": 0.005146480265159056}}, "area": {"large": {"n": 1, "mean": 0.7662558860762865, "median": 0.7662558860762865}, "medium": {"n": 12, "mean": 0.2371030536163612, "median": 0.2101597230356717}, "small": {"n": 7, "mean": 0.06919617604101426, "median": 1.5431018420265576e-05}}}}
FROZEN_VISUAL best完整train−dev差=+0.263518，曲线monitor仅原4训练pair，不是完整train。实际13轮/377更新，成本：{"training_loop_s": 961.9986938377842, "train_updates_s": 439.1370845064521, "routine_evaluation_s": 475.3580996338278, "final_evaluation_s": 657.5422824090347, "peak_allocated_bytes": 1908407296, "peak_reserved_bytes": 2042626048, "parameters": {"lora": 0, "decoder": 2094017, "frozen": 305952048, "checkpointed_blocks": 0}}
FROZEN_VISUAL完整fake覆盖/背景：{"video_mean_GT_recall": 0.24109604529484258, "pooled_GT_recall": 0.2896427841763955, "pooled_BG_FPR": 0.01743298519698416}
LORA_ADAPT分布/分组：{"mean": 0.26852226235145576, "median": 0.1564499619971244, "dice_below_0_05": 8, "groups": {"type": {"inpainting": {"n": 10, "mean": 0.24467199700289877, "median": 0.1564499619971244}, "splicing": {"n": 10, "mean": 0.29237252770001276, "median": 0.22736382666742677}}, "area": {"large": {"n": 1, "mean": 0.8297044631552579, "median": 0.8297044631552579}, "medium": {"n": 12, "mean": 0.26119067212335295, "median": 0.1564499619971244}, "small": {"n": 7, "mean": 0.20092181691337446, "median": 0.08511350993566129}}}}
LORA_ADAPT best完整train−dev差=+0.466453，曲线monitor仅原4训练pair，不是完整train。实际30轮/870更新，成本：{"training_loop_s": 29811.042979296297, "train_updates_s": 21840.564238172024, "routine_evaluation_s": 7824.9941195761785, "final_evaluation_s": 1583.1631843531504, "peak_allocated_bytes": 10510061568, "peak_reserved_bytes": 10806624256, "parameters": {"lora": 1579312, "decoder": 2094017, "frozen": 304372736, "checkpointed_blocks": 0}}
LORA_ADAPT完整fake覆盖/背景：{"video_mean_GT_recall": 0.325814678638808, "pooled_GT_recall": 0.3107237581596428, "pooled_BG_FPR": 0.007870180429808539}

|开发fake|类型|目标面积组|冻结Dice|LoRA Dice|差|
|---|---|---|---:|---:|---:|
|an:4xd3r9|inpainting|medium|0.035902|0.021983|-0.013918|
|an:7e2p57|splicing|small|0.000000|0.001175|+0.001175|
|an:95q8g9|inpainting|medium|0.007833|0.152294|+0.144461|
|an:4rvro9|splicing|medium|0.757729|0.874674|+0.116945|
|an:4xd5r9|inpainting|medium|0.233908|0.000296|-0.233612|
|an:71g867|inpainting|medium|0.246036|0.189381|-0.056655|
|an:7e2e57|splicing|medium|0.008307|0.008775|+0.000469|
|an:4wgx37|inpainting|large|0.766256|0.829704|+0.063449|
|an:73rqr7|splicing|medium|0.368900|0.633823|+0.264924|
|an:4z1q27|splicing|small|0.000000|0.437706|+0.437706|
|an:40elg7|inpainting|medium|0.013289|0.000000|-0.013289|
|an:4qwdw4|inpainting|medium|0.478685|0.468037|-0.010648|
|an:4z1v27|splicing|small|0.000015|0.000000|-0.000015|
|an:7e83j7|splicing|small|0.001986|0.017402|+0.015416|
|an:4xend7|inpainting|medium|0.186412|0.160606|-0.025805|
|an:4vge14|splicing|small|0.220419|0.495442|+0.275023|
|an:4ny267|splicing|small|0.000000|0.369614|+0.369614|
|an:7og5g4|inpainting|medium|0.000000|0.000000|+0.000000|
|an:7p8l14|splicing|small|0.261953|0.085114|-0.176839|
|an:7ogeg4|inpainting|medium|0.508238|0.624419|+0.116182|
旧24pair best/12在新dev一次参考评价耗时=296.7495820540935秒；旧dev Dice0.312192仅历史记录。参考模型与新模型的训练队列、计算配置及选模过程不同，不视作单独扩量的严格因果消融。
原8pair开发子集仅历史接续、不用于选模：{"FROZEN_VISUAL": {"mean": 0.2569960842268586, "median": 0.1349047073269559, "dice_below_0_05": 4, "groups": {"type": {"inpainting": {"n": 5, "mean": 0.2579866862884953, "median": 0.23390784448013677}, "splicing": {"n": 3, "mean": 0.2553450807907975, "median": 0.008306602686078551}}, "area": {"large": {"n": 1, "mean": 0.7662558860762865, "median": 0.7662558860762865}, "medium": {"n": 6, "mean": 0.21495213128976376, "median": 0.1349047073269559}, "small": {"n": 1, "mean": 0.0, "median": 0.0}}}, "videos": [{"video_id": "c98cb33f48392c108c40f206", "pair_id": "an:4xd3r9", "generator": "inpainting", "spatial_dice": 0.035901570173774995, "spatial_iou": 0.019333021756301524, "pixel_ap": 0.027391818199127525}, {"video_id": "e55c9628ed7641f96707430e", "pair_id": "an:7e2p57", "generator": "splicing", "spatial_dice": 0.0, "spatial_iou": 0.0, "pixel_ap": 0.0007345305973372364}, {"video_id": "37b22da558e1daee1e72b707", "pair_id": "an:95q8g9", "generator": "inpainting", "spatial_dice": 0.00783251218743887, "spatial_iou": 0.003980092140456933, "pixel_ap": 0.096793849063258}, {"video_id": "a2a08b465a8cea8a95403ee7", "pair_id": "an:4rvro9", "generator": "splicing", "spatial_dice": 0.757728639686314, "spatial_iou": 0.6106871443860455, "pixel_ap": 0.8639970444049622}, {"video_id": "2649719f53fe1a54aa56ab8f", "pair_id": "an:4xd5r9", "generator": "inpainting", "spatial_dice": 0.23390784448013677, "spatial_iou": 0.15567216722755098, "pixel_ap": 0.39453813976979873}, {"video_id": "8a6ce2b60f7cb1353c2919cc", "pair_id": "an:71g867", "generator": "inpainting", "spatial_dice": 0.24603561852483927, "spatial_iou": 0.16009822829973652, "pixel_ap": 0.5259576801446704}, {"video_id": "e73bc60d4e2c5b994ee45ba4", "pair_id": "an:7e2e57", "generator": "splicing", "spatial_dice": 0.008306602686078551, "spatial_iou": 0.0062285117261568395, "pixel_ap": 0.023115215860618297}, {"video_id": "f2cd41132415857399b19518", "pair_id": "an:4wgx37", "generator": "inpainting", "spatial_dice": 0.7662558860762865, "spatial_iou": 0.6376991832555448, "pixel_ap": 0.8764616035544852}]}, "LORA_ADAPT": {"mean": 0.25978519462126864, "median": 0.08713838975956074, "dice_below_0_05": 4, "groups": {"type": {"inpainting": {"n": 5, "mean": 0.23873149751650224, "median": 0.1522936453651448}, "splicing": {"n": 3, "mean": 0.29487468979587933, "median": 0.008775258866566254}}, "area": {"large": {"n": 1, "mean": 0.8297044631552579, "median": 0.8297044631552579}, "medium": {"n": 6, "mean": 0.2079003778011411, "median": 0.08713838975956074}, "small": {"n": 1, "mean": 0.0011748270080447403, "median": 0.0011748270080447403}}}, "videos": [{"video_id": "c98cb33f48392c108c40f206", "pair_id": "an:4xd3r9", "generator": "inpainting", "spatial_dice": 0.021983134153976683, "spatial_iou": 0.012299513696897564, "pixel_ap": 0.03075948315733803}, {"video_id": "e55c9628ed7641f96707430e", "pair_id": "an:7e2p57", "generator": "splicing", "spatial_dice": 0.0011748270080447403, "spatial_iou": 0.000612714755391512, "pixel_ap": 0.0018172510784130835}, {"video_id": "37b22da558e1daee1e72b707", "pair_id": "an:95q8g9", "generator": "inpainting", "spatial_dice": 0.1522936453651448, "spatial_iou": 0.09946714640126422, "pixel_ap": 0.5182635481240284}, {"video_id": "a2a08b465a8cea8a95403ee7", "pair_id": "an:4rvro9", "generator": "splicing", "spatial_dice": 0.8746739835130269, "spatial_iou": 0.777565833842314, "pixel_ap": 0.9619828024776007}, {"video_id": "2649719f53fe1a54aa56ab8f", "pair_id": "an:4xd5r9", "generator": "inpainting", "spatial_dice": 0.00029566592860923077, "spatial_iou": 0.00014919002036889846, "pixel_ap": 0.3704237557341251}, {"video_id": "8a6ce2b60f7cb1353c2919cc", "pair_id": "an:71g867", "generator": "inpainting", "spatial_dice": 0.1893805789795227, "spatial_iou": 0.12805707887422543, "pixel_ap": 0.5307536036094028}, {"video_id": "e73bc60d4e2c5b994ee45ba4", "pair_id": "an:7e2e57", "generator": "splicing", "spatial_dice": 0.008775258866566254, "spatial_iou": 0.004823081594884247, "pixel_ap": 0.09489021557517598}, {"video_id": "f2cd41132415857399b19518", "pair_id": "an:4wgx37", "generator": "inpainting", "spatial_dice": 0.8297044631552579, "spatial_iou": 0.7217714789430097, "pixel_ap": 0.9152479929916598}]}}
固定25/50/75%边界（原生2px，仅固定样例）和完整背景统计须分开解释：{"FROZEN_VISUAL": {"interior_recall": 0.2653728187660628, "GT_boundary_recall": 0.17132007498004928, "near_boundary_BG_FPR": 0.15438364219056103, "far_BG_FPR": 0.021872044417634035}, "LORA_ADAPT": {"interior_recall": 0.37490313549026333, "GT_boundary_recall": 0.2507113605640273, "near_boundary_BG_FPR": 0.2186953461386552, "far_BG_FPR": 0.009916864371316893}}
吞吐实测及外推（非正式耗时）：{"identity": {"config_sha256": "6fddea079a9bbca771e707cdbd6d843321d7c2ac47bd7776a77d0bc90013fe9e", "manifest_sha256": "0acd96c4d09ca880b08d2a0b623d8ed7f1057721354b474469ab0255b4949968"}, "records": [{"status": "ok", "strategy": "all", "chunk_frames": 2, "peak_allocated_bytes": 1814404608, "peak_reserved_bytes": 1941962752, "loss": 0.25132977012544877, "bce": 0.07819135559515822, "dice": 0.17313841614458295, "gradient_norm": 4.455781936645508, "gradient_abs_sum": {"lora": 2080.230396911502, "decoder": 496.57180676981807}, "seconds": 26.168827301822603, "frames": 180, "bn_status": {"strategy": "FIXED_BN", "layers": 5, "buffers_equal": 1, "all_eval": 1, "affine_trainable": 1}, "frames_per_second": 6.8784129271036685, "headroom_ok": 1}, {"status": "ok", "strategy": "none", "chunk_frames": 2, "peak_allocated_bytes": 5870822400, "peak_reserved_bytes": 6190792704, "loss": 0.25132977012544877, "bce": 0.07819135559515822, "dice": 0.17313841614458295, "gradient_norm": 4.455782890319824, "gradient_abs_sum": {"lora": 2080.2308516204357, "decoder": 496.57179134711623}, "seconds": 19.178033619187772, "frames": 180, "bn_status": {"strategy": "FIXED_BN", "layers": 5, "buffers_equal": 1, "all_eval": 1, "affine_trainable": 1}, "frames_per_second": 9.385738057102403, "headroom_ok": 1}, {"status": "ok", "strategy": "none", "chunk_frames": 4, "peak_allocated_bytes": 10479915008, "peak_reserved_bytes": 10802429952, "loss": 0.25157967435403, "bce": 0.07833876160697804, "dice": 0.1732409116294649, "gradient_norm": 4.466766834259033, "gradient_abs_sum": {"lora": 2085.161223605275, "decoder": 495.9341443963349}, "seconds": 18.030607811175287, "frames": 180, "bn_status": {"strategy": "FIXED_BN", "layers": 5, "buffers_equal": 1, "all_eval": 1, "affine_trainable": 1}, "frames_per_second": 9.98302452613033, "headroom_ok": 1}, {"status": "ok", "strategy": "none", "chunk_frames": 8, "peak_allocated_bytes": 19682726400, "peak_reserved_bytes": 20210253824, "loss": 0.2515328350580401, "bce": 0.07832621215946145, "dice": 0.17320662273301018, "gradient_norm": 4.467588901519775, "gradient_abs_sum": {"lora": 2085.5377859026194, "decoder": 495.8147089295089}, "seconds": 18.322883286513388, "frames": 180, "bn_status": {"strategy": "FIXED_BN", "layers": 5, "buffers_equal": 1, "all_eval": 1, "affine_trainable": 1}, "frames_per_second": 9.823781398667181, "headroom_ok": 0}, {"condition": "FROZEN_VISUAL", "strategy": "none", "chunk_frames": 4, "peak_allocated_bytes": 1900902400, "peak_reserved_bytes": 2048917504, "loss": 0.2513179702684283, "bce": 0.07818545250015127, "dice": 0.17313251429133947, "gradient_norm": 0.9034111499786377, "gradient_abs_sum": {"lora": 0, "decoder": 496.43808203935623}, "seconds": 0.7718234928324819, "frames": 180}], "execution": {"chunk_frames": 4, "checkpoint_strategy": "none"}, "parameters": {"FROZEN_VISUAL": {"lora": 0, "decoder": 2094017, "frozen": 305952048, "checkpointed_blocks": 0}, "LORA_ADAPT": {"lora": 1579312, "decoder": 2094017, "frozen": 304372736, "checkpointed_blocks": 0}}, "evaluation_rates": {"FROZEN_VISUAL": {"seconds_per_frame": 0.0031142417910612293, "exact_AP_seconds_per_frame": 0.009658492827374075}, "LORA_ADAPT": {"seconds_per_frame": 0.04138329577528768, "exact_AP_seconds_per_frame": 0.04777649894563688}}, "projection_max30": {"FROZEN_VISUAL": {"train30_s": 936.7364458010221, "monitor_dev30_s": 540.1963810774809, "periodic_AP_extra_s": 194.20719375362, "final_best_last_s": 236.20810058626037}, "LORA_ADAPT": {"train30_s": 21883.14768016307, "monitor_dev30_s": 7178.3464851814015, "periodic_AP_extra_s": 189.72469728328286, "final_best_last_s": 1168.4220582144956}}, "available_bytes_before_probe": 23435804672, "limits": "one complete fixed training pair per compute combination; at most4; all smoke weights discarded; budgets extrapolate, exclude IO and shared-machine variation", "frozen_execution_config_sha256": "be9ccc553edb8752f84a73e3857a85ca5b87372187604afa352f070d056b45a6", "configuration_note": "identity retains the original pre-probe config SHA; frozen_execution_config_sha256 identifies the final config after selecting measured none/chunk4; measurements unchanged"}
新增冻结特征成本（仅新train/dev，不含旧缓存或保留测试）：{"videos": 34, "frames": 4244, "seconds": 282.15982314478606, "bytes": 11898958080}
报告前新增独立输出存储=17105754032字节；旧缓存直接引用，无新MoGe/CoTracker，无逐轮DINO特征。
下一步建议：先在更多独立source上验证当前视觉适配收益的稳定性。
只作单seed描述，不能把背景误报降低本身写成定位改善；不自动执行后续训练或测试评分。

## 完成后的只读审阅（2026-10-31）

读取已有结果、曲线及固定样例，未重跑训练、评价或特征提取。结果提交`6c61a889`已核对远端一致；数据目录状态为COMPLETE/report_complete，原退出码为0。

同一20条开发fake的best逐视频Dice：LoRA相对冻结视觉11条提高、1条持平、8条下降。
inpainting均值差−0.002984，splicing均值差+0.130442，收益并非各编辑类型一致。
Dice低于0.05的样本由10条降到8条，仍有明显失败样本。
完整训练与开发Dice差从0.263518增至0.466453；曲线中的训练monitor仅含原4个pair，不能冒充完整训练集指标。
LoRA在best第26轮后，开发Dice到last第30轮由0.268522降至0.196182，GT召回由0.325815降至0.211789，
同时real误报由0.011476降至0.004343。后期更少预测正区域未带来更好重叠。

固定25/50/75%样例统计显示内部与边界GT召回提高、远背景误报降低，但近边界背景误报增加；该统计不能外推为全视频边界指标。
另按manifest顺序查看前两条开发fake的固定50%样例：`an:4xd3r9`与`an:7e2p57`中预测仍主要漏掉或偏离GT，
响应更集中不等于位置正确。以上样例未按结果挑选。

实际训练仅29个pair，比原训练增加5个，未达到60个目标；旧24pair模型的新dev参考成绩也不能隔离扩量的因果效果。
唯一下一步建议仍是补充独立source上的空间监督，验证当前视觉适配收益的稳定性；本轮不执行后续实验。
