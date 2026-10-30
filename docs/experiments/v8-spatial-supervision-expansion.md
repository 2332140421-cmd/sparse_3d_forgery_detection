# V8空间定位监督扩量匹配对照

2026-10-30，PREFLIGHT_COMPLETE；分支`v8-spatial-supervision-expansion`，起点`eedfd9d`。
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
尚无本轮最终成绩，最终表及实际成本由完成报告补齐。不追加新训练、额外seed、三维或test评分。
