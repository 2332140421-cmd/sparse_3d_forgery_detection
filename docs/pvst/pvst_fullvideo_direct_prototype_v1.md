# PVST完整视频原始块原型 v1：资源测量收尾

2026-11-24。本次仅CPU读取指定结果目录的summary.json、measurements.csv和三个病例result.json，核对记录并更新文档。新增模型前向、优化更新、冻结特征提取、MoGe、检查点加载均为0。没有启动监控或后续实验。

## 终态与来源

本轮开始一次性确认：无资源测量进程在运行；三个病例结果均PASS，summary为COMPLETE、pending_cases为空；两次终端调用保存的Python退出码均为0。未继续轮询。

本报告唯一结果根：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run`。未读取或合并上级默认目录的首次环境失败记录。机器原始产物保持不变。

- 小病例执行提交：`4384cfddd3387b602bf91b275eaac7151924bfda`。
- 中、高病例执行提交：`4e997b4e19cd50527d835219d3117b3fc9fb8631`。
- 两版本的`fullvideo_direct_model.py`相同，后续提交仅完善环境/记录/交接；三病例六模式初值hash相同：`0f6e601206cd848859d412e62e4ce73fb6f7ffb3d072bf9db487857e13b564e2`。
- 本次文档收尾起点：`4e997b4e19cd50527d835219d3117b3fc9fb8631`；最终发布HEAD见Git回执。
- `summary.json`各病例与各自result.json一致；measurements.csv全部12条step的loss、耗时及显存与JSON逐项一致；T、每帧计数总和与M一致。只核对小型记录，没有扫描观测缓存或加载权重。

## 已实现结构与测量边界

入口`research_tools.xyz_pilot.fullvideo_direct_probe`调用独立`FullVideoDirect`；授权见ADR-0055/根契约17.12。旧8槽单目标模型及外部官方代码不改。
冻结扩充空间epoch01的通用DINO、适配LoRA、官方HighResContextHead，始终eval/no_grad。同一次原空间前向捕获1024维patch、predictor真实输入detail32和工作logit z0。
工作RGB长边384，ImageNet归一化，按14补齐；按实际RGB算DINO pad，不复用DFormer pad。例如216×384对应224×392，patch1024×16×28，detail32×128×224。
新patch投影1024→128/GELU，双线性插值至DINO补齐尺寸后裁回工作网格；自身MoGe的xyz_provider在FP32只除scale=12.880994319915771一次，以h=.5/origin0三轴floor路由所有有效点与非空块。几何调制、逐点MLP及FP32学习池化生成C128。
全片C拼接，XYZ块中心与距首帧实际PTS编码加入Q/K，V使用内容。两层D128/4头/FF256、dropout0。FRAME_ONLY逐帧原始块交互；FULL_VIDEO全部原始块直接双向交互，无中继、筛块、8槽截止或只返回末帧R。
各帧R投影32、按points/index回填，与冻结detail32融合，64→32卷积/GN/GELU→1通道delta；末层weight/bias零初始化，z=z0+delta。全部帧定位，无效几何保留视觉路径；没有帧分类/全块均值分类门控。冻结二维与logit修正是本原型具体选择。
各模式同一新初值、同一seed20260909，新AdamW(lr1e-4/wd.01/clip1)。每帧BCE均值＋soft Dice(eps1)，在合法监督帧上平均，整片一次backward/step；40y1x9_fake[6,10)保留上下文与输出，不监督。新模块481,954参数。
按帧编码/解码及关系层使用非重入checkpoint，明确绑定逐帧输入，不detach。冻结观测在CPU，准备后释放DINO再测主网络；所有病例Query chunk=0，未启用备用1024。

## 六条件完整汇总

F/B/U分别为前向/反向/optimizer时间，单位秒；显存A/R为峰值allocated/reserved，单位GiB。梯度列指第2步要求的全部模块组有限且非零，首步上游梯度为零符合delta末层零初始化。

|病例|模式|完整T|合法监督|M|步数|第1步 F/B/U(s)|第2步 F/B/U(s)|峰值A/R(GiB)|实际后端|Q分块|梯度|退出码|
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---:|
|4ym5n4_fake|FRAME_ONLY|96|96|735|2|0.577070/1.778262/0.018509|0.437332/1.622036/0.000545|0.379266/0.650391|SDPA Flash|0（不分块）|PASS|0|
|4ym5n4_fake|FULL_VIDEO|96|96|735|2|0.341489/1.344860/0.001098|0.339413/1.346775/0.000538|0.379266/0.662109|SDPA Flash|0（不分块）|PASS|0|
|40y1x9_fake|FRAME_ONLY|150|146|22924|2|0.793443/2.047413/0.017637|0.607654/1.977857/0.000563|0.389193/0.769531|SDPA Flash|0（不分块）|PASS|0|
|40y1x9_fake|FULL_VIDEO|150|146|22924|2|0.469756/1.572075/0.001218|0.455784/1.577190/0.000544|0.389176/0.832031|SDPA Flash|0（不分块）|PASS|0|
|9mygy9_fake|FRAME_ONLY|120|120|50921|2|0.579388/1.630255/0.001078|0.516406/1.661415/0.000529|0.461936/0.785156|SDPA Flash|0（不分块）|PASS|0|
|9mygy9_fake|FULL_VIDEO|120|120|50921|2|0.415367/1.349893/0.001146|0.399333/1.350923/0.000554|0.505254/0.759766|SDPA Flash|0（不分块）|PASS|0|

后端由实际profiler记录`aten::_scaled_dot_product_flash_attention`和对应backward确认，不是根据启用开关推断。全部使用PyTorch内置实现，无独立FlashAttention安装。无失败模式、无未完成步；没有把缺测项填0。
首步上游梯度为零，但AdamW weight decay仍可能改变参数，因此以第2步有限非零梯度及实际更新作为链路证据。全部六模式第二步visual、geometry、point、score、value、layers、pixel_context、correction满足检查；第一步初始z与缓存z0最大差全部为0。

### 全片连接与范围证据

|病例|每帧非空块数范围|有效点帧次|FRAME_ONLY首帧R对后续C梯度最大值|FULL_VIDEO对应值|
|---|---|---:|---:|---:|
|4ym5n4_fake|7–8|7962624|0.000000000|0.513935983|
|40y1x9_fake|45–323|8843496|0.000000000|0.682840645|
|9mygy9_fake|414–439|8960037|0.000000000|0.652332842|

这确认R层的跨未来帧依赖及同帧模式屏蔽，没有用初始化delta=0时的相同logit判断连接。它不是顺序理解或定位效果证据。全部帧工作输出尺寸已有检查；原生恢复为FP32 bilinear、align_corners=False。

## FULL_VIDEO相对FRAME_ONLY的资源变化

下表使用第2步F+B+U减少首步冷启动影响，但每条件只有2步，且FRAME_ONLY先跑、FULL_VIDEO后跑，不是稳定吞吐测量。

|病例|第2步FRAME_ONLY秒|第2步FULL_VIDEO秒|时间差秒（相对变化）|峰值allocated差MiB|峰值reserved差MiB|
|---|---:|---:|---|---:|---:|
|4ym5n4_fake|2.059912|1.686726|-0.373187（-18.12%）|+0.000000|+12.000|
|40y1x9_fake|2.586074|2.033517|-0.552556（-21.37%）|-0.016602|+64.000|
|9mygy9_fake|2.178350|1.750810|-0.427539（-19.63%）|+44.357422|-26.000|

高病例allocated由0.461936升至0.505254 GiB，增加44.36 MiB、约9.38%；中病例近似不变，小病例相同。reserved变化分别+12、+64、−26 MiB，受allocator缓存和执行顺序影响，不能把reserved下降解释为全片注意力内存复杂度更低。
本次第2步FULL_VIDEO总时间反而少18.12%–21.37%。FRAME_ONLY有大量逐帧小算子/注意力调用，FULL_VIDEO合并为少量大调用，可能降低调度开销；这是合理解释而非已隔离验证的原因。时间还受冷启动、CPU profiler、checkpoint重算和并发状态影响，不能据此宣称FULL_VIDEO普遍更快或长期训练更便宜；全片两两交互的理论二次计算量没有消失。
前向包括可训练块编码、关系、全部帧定位及loss；loss反向、更新分列。另各模式有一次结构依赖VJP，不包含在表中F/B/U：

|病例|FRAME_ONLY额外VJP秒|FULL_VIDEO额外VJP秒|
|---|---:|---:|
|4ym5n4_fake|0.230205|0.003269|
|40y1x9_fake|0.342425|0.018039|
|9mygy9_fake|0.253578|0.070635|

主网络显存/时间不含在线DINO；已使用非重入checkpoint和实际高效SDPA。不是未优化的朴素M×M权重矩阵成本，也不是包含视觉前向的端到端训练成本。

### 既有冻结观测准备（本轮没有新增）

|病例|既有真实冻结前向数|准备秒|缓存字节|冻结权重/缓冲区|
|---|---:|---:|---:|---|
|4ym5n4_fake|96|8.601568|495868800|不变|
|40y1x9_fake|150|9.686899|774795000|不变|
|9mygy9_fake|120|7.827928|619836000|不变|

累计366次冻结前向、12次可丢弃预检更新发生在此前资源测量阶段；本轮均未重做，新增MoGe仍为0。中高两病例的既有终端调用总墙钟150.002929秒（invocations/result_1795468028347656690.json），包含准备、CPU路由/读取、记录等，不用F/B/U之和冒充整次墙钟。小病例原调用未保存monotonic总墙钟，本报告不补造精确值。

## 资源判断与唯一下一步

1. **可运行：是，限这三个完整病例。** 最大实测50,921个原始块、120帧，全片双向模式完成2次有效更新，梯度及输出检查通过。三个病例没有OOM或失败阶段。不能从两步小幅loss变化推断效果或收敛。
2. **全片成本：本次增量可控。** 以六条件实测为准，高病例allocated增加约44.36MiB；没有观察到持续时间增加，但不据此作普遍速度承诺。
3. **无需为已测三例再新增精确显存优化。** 当前已用SDPA与非重入checkpoint，默认不分Query全部通过，峰值allocated最高约.505GiB、reserved最高约.832GiB。没有资源证据要求启动Q=1024备用路径，更不应改中继、删块或增h。其他更长/更多块的视频仍可能需要精确优化，当前未测。
4. **适合进入完整数据定位对照的准备阶段，不等于已经具备全89来源运行保证。** 三条训练来源没有覆盖整个队列的长度/占据块极值、缓存范围或长期训练吞吐；仍未产生定位效果证据。

**唯一下一步建议：准备同一冻结视觉底座下、完整来源隔离数据上的FRAME_ONLY vs FULL_VIDEO匹配定位对照协议与终端脚本。** 保持XYZ切块、联合编码、定位路径和监督范围一致，匹配有效监督帧呈现量，预先固定预算及末次评价；在准备中先依据已有清单/几何记录列明全队列T/M范围和观测缺口，未覆盖的资源极值再单独授权处理。不要因三例PASS直接开全量训练，也不新增第三个结构或改Query配置。本轮不执行这些准备/计算。

## 产物与文档检查

- 原始程序报告：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/report.md`。该报告由原脚本生成，不覆盖。
- 本轮汇总CSV：`docs/pvst/pvst_fullvideo_direct_resource_summary_v1.csv`（6行，含完整精度逐步时间/显存及检查结果）。
- 实际读取：`summary.json`、`measurements.csv`、`cases/{4ym5n4_fake,40y1x9_fake,9mygy9_fake}/result.json`和指定目录内小型调用退出记录。
- 仅CPU做跨表一致性、有限数、T/M/步数检查与git diff --check；没有模型测试、权重恢复或新推理。仅提交本报告及轻量CSV。
- 本轮计算读取已结束；Git发布状态由最终回执独立报告。后续长训练仍由用户终端/tmux启动，Codex不持续轮询。
