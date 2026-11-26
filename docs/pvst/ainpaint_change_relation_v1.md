# AINPAINT视觉底座上的全片变化关系定位实验v1

状态：实现、9项CPU检查、固定真实GPU预检与最终CPU计划检查通过。**正式训练尚未启动，由用户终端启动。**
本轮不报告开发收益，不追加MumPy或学习率实验。路线依据是用户待验证的PVST设计；AINPAINT为母体，STIL/P4Transformer仅作为特征变化/局部时空再全局交互的思想参照，不能称这些论文已证明本新组合有效。

## 实际结构与三组对照

```text
只读epoch01 DINO/LoRA patch1024 + 已有MoGe/成员映射
 → 1024→128投影/GELU → 插值DINO pad尺寸 → 裁工作画面
 → 绝对/块内坐标MLP(gamma,beta)调制视觉 → 点MLP → 块内FP32学习softmax聚合 C128
 → 独立变化轮1 → 全局交互1 → 独立变化轮2 → 全局交互2 → 逐块R128
 → LN → 成员回投影工作像素 → 右下补零到DINO pad尺寸
 → 在官方HighResContextHead的512/128两层零初始化1×1融合
 → 官方后续卷积/BN/激活/dropout/predictor → pad尺寸插值 → 裁工作尺寸
 → 全部帧输出；计分时FP32恢复原生
```

模块位于research_tools/xyz_pilot/change_relation_model.py；change_relation.py为独立启动/保存/评价入口。
立即caller是新shell脚本。旧Direct模型只有浅logit修正，不能表达本轮双层特征融合和变化比较，因此新增独立模块；未改旧模型、官方源码或实验产物。
不实例化DINO/MoGe，不使用旧训练C/R或旧XYZ末次权重。旧detail32未用于新解码，z0只用于参照和预览。

|条件|分组|局部变化槽|全局连接|
|---|---|---|---|
|A_CURRENT_XY|14像素XY格|仅自身m=Va、差分0、dt0|每帧分别两层，时间位置为零|
|B_FULL_XY|同一XY格|全视频PTS选择参考帧|全片原始块双向|
|C_FULL_XYZ|原scale/h等尺寸XYZ立方块|同B|同B|

三组同构、相同可训练参数3,346,949：官方head2,094,017，其余新增模块1,252,932。
A不会退回直接调用旧二维预测，仍训练相同空间关系、融合、块辅助头和定位头。A未使用的跨帧Q/K/时间参数可能无梯度，是结构干预预期，不添加无效损失。
B−A只检验整套时空变化关系；C−B检验组织及几何编码方案，不能分离纯深度数值贡献。

## 坐标、有效性、分组与稀疏C

C保存为每帧[Nt,128]列表，另有points/index/keys、centers、counts、PTS与帧号；不是巨型稠密C张量。

- XYZ直接读原缓存points/keys/index；provider XYZ仅经既有读取路径除scale=12.880994319915771一次。h=.5、origin0，keys=floor(XYZ_normalized/.5)。输入绝对归一化XYZ、相对XYZ_normalized/.5−keys−.5；中心(keys+.5)*.5。
- XY：gx=floor(x/14)、gy=floor(y/14)；key=(gx,gy,0)。输入绝对((x+.5)/W,(y+.5)/H,0)，相对((x+.5)/14−gx−.5,(y+.5)/14−gy−.5,0)。位置中心为((gx+.5)*14/W,(gy+.5)*14/H,0)。最后部分格仍以完整14网格中心定义，中心可以略超工作边界，但只有真实有效工作像素参与；padding没有伪造点。
- XY近邻用整数XY格坐标距离，XYZ用三维立方格坐标距离；任意展平编号不作为距离。CPU cKDTree只确定每帧最近的至多8个其他块，少于8用实际数量，单块才用自身；等距并列不代表物理关联。
- 三组共享旧几何有效像素支持，以匹配关系覆盖。XY数值编码不读XYZ，但共享几何有效性门控，因此不是完全独立于深度有效性的数据消融。视觉主路、像素loss和评价始终全画面；无效几何处关系map显式为零。
- 点数仅用于池化归一化、监督和报告，不把数量作为手工真假输入。GT从不决定分组、对应或近邻。
- 原单目几何没有真实米制标定/精确世界配准的保证，不能据共享格号宣称跟踪同一物理表面。保留尺度漂移限制，不增加共有键要求或状态分类器。

## 两轮变化关系的Q/K/V与形状

每轮独立参数，D128/4头/头32；前轮输出是后轮输入。

1. PTS偏移固定−.8/−.2/+.2/+.8秒，最近实际帧、并列选较早，超界方向丢弃，当前和重复帧去除。候选若没有有效块则不可读取；无可用参考时明确用自身槽。没有未来截止限制，B/C允许全片未来。
2. 当前块Q=Linear_q(LN(Ca)+position_a)，候选帧所有块K=Linear_k(LN(Cb)+position_b)，V=Linear_v(LN(Cb))。SDPA产生m[Nt,128]；同一个Linear_v(LN(Ca))给Va，不在不同特征空间直接相减。
3. 实际有符号dt经1→32→32时间MLP；[Va,m,m−Va,time32]共416维→256/GELU→128，得到d。
4. 用LN(d)经独立邻域Q/K、softmax学beta（实际邻居轴），Value直接用原d，得到d_bar。没有额外V投影破坏差分空间。[LN(Ca),d,d−d_bar]共384维→256/GELU→128为r槽。
5. LayerNorm/Linear标量对至多4槽softmax，聚合消息并残差更新C。槽是消息维，不把全局token扩成4M。
6. Global沿原DirectLayer：LN内容＋位置入Q/K，内容入V，4头SDPA、out投影残差、LN→128→256/GELU→128 FFN残差。位置为真实中心编码＋PTS−首PTS秒数编码；A按帧执行且时间编码项置零。

B/C全局张量[M,128]，无中继/筛token/抽帧/共有键交集；全部帧和块更新两轮。未显式保存M×M注意力或M×M×D差分。默认Query chunk=0，torch内置SDPA分派，无注意力权重；未做独立kernel profiler，所以不宣称实际Flash后端。编码/变化/全局/解码使用非重入checkpoint；循环帧及模块显式绑定，梯度不detach。

## 官方定位头与监督

官方文件：`/media/njupt/8FCD3869E2FAB06F/chenyh/external/AINPAINT-CVIU26/ainpaint/modules/context_decoder.py`。
严格加载已训练二维epoch01的全部head参数与BN buffers；不重写近似官方头。
在proj→proj_norm→proj_relu的512通道和up2后的128通道分别加Conv1×1(LN(R回投影))。
两个fusion无bias、权重零初始化。关系map先在工作尺寸回填，再右下补零至pad14尺寸，再resize到解码层；不是把工作区域拉伸覆盖padding。
后续官方up3/final_conv_block/dropout/predictor及输出插值保持，最终不再是z0＋浅修正。

head所有卷积/BN affine可训练；每次model.train()都将5个BN置eval，running_mean/var/num_batches_tracked固定。
官方dropout=.1在训练开启、评价关闭，checkpoint保留其RNG。DINO/LoRA只作为已冻结观测来源，不在本模型实例中。

L=逐监督帧平均(pixel BCE + softDice epsilon1 + .1 blockBCE)，再组成完整单视频loss，一次backward/AdamW更新。
辅助头为LN(R)→Linear128→1正常初始化；块软目标g/n，块BCE乘成员数n、除每帧有效成员总数，无块时块项0但保留像素监督。
不设“碰到GT即全块伪造”、类别权重、平滑/一致性/分类loss。40y1x9_fake[6,10)仅上下文/输出、不进入任何监督项。
第一步融合为零，但辅助头使块编码/关系立即收到梯度，预检中已经确认，不依赖第二步才接通整条块支路。

## 数据、初始化和训练预算

直接复用冻结89训练来源/178视频、19开发来源/38视频、每轮22878合法监督帧/开发5152唯一帧；94测试保护来源不参与。
原4/新增15、对象16/背景3分组沿用，来源隔离，不重新70/30或人工筛选。
原epoch01二维R0指标直接引用frozen_reference.json；不是新的epoch0训练成绩，未全开发回放R0。

官方head源：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_expanded_spatial_training_v1/epoch01.pt`。
新增模块seed20260909。公共未更新初态model SHA：`2fdb8dcf60c32fc398ef8a093be3df52676805ee654b23a1acb245db30df2cb9`；head SHA `8a921b8cd1e295006adbff34c425d0c3f62b3e562b9b135788928abfc384f8de`。
root initial.pt是在预检更新前保存的共同初态，三组均严格加载；不是丢弃更新权重，也不是旧XYZ终点。预检更新后的权重文件已删除。

AdamW默认betas/eps，head1e-5/new1e-4、wd.01、clip1；BF16前向，FP32路由/学习池化/损失/原生排序计分。
共同cosine在完成更新索引0..1779从1降至.1，直接由检查点updates精确恢复，不在恢复时重启计划。
10个完整无放回视频epoch、178视频/轮，每组1780有效更新/228780合法监督帧呈现。共同orders.json预先保存，不套用MumPy步数。
每轮全开发评价及共同32目标/8来源观察。观察目标只限定计分，输入仍是其自身完整视频，同视频一次关系前向。
AP正GT fake帧→视频→来源等权，原生FP32；固定0.5的Dice/IoU/召回、real FPR/fake背景FPR及面积；空GT不补AP。
末轮10主结果，各自新增15开发AP、同分取早选择仅作附表。开发已被多次查看，不称最终测试。

## 缓存与工程范围

源：`derived/pvst_fullvideo_matched_localization_v1/`的只读visual/geometry/protocol。
现有VideoCache/WorkVideo读取patch、原路由/坐标和GT，不运行旧prepare、MoGe、DINO或DFormer；旧detail/z0虽由复合包读取，但不复制整库、不用于新可训练中间层。无新C/R缓存，全部在线保留梯度。

预检发现CPU三元组unique使XY组织约9.4–9.8秒/96帧；优化为pack=gx*ceil(H/14)+gy的一维unique，并恢复同一字典序keys/index。
不是改分组或重排token：全部96真实帧逐项torch.equal核对keys和index完全一致，部分边缘/无效支持CPU契约也通过。
同病例CPU读取0.355秒、组织0.206秒、合计0.561秒；不同缓存热度的短测不当稳定吞吐。
因此不需要复制一套大量中间观测/无关RGB/detail缓存。

GPU证据来自优化前完全相同的三元组路由定义；最终仅做上述精确整数排序优化，以及入口的状态/配置/预检身份保护。
保留smoke内execution_identity.json、execution_model.py.txt、execution_entry.py.txt和xy_optimization.json，未回填原GPU执行身份。
在本新实验尚无训练检查点的前提下，仅登记最终源码摘要到其自身active identity；preflight_finalization.json逐项记录旧/新摘要和原因，preflight_delivery.json绑定最终模型和配置。
没有改旧实验identity，不提供忽略源码开关。后续源码/配置不匹配会明确停止。

## 短预检证据

9项CPU检查通过：路由/GT独立性、PTS双向/边界、8邻域、软目标展开BCE等价、两个融合尺度/padding/官方头完全一致、BN模式、A阻断/B跨帧依赖、checkpoint梯度一致性和排除监督、恢复/JSON报告接口。
另有CPU模拟的完整10轮driver回归：1780更新、228780曝光、日志kwargs、末次余弦lr和checkpoint epoch11/position0均通过；此为mock调度，不是训练或效果证据。

真实GPU固定6个病例条件、每个2步，共12次可丢弃完整视频更新；三低病例各一次完整上下文恢复后推理，仅frame0计分做接口检查。
此外三组各两次固定首帧解码用于零融合与官方头参照。没有完整开发推理，没有正式训练。

结果：`derived/pvst_ainpaint_change_relation_v1/smoke/1795699295143043679/result.json`。
日志：新根`invocations/20261126T212132_888630.log`，Python与tee退出0；预检共约111.68秒。
三低病例的适配头/官方头、适配头/缓存z0最大绝对差全部0（相同BF16前向，FP32比较）。
所有病例BN buffers未变、完整模型/AdamW/RNG检查点严格恢复，丢弃权重已删除。
第一步encoder/rounds/aux均有限非零，证明辅助监督到达联合编码与关系；第二步两个fusion及head等主要生效路径梯度有限非零。A时间路径未使用属预期。

以下表格为实际短测，IO包含读取与CPU组织；主网计时含编码、关系、所有帧解码、反向和更新，不包含在线DINO（本轮没有运行DINO）。冷/热两步不能冒充完整训练吞吐。

|病例|条件|T|M|合法帧|步|前向秒|反向秒|更新秒|总秒|IO秒|allocated/reserved GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|
|4ym5n4_fake|A_CURRENT_XY|96|43008|96|1|0.702|1.609|0.035|2.393|9.815|0.437/0.820|
|4ym5n4_fake|A_CURRENT_XY|96|43008|96|2|0.659|1.405|0.001|2.072|9.815|0.463/0.832|
|4ym5n4_fake|B_FULL_XY|96|43008|96|1|0.879|1.899|0.002|2.791|9.447|0.453/0.766|
|4ym5n4_fake|B_FULL_XY|96|43008|96|2|0.793|1.907|0.001|2.710|9.447|0.480/0.777|
|4ym5n4_fake|C_FULL_XYZ|96|735|96|1|0.870|2.702|0.002|3.583|0.287|0.398/0.678|
|4ym5n4_fake|C_FULL_XYZ|96|735|96|2|0.802|2.684|0.001|3.496|0.287|0.423/0.688|
|4ro5o4_fake|C_FULL_XYZ|125|518262|125|1|2.853|9.223|0.002|12.088|2.476|4.393/5.062|
|4ro5o4_fake|C_FULL_XYZ|125|518262|125|2|2.825|9.231|0.001|12.067|2.476|4.415/5.074|
|4nyk27_fake|C_FULL_XYZ|240|960|240|1|2.200|7.414|0.006|9.642|3.019|0.445/0.818|
|4nyk27_fake|C_FULL_XYZ|240|960|240|2|2.135|7.513|0.001|9.665|3.019|0.470/0.828|
|48o3k9_fake|C_FULL_XYZ|210|840|210|1|2.000|7.476|0.002|9.492|3.074|0.509/0.801|
|48o3k9_fake|C_FULL_XYZ|210|840|210|2|2.051|7.482|0.001|9.549|3.074|0.534/0.811|

真实最大M来自原queue_extrema：4ro5o4_fake，T125，Nt范围3082–4851，M518262，未用旧约5万块病例冒充。
最长4nyk27_fake是240帧/M960；最大工作像素总量48o3k9_fake是210帧、256×384、M840。这些均完整输入，没有截帧/丢块/Query降级；不能据6个短病例直接声称全部队列每步都已资源验收。
低XY的M43008和低XYZ的M735均为实际分组，不用旧约747数字代填。

## 启动与恢复

工作区：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen`。
结果根：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_ainpaint_change_relation_v1/`。

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen
bash scripts/run_pvst_ainpaint_change_relation_v1.sh
```

依次A→B→C，不并发争抢GPU。tmux：

```bash
tmux new-session -s pvst-change 'cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen && bash scripts/run_pvst_ainpaint_change_relation_v1.sh'
```

中断恢复（从最近完成视频边界；半个视频重做，恢复权重/优化器/RNG与位置）：

```bash
bash scripts/run_pvst_ainpaint_change_relation_v1.sh --resume
```

只运行指定一组；其他组不启动：

```bash
bash scripts/run_pvst_ainpaint_change_relation_v1.sh --condition B_FULL_XY
# 已有该组检查点时
bash scripts/run_pvst_ainpaint_change_relation_v1.sh --condition B_FULL_XY --resume
```

纯CPU报告（不导入模型或读取检查点）：

```bash
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -m research_tools.xyz_pilot.change_relation_report --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_ainpaint_change_relation_v1
```

wrapper的`--report-only`也不实例化模型/前向，但上面独立CPU命令完全不导入torch。`--plan-only`仅计划；本轮最后已执行并退出0。

每视频更新原子last.pt含全部模型、AdamW、RNG、epoch/视频位置/records；每轮epochXX.pt、dev/monitor原生JSON/CSV、末次/开发选择、来源差值和54固定预览自主生成。
快照的模型、优化器和学习率位置相互匹配。已完成条件跳过；未完成权重需要显式resume，身份不符停止，不无限重试。
日志python -u＋tee、PIPESTATUS[0]保存Python真实退出码；实时输出条件、轮/视频/step、T/M、三项loss、lr、耗时、峰值显存及最近20步训练ETA；评价另输出每视频进度。
异常保存failure/退出状态，原错误不清理，计算与Git发布状态分开。

按优化后低病例IO＋第二步总时长外推，三组纯训练合计约4.8小时，但低病例只有96帧，C高M成本更高，不能当完整队列实测。
考虑实际更长视频、逐轮38视频原生AP、16视频全上下文训练观察、检查点与IO，建议为三组完整任务预留约10–18小时甚至更多；首轮实测进度为准，不保证吞吐或用最坏一步简单乘全部预算。

本轮新增骨干/MoGe/DFormer/视频解码均0；只复用观测并在短预检中真实计算新C/R及可训练head，不能称新增模型计算0。
没有全队列CPU导出，未占用一套重复视觉缓存；原大缓存留原目录，只读。新初值/日志/短预检/预测留新结果根，不入Git。

最终状态：READY_FOR_USER_TRAINING，正式A/B/C训练目录均尚不存在。待用户启动；不持续轮询，不自动开正式训练或追加模型。
