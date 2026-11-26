# 官方MumPy架构在ANAKIN上的增广策略对照

状态：实现、CPU检查和真实短预检完成，退出码0；**正式10轮尚未启动，由用户终端/tmux启动**。2026-11-26。

## 问题与执行边界

只检验把官方可选SingleAugment换为默认DoubleAugment，能否改善多来源定位及后期稳定性。
不改变学习率计划、损失、初始化、架构、224输入、来源、预算或阈值；不增加XYZ/C、光流、分类或其他seed。
旧SingleAugment程序默认仍是single，旧正式结果、缓存、日志和身份不改；本次使用新分支、新配置和输出目录。
这是官方架构的ANAKIN增广策略对照，不称作者原数据及全部训练协议复现，更不证明三维/完整视频C贡献。

实际基线为已完成的SingleAugment：全部19来源首轮AP/Dice/召回0.283041/0.107967/0.109043，末轮0.207931/0.045842/0.035154；本轮未重跑旧模型。

## 官方源码核对与最小类别修正

官方checkout：`/media/njupt/8FCD3869E2FAB06F/chenyh/third_party/Mumpy`，commit `bc9280a0839f97375b8dda2dcd6d5b5052083f13`；不修改官方文件。
`args.py`默认singleAugment=False，`dataloaders/universaldataset.py`在增强开启时调用RandAugment(1,7,DoubleAugmentStrategy())；
`train_davis.sh`没有singleAugment选项，args实际augment默认True。旧适配调用OneAugmentStrategy。

`utils/randaugment.py`的m=7被保存但实际调用没有把它用于幅度缩放；本适配如实沿用。
官方Double策略独立用random.choices抽两组，每组k=1：先非形变，再形变；不自行加Identity或调概率。

|组|操作|采样与实际规则|
|---|---|---|
|非形变|HFlip、VFlip、PsccAug|各1/3；前两项幅度1，Pscc幅度uniform[0,1]但内部不用；内部numpy randint(0,7)实际仅0–6七种变换，保留原行为|
|形变|RandomCrop|1/4，v~uniform[20,220]；非空mask由其包围范围随机取外侧裁剪，空mask转OriginalRandomCrop|
|形变|RandomRotate|1/4，v~uniform[0,180]后int；沿原OpenCV仿射及中心内裁逻辑，可能截去前景|
|形变|OriginalRandomCrop|1/4，边长int(v)，v~uniform[20,220]；三个分支等概率取随机左上/右下/中心裁剪，不受GT保护，可能空裁剪|
|形变|RandomScaleCrop|1/4，v~uniform[20,220]，原base_size512、短边randint(256,1024)，保持比例；非空mask再走CornerCrop，空maskCornerCrop原样返回整张缩放图|

不把名称当数学定义：RandomScaleCrop在本输入上短边至少256、v至多220，padding分支通常不触发；它不会强制输出v×v，最后沿旧Clips整图resize224。
所有三帧使用同一个形变/裁剪参数，RGB保留官方实际插值（Rotate/OpenCV线性、scale三次，CornerCrop默认PIL RGB缩放）；
不把它改写成另一个“类似”策略。

必要mask修正在独立double_augment.py：
- RandomRotate仅mask的warpAffine改INTER_NEAREST；仿射矩阵、裁剪坐标及RGB路径不改。
- RandomScaleCrop的mask初次resize改NEAREST。
- CornerCrop裁剪后mask恢复原尺寸改NEAREST。
- Clips最后mask resize224沿用NEAREST；已有RandomCrop和OriginalRandomCrop只裁剪，其他翻转/直角变换本身保持类别。

因此会保留二值类别；修正后的mask边界也可能影响其后的官方GT包围框，这是类别插值修正的自然结果，不增加新的ROI规则。
只在训练使用目标mask引导作者裁剪。空裁剪照常训练，不丢弃、不重抽、不造标签、不调整loss权重；real空mask单列。
开发/推理Clips不构造增强器，完整224画面评价，不用开发GT决定裁剪、尺度、区域或阈值，无TTA。
GT不进入三视图Encoder或频率输入，fake没有配对real参考输入。

## 公共协议与复用

复用旧formal/protocol.json、config.json、orders.json、monitor.json和initialization.json；
新入口生成的协议/订单/观察与旧JSON逐项相等才通过，非增广训练配置必须相等。
89训练来源、178视频，每轮22878合法目标；19开发来源、38视频、5152唯一帧；94测试保护来源不读取。
40y1x9_fake[6,10)不作为监督/计分目标，但自身RGB允许上下文，如目标10取[9,10,11]。
三帧夹取边界与中帧输出不变；原4开发、新增15、对象16、背景3分组不变。
训练观察固定32目标/8来源，不当完整训练结果。

共同seed20260909；官方作者链接的通用weight.pth及全部新模块初值不变。
实际全模型初始化SHA256与旧记录一致：`ff32e887f8aa34e44fd2df3d894e25f967078e978550c9933ae5ca2c97f92a6c`。
不从旧epoch01/10继续；新三个SGD优化器与多项式调度均从新状态开始。
encoder/CVA/decoder及三视图、DWTI、globalblocks、DCT与多金字塔Decoder均沿原完整实现。

FP32；WeightedFocal(alpha=[1,1],gamma2)＋官方softIoU helper(epsilon1e-6)各1。
SGD encoder1e-3、decoder/CVA1e-2，momentum.9、wd1e-4，PolynomialLR power.9/min1e-5/no warmup，不加clip或调度变更。
micro2累积有效12，尾批6保留并按实际目标数归一化；10轮/1907更新每轮/总19070次联合更新、228780目标呈现。
每轮观察与5152完整开发，按新增15开发AP选检查点，同分取早；末轮与选择结果分列，不称独立测试。
AP沿正GT fake帧→视频→来源等权宏平均，原生FP32排序，固定logit≥0；两类FPR及fake预测面积单列。

只读缓存：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_mumpy_baseline_v1/rgb224`。
读取原npy采用mmap_mode=r，原marker身份及数组非空检查保留；缺失直接报错，不生成替代缓存。
新准备统计写新输出目录，旧目录不写入。沿用旧长边384 RGB再缩224、官方归一化；不等同原生直接224。
新增MoGe/DINO/DFormer/视频解码/权重数据下载均0。

## 保存的增广统计与复算

每轮epochXX_augmentation.json包含操作次数、增强耗时、fake原非空前后前景占比和变空次数、原空fake/real空mask计数。
占比均在实际224目标网格统计，均值及累计和/样本数同时保存，便于复算；不存海量增强图。
每目标应正好计数两项操作；恢复按已完成epoch边界，不叠加中断epoch的重复计数。

旧report逻辑复用，显式double时调用纯CPU double_comparison：
comparison/groups.csv、sources.csv、epochs.csv、double_minus_single_sources.csv、double_minus_single_groups.csv、augmentation_by_epoch.json及report.md。
比较两组首轮、last_available及各自开发选择；未到第10轮的last_available明确不当末次结果。
保留全部19及既有分组，比较AP/Dice/IoU/召回/两类FPR/面积；只读Single既有JSON，不重新推理旧模型或二维参照。
正式末次/选择均使用原54固定预览目标，概率统一0–1及0.5二值图。

## 必要检查与真实短预检

CPU共17项通过（原9项＋前轮CPU分析2项＋新增6项）：
官方策略/范围与四种形变路径、三帧同步、mask二值/对应、空mask、统计不重采样、旧Single默认、
公共非增广配置与订单一致、初始化hash拒绝不匹配、只读缓存缺失不准备、真实旧缓存只读、评价不增强、CPU比较导出。
另有bash语法和git diff检查。无全开发推理或重复旧基线。

真实预检目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_mumpy_doubleaugment_v1/smoke/1795670527117950722/`。
日志：新结果根invocations/20261126T132204_2862465.log；Python/tee退出0。
固定训练目标4ym5n4_fake/0、40y1x9_fake/10、40y1x9_real/0，各重复4次组成有效12；共2次可丢弃更新、24目标呈现。

|step|有效目标|Focal|softIoU|总loss|CPU组batch秒|前向含loss秒|反向秒|更新秒|总秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|12|.143440|.949085|1.092525|.106416|.592222|.437128|.016674|1.209548|
|2|12|.094476|.962742|1.057217|.149801|.423731|.322028|.016332|.964274|

计时在必要CUDA边界同步；总秒另含梯度核查与组织开销，两步冷/热混合不当稳定吞吐。
峰值allocated6069874688字节（5.6530GiB）、reserved6314524672字节（5.8809GiB）。
第二步各主要分支有限非零：三视图0.06198/0.06363/0.05784、Swin0.04158、CVA0.17363、CVA内部2.85e-5、global0.06897、频率解码0.05132、定位解码1.26953（梯度L2）。
第一步CVA内部因原proj_out零初值为0，第二步已非零，不改初始化。

24个目标实际抽到HFlip8/VFlip11/Pscc5、RandomCrop7/RandomRotate5/OriginalRandomCrop3/RandomScaleCrop9。
16个原非空fake平均前景占比0.028938→0.082615；增强后变空0；real空8次仍为空。这里只是小规模路径证据，不预测正式空裁剪比例。
增广本身累计0.15525秒，batch准备含IO/归一化另计。复用3视频396缓存帧、新建RGB/GT0、视频解码0、前端0。

共12次训练micro前向/反向、3次短评价前向及3次固定预览前向，实际检查点保存/严格恢复（完整模型、三SGD/三scheduler/RNG）、JSON/CSV及HTML报告收尾均通过。
预检检查点标epoch0，只在smoke目录，不是正式可评价epoch1–10；正式入口不会加载该权重。

两步按当前实测总步时粗推训练本体约20722秒（5.76小时），不含全开发原生AP、训练观察、写权重和完整队列吞吐差异。
完整运行建议预留约6–10小时甚至更多，非保证；不直接以旧5小时作保证。
正式日志每10更新输出实时loss分项、epoch/总轮、step/总step、lr、耗时和最近100步ETA，每轮输出开发指标。

少量CPU固定增强示例（非正式输入缓存）：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_mumpy_doubleaugment_v1/augmentation_examples/index.html`。
展示三帧、增强mask及原mask，种子和操作记录在manifest.json；数据图不提交Git。

## 用户启动、恢复与收尾

工作区：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen`。
新结果根：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_mumpy_doubleaugment_v1/`。
正式数据写formal/，日志invocations/，旧Single根只作为cache-root及CPU指标参照。

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen
bash scripts/run_pvst_anakin_mumpy_doubleaugment_v1.sh
```

```bash
tmux new-session -s mumpy-double 'cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen && bash scripts/run_pvst_anakin_mumpy_doubleaugment_v1.sh'
```

中断后从最近已完成epoch继续（中断epoch从该epoch开头重做，不称逐micro恢复）：

```bash
bash scripts/run_pvst_anakin_mumpy_doubleaugment_v1.sh --resume
```

仅评价新条件的末次（不重新评价旧Single）：

```bash
bash scripts/run_pvst_anakin_mumpy_doubleaugment_v1.sh --evaluate-only --checkpoint /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_mumpy_doubleaugment_v1/formal/epoch10.pt
```

只读保存结果整理程序报告与Single对照，不构建模型/读取权重或调用前向：

```bash
bash scripts/run_pvst_anakin_mumpy_doubleaugment_v1.sh --report-only
```

launcher复用原python -u/tee、原环境overlay、PIPESTATUS[0]真实Python退出码；带输出锁，失败保存failure/status，不自动无限重试。
模型/优化器/调度器/RNG及身份按原严格逻辑保存恢复，不清理旧失败或迁移旧身份。新条件code_sha256包含新增增广及CPU对照文件；改源码后不会静默恢复旧身份。
预检与正式状态分开；本次formal目录尚不存在。只提交本轮源码、测试、轻量配置/文档，媒体、缓存和权重留服务器。
完成普通push后停止，由用户启动，不进行正式训练或持续监控。
