# 完整视频固定权重context诊断：脚本交付

状态：**完整诊断已完成，退出码0；CPU结果分析已补齐**。2026-11-25。独立分支`pvst-fullvideo-context-diagnosis-v1`，起点`57665d9a0318f75ad6338017f6f449bc5cc24b79`。只扩展诊断入口，不改变§17.12/17.13的模型、数据、训练或旧结果。

## 问题、最小实现与边界

问题是总修正有多大、读出对完整context有多敏感、四视频训练观察高分是否已有于z0。唯一新模块`research_tools/xyz_pilot/fullvideo_context_diagnosis.py`由同名shell脚本调用，输出逐帧/视频/来源指标、分区幅度/翻转、回放与自动报告。旧训练入口会准备数据或写原状态，不能直接用于本诊断；薄入口只复用原FullVideoDirect、VideoCache、native_mask、指标及汇总。

每模式独立严格加载`FRAME_ONLY/epoch03.pt`和`FULL_VIDEO/epoch03.pt`：epoch3、position178、534更新、原runtime identity以及对应`epoch03_dev.json`的完整state hash均须一致。优化器/RNG虽在原检查点文件中，不恢复、不使用。当前HEAD不同不构成模型失配：比较真正的计算源码、原配置、观测与模型摘要，不修改旧identity。全部参数requires_grad=False，eval/inference_mode、CUDA BF16/FP32原生计分，无优化器。

每条视频只编码一次并计算一次关系，C/R不跨权重共享。逐帧适配器在原decode的correction入口用只读hook取得准确的`[detail32,context32]`，NORMAL直接调用原decode；CTX_ZERO仅将后32通道换为zeros_like，通过同一个correction加同一z0。该位置已完成pixel_context（含bias）与像素回填；不是R=0，也不是只移除注意力消息。每视频首帧用同份输入重放correction验证NORMAL严格一致，原固定54目标另重复原decode一次。两者均不重算关系或AP。

`Δ_total=NORMAL-z0`、`Δ_context=NORMAL-CTX_ZERO`、`Δ_zero_remaining=CTX_ZERO-z0`在工作网格FP32 logit统计；fake GT/背景、real全帧、fake工作空mask单列。空区域值为null，帧数/像素数仍保存。工作空mask并不排除原生正GT帧的AP。

mean/mean_abs/逐帧绝对值p95按帧→视频→来源等权，max_abs取最大；不是整库像素合池p95。翻转计数求和，同时报告分层平均比例及按区域像素总数计算的比例。context/detail在相同有效几何像素上的mean-abs、RMS、覆盖率有逐帧/视频/来源/宏表。幅度不是信息量，三种差值不是独立因果贡献，可能存在补偿。

NORMAL/CTX_ZERO在同一帧共享一次原生GT读取，连续FP32 logit恢复后AP计分，阈值固定logit≥0；沿原帧→视频→来源等权。完整报告包括全部19、原4、新增15、对象16及外观背景3，全部19来源，不合池，不把空GT AP填0/1。NORMAL回放原epoch03的分组/来源差值均输出；回放明显不一致时先查输入/权重/模式，不能自动归因或反复重跑找更近值。

观察集只从`monitor.json`的4dpnv4/4qw154各real/fake缓存读取z0与GT，在CPU用原双线性FP32恢复公式计分，再与已保存两模式各轮monitor并列。CPU恢复可有浮点差异；不运行训练观察的context网络，不引用旧356帧观察。

## CPU检查与真实短预检

8项必要CPU测试覆盖：带非零投影bias的完整context置零、原正常读出一致、同C/R路径、原生计分一致、空区域null/有向翻转、无网络的观察CPU恢复、真实finalize→JSON/CSV/report路径。测试仅用小合成张量/临时目录，非全项目测试。

已在原RTX4090/v8+既有overlay完成授权小病例：`4ym5n4_fake`全部96帧、T=96、M=735、工作216×384、原生360×640；未截帧。两模式使用各自末次已训练权重，未重初始化修正头。

|项目|FRAME_ONLY|FULL_VIDEO|
|---|---:|---:|
|视频整体预检秒（含加载/计分）|6.081389|4.285490|
|编码/关系秒|0.405668|0.195010|
|适配器NORMAL最大绝对误差|0|0|
|三固定帧重复decode最大绝对误差|0|0|
|正常context mean-abs（有效像素）|.831485|.855986|
|同支持detail mean-abs|.739800|.739800|
|C/R、参数、缓冲区未变|通过|通过|
|退出码|0|0|

全调用程序耗时10.683720秒，Python/tee退出均0；这是一次小训练病例接口预检，不能代表开发总体或稳定吞吐。其正常和置零输出均已完整计分用于验收路径，但不在此将单训练视频AP差写成研究结论。

真实调用：检查点加载2；完整视频编码/关系2；逐帧encode192；NORMAL decode192；CTX_ZERO correction192；重复NORMAL decode6；适配器correction重放2。因而原decode共198次，correction共392次。另有CPU小合成接口测试，不算开发队列推理。新增训练/优化更新、MoGe和DINO提取均0。

短预检原始实现SHA256：`9c336fb8f43f42c711f959267a895d0d42875d1ac79343265bd3e6ae24b04b45`。其后仅补充日志分段计时、汇总方向计数、元数据身份断言及已完成结果的CPU收尾兼容；模型数学/适配器/计分未改。保留原smoke identity，不回填为后来源码；交付版本再经CPU回归检查，不重复GPU预检。

预检产物：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_context_diagnosis_v1/smoke/`

预检日志：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_context_diagnosis_v1/invocations/20261125T162749_1808365.log`

## 用户终端启动与恢复

完整诊断由用户启动，Codex不启动后台任务、不持续轮询。使用原v8 Python和AINPAINT/rgbd overlay，不安装依赖。

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen
bash scripts/run_pvst_fullvideo_context_diagnosis_v1.sh
```

可选tmux：

```bash
tmux new-session -s pvst-context 'cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen && bash scripts/run_pvst_fullvideo_context_diagnosis_v1.sh'
```

中断后的同配置续跑（不是训练恢复）：

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen
bash scripts/run_pvst_fullvideo_context_diagnosis_v1.sh --resume
```

完成的逐视频JSON在全部指标/统计写好后原子提交。`--resume`核对权重、协议、配置、诊断代码及视频/观测身份，只复用完整同身份结果；未完成视频重新从头诊断，不修改旧实验，也不无限重试。已有视频结果但没给`--resume`会清楚退出。不同源码/权重不混用；需要新结果目录时用`--output /新的绝对路径`，不得放旧实验根内。smoke位于独立子目录，不充作开发完成结果。

完整队列预计新增76次完整视频编码/关系、10304次逐帧encode、10304次NORMAL decode、10304次CTX_ZERO correction、108次固定目标重复decode、76次首帧适配器correction校验、2次检查点加载（全部复用时实际加载可为0）。逐视频计数与本次调用计数分开记录，不将额外解码称为0计算。

脚本python -u，终端实时输出模式/视频/帧/T/M/关系/读出/计分/耗时，并tee日志；用PIPESTATUS[0]保留Python实际退出码，tee错误单列。文件锁拒绝并发，不等待GPU；异常留独立failure JSON与traceback，不删旧失败。用户不需要Codex在线即可完成原生计分、观察CPU基线、汇总与程序报告。

## 完整运行产物（现已生成）

根：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_context_diagnosis_v1/`

- 日志：`invocations/<时间_PID>.log`；同stem的`.exit_code`与`.launcher_exit_code`。
- 程序报告：`development/report.md`；状态与调用计数：`development/status.json`、`summary.json`。
- 逐视频恢复点：`development/<模式>/videos/<video_id>/result.json`。
- 原生定位：每模式NORMAL/CTX_ZERO的frames/videos/sources JSON/CSV；`primary.csv`、`source_metrics.csv`、`comparison.csv`、`replay_sources.csv`。
- 修正幅度/翻转：`region_frames.csv`、`region_videos.csv`、`region_sources.csv`、`region_macro.csv`；输入幅度：`input_frames.csv`、`input_videos.csv`、`input_sources.csv`、`input_macros.csv`。
- 读出误差：`decode_checks.csv`；训练观察：`monitor_z0_CPU.json`及逐层级CSV、`monitor_comparison.csv`。

默认不保存全库原生预测，不建设新预览网站。自动报告将数字幅度、对context敏感性、定位变化及z0已有观察能力分别列出，不用幅度阈值宣告故障；最终原因仍需结合回放、重复误差、翻转与逐来源指标。即使置零掉分，也不是三维或时间独立有效；即使无影响，也不能证明C无信息。本轮不训练、不追加结构或其他探针。

最终8项CPU测试、shell语法及实际元数据核对通过，交付检查记录在新根`delivery_checks.json`。旧smoke执行身份原样保留，不将后续源码摘要回填其中；完整development的续跑使用交付版本自身身份。

脚本交付时完整诊断尚未启动；随后用户终端运行已正常完成，结果分析如下。Git发布状态以交付回执为准。


## 用户完整运行后的CPU结果分析

只读取已有JSON/CSV，无新增模型前向、检查点加载、训练、深度或冻结特征提取。原程序报告、逐视频结果和日志不改。status为COMPLETE、exit_code=0，完整调用耗时808.692835秒（约13分29秒；summary测量主体807.741964秒，范围略有区别）。每模式38视频/5152帧，合计76次完整视频编码/关系、10304次encode、10304次正常decode、10304次置零correction、108次重复decode、76次适配器correction校验；加载两份末次检查点。全部逐视频记录C/R、参数与缓冲区不变，context置零验证通过。

正常回放宏AP差：FRAME_ONLY +1.07785e-7，FULL_VIDEO −5.31905e-8；宏Dice差分别−1.62889e-7、+2.70948e-7。适配器一致与固定目标重复decode的最大差均0。指标回放差远小于此处NORMAL/CTX_ZERO差值，未发现影响本轮解释的回放偏差；这不是对所有GPU数值路径的逐像素确定性证明。

### 开发19来源：同权重读出

所有指标为比例，real FPR不是百分数。NORMAL与CTX_ZERO只比较同一模式权重；不同模式仍有训练权重差异。

|模式|读出|AP|Dice|IoU|召回|real FPR|fake背景FPR|预测面积|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|FRAME_ONLY|NORMAL|.37912627|.25196772|.18541597|.26961005|.00230974|.00911582|.05927695|
|FRAME_ONLY|CTX_ZERO|.38089260|.24990184|.18415612|.26722179|.00230325|.00929484|.05993451|
|FULL_VIDEO|NORMAL|.37960891|.25052481|.18446248|.26796388|.00232365|.00904740|.05927180|
|FULL_VIDEO|CTX_ZERO|.38130772|.24839548|.18290532|.26442715|.00225645|.00917857|.05972400|

NORMAL−CTX_ZERO：FRAME_ONLY AP −.00176633、Dice +.00206588、召回 +.00238827；FULL_VIDEO AP −.00169881、Dice +.00212933、召回 +.00353672。context保留时AP方向分别9升10降、5升14降；Dice方向11升3降5平、9升5降5平（包括数值很小的变化，不冒称实质收益来源数）。两模式fake背景FPR略降、real FPR略升，有操作点取舍；不能概括为关闭context全面更好，也不能因Dice略升宣称排序改善。

对象16组NORMAL−CTX_ZERO AP分别约−.00201063、−.00201945，Dice分别+.00305485、+.00293801；没有被总体中外观/背景高分掩盖。五个既定范围与全部19来源的完整指标仍在development/primary.csv、source_metrics.csv、comparison.csv和程序报告。

FULL_VIDEO的历史四来源：猫NORMAL−ZERO AP +.00049609、Dice +.01050929、召回 +.01056183，局部有帮助但NORMAL召回仍约.396；骑马AP −.00068810、Dice +.00018611、召回−.00094743；过街两种读出Dice/召回仍0；7ogeg4 AP −.00395697、Dice−.00009480、召回−.00218650。不是多来源一致恢复。

### 总修正与context依赖：不能用AP接近代替测量

|模式|fake GT总修正mean-abs|GT context敏感性mean-abs|GT零context剩余mean-abs|context输入mean-abs|detail输入mean-abs|
|---|---:|---:|---:|---:|---:|
|FRAME_ONLY|.17158040|.18499899|.11162891|1.05603751|.78248465|
|FULL_VIDEO|.13336516|.17270990|.15981304|1.11979233|.78248465|

context/detail在同一有效几何支持计算，宏覆盖率约.917928。总修正和context敏感性可分辨地非零，明显不是本次重复decode误差（0）；但数值幅度本身不证明判别信息或收益。FULL_VIDEO GT内总修正均值−.03287431，context差均值+.10753551，零context剩余均值−.14040982：存在方向相反、均值抵消的现象，不能把总修正均值小解释成context未使用。三者不是独立因果贡献，尚不能定位某层失效或解释全部原因。

翻转统计中按全部区域像素求出的比例，与按来源等权的召回不是同一加权方式，不能直接互相推导。例如大面积来源的GT翻转方向可能与来源宏召回方向不同。保留各层级表，不合池替换主指标。

### 训练观察：高分主要已在二维起点

同一monitor四视频来自两个训练来源，不代表全训练集。

|条件|AP|Dice|召回|
|---|---:|---:|---:|
|冻结z0（CPU恢复）|.89423354|.78053928|.75270795|
|FRAME_ONLY第3轮|.89469905|.78355285|.75354900|
|FULL_VIDEO第3轮|.89528286|.78285866|.75289904|

AP相对z0增量只有+.00046551和+.00104933。高观察AP主要已经存在于冻结二维底座，不能据此证明新块关系支路充分学会定位，更不能仅凭高观察/低开发的水平差距断言新支路过拟合。CPU恢复与旧GPU可能有小数值差，未用旧356帧清单补填。

### 唯一下一步建议（仅建议，本轮不实施）

补一个从训练开始就固定context=0的DETAIL_ONLY定位修正参照：冻结相同z0/detail，训练同一个correction(detail,0)，沿用原89/19来源、公共修正头初值、视频顺序、3轮/534更新、loss和评价，和已完成的FRAME_ONLY/FULL_VIDEO作固定预算比较。它是新增一个条件，不重跑本轮诊断，不自动启动。

理由：当前已证明context参与输出，但AP没有稳定增量；推理时突然置零又改变了修正头输入分布。这个缺失参照能检验XYZ/context整条支路是否胜过等预算的纯视觉定位修正，而不是把CTX_ZERO的略高AP直接当作可部署替代。有效参数量不同应如实报告；它仍不能独立证明几何或时间的贡献。现有结果不足以指定故障层，不先调学习率、改关系结构或扩大数据。

程序报告绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_context_diagnosis_v1/development/report.md`

本轮完成CPU归纳与文档收尾；新增模型前向、权重加载、训练、MoGe、DINO提取全部0。下一轮没有实施或启动。
