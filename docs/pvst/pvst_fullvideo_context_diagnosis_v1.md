# 完整视频固定权重context诊断：脚本交付

状态：**脚本已交付，完整诊断尚未启动**。2026-11-25。独立分支`pvst-fullvideo-context-diagnosis-v1`，起点`57665d9a0318f75ad6338017f6f449bc5cc24b79`。只扩展诊断入口，不改变§17.12/17.13的模型、数据、训练或旧结果。

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

## 完整运行将生成（目前未生成）

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

脚本已交付，完整诊断尚未启动。Git发布状态以交付回执为准。
