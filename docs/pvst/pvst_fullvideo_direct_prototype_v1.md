# PVST完整视频原始块原型 v1

## 完成范围与身份

本轮已实现独立模型、冻结观测准备及资源入口，实跑完整低token训练来源病例。中高病例未启动；没有正式训练、开发集效果比较或持续监控。
实际发生4次可丢弃AdamW优化更新，不能称为“完全未训练”。没有保存预检权重。

实测执行提交：`4384cfddd3387b602bf91b275eaac7151924bfda`；最终发布提交见Git回执。实测后仅补充终端失败阶段/配置/墙钟记录、原结果防覆盖及显式观测路径参数，模型数学不变；这些记录逻辑经过CPU回归测试，没有因此重复GPU预检。

仓库：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen`；分支`pvst-fullvideo-direct-prototype-v1`。根契约§17.12、ADR-0055及AGENTS登记本轮独立授权。

## 实现与旧模型的区别

入口：`research_tools.xyz_pilot.fullvideo_direct_probe`；模型：`fullvideo_direct_model.FullVideoDirect`。旧入口和官方源码不改。

1. 严格复用`ainpaint_spatial.build/load_adaptation`恢复扩充空间epoch01；官方通用DINO、LoRA和原head全部冻结、eval/no_grad。其通用权重SHA和外部两个仓库revision与检查点记录核对。
2. 同一次原`SpatialBaseline.forward`通过局部`token_grid`包装捕获patch网格，predictor的pre-hook捕获其真实输入detail32，原前向返回工作z0。没有另写近似定位头或额外第二次视觉前向；作用域结束恢复包装并移除hook。
3. DINO padding按实际RGB重新计算：工作216×384→224×392；没有误用DFormer的224×384。patch实际1024×16×28、FP32；detail实际32×128×224、BF16；z0为216×384、FP32。缓存按实际输出dtype保存。
4. 原视频自身`xyz_provider`只除scale=12.880994319915771一次；FP32按h=.5、origin0三轴floor，保留全部有效点/非空块、keys/points/index及帧切片/真实PTS。单目相机坐标不是世界配准或米制体积。没有GT路由/共有键要求/新MoGe。
5. 可训练patch 1024→128/GELU，插值至DINO pad后裁工作网格；沿旧公式用绝对XYZ和块内相对位置生成gamma/beta，逐点MLP、单头FP32 softmax池化得到C128。没有假四级金字塔、旧XYZ训练权重或成熟邻域3D编码。
6. 拼接完整96帧原始块；两层D128/4头、FF256、dropout0。Q/K包含可学习块中心和距首帧的实际PTS编码，V为归一化内容。FULL_VIDEO原始块直接双向SDPA；FRAME_ONLY同一参数逐帧计算。没有中继、筛块、因果断言或只取末帧R。
7. 各帧R→32回填；冻结detail32先插值至对应DINO pad再裁工作网格；concat64→Conv3×3(32)/GN4/GELU→Conv1×1(1)形成delta。末层weight/bias为0，z=z0+delta。冻结二维与logit修正是本次具体选择，不是用户思想必须如此。
8. 所有帧输出；无效几何context零，视觉仍保留。没有帧分类、全块均值分类或mask门控。合法帧全画面BCE均值＋soft Dice(eps1)，再按合法帧平均，全片一次backward/step。40y1x9_fake[6,10)保留上下文/输出但不监督。

按帧非重入checkpoint编码/解码，循环帧输入用默认参数显式绑定；关系层checkpoint，不detach跨帧表示。冻结观测在CPU按需送GPU；完成准备后释放原DINO/head再测主网络。默认不分Query，本次chunk=0；可手动选择唯一备用1024，每个Q块仍读取全部K/V并完整归一化，不产生M×M bias/mask或返回注意力权重。

新增可训练模型参数：481,954。两模式同一初值hash：`0f6e601206cd848859d412e62e4ce73fb6f7ffb3d072bf9db487857e13b564e2`；各自新AdamW(lr1e-4/wd.01/clip1)，seed20260909，BF16前向/FP32池化及loss；原二维optimizer不使用。

## 小病例实测

病例`4ym5n4_fake`，完整96帧、合法监督96帧；每帧7–8块，全片**735块**，有效点帧次合计7,962,624。旧747块来自另一尺度，仅用于选病例，未拿来充当本轮结果。

|模式|步|loss|前向秒|反向秒|AdamW秒|峰值allocated GiB|峰值reserved GiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|FRAME_ONLY|1|0.410608709|0.5771|1.7783|0.0185|0.3757|0.6465|
|FRAME_ONLY|2|0.410593212|0.4373|1.6220|0.0005|0.3793|0.6504|
|FULL_VIDEO|1|0.410608709|0.3415|1.3449|0.0011|0.3757|0.6621|
|FULL_VIDEO|2|0.410593301|0.3394|1.3468|0.0005|0.3793|0.6621|

计时含必要CUDA同步；前向含联合块编码、全片关系、全部帧定位与全片loss，反向含checkpoint重算，optimizer单列。CPU profiler收集实际算子派发，会产生额外开销。第1步包含冷启动，两步不是稳定吞吐；两模式不据此作速度优劣结论。
主网络不包含在线DINO，不能把上述数值称为包含在线DINO的端到端成本。关系结构VJP额外单列，FRAME_ONLY约.2302秒、FULL_VIDEO约.00327秒，不混入loss反向时间。

冻结视觉准备：**96次真实原空间前向**，8.6016秒（含模型恢复/身份核对/冻结核验及保存），缓存495,868,800字节，约472.90MiB；新增MoGe=0。xyz_provider来自旧观测包，DFormer视觉输出不作为本原型输入。
主网络完整前向4次、loss backward 4次、AdamW step 4次；另各模式一次关系依赖VJP。checkpoint导致内部编码/解码/关系重算，以上前向数指顶层模型调用，不把重算说成零计算。CPU单测的合成张量调用另计，不是实际视频推理。
正确环境调用从日志命名04:20:46至退出文件时标04:21:48，墙钟约62秒（文件时标近似，含CPU路由、哈希、导入与记录）；不等于纯GPU累计。未来入口会直接记录monotonic总墙钟。

### 验收证据

|检查|FRAME_ONLY|FULL_VIDEO|
|---|---:|---:|
|初始化全部帧z−缓存z0最大绝对差|0.0|0.0|
|首帧R对后续块C的梯度最大绝对值|0.000000000|0.513935983|
|完整帧/点/非空块、全部工作输出尺寸|PASS|PASS|
|第2步visual/geometry/point/score/value/layers/pixel_context/correction有限非零梯度|PASS|PASS|
|第2步块编码/关系/修正参数实际变化|PASS|PASS|
|优化更新次数/退出码|2 / 0|2 / 0|

首步上游梯度确为0、末层有梯度，符合零初始化链路；AdamW对有零梯度张量仍可能weight decay，不能用首步参数变化冒充上游学习证据。第2步才确认全部有效路径。冻结DINO/LoRA/head含BN的state hash前后相同；主网络测量时原视觉模型已释放，冻结缓存只读。
关系依赖检查直接作用在R和C上：FULL_VIDEO首帧可读取后续块，FRAME_ONLY跨帧依赖为0；不利用初始delta=0的logit来判断时间路径。这不是顺序理解、泛化或伪造判别能力证明。
原生恢复约定为工作logit先FP32，再bilinear(align_corners=False)恢复各帧native_hw。本次检查全部工作输出尺寸及首帧恢复形状；没有开发集计分、AP或效果比较。
实际profiler派发含`aten::_scaled_dot_product_flash_attention`及其backward（PyTorch2.5.1+cu124、RTX4090）；这是本病例实际派发证据，不是根据开关猜测。没有安装独立FlashAttention。中高病例后端以其日志实录为准。

## 失败记录与产物位置

首次执行`ec9569a`未带原实验依赖overlay，构建前出现`ModuleNotFoundError: peft`，0次视觉提取、0次优化更新，退出1。其错误记录打印又暴露同名stage参数错误，已修复并加回归测试。
原空间run.sh已使用`envs/v8-ainpaint-packages`及`envs/rgbd-packages`，新脚本现在原样加入这两个现有路径、XFORMERS_DISABLED=1等设置，没有安装新包。首次数据和日志不覆盖；正确环境结果另存：

- 首次失败：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/cases/4ym5n4_fake/result.json`
- 首次日志：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/invocations/20261124T041920_999463.log`（Python退出1）
- 成功程序报告：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/report.md`
- 成功汇总：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/summary.json`
- 成功逐病例：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/cases/4ym5n4_fake/result.json`
- 成功逐步CSV：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/measurements.csv`
- 成功日志：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/invocations/20261124T042046_1003636.log`（Python与tee均退出0）
- 冻结缓存：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/observations/4ym5n4_fake`

适配检查点SHA256：`3987d1bfb4ac9f233a73e14223ec4e8a0913b2d203ca7312926de20d22380598`；源manifest/provider配置、视频/mask、官方权重/revision、完整PTS与尺寸写入缓存identity.json。
根目录原report/summary保留首次失败，不误读为成功结果；本轮成功入口见上述original_environment_run及交付索引README。没有预检权重或全量预测提交Git。

## 中高病例：待用户终端启动

未提取这两条的新DINO缓存、未测其真实块数与显存；不能宣称全片资源全面通过。脚本顺序处理，不等GPU、不自动重试。已有同身份PASS病例复用；已有失败/不同配置拒绝覆盖。

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen
scripts/run_pvst_fullvideo_direct_probe_v1.sh \
  --cases 40y1x9_fake 9mygy9_fake \
  --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run
```

可选tmux启动（不依赖Codex会话）：

```bash
tmux new-session -s pvst-fullvideo 'cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen && bash scripts/run_pvst_fullvideo_direct_probe_v1.sh --cases 40y1x9_fake 9mygy9_fake --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run'
```

终端实时显示病例/阶段/帧/T/M/模式/step/loss/lr/耗时/显存，tee同步写invocations时间戳日志，保存Python实际退出码。每病例JSON、逐步CSV和程序报告自动导出，无需Codex盯进程。
若默认精确SDPA发生资源失败，先保留失败证据；用户决定是否另开显式输出目录用固定`--query-chunk 1024`，可用`--observations /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_direct_prototype_v1/original_environment_run/observations`复用身份一致冻结观测。本轮没有自动执行该备用路径，也没有截视频、增大h或删块。

## 检查与结论边界

8项CPU契约测试：旧MHA/SDPA输出和输入/参数梯度；精确Query分块；未来依赖；初始化与排除监督；checkpoint逐帧绑定与梯度；单尺度路由/无效视觉旁路；失败报告；旧结果不被新失败覆盖。另做bash -n、Python编译和git diff --check。
小病例证明该具体结构在完整96帧上可运行、连通且能更新；没有证明学习充分或任何定位收益。735块不能代表约2万/5万块病例，更不能代表全89来源。二维冻结、logit修正、单尺度与简洁逐点池化都是本次实现选择。
计算终态：小病例PASS、4次预检更新已丢弃、退出0、短任务已结束；中高待用户启动。Git发布与计算状态分开，发布HEAD及远端核对见最终回执。到此停止，不启动正式训练或持续监控。
