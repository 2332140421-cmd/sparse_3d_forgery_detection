# 逐head即时QKV：限定资源验收

只新增默认False的head_qkv_stream。self.qkv及参数身份/state_dict保留；按原[3,heads,64]布局，从weight/bias可微选Q/K/V三个64行片段，拼当前192行小权重，每head复用该权重。节点块仍8192，每块完整D维原LayerNorm（FP32累积/原epsilon/仿射）＋F.linear，拼当前[N,192]后立刻原CSR attention。节点块checkpoint嵌在“当前head投影＋attention”的统一非重入checkpoint内；head显式传参，RNG保留，不复制x、不缓存其他head QKV。

沿用CPU head输出暂存/CPU stack/完整attended一次回GPU，以及原CSRSoftmax/CSRFromValues、逻辑padding、BF16/逐head FP32、外层saved-tensor CPU卸载、原后续投影/残差/FFN。新开关只允许CSR＋CPU head路径，无静默gather回退。正式XYZ仍null，正式模型0/2。

12项直接相关CPU检查通过，覆盖256/512、FP32/BF16、dropout内外checkpoint、尾块、完整D归一化/选行、所有参数梯度与每head QKV行梯度、参数身份及默认关闭。TorchDispatch覆盖前向/反向重计算：旧路径可观测[N,3D]；新路径禁止[N,3D]/[N,3,H,64]激活及逻辑N×N，允许[3D,D]参数和正常N×D特征。沿用原固定容差。

独立入口 `python -m research_tools.pvst.head_qkv_stream_check`：仅四组直接相关CUDA小图（两宽度×两精度，dropout开启）→唯一规模131072节点/7475256边、512/8head的匹配比较。两者均CPU head输出暂存，只有QKV开关不同。复用既有benchmark/预热/同步计时，全部参数梯度比较，大图输出/输入梯度仅固定64行，小图全量。小型GPU累计限300秒，无搜索或重试。

正确性/布局与实际head QKV形状通过后，同任务一次原max_window_check --head-qkv-stream：video782426ea0330c7fea684953c、ANAKIN/train/real/an:4wdo37，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4，节点2970629/1488042。原seed20260909、attention512/QKV8192/FFN8192/point32768/tile256、真实监督/归一化、初始化、AdamW保持，三个执行开关显式开启。父进程worker启动起900秒，前向/反向/有限梯度/裁剪/一次AdamW及真实参数改变全部完成才PASS。权重丢弃不保存，不训练。

所有新产物在derived/pvst_head_qkv_stream_check_v1。源码先提交，运行期间不改；自动中文报告、小型JSON、状态页和普通push；计算/发布分列。旧资产、协议、错误/退出记录、config及用户无关JSON只读保留。失败停止，不追加其他优化、条件或正式训练。
