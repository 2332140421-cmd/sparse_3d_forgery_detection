# PVST v0.3 逐head即时QKV执行验收

**状态：OOM；正式模型0/2。** 无正式训练、前端提取或效果评价。
实现：保留原qkv Linear及参数身份，当前head可微选择Q/K/V各64行，合为192行权重；每8192节点仍对完整D维执行原FP32累积LayerNorm，再F.linear。当前head分块归一化/投影和CSR attention处于同一非重入checkpoint，节点块另用原checkpoint；head作为显式参数传入，重计算不读取循环尾值。
只物化当前[N,192]，不在head间缓存QKV、不复制完整x、不创建新的Parameter。当前head输出沿用同步CPU暂存/CPU stack/完整attended回GPU，保留CSRSoftmax、CSRFromValues、SIGFPE填充列、BF16外层/逐head FP32、dropout及外层saved-tensor CPU卸载。新开关默认False，只允许CSR＋CPU head路径，不静默回退；正式XYZ仍null。
## 正确性

沿用FP32输出atol2e-5/rtol2e-4、梯度2e-5/3e-3，BF16输出.04/.03、梯度.003/.08，不放宽。小图全量比较输出/输入/全部参数梯度；参数身份、键和数量不变；各head原qkv参数行梯度范数非零。CPU单独检查完整D LayerNorm及Q/K/V选行/排列、尾块、关闭路径与默认值。
|精度/宽度/dropout|输出max abs|输入梯度max abs|全部参数梯度max abs|候选[N,3D]事件数|
|---|---:|---:|---:|---:|
|torch.float32/256/0.1|0|1.1641532e-10|2.3283064e-10|0|
|torch.float32/512/0.1|4.7683716e-07|1.1641532e-10|3.4924597e-10|0|
|torch.bfloat16/256/0.1|0|3.8146973e-06|1.0490417e-05|0|
|torch.bfloat16/512/0.1|0|3.8146973e-06|5.364418e-06|0|

TorchDispatch覆盖前向、完整反向、内外checkpoint重算，禁止[N,3D]/[N,3,H,64]激活、_to_dense和逻辑N×N；允许正常N×D特征与[3D,D]参数。小图含负坐标、空帧、非均匀PTS和尾块。实际最大绝对/相对参考尺度及逐参数误差保留JSON。
## 匹配单层

两组均当前CSR＋桥接＋CPU head输出暂存，仅head_qkv_stream不同；实际N131072/E7475256、dim512/8head/64。同seed/输入/权重/dropout.1/BF16/checkpoint，独立进程各一次预热后同步计时，无profiler/多规模搜索。
|路径|前向秒|反向秒|peak allocated/reserved GiB|测量峰值RSS GiB|实际QKV形状/dtype|
|---|---:|---:|---:|---:|---|
|full|0.183609|0.781172|2.2167/2.5137|1.9841|[131072, 1536]/torch.bfloat16|
|stream|0.222360|0.867027|1.5325/1.9277|1.9819|[131072, 192]/torch.bfloat16|

大图配对：全部参数梯度；输出/输入梯度固定每2048节点一行64行，不称全量输出比较。小图均逐元素核验。QKV实际形状/字节、前向与重算记录、起始allocated及主机资源保留机器结果。小型GPU累计墙时=12.620467348955572秒/300秒。
## 唯一完整窗口

进入次数=1；现有父进程900秒worker限时，不重试。
状态 **OOM**。XYZ_PVST、ANAKIN/train/real/an:4wdo37、video782426ea0330c7fea684953c，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4。实际fine/coarse=2970629/1488042。
seed20260909；原AdamW/初始化/真实8帧均值frame BCE、AN real合法全零GT pixel BCE/softDice；attention512/QKV8192/FFN8192/point32768/tile256；CPU saved-tensor卸载、CPU head暂存和head_qkv_stream均True，BF16/逐head FP32保持。17,821,672参数及原state_dict键hash由worker核对。
|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/主机可用 GiB|
|---|---:|---:|---:|---:|
|worker_start|None|1.881|0.000/0.000|0.514/47.996|
|model_built|0.8385280109941959|2.721|0.066/0.080|0.676/47.831|
|data_loaded|0.6144744781777263|3.336|1.635/1.713|1.276/47.307|
|forward_complete|21.779683273285627|25.117|17.491/19.723|14.767/34.590|

|层|N/E|完成head|当前head QKV形状/bytes|完整[N,3D]|CPU head payload GiB|
|---|---:|---:|---|---|---:|
|fine_time.0|2970629/37042228|4|[2970629, 192]/1140721536|False|1.4165|
|fine_time.1|2970629/31105643|4|[2970629, 192]/1140721536|False|1.4165|
|fine_time.2|2970629/28847555|4|[2970629, 192]/1140721536|False|1.4165|
|coarse_time.0|1488042/26982715|8|[1488042, 192]/571408128|False|1.4191|
|coarse_time.1|1488042/22781330|8|[1488042, 192]/571408128|False|1.4191|
|coarse_time.2|1488042/20707343|8|[1488042, 192]/571408128|False|1.4191|

更新次数=0；真实参数改变=False；AdamW状态参数数=未到达。
worker至末态=141.69727745093405秒，父进程总墙时=144.29580196924508秒；returncode/signal=2/None；GPU释放=True。
末态资源/关键梯度/失败上下文：
```json
{
  "phase": "backward",
  "peak_allocated_bytes": 22668686848,
  "peak_reserved_bytes": 23400022016,
  "host": {
    "available_bytes": 36405739520,
    "rss_bytes": 16202473472,
    "peak_rss_bytes": 19242926080
  },
  "gradients": null,
  "parameter_delta_norms": null,
  "failure_operator_context": [],
  "error": "CUDA out of memory. Tried to allocate 1.42 GiB. GPU 0 has a total capacity of 23.51 GiB of which 861.50 MiB is free. Process 3903557 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 21.45 GiB memory in use. Of the allocated memory 20.61 GiB is allocated by PyTorch, and 371.24 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)"
}
```
最后阶段=backward；完整调用栈、当前层/head/QKV shape及5秒采样保留full_window/worker_result.json、live.json、worker.log。未完成耗时不填0；失败时已观测峰值不代表完整更新峰值上界。
上轮全量QKV＋CPU head暂存同窗口：前向20.641秒，反向coarse第三层QKV cat重算OOM，申请4.25GiB时余625.31MiB；峰值allocated/reserved20.876/21.811GiB，AdamW0。旧反向未完成，不计算精确完整加速比，不从单层直接外推整模型。

结论：尚未完成规定窗口一次真实更新，不建议直接训练；按本轮停止条件结束，不追加SpMM/索引/尺度/其他优化。
错误/门槛：无
源码：`8ceedb3ccf1fcf3a755991ce042fcf588273e6a9`；产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_head_qkv_stream_check_v1`。冻结缓存复用，无下载/前端/视觉重提取、坐标审计/预览或效果指标，验收权重丢弃无checkpoint。原报告、协议、退出码、config和用户无关JSON保留，正式模型0/2；计算与发布分列。
