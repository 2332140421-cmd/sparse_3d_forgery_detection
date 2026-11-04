# PVST v0.3 逐head输出CPU暂存资源验收

状态：**OOM**；正式模型仍0/2。本轮只有存储路径变化，无训练、评价或前端/冻结视觉提取。
新增默认关闭head_output_offload：GPU逐head原CSRSoftmax/CSRFromValues/SpMM及checkpoint → 保留autograd的同步to(cpu) → CPU按原head顺序stack/reshape → 完整attended一次同步回GPU。dtype不变，不用detach/no_grad，不改QKV/索引/初始化/损失/宽度/像素/点/节点/边/上下文。CPU列表没有GPU输出引用；成员变量只记标量/字符串，不记Tensor。
原已验证外层saved-tensor CPU卸载、BF16/逐head FP32、SIGFPE padding、dropout和checkpoint RNG保持。正式config XYZ=null且无新开关字段，正式默认False。

**最终结论：正确性和匹配单层通过；唯一完整窗口OOM，AdamW更新0次。暂不建议在当前4090上进入两条件各一轮。**

本次明确失败于反向的外层checkpoint重计算：`Temporal.apply → SparseLayer.forward → projected=torch.cat(parts)`。保存的活动层是 **coarse_time第3层（dilation4，512维，N=1,488,042）**。此时尚在norm+QKV拼接，**不对应某个正在执行的head**；上轮coarse第7head SpMM不是本次报错位置。拟拼接shape为 **[1,488,042,1,536]、BF16，约4.257 GiB**，由节点数和QKV宽度推导；cat未成功，不冒称已有该输出Tensor。

CUDA报错申请 **4.25 GiB** 时仅余 **625.31 MiB**；错误瞬间PyTorch allocated20.87 GiB、reserved未分配344.85 MiB。已观测allocated/reserved峰值 **20.876/21.811 GiB**。完整更新未完成，这不是完整更新峰值上界。

前向 **20.641秒**；反向未完成，从前向阶段结束至worker末态约 **45.609秒**（含异常处理/收尾，不能当完整反向耗时）；梯度检查/裁剪、AdamW及优化器状态建立均未到达。worker至末态 **69.647秒**，父进程总墙时 **71.573秒**，真实退出码 **2**，无终止信号，非TIMEOUT。主机进程峰值RSS **17.572 GiB**，末态RSS **16.225 GiB**、主机可用 **33.044 GiB**。

## 正确性

沿用FP32输出atol2e-5/rtol2e-4、梯度atol2e-5/rtol3e-3；BF16输出.04/.03、梯度.003/.08，不事后放宽。小图含非均匀PTS、空帧、负坐标和尾块，当前CSR＋桥接GPU拼接与CPU暂存逐项全量对比。
|dtype/宽度/dropout|输出max abs|输入梯度max abs|全部参数梯度最大误差|
|---|---:|---:|---:|
|torch.float32/256/0|0.0|0.0|0.0|
|torch.float32/256/0.1|0.0|0.0|0.0|
|torch.bfloat16/512/0.1|0.0|0.0|0.0|

参数量/state_dict键与权重保持；CPU测试核对head顺序/reshape、默认关闭与全层开关接入。CUDA检查CPU拷贝requires_grad=True、ToCopyBackward0、原dtype，输入/QKV/bias/投影/FFN全部梯度存在且正确；dropout开启时内外checkpoint复算与RNG一致。TorchDispatch拒绝_to_dense及逻辑N×N矩阵。
## 匹配单层

同已验证131072节点/7475256边、512维/8head/64维head图与PTS。GPU拼接和CPU暂存使用相同CSR＋桥接、seed/权重/输入/dropout.1/BF16/内外checkpoint；独立进程、各一次预热与同步计时。没有使用旧COO作为对照，没有多规模或chunk搜索。
|路径|前向秒|反向秒|peak allocated/reserved GiB|测量期峰值RSS GiB|实际CPU head payload MiB|stack设备|
|---|---:|---:|---:|---:|---:|---|
|gpu|0.101050|0.656945|2.2792/2.4941|1.3911|0.0|cuda:0|
|cpu|0.155992|0.773836|2.2323/2.5723|1.8934|128.0|cpu|

head_storage中CPU字节数为实际拷贝Tensor的numel×element_size累计，不能与CPU进程峰值RSS等同；CPU heads和CPU stack曾短暂同时存在。D2H/CPU stack/H2D耗时、每head后allocated/reserved和回传前后内存保留JSON，只记数值，不增加额外同步或逐head落盘。

实际存储证据：匹配单层GPU拼接的head列表累计128 MiB；CPU路径列表GPU字节为0，CPU payload128 MiB。GPU路径各head后allocated依次增加16 MiB；CPU路径八次均为694,027,776字节不增长，最后相差恰好128 MiB。CPU stack完成后GPU allocated仍为694,027,776，完整attended回传后变为828,245,504字节（增加128 MiB）。这些是实际阶段记录，不只凭调用.cpu()或形状估算。

整层allocated峰值只下降 **48 MiB（2.06%）**，不是128 MiB；因为全局峰值还有QKV和稀疏/反向临时量。reserved反而2.494→2.572 GiB，因此不以reserved变化宣称节省。匹配单层测量期峰值RSS约1.391→1.893 GiB；包含报告快照后的进程峰值另在JSON保留。

大图配对核对全部参数梯度，输出/输入梯度取固定每2048节点一行共64行；小图逐元素全量，未将大图抽样称全量。配对误差：
```json
{
  "output_rows": {
    "max_abs": 0.0,
    "reference_max_abs": 4.125,
    "relative_to_max": 0.0
  },
  "input_gradient_rows": {
    "max_abs": 0.0,
    "reference_max_abs": 1.2293457984924316e-07,
    "relative_to_max": 0.0
  },
  "norm.weight": {
    "max_abs": 1.278749550692737e-09,
    "reference_max_abs": 2.345000575587619e-05,
    "relative_to_max": 5.4530884299348335e-05
  },
  "norm.bias": {
    "max_abs": 1.775333657860756e-09,
    "reference_max_abs": 0.0002966161700896919,
    "relative_to_max": 5.985289532003343e-06
  },
  "qkv.weight": {
    "max_abs": 1.1920928955078125e-07,
    "reference_max_abs": 1.519918441772461e-05,
    "relative_to_max": 0.00784313725490196
  },
  "qkv.bias": {
    "max_abs": 4.76837158203125e-07,
    "reference_max_abs": 0.00061798095703125,
    "relative_to_max": 0.0007716049382716049
  },
  "bias.0.weight": {
    "max_abs": 5.960464477539063e-08,
    "reference_max_abs": 1.8596649169921875e-05,
    "relative_to_max": 0.003205128205128205
  },
  "bias.0.bias": {
    "max_abs": 1.4901161193847656e-08,
    "reference_max_abs": 1.4483928680419922e-05,
    "relative_to_max": 0.00102880658436214
  },
  "bias.2.weight": {
    "max_abs": 5.960464477539063e-08,
    "reference_max_abs": 6.389617919921875e-05,
    "relative_to_max": 0.0009328358208955224
  },
  "bias.2.bias": {
    "max_abs": 8.149072527885437e-09,
    "reference_max_abs": 6.100162863731384e-08,
    "relative_to_max": 0.13358778625954199
  },
  "proj.weight": {
    "max_abs": 2.384185791015625e-07,
    "reference_max_abs": 5.53131103515625e-05,
    "relative_to_max": 0.004310344827586207
  },
  "proj.bias": {
    "max_abs": 0.0,
    "reference_max_abs": 0.00103759765625,
    "relative_to_max": 0.0
  },
  "norm2.weight": {
    "max_abs": 7.421476766467094e-10,
    "reference_max_abs": 0.00031939923064783216,
    "relative_to_max": 2.323573776747751e-06
  },
  "norm2.bias": {
    "max_abs": 8.87666828930378e-10,
    "reference_max_abs": 0.0002966000174637884,
    "relative_to_max": 2.992807743306193e-06
  },
  "ffn.0.weight": {
    "max_abs": 4.76837158203125e-07,
    "reference_max_abs": 8.106231689453125e-05,
    "relative_to_max": 0.0058823529411764705
  },
  "ffn.0.bias": {
    "max_abs": 1.1920928955078125e-07,
    "reference_max_abs": 0.0002498626708984375,
    "relative_to_max": 0.00047709923664122136
  },
  "ffn.3.weight": {
    "max_abs": 9.5367431640625e-07,
    "reference_max_abs": 0.0002803802490234375,
    "relative_to_max": 0.003401360544217687
  },
  "ffn.3.bias": {
    "max_abs": 0.0,
    "reference_max_abs": 0.0010833740234375,
    "relative_to_max": 0.0
  }
}
```
放行要求：正确性/layout及配对一致、真实CPU列表/CPU stack/dtype/梯度连接成立，无OOM；测量期峰值RSS相对GPU路径增加不超过预先记录1GiB安全界（payload128MiB、CPU stack另128MiB），超出则停止。若整层allocated峰值未下降，结合每head与拼接阶段解释，不以.cpu()调用冒充节省，也不另加其他优化。
小型GPU子进程累计墙时：11.35923863016069秒，限300秒。
## 唯一完整窗口

进入次数：1；现有父进程900秒限时，不重试。
状态 **OOM**。XYZ_PVST，ANAKIN/train/real/an:4wdo37，video782426ea0330c7fea684953c；720×1280；history[104,120)、target[120,128)，fine1/8/coarse1/4；实际fine/coarse=2970629/1488042。
配置seed20260909/BF16/逐head FP32；attention512/QKV8192/FFN8192/point32768/tile256；外层saved-tensor CPU卸载及head_output_offload均True。参数17,821,672与state_dict键hash核对；原8目标帧均值frame BCE和本ANAKIN real合法全零GT pixel BCE/softDice。只用于执行检查，不报告效果指标。
|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|当前allocated/reserved GiB|RSS/主机可用 GiB|
|---|---:|---:|---:|---:|---:|
|worker_start|None|1.991|0.000/0.000|0.000/0.000|0.514/48.017|
|model_built|0.7850326234474778|2.777|0.066/0.080|0.066/0.080|0.676/47.875|
|data_loaded|0.6178060239180923|3.396|1.635/1.713|1.598/1.713|1.282/47.383|
|forward_complete|20.640938526950777|24.039|19.623/21.811|13.037/21.598|14.588/34.789|

|层|dilation|N/E|CPU head payload GiB|stack设备|D2H/stack/H2D秒|回GPU前后allocated GiB|
|---|---:|---:|---:|---|---:|---:|
|fine_time.0|1|2970629/37042228|1.4165|cpu|0.4767/0.1247/0.1155|14.4393/15.8558|
|fine_time.1|2|2970629/31105643|1.4165|cpu|0.4910/0.1395/0.1167|14.3523/15.7688|
|fine_time.2|4|2970629/28847555|1.4165|cpu|0.4893/0.1281/0.1163|14.3201/15.7366|
|coarse_time.0|1|1488042/26982715|1.4191|cpu|0.5072/0.1354/0.1149|15.7663/17.1854|
|coarse_time.1|2|1488042/22781330|1.4191|cpu|0.5032/0.1411/0.1215|15.7096/17.1287|
|coarse_time.2|4|1488042/20707343|1.4191|cpu|0.6248/0.1594/0.1336|15.6845/17.1036|

梯度、更新及失败上下文：
```json
{
  "gradients": null,
  "parameter_update_verified": null,
  "optimizer_state_parameters": null,
  "parameter_delta_norms": null,
  "failure_operator_context": [],
  "error": "CUDA out of memory. Tried to allocate 4.25 GiB. GPU 0 has a total capacity of 23.51 GiB of which 625.31 MiB is free. Process 3903557 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 21.68 GiB memory in use. Of the allocated memory 20.87 GiB is allocated by PyTorch, and 344.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)",
  "phase": "backward",
  "elapsed_s": 69.6474349591881,
  "host": {
    "available_bytes": 35480539136,
    "rss_bytes": 17421811712,
    "peak_rss_bytes": 18867466240
  },
  "peak_allocated_bytes": 22415039488,
  "peak_reserved_bytes": 23418896384
}
```
更新次数=0；父进程总墙时=71.57329954579473秒；returncode/signal=2/None；GPU进程释放=True。
最后阶段=backward。未完成阶段不填0；失败已观测峰值不当作完整更新峰值上界。完整错误和最后层/shape见full_window/worker_result.json、live.json与worker.log，5秒GPU采样/曲线保留独立数据目录。
上次当前CSR＋桥接同窗口前向16.768秒、反向coarse dilation4第7head SpMM重算OOM，申请364MiB时剩298.12MiB；allocated/reserved峰值21.174/21.818GiB，AdamW0。旧反向未完成，不计算精确完整加速比；head输出1.06GiB等先验形状数不冒充实测释放。

## 结论

未完成完整更新，不建议直接进入两条件各一轮。按本轮限定停止，不自动改SpMM、索引类型、尺度或其他存储，不追加训练或资源测试。
错误/门槛：无
执行源码：`7791399153899d20cb50f7d4ecfa92846459e4b2`。本轮产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_head_output_offload_check_v1`。原报告/错误/退出码/协议/config和缓存不覆盖；用户无关JSON保留。验收权重丢弃，不保存checkpoint；计算状态与publication.json单列。

本轮8项直接相关CPU检查通过（3.39秒）。CUDA小图FP32/dropout关闭、FP32/dropout开启、BF16/dropout开启三组输出/输入/全部参数梯度最大误差均0；大图固定64行输出/输入梯度误差0，全部参数梯度最大误差9.536743e-7，满足冻结容差。

六层前向CPU head payload每fine层1.4165 GiB、每coarse层1.4191 GiB，累计 **8.5068 GiB**。这是实际CPU Tensor字节的累计传输量，**不是8.51 GiB同时驻留，也不是进程峰值RSS**；每层CPU stack另分配同等字节，短暂与heads重叠后释放。

六层前向阻塞D2H调用累计 **3.092秒**，CPU stack **0.828秒**，H2D **0.719秒**。D2H阻塞时间含等待此前排队GPU计算，不等于纯PCIe传输耗时，不据此计算带宽。匹配单层的前向和反向均变慢，完整前向较上次16.768秒增至20.641秒；两次失败点不同，不能计算精确完整更新加速/减速比。

**对4090的判断：CPU暂存有效移除了head输出累计GPU驻留，但剩余阻塞已是coarse QKV重算拼接的4.25GiB申请；当前配置仍未通过一次真实更新。保持停止，不自动实施QKV、SpMM、索引、尺度或其他修改。**

完整调用栈与资源曲线位于`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_head_output_offload_check_v1/full_window`。原报告、退出历史、config和用户JSON均经SHA核对未改动。
