# PVST v0.3 固定CSR values梯度桥接验收

2026-11-05；状态 **OOM**；正式模型仍0/2，不启动训练或性能评价。
## 修复与结构

原直接CSR构造的SparseCompressedTensorBackward0会_to_dense，再提取values梯度。新增CSRFromValues只封装固定结构构造：A[row_e,col_e]=v_e；其反向从真实CSR上游梯度返回对应边的grad_values。shape/crow/col均核对，不凭nnz假定顺序；CUDA结构核对用异步断言，避免每head主机同步。非空结构改变明确报错，不静默匹配或稠密化；本路径实测CSR结构完全一致。
None梯度保留None；空CSR上游代表数学零梯度，明确返回原边上的零。支持空边/空行、单边、显式零值及无梯度输入，不把dropout后的零值边删除。没有改写SpMM、sampled_addmm或整套attention反向，只有CSR构造values的一阶桥接。
现有CSRSoftmax保持不变；sampled_addmm输出仍直接可微values()，完整路径证明无需额外取值桥接。最终路径：CSR sampled_addmm → values+bias → CSRSoftmax → 原dropout → CSRFromValues → CSR sparse.mm。保留逐head FP32、外层BF16、SIGFPE无边填充列、全部邻接/节点/点/像素/宽度/PTS、初始化及损失。正式XYZ=null，候选仍默认关闭。
CSRSoftmax仅保存p/crow；桥接仅保存crow/col引用和逻辑shape。临时行号及算术张量仍有成本；del局部变量不意味着autograd保存内容消失，节省以下列实测allocated为准。无显式COO构造，不承诺后端内部完全无稀疏布局转换。
## 小图正确性与布局

沿用已有固定容差：FP32输出2e-5/2e-4，梯度2e-5/3e-3；BF16输出.04/.03，梯度.003/.08（atol/rtol）。桥接FP64对照1e-10/1e-9；gradcheck eps1e-6/atol1e-5/rtol1e-3。未事后放宽。
桥接与SpMM：
```json
{
  "incoming_gradient": {
    "layout": "torch.sparse_csr",
    "crow": [
      0,
      2,
      2,
      3,
      5
    ],
    "col": [
      0,
      2,
      1,
      0,
      3
    ]
  },
  "gradcheck": true,
  "output_max_abs": 0.0,
  "values_grad_max_abs": 3.469446951953614e-18,
  "V_grad_max_abs": 0.0,
  "empty_edges_rows": true,
  "explicit_zero_edges": true,
  "padding_columns": 2
}
```
|精度/dim/dilation|输出max abs|输入梯度max abs|QKV梯度max abs|bias梯度max abs|投影梯度max abs|
|---|---:|---:|---:|---:|---:|
|torch.float32/256/1|4.7683716e-07|1.1641532e-10|8.1490725e-10|2.0954758e-09|6.9849193e-10|
|torch.float32/256/2|4.7683716e-07|1.1641532e-10|9.3132257e-10|2.5029294e-09|5.8207661e-10|
|torch.float32/256/4|4.7683716e-07|1.1641532e-10|5.8207661e-10|3.259629e-09|6.9849193e-10|
|torch.bfloat16/512/1|0.03125|5.7220459e-06|7.6293945e-06|3.8258731e-06|1.5258789e-05|

小图包含非均匀PTS、负坐标、不同帧节点/空帧、跨帧query与尾块；FP32覆盖三种dilation，BF16覆盖512/8head。TorchDispatch检查前向、完整反向及checkpoint重计算，拒绝_to_dense和逻辑N×N/N×(N+padding)张量，同时允许正常N×D特征。普通与强制逻辑padding两条小图均检查；没有重做旧SIGFPE定位。
布局通过：True；padding布局通过：True；dropout/checkpoint：{"dropout": 0.1, "max_output_error": 0.0, "finite_gradients": true, "checkpoint_rng": "PASS"}。
全部参数误差及相对参考最大幅度保留JSON；CPU直接相关检查见cpu_tests.txt。无新增模型参数/state_dict键。
## 唯一规模单层内存与速度

实际131,072节点、dim512、8head、head_dim64，四帧真实已验证坐标图/非均匀PTS；旧COO候选仅作tests中的本次参考。两者独立进程、同seed/权重/输入/dropout.1/BF16/FP32/内外checkpoint，一次预热后各一次非profiler同步计时。
|实现|实际N/E|前向秒|反向秒|起始allocated GiB|peak allocated/reserved GiB|
|---|---:|---:|---:|---:|---:|
|old|131072/7475256|0.149002|0.837250|0.1575|2.4317/2.6699|
|new|131072/7475256|0.099172|0.653441|0.1575|2.2792/2.4941|

大图一致性：全部参数梯度比较；输出及输入梯度比较固定每2048节点一行，共64行，不冒称大图逐元素全量。小图输出/输入/参数梯度均全量核对。配对误差：
```json
{
  "output_rows": {
    "max_abs": 0.0078125,
    "reference_max_abs": 4.125,
    "relative_to_max": 0.001893939393939394
  },
  "input_gradient_rows": {
    "max_abs": 4.656612873077393e-10,
    "reference_max_abs": 1.2293457984924316e-07,
    "relative_to_max": 0.003787878787878788
  },
  "norm.weight": {
    "max_abs": 3.4924596548080444e-09,
    "reference_max_abs": 2.3449960281141102e-05,
    "relative_to_max": 0.00014893243369869355
  },
  "norm.bias": {
    "max_abs": 4.103640094399452e-09,
    "reference_max_abs": 0.0002966150059364736,
    "relative_to_max": 1.3834903872929254e-05
  },
  "qkv.weight": {
    "max_abs": 1.7881393432617188e-07,
    "reference_max_abs": 1.537799835205078e-05,
    "relative_to_max": 0.011627906976744186
  },
  "qkv.bias": {
    "max_abs": 9.5367431640625e-07,
    "reference_max_abs": 0.00061798095703125,
    "relative_to_max": 0.0015432098765432098
  },
  "bias.0.weight": {
    "max_abs": 2.9802322387695312e-08,
    "reference_max_abs": 1.8596649169921875e-05,
    "relative_to_max": 0.0016025641025641025
  },
  "bias.0.bias": {
    "max_abs": 2.9802322387695312e-08,
    "reference_max_abs": 1.4483928680419922e-05,
    "relative_to_max": 0.00205761316872428
  },
  "bias.2.weight": {
    "max_abs": 2.384185791015625e-07,
    "reference_max_abs": 6.389617919921875e-05,
    "relative_to_max": 0.0037313432835820895
  },
  "bias.2.bias": {
    "max_abs": 2.2268068278208375e-08,
    "reference_max_abs": 6.28642737865448e-08,
    "relative_to_max": 0.35422453703703705
  },
  "proj.weight": {
    "max_abs": 3.5762786865234375e-07,
    "reference_max_abs": 5.53131103515625e-05,
    "relative_to_max": 0.00646551724137931
  },
  "proj.bias": {
    "max_abs": 7.62939453125e-06,
    "reference_max_abs": 0.00103759765625,
    "relative_to_max": 0.007352941176470588
  },
  "norm2.weight": {
    "max_abs": 2.444721758365631e-09,
    "reference_max_abs": 0.00031940004555508494,
    "relative_to_max": 7.654105853732589e-06
  },
  "norm2.bias": {
    "max_abs": 2.153683453798294e-09,
    "reference_max_abs": 0.0002965993480756879,
    "relative_to_max": 7.26125484688761e-06
  },
  "ffn.0.weight": {
    "max_abs": 4.76837158203125e-07,
    "reference_max_abs": 8.106231689453125e-05,
    "relative_to_max": 0.0058823529411764705
  },
  "ffn.0.bias": {
    "max_abs": 9.5367431640625e-07,
    "reference_max_abs": 0.0002498626708984375,
    "relative_to_max": 0.003816793893129771
  },
  "ffn.3.weight": {
    "max_abs": 1.9073486328125e-06,
    "reference_max_abs": 0.0002803802490234375,
    "relative_to_max": 0.006802721088435374
  },
  "ffn.3.bias": {
    "max_abs": 2.384185791015625e-07,
    "reference_max_abs": 0.0010833740234375,
    "relative_to_max": 0.00022007042253521127
  }
}
```
放行门槛在执行前固定：peak allocated实际下降、反向≤旧1.25倍，且全部正确性/layout通过。小型GPU子进程累计墙时11.557574870996177秒/290秒（为预先极小接口探针留约10秒预算，探针墙时未单列）。
## 完整窗口

进入次数：1；唯一900秒父进程限时，不重试。
状态 **OOM**；ANAKIN/train/real/an:4wdo37，video782426ea0330c7fea684953c；720×1280，历史[104,120)、目标[120,128)，fine1/8、coarse1/4。
实际fine/coarse：2970629/1488042；参数17,821,672及state_dict键hash由入口核对；8帧原生pixel输出、原frame BCE+AN real全零空间GT的pixel BCE/Dice损失，仅作执行检查。
|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|当前allocated/reserved GiB|RSS/主机可用GiB|
|---|---:|---:|---:|---:|---:|
|worker_start|None|1.866|0.000/0.000|0.000/0.000|0.513/48.304|
|model_built|0.8372179046273232|2.704|0.066/0.080|0.066/0.080|0.676/48.094|
|data_loaded|0.6304891966283321|3.336|1.635/1.713|1.598/1.713|1.267/47.603|
|forward_complete|16.767977717332542|20.105|19.623/21.818|13.037/20.172|14.565/34.578|

六层运行证据（全部通过才允许PASS）：
```json
[
  {
    "layer": "fine_time.0",
    "width": 256,
    "dilation": 1,
    "enabled": true,
    "head_source_sha256": "f02e95451f0267eb9e2fde333e6126542c719821d11f540c61b1a922ba09c061",
    "nodes": 2970629,
    "edges": 37042228,
    "average_degree": 12.46948979492222,
    "topology_bytes": 616440688,
    "backend": "CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head",
    "softmax_impl": "research_tools.pvst.sparse_attention.CSRSoftmax",
    "values_bridge": "research_tools.pvst.sparse_attention.CSRFromValues",
    "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
    "logical_padding_columns": 0,
    "completed_heads": 4
  },
  {
    "layer": "fine_time.1",
    "width": 256,
    "dilation": 2,
    "enabled": true,
    "head_source_sha256": "f02e95451f0267eb9e2fde333e6126542c719821d11f540c61b1a922ba09c061",
    "nodes": 2970629,
    "edges": 31105643,
    "average_degree": 10.471062862444283,
    "topology_bytes": 521455328,
    "backend": "CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head",
    "softmax_impl": "research_tools.pvst.sparse_attention.CSRSoftmax",
    "values_bridge": "research_tools.pvst.sparse_attention.CSRFromValues",
    "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
    "logical_padding_columns": 0,
    "completed_heads": 4
  },
  {
    "layer": "fine_time.2",
    "width": 256,
    "dilation": 4,
    "enabled": true,
    "head_source_sha256": "f02e95451f0267eb9e2fde333e6126542c719821d11f540c61b1a922ba09c061",
    "nodes": 2970629,
    "edges": 28847555,
    "average_degree": 9.710924858001453,
    "topology_bytes": 485325920,
    "backend": "CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head",
    "softmax_impl": "research_tools.pvst.sparse_attention.CSRSoftmax",
    "values_bridge": "research_tools.pvst.sparse_attention.CSRFromValues",
    "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
    "logical_padding_columns": 0,
    "completed_heads": 4
  },
  {
    "layer": "coarse_time.0",
    "width": 512,
    "dilation": 1,
    "enabled": true,
    "head_source_sha256": "f02e95451f0267eb9e2fde333e6126542c719821d11f540c61b1a922ba09c061",
    "nodes": 1488042,
    "edges": 26982715,
    "average_degree": 18.13303320739603,
    "topology_bytes": 443627784,
    "backend": "CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head",
    "softmax_impl": "research_tools.pvst.sparse_attention.CSRSoftmax",
    "values_bridge": "research_tools.pvst.sparse_attention.CSRFromValues",
    "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
    "logical_padding_columns": 0,
    "completed_heads": 8
  },
  {
    "layer": "coarse_time.1",
    "width": 512,
    "dilation": 2,
    "enabled": true,
    "head_source_sha256": "f02e95451f0267eb9e2fde333e6126542c719821d11f540c61b1a922ba09c061",
    "nodes": 1488042,
    "edges": 22781330,
    "average_degree": 15.309601476302417,
    "topology_bytes": 376405624,
    "backend": "CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head",
    "softmax_impl": "research_tools.pvst.sparse_attention.CSRSoftmax",
    "values_bridge": "research_tools.pvst.sparse_attention.CSRFromValues",
    "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
    "logical_padding_columns": 0,
    "completed_heads": 8
  },
  {
    "layer": "coarse_time.2",
    "width": 512,
    "dilation": 4,
    "enabled": true,
    "head_source_sha256": "f02e95451f0267eb9e2fde333e6126542c719821d11f540c61b1a922ba09c061",
    "nodes": 1488042,
    "edges": 20707343,
    "average_degree": 13.915832348818112,
    "topology_bytes": 343221832,
    "backend": "CSR sampled_addmm/CSRSoftmax/CSRFromValues/CSR mm FP32 per head",
    "softmax_impl": "research_tools.pvst.sparse_attention.CSRSoftmax",
    "values_bridge": "research_tools.pvst.sparse_attention.CSRFromValues",
    "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
    "logical_padding_columns": 0,
    "completed_heads": 8
  }
]
```
梯度/更新/失败上下文：
```json
{
  "gradients": null,
  "optimizer_state_parameters": null,
  "parameter_delta_norms": null,
  "parameter_update_verified": null,
  "failure_operator_context": [
    {
      "function": "sparse_head",
      "head": 6,
      "q_shape": [
        1488042,
        64
      ],
      "edges": 20707343,
      "dtype": "torch.bfloat16"
    }
  ],
  "phase": "backward",
  "elapsed_s": 63.92112171743065,
  "host": {
    "available_bytes": 37036830720,
    "rss_bytes": 15905071104,
    "peak_rss_bytes": 15899230208
  },
  "peak_allocated_bytes": 22735592448,
  "peak_reserved_bytes": 23427284992
}
```
实际更新次数：0；worker正常末态墙时：63.92112171743065；父进程总墙时：65.49554091691971；returncode/signal：2/None。
GPU开始/结束及释放：
```json
{
  "gpu_start": {
    "unix_time": 1793813038.4072278,
    "worker_pid": null,
    "gpu_total_free_used_util": "24564, 22850, 1223, 9",
    "compute_pid_mib": "3903557, 507"
  },
  "gpu_after": {
    "unix_time": 1793813103.9429748,
    "worker_pid": 4103717,
    "gpu_total_free_used_util": "24564, 22850, 1223, 7",
    "compute_pid_mib": "3903557, 507"
  },
  "worker_gpu_released": true
}
```
最后阶段：backward。失败记录：
```
Traceback (most recent call last):
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/pvst/max_window_check.py", line 169, in worker
    phase='backward';tick=time.monotonic();loss.backward();torch.cuda.synchronize();boundary('backward_complete',time.monotonic()-tick)
                                           ^^^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/utils/checkpoint.py", line 1125, in unpack_hook
    frame.recompute_fn(*args)
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/utils/checkpoint.py", line 1519, in recompute_fn
    fn(*args, **kwargs)
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/pvst/sparse_attention.py", line 130, in sparse_head
    result=torch.sparse.mm(matrix,vf);emit('sparse_mm_complete',result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 364.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 298.12 MiB is free. Process 3903557 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 21.99 GiB memory in use. Of the allocated memory 20.96 GiB is allocated by PyTorch, and 578.56 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```
旧同窗口COO候选前向18.064秒、反向未完成，约65.56秒OOM、AdamW未执行，已观测allocated峰值21.25GiB。新旧都逐head FP32核心，但softmax计算次序不同，不保证训练逐位相同；不把旧失败时间当完整成本算精确加速比，不从单层直接外推整模型。

不建议直接进入两条件各一轮。未达到完整更新放行条件；按本轮停止条件结束，不继续叠加优化或重试。已观测峰值不代表完整更新峰值上界。
错误/门槛：无
源码：`a6199da86044401bbaa551a377ea800d8ee64ea6`；产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_csr_values_bridge_check_v1`。缓存/前端/视觉特征均复用，不读测试集，不下载，不做坐标审计或预览；验收权重与优化器丢弃，无checkpoint。原报告、退出历史、正式config和用户无关JSON保持；计算与publication分列。
