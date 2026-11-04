# PVST v0.3 SliceBackward取证与原生split修复

**候选结果：OOM；正式模型0/2。** 原实现完整诊断1次，修复后完整验收至多1次；无正式训练/性能指标。

**最终结果：原1.42GiB申请已原生定位；三处split修复与正确性核验通过；唯一候选最大窗口仍OOM，AdamW更新0次。**

原诊断明确为 **SliceBackward0 → slice_backward → zeros**，不是仅凭“与[N,D]大小相符”推断。小图在256/512维、FP32/BF16四组中，目标完整节点slice_backward由 **40/64次降至0**；合法的参数权重切片未移除。

新失败已转为 **ToCopyBackward0 → at::native::to/_to_copy → empty_strided/allocator**，申请 **364MiB** 时设备剩 **255.38MiB**，已不是本次修复的slice零缓冲路径。此为设备/精度拷贝反向分配，但C++栈未打印该ToCopy节点的输入shape、source/target device及dtype；不能只凭364MiB认定CPU→GPU传回或某个head。最后活动记录为coarse dilation1的head0稀疏attention重算，不能当作失败节点身份。原生完整栈留新数据目录，不追加第三次窗口/新profiler。

候选前向 **22.230秒**；反向未完成，从前向阶段结束至worker落盘末态约 **40.155秒**（含错误符号化/收尾，不是完整反向耗时）；梯度检查/裁剪、AdamW和优化器状态建立均未到达。worker末态 **65.640秒**，父进程总墙时 **66.862秒**。

**真实退出历史：**原诊断returncode=2；候选已捕获CUDA OOM并保存worker_result，随后进程最终 **returncode=-6，SIGABRT(6)**，不是正常退出2或0，且不是父进程超时终止。发生abort的进一步机制本轮未取证，不改写原始退出记录。primary计算结果保留OOM，原生退出信号另列；本任务GPU进程已释放。

候选已观测allocated/reserved峰值 **21.110/21.791GiB**，不能作为完整更新峰值上界；主机峰值RSS **17.865GiB**。split并未降低本组单层allocated峰值，但反向时间由0.863425降至0.493071秒（约42.9%缩短）；前向0.213057→0.237282秒。不得拿两次失败墙时计算完整更新加速比。

**对4090的判断：当前仍未通过一次真实更新，不应直接训练。原重复切片问题已确证并修复，剩余限制是ToCopy反向的显存分配；本轮停止，不自动修改拷贝、SpMM、索引、尺度或其他存储。**

## 已确认的申请来源

原HEAD `51d0f298cd4aeadcb1944d7ef3308d0f149974f0`、相同最大窗口/初始化/开关，仅启动前添加TORCH_SHOW_CPP_STACKTRACES=1，没有allocator history或全程profiler。原配置再次OOM，1.42GiB申请来自 **SliceBackward0 → slice_backward_wrapper → at::native::slice_backward → zeros → CUDA allocator malloc**。不是只根据申请量或最近FFN记录猜测。
关键原生栈（完整证据留diagnostic/worker_result.json）：
```
#19 at::native::zeros_symint(c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from ??:0
#24 at::native::slice_backward(at::Tensor const&, c10::ArrayRef<long>, long, long, long, long) from ??:0
#34 torch::autograd::generated::details::slice_backward_wrapper(at::Tensor const&, c10::ArrayRef<c10::SymInt> const&, long, std::optional<c10::SymInt>, std::optional<c10::SymInt>, c10::SymInt) from :0
#35 torch::autograd::generated::SliceBackward0::apply(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) from ??:0
```
确切slice的input_sizes/dtype未被C++异常文本打印，不能唯一指定fine/coarse或FFN/QKV/投影中的哪一个调用；两尺度完整BF16节点特征均约1.42GiB，属于形状线索。没有内存snapshot，驻留GPU张量不能逐个归类；已记录allocated/reserved、last trace、CPU head payload与当前head QKV shape，不把形状估算当精确驻留清单。
对应安装PyTorch2.5.1+cu124的官方v2.5.1源码核对：[TensorShape.cpp slice_backward](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/TensorShape.cpp#L2381-L2385)建立input_sizes零梯度后回填切片；[derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.5.1/tools/autograd/derivatives.yaml)映射slice.Tensor；[FunctionsManual.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/autograd/FunctionsManual.cpp#L1979-L2047)的split反向汇总分块梯度，仍需要完整输入梯度，并非零内存。未升级依赖。
小规模实际执行：[41,256]以11/7节点重复切片，分别4/6次完整尺寸slice_backward；一次split为SplitBackward0、无对应slice_backward。当前真实SparseLayer小图原路径共40次[41,256]反向分配，不将累计字节当作同时驻留峰值。CPU原始记录见small_original.json。
## 修复与正确性

只增加默认False的node_split，三个已有大激活入口统一一次split：streamed_head的norm+QKV、attention投影/残差的两个输入、FFN。原qkv权重行切片、像素tile等保持；无全仓替换、CopySlices预分配、自定义autograd、原地修改多输出view或输入复制。共享小方法node_parts只供这三处调用并记录实际view grad_fn。
逐head QKV、CPU head暂存、CSRSoftmax/CSRFromValues、全部节点/边/像素/时序/宽度、checkpoint/dropout/RNG、初始化/损失/参数身份和键不变。split仍可能暂存各块梯度并cat成完整输入梯度，不预填峰值减半或PASS。
原容差FP32输出2e-5/2e-4、梯度2e-5/3e-3；BF16输出.04/.03、梯度.003/.08（atol/rtol），未放宽。
|dtype/宽度|输出max abs|输入梯度max abs|参数梯度max abs|原/新全节点slice_backward次数|
|---|---:|---:|---:|---:|
|torch.float32/256|0|0|0|40/0|
|torch.float32/512|0|0|0|64/0|
|torch.bfloat16/256|0|0|0|40/0|
|torch.bfloat16/512|0|0|0|64/0|

小图完整SparseLayer、dropout开启、内外checkpoint及尾块，含全部norm/QKV/bias/projection/FFN参数梯度；参数身份/数量/state_dict与默认关闭复现核对。Dispatch检查前向/反向重算，不要求所有SliceBackward为零，仅禁止对应全节点激活的重复分配和旧[N,3D]/逻辑N×N稠密化。
## 唯一匹配单层

实际N131072/E7475256、dim512/8head。两组均逐head QKV＋CPU head暂存＋CSR桥接；仅node_split不同，同seed/输入/权重/BF16/dropout/checkpoint，独立进程一次预热后同步计时，无搜索。
|路径|前向秒|反向秒|peak allocated/reserved GiB|测量峰值RSS GiB|
|---|---:|---:|---:|---:|
|slice|0.213057|0.863425|1.5325/1.7910|1.8782|
|split|0.237282|0.493071|1.5325/1.8320|2.0131|

大图全部参数梯度核对，输出/输入梯度固定64行（每2048节点一行），不称全量输出比较；完整小图全量。实际分块view记录保留slice.json/split.json，SliceBackward/SplitBackward原生行为来自小图执行，不在大图计时中开profiler。小GPU累计墙时=12.110893541015685秒/300秒。
## 完整窗口

两次均XYZ_PVST，video782426ea0330c7fea684953c、ANAKIN/train/real/an:4wdo37，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4；fine2970629/coarse1488042。seed20260909、attention512/QKV8192/FFN8192/point32768/tile256、BF16/逐head FP32、外层CPU卸载/CPU head/逐headQKV原样，候选仅增加node_split。正式XYZ=null。

原实现诊断：**OOM**；源码`51d0f298cd4aeadcb1944d7ef3308d0f149974f0`；returncode/signal=2/None。
|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/主机可用GiB|
|---|---:|---:|---:|---:|
|worker_start|None|1.859|0.000/0.000|0.513/48.399|
|model_built|0.826927131973207|2.687|0.066/0.080|0.676/48.255|
|data_loaded|0.6201891805976629|3.309|1.635/1.713|1.291/47.707|
|forward_complete|21.910781230777502|25.221|17.491/19.723|14.744/33.877|
更新次数=0；参数改变核对=False；优化器状态参数数=未到达；worker至末态=142.14252857584506秒；父进程总墙时=144.80347159691155秒。
最后阶段=backward；错误首行=['CUDA out of memory. Tried to allocate 1.42 GiB. GPU 0 has a total capacity of 23.51 GiB of which 824.38 MiB is free. Process 3903557 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 21.49 GiB memory in use. Of the allocated memory 20.61 GiB is allocated by PyTorch, and 411.24 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)']。

修复后验收：**OOM**；源码`dc986df3daf4b62756596dd600d51cda9a4d7ac3`；returncode/signal=-6/6。
|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/主机可用GiB|
|---|---:|---:|---:|---:|
|worker_start|None|1.834|0.000/0.000|0.514/47.704|
|model_built|0.7983589414507151|2.634|0.066/0.080|0.676/47.531|
|data_loaded|0.6182875297963619|3.253|1.635/1.713|1.274/47.091|
|forward_complete|22.230397945269942|25.485|17.491/20.680|14.733/33.522|
更新次数=0；参数改变核对=False；优化器状态参数数=未到达；worker至末态=65.6397415716201秒；父进程总墙时=66.86215801071376秒。
最后阶段=backward；错误首行=['CUDA out of memory. Tried to allocate 364.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 255.38 MiB is free. Process 3903557 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 22.05 GiB memory in use. Of the allocated memory 20.90 GiB is allocated by PyTorch, and 686.87 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)']。

六层实际split入口：
```json
[
  {
    "layer": "fine_time.0",
    "nodes": 2970629,
    "edges": 37042228,
    "completed_heads": 4,
    "blocking_execution": {
      "head_QKV": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 8192,
        "last_shape": [
          5125,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_x": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 512,
        "last_shape": [
          5,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_attended": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 512,
        "last_shape": [
          5,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "FFN": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 8192,
        "last_shape": [
          5125,
          256
        ],
        "dtype": "torch.bfloat16"
      }
    }
  },
  {
    "layer": "fine_time.1",
    "nodes": 2970629,
    "edges": 31105643,
    "completed_heads": 4,
    "blocking_execution": {
      "head_QKV": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 8192,
        "last_shape": [
          5125,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_x": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 512,
        "last_shape": [
          5,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_attended": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 512,
        "last_shape": [
          5,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "FFN": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 8192,
        "last_shape": [
          5125,
          256
        ],
        "dtype": "torch.bfloat16"
      }
    }
  },
  {
    "layer": "fine_time.2",
    "nodes": 2970629,
    "edges": 28847555,
    "completed_heads": 4,
    "blocking_execution": {
      "head_QKV": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 8192,
        "last_shape": [
          5125,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_x": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 512,
        "last_shape": [
          5,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_attended": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 512,
        "last_shape": [
          5,
          256
        ],
        "dtype": "torch.bfloat16"
      },
      "FFN": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          2970629,
          256
        ],
        "chunk": 8192,
        "last_shape": [
          5125,
          256
        ],
        "dtype": "torch.bfloat16"
      }
    }
  },
  {
    "layer": "coarse_time.0",
    "nodes": 1488042,
    "edges": 26982715,
    "completed_heads": 8,
    "blocking_execution": {
      "head_QKV": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 8192,
        "last_shape": [
          5290,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_x": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 512,
        "last_shape": [
          170,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_attended": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 512,
        "last_shape": [
          170,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "FFN": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 8192,
        "last_shape": [
          5290,
          512
        ],
        "dtype": "torch.bfloat16"
      }
    }
  },
  {
    "layer": "coarse_time.1",
    "nodes": 1488042,
    "edges": 22781330,
    "completed_heads": 8,
    "blocking_execution": {
      "head_QKV": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 8192,
        "last_shape": [
          5290,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_x": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 512,
        "last_shape": [
          170,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_attended": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 512,
        "last_shape": [
          170,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "FFN": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 8192,
        "last_shape": [
          5290,
          512
        ],
        "dtype": "torch.bfloat16"
      }
    }
  },
  {
    "layer": "coarse_time.2",
    "nodes": 1488042,
    "edges": 20707343,
    "completed_heads": 8,
    "blocking_execution": {
      "head_QKV": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 8192,
        "last_shape": [
          5290,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_x": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 512,
        "last_shape": [
          170,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "projection_attended": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 512,
        "last_shape": [
          170,
          512
        ],
        "dtype": "torch.bfloat16"
      },
      "FFN": {
        "mode": "split",
        "view_grad_fn": "SplitBackward0",
        "input_shape": [
          1488042,
          512
        ],
        "chunk": 8192,
        "last_shape": [
          5290,
          512
        ],
        "dtype": "torch.bfloat16"
      }
    }
  }
]
```
关键梯度、内存和失败上下文：
```json
{
  "gradients": null,
  "parameter_delta_norms": null,
  "host": {
    "available_bytes": 34513686528,
    "rss_bytes": 17643102208,
    "peak_rss_bytes": 19182764032
  },
  "peak_allocated_bytes": 22667201024,
  "peak_reserved_bytes": 23397924864,
  "failure_operator_context": []
}
```

诊断仅增加C++错误栈，符号化/错误收尾可能增加成本，不能把诊断墙时当无诊断吞吐。候选验收也保留仅异常时生成的C++错误栈，不启用allocator history或全程profiler；如失败，总墙时含错误符号化/收尾成本。两个worker均原父进程900秒限时，不重试；未完成阶段不填0，失败观测峰值不冒充完整更新上界。
结论：未完成规定最大窗口一次更新，不建议直接正式训练；本轮停止，不再叠加其他优化。
错误/门槛：无
产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_split_backward_check_v1`。原资产/协议/失败日志/退出码/config和用户无关JSON保留，权重丢弃无checkpoint，正式模型0/2；计算与Git发布分列。

11项CPU直接相关检查通过（5.27秒）；CUDA四组小图输出/输入/全部参数梯度最大绝对误差均0，大图固定64行输出/输入及全部参数梯度最大误差1.907349e-6，满足既定容差。单层实测peak allocated两组均1.5325GiB，reserved分别1.7910/1.8320GiB，主机测量期峰值RSS1.8782/2.0131GiB。无全队列、前端、特征重提取或正式训练。原资产/config/旧退出历史及用户JSON经SHA核对未变。
