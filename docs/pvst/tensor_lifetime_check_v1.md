# PVST v0.3 无用view清理与训练前验收

**状态：OOM；正式模型0/2，不启动训练。**
源码仅在最后使用后删除projection循环xb/ab/args、已使用的分块列表和FFN尾部part/x；没有永久清理开关，没有绕过autograd。共享view的payload不等于底层storage，去重按storage_id。原CSR、逐headQKV、CPU head、split和外层卸载全部保留。
显式转换仅增加元数据与ToCopy节点前/后hook，闭包不捕获Tensor；记录Q/K/V BF16→FP32、核心输出FP32→BF16、GPU head→CPU、CPU stack→GPU。pending_copies保存已开始但未完成的节点，不凭364MiB猜身份。checkpoint内部恢复未新增拷贝实现或自定义反向。复制事件日志上限1024，生命周期边界另限256条，并保留当前pending，非完整profiler/allocator history。
## 正确性与存活证据

使用本分支固定父提交b7ab0b8的真实SparseLayer作为只读参考，原参数/键/数量/身份、dropout/RNG、尾块、checkpoint前反向与全部梯度核对；沿用原FP32/BF16容差，未放宽。
|dtype/宽度|输出max abs|输入梯度max abs|参数梯度max abs|
|---|---:|---:|---:|
|torch.float32/256|0|0|0|
|torch.float32/512|0|0|0|
|torch.bfloat16/256|0|0|0|
|torch.bfloat16/512|0|0|0|

单层存活检查只隔离真实project_block与checkpoint；attended=x*1.5用于明确存储归属，不作为模型效果或吞吐。无gc.collect/empty_cache。
```json
{
  "status": "PASS",
  "records": [
    {
      "outer_checkpoint": false,
      "phase": "forward",
      "source": {
        "shape": [
          131072,
          512
        ],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "requires_grad": true,
        "grad_fn": "MulBackward0",
        "payload_bytes": 134217728,
        "storage_bytes": 134217728,
        "storage_id": "41484812288"
      },
      "view": {
        "shape": [
          512,
          512
        ],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "requires_grad": true,
        "grad_fn": "SplitBackward0",
        "payload_bytes": 524288,
        "storage_bytes": 134217728,
        "storage_id": "41484812288"
      },
      "allocated_before_name_delete": 423786496,
      "allocated_after_name_delete": 423786496,
      "allocated_after_view_cleanup": 423786496
    },
    {
      "outer_checkpoint": true,
      "phase": "forward",
      "source": {
        "shape": [
          131072,
          512
        ],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "requires_grad": true,
        "grad_fn": "MulBackward0",
        "payload_bytes": 134217728,
        "storage_bytes": 134217728,
        "storage_id": "41350594560"
      },
      "view": {
        "shape": [
          512,
          512
        ],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "requires_grad": true,
        "grad_fn": "SplitBackward0",
        "payload_bytes": 524288,
        "storage_bytes": 134217728,
        "storage_id": "41350594560"
      },
      "allocated_before_name_delete": 432306176,
      "allocated_after_name_delete": 432306176,
      "allocated_after_view_cleanup": 298088448
    }
  ],
  "note": "actual project_block, native splits and checkpoint; synthetic attended=x*1.5 only isolates storage ownership, not a model/performance result"
}
```
## 逐case真实更新

### max_window

**OOM**；完成更新0；optimizer.step成功返回次数0；returncode/signal=2/None。
|更新/阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/可用GiB|
|---|---:|---:|---:|---:|
|0/worker_start|None|0.886|0.000/0.000|0.513/51.184|
|0/model_built|0.7840681793168187|1.671|0.066/0.080|0.676/51.039|
|0/data_loaded|0.5869001671671867|2.259|1.635/1.713|1.264/50.536|
|1/forward_complete|21.12118191085756|23.381|17.491/19.723|14.714/36.653|
实际fine/coarse=2970629/1488042；worker至末态=63.36903218738735秒，父进程墙时=66.66513255145401秒。
每次更新loss/梯度/参数变化、pending转换及错误：
```json
{
  "updates": [],
  "pending_copies": [
    {
      "kind": "copy_backward_pre",
      "event": 697,
      "phase": "backward",
      "update_index": 1,
      "layer": "fine_time.2",
      "head": 1,
      "conversion": "head_output",
      "source": {
        "shape": [
          2970629,
          64
        ],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "requires_grad": true,
        "grad_fn": "ToCopyBackward0",
        "payload_bytes": 380240512,
        "storage_bytes": 380240512,
        "storage_id": "19362817024"
      },
      "target": {
        "shape": [
          2970629,
          64
        ],
        "dtype": "torch.bfloat16",
        "device": "cpu",
        "requires_grad": true,
        "grad_fn": "ToCopyBackward0",
        "payload_bytes": 380240512,
        "storage_bytes": 380240512,
        "storage_id": "123977161498688"
      },
      "copy_id": "fine_time.2:1:head_output_after:62"
    }
  ],
  "error_first_line": [
    "CUDA out of memory. Tried to allocate 122.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 54.19 MiB is free. Including non-PyTorch memory, this process has 22.77 GiB memory in use. Of the allocated memory 21.55 GiB is allocated by PyTorch, and 763.14 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)"
  ],
  "native_frames": [
    "#13 at::native::empty_cuda(c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>) from ??:0",
    "#20 at::native::empty_like(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, std::optional<c10::MemoryFormat>) from ??:0",
    "#21 at::native::layer_norm_cuda(at::Tensor const&, c10::ArrayRef<long>, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, double) from ??:0",
    "#28 at::native::layer_norm_symint(at::Tensor const&, c10::ArrayRef<c10::SymInt>, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, double, bool) from ??:0",
    "#60 torch::autograd::generated::AmaxBackward0::apply(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) from ??:0"
  ],
  "host": {
    "available_bytes": 37530648576,
    "rss_bytes": 17642135552,
    "peak_rss_bytes": 19176407040
  }
}
```

## 结论边界

最大窗口固定XYZ video782426ea0330c7fea684953c、ANAKIN/train/real/an:4wdo37、720×1280、history[104,120)、target[120,128)、fine1/8/coarse1/4，节点2970629/1488042。seed20260909、BF16/逐head FP32、attention512/QKV8192/FFN8192/point32768/tile256，真实8帧均值frame BCE及合法AN real零GT pixel BCE/Dice不变。
同一最大窗口最多两次正常zero_grad/重新前向反向更新，900秒总时限；第一次失败不做第二次。只有两次通过才补同一非空GT训练片段XYZ/UV各一次、各300秒。优先冻结clips，缺失则稳定身份排序冻结train队列stride8首合格片段，绝不读开发/测试标签。UV只走UV reader，scale1/32，不读XYZ派生输入。
未完成阶段不填0；捕获OOM只落盘已有CPU证据，不继续CUDA计算。SIGABRT与计算状态分记，资源检查不计正式模型；不拿最大窗口时间冒充全队列平均。
当前结论：尚未完成全部规定验收，不具备直接启动原定两条件训练贯通的依据；本轮停止，不追加优化。
源码`39c1568f6462e7aca042450a36ce608353eca356`；产物`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_tensor_lifetime_check_v1`；旧缓存/协议/错误/退出历史/config和用户JSON保持。计算与publication分列。错误/门槛：无
