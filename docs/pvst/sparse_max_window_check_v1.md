# PVST v0.3 修复后稀疏attention最大窗口资源检查

状态：**OOM**；正式模型仍0/2，本轮无正式训练、评价、前端或单层重复测试。
源码：`a1c3cdc2e2f566e4d3a6a0513ffb53d7fc6267a0`。视频782426ea0330c7fea684953c；ANAKIN/train/real/an:4wdo37；720×1280；历史[104,120)、目标[120,128)。
局部XYZ h=1/8、coarse=1/4；seed20260909；外层BF16、逐head FP32 CSR sampled_addmm → COO softmax → CSR mm；保留逻辑填充列修复。
attention512 / QKV8192 / FFN8192 / point32768 / tile256；原外层checkpoint CPU卸载。完整点、像素、邻居及128/256/512宽度不变；正式XYZ尺度仍null。
真实损失为8目标帧均值frame BCE + pixel BCE + soft Dice loss；ANAKIN real使用既有全零空间GT。损失仅作执行检查。

实际fine/coarse：2970629/1488042；预期2,970,629/1,488,042。
父进程墙时：67.171秒；returncode=2；signal=None。
最后阶段：backward；实际更新次数：0；参数改变核对：False。

|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|当前 allocated/reserved GiB|RSS/主机可用 GiB|
|---|---:|---:|---:|---:|---:|
|worker_start|None|1.807|0.000/0.000|0.000/0.000|0.514/47.958|
|model_built|0.7822061814367771|2.593|0.066/0.080|0.066/0.080|0.677/47.844|
|data_loaded|0.6211445834487677|3.215|1.635/1.713|1.598/1.713|1.271/47.390|
|forward_complete|18.0643613897264|21.281|20.870/21.809|13.037/21.637|14.562/35.350|

六层构建核对：6/6；完成前向的运行证据如下。未完成层不冒称通过。
|层|dilation|节点|有效边E|平均度|拓扑MiB|完成head|逻辑填充列|
|---|---:|---:|---:|---:|---:|---:|---:|
|fine_time.0|1|2970629|37042228|12.469|587.884|4|0|
|fine_time.1|2|2970629|31105643|10.471|497.299|4|0|
|fine_time.2|4|2970629|28847555|9.711|462.843|4|0|
|coarse_time.0|1|1488042|26982715|18.133|423.076|8|0|
|coarse_time.1|2|1488042|22781330|15.310|358.968|8|0|
|coarse_time.2|4|1488042|20707343|13.916|327.322|8|0|

梯度、损失及更新证据：
```json
{
  "loss_components": {
    "total": 1.7414692640304565,
    "frame_BCE": 0.7357272505760193,
    "pixel_BCE": 0.005926134530454874,
    "soft_Dice_loss": 0.9998159408569336
  },
  "gradients": null,
  "parameter_delta_norms": null,
  "optimizer_state_parameters": null
}
```
末次状态与失败位置（已观测峰值不是未完成更新的峰值上界）：
```json
{
  "phase": "backward",
  "elapsed_s": 65.55915680062026,
  "peak_allocated_bytes": 22812468736,
  "peak_reserved_bytes": 23416799232,
  "host": {
    "available_bytes": 37161975808,
    "rss_bytes": 15883857920,
    "peak_rss_bytes": 15879180288
  },
  "trace": {
    "fine_time": {
      "active_layer": {
        "dilation": 4,
        "nodes": 2970629,
        "width": 256,
        "dtype": "torch.bfloat16"
      },
      "layers": [
        {
          "operation": "norm2+FFN block",
          "input_shape": [
            5125,
            256
          ],
          "hidden_shape": [
            5125,
            1024
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "operation": "norm2+FFN block",
          "input_shape": [
            5125,
            256
          ],
          "hidden_shape": [
            5125,
            1024
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "operation": "norm2+FFN block",
          "input_shape": [
            5125,
            256
          ],
          "hidden_shape": [
            5125,
            1024
          ],
          "dtype": "torch.bfloat16"
        }
      ],
      "edge_stats": [
        {
          "nodes": 2970629,
          "edges": 37042228,
          "average_degree": 12.46948979492222,
          "topology_bytes": 616440688,
          "dilation": 1,
          "backend": "CSR sampled_addmm/COO softmax/CSR mm FP32 per head",
          "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
          "logical_padding_columns": 0,
          "completed_heads": 4
        },
        {
          "nodes": 2970629,
          "edges": 31105643,
          "average_degree": 10.471062862444283,
          "topology_bytes": 521455328,
          "dilation": 2,
          "backend": "CSR sampled_addmm/COO softmax/CSR mm FP32 per head",
          "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
          "logical_padding_columns": 0,
          "completed_heads": 4
        },
        {
          "nodes": 2970629,
          "edges": 28847555,
          "average_degree": 9.710924858001453,
          "topology_bytes": 485325920,
          "dilation": 4,
          "backend": "CSR sampled_addmm/COO softmax/CSR mm FP32 per head",
          "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
          "logical_padding_columns": 0,
          "completed_heads": 4
        }
      ]
    },
    "coarse_time": {
      "active_layer": {
        "dilation": 4,
        "nodes": 1488042,
        "width": 512,
        "dtype": "torch.bfloat16"
      },
      "layers": [
        {
          "operation": "norm2+FFN block",
          "input_shape": [
            5290,
            512
          ],
          "hidden_shape": [
            5290,
            2048
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "operation": "norm2+FFN block",
          "input_shape": [
            5290,
            512
          ],
          "hidden_shape": [
            5290,
            2048
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "operation": "norm2+FFN block",
          "input_shape": [
            8192,
            512
          ],
          "hidden_shape": [
            8192,
            2048
          ],
          "dtype": "torch.bfloat16"
        }
      ],
      "edge_stats": [
        {
          "nodes": 1488042,
          "edges": 26982715,
          "average_degree": 18.13303320739603,
          "topology_bytes": 443627784,
          "dilation": 1,
          "backend": "CSR sampled_addmm/COO softmax/CSR mm FP32 per head",
          "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
          "logical_padding_columns": 0,
          "completed_heads": 8
        },
        {
          "nodes": 1488042,
          "edges": 22781330,
          "average_degree": 15.309601476302417,
          "topology_bytes": 376405624,
          "dilation": 2,
          "backend": "CSR sampled_addmm/COO softmax/CSR mm FP32 per head",
          "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
          "logical_padding_columns": 0,
          "completed_heads": 8
        },
        {
          "nodes": 1488042,
          "edges": 20707343,
          "average_degree": 13.915832348818112,
          "topology_bytes": 343221832,
          "dilation": 4,
          "backend": "CSR sampled_addmm/COO softmax/CSR mm FP32 per head",
          "head_impl": "research_tools.pvst.sparse_attention.sparse_head",
          "logical_padding_columns": 0,
          "completed_heads": 8
        }
      ]
    }
  }
}
```
错误与调用栈：
```
Traceback (most recent call last):
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/pvst/max_window_check.py", line 168, in worker
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
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/pvst/sparse_attention.py", line 58, in sparse_head
    weights=torch.sparse.softmax(scores,dim=1)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 316.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 174.00 MiB is free. Process 3903557 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 22.11 GiB memory in use. Of the allocated memory 21.08 GiB is allocated by PyTorch, and 571.67 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```
启动/结束GPU：
```json
{
  "start": {
    "unix_time": 1793806798.4032412,
    "worker_pid": null,
    "gpu_total_free_used_util": "24564, 22845, 1227, 10",
    "compute_pid_mib": "3903557, 507"
  },
  "after": {
    "unix_time": 1793806865.6137257,
    "worker_pid": 3818469,
    "gpu_total_free_used_util": "24564, 22846, 1227, 7",
    "compute_pid_mib": "3903557, 507"
  },
  "worker_gpu_released": true
}
```

同窗口旧query SDPA：前向61.7257秒；反向至少827.776秒仍未完成；900秒TIMEOUT，AdamW未执行。当前逐head FP32核心的精度路径不同，数学目标保持，不宣称逐位数值相同，也不以旧下界计算精确完整加速比。

本次真实CUDA OOM，停止；不建议直接进入两条件各一轮。先根据明确失败算子与内存证据决定下一步，本轮不继续修复。
旧日志/退出码/缓存及正式配置保持原样；新模型和优化器丢弃，无checkpoint保存。计算状态与publication.json中的Git发布状态分开。
独立产物目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_v03_sparse_max_window_check_v1`；resource_curve.png为约5秒设备采样；result.json、stages.json、live.json保留完整证据。
