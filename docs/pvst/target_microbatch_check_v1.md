# PVST v0.3 目标帧4＋4微批资源验收

**状态：OOM；正式模型0/2，不启动训练。**

只改变目标帧执行：A历史[104,120)、目标[120,124)；B历史[108,124)、目标[124,128)。每组20帧，逻辑更新40帧次、24唯一输入帧；历史重算，不复用可学习特征、不提前step。
三层因果最大回溯2+4+8=14帧，加帧级visual的2帧，完整16帧。context读写仅同帧，BN固定、LayerNorm逐位置，XYZ为冻结训练全局scale；真实PTS与绝对q保持。没有窗口统计或未来帧依赖。
losses逐帧计算：frame BCE标量、pixel BCE原生全像素均值、softDice每帧全像素sum/epsilon1；缺GT仅空间项0，不改变frame分母。clip_terms对目标帧取均值，本AN real8帧均有合法零GT，各分项每微批乘4/8。未来AF/AN外层配对权重仍须保留。
诊断关闭所有Dropout（含MHA与attention内部），两份诊断模型同非零输出头验证上游梯度；资源验收恢复原Dropout与规定初始化。分组改变随机轨迹，不称逐步完全相同。

执行源码：`682595fdf813f7b6fdbcc32ff5916a94a3c13903`；冻结配置XYZ仍null；1/8只作局部资源配置。新增前端调用0。

## 小规模语义核验
```json
{
  "status": "PASS",
  "comparisons": [
    {
      "status": "PASS",
      "dtype": "torch.float32",
      "errors": {
        "frame_logit": 2.9802322387695312e-08,
        "pixel_logits": 1.7881393432617188e-07,
        "loss_components": 5.960464477539063e-08
      },
      "gradient_max_abs": 2.308189868927002e-05,
      "gradient_max_abs_over_reference_max": 2.1956521739130435,
      "key_gradients": {
        "geo.0.weight": {
          "max_abs": 1.3560056686401367e-06,
          "reference_norm": 0.20043820142745972
        },
        "vfe.0.weight": {
          "max_abs": 4.470348358154297e-08,
          "reference_norm": 2.1980888843536377
        },
        "fine_time.layers.0.qkv.weight": {
          "max_abs": 1.862645149230957e-09,
          "reference_norm": 0.32990363240242004
        },
        "coarse_time.layers.0.qkv.weight": {
          "max_abs": 1.3969838619232178e-09,
          "reference_norm": 0.28859755396842957
        },
        "pixel_out.weight": {
          "max_abs": 2.384185791015625e-07,
          "reference_norm": 7.416090488433838
        },
        "frame_residual.6.weight": {
          "max_abs": 2.384185791015625e-07,
          "reference_norm": 5.045561790466309
        }
      },
      "tolerances": {
        "output_atol": 2e-05,
        "output_rtol": 1e-05,
        "gradient_atol": 2e-05,
        "gradient_rtol": 3e-05
      },
      "loss_terms_reference": [
        1.9465551376342773,
        0.6052652597427368,
        0.6090775728225708,
        0.7322123646736145
      ],
      "loss_terms_microbatch": [
        1.9465551376342773,
        0.605265200138092,
        0.6090775728225708,
        0.7322124242782593
      ],
      "diagnostic_heads": "same nonzero weights only in diagnostic models; all Dropout and MHA dropout=0; train/checkpoint retained"
    },
    {
      "status": "PASS",
      "dtype": "torch.bfloat16",
      "errors": {
        "frame_logit": 0.001953125,
        "pixel_logits": 0.001953125,
        "loss_components": 8.559226989746094e-05
      },
      "gradient_max_abs": 0.01171875,
      "gradient_max_abs_over_reference_max": 1.724390243902439,
      "key_gradients": {
        "geo.0.weight": {
          "max_abs": 0.0008544921875,
          "reference_norm": 0.1985328197479248
        },
        "vfe.0.weight": {
          "max_abs": 0.002197265625,
          "reference_norm": 2.1885805130004883
        },
        "fine_time.layers.0.qkv.weight": {
          "max_abs": 6.103515625e-05,
          "reference_norm": 0.3298802971839905
        },
        "coarse_time.layers.0.qkv.weight": {
          "max_abs": 3.0517578125e-05,
          "reference_norm": 0.28844836354255676
        },
        "pixel_out.weight": {
          "max_abs": 0.01171875,
          "reference_norm": 7.4253764152526855
        },
        "frame_residual.6.weight": {
          "max_abs": 0.0078125,
          "reference_norm": 5.04946756362915
        }
      },
      "tolerances": {
        "output_atol": 0.02,
        "output_rtol": 0.02,
        "gradient_atol": 0.2,
        "gradient_rtol": 0.03
      },
      "loss_terms_reference": [
        1.947211742401123,
        0.6061400175094604,
        0.6089023947715759,
        0.7321693897247314
      ],
      "loss_terms_microbatch": [
        1.9471261501312256,
        0.6061416864395142,
        0.6088321208953857,
        0.7321523427963257
      ],
      "diagnostic_heads": "same nonzero weights only in diagnostic models; all Dropout and MHA dropout=0; train/checkpoint retained"
    }
  ]
}
```

## 实际微批阶段
|更新/微批/阶段|耗时秒|allocated/reserved GiB|已观测峰值A/R GiB|RSS GiB|
|---|---:|---:|---:|---:|
|0/None/worker_start|None|0.000/0.000|0.000/0.000|0.516|
|0/None/model_built|0.8091107662767172|0.066/0.080|0.066/0.080|0.679|
|1/A/before_microbatch_load|None|0.066/0.080|0.066/0.080|0.679|
|1/A/data_loaded|0.5235004127025604|1.343/1.443|1.379/1.443|1.211|
|1/A/forward_complete|19.621444502845407|7.587/13.924|11.349/13.924|14.521|

## 结果与边界
成功更新0次，step成功调用0次；原24帧联合检查不重跑。
```json
{
  "update_records": [],
  "microbatches": [],
  "error": "CUDA out of memory. Tried to allocate 606.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 372.38 MiB is free. Process 290469 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 21.94 GiB memory in use. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 889.84 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)\nException raised from malloc at ../c10/cuda/CUDACachingAllocator.cpp:1318 (most recent call first):\nC++ CapturedTraceback:\n#4 std::_Function_handler<std::shared_ptr<c10::LazyValue<std::string> const> (), c10::SetStackTraceFetcher(std::function<std::string ()>)::{lambda()#1}>::_M_invoke(std::_Any_data const&) from Logging.cpp:0\n#5 c10::Error::Error(c10::SourceLocation, std::string) from ??:0\n#6 c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc(signed char, unsigned long, CUstream_st*) from CUDACachingAllocator.cpp:0\n#7 c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::malloc(void**, signed char, unsigned long, CUstream_st*) from :0\n#8 c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::allocate(unsigned long) from :0\n#9 at::TensorBase at::detail::_empty_strided_generic<c10::ArrayRef<long> >(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType) from :0\n#10 at::detail::empty_strided_generic(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType) from ??:0\n#11 at::detail::empty_strided_cuda(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ScalarType, std::optional<c10::Device>) from ??:0\n#12 at::detail::empty_strided_cuda(c10::ArrayRef<long>, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from ??:0\n#13 at::native::empty_strided_cuda(c10::ArrayRef<long>, c10::ArrayRef<long>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from ??:0\n#14 at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from RegisterCUDA.cpp:0\n#15 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>), &at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> > >, at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from RegisterCUDA.cpp:0\n#16 at::_ops::empty_strided::redispatch(c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from ??:0\n#17 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>), &at::(anonymous namespace)::empty_strided>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool> > >, at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from RegisterBackendSelect.cpp:0\n#18 at::_ops::empty_strided::call(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) from ??:0\n#19 at::empty_strided(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions) from :0\n#20 at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from ??:0\n#21 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>), &at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd___to_copy>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat> > >, at::Tensor (at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from RegisterCompositeExplicitAutograd.cpp:0\n#22 at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from ??:0\n#23 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>), &at::(anonymous namespace)::_to_copy>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat> > >, at::Tensor (at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from RegisterBackendSelect.cpp:0\n#24 at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from ??:0\n#25 torch::autograd::VariableType::(anonymous namespace)::_to_copy(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from VariableType_0.cpp:0\n#26 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>), &torch::autograd::VariableType::(anonymous namespace)::_to_copy>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat> > >, at::Tensor (c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from VariableType_0.cpp:0\n#27 at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) from ??:0\n#28 at::native::to(at::Tensor const&, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) from ??:0\n#29 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>), &at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd_dtype_to>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat> > >, at::Tensor (at::Tensor const&, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) from RegisterCompositeImplicitAutograd.cpp:0\n#30 at::_ops::to_dtype::call(at::Tensor const&, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) from ??:0\n#31 torch::autograd::THPVariable_to_type(_object*, c10::ScalarType, std::optional<c10::MemoryFormat>) from python_variable_methods.cpp:0\n#32 torch::autograd::THPVariable_float(_object*, _object*, _object*) from python_variable_methods.cpp:0\n#33 method_vectorcall_VARARGS_KEYWORDS from /usr/local/src/conda/python-3.12.7/Objects/descrobject.c:365\n#34 _PyObject_VectorcallTstate from /usr/local/src/conda/python-3.12.7/Include/internal/pycore_call.h:92\n#35 _PyEval_EvalFrameDefault from /croot/python-split_1728047642222/work/build-static/Python/bytecodes.c:2715\n#36 _PyEval_EvalFrame from /usr/local/src/conda/python-3.12.7/Include/internal/pycore_ceval.h:89\n#37 PyCFunction_Call from /usr/local/src/conda/python-3.12.7/Objects/call.c:387\n#38 _PyObject_VectorcallTstate from /usr/local/src/conda/python-3.12.7/Include/internal/pycore_call.h:92\n#39 PyObject_CallFunctionObjArgs from /usr/local/src/conda/python-3.12.7/Objects/call.c:957\n#40 torch::autograd::PySavedVariableHooks::call_unpack_hook() from ??:0\n#41 torch::autograd::SavedVariable::unpack(std::shared_ptr<torch::autograd::Node>) const from ??:0\n#42 torch::autograd::generated::SparseAddmmBackward0::apply(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) from ??:0\n#43 torch::autograd::Node::operator()(std::vector<at::Tensor, std::allocator<at::Tensor> >&&) from :0\n#44 torch::autograd::Engine::evaluate_function(std::shared_ptr<torch::autograd::GraphTask>&, torch::autograd::Node*, torch::autograd::InputBuffer&, std::shared_ptr<torch::autograd::ReadyQueue> const&) from ??:0\n#45 torch::autograd::Engine::thread_main(std::shared_ptr<torch::autograd::GraphTask> const&) from ??:0\n#46 torch::autograd::Engine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool) from ??:0\n#47 torch::autograd::python::PythonEngine::thread_init(int, std::shared_ptr<torch::autograd::ReadyQueue> const&, bool) from ??:0\n#48 execute_native_thread_routine from thread48.o:0\n#49 start_thread from ./nptl/pthread_create.c:442\n#50 __clone3 from ./misc/../sysdeps/unix/sysv/linux/x86_64/clone3.S:81\n",
  "failure_phase": "backward",
  "failure_microbatch": "A",
  "process": {
    "pid": 525626,
    "returncode": 2,
    "termination_signal": null,
    "timed_out": false,
    "wall_s": 59.99331793747842,
    "timeout_s": 900,
    "samples": [
      {
        "unix_time": 1793856564.591882,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 12927, 11146, 96",
        "compute_pid_mib": "290469, 507\n525626, 9938"
      },
      {
        "unix_time": 1793856569.6284738,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 8139, 15934, 64",
        "compute_pid_mib": "290469, 507\n525626, 14718"
      },
      {
        "unix_time": 1793856574.664455,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 8131, 15942, 42",
        "compute_pid_mib": "290469, 507\n525626, 14722"
      },
      {
        "unix_time": 1793856579.6978002,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 8113, 15960, 11",
        "compute_pid_mib": "290469, 507\n525626, 14726"
      },
      {
        "unix_time": 1793856584.7341585,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 8101, 15972, 53",
        "compute_pid_mib": "290469, 507\n525626, 14740"
      },
      {
        "unix_time": 1793856589.7704256,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 4519, 19554, 100",
        "compute_pid_mib": "290469, 507\n525626, 18322"
      },
      {
        "unix_time": 1793856594.8125532,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 3819, 20254, 68",
        "compute_pid_mib": "290469, 507\n525626, 19022"
      },
      {
        "unix_time": 1793856599.844735,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 373, 23700, 40",
        "compute_pid_mib": "290469, 507\n525626, 22468"
      },
      {
        "unix_time": 1793856604.9099638,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 373, 23700, 7",
        "compute_pid_mib": "290469, 507\n525626, 22468"
      },
      {
        "unix_time": 1793856609.9446166,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 373, 23700, 7",
        "compute_pid_mib": "290469, 507\n525626, 22468"
      },
      {
        "unix_time": 1793856614.978259,
        "worker_pid": 525626,
        "gpu_total_free_used_util": "24564, 373, 23700, 7",
        "compute_pid_mib": "290469, 507\n525626, 22468"
      }
    ]
  },
  "host": {
    "available_bytes": 37627506688,
    "rss_bytes": 15889297408,
    "peak_rss_bytes": 18433126400
  }
}
```
本轮只验收规定XYZ窗口，不追加UV、非空GT或其他样本。尚未完成两次更新；下一步仅依据本轮实际失败阶段/算子和资源证据决定，不自动缩小微批或改算子。
旧24帧失败峰值为未完成更新的观测值，不能拿旧失败耗时当作完整更新时间或计算总加速比。全部原节点/边、宽度、尺度、输入像素、损失及缓存保持。
产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_target_microbatch_check_v1`。计算状态与publication分开。
