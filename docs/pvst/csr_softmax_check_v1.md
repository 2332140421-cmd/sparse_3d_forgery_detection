# PVST v0.3 CSR行分段softmax：限定修复与阻塞报告

2026-11-05；状态 **REPRESENTATION_BLOCKED**。独立CSR softmax数值/梯度验证通过；直接CSR权重构造的反向违反禁止稠密N×N的要求，因此未放行实现。
**未运行131,072节点单层内存对比，未进入900秒完整窗口，未执行AdamW。正式模型仍0/2。**
## 实现与停止原因

在现有sparse_attention.py增加CSRSoftmax，仅封装行softmax。使用CSR offsets上的segment_reduce(max/sum)，p=exp(s-row_max)/row_sum；只保存p和crow。反向返回p*(g-row_sum(p*g))，不切断Q/K或relative bias梯度；每方向生成一次E长int64行号并复用，用完释放。空行无概率，非空行不加epsilon。FP32核心保持，FP64仅用于gradcheck；只支持所需一阶梯度。
前向p.div_只修改Function中新生成、无上游别名的exp私有缓冲；不覆盖输入、保存的上游张量或反向所需概率。实际保存内容由测试核对为概率p与crow两项；临时row为8E字节，概率4E字节及其他临时浮点张量仍有成本。
新head只作为tests/pvst_csr_candidate.py中的必要复现参考，显式路径为CSR sampled_addmm → CSR.values()+bias → CSRSoftmax → 原dropout → CSR constructor → sparse.mm，不构造COO。原正式sparse_head保持上次提交实现，候选仍默认关闭；没有将未通过的替代路径接入六层。
CPU回归首先发现direct CSR constructor对应SparseCompressedTensorBackward0调用aten._to_dense；随后在极小CUDA图确认相同行为。不是CSR softmax解析梯度错误，也不是segment_reduce缺乏CUDA支持。未声称所有底层算子无COO。
CUDA实际证据：
```json
{
  "torch": "2.5.1+cu124",
  "capability": {
    "CUDA_max_sum_backward": true,
    "rows": [
      0,
      0,
      2
    ],
    "gradient": [
      2.0,
      1.0,
      2.0
    ]
  },
  "constructor_grad_fn": "SparseCompressedTensorBackward0",
  "dense_constructor_backward": [
    {
      "op": "aten._to_dense.default",
      "input_layout": "torch.sparse_csr",
      "input_shape": [
        17,
        18
      ],
      "output_shape": [
        17,
        18
      ],
      "dtype": "torch.float32"
    }
  ],
  "dense_candidate_backward": [
    {
      "op": "aten._to_dense.default",
      "input_layout": "torch.sparse_csr",
      "input_shape": [
        17,
        17
      ],
      "output_shape": [
        17,
        17
      ],
      "dtype": "torch.float32"
    }
  ],
  "candidate_finite_gradients": true
}
```
上例直接构造17×18矩阵的反向出现17×18稠密梯度；完整小head的17×17反向也出现稠密矩阵。131,072节点逻辑shape[131072,131073]若以FP32稠密化，单张量形状成本约64.0005GiB（仅尺寸计算，未分配/未实测），明显违反本轮表示约束。
本轮只允许小型softmax自定义反向；为了绕过此额外CSR构造问题，可能还需显式values梯度桥接或其他经核实的构造接口，已超出“仅封装CSR行softmax”的批准边界。本轮不继续实现，不添加CSR构造/SpMM自定义反向，也不把整套attention包进Function。

## 正确性

预声明沿用已有容差：FP32输出atol2e-5/rtol2e-4、梯度atol2e-5/rtol3e-3；BF16输出atol.04/rtol.03、梯度atol.003/rtol.08。独立行softmax输出/梯度atol1e-6/rtol1e-5；FP64 gradcheck eps1e-6/atol1e-5/rtol1e-3。未事后放宽。
|精度/dim/dilation|输出max abs|输入梯度max abs|QKV梯度max abs|bias梯度max abs|投影梯度max abs|
|---|---:|---:|---:|---:|---:|
|torch.float32/256/1|4.7683716e-07|1.1641532e-10|8.1490725e-10|2.0954758e-09|6.9849193e-10|
|torch.float32/256/2|4.7683716e-07|1.1641532e-10|9.3132257e-10|2.5029294e-09|5.8207661e-10|
|torch.float32/256/4|4.7683716e-07|1.1641532e-10|5.8207661e-10|3.259629e-09|6.9849193e-10|
|torch.bfloat16/512/1|0.03125|5.7220459e-06|7.6293945e-06|3.8258731e-06|1.5258789e-05|

独立softmax/CUDA FP64 gradcheck/dropout：
```json
{
  "softmax": {
    "probability_max_abs": 0.0,
    "logits_gradient_max_abs": 5.960464477539063e-08
  },
  "cuda_fp64_gradcheck": true,
  "dropout": {
    "dropout": 0.1,
    "max_output_error": 0.0,
    "finite_gradients": true,
    "checkpoint_rng": "PASS"
  }
}
```
小图包含非均匀PTS、负坐标、不同帧节点数/空帧、跨帧查询和尾块，FP32覆盖dilation1/2/4，BF16覆盖512/8head；复用既有小图和固定容差，不重跑旧GPU大矩阵。dropout=.1，checkpoint开/关的RNG与梯度核对保持。
CPU直接相关测试结果另见cpu_tests.txt；布局阻塞测试明确断言当前API存在稠密化，不把它改写为“不生成稠密”的通过。17,821,672模型参数/原state_dict及原邻接保持；本轮仅新增无参数Function，无模型参数更改。

## 资源与决策

131,072/512新旧前向、反向、起始allocated和峰值：**未运行**，原因是此前表示正确性门槛失败。不能用理论存储字节或单独softmax保存内容宣称单层/整模型峰值下降。
完整窗口前向、反向、AdamW和峰值：**未运行**，没有新的OOM/TIMEOUT。旧18.064秒前向、约65.56秒OOM、allocated峰值21.25GiB的记录原样保留，不计算加速比。
本轮CUDA诊断子进程墙时2.718秒，returncode=0，signal=None，timeout=False；此前3元素微型能力检查墙时未单独记录，不补测；本次有界子进程远低于300秒预算。无大规模分配、无正式训练、无前端/冻结特征提取、无新权重或checkpoint。
**结论：本轮未完成可放行的显存修复；不建议进入两条件各一轮。唯一下一步建议：先授权并验证一个仅针对CSR构造values梯度的最小桥接，确认不稠密化，再继续既定单层内存门槛。** 本轮停止，不实施该建议。
源码身份：`a04996cc6b39315ee4a4eeb55edf1acb764f2e95`；新产物目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_csr_softmax_check_v1`。原缓存、正式config、旧报告/退出历史及用户无关JSON保留；计算与Git发布状态分列。
