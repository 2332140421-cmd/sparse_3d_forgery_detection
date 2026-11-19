# ADR-0052：固定空间模型权重的BN推理统计诊断

仅原AINPAINT_SPATIAL final12845权重、原dev4完整824帧，RUNNING_BN与IMAGE_BN各一次。
整体eval/no_grad；IMAGE仅将官方head五个BN的forward临时改为F.batch_norm，
running_mean/var=None、training=True、momentum0、保留affine/eps；finally恢复。
不改DINO LayerNorm或dropout，不更新参数、buffer、计数，不训练/校准或读训练GT。
原BF16、pad14/crop、原生FP32指标与24固定预览保持；概率0–1、差分[-1,1]不拉伸。
先少量固定中帧回放及同输入重复，记录旧logit未保存的限制；完整RUNNING回放原逐帧指标。
可在同一次RUNNING前向被动记录BN输入标量摘要。无第三种条件/组合搜索/阈值调整。
此诊断只扩展17.8固定权重推理范围，不改变原正式成绩、模型或默认行为。
