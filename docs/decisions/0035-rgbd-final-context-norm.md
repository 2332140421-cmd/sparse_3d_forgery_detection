# ADR-0035：最终关系token共享通道归一化

状态：2026-11-08 用户授权单条件RGBD_SLICE_RELATION_NORM、seed20260909。
起点949fea70e5e1f6b819512c7382901743a6076ff0，仍在pvst-rgbd-slice-relations。

研究问题：在完整时间→深度关系输出后，仅统一两个上下文残差入口的特征尺度，是否改善多pair定位和固定监测稳定性。既有B单pair能拟合，frame_context梯度偏大只是动机，不预设根因。

唯一核心改动：C_norm(i,t)=LayerNorm_512(C(i,t))，共享模块，eps=1e-5（本工程torch.nn.LayerNorm默认）、affine=True，初始weight1/bias0。先逐token沿512通道归一化，再用于pixel_context和frame_context(mean(C_norm))；不是先均值后归一化，不跨时间/片统计，不改坐标、mask、slice_id或logits。两个残差投影仍零初始化。旧A/B默认不启用、旧B checkpoint仍strict加载；新增1024参数，NORM共33,265,335个可训练参数。

共同初始化仍由原make+common_initial加载；新增模块放在旧模块构造之后且常量初始化不消耗RNG。检查全部共有state逐项完全相同、CPU/CUDA RNG状态相同、新增仅两键、旧参数优化器分组不变、新norm两参数按现有新模块lr1e-4/wd.01处理。

其余全部沿ADR-0034：原8训练/4开发pair、48条24原生帧片段/epoch、数据/PTS/GT、MoGe高分辨率缓存、DFormer微调和固定BN、池化4×128、两组512/8head关系、原生解码旁路、AdamW lr1e-5/1e-4、三项等权loss、BF16/checkpoint/可微CPU暂存、clip1、max30/min10/patience8、开发Dice选best。旧结果/诊断只读，不重训旧A/B、不重提前端、不再做单pair40/150步拟合。两次现有非空GT24帧smoke通过后，丢弃权重，从共同初始化只训练NORM。

按功能复用现有模型/正式训练循环；新薄入口context_norm负责必要检查、旧JSON/预测图比较与报告，立即调用者为CLI，输出独立derived/pvst_rgbd_slice_context_norm_v1。若直接调用旧完整入口会重提准备/拟合并训练旧A/B，因此用此薄入口限定一条件，不建设新框架。

每更新记录剪裁前总/两残差/norm梯度范数及实际clip系数；原始C与C_norm RMS、pixel投影RMS、实际帧残差mean-abs从已有前向分离成GPU标量，更新末一次CPU拷贝；不增加前向、不改变Dropout调用或梯度、不在片循环同步。RMS按实际占据token通道/投影元素，不是像素面积；帧残差按帧平均，原路径无观测残差0。旧B无此尺度日志，不补造。训练监测仍仅同一个实际参与反向的固定pair首段，完整曲线只代表该pair。

沿原完整开发逐帧评价与窗口边界、tail、宏平均、AP定义、固定logit0与视频log-mean-exp；NORM best和last分列，每行同checkpoint。旧A/B仅读正式JSON；固定旧B预测JPEG直接拼图，概率色阶均0–1，不为拼图重评分。报告四条fake的Dice/IoU/AP及差值，尤其两条inpainting；监测覆盖骤降、检测权衡和收益集中性需与梯度尺度分开判断。无改善直说不支持此接入有效，不追加调参。

本ADR只对新条件取代ADR-0034“最终C直接入两个残差”的路径；旧A/B、旧合同分数与资产保持。计算完成和Git发布分开记录。异常保存真实状态，不自动改精度/规模重试。
