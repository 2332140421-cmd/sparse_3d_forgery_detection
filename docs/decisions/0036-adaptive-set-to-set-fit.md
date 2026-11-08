# ADR-0036：自适应集合对集合关系与限定拟合

状态：用户2026-11-08授权实现并运行；起点fecadfdb091d668b1c6ed3cf6e5cdc698b1c03b2，新分支pvst-set-to-set-relations。

仅对新入口取代ADR-0034的同片时间→同帧深度关系及原global帧分类捷径；不自动启用ADR-0035最终context_norm，不改变旧A/B/NORM及其state_dict、结果或默认行为。
研究问题、形状和边界见[设计](../pvst/set_to_set_design_v1.md)。集合是重叠软路由，不是物体/Part发现或固定互斥分区；不承诺无损/穷举子集/物理因果。此范围明确允许潜在集合及独立关系检测头，不恢复V7/V8/V9数值模型。

唯一新模块set_relations的调用者是RGBDSetSlices，输出当前逐片512维上下文和关系GRU帧logit；现有PairRelation不能表达非固定行列、重叠角色集合，因此新增此模块而非通用图框架。
8单元×A/B×4状态，每状态512，8head，两块联合Transformer。独立sigmoid参与，带log参与偏置读取全部当前/历史token和可学习空状态，不截邻居。深度用绝对片号×.125，时间用真实PTS差。当前逐片读取64个联合状态；关系单元z学习汇聚g，经真实dt编码和512GRU到检测头。不detach历史。

保留官方DFormer-S、完整多尺度原生解码、所有原生有效点SlabPool4×128、像素旁路、高分辨率几何、固定BN统计、BF16/checkpoint/CPU保存激活；共享Local从common_initial逐键strict加载并校验哈希，之后明确删除已被新帧头取代的local.frame。新pool/sets/pixel_context按seed20260909初始化，pixel_context仍零初始化，不载入已拟合权重。

仅训练清单an:7ogg04、an:728qx7各fake/real首24原始帧；四段固定平衡循环，每段一次AdamW，总计最多80次或worker启动1800秒。lr1e-5/1e-4、wd.01、clip1、三项逐帧loss等权沿现有口径。初始/每10步/结束固定评价这四段，不读开发/测试，不重跑前端。因果/元素重排/梯度/空帧/RNG检查先行，一次最终关系消息置零检查消费情况，不将其当效果消融。失败保留真实退出码且不自动重试。

新薄入口set_fit_check复用Video/load_clip/forward/optimizer/terms/指标/preview，独立父进程仅限时自己的worker进程组，随后报告与提交推送；不复制旧正式训练runner。诊断资产独立derived/pvst_set_to_set_fit_check_v1，计算和发布状态分离。正式训练、额外seed/样本/超参搜索均未授权。
