# ADR-0059 AINPAINT视觉底座的全片变化关系定位

状态：用户附件明确授权，2026-11-26（实际系统日期以运行记录为准）。

问题：在固定已训练视觉底座上，完整变化关系及XYZ组织是否带来定位增量？邻近相关与变化稳定只是候选假设。

最小实现：change_relation_model.py由change_relation.py调用，读取既有patch/成员数据，输出全部帧logit和软块监督；旧Direct模型只有浅logit修正，不足以表达本轮指定设计。
change_relation.py复用旧缓存/计分/检查点工具，独立保存三组训练状态；change_relation_report.py只整理保存结果，避免CPU报告触发模型。
不添加通用provider/注册表/作业框架。

决定：A同帧XY，B全片XY，C全片XYZ；同构D128/两轮4头，当前与参照共V空间差分，经8近邻变化比较、学习槽聚合再global SDPA。
不要求跨帧共有键、点跟踪、真实必须平滑；无GT路由。XY和XYZ只在相同几何有效像素上比较，视觉定位和评价全画面。
官方HighResContextHead的proj/norm/relu后512与up2后128层加零初始化无bias1x1融合，继续官方解码，不叠加z0浅修正。
全head参数可训练、BN buffers固定；原dropout在train开启/eval关闭。块头正常初始化，软g/n目标/成员权重，仅合法帧监督。
原89/19协议，冻结DINO/LoRA观测，共同随机新增模块，head来自原epoch01，10轮/1780视频更新，共228780合法帧呈现。

仅在本独立范围取代ADR-0055/0056的浅logit修正、两条件及三轮，不改变旧默认/结果。仍不证明纯深度或三维必要性。
资源保持原数学含义，失败不静默降规格。Codex只固定短预检；正式训练用户启动，模型/缓存大资产不入Git。
