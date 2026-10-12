# ADR-0019：主方法固定5轮完整视频适配

用户主动收缩计算范围以降低时间成本，不基于验证表现选seed/checkpoint。
原ADR-0018六模型20轮计划及恢复文件保留并标记PAUSED_SCOPE_CHANGE。
新运行只微调GLOBAL_COMPONENT、三个原seed、固定5epochs，最后epoch。

调用同一runner，显式传入conditions/epochs；不复制训练流程。严格核验后仅将
原5轮以内合法checkpoint的计划身份改为新身份，并记录迁移来源/hash/差异。
模型、optimizer、RNG、source顺序、已提交pair、loss和既有训练耗时全部保留。
不改旧文件，不重跑既有验证或缓存，不detach跨块梯度。

复用既有video评价函数的可选条件参数，单条件不要求CELL。主要比较为
微调−冻结COMPONENT，读旧分数；CELL未同样微调，不能称匹配消融。
不废弃历史ADR、不改变AGENTS的学术最小实现边界。
