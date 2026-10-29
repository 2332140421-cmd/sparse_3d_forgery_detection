# ADR-0025：固定decoder BN运行统计的匹配对照

状态：接受；日期2026-10-29。

## Context

限定BN诊断发现同checkpoint仅改变统计模式即可改变部分样本输出，但不支持直接改用
BATCH_STATS正式评价。用户授权只改变训练/评价BN策略，重新训练两个完整条件。

## Decision

按`../v8/spatial_context_frozen_bn_contract.md`，在独立分支/目录从同一官方初始化开始。
decoder BN始终eval，running_mean/var/num_batches_tracked固定为官方值；
weight/bias与其他decoder参数仍训练，Dropout训练时开启。完整几何/上下文路径不变。
默认选项False保留原行为，train重载覆盖反复train/eval和resume；每轮断言buffers与初始化相同。
直接引用原64视频视觉与前端缓存，epoch0复用并用固定样例核对，不新提取或复制大缓存。

## Consequences

仅BN策略改变，数据、初始化、loss、pair顺序、预算和选模不变。
最终引用旧结果/旧history与本轮比较，不重新运行旧模型；单seed、开发队列不作显著性推断。
缓存元信息检查不冒称重新全量SHA验收，保存来源与size/mtime并在结束核对只读资产。

## Supersedes

只在新FIXED_BN实验取代ADR-0024 decoder训练时更新BN统计的策略；
保留原BN实验、诊断、默认入口和历史分数，不改变其他方法定义。
