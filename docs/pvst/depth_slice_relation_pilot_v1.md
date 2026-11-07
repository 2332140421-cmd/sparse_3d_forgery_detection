# 深度切片关系学习首轮实验

状态：RUNNING。本页为启动核验，不是最终成绩；实时状态以独立运行目录progress.json为准。
执行源码：`35914bb1f469d896f77b91954780a5116179bc49`；单条件DEPTH_SLICE_RELATION，seed20260909。

协议：[ADR-0033](../decisions/0033-depth-slice-relation-pilot.md)、[冻结配置](depth_slice_pilot_config.json)。
新方法不是旧PVST体素恢复；固定h1/8、d0=0、D512、两组独立8head时间→深度显式关系。
官方ImageNet ResNet18冻结、原生RGB正常stride特征；可学习原生XYZ/UV128→256、视觉几何融合128、按片编码后mean/max→512；保留全部原生像素旁路与独立像素/帧头。可训练参数13,789,378。

数据复用已登记ANAKIN训练8 pair/16视频/2138帧，其中1065正帧；开发4 pair/8视频/924帧，其中462正帧。
训练fake7条全程非空、1条混合空/非空；开发fake4条全程非空。全程非空开发样本不能验证复杂时间边界。
源身份隔离，无保留测试集读取、无新场景过滤或下载。24条视频mask尺寸/数值编号、原生解码PTS与RGB缓存已核验；24条原生ImageNet ResNet特征缓存已完成，MoGe未重跑。
训练固定每视频首/中/尾3段24连续帧，共48片段/epoch；不是整个训练视频逐帧覆盖。监测固定前2pair完整视频，称训练监测子集。开发完整合法帧每帧计一次；尾段补左历史，常规24帧边界重置上下文，见ADR。

6项CPU检查通过。真实含非空GT的an:7ogg04、[0,24)片段两次更新通过：

|检查步|总loss|frame BCE|pixel BCE|Dice loss|帧头梯度范数|帧logit范围|
|---|---|---|---|---|---|---|
|1|1.414356|0.073420|0.452971|0.887965|9.592404|1.484375–3.8125|
|2|1.221428|6.10e-9|0.267287|0.954141|0（BCE饱和）|17.625–23.75|

每步梯度存在且有限，全部核心分支两步内至少一次非零，pixel参数确实改变、AdamW状态已建立。验收权重丢弃，正式训练从规定初始化开始。
该片段已观测allocated峰值1.3404GiB，不是全训练峰值；总loss下降也不等于Dice改善（第二步Dice loss上升）。

此前两次准备检查退出1均保留于独立目录execution_history：
00_bf16_context_scatter为FP32目标/BF16上下文index_copy类型错误，已修复并补BF16反向测试；
01_overstrict_per_step_gradient_check为第一次验收更新后第二步帧BCE饱和，逐步非零断言过严。
现检查逐步存在性/有限性以及跨两步分支连通性，不改初始化、监督或损失，不删除原退出历史。

独立入口自动执行epoch0→最多30轮（至少10轮、Dice耐心8）→同一best全任务评价→中文报告→文档提交与普通push。
当前尚无正式epoch/best成绩，不把验收响应作为检测性能。

完整运行目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_depth_slice_relation_pilot_v1/`。
查看`progress.json`、`run.log`和后续`history.csv`；完成后`report.md`、`best_dev.json`、`best_monitor.json`、`samples/index.html`、`publication.json`。
计算与Git发布状态分开。相机运动/深度漂移未解决，没有二维匹配对照不主张三维增益。不追加其他实验。
