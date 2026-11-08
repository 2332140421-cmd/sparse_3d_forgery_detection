# 最终关系上下文LayerNorm匹配实验

状态：准备启动，尚无新正式成绩。附件12份内容一致，本轮只执行一份协议、一个条件。

见[ADR-0035](../decisions/0035-rgbd-final-context-norm.md)、[独立配置](rgbd_slice_context_norm_config_v1.json)。唯一核心改动为完整时间—深度关系后共享LayerNorm(512,eps1e-5)，逐token归一化后供两残差消费；新增1024参数，预计33,265,335。坐标、切片、数据、loss与旧模型默认不改。

3项直接相关CPU检查通过：共有参数和RNG一致、旧B真实best checkpoint严格加载、逐token且先norm再mean/两头同输入/空片、标量观察器不消费RNG并保持反向。接下来独立入口只做两次现有24帧更新检查，然后自动训练新条件并评价best/last。

入口：`python -m research_tools.rgbd_slice.context_norm`。
原8/4 pair和48条训练片段、原共同初始化与高分辨率几何均只读复用，不重做前端或单pair拟合，不重训旧A/B。旧B预测直接用保存图像拼图。

产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_context_norm_v1/`。实时progress.json/run.log、每更新诊断update_history.jsonl及正式history/最佳与末轮指标，结束后报告更新并普通commit/push。
