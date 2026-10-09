# ADR-0015：V8 全视频连续帧工程 pilot

状态：用户授权执行；不替代ADR-0013/0014的历史实验。

目标是验证全帧队列、跨计算块的时序状态延续和逐帧/视频响应接口。
新增独立 `research_tools/v8/full_video_streaming/`，直接调用者为其runner；
输出为两个视频的原始PTS、帧输入、格响应、帧响应、视频累计状态及工程报告。
简单窗口forward会每块重置GRU且没有恢复/视频累计，故需要最小逐帧状态接口，
不建设通用调度器、数据库、provider体系或新训练框架。

唯一参考是已完成局部几何pilot中MEAN_REPLICATED seed20260909第100epoch。
不新增参数、不训练；真实内容/padding隔离、粗分组、空间消息与GRU复用其冻结定义。
PyAV从视频首帧至正常EOF顺序解码，保存原始PTS/time_base和视频相对秒数；
不按旧窗口、分数或图像内容选帧。工具近期输入只含前两帧和当前帧，重复端点明确标记。
单帧几何只有在视频SHA、原始帧、预处理图像和冻结provider配置一致时才可复用；
旧窗口的运动、关联和hidden不能作为连续上下文复用。

forward_step不以块内index=0判定首帧。真实首帧aux=0；其他帧使用原no-history/关联路径。
frame响应=logsumexp(cell_logit+log(area))-log(total_area)。视频响应以log域累计各帧
未归一化分子及真实面积，等价于对全部帧全部格的一次性聚合，零面积严格排除。
响应是弱监督冻结模型的工程读出，不是帧/pixel真值或新AUROC结果。

相同输入的旧16帧forward、块长7/16/31、帧边界保存恢复和未来扰动前缀必须通过。
真实权重smoke在默认TF32路径观察到batch-dependent最大格logit误差0.000988；
本轮推理统一禁用TF32后误差0.000000954，保持原容差atol/rtol=2e-5，未放宽容差。
这是推理数值设置，不修改旧模型/分数或训练设置；log域累计用float64以稳定长序列。

只完整处理按train source/pair身份排序选出的第一对长度≥63帧的视频；
其他视频仅建立清单及元数据成本估计。COMPLETE范围为two_video_engineering_pilot。
没有全视频训练、独立泛化、帧级/像素定位或全帧component图验证。
