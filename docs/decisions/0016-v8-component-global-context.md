# ADR-0016：动态component与singleton格的全帧关系匹配对照

状态：接受；仅适用于用户授权的 `v8-component-global-context` 实验分支。

问题：相同几何/RGB/状态和时间路径下，动态成员分区是否改善全帧区域关系学习？

在MEAN_REPLICATED路径的固定格邻接之后，以同一模块替换原component residual：
面积加权区域hidden128、有效几何点计数加权XYZ3、面积比例1、成员数/4共133维；
投影128、单层4头同帧attention、residual+LayerNorm，广播后只调用一次原residual MLP。
其余cell GRU、多前驱、delta_t、分类头及面积LSE不变。条件只改变成员分区。

不增加永久component轨迹、全局token、新前端或额外损失。使用同801窗口、同训练统计，
两条件参数/初始化/数据顺序匹配；6模型100 epochs；窗口监督与评价。缓存只读复用。

实现只需模型模块（直接调用既有patch编码器）和薄运行入口（复用加载、权重、指标、
checkpoint工具）；GPU预载均值和mask避免不必要的dense XYZ读取。无需新数据平台。

本决定扩展V8，未取代V6/V7冻结方法；旧V8窗口与全视频工程结果保留，不修改旧行为。
旧MEAN_REPLICATED无全帧attention，故历史分数不是本轮匹配对照；新结果不证明纯三维、
时间机制或定位的独立贡献。全视频连续帧训练不在本轮范围。
