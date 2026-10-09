# V8 当前状态与边界

## 用户最终目标

V8 的目标是面向全视频连续帧：每个约定处理帧的全部真实内容都有区域归属；动态 components 利用当前三维结构和近期运动形成；模型学习局部结构、整帧组织及连续演化；观测缺失保留而不删除。

此目标没有因当前 pilot 范围而缩减。

## 当前已实现范围

当前实现处理 256×256 letterbox 画布上的16×16基础格。每个被选中的窗口含16个时刻；每帧从原视频重新生成MoGe dense geometry、CoTracker短历史状态和冻结ResNet18空间特征。FULL使用格级XYZ mean、观测状态、逐帧动态component、固定格消息、cell级多对多关联和GRU；RGB_2D不读取XYZ、geometry mask或geometry-derived component pooling。监督与正式评价均为窗口级。

当前仍是继承的801窗口队列，不是独立全视频帧队列；GRU hidden每窗口重置；没有经帧级监督/评价的正式frame detector；dense XYZ在可学习编码前已压为每格mean。

## 正式基准

唯一干净正式基准为数据目录 `v8_full_coverage_observation_field_identityfix_v1`：718 train windows/122 sources，83 validation windows/14 sources，FULL与RGB_2D各3 seed、100 epochs。三seed平均logit的validation source-macro AUROC分别为0.785714和0.761905。

修复前 `v8_full_coverage_observation_field_v1` 含训练输入身份错配，状态为 `PRE_FIX_TRAIN_INPUT_IDENTITY_MISMATCH`，仅作历史记录，不参与正式模型选择或结论。

## 旧路线边界

允许继承原始视频、官方标签、source/split/window身份，以及通过视频SHA、source、role与frame identity核验的V8 raw frontend cache。当前V8未读取V7的XYZ、轨迹、component、S/Q、视觉特征、standardizer或检测模型，也未用修复前V8 checkpoint、训练统计或分数形成当前正式结果。

当前队列的window IDs、源帧身份和原视频映射仍来自旧开发实验 manifests；这是明确的样本范围依赖，而不是数值特征复用。因此当前结果不是独立全视频队列，也不是sealed test。

## 下一步边界

下一次只应先验证“格内mean之前的轻量局部XYZ编码”是否提供增量，并与mean-replicated等容量控制匹配。独立全视频连续帧接口应作为另一项后续工作，不能与局部几何改造同时进入一个正式比较。

## 后续独立工程验收记录

`v8-full-video-streaming`分支已完成独立的一对1LARL real/fake连续帧工程pilot，共1828帧，跨块保持hidden，冻结MEAN_REPLICATED模型，无训练或新性能评价。该范围不替换上述窗口正式基准，也不代表272视频全量完成。详见`docs/experiments/v8-full-video-streaming-pilot.md`。
