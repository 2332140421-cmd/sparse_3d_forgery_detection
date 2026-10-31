# ADR-0028：V8二维／三维时序结构匹配检测

状态：接受；2026-10-31，用户明确授权。

研究问题：同一冻结视觉、数据、容量、时间机制和预算下，三维时序结构是否改善二维结构的视频检测？
只比较STRUCTURE_2D/STRUCTURE_3D，三个配对seed；从头初始化检测网络，不加载旧窗口或定位权重。

新入口`research_tools/v8/temporal_detection_core/run.py`直接调用现有GlobalContextModel、VideoTraining及时间损失。
必要的数据模块输出本轮mmap输入，模型模块只调整等宽输入；旧缓存缺逐点三维位移，不能切旧特征向量得到合法二维对照。
保持现有四邻接component最大4格、全帧attention、因果GRU、全内容格面积LSE；完整跨chunk梯度。
每条件80维cell输入（结构坐标/位移补零至3维），共享隐藏128、4头attention、分类头及参数量。

2D仅读取冻结RGB、UV、同批短历史可见跟踪对应和二维位移；其标准化、分组不读取XYZ派生量。
3D读取对应XYZ及同点三维位移；有效性要求两个端点可用。没有几何时保留RGB、当前UV、格与帧，并显式表示缺失。
位移是相机观测变化，不是世界速度；真实delta_t保留原GRU输入。XYZ均值压缩沿用V8 MEAN_REPLICATED，不能声称稠密表面细节未经压缩。
训练集分别拟合表示尺度与component阈值，RGB/时间等公共量规则一致；无dev/test统计校准。

48/20/20 source隔离研究子集，保留原24/8角色；官方区间按原生PTS闭区间并集。
测试选择官方test-only source，排除官方train任一编辑和全部已知旧使用source。
旧122训练source身份未恢复，明确说明其独立性依赖历史official-train协议，而不是已恢复完整历史交集。
原片三固定时刻预审仅为场景资格，不根据fake异常或分数筛选。

AdamW lr/wd均1e-4，最多30轮，dev视频AUROC最高选best、patience8、clip1；完整pair等权，
video BCE+0.5视频内类别均衡frame BCE。六模型完成后才对各best测试一次；last只评价train/dev。
标准视频AUROC/AP/ACC/F1/Precision/Recall/混淆矩阵，pooled帧AUROC/AP/F1；阈值logit≥0。
三个seed独立成绩、mean±样本std及配对3D−2D，不以集成替代。

仅本实验取代ADR-0018/0019的旧窗口初始化和5轮适配、ADR-0020单seed探索范围；不覆盖旧协议或历史分数。
ADR-0027定位扩量结果和ANAKIN保留测试边界不变；本实验测试来自ActivityForensics，不读取ANAKIN保留测试。
实验完成先落盘，Git发布失败单独记录并保留实际退出码；不以网络失败改写实验状态。
