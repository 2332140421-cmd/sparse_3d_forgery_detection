# V8 局部几何编码匹配对照合同

仅适用于用户授权的 `v8-local-geometry-encoding` 开发分支。

唯一主比较为 `DENSE_LOCAL − MEAN_REPLICATED`。同一801窗口（718 train/83 validation），
同一16时刻、source隔离、标签、RGB、77维格级输入、components、前驱关联、状态、
标准化、网络参数量和初始化。禁止读取V7数值资产或修复前V8模型/统计/分数。

从identity-fix原始缓存构造content mask和有效geometry，真实内容无几何位置仍保留。
格均值只用content内可计算XYZ；分组按原四邻接、可用XYZ/运动代价、训练尺度和
25%分位阈值，最多4格，剔除纯padding。两个条件共享分组。

局部输入为绝对XYZ3、相对固定cell均值XYZ3、geometry-valid1。绝对连续量按训练集
valid-content像素mean/std标准化；相对量按训练集valid-content像素相对偏移RMS缩放。
无逐帧或逐格尺度拟合。MEAN_REPLICATED在同一valid位置复制mean、相对项全零。

可学习编码为Conv2d(7,32,3,padding=1)→ReLU→Conv2d(32,32,3,padding=1)→ReLU；
每层后content外激活置零；content-masked mean/max拼为64维。geometry-invalid内容
仍进入pooling。64维与相同77维格输入拼接141维，hidden保持128。

格统计、空间消息、component sums/counts和历史候选排除纯padding；无前驱走
no-history。最终area=0严格以负无穷排除logsumexp。保留冻结RGB主干在letterbox上
提取的特征，不宣称视觉骨干完全不响应padding。

沿用窗口级source/class weighted BCE、AdamW lr=1e-3/weight_decay=1e-4、clip1、
effective batch8、100 epochs最终epoch、seed20260909/10/11。几何编码端到端训练。
checkpoint保存optimizer、RNG、数据顺序生成器和输入/config/statistics身份；不从旧
V8模型恢复。三seed先平均logit，阈值0，source配对bootstrap10000次/seed20260909。

此实验不单独估计padding修正收益，不建设全视频队列，不提供帧/像素真值定位证据。
