# V8检测／空间定位输出分离对照 v1

2026-10-27；ADR-0023；以本轮用户完整指令为准。

问题：共享像素输出是否限制定位可拟合性？只比较完整COMPONENT_LOC的COUPLED与SEPARATE_HEADS。
共有256结构、512像素RGB与skip、真实dense XYZ、规则component、局部消息、全帧attention、
128维因果GRU与完整视频梯度；不detach共享特征。像素coarse+residual不改变。

COUPLED保持原pixel→beta1内容面积归一化LSE→frame/video。
SEPARATE_HEADS从[T,256,128] GRU hidden经独立Linear128→64/ReLU/Linear64→1，
按原cell内容面积作beta1归一化LSE，再按帧内容面积聚合video。
不读最终pixel logit，不加回/门控pixel，也不按video阈值隐藏mask。
输出分离仅解除末端直接耦合，共享模块仍受两个任务的共同影响。

共同参数使用seed20260909，与原paired_initialization重新随机构建逐参数SHA相等后加载。
新检测head从coarse head深拷贝，参数独立，不改变共有随机数消耗。两条件共同初始化逐项相同，
各362658与370979参数，增加8321；不加无用模块凑容量。

队列只取原冻结manifest明确train_monitor的16视频（每库4pair），dev原32视频全部保留。
按原身份、帧、PTS、标签、padding读取全部帧；无新采样、下载、解码或前端调用。
只读引用原cache，缺少fitted component/cache或身份不符即报错，不补算。
原统计/grouping固定，来自原96训练视频，不重新拟合；本轮只有16视频更新参数，无dev拟合。
缓存复用检查canonical身份、数组header/帧数、PTS、component sidecar身份/SHA及mask SHA；
不重新哈希全部大型原始数组。记录原产物小文件hash及checkpoint大小/mtime，结束复核未覆盖。

L=video BCE+0.5frame BCE+pixel；pixel为已知帧内容像素BCE+非空GT soft Dice，
平滑1，空GT Dice0且保留BCE；AF fake像素监督跳过，real明确背景，ANAKIN fake公开编辑区域。
每视频归一化，每pair真假0.5各一次反传后step；两库交替，逐轮顺序固定SEED+epoch且两条件同序。
AdamW1e-4、weight_decay1e-4、clip1，像素BF16/结构FP32；固定30轮，不早停，不加预算。
主比较同第30轮train/dev；最高dev ANAKIN Dice的best仅补充，同epoch报告所有任务。

原入口run.py复用同一训练、恢复和原尺寸评价，默认旧条件和计算路径保持。
每轮评价完整16训练/32dev，保存全逐视频/帧结果、online与轮末评价分开；日志增pixel BCE/Dice，
总pixel仍按原表达式计算，不因日志拆分改变loss。pixel AP每5轮（两个集合）及最终精确计算，其他轮null。
视频/帧固定logit>=0，空间预测先回映原尺寸再阈值0；正GT帧均值→视频均值，空GT FP单列。

best/last对全部ANAKIN train/dev在固定25/50/75%保存原始float32压缩pixel logit、
GT512/nativeGT、content512、RGB512、PTS、身份、映射；不存全部帧pixel张量。
两条件同一绝对logit色阶[-6,+6]仅影响显示，二值0；索引含real，不用检测预测门控。
逐帧检测分数、梯度裁剪前范数、lr、更新数、耗时、完整帧数和峰值allocated写盘。
每pair checkpoint保存model/optimizer/AMP/RNG及epoch/pair位置，跨块hidden不截断。

新增head_decoupling.py仅由当前run调用，负责固定子集/只读资产核查、两pair smoke和对照报告。
现有run没有本轮只读准备和固定输出对照，因此该小模块必要；不另建训练循环或通用框架。
smoke在两条件上各完成AF与ANAKIN完整pair更新/评价，丢弃其权重；预算基于实测，不删模块。
先测试并commit/push，再独立tmux运行；Python自动完成两条件、报告后停止。

本合同仅取代ADR-0022在本新对照中的“全部检测来自pixel”和“原96视频/早停”限制。
原实验资产/结果/合同保持历史意义；不将本轮与历史COMPONENT_ADAPTED成绩直接作模块因果比较。
