# ADR-0041：冻结整片／片内固定方格的匹配定位训练

状态：接受，用户当前授权；起点72ac39fade910839fed5067aac1859bf378cb39d。
研究问题：整片表达上限低时，预先固定的片内方格能否改善实际定位？不把GT oracle当模型成绩。
直接调用者slab_grid_experiment；grid_localization只提供共享冻结池化、原参数分块读出和计数指标。
原pool的整片索引与MHA全查询不能直接支持本实验；仅新增必要辅助模块和独立runner，不改旧默认。

冻结来源RGB_DEPTH_PARTITION last epoch12/576，原train8/48段1152唯一帧，dev4/8视频924唯一帧。
每帧一次BF16局部RGB编码，同一score/value冻结权重。SLAB h=.125/Z/s/valid保持；GRID为原生
L=ceil(min(H,W)/16)，左上角对齐，保留边缘格与全部有效像素。各单元四头完整softmax，分块只累计数值。
整片C核对上次sets入口缓存（FP32 atol2e-5/rtol3e-5），GRID对照oracle逐帧单元/GT守恒。
缓存仅两组FP32 C512、绝对深度ID/PTS/n/g，小型统计；按视频落盘、只当前帧上GPU，无像素特征落盘。

读出为相同CURRENT_ONLY参数/初始状态（512/8head、绝对深度与dt=0共享位置MLP、C/R独立LN、差/乘积评分）。
无XY编码、历史、旧GRU/头；所有同帧非自身单元均可作K/V，N=1才用原空参照。
FP32 PyTorch efficient SDPA，查询块512，只生成块mask（SDPA True=可读，与MHA相反）。
训练每块non-reentrant checkpoint，完整Q/K/V及评分头梯度；不物化全帧N×N注意力/权重。

统一计数像素BCE：sum[w_pos*g*softplus(-s)+(n-g)*softplus(s)]/整段有效像素数。
train有效负/正像素比只算一次，两组同值；逐帧反向、每段一次AdamW，GT不作输入。
seed20260909、原12轮orders、48步/轮，共576步，lr1e-4/wd.01；不改阈值或调参，不用开发选best。
此处仅取代本新实验16.7片正负BCE，不改旧结果、入口或监督协议。

验收：小图旧MHA/新SDPA输出输入及参数梯度、mask/空参照、池化与逐像素BCE/AP核对；
最大GRID训练单帧所属完整24帧片段一次AdamW，若dev最大帧更大再推理该帧。验收权重丢弃。
语义/OOM失败立即停止。吞吐门槛预登记：按该最大片段更新时间×576保守估计方格训练若超过4小时，
视为本轮吞吐不合适而停止（资源执行门槛，不改变训练预算或缩小设计）。正常通过才两组正式训练。

评价原native几何有效范围，单元分数全回填；AP用同分数组n/g聚合精确计算，先核对sklearn逐像素。
fake正GT帧→视频→宏平均；无有效GT不补AP0。Dice/IoU/召回与real FPR阈值logit>=0。
逐轮开发、最后原48段回测。GT覆盖一致；GT上限分表。不读保留测试/新前端/权重/数据。
计算/发布独立，普通push必要代码与文档，缓存和权重不提交。无收益不追加格长、历史或训练。
