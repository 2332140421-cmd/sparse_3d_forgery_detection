# 方格池化前后冻结特征读出诊断

状态：执行前协议。基于e4b3b02e337527bfe4b74216199f0c16dd08494f独立分支，不覆盖后续人工审阅记录。

仅加载RGB_DEPTH_PARTITION last epoch12/576，主模型eval/no_grad；只执行Local.encode与native_sample，不执行旧关系或像素头。训练原48段1152唯一帧，开发924唯一帧，全部保留。方格C直接读已PASS缓存。A为256维真实池化输入线性读出；B为A全部有效成员logit的格内均值；C为缓存C512的线性读出；D引用同口径正式方格关系final_dev，禁止重跑或替换。

训练每帧在几何有效范围按seed20260909从正负区各取最多128个不放回像素，两模型索引完全相同。各仅拟合一次训练StandardScaler+L2 LogisticRegression(C=1,solver=lbfgs,max_iter=1000,class_weight=balanced)，未收敛即停止，不做搜索。开发不参与拟合。AP按正GT帧→视频→四fake宏平均；不报告不同线性读出零阈值Dice/FPR。

三项必要小图检查通过：训练采样重现/隔离、native_sample索引与方格计数/回填AP、显式标准化FP32读出等价（包括高偏置特征）。沿用atol=rtol=1e-5。逐帧核对真实PTS、原生尺寸、方格顺序及n/g；参数/缓冲和冻结缓存hash前后不变。

独立入口：`PYTHONPATH=. python -u -m research_tools.rgbd_slice.grid_feature_readout_probe`。
产物：`derived/pvst_grid_feature_readout_probe_v1/`。计算与发布状态分开，原结果只读，不训练主模型、不重提前端、不追加实验。

执行记录：首次启动缺原rgbd-packages路径，在构造前退出1；补齐原环境后两个读出均收敛，但C折入原始权重在1/1024分数超差（最大绝对3.41402724e-5）。保留该退出1，不放宽容差。改为显式训练标准化后FP32线性打分；由于旧断言前未持久化C系数，需要同协议重新拟合，两个完成尝试分别记录，不作为参数搜索或额外实验条件。
