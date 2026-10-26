# V8 完整联合定位协议 v1

2026-10-26；ADR-0022。以用户本轮完整指令为准。

目标为公开定义的编辑区域，不是肉眼可见异常。用户已人工确认核验批ANAKIN的mask覆盖
编辑位置、无明显错位且有像素级细节；取代此前将本批笼统概括为粗矩形的预审判断。
不改变3项帧号/帧数未解决的技术记录。旧人工接口标签仍记录visible_forgery_anomaly，
本轮标签为edit_region，不混用。

科学问题：相同RGB/XYZ、损失与预算下，规则component上下文是否改善编辑区域定位？
CELL_LOC和COMPONENT_LOC唯一条件差异为区域成员分区；都含三维信息。
复用GlobalContextModel及VideoTraining的完整可微路径、128维hidden、cell消息、
全域4头attention、多前驱GRU和真实delta_t。新增返回hidden接口默认关闭，不改变旧调用。

结构路径256；像素路径512。RGB两层3×3卷积32通道，第一层stride2；512RGB另有1×1的8通道skip。
格均值压缩前XYZ构成7通道：训练统计归一化绝对XYZ、相对格均值XYZ、valid状态。
两层3×3几何卷积32通道，256分辨率。无效XYZ明确占位，RGB保持；每层content门控。
与128通道GRU hidden融合，经1×1到32、3×3到32；上采样512、拼接RGB skip、
3×3到16、1×1输出残差；最终为上采样cell logit+残差。所有聚合从最终像素logit进行。
padding不进入卷积有效输入、区域池化/有效attention键、损失或指标。

视频LSE为全部内容像素/帧的面积归一化LSE，beta1。L=L_video+0.5L_frame+L_pixel。
每视频等权；L_frame为所有有效帧平均BCE；L_pixel为标注帧内容像素BCE与Dice均值，Dice平滑常数1。
空GT保留BCE，Dice损失为0，不用空背景提高正GT Dice主指标。
AF fake只用视频与时间区间，像素监督缺失；两库真实原片是已声明全背景空间负例。
ANAKIN edited逐帧mask转帧标签；区间端点采用闭区间，按源视频归零后的实际PTS匹配。
GT最近邻到512；预测先回映原图再以logit0分割。非空GT帧IoU/Dice先帧均值后视频均值；
空GT另报FP像素比例。pixel AP每5轮和最终逐帧精确计算后先视频平均，不称pooled/实例AP。

seed20260909。AF Charades Wan/FCVG各train12、dev4 source pair；ANAKIN train24/dev8，
优先has-masks官方train/validation，以URL分组；排除已知不符/未解决样本。
新队列不称旧244/28复现。候选顺序和技术失败替补规则预先固定，正式训练后不得替换。
前端仅取冻结队列全部实际帧，不降采样；统计/grouping仅训练数据，共用于两条件。
无高维像素embedding缓存。可微分块checkpoint保持hidden和梯度，无截断反传。

AdamW lr1e-4 weight_decay1e-4，clip1，最多30轮，dev ANAKIN Dice连续8轮未超过0.002改善早停。
最佳按最高dev Dice选同一epoch全部指标；0.002仅用于patience，不阻止保存更高Dice。
一个完整pair一次更新，真假各0.5；A/B交替、每轮确定性同序。
开发不缩减，每库固定4个train_monitor pair；日志online_train与monitor分开。
只运行两条件各一seed。不搜索阈值或损失权重。

先用两库各一个完整训练pair做真实更新/评价资源smoke，丢弃smoke权重。
若外推两条件30轮训练及常规评价超过48小时，正式开始前两库train同步减至16；仍超过则12。
精确pixel AP的额外成本未在资源smoke实测时单列未知，不计为0。
这是预注册的成本阈值，只减训练数，不改模型、分辨率、完整时序或监督；dev仍各8。
冻结初始参数、最终清单与统计身份，保存best/last及optimizer、AMP、RNG、epoch/pair位置。
每pair边界可精确恢复，半pair意外中断从上次完整pair重算，不能跳过未更新梯度。

入口run负责准备、smoke、训练、评价和自动报告；提交代码/清单后独立tmux运行。
不持续轮询epoch，不自动加预算。完成后再审阅结果并提交小型摘要。

报告汇总为两个数据集的video指标等权平均，frame AUROC/AP也按数据集等权平均；
真实帧FPR先在真实视频内取比例，再视频平均。原始尺寸空间空mask像素FP单列。
报告用train_monitor减dev Dice超过0.15标记明显差距，仅作描述性过拟合提示，不参与选模。

检测frame指标在每个数据集内pooled全部有效帧，video指标按视频，dataset_macro等权平均；
pixel混淆矩阵是已知原始尺寸GT像素的pooled计数，不替代正GT视频均值主指标。
视频/帧Precision无预测正例时按sklearn zero_division=0固定定义，未计算指标保留null和原因。
正式输出的曲线依赖Matplotlib3.10.7；预训练前端与PyTorch版本保持已验证版本。
