# V8 按source隔离的冻结测试

起点55905ae7a7d6e4724bdce300e5618354dc5d7a3c；分支v8-independent-source-test。
独立输出`derived/v8_independent_source_test_v1`。本次无训练、拟合或阈值搜索。

## 资格和冻结

纠正此前准备文档的保守排除：review_manifest字段及README只记录媒体/溯源，
人工作业字段为空，不能凭登记排除。只查直接相关V7测量/评分及当前V8输入清单，
不做全服务器历史审计。19个本地官方test source中，0DVVD/0FM93/0KTWY/0PU21
有实际历史测量或检测分数，排除。0HV07曾运行前端但没有有效结构/模型评分，
periodic训练按paired_eligible筛选不纳入它；保留，不把缺失占位算性能使用。
历史模型未保存完整输入指纹的范围不能完全追溯，因此不称严格从未接触。

全部15个合格本地source纳入，30视频；hash seed20260909排序，fake官方记录
稳定排序选唯一可用变体。清单在评分前保存，预计19607元数据帧；不补到20对，
不按分数、几何成功率或fake观感筛选。官方Charades scene/actions支持室内自然
活动范围，未声称人工核验了全部动作。视频内容SHA与source/role共同标识，不使用
pair_id单键。名单和资格证据保留数据目录。

Vidu15个fake，训练只见wan/fcvg。这是同数据族独立source与未见生成器的联合迁移，
不是跨数据集，不能独立分解两个因素；小子集不代表完整benchmark。

## 数值路径

一次原生全视频前端，MoGe/CoTracker/ResNet、standardizer/grouping与原连续帧
实现不变。无抽帧、截片或未来上下文；全部非padding格（包括无几何/无历史）参与。
直接复用`full_video_global_context.runner.process`，只参数化总视频计数与禁用
已通过的重复大smoke，额外按CUDA events记录各模型设备耗时；原入口默认行为保留。

明确九份权重：COMPONENT_ADAPTED=GLOBAL_COMPONENT epoch5×3seed；
COMPONENT_WINDOW=GLOBAL_COMPONENT epoch100×3seed；CELL_WINDOW=GLOBAL_CELL
epoch100×3seed。strict加载并核对condition/seed/epoch/hash，eval冻结。
真实全帧attention和原多前驱GRU，hidden跨64帧块；各seed独立完整面积LSE，
随后平均最终video logit，阈值0。无模型平台或新前端。

主要结果为ADAPTED video AUROC及自身95%CI；辅助AP、FPR、Recall、Balanced
Accuracy、P/F1/Accuracy/混淆矩阵及逐seed。仅两配对：ADAPTED−WINDOW与
COMPONENT_WINDOW−CELL_WINDOW；前者适配影响，后者同训练阶段分区对照。
source簇bootstrap10000次/seed20260909，同次采样保留真假并用于全部组。
不将适配−CELL称匹配消融，不做无真值的帧/像素定位。

## 验证与执行

3项本轮定向测试通过；旧梯度、padding、连续状态验证只读引用，不重跑。
正式入口另做两帧九模型加载与连续/块输出核对，容差2e-5。

```bash
PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$PWD .venv/bin/python -u -m research_tools.v8.full_video_global_context.independent_test --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_independent_source_test_v1 --resume
```

唯一任务，输出目录flock防重；TERM在当前64帧块保存后停止。成功视频不重复处理。
实际结果以全部30视频、九权重评分、评价及report完成后的final_status为准。
本轮已完成，最终状态COMPLETE，退出码0，无残留实验进程或screen。

## 实际结果

15 source、30视频（15 real/15 fake）、19607实际连续帧；九份已有权重全部评分。
全部原生PTS，估算时间戳0，实际帧数与元数据一致；前端19607帧各执行一次。
270条seed分数加90条集成分数，360个唯一键全部有限。30份视频内容SHA均唯一，
与当前训练/验证SHA无重叠。无新训练、统计拟合、下载或阈值调整。

|三seed最终视频logit均值|video AUROC|AP|FPR|Recall|Balanced Accuracy|Precision|F1|Accuracy|TN/FP/FN/TP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|COMPONENT_ADAPTED|0.791111|0.830258|0.266667|0.600000|0.666667|0.692308|0.642857|0.666667|11/4/6/9|
|COMPONENT_WINDOW|0.817778|0.833092|0.600000|1.000000|0.700000|0.625000|0.769231|0.700000|6/9/0/15|
|CELL_WINDOW|0.720000|0.714443|0.400000|0.800000|0.700000|0.666667|0.727273|0.700000|9/6/3/12|

主方法自身source簇bootstrap95%CI为[0.6666666666666666, 0.9067777777777762]。
两个预声明AUROC配对差（10000次/seed20260909，同次source采样保留真假）：

- ADAPTED−COMPONENT_WINDOW：−0.026666666666666616，CI
  [−0.13333333333333341, 0.053333333333333344]；三个seed差
  +0.053333、−0.044444、−0.013333，方向不一致。
- COMPONENT_WINDOW−CELL_WINDOW：+0.09777777777777774，CI
  [−0.0444444444444444, 0.20000000000000007]；三个seed差
  +0.075556、+0.022222、+0.075556，均正但CI跨0。

每seed测试AUROC（按20260909/10/11）：适配0.813333/0.728889/0.764444；
窗口COMPONENT 0.760000/0.773333/0.777778；窗口CELL 0.684444/0.751111/0.702222。
各seed的全部辅助指标见数据目录report.md与evaluation/metrics.csv，不挑最好seed。

适配FP由9降至4，但FN由0增至6，Recall与F1下降；不支持适配带来稳定额外收益。
逐source单对真假排序：适配−窗口提高/平/降=0/14/1（55NRK下降），窗口COMPONENT−
CELL=0/14/1（0HV07下降）。pooled排序增量不等于逐source普遍改善；可能涉及跨source
分数排序，不能独立分辨尺度与表示因素。保留全部不利样本，不事后删除0HV07。
可保留当前架构作为初步迁移候选，但不称5轮适配已经验证有效，也不根据本测试
切换模型/seed。值得另行决策有限的真正跨数据集冻结验证，本轮不执行。

测试fake均为官方Vidu（15/15在test@vidu.txt中匹配），属于训练未见生成方法；
场景6类：Bedroom6、Closet3、Living room2、Pantry2、Kitchen1、Home Office/Study1。
只代表这个小子集；source与生成器同时变化、历史准备性接触及元数据审核不等于
严格盲测，不能宣称跨数据集泛化或时空定位性能。

## 实际成本与交付

运行墙钟6869.164秒（1.908小时）；前端阶段计时之和5939.510秒，九模型推理墙钟
598.369秒，评价24.754秒；provider初始化2.753秒另有记录。分组CUDA events区间
累计适配206.960秒、窗口COMPONENT193.044秒、窗口CELL193.072秒，不作为额外墙钟相加。
本轮下载0秒/0字节；紧凑缓存984920179字节（0.917GiB），新增目录生成时快照
1117440299字节（1.041GiB）；GPU峰值allocated1.973GiB、reserved2.996GiB。
约0.350秒/帧，较旧估计略高（九而非六权重），未明显超出约20对预算预期。

3项定向测试通过；两帧九权重连续/块输出最大误差2.773147400603193e-8，容差2e-5。
未重跑旧大smoke、梯度测试或前端可靠性评测。代码数值路径运行中固定。

数据目录保留`report.md`、`video_manifest.json`、`eligibility_audit.csv`、
`eligibility_evidence.json`、`protocol.json`、`loading_check.json`、`costs.json`、
`evaluation/{summary.json,metrics.csv,video_scores.csv,per_source_metrics.csv}`及final_status。
模型/缓存/视频/长日志均不进入Git，不生成ZIP，不自动启动后续实验。
