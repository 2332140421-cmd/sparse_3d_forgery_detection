# V8 局部几何编码匹配对照 pilot

状态：COMPLETE。起始 HEAD `a50ef86ed8b0b3f7623cd0ad0cc40fc416a774a6`；独立分支
`v8-local-geometry-encoding`；正式训练源码提交 `3cffac71c1cd5a72132df55fac2809863b888ad1`。
旧 identityfix 六模型、标准化和原始缓存保持不变，无 V7 数值资产复用，无前端重跑。

## 冻结方法与数据

主问题：同样的粗 components、RGB、时间路径与训练规则下，格内真实三维形态是否优于格均值复制？
两条件共用逐像素 content/valid mask、77维格输入、训练统计、component 分组与前驱关联；
只改变7通道几何输入：DENSE_LOCAL 为 absolute XYZ、relative XYZ、geometry-valid；
MEAN_REPLICATED 在有效位置复制格均值，relative XYZ 为零。无效几何使用有限占位及独立状态，
不删除真实内容。两层3×3卷积（7→32→32）、content mean+max 得到64维，拼接为141维；
hidden=128，空间/component residual、GRU及面积聚合沿用 V8。两条件均为236,513参数。

公共 padding 修正仅作用于本 pilot：content-only 标准化/几何均值/分组统计；
纯padding不作为合并桥梁；空间消息及度数同步mask；component pooling排除padding；
前驱排除padding后重新归一化；面积零位置严格排除LSE。
这些不是冻结 ResNet18 对letterbox画布完全无响应的保证，也未单独估计padding修正收益。

801窗口身份、标签、原视频SHA、源帧与PTS逐项保持一致。
训练718窗口/122 source（363 real/355 fake），验证83窗口/14 source（42 real/41 fake）；
每窗16个冻结时刻，source无交集。有效内容格-时刻训练1,958,400、验证221,184；
全部无几何的内容格-时刻为0，但训练像素中14,422个几何无效观测仍保留状态。
验证51,757,056个内容像素-时刻几何均有效；这只说明输入支撑，不是精度结论。

AdamW，lr=0.001，weight_decay=0.0001，有效batch=8，microbatch=8，clip=1；
原source/class加权窗口BCE，100 epochs，最后epoch，无验证选模或阈值搜索。
seeds=20260909/20260910/20260911；每seed两条件共享参数初始化hash和数据排序规则。
标准化只在训练内容上拟合；两条件共用距离/坐标统计。
训练原生CUDA，无新增混合精度。恢复包含optimizer、loss、RNG、排序生成器和完整输入/配置身份。

## 结果

三seed先平均logit，再计算指标；固定logit≥0，fake为正类。
macro与pooled使用相同窗口总体，主指标为验证source等权AUROC。

| split | condition | macro AUROC | pooled AUROC | AP | Precision | Recall | F1 | ACC | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | DENSE_LOCAL | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 363/0/0/355 |
| train | MEAN_REPLICATED | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 363/0/0/355 |
| validation | DENSE_LOCAL | 0.738095 | 0.663182 | 0.721144 | 0.740741 | 0.487805 | 0.588235 | 0.662651 | 35/7/21/20 |
| validation | MEAN_REPLICATED | 0.817460 | 0.694541 | 0.726050 | 0.625000 | 0.487805 | 0.547945 | 0.602410 | 30/12/21/20 |

主要配对DENSE_LOCAL−MEAN_REPLICATED：**−0.07936507936507936**；
source paired bootstrap（10,000次，seed20260909）95% CI
**[−0.2777777777777778, 0.10317460317460317]**。
14 source中3提高、6持平、5下降；不是一致正向增益。

| seed | DENSE macro | MEAN macro | 差值 |
|---|---:|---:|---:|
| 20260909 | 0.650794 | 0.761905 | −0.111111 |
| 20260910 | 0.825397 | 0.650794 | +0.174603 |
| 20260911 | 0.793651 | 0.746032 | +0.047619 |

两个seed正向、一个负向；逐seedAUROC平均与平均logit后AUROC不是同一统计。
集成pooled AUROC下降0.031359，AP下降0.004906；F1增加0.040290来自FP由12降至7，
FN仍为21，不抵消主排序指标下降。

## 验证、成本与结论

7项定向测试通过：条件信息隔离、形态/邻接响应、padding及缺失路径、有限非零梯度、
patch分块/保存重载、microbatch梯度等价、恢复身份与排序状态。
真实real/fake smoke保存重载误差0；batch8首step0.540685秒、allocated峰值3.637GiB
（首step不当成稳态吞吐）。六模型均100个有限loss记录，optimizer/RNG/排序状态可读取；
6408条逐seed/集成、训练/验证评分有限且唯一。无失败或中断恢复。

GPU RTX4090D；正式训练6,745.396秒（112.42分钟），模型前向评分55.255秒，
评价汇总1.491秒；准备累计465.161秒，smoke累计3.771秒，报告0.003秒。
正式all调用6,832.986秒（包含恢复输入核验与评分，不等于完整准备成本）。
最大allocated显存4.350GiB；平均epoch11.242秒，约63.87训练窗口/秒（含backward等epoch工作）。
未将resume调用耗时冒充完整训练成本；没有重新提取MoGe/CoTracker/RGB。

工程完整跑通，但本pilot**未获得真实格内三维形态优于格均值复制的验证证据**。
区间跨0不证明完全无效或等价；seed不一致、不足以断定四维压缩、前端噪声或过拟合为唯一根因。
训练集拟合良好不等于跨source泛化。当前验证已多轮开发使用，不是sealed test。
旧identityfix FULL仅作历史参照，新旧同时有padding与编码器差异，不能作纯局部形态因果比较。
本轮未验证全视频连续帧、全帧component图、帧/像素定位或独立未知source泛化。
完成后结束，不追加条件、调参或扩数据。

## 产物与恢复入口

数据目录：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_local_geometry_encoding_pilot_v1/`。
`protocol.json`、`inputs/manifest.json`、`support_audit.json`、`standardizer.json`、
`grouping_config.json`、`training_weights.json`记录身份/支撑/统计。
`models/fold_models.json`、各模型loss CSV、`scores/window_scores.csv`、
`evaluation/{summary.json,metrics.csv,per_source_metrics.csv,per_source_differences.csv,train_loss.svg}`、
`stage_costs.json`、`final_status.json`、`report.md`、`review_summary.md`保存完整结果。
权重、NPY/NPZ及缓存不提交Git。

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
bash research_tools/v8/local_geometry_pilot/run_all.sh --resume --stage all \
  --baseline /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_identityfix_v1 \
  --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_local_geometry_encoding_pilot_v1
```

身份相符的最终模型与评分可复用；该命令是恢复说明，本轮完成后未再执行。
