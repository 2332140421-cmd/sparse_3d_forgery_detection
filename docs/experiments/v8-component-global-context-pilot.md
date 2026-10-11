# V8 动态component全帧关系匹配对照

独立分支 `v8-component-global-context`，起始HEAD
`151e300549ab2f238e46b184da3647abdbe50647`。合同见
`docs/v8/component_global_context_contract.md`、ADR-0016。

两条件GLOBAL_COMPONENT与GLOBAL_CELL共享MEAN_REPLICATED、141维cell输入、
原固定格消息、区域节点属性/单层4头同帧attention、cell多前驱GRU与面积LSE。
唯一条件差异为动态component成员分区或active singleton格。旧component residual
不额外调用。每条件319,969参数，配对初始化、数据顺序、标准化完全相同。

只读identityfix和local_geometry合法V8缓存，801窗口身份逐项核对。
训练718窗口/122 source，验证83窗口/14 source（42 real/41 fake），每窗16时刻。
不运行任何前端、不读取V7数值资产、不改全视频接口、不用旧模型初始化。

AdamW lr=1e-3/wd=1e-4、clip1、有效batch8/物理batch8、原source/class weighted
窗口BCE，100 epochs/最后epoch，三个seed20260909/10/11。主评价先平均三seed
logit，再计算source-macro AUROC；配对source bootstrap10000/seed20260909，阈值0。

正式训练前19项本轮及直接相关测试通过。真实1LARL real/fake smoke完成训练更新、
optimizer保存恢复、评分与严格JSON评价落盘，重载误差0。GPU resident输入约2.73GiB，
smoke有效batch8 step=0.113885秒，allocated峰值6.447GiB；首步估计六模型约102.5分钟，
不是最终训练实测。未重复dense XYZ副本；仅预载mean/mask/shared。

输出目录：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_component_global_context_pilot_v1/`。

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
bash research_tools/v8/component_global_context_pilot/run_all.sh --resume \
  --baseline /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_identityfix_v1 \
  --reference /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_local_geometry_encoding_pilot_v1 \
  --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_component_global_context_pilot_v1
```

## 完成结果

状态COMPLETE：6/6正式模型，每个100 epochs和9000 optimizer steps；无失败、跳过或
训练重试。正式数值源码提交`1d4cf4395a98df0192942a0fd4dcb914b2966a66`。
首次smoke的protocol在基底151e300尚未提交新增代码时建立；其实际源码hash与正式
提交完全一致，`execution_versions.json`明确区分两个版本，不改写原protocol历史。

三seed先平均window logit，fake为正类，阈值logit≥0。训练363 real/355 fake，
验证42 real/41 fake；macro和pooled使用同一窗口总体，无单类别source或缺失评分。

| split | condition | macro AUROC | pooled AUROC | AP | Precision | Recall | F1 | ACC | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | GLOBAL_COMPONENT | 1.000000 | 0.999984 | 0.999984 | 1.000000 | 0.994366 | 0.997175 | 0.997214 | 363/0/2/353 |
| train | GLOBAL_CELL | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 363/0/0/355 |
| validation | GLOBAL_COMPONENT | 0.896825 | 0.856562 | 0.894838 | 0.868421 | 0.804878 | 0.835443 | 0.843373 | 37/5/8/33 |
| validation | GLOBAL_CELL | 0.904762 | 0.721835 | 0.697592 | 0.666667 | 0.731707 | 0.697674 | 0.686747 | 27/15/11/30 |

主比较GLOBAL_COMPONENT−GLOBAL_CELL：**−0.007936507936507948**；source配对bootstrap
10000次/seed20260909，95% CI **[−0.15079365079365079, 0.12698412698412698]**。
14 source中2提高、10持平、2下降；不支持稳定的macro优势，也不能认定等价或无效。

| seed | COMPONENT macro | CELL macro | 差值 | COMPONENT pooled | CELL pooled | COMPONENT AP | CELL AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260909 | 0.873016 | 0.833333 | +0.039683 | 0.889082 | 0.738966 | 0.897741 | 0.759483 |
| 20260910 | 0.920635 | 0.817460 | +0.103175 | 0.847271 | 0.727642 | 0.884035 | 0.738455 |
| 20260911 | 0.801587 | 0.801587 | 0.000000 | 0.836818 | 0.694541 | 0.875632 | 0.605577 |

逐seed macro均值为0.865079/0.817460（差+0.047619），不是预声明集成指标。
平均logit后AUROC改变排序，主差转为−0.007937；没有按更有利口径更换主结果。

| source | real/fake | COMPONENT AUROC | CELL AUROC | 差值 |
|---|---|---:|---:|---:|
| 00ZCA | 3/3 | 0.555556 | 0.000000 | +0.555556 |
| 0RJKT | 3/2 | 1.000000 | 1.000000 | 0.000000 |
| 0YOXV | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 129SP | 3/3 | 0.333333 | 1.000000 | −0.666667 |
| 13MWT | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 14TJT | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 170OQ | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 19ZJB | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 1CITG | 3/3 | 0.666667 | 1.000000 | −0.333333 |
| 1G9R7 | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 1JGRO | 3/3 | 1.000000 | 0.666667 | +0.333333 |
| 1K0SU | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 1L376 | 3/3 | 1.000000 | 1.000000 | 0.000000 |
| 1TW98 | 3/3 | 1.000000 | 1.000000 | 0.000000 |

## 实际组织、成本与解释

训练帧区域数[min,P25,median,P75,max]：COMPONENT为43/95/130/151/250，
CELL为160/160/160/192/256；验证分别43/96/124/150/188与160/160/160/160/192。
区域分区确实进入scatter pooling、有效节点mask和全帧attention；不是仅变更名称。
训练组-帧记录中1/2/3/4格为1130838/118445/62124/101075，验证为
127549/13837/6915/11304，最多四格限制生效。无效几何与无历史内容保留，padding排除。

正式训练6378.706秒（106.312分钟），正式all调用6447.816秒；评分21.687秒、指标
汇总2.772秒、案例图0.397秒。最大训练allocated显存7.150GiB（含2.729GiB预载输入），
新增存储约58.57MiB，仍有约319.16GiB可用。没有降样本、帧数、epochs或batch。
输入身份核对累计10.454秒、smoke计算保存检查累计2.631秒；首次smoke预载未独立
累计保存，不把正式调用耗时冒充整个实现任务墙钟。19项测试通过；6408条唯一有限评分、
六checkpoint及配对初始化/100轮顺序hash均已核验。进程正常退出，GPU空闲。

本轮主指标未获得动态分组优势，但辅助指标有一致正向信号：pooled/AP/F1/ACC分别
提高0.134727/0.197246/0.137769/0.156627；FP由15降至5、FN由11降至8。
三个seed的pooled和AP均提高。应保留这些实用信号，但不以辅助指标替代macro主结论；
当前证据不足以宣布稳定分组增益或概率校准改善。训练拟合很高而验证仍有差距，
不能据此指定过拟合、前端或训练不足为唯一原因。

历史局部几何与identityfix分数仅作独立背景，不是新attention的匹配消融；区域属性
与全帧关系路径都与历史模型不同。当前14 source验证已被多轮开发使用，不是sealed
test。本轮没有单独证明三维、attention或时间机制的贡献，没有帧/像素真值定位评价。
全视频streaming接口保持原状，连续帧训练需另行冻结范围，完成后不自动追加实验。

完整中文`report.md`及`evaluation/{summary.json,metrics.csv,per_source_metrics.csv,
per_source_differences.csv,train_loss.svg}`、`scores/window_scores.csv`保存在上述数据目录。
固定案例位于`evaluation/examples/index.html`：按role取首window ID、时刻索引0/8，
统一[-10,10]的三seed cell-logit均值色标。没有浏览器视觉验收，不把图像当定位证据。
