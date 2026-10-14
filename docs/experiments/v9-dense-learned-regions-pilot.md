# V9 稠密局部表示与学习区域 pilot

起点`9a9b6b8efc41a0c752b070c440ab3c6a69a4a663`，原分支干净且同步。
新分支`v9-dense-learned-regions`，按ADR-0021独立执行，不改旧产物。

244训练视频122source，冻结976×32帧片段，31232次帧使用、31144去重帧；
28验证视频14source全部27505帧。fake两段最长正区间内部、最长负区间中部、
首个边界跨越片段；real按同source实际PTS匹配，原生连续帧，不给fake上下文正标签。

RGB_QUERY/XYZ_QUERY/XYZ_PARTITION共享1373057可训练参数、seed20260909，
从头训练20轮。ResNet18保持原224输入预处理，但保留完整layer1空间图；
稠密XYZ不是格均值复制。XYZ含显式有效性，缺失内容像素仍编码/分类。
局部128+视觉128+RGB/UV融合256，64query归一化轴对照，区域attention和GRU256，
像素注意力解码及旁路，beta1归一化LSE。parent内全部片段统一帧类别平衡，
真实/伪造parent各半，完整source pair后AdamW step。

CPU测试保护边界、分区、匹配参数、因果性、分块/逐帧路径、梯度和重载。
GPU真实1LARL pair smoke更新丢弃；前端仅MoGe和冻结ResNet，无CoTracker。
初查训练28421帧、验证26588帧
需要重新生成稠密XYZ；其余只在身份、画布、配置核对后复用。

```bash
PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:$PWD \
  .venv/bin/python -u -m research_tools.v9.dense_learned_regions.run --stage all
```

输出`derived/v9_dense_learned_regions_pilot_v1/`；GPU bf16 autocast、FP32参数/损失。
空间microbatch最高32帧，non-reentrant checkpoint保留完整clip内时间梯度。
最终预训练资源smoke4.16s/source pair，3×20轮约8.45小时；这是外推，不是正式实测。
峰值allocated9.11GiB、reserved11.07GiB。真实稠密XYZ在16像素patch内存在非零变化，
GPU空间批处理与逐帧参考logit最大误差3.34e-5。4项机制测试通过。
旧完整视频缓存只有mean，不能恢复稠密输入。实测冻结视觉图GPU生成约0.223ms/帧，
读取缓存约0.303ms/帧且需额外21.93GiB，所以正式任务选择GPU在线生成，不扩建视觉缓存。
早期smoke块已有少量FP16视觉图不再消费，保留其RGB/XYZ/mask，避免重提合法几何。
不持久化256维像素embedding或软区域矩阵；真实成本/完成度以运行产物为准。

## 正式结果

数值执行版本`eb7dd69f5684ed7815f26ef36b68873049cbbf4d`。
3/3模型各20轮、2440 optimizer steps；三条件各完整评价28视频/27505原生帧。
84份响应唯一、有限，PTS/标签与冻结清单一致，视频LSE从帧分数复算误差0；
54张固定画面图身份与清单一致。最终4项机制测试通过，旧资产hash保持不变。

|条件|video AUROC|video AP|video F1|TN/FP/FN/TP|frame AP|frame AUROC|fake-video frame AP|real-video frame FPR|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|RGB_QUERY|0.622449|0.676159|0.480000|9/5/8/6|0.353566|0.671428|0.585068|0.257422|
|XYZ_QUERY|0.586735|0.680109|0.518519|8/6/7/7|0.387858|0.634282|0.608517|0.230528|
|XYZ_PARTITION|0.642857|0.735391|0.625000|6/8/4/10|0.397832|0.630117|0.601490|0.301336|

14source簇配对bootstrap10000次/seed20260909，pooled video AUROC：
XYZ_QUERY−RGB_QUERY=−0.035714，95%CI[−0.147959,0.035714]；
XYZ_PARTITION−XYZ_QUERY=+0.056122，95%CI[0,0.132781]，下界保存值为精确0。
没有三维视频排序增益证据；竞争分区方向正向但CI含0、FP增加。

预选18帧/条件的最大query质量中位数依次0.8385/0.3171/0.9951，
有效区域数中位数2.05/5.12/1.04。QUERY按仅用于描述的像素行归一化权重统计，
不是改写模型；PARTITION呈明显单query集中，不把点估计改善解释为有效物理分区。
当前版本暂不升级为论文主方法；不追加正则、seed、轮次或外部实验。

前端累计已记录块处理2911.70s；训练统计548.25s；正式epoch墙钟25792.19s
（约7.16小时，pair处理计时25743.70s）；评分/诊断586.34s。
最后完整恢复调用29794.75s，非所有前期准备/维护的精确总墙钟；早期smoke总成本未完整记录。
峰值allocated9.1614GiB/reserved11.1367GiB，新增约55.604GiB。
冻结训练976片段、31232帧使用、31144去重帧，重叠88次，无短片段；
58393帧新MoGe、256帧合格旧稠密缓存，共58649帧，不运行CoTracker。
完整报告、精确指标、配对与最终核验位于上述独立输出目录。
