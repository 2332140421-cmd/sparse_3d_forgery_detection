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
