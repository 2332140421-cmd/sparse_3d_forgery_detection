# 固定decoder BN运行统计：单seed匹配对照合同

2026-10-29，用户授权，ADR-0025。基础方法沿用[原合同](pretrained_spatial_context_contract.md)，
只改变BN统计策略，不增加模型条件或辅助loss。

- 分支`v8-spatial-context-frozen-bn`；输出`derived/v8_pretrained_spatial_context_frozen_bn_v1`。
  原预训练实验`derived/v8_pretrained_spatial_context_pilot_v1`及原joint缓存均只读。
- 两条件仍为SPATIAL_TRANSFER和SPATIAL_3D_CONTEXT，标记FIXED_BN。
  只从同一官方AINPAINT初始化开始；不加载旧best/last继续训练。
- `freeze_decoder_bn=True`固定decoder五层BN的running_mean/running_var/num_batches_tracked；
  每次model.train()后BN仍eval。weight/bias仍可训练，其他decoder参数不冻结，
  Dropout保持原训练开启/评价关闭。不用momentum0代替，不校准、不重置、不换归一化。
- 默认False保留旧配置行为。每轮及smoke/resume检查BN模式、buffers逐项相等及仿射可训练。
  checkpoint重算依旧保留完整梯度，不能截断视频或移除dense XYZ、component、全帧关系、GRU。
- 原ANAKIN训练24pair/48视频6178帧、开发8pair/16视频1806帧全部处理；
  原96训练统计/grouping复用，无AF更新、test选模、重新挑样本。
- `features_root`直接指向旧视觉缓存，默认仍为本输出/features。
  新路径只恢复官方decoder，mmap读取官方checkpoint必要张量，不构建DINO或下载权重。
  逐视频检查身份、PTS、shape/dtype、header和旧完整验收SHA记录，不重扫大文件。
  原小型结果/元数据SHA及大文件size/mtime在结束再次检查，不复制20.85GiB或130GiB。
- 完整epoch0指标按SHA复用；固定95qlg9 real/fake中帧用原2帧chunk、BF16输出
  对照保存的epoch0 logits，并验证增强零P输出相同，不重复完整epoch0精确AP。
- seed20260909，AdamW lr=wd=1e-4，clip1，BF16/结构FP32，chunk2；
  全视频空间BCE+非空GT Dice、视频平均、pair真假各0.5；每完整pair一次更新。
  相同确定性pair顺序，最多30轮，dev Dice patience8/min_delta0.002，best最高且允许epoch0。
- 两模型各做原首完整ANAKIN pair两次更新smoke，验证BN仿射/卷积更新与第二步几何/GRU梯度，
  buffers精确不变；丢弃smoke，正式恢复共同官方decoder和原未训练分支初始化。
- 原轮末train/dev、AP日程、best/last完整AP、固定25/50/75% F32样例与绝对色阶不变。
  增加每轮简洁BN断言，不打印完整buffers。
- 自动报告原BN与FIXED_BN各自best/last、三组差值、8条配对与预先固定弱6条、
  同轮history曲线、GT覆盖/背景/固定边界、实际训练成本；原成绩只引用不重推理。
  弱6条为除原强例4rvro9/4wgx37外的6条开发fake，不改变主指标或样本。
- 测试/smoke通过后先提交推送，再独立tmux单入口执行，不逐轮Codex轮询或调参。
  本轮仅两个条件/一个seed，不追加BN策略、模型或数据搜索。

复用原SpatialModel、资产入口、训练循环、恢复、指标和预览；新路径选项仅供当前
固定BN匹配实验直接调用，输出为独立实验目录，不建立新训练框架。
