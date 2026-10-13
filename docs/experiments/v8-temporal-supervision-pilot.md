# V8 时间区间辅助监督单seed pilot

起点`d9036e62d4856f6386e7c11d840994669d7cfeef`，干净分支创建
`v8-temporal-supervision-pilot`。独立输出`derived/v8_temporal_supervision_pilot_v1/`。
仅增加聚合帧BCE，固定系数0.5，原模型319969参数和视频BCE不变。
复用`VideoTraining.video(return_cells=True)`和原source-pair训练/恢复循环；仅增加
显式单seed及训练step参数，旧入口默认行为不变，不复制pipeline或全量核验/提取缓存。

训练244视频/122source/206779帧，官方正帧32181（约15.56%）；
验证28视频/14source/27505帧，正帧4046（约14.71%）。fake全部正负帧均存在。
原视频路径+source+EXACT精确匹配官方区间；时间原点均0，缓存原生PTS，闭区间并集。
不以平均FPS推标签，不以无效观测删帧。区域/模型/统计/优化/完整梯度均保持。
seed20260909、5轮、chunk64、最后epoch，从原窗口epoch100初始化，不从旧5轮继续训练。
旧对照同seed5轮，初始SHA/统计/训练清单一致，评分时复现原保存视频logit。

先短序列跨块梯度及一个1LARL真实完整pair训练检查，正式模型重新初始化。
随后一个正式模型、28验证视频新旧相同接口帧/视频评价与报告；不重训旧模型。
帧AP非视频AP/时间检测mAP，无空间指标或外部扩测。结果完成后补入，不预填COMPLETE。

```bash
.venv/bin/python -u -m research_tools.v8.full_video_training.temporal --resume
```
