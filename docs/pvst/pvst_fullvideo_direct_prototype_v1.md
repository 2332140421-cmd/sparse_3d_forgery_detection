# PVST完整视频原始块原型 v1

本轮按用户固定设计实现，见ADR-0055及根契约17.12。独立模型与资源入口，不修改旧8槽模型。

冻结扩充空间epoch01的DINO/LoRA/官方head，原前向hook取得patch1024、predictor输入detail32和z0；
新patch投影128与XYZ几何调制/逐点MLP/FP32学习池化形成C。scale=12.880994319915771/h=.5/origin0。
两层D128/4头、FF256的SDPA，Q/K含块中心与真实PTS，V为内容；FRAME_ONLY逐帧，FULL_VIDEO直接全片双向。
全部帧R投影32回填，与冻结detail32融合，零初始化delta末层叠加z0。无帧分类、全块均值门控、GT输入。
全片合法监督帧BCE＋Dice(eps1)平均，一次backward/AdamW更新。40y1x9 fake6–9只上下文及输出。

本轮只实跑完整低token病例，两模式各2次可丢弃优化更新；不是正式训练或效果比较。
中高病例由用户终端启动，不轮询监控。实测与命令在完成短预检后补记。
