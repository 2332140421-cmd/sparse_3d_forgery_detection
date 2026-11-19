# ADR-0051：通用DINOv2＋官方AINPAINT空间头的ANAKIN基线

用户明确授权单组AINPAINT_SPATIAL，保持原扩充11训练来源、4开发来源和原订单前12845项。
使用官方DINOv2 ViT-L/14 registers通用权重，PEFT LoRA r4/alpha8/dropout.25、qkv/proj/fc1/fc2，
仅训练LoRA和原HighResContextHead。不得加载AINPAINT取证checkpoint、RAFT或时序/PVST支路。
完整工作RGB ImageNet归一化，右下padding至14倍数，x_norm_patchtokens严格网格检查；
官方头输出padding尺寸，先裁回工作网格，再原生bilinear恢复。pixel BCE+Dice各1、epsilon1。
seed20260909、新AdamW1e-4/wd.01/clip1，BF16/FP32，batch1无累积，5轮共12845更新。
每轮前100既定训练目标仅无梯度train前向更新BN，不reset统计，额外500次另计；
每轮固定共同28/新增16训练观察，最后824开发与固定24预览，无best/阈值搜索。
保存LoRA、完整head含BN、优化器/RNG和通用骨干身份；通用原参数必须不变。
只在本独立二维能力参照范围扩展root17.7的DFormer模型限制，不取代任何旧方法默认或结果。
本实验匹配数据/监督曝光/全画面计分，不是XYZ有无单因素消融，不复现作者裁剪和loss设置。
