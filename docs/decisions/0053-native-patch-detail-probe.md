# ADR-0053：相同局部视野与预算下的原生细节输入对照

用户授权仅NATIVE_PATCH/LOWDETAIL_PATCH，各12845更新；原11训练/4开发来源及订单不变。
单次288原生坐标crop，real均匀；正GT fake按.5概率均匀选正像素并均匀选包含它的合法crop，
其余均匀，无面积门槛/拒绝。种子20260909，提前保存共享crop计划与每轮前100的BN计划。
LOW必须整帧按原work_hw/INTER_AREA缩小，再整帧INTER_LINEAR恢复，最后裁同坐标；GT原生不退化。
小尺寸原生黑padding/valid loss；模型归一化再pad14，输出先裁回288。共用通用DINO+新LoRA/head初值，
原BCE+Dice eps1、AdamW1e-4/wd.01/clip1/BF16；每组BN额外500前向，running BN评价。
只做末次原824帧完整原生盲滑窗，tile288/stride144/末端贴边，FP32 logits均匀平均后计分，
无GT选tile、无开发调参/best。推理批量8，资源不足只可降低推理batch，不改视野/预算/帧数。
只准备必要原生RGB/GT和少量真实前后向，随后自动训练两组；不提MoGe/DFormer，不改C/XYZ/关系。
本新输入策略仅扩展17.8全图384和17.9无训练范围；旧模型/默认/成绩保持。旧全图仅历史，
局部策略差异不能全部归为细节，不声称频率因果、三维/时序有效或80%目标。
