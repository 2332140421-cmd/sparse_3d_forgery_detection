# 冻结像素消息开发检测对照：协议

基于766aa06ecd59494d6b0ff8b62d11b87045badebe，分支pvst-pixel-message-dev-check。
只加载原RGB_DEPTH_PARTITION last epoch12/576，原开发4对/8视频/924计分帧，完整BF16正常路径、真实PTS/原因果窗口和尾段左上下文。
问题：像素top-1%检测是否依赖当前集合上下文投影？新增薄CLI直接调用原evaluate并输出两条件评分，不修改原模型/正式入口。

- NORMAL：不改变pixel_context输出。
- PIXEL_MESSAGE_ZERO：forward hook在pixel_context仿射投影后zeros_like，含bias一起清零，原形状/dtype/device。保留集合关系、GRU、原帧头及局部RGB。
- 这是整个pixel_context(C+M)输出干预，与旧pixel_context(C)不同，不拆分C/M/bias自身贡献。
- 逐帧所有原生像素FP32 logit，k=max(1,ceil(.01×H×W))，最高k个均值；视频沿原log-mean-exp。GT只作标签，不用于选像素，不按valid删画面。
- 每条件只前向一次开发队列，尾段上下文不重复计分；无训练、探针拟合、前端、阈值/比例搜索或新样本。
- 检查资产/源码hash、正确模式strict加载、投影输出确为零、原帧头最大偏差及排序变化、权重/buffers不变、hook移除。
- NORMAL先与旧像素证据完整精度结果核对；允许报告微小重算差异，但若足以影响关键排序判断则不作消息归因，不自动重跑。

输出derived/pvst_pixel_message_dev_check_v1，报告原生924帧、分数CSV、逐视频/逐pair、资源及发布状态；不保存完整像素响应。
解释结合总体指标和多pair方向，单视频驱动/差异小不将成绩归给集合关系；关闭更好则如实报告。
仅冻结权重的推理依赖，不证明深度/时序/集合各自贡献，不替代匹配重训。计算状态与Git发布状态分开。

当前尚无本次结果。
