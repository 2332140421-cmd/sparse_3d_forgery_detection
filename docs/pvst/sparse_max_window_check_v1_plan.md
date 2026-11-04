# 修复后稀疏attention最大窗口：限定执行

本次仅复用 `research_tools.pvst.max_window_check --sparse`，900秒父进程限时，独立输出 `derived/pvst_v03_sparse_max_window_check_v1/`。不运行坐标审计、预览、单层GPU测试、评价或正式训练。

唯一输入：ANAKIN/train/real，video `782426ea0330c7fea684953c`，pair `an:4wdo37`；720×1280；历史[104,120)，目标[120,128)，24帧。fine h=1/8、coarse h=1/4，预期节点2,970,629/1,488,042。正式XYZ尺度保持null。

沿用seed20260909、BF16外层、逐head FP32稀疏核心及逻辑填充列修复，attention512/QKV8192/FFN8192/point32768/tile256，原外层checkpoint CPU卸载。模型参数17,821,672及state_dict结构保持；全部点、像素、邻居、128/256/512宽度和损失不变。

入口核对六层开关与真实执行证据（dilation、节点、边、拓扑字节、后端、完成head和填充列）；一次真实前向/反向/裁剪/AdamW后检查优化器状态及参数差值。零初始化导致上游零梯度允许，梯度不存在单独记录。检查后不保存权重。

本轮源码只增加配置接入、运行证据及独立报告；4项入口CPU检查通过，不重复已通过的单层GPU验收。运行前提交源码，运行期间不修改。结果仅分PASS/OOM/TIMEOUT/FAILED；失败不重试、不自动修复；旧记录及用户无关JSON改动保留。正式模型仍0/2。
