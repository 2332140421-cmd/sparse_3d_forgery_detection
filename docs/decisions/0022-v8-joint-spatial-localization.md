# ADR-0022：V8双分辨率编辑区域联合定位

状态：接受；日期2026-10-26。

依据用户本轮指令，采用两个完整CELL_LOC/COMPONENT_LOC条件，保留规则component、
原V8完整结构与跨块梯度，增加256dense几何/512像素定位分支；从配对随机初始化联合训练。
具体冻结结构、标签、损失、队列、预算及指标见`../v8/joint_spatial_contract.md`。

取代ADR-0021当前阶段的仅接口准备限制，以及spatial_localization_plan中可见异常目标、
256单分辨率候选和先骨干再定位的建议。ADR-0021的历史smoke和人工标签不重命名或删除。
用户人工确认本批ANAKIN mask覆盖编辑位置且有像素细节，取代此前粗矩形预审概括。
技术上未解决的3项帧对应仍排除。旧模型forward默认行为、历史训练和分数保持不变。

工程使用data/model/metrics/run四个小模块，调用方为本轮CLI，输出冻结manifest、
外部逐视频缓存、两模型checkpoint和报告。旧缓存无本轮512RGB/dense几何/空间GT，
不能替代新缓存；旧推理inference_mode不能用于可微训练。无通用训练框架。
