# ADR-0029：PVST-Forensics v0.2

接受；2026-11-02用户完整规格保存于[设计合同](../pvst/design_v02.md)。

问题：全部像素先逐点学习、再按空间组织及学习局部时空变化，能否支撑检测与定位？
新方法与旧V8格均值/固定格、V9查询分区不同，不重命名旧模型。
精确规格以设计文件为准；UV/XYZ匹配、双尺度三层因果attention和原生定位旁路均不可削减。

直接入口research_tools/pvst；原生全像素计算及跨tile全局scatter不可被旧网络复用替代。
只复用原始媒体、标签、真实稠密XYZ、兼容无LoRA token及通用指标函数。
单seed20260909，32训练/16开发视频，训练仅frame BCE与ANAKIN空间BCE+softDice。
AF dev视频AUROC唯一选best，最多30轮，patience8；没有video BCE或测试集访问。

用户允许AMP、checkpoint、时间halo与等价空间tile。24GB无法容纳时报告真实测量，停止正式训练等待硬件决定。
实验状态与Git发布分开，保留真实退出历史；不自动增加其他条件或后续实验。
本ADR仅在独立PVST新方法中取代早期V7/旧V8默认方法限制，旧ADR不删除、旧实验不覆盖。
