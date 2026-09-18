# V7 冻结视频特征与结构信息最小三条件 pilot

本实验的可执行入口是 `research_tools/v7/visual_structure_pilot/run_all.sh`，
结果写入数据目录 `derived/v7_activityforensics_visual_structure_pilot_v1/`。
它只读取 observation-support pilot 的 R17 结构特征和 source128 的已冻结窗口/视频
身份，不修改正式检测链，也不运行前端、tracking、depth、pose 或 segmentation。
正式数值以数据目录中的 `report.md`、`evaluation/summary.json` 和
`visual_feature_manifest.json` 为准。本文件只记录代码入口和研究边界，
不复制运行时的大型特征、权重或模型文件。
