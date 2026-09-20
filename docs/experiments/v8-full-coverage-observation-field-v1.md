# V8 Full-Coverage 3D Observation Field v1

本实验在 `v8-full-coverage-observation-field` 分支独立实现。协议、真实产物、
阶段状态和运行命令见数据目录下的 `protocol.json`、`stage_status.json` 与
`report.md`。V7 的粒子/特征/模型缓存不作为 V8 输入；当前开发队列仅复用窗口
身份和原始 ActivityForensics 媒体。

报告必须区分：原始窗口、前端成功/失败、全覆盖格数量、缺失状态、训练模型和
评价覆盖。若官方 MoGe/CoTracker 依赖或资源不可用，状态保留为 BLOCKED/PARTIAL，
不以旧前端或随机特征替代。
