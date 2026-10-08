# ADR-0014: V8 格内三维形态匹配对照

状态：accepted，仅适用于局部几何pilot分支。

用户要求检查可学习编码前mean压缩是否丢失可用判别信息。采用两个完全等容量、
共用数据和padding修正的条件，仅改变真实局部XYZ与格均值复制。使用缓存中的
16×16 patch与两层3×3卷积、content mean/max，维持原128维格/GRU网络及窗口loss。

公共padding修正只在新模块实施；旧identity-fix模型和统计不改。协议与结果保存
独立目录，不能将新旧版本差异全部解释为局部三维贡献。恢复时严格绑定输入内容、
统计、条件与配置。具体合同见 `docs/v8/local_geometry_contract.md`。

本ADR在新分支范围内补充ADR-0013，不追溯更改旧实验，不替换V7设计。
