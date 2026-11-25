# ADR-0057 官方MumPy完整定位结构的ANAKIN迁移

状态：用户明确授权，2026-11-26。

## 研究问题与最小实现

本轮建立可追溯的已发表视频时空定位结构参照，不调整旧PVST关系探针。`research_tools/mumpy_anakin/model.py`只适配官方Encoder/Decoder初始化、连续logit与官方损失/参数组；`data.py`提供原89/19清单的同视频三帧RGB/中帧GT；`experiment.py`组织用户终端启动的单组训练、评价、恢复和报告。旧入口没有官方MumPy模型或三帧目标接口，因此不能直接替代这些薄适配。

## 决定

固定官方OUC-VAS/Mumpy提交`bc9280a0839f97375b8dda2dcd6d5b5052083f13`；只接受核验后的作者公开通用初始化，不接受旧PVST/AINPAINT或已训练取证模型。
完整三时间视图、可变形跨视图交互、全局时间Transformer、DCT频率辅助与多金字塔定位解码保持官方计算路径。窗口不是XYZ块，不声称全视频C实现。

数据、10轮/有效batch12（脚本6个source样本由collate拼成12片段）、评价/选权重及运行边界见根契约17.14和`anakin_mumpy_config_v1.json`。保留作者SGD参数分组、实际focal/IoU命名和多项式形式；明确迁移使用官方可选SingleAugment，避免默认DoubleAugment中的GT引导裁剪；整帧224及原生计分，不做开发ROI。

官方softIoULoss封装将recall=False误作第三位置参数epsilon=0。为合法real/空mask数值稳定，直接调用作者softIoU helper并显式e=1e-6、recall=False；focal保持发布代码的alpha=[1,1]和gamma2，不改称BCE/Dice。

本轮允许必要公开代码/初始化下载、小型依赖overlay及短真实更新，预检更新丢弃。正式长训练只由用户启动，成功/异常自主保存，不依赖Codex监控。

本ADR仅扩展独立已发表基线的授权，不推翻或删除旧ADR、默认方法、模型与数据角色，不支持三维/时间独立贡献的预设结论。
