# 深度切片关系学习首轮实验



状态：FAILED。独立方法DEPTH_SLICE_RELATION，单seed20260909；不是旧PVST体素实验。

数据：{"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}

工作流：冻结官方ImageNet ResNet18原生stride32局部视觉→回映每个原生像素；XYZ/UV逐点128→256编码→融合128→同深度片mean/max→512维C→两组时间/深度显式8head成对关系→像素局部旁路+片上下文定位/独立帧头。无效几何保留视觉路径。d=Z/1.9888148307800293，h=1/8，d0=0；单位非米，无新增配准。

训练片段：每视频预先固定首、中、尾最多3段24连续原生帧。开发完整原生帧，每帧一次；24帧因果窗口，最后尾段补左历史，常规边界重置历史。frame BCE+pixel BCE+softDice等权，逐帧/片段均值；空GT Dice使用epsilon1仍参与。

全部指标固定logit≥0；视频log-mean-exp温度1；空间Dice/IoU：正GT帧均值后fake视频宏平均；pixel AP：正GT帧AP均值后fake视频宏平均，并非AP@IoU。

监测：固定前两个训练pair的完整视频，称训练监测子集，不代表全训练集。best仅按开发Dice，min10/max30/patience8。

完成摘要：{"status": "FAILED", "error": "index_copy_(): self and source expected to have the same dtype, but got (self) Float and (source) BFloat16", "execution_head": "d15956f40b6cf72aa5750ceaa46d69af2e46f416", "wall_s": 35.66993834730238, "peak_allocated": 645201408, "peak_reserved": 939524096, "data": {"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}, "storage_bytes": 1723157427}



## 结论边界

相机运动本轮允许，不证明深度可比性已解决；此前CPU审计显示漂移/跨片问题。没有匹配二维对照，不主张三维优于二维。全程编辑样本不能检验复杂时间边界。检测监督只来自逐帧GT是否含编辑区，ANAKIN裁剪边界未作为伪造区间。

完整资产与固定样例：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_depth_slice_relation_pilot_v1`。

首次必要检查在BF16片上下文index_copy类型不一致处退出1，正式训练尚未开始；已修复显式dtype对齐并通过5项CPU检查（含BF16反向），将复用全部冻结视觉缓存重新执行必要检查。原错误/退出历史保存在运行目录execution_history/00_bf16_context_scatter；本页不是最终成绩。
