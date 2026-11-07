# 深度切片关系学习首轮实验



状态：FAILED。独立方法DEPTH_SLICE_RELATION，单seed20260909；不是旧PVST体素实验。

数据：{"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}

工作流：冻结官方ImageNet ResNet18原生stride32局部视觉→回映每个原生像素；XYZ/UV逐点128→256编码→融合128→同深度片mean/max→512维C→两组时间/深度显式8head成对关系→像素局部旁路+片上下文定位/独立帧头。无效几何保留视觉路径。d=Z/1.9888148307800293，h=1/8，d0=0；单位非米，无新增配准。

训练片段：每视频预先固定首、中、尾最多3段24连续原生帧。开发完整原生帧，每帧一次；24帧因果窗口，最后尾段补左历史，常规边界重置历史。frame BCE+pixel BCE+softDice等权，逐帧/片段均值；空GT Dice使用epsilon1仍参与。

全部指标固定logit≥0；视频log-mean-exp温度1；空间Dice/IoU：正GT帧均值后fake视频宏平均；pixel AP：正GT帧AP均值后fake视频宏平均，并非AP@IoU。

监测：固定前两个训练pair的完整视频，称训练监测子集，不代表全训练集。best仅按开发Dice，min10/max30/patience8。

完成摘要：{"status": "FAILED", "error": "SMOKE_GRADIENT_PATH:{'geo': {'present': True, 'finite': True, 'norm': 0.1475275754928589}, 'fusion': {'present': True, 'finite': True, 'norm': 0.2686961889266968}, 'token': {'present': True, 'finite': True, 'norm': 0.08095058798789978}, 'time': {'present': True, 'finite': True, 'norm': 0.13109533488750458}, 'depth': {'present': True, 'finite': True, 'norm': 1.1420923471450806}, 'context': {'present': True, 'finite': True, 'norm': 1.1309572458267212}, 'pixel': {'present': True, 'finite': True, 'norm': 1.4781862497329712}, 'frame_head': {'present': True, 'finite': True, 'norm': 0.0}, 'frame_visual': {'present': True, 'finite': True, 'norm': 0.0}}", "execution_head": "963b412cb5f369ee4d2c109e3d5d81cdf743b84b", "wall_s": 17.328498259186745, "peak_allocated": 1439262208, "peak_reserved": 1824522240, "data": {"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}, "storage_bytes": 1723172319}



## 结论边界

相机运动本轮允许，不证明深度可比性已解决；此前CPU审计显示漂移/跨片问题。没有匹配二维对照，不主张三维优于二维。全程编辑样本不能检验复杂时间边界。检测监督只来自逐帧GT是否含编辑区，ANAKIN裁剪边界未作为伪造区间。

完整资产与固定样例：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_depth_slice_relation_pilot_v1`。

第二次必要检查已完成一次验收更新，第二次反向各分支梯度存在且有限，但全正标签片段上的帧头梯度为零，被逐步非零断言过严拦截。当前检查要求每步梯度存在且有限、两步内每个核心分支至少一次非零，并保存逐步logit与loss；不改变模型、初始化、数据或损失。原退出1保留在execution_history/01_overstrict_per_step_gradient_check，尚无正式epoch成绩。
