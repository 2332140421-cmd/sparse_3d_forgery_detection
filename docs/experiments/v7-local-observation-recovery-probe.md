# V7 04LAX fake 局部观测恢复诊断

本实验是单案例的观测层诊断，不是随机抽样评价、检测器训练或 AUROC
实验。案例固定为 `04LAX` fake 的 `0002_MANIP_25::fake`，使用已有
289 点密度诊断产物。两个用户时刻按真实 PTS 映射到源帧 487 和 488：
`16.249582916249583` s 与 `16.28294961628295` s，二者相邻，间隔约
0.0333667 s。查询初始化帧是窗口首帧 476（15.88254921588255 s），
所以第一张核验时刻不是初始化帧。

## 条件

- **O**：原有 `density289` 数组、H 支撑和原配置，完全复用，不重跑检测器。
- **R**：在源帧 488 重新用同一 BootsTAPIR 和 17×17（289）网格查询；新
  ID 不与 O 连接。当前实现只保存 2D UV/visibility，未把它伪装成 3D
  几何结果。
- **T**：唯一替代候选是官方 CoTracker3 online；若官方运行时不可用则保留
  阻塞证据，不替换为其他跟踪器。

## 现有数组证据

O 的源帧 487 为 `289/289/289`（可用 UV/visibility/geometry），源帧
488 为 `86/86/86`，因此“消失”首先出现在已有 visibility/UV 有效性层，
并不是页面单独过滤造成的。canonical NPZ 对不可见点已将 UV 写为 NaN，
因此原始 tracker 的未掩码预测 UV 不可恢复；不能据此断言 tracker 没有输出
预测坐标。逐层 CSV 位于数据目录的 `frame_layer_counts_*.csv`。

## ROI 与边界

对话截图是浏览器截图，当前没有可复核的浏览器到源视频像素变换文件，故
ROI 标记为 `PENDING_SOURCE_PIXEL_MAPPING`，没有编造 ROI 内外统计。页面只
显示真实源帧、PTS、保存的 UV/visibility/geometry 和已保存 H 关系；缺失不
补 XYZ，不把单案例现象称为伪造检测或定位成功。

## 产物与访问

运行脚本：

```bash
PYTHONPATH=src .venv/bin/python -m research_tools.v7.local_observation_recovery_probe.probe
python3 -m http.server 8765 --bind 127.0.0.1 --directory /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_observation_recovery_probe_v1
```

然后访问 `http://127.0.0.1:8765/review/`。真实前端、模型权重、视频和
NPZ 均留在数据盘，不进入 Git。
