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
  ID 不与 O 连接。已用同一 Depth Pro、首帧内参策略和 Open3D RGB-D
  odometry 形成独立三维状态；由于新 ID 没有旧 H/B 成员映射，不宣称恢复
  O 的跨失踪物理对应。
- **T**：唯一替代候选是官方 CoTracker3 online。官方源码/权重可用时按官方
  chunk API 形成独立 2D UV/visibility；本案例不把 T 的 2D 输出扩展成三维
  或 H/B 结果。若官方资源不可用，运行状态保留明确阻塞证据。

## 现有数组证据

O 的源帧 487 为 `289/289/289`（可用 UV/visibility/geometry），源帧
488 为 `86/86/86`，因此“消失”首先出现在已有 visibility/UV 有效性层，
并不是页面单独过滤造成的。canonical NPZ 对不可见点已将 UV 写为 NaN，
因此原始 tracker 的未掩码预测 UV 不可恢复；不能据此断言 tracker 没有输出
预测坐标。逐层 CSV 位于数据目录的 `frame_layer_counts_*.csv`。

## 历史与评估阶段

源帧 488 的 PTS 相对初始化帧 476 为约 0.4004004 s，属于首个 0.5 s
历史窗口；既有 detail 的模型目标从更晚的帧 491 等开始。因此这两个用户时刻
不能直接当成已经被模型评分的目标时刻。visibility、geometry 和 relation
support 是同一观测链的不同筛选层，不能描述成三个独立失败证据。

## 本轮 T 轨迹核验与页面修复

T 并非静态网格或面板占位：`conditions/T_cotracker3_online.npz` 的状态为
`COMPLETE_2D_TRACKS_NO_3D`，由官方 CoTracker3 online（源码 commit
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`、官方
`scaled_online.pth`）生成。源帧到数组索引核对为
`487→11`、`488→12`、`498→22`；页面现在从
`review/trajectory_data.json` 按源帧查找该索引，不使用初始 query 广播，且保留
visibility 与预测 UV 的区别。三帧的 UV 数组均不完全相同：

| 源帧对 | 全体点位移 median / p95 / max (px) | 粗人物框内位移 median / p95 / max (px) |
|---|---:|---:|
| 487→488 | 0.780046 / 3.624831 / 24.441378 | 1.734038 / 4.391930 / 24.441378 |
| 488→498 | 0.503594 / 2.541998 / 3.649551 | 0.793961 / 2.835928 / 3.649551 |
| 487→498 | 1.084733 / 5.726220 / 25.163301 | 2.316031 / 6.133122 / 25.163301 |

这些非零位移只说明保存数组发生变化，不证明跟踪正确。T 仍是二维诊断，
`XYZ`、H/B 分组和 triplet 支撑均为未计算。页面状态已分开显示二维跟踪、XYZ、
H/B grouping 和 triplet support；可选点 ID 与最近五帧尾线均按条件本地数组工作，
不把 O/R/T 的同号 ID 当成同一物理点。R 的 geometry 计数来自独立
`R_geometry.npz` 的 UV、visibility、深度采样、位姿变换和有限 XYZ 联合有效性；
R 尚未形成新轨迹的 H/B/triplet 支撑。

## ROI 与边界

对话截图只用于给出源帧 488 的粗略建议框，状态是
`PENDING_USER_CONFIRMATION`，不是空间真值。页面显示未标注的源帧原图、原
像素坐标读数、可编辑框和一次性下载确认 JSON。用户确认前不输出 ROI 内外
统计；固定 query ID 的后续存续和当前落框数量保持分开。页面仍显示真实源帧、
PTS、保存的 UV/visibility/geometry 和已保存 H 关系；缺失不补 XYZ，不把单
案例现象称为伪造检测或定位成功。

## 产物与访问

运行脚本：

```bash
PYTHONPATH=src .venv/bin/python -m research_tools.v7.local_observation_recovery_probe.probe
python3 -m http.server 8765 --bind 127.0.0.1 --directory /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_observation_recovery_probe_v1
```

然后访问 `http://127.0.0.1:8765/review/`。真实前端、模型权重、视频和
NPZ 均留在数据盘，不进入 Git。
