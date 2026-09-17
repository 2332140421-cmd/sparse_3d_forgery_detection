# V7 04LAX R 查询密度匹配对照

本实验是单案例、只读模型诊断：在同一 `04LAX` fake 窗口、同一 query
起点和同一帧范围内，将已核验的 17×17 R 查询与实际轴坐标逐邻接插入
得到的 33×33 查询比较。它不训练模型、不改变正式 `src` 检测链，也不
产生 AUROC 或空间真值评价。

入口：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
PYTHONPATH=src:. /root/autodl-tmp/envs/v7-explicit-geometry/bin/python \
  -m research_tools.v7.query_density_probe.pilot
```

结果写入数据盘：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_query_density_probe_v1/`。
R17 复用旧 R 的 2D UV/visibility；R33 只运行一次 1089-query BootsTAPIR。两者
随后共享一次 Depth Pro、固定首帧内参和 Open3D pose，再各自按当前分组、五时刻
共同成员、历史尺度和 Q 规则构造结构。旧 R_geometry 只作历史参照，公平读出使用
同一份新共享几何。

`query_manifest.json` 明确记录原查询和新增查询；`original_query_consistency.csv`
只描述联合查询中原 289 点的 UV/visibility 变化，不把非零位移解释为跟踪正确性。
`roi_support_comparison.csv` 仅使用已有逐帧 `CONFIRMED` 粗矩形，SKIP/PENDING 不计入。
模型输出沿用冻结 observation-support 模型和标准化，属于单案例开发读出，不是定位
概率或 sealed-test 结果。

若前端、几何和支撑已完成而仅需恢复模型读出，可使用不重跑前端的入口：

```bash
PYTHONPATH=src:. /root/autodl-tmp/envs/v7-explicit-geometry/bin/python \
  -m research_tools.v7.query_density_probe.pilot --resume-readout
```
