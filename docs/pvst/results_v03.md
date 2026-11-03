# PVST v0.3 单轮贯通报告

计算状态：RESOURCE_BLOCKED；模型完成0/2；每条件最多一轮，不是最终收敛结论。
v0.3改变XYZ结构分支统一体素尺度及片段监督/更新方式，不与v0.2全视频训练宣称等价。扩大体素降低块级结构空间分辨率，点旁路不能保证精度不下降。
逐点9维坐标→geo128/局部卷积→VFE及fine256/coarse512；geo点旁路与逐像素视觉128和所属时空块256进入原像素解码器。全部像素有预测，全部有效点归块，无随机删点/块或邻接截断。几何为dense256场回映原生，并非新增独立高分辨率观测。
冻结体素尺度：{'UV_PVST': 0.03125, 'XYZ_PVST': None}；候选预算统计：[{"h": 0.03125, "fine_max": 6533459, "coarse_max": 4907360, "within_budget": 0}, {"h": 0.0625, "fine_max": 4907360, "coarse_max": 2970629, "within_budget": 0}, {"h": 0.125, "fine_max": 2970629, "coarse_max": 1488042, "within_budget": 0}]
复用48视频22,074帧（训练32、开发16），新增前端调用0；训练片段0，去重目标帧0；每视频至多4个互不重叠目标段，历史不监督。
训练每更新AF/AN顺序反向后对参与片段等权平均；片段内frame BCE及AN逐帧pixel BCE/Dice分别取目标帧均值，无旧全视频分母/类别权重。

|条件|checkpoint|集合|AF video AUROC/AP/F1/ACC|AF frame AUROC/AP/F1|AN Dice/IoU/pixel AP/real FPR|
|---|---|---|---|---|---|

资源记录（非profiler；两更新/两微批；权重丢弃）：

尚无一轮训练或评价指标；资源/尺度未通过则停止，不用探针loss代替正式成绩。

资源简表（前向/反向均合计4微批；吞吐按32次目标帧处理计，不是独立帧数）：
|条件|视频/目标|卸载|结果|fine/coarse|allocated/reserved GiB|前向/反向秒|目标帧/秒|
|---|---|---|---|---|---|---|---|

坐标计数秒：489.2232668567449
资源case总秒：0
4090建议：当前未通过全部验收，不启动或追加训练；具体阻塞见下。
本轮输出目录：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_point_voxel_v03
完整视频评价使用固定logit>=0，正GT帧先视频内均值再视频均值，空GT/real误报分列；AF无空间GT不报像素定位。固定图25/50/75% PTS、统一概率0–1；不按模型峰值挑帧。

## 训练坐标预算审计
|XYZ h|fine median/p95/max|coarse median/p95/max|
|---|---|---|
|0.03125|97,744.0 / 3,257,015.5 / 6,533,459.0|27,365.5 / 1,955,117.7 / 4,907,360.0|
|0.0625|27,365.5 / 1,955,117.7 / 4,907,360.0|7,176.5 / 1,078,222.6 / 2,970,629.0|
|0.125|7,176.5 / 1,078,222.6 / 2,970,629.0|1,872.5 / 552,412.6 / 1,488,042.0|

最粗1/8最大窗口：{"video_id": "782426ea0330c7fea684953c", "target_range": [120, 128], "context_range": [104, 120], "native_hw": [720, 1280], "fine": 2970629, "coarse": 1488042}
UV原1/32精确原生UV格点占据计数（坐标解析计数，不是GPU实测）：{"train": {"h": 0.03125, "windows": 1932, "fine_min_median_p95_max": [32768.0, 98304.0, 98304.0, 98304.0], "coarse_min_median_p95_max": [8192.0, 24576.0, 24576.0, 24576.0]}, "dev": {"h": 0.03125, "windows": 843, "fine_min_median_p95_max": [32768.0, 98304.0, 98304.0, 98304.0], "coarse_min_median_p95_max": [8192.0, 24576.0, 24576.0, 24576.0]}}

CPU模型身份核对（无前向/训练）：{
  "scope": "CPU UV actual official decoder/model; no forward or training",
  "parameters": 17821672,
  "state_dict_keys": 252,
  "state_dict_keys_sha256": "31beab46b45b34ed53ff89a70293a0e9e93abb86721dc51902e9c4c1808af665",
  "official_bn_unchanged": true
}


## 必要正确性
```
..............FFN_PARITY torch.float32 output/input/params max_abs [2.980232238769531e-07, 4.0745362639427185e-10, 5.820766091346741e-10, 4.656612873077393e-10, 1.1641532182693481e-09, 9.313225746154785e-10, 8.731149137020111e-11, 8.731149137020111e-11, 2.3283064365386963e-10, 4.656612873077393e-10, 9.313225746154785e-10, 9.313225746154785e-10, 5.529727786779404e-10, 4.656612873077393e-10, 1.0477378964424133e-09, 4.656612873077393e-10, 3.259629011154175e-09, 9.313225746154785e-10]
.FFN_PARITY torch.bfloat16 output/input/params max_abs [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3283064365386963e-10, 0.0, 1.52587890625e-05, 7.62939453125e-06, 3.0517578125e-05, 3.0517578125e-05]
.NESTED_DROPOUT_CHECKPOINT max_abs 2.7939677238464355e-09
....OUTPUT_PROJECTION_PARITY torch.float32 output/input/params max_abs [5.960464477539062e-07, 3.4924596548080444e-10, 3.4924596548080444e-10, 3.637978807091713e-10, 1.3969838619232178e-09, 5.820766091346741e-10, 1.1641532182693481e-10, 6.548361852765083e-11, 3.4924596548080444e-10, 3.4924596548080444e-10, 5.238689482212067e-10, 6.984919309616089e-10, 4.3655745685100555e-10, 2.3283064365386963e-10, 6.984919309616089e-10, 2.3283064365386963e-10, 1.3969838619232178e-09, 4.656612873077393e-10]
.OUTPUT_PROJECTION_PARITY torch.bfloat16 output/input/params max_abs [0.0, 3.814697265625e-06, 1.9615981727838516e-06, 1.430511474609375e-06, 1.52587890625e-05, 7.62939453125e-06, 1.9073486328125e-06, 2.6971101760864258e-06, 7.62939453125e-06, 1.4230608940124512e-05, 1.52587890625e-05, 1.52587890625e-05, 1.1641532182693481e-10, 0.0, 7.62939453125e-06, 3.814697265625e-06, 1.52587890625e-05, 1.52587890625e-05]
.ALL_NODE_CHECKPOINT_PARITY max_abs 2.7939677238464355e-09
...COMPACT_OLD_PARITY 1 torch.float32 output/input/parameters [4.76837158203125e-07, 1.0477378964424133e-09, 6.752088665962219e-09]
.COMPACT_OLD_PARITY 2 torch.float32 output/input/parameters [4.76837158203125e-07, 1.0477378964424133e-09, 6.984919309616089e-09]
.COMPACT_OLD_PARITY 4 torch.float32 output/input/parameters [5.364418029785156e-07, 1.0477378964424133e-09, 6.984919309616089e-09]
.COMPACT_OLD_PARITY 1 torch.bfloat16 output/input/parameters [0.0, 1.52587890625e-05, 6.103515625e-05]
.COMPACT_OLD_PARITY 2 torch.bfloat16 output/input/parameters [0.0, 1.52587890625e-05, 6.103515625e-05]
.COMPACT_OLD_PARITY 4 torch.bfloat16 output/input/parameters [0.0, 0.0, 6.103515625e-05]
..V03_QKV_PARITY torch.float32 [1.5497207641601562e-06, 1.430511474609375e-06, 3.814697265625e-06]
.V03_QKV_PARITY torch.bfloat16 [0.0, 0.0, 0.125]
.......
40 passed in 6.82s

```

三个授权候选均超训练坐标预算；未冻结XYZ尺度，未读取开发坐标选尺度，未运行GPU资源探针/一轮训练/epoch0或epoch1评价。不能提供未测显存、吞吐、分数或30轮成本。未扩展到更粗尺度、删点或改变其他规格。旧4090失败不是本轮重新发生的OOM。

错误：
```
Traceback (most recent call last):
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/pvst/v03.py", line 242, in main
    if counts['selected_h'] is None:status='RESOURCE_BLOCKED';raise RuntimeError('NO_AUTHORIZED_SCALE_WITHIN_BUDGET')
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: NO_AUTHORIZED_SCALE_WITHIN_BUDGET

```
