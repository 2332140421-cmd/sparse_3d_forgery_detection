# 冻结 GRU 读出定位检查

使用 `pvst_rgb_partition_anakin_dev_v1/RGB_DEPTH_PARTITION/last.pt`（第12轮／576步），以原24帧评价路径完成一次前向。分别记录 GRU 的实际输入 `u=g+time_embedding`、输出 `h` 和原分类头分数。权重、输入、帧顺序和精度均沿用原实验。

- 训练：原8对、48段、1152帧。开发：原4对、924帧；逐帧仅计一次。
- 两个512维读出均在训练帧上拟合标准化和L2逻辑回归（C=1），不根据开发集选择配置。
- 报告训练／开发帧及视频AUROC、AP和逐对视频分差。视频以原 log-mean-exp 聚合，帧标签来自官方mask是否非空。

在原4090实验环境，从仓库根目录运行：

```bash
PYTHONPATH=. python -u -m research_tools.rgbd_slice.gru_readout_probe
```

结果保存在原运行同级的 `pvst_gru_readout_probe_v1/`。包括 `report.md`、`summary.json`、`frames.csv` 及按CSV顺序存放的 `features.npz`。重复运行不会覆盖该目录。原模型代码、配置、清单及权重身份不匹配时停止。若原帧头复现偏差足以改变排序，应先检查输入和运行环境。

若开发集 u 读出优于 h，优先查时序更新；若 h 优于原头且接近 u，优先查分类头；若二者均弱，优先查进入 GRU 前的关系汇聚。结合四对逐视频分差判断结果是否由单条视频驱动。
