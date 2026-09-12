# V7 固定时间网格：标注时段与分数时间响应描述性分析

本报告只分析已有固定时间网格冻结模型产物，不重跑前端、模型前向或训练。分析范围固定为完整 source `0BX9N`、`0FO58`，条件固定为 `H_MEAN_A`、`B_MEAN_A`、`B_MEAN_C`。这是事后描述性分析，不用于调参、阈值选择或新的主指标。

## 1. 输入与口径

实际读取的文件及字段如下：

| 文件 | 用途 | 关键字段 |
| --- | --- | --- |
| `derived/v7_activityforensics_fixed_grid_frozen_probe_v1/scores/window_scores.csv` | 逐窗口计划、类别与三条件 logit | `window_id`、`source_id`、`source_video_id`、`role`、`interval_start_s`、`interval_end_s`、`annotation_category`、`H_MEAN_A`、`B_MEAN_A`、`B_MEAN_C`、对应 `*_status` |
| `derived/v7_activityforensics_fixed_grid_frozen_probe_v1/manifests/grid_windows.json` | 冻结网格和时间身份核对 | `window_id`、`nominal_start_s`、`nominal_end_s`、`frame_indices`、`timestamps_s`、`annotation_category` |
| `derived/v7_activityforensics_fixed_grid_frozen_probe_v1/manifests/videos.json` | 视频路径及原始修改区间 | `source_id`、`video_id`、`role`、`path`、`annotation_intervals_relative_s` |
| `derived/v7_activityforensics_fixed_grid_frozen_probe_v1/evaluation/time_curve_index.csv` | 现有曲线索引 | `source_id`、`path`、`window_count`、`note` |
| `derived/v7_activityforensics_fixed_grid_frozen_probe_v1/evaluation/time_curves/source_0BX9N.svg`、`source_0FO58.svg` | 已生成的 real/fake 时间曲线 | 同一 source 共用纵轴；缺失分数处断线；黄色区间为数据集标注 |

输入快照（SHA-256）：

```text
scores/window_scores.csv  a5c22335a45fd20a260e0a583a846eecb1d9e776b216ef5f75e3891b8cf4fb77
manifests/grid_windows.json  000b4a722f9efd9ffc7b074772d0ec384fdf9250ec4164d6ccac56837307eca7
coverage/window_support.csv  d59bd61f73c4fb6edb1ba9bb43a489ea8b43cfcd8f83115b117c7338963945b6
evaluation/summary.json  9f4989900ec0d32c7de72a0fcf0ffd164669590e812838bd23b79c486b139d78
evaluation/time_curve_index.csv  a25dcf9dd49633a33c897f0a11be157bf5d9f92f5276f22fc9adaa39e2d0f99f
```

`planned` 是该类别在固定网格中的全部行数；`scored` 是对应条件字段中有限且非空的 logit；`missing = planned - scored`。缺失不填 0。`>=0` 是固定阈值 `logit >= 0` 的比例，不是重新选择的阈值，也不是假阳性率。P25/P50/P75 对有效 logit 使用排序后的线性插值。

网格类别沿用已有字段：real 全部为 `REAL_NEGATIVE`；fake 的 `FAKE_MANIPULATION` 表示窗口完整落入标注修改区间，`BOUNDARY_MIXED` 与 `OUTSIDE_ANNOTATED_MANIPULATION` 只作描述。区间外 fake 不作为确定负类，边界窗口不并入主标签结论。

## 2. 两个 source 的视频与标注时间

| source | real 视频 | fake 视频 | fake 标注修改区间（相对视频秒） |
| --- | --- | --- | --- |
| `0BX9N` | `datasets/v7_core_candidates/activityforensics_charades_v1/source/charades/videos/0BX9N.mp4` | `datasets/v7_core_candidates/activityforensics_charades_v1/source/activityforensics/raw/video/02_wan/0BX9N+11.80=21.40=charades@train_delete@0BX9N@552@wan.mp4` | `[11.8, 21.4]` |
| `0FO58` | `datasets/v7_core_candidates/activityforensics_charades_v1/source/charades/videos/0FO58.mp4` | `datasets/v7_core_candidates/activityforensics_charades_v1/source/activityforensics/raw/video/03_fcvg/0FO58+1.30=13.10=charades@train_delete@0FO58@1025@fcvg.mp4` | `[1.3, 13.1]` |

real/fake 只按 source 汇总分布；没有把 real 与 fake 当作逐帧物理配对，也没有假定动作在时间上相同。

## 3. 按 source、条件和时间类别的描述统计

### 0BX9N

| condition | 类别 | planned | scored | missing | median | P25 | P75 | logit≥0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H_MEAN_A | REAL_NEGATIVE | 45 | 45 | 0 | -0.2947 | -0.3689 | -0.1930 | 0.067 |
| H_MEAN_A | FAKE_MANIPULATION | 17 | 17 | 0 | -0.3071 | -0.3521 | -0.2474 | 0.000 |
| H_MEAN_A | OUTSIDE_ANNOTATED_MANIPULATION | 24 | 24 | 0 | -0.3392 | -0.4180 | -0.2198 | 0.167 |
| H_MEAN_A | BOUNDARY_MIXED | 4 | 4 | 0 | -0.3065 | -0.3741 | -0.2417 | 0.000 |
| B_MEAN_A | REAL_NEGATIVE | 45 | 45 | 0 | -0.3432 | -0.4196 | -0.2357 | 0.067 |
| B_MEAN_A | FAKE_MANIPULATION | 17 | 17 | 0 | -0.3613 | -0.4132 | -0.2743 | 0.000 |
| B_MEAN_A | OUTSIDE_ANNOTATED_MANIPULATION | 24 | 24 | 0 | -0.3797 | -0.4735 | -0.2316 | 0.167 |
| B_MEAN_A | BOUNDARY_MIXED | 4 | 4 | 0 | -0.3488 | -0.4199 | -0.2743 | 0.000 |
| B_MEAN_C | REAL_NEGATIVE | 45 | 45 | 0 | -0.0819 | -0.1705 | 0.0337 | 0.289 |
| B_MEAN_C | FAKE_MANIPULATION | 17 | 17 | 0 | -0.0930 | -0.1902 | -0.0176 | 0.235 |
| B_MEAN_C | OUTSIDE_ANNOTATED_MANIPULATION | 24 | 24 | 0 | -0.0884 | -0.2384 | 0.0297 | 0.333 |
| B_MEAN_C | BOUNDARY_MIXED | 4 | 4 | 0 | -0.1259 | -0.1772 | -0.0331 | 0.250 |

### 0FO58

| condition | 类别 | planned | scored | missing | median | P25 | P75 | logit≥0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H_MEAN_A | REAL_NEGATIVE | 65 | 63 | 2 | -0.0643 | -0.4366 | 0.2679 | 0.476 |
| H_MEAN_A | FAKE_MANIPULATION | 22 | 21 | 1 | 0.5120 | 0.3285 | 0.6836 | 0.905 |
| H_MEAN_A | OUTSIDE_ANNOTATED_MANIPULATION | 39 | 36 | 3 | -0.2741 | -0.4264 | 0.0421 | 0.306 |
| H_MEAN_A | BOUNDARY_MIXED | 4 | 4 | 0 | 0.1812 | -0.0169 | 0.4871 | 0.750 |
| B_MEAN_A | REAL_NEGATIVE | 65 | 63 | 2 | -0.0845 | -0.4225 | 0.2512 | 0.429 |
| B_MEAN_A | FAKE_MANIPULATION | 22 | 21 | 1 | 0.4749 | 0.2954 | 0.5877 | 0.952 |
| B_MEAN_A | OUTSIDE_ANNOTATED_MANIPULATION | 39 | 36 | 3 | -0.2474 | -0.4286 | 0.1083 | 0.333 |
| B_MEAN_A | BOUNDARY_MIXED | 4 | 4 | 0 | 0.1801 | -0.0386 | 0.4939 | 0.750 |
| B_MEAN_C | REAL_NEGATIVE | 65 | 63 | 2 | -0.1388 | -0.3813 | 0.0779 | 0.349 |
| B_MEAN_C | FAKE_MANIPULATION | 22 | 21 | 1 | 0.4520 | 0.1905 | 0.4859 | 0.905 |
| B_MEAN_C | OUTSIDE_ANNOTATED_MANIPULATION | 39 | 36 | 3 | -0.0933 | -0.2179 | 0.0094 | 0.306 |
| B_MEAN_C | BOUNDARY_MIXED | 4 | 4 | 0 | 0.1387 | 0.0079 | 0.2667 | 0.750 |

### real 与 fake 整体分布的对照

下面的 fake 行包含该 source fake 视频的三类窗口（MANIP、BOUNDARY、OUTSIDE），仅用于判断整体 source/video 差异，不把它当作时间定位指标。

| source | condition | real n / median | fake n / median | fake−real median | real logit≥0 | fake logit≥0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0BX9N | H_MEAN_A | 45 / -0.2947 | 45 / -0.3292 | -0.0344 | 0.067 | 0.089 |
| 0BX9N | B_MEAN_A | 45 / -0.3432 | 45 / -0.3686 | -0.0253 | 0.067 | 0.089 |
| 0BX9N | B_MEAN_C | 45 / -0.0819 | 45 / -0.0948 | -0.0129 | 0.289 | 0.289 |
| 0FO58 | H_MEAN_A | 63 / -0.0643 | 61 / 0.0367 | 0.1010 | 0.476 | 0.541 |
| 0FO58 | B_MEAN_A | 63 / -0.0845 | 61 / 0.0696 | 0.1541 | 0.429 | 0.574 |
| 0FO58 | B_MEAN_C | 63 / -0.1388 | 61 / 0.0061 | 0.1449 | 0.349 | 0.541 |

0BX9N 的 fake 整体中位数略低于 real；0FO58 则略高，且 real 分布也有高分。因此把两个 source 混合后看到的 fake/real 差异不能单独解释为修改时段响应。

## 4. 修改区间内外的时间响应

每个 fake 视频只有一个已核对的修改区间。`inside` 只取 `FAKE_MANIPULATION`，`outside` 只取 `OUTSIDE_ANNOTATED_MANIPULATION`；边界窗口单列，不参与差值。这里没有对重叠窗口做显著性检验。

| source | interval (s) | condition | inside n / median | outside n / median | inside−outside | boundary n / median |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0BX9N | [11.8, 21.4] | H_MEAN_A | 17 / -0.3071 | 24 / -0.3392 | 0.0321 | 4 / -0.3065 |
| 0BX9N | [11.8, 21.4] | B_MEAN_A | 17 / -0.3613 | 24 / -0.3797 | 0.0185 | 4 / -0.3488 |
| 0BX9N | [11.8, 21.4] | B_MEAN_C | 17 / -0.0930 | 24 / -0.0884 | -0.0046 | 4 / -0.1259 |
| 0FO58 | [1.3, 13.1] | H_MEAN_A | 21 / 0.5120 | 36 / -0.2741 | 0.7861 | 4 / 0.1812 |
| 0FO58 | [1.3, 13.1] | B_MEAN_A | 21 / 0.4749 | 36 / -0.2474 | 0.7222 | 4 / 0.1801 |
| 0FO58 | [1.3, 13.1] | B_MEAN_C | 21 / 0.4520 | 36 / -0.0933 | 0.5453 | 4 / 0.1387 |

描述性结论是：0FO58 的修改区间内三种条件均明显高于该 fake 视频的区间外窗口；0BX9N 的差异接近 0，且二阶条件略低。该模式与 source 强相关，不能据此宣称普遍的时间定位能力或机制。

现有 real 分布（有效窗口）为：

| source | condition | n | median | P25 | P75 | logit≥0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0BX9N | H_MEAN_A | 45 | -0.2947 | -0.3689 | -0.1930 | 0.067 |
| 0BX9N | B_MEAN_A | 45 | -0.3432 | -0.4196 | -0.2357 | 0.067 |
| 0BX9N | B_MEAN_C | 45 | -0.0819 | -0.1705 | 0.0337 | 0.289 |
| 0FO58 | H_MEAN_A | 63 | -0.0643 | -0.4366 | 0.2679 | 0.476 |
| 0FO58 | B_MEAN_A | 63 | -0.0845 | -0.4225 | 0.2512 | 0.429 |
| 0FO58 | B_MEAN_C | 63 | -0.1388 | -0.3813 | 0.0779 | 0.349 |

## 5. 缺失与可复核窗口

选定 source 共 220 个固定网格窗口（0BX9N 90、0FO58 130）。0BX9N 三条件全部有分数；0FO58 有 124 个窗口有分数、6 个窗口在三条件同时为 `NO_MODEL_SCORE`，其 `B_MEAN_A_status` 为 `NO_VALID_TRIPLET`。缺失保持为空，不能解释为低分或伪造。

缺失窗口为：

| window_id | video | interval (s) | category |
| --- | --- | --- | --- |
| `0FO58::fake::grid0009` | fake | [4.5, 5.5] | FAKE_MANIPULATION |
| `0FO58::fake::grid0031` | fake | [15.5, 16.5] | OUTSIDE_ANNOTATED_MANIPULATION |
| `0FO58::fake::grid0041` | fake | [20.5, 21.5] | OUTSIDE_ANNOTATED_MANIPULATION |
| `0FO58::fake::grid0064` | fake | [31.68, 32.68] | OUTSIDE_ANNOTATED_MANIPULATION |
| `0FO58::real::grid0031` | real | [15.5, 16.5] | REAL_NEGATIVE |
| `0FO58::real::grid0064` | real | [31.72, 32.72] | REAL_NEGATIVE |

缺失是结构支撑不足，不代表视频时间段没有伪造或没有运动。已有的 `evaluation/time_curves/*.svg` 按真实视频时间排列，缺失处分段断线；黄色背景是数据集标注区间，固定阈值为 `logit = 0`，不是预测出的区间。索引文件还保留了 source、role 和窗口数，未对曲线做平滑或跨缺失插值。

## 6. 按分数选择的人工核对案例

以下每个 source 最多四个案例，固定以 `B_MEAN_A` 选择：修改区间内最高/最低、real 最高，以及可用时的一个无支撑窗口；同时列出另外两个条件。它们是诊断选例，不是无偏抽样、成功率或定位证据，也没有人工确认“可见失真部位”。

| source | case | window_id | video | interval (s) | category | H_MEAN_A | B_MEAN_A | B_MEAN_C |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 0BX9N | fake MANIP 最高 | `0BX9N::fake::grid0034` | fake | [17.0, 18.0] | FAKE_MANIPULATION | -0.0631 | -0.0563 | 0.1148 |
| 0BX9N | fake MANIP 最低 | `0BX9N::fake::grid0035` | fake | [17.5, 18.5] | FAKE_MANIPULATION | -0.4374 | -0.4944 | -0.2007 |
| 0BX9N | real 最高 | `0BX9N::real::grid0009` | real | [4.5, 5.5] | REAL_NEGATIVE | 0.7638 | 0.8610 | 1.0740 |
| 0FO58 | fake MANIP 最高 | `0FO58::fake::grid0014` | fake | [7.0, 8.0] | FAKE_MANIPULATION | 1.3826 | 1.6084 | 1.6835 |
| 0FO58 | fake MANIP 最低 | `0FO58::fake::grid0011` | fake | [5.5, 6.5] | FAKE_MANIPULATION | -0.0291 | -0.0558 | -0.5576 |
| 0FO58 | real 最高 | `0FO58::real::grid0033` | real | [16.5, 17.5] | REAL_NEGATIVE | 3.9189 | 4.2941 | -5.9485 |
| 0FO58 | 无支撑示例 | `0FO58::fake::grid0009` | fake | [4.5, 5.5] | FAKE_MANIPULATION | NA | NA | NA |

## 7. 可视化索引与访问

已生成轻量索引（不复制视频、不生成 ZIP）：

```text
/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_fixed_grid_frozen_probe_v1/evaluation/temporal_response_index.html
```

建议从数据根启动本地静态服务器，以便同时访问既有曲线和四个已存在的视频：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory /root/autodl-tmp/data/sparse_3d_forgery_detection
```

浏览器打开：

```text
http://127.0.0.1:8765/derived/v7_activityforensics_fixed_grid_frozen_probe_v1/evaluation/temporal_response_index.html
```

索引直接链接 `source_0BX9N.svg`、`source_0FO58.svg` 以及 `manifests/videos.json` 中实际存在的 real/fake MP4。若服务器从其他目录启动，视频链接可能不可访问；报告中的绝对路径仍是实际文件位置。

## 8. 结论与限制

1. **是否有修改区间内分数偏高的迹象？** 有，但只在 `0FO58` 明显：三条件 inside−outside 中位数为 `0.5453–0.7861`；`0BX9N` 为 `-0.0046–0.0321`，没有一致升高。
2. **是否主要反映 real/fake 整体差异？** 不能排除。两个 source 的 fake−real 整体中位数方向相反，且 `0FO58` real 也出现很高分，说明 source/video 整体差异和时间段差异混在一起。
3. **无支撑的影响？** 0FO58 有 6 个前端完成但无有效 triplet 的窗口，其中 1 个位于修改区间内；缺失改变可比较样本，不能当作低分。
4. **能否据此定位失真过程？** 不能。这里没有空间真值，没有人工逐窗口可见性核对，且只有两个完整 source；窗口分数只表示冻结模型响应，不是校准概率、像素定位或物理机制。
5. **为什么不作显著性检验？** 固定网格窗口重叠，不能把它们当独立样本；两个 source 也不足以给出稳定泛化保证。

因此，本轮支持的最强表述是：在已有两个完整 source 的事后描述中，`0FO58` 的标注修改时段与较高冻结 logit 同时出现，而 `0BX9N` 不显示同样模式；结果不足以区分时间响应与 source/video 整体差异，也不足以支持普遍时间定位或空间定位结论。
