# V7 Vript current-design feasibility fast path

Status: `SUPPORTED` as a private engineering gate; not final Core-domain evidence.

## 1. Research Question

在不改变 V7 formal frontend、persistent particles、dynamic-component 或 `S_t/ΔS_t/Δ²S_t` 实现的前提下，检查冻结的 Vript train-real 小样本是否能形成足够连续的显式 XYZ、持续组件和结构状态。只运行到 `S_t`、`ΔS_t`、`Δ²S_t`，不训练 normality。

## 2. Population

- 32 frozen metadata-prefiltered Vript train-real identities，未重新抽样。
- Fast path 按 frozen order 取得前 8 个 `MEDIA_VALID`。
- `UNREVIEWED_METADATA_CORE_CANDIDATES`；contact sheet 已生成，但人工 RGB review 尚未完成。
- fake count = 0。

### Probe source IDs

- `Vript:00015:-C_-HNTztXI-Scene-005`
- `Vript:00019:-Z3priQFMeE-Scene-021`
- `Vript:00024:-W954LYAU1k-Scene-006`
- `Vript:00033:-YFKqKdD3_A-Scene-006`
- `Vript:00039:-alqtU2K164-Scene-040`
- `Vript:00045:-MP2E1IiHXg-Scene-030`
- `Vript:00056:-F6BoGRCF8g-Scene-006`
- `Vript:00067:-4JeJ5Cm5Mw-Scene-050`

## 3. Data Acquisition

- Official source: `Mutonix/Vript`, revision `acc278efb0ee249d646eef8a6b023595ca7efb93` (repository commit `acc278efb0ee249d646eef8a6b023595ca7efb93`).
- Target→shard/member 使用既有 GenVidBench frozen identity map 与官方 Vript ZIP central-directory exact member；无模糊替换。
- 访问 8 个 distinct official small ZIP shards；官方完整 shard metadata 总大小 `7,143,412,526` bytes（约 6.65 GiB），低于 10 GiB budget。
- 实际只按 HTTP range 读取 ZIP central directory 与精确目标 member；提取视频 payload 合计 `14,383,004` bytes。未永久解压整个 shard，未下载 fake/HD-VG。
- 输出：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_vript_core_feasibility_v1/`。

## 4. Protocol

- 每段视频使用真实 frame PTS；anchor 为视频时间线 50%。
- 物理时间窗：0.5 s、1.0 s、2.0 s；不排序、不 padding、不重复帧。
- 8 videos × 3 timescales = 24 windows，全部 `AVAILABLE` 并完成 frontend。
- Explicit baseline unchanged: online BootsTAPIR, Apple Depth Pro, causal first-frame focal, centered principal point, zero distortion, adjacent Open3D RGB-D odometry, world fixed to camera 0。
- Particles unchanged: 64 regular-grid tracks。ComponentConfig unchanged: `max_initial_distance=1.0 m`, `max_relative_change=0.05 m`, `minimum_overlap=8`, `minimum_size=3`。

## 5. Explicit Geometry Results

| source_id | frames | duration (s) | resolution | effective FPS | contact sheet |
|---|---:|---:|---|---:|---|
| `Vript:00015:-C_-HNTztXI-Scene-005` | 269 | 8.933 | 1280×720 | 30.000 | `Vript_00015_-C_-HNTztXI-Scene-005.png` |
| `Vript:00019:-Z3priQFMeE-Scene-021` | 443 | 14.748 | 1280×720 | 29.970 | `Vript_00019_-Z3priQFMeE-Scene-021.png` |
| `Vript:00024:-W954LYAU1k-Scene-006` | 121 | 4.004 | 1280×720 | 29.970 | `Vript_00024_-W954LYAU1k-Scene-006.png` |
| `Vript:00033:-YFKqKdD3_A-Scene-006` | 425 | 14.147 | 1280×640 | 29.970 | `Vript_00033_-YFKqKdD3_A-Scene-006.png` |
| `Vript:00039:-alqtU2K164-Scene-040` | 124 | 4.920 | 1280×628 | 25.000 | `Vript_00039_-alqtU2K164-Scene-040.png` |
| `Vript:00045:-MP2E1IiHXg-Scene-030` | 445 | 14.815 | 1280×720 | 29.970 | `Vript_00045_-MP2E1IiHXg-Scene-030.png` |
| `Vript:00056:-F6BoGRCF8g-Scene-006` | 329 | 10.944 | 1280×720 | 29.970 | `Vript_00056_-F6BoGRCF8g-Scene-006.png` |
| `Vript:00067:-4JeJ5Cm5Mw-Scene-050` | 956 | 38.200 | 1280×628 | 25.000 | `Vript_00067_-4JeJ5Cm5Mw-Scene-050.png` |

- All 8 media are nonzero, openable, RGB-decodable; full decoded PTS strictly increasing.
- All 24 probe windows completed. Pose-chain success rate is 1.0 at every timescale.

## 6. Dynamic Component Results

| timescale | windows | geometry median (IQR; p10–p90) | pose chain | tracking persistence median (IQR; p10–p90) | windows with component | no-component | component fraction median | component size median |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.5 s | 8 | 0.992 (0.164; 0.728–1.000) | 1.000 | 0.984 (0.289; 0.416–1.000) | 7/8 | 0.125 | 0.867 | 47.0 |
| 1.0 s | 8 | 0.980 (0.207; 0.626–0.999) | 1.000 | 0.945 (0.344; 0.252–0.989) | 8/8 | 0.000 | 0.984 | 43.0 |
| 2.0 s | 8 | 0.942 (0.223; 0.556–0.998) | 1.000 | 0.789 (0.551; 0.164–0.984) | 8/8 | 0.000 | 0.914 | 18.5 |

Component discovery remained label-blind and used only the current fixed configuration. No semantic or authenticity input was added.

## 7. Structural State Results

| timescale | S_t valid-frame median (IQR; p10–p90) | valid ΔS | valid Δ²S | Δ²S magnitude median (IQR; p10–p90) |
|---:|---:|---:|---:|---|
| 0.5 s | 1.000 (0.000; 0.700–1.000) | 142 | 131 | 39.850 (112.465; 5.329–236.316) |
| 1.0 s | 1.000 (0.000; 0.862–1.000) | 364 | 350 | 44.835 (85.802; 7.520–202.979) |
| 2.0 s | 1.000 (0.011; 0.824–1.000) | 853 | 837 | 43.196 (92.960; 4.705–200.486) |

Δ²S magnitude 只作为分布报告，不用于判断哪个 timescale“更好”。

## 8. Timescale Comparison

- `1.0 s` 是当前三者中最平衡的可观测尺度：8/8 窗口有组件、median geometry coverage `0.980`、median tracking persistence `0.945`、S_t valid median `1.000`。
- `0.5 s` 存在 1/8 no-component 窗口，短窗口对组件形成略显不足；其 median coverage 仍为 `0.992`。
- `2.0 s` 组件持续性最好地保持为 8/8，但 tracking persistence 降至 median `0.789`、geometry median `0.942`，显示长窗口有累积前端/对应退化迹象。
- 这是当前小样本的工程可行性比较，不是语义或真假结论。

## 9. Measurement-Outlier Diagnostic

- 仅作诊断的 extreme XYZ step threshold：`5.0 m`；未进入 mask、component 或 score。
- per-window extreme-step rate median `0.000000`，最大 `0.008371`。
- extreme rate > 0.5 的 component 窗口：`0`；no-component 窗口总数：`1`。
- 因此没有证据表明单次巨大 XYZ step 主导本轮 component/S_t 失败；该诊断不等同于 absolute XYZ accuracy。

## 10. Engineering Feasibility Conclusion

`SUPPORTED`

至少一个 timescale（本轮为 1.0 s）同时满足 private gate：≥6/8 videos 有 component、component 窗口 median S_t valid-frame fraction ≥0.75、≥6/8 videos 具有至少 3 个 valid Δ²S observations、median geometry coverage ≥0.80，且失败没有被单次巨大 XYZ outlier 明显支配。

该结论只支持继续当前 V7 private engineering path，不是论文 Core Domain 证据，也不是 fake detection 结果。

## 11. Limitations

- N=8，且只来自 frozen metadata prefilter；人工 RGB review 仍待完成。
- fake = 0；没有 normality training、AUROC、fake detector 或真假比较。
- 没有 GT XYZ，深度尺度和相机假设仍继承当前 baseline 限制。
- 组件、`S_t` 表示和正常性模型仍是 V7 当前 baseline，不因本轮结果而冻结为最终方法。

## 12. Immediate Next Step

完成这 8 个候选的快速人工 review；若保持 Core Domain 资格，则扩大 real Core train population，随后进入 real-only normality training。本轮不自动进入训练。
