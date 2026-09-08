# V7 real-only structural normality — unpaired generated-video response

## 1. Research Question

冻结的 real-only structural-evolution normality 是否对真实生成视频产生异常响应？

## 2. Scientific Boundary

这是 Vript held-out real_val 与 GenVidBench Pair2 SVD/CogVideo 的 unpaired source/domain comparison；不是最终 paired benchmark，也没有 structural-anomaly ground truth。

## 3. Frozen Method

M1=ΔS、M2=Δ²S、M3=[ΔS,Δ²S]；model SHA256=`4e02e45bd56be8a7081c2c2af5046fbe7697434b110d016313b4d502b3d32847`；fitting population=real_train_only；fake fitting count=0；无 refit。

1.0 s true-PTS windows、25/50/75% anchors、现有 ComponentConfig、S_t/ΔS/Δ²S 和前端均保持冻结。

## 4. Population and Representation

real N=8 held-out Vript sources；fake N=10（SVD=5、CogVideo=5），fake underlying Pair2 ordinal N=5。两类均使用 geometry → tracking → component → S validity → frozen score；所有 10 个 fake 均保留在 per-video 表，00042 CogVideo 因三窗口均无 component 而无可评分 M1/M2/M3 值。

real quality summary: `{"component_success_fraction": {"IQR": 0.08333333333333337, "N": 8, "max": 1.0, "median": 1.0, "min": 0.6666666666666666, "p10": 0.6666666666666666, "p90": 1.0}, "geometry_coverage": {"IQR": 0.14105499834217516, "N": 8, "max": 1.0, "median": 0.9145833333333333, "min": 0.6754807692307693, "p10": 0.7087139423076922, "p90": 0.9912500000000001}, "pose_success": {"IQR": 0.0, "N": 8, "max": 1.0, "median": 1.0, "min": 1.0, "p10": 1.0, "p90": 1.0}, "s_valid_fraction": {"IQR": 0.0, "N": 8, "max": 1.0, "median": 1.0, "min": 0.34615384615384615, "p10": 0.8038461538461539, "p90": 1.0}, "tracking_persistence": {"IQR": 0.10546875, "N": 8, "max": 1.0, "median": 0.9453125, "min": 0.84375, "p10": 0.8546875, "p90": 1.0}}`
fake quality summary: `{"component_success_fraction": {"IQR": 0.0, "N": 10, "max": 1.0, "median": 1.0, "min": 0.0, "p10": 0.9, "p90": 1.0}, "geometry_coverage": {"IQR": 0.15460069444444446, "N": 10, "max": 0.9953125, "median": 0.8984375, "min": 0.5642361111111112, "p10": 0.6439236111111112, "p90": 0.99171875}, "pose_success": {"IQR": 0.0, "N": 10, "max": 1.0, "median": 1.0, "min": 1.0, "p10": 1.0, "p90": 1.0}, "s_valid_fraction": {"IQR": 0.0, "N": 10, "max": 1.0, "median": 1.0, "min": 0.0, "p10": 0.9, "p90": 1.0}, "tracking_persistence": {"IQR": 0.28515625, "N": 10, "max": 0.984375, "median": 0.7890625, "min": 0.203125, "p10": 0.315625, "p90": 0.9703125}}`

## 5. Fake Response Results

主 video/source score 按已有 real-val 实际 artifact 使用：每个视频先取每个窗口的 component-time p95，再对窗口 p95 取 source median；这与 ADR 中简写的 primary p95 存在表述差异，未静默改用跨窗口 pooled p95。

| model | real median/IQR | fake median/IQR | AUROC | Cliff's delta | cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| M1_delta_s | 0.427525/1.43197 | 0.0356123/0.53524 | 0.333333 | -0.333333 | 0.078125–0.6125 |
| M2_delta2_s | 0.327589/1.66414 | 0.00300128/0.0493664 | 0.194444 | -0.611111 | 0–0.444444 |
| M3_delta_s_delta2_s | 0.672858/1.98637 | 0.0534897/0.638754 | 0.347222 | -0.305556 | 0.09375–0.625 |

## 6. Generator-specific Results

### svd (N=5 fake; same N=8 real_val reference)
- M1_delta_s: fake N=5 (scored=5), median=0.0226569, IQR=1.4223, AUROC=0.375, Cliff's delta=-0.25
- M2_delta2_s: fake N=5 (scored=5), median=0.00269984, IQR=0.220319, AUROC=0.25, Cliff's delta=-0.5
- M3_delta_s_delta2_s: fake N=5 (scored=5), median=0.0261533, IQR=1.50822, AUROC=0.375, Cliff's delta=-0.25

### cogvideo (N=5 fake; same N=8 real_val reference)
- M1_delta_s: fake N=5 (scored=4), median=0.139604, IQR=0.287797, AUROC=0.28125, Cliff's delta=-0.4375
- M2_delta2_s: fake N=5 (scored=4), median=0.00951193, IQR=0.0220801, AUROC=0.125, Cliff's delta=-0.75
- M3_delta_s_delta2_s: fake N=5 (scored=4), median=0.238569, IQR=0.43586, AUROC=0.3125, Cliff's delta=-0.375

## 7. Real-only Empirical Tail Response

- M1_delta_s: tau95=56.5956, real-max=86.1607, overall fake>tau95=0, overall fake>real-max=0; SVD={"fake_gt_real_max_fraction": 0.0, "fake_gt_tau95_fraction": 0.0}, CogVideo={"fake_gt_real_max_fraction": 0.0, "fake_gt_tau95_fraction": 0.0}
- M2_delta2_s: tau95=37.4962, real-max=56.6655, overall fake>tau95=0, overall fake>real-max=0; SVD={"fake_gt_real_max_fraction": 0.0, "fake_gt_tau95_fraction": 0.0}, CogVideo={"fake_gt_real_max_fraction": 0.0, "fake_gt_tau95_fraction": 0.0}
- M3_delta_s_delta2_s: tau95=57.6406, real-max=87.4432, overall fake>tau95=0, overall fake>real-max=0; SVD={"fake_gt_real_max_fraction": 0.0, "fake_gt_tau95_fraction": 0.0}, CogVideo={"fake_gt_real_max_fraction": 0.0, "fake_gt_tau95_fraction": 0.0}

## 8. Measurement-quality Confounding

score-quality Spearman: `{"M1_delta_s": {"component_success_fraction": -0.2608745973749755, "geometry_coverage": -0.5052116216296061, "pose_success": null, "s_valid_fraction": -0.408248290463863, "tracking_persistence": -0.26814450701075104}, "M2_delta2_s": {"component_success_fraction": -0.3354101966249685, "geometry_coverage": -0.3825874416224202, "pose_success": null, "s_valid_fraction": -0.408248290463863, "tracking_persistence": -0.01722028944105741}, "M3_delta_s_delta2_s": {"component_success_fraction": -0.2608745973749755, "geometry_coverage": -0.5199265232304685, "pose_success": null, "s_valid_fraction": -0.408248290463863, "tracking_persistence": -0.287824837800531}}`

The per-video table is `metrics/per_video_response.csv` and `.json`; no video was dropped by anomaly score. Quality is reported descriptively and was not used to refit, filter, or tune the model.

## 9. First- vs Second-order Evidence

M1=ΔS、M2=Δ²S、M3=[ΔS,Δ²S] 的比较只按上述 frozen scores 解读；M3 的 condition number 保持 frozen，不因 fake 结果重训或重新 regularize。

## 10. Interpretation and Limitations

该结果只能回答 frozen structural normality 是否对当前真实生成视频产生 exploratory response，不能证明结构异常 GT、最终跨域检测能力或定位能力。限制包括 unpaired source/domain、real N=8、fake N=10/5 ordinals、仅 SVD/CogVideo、M3 高 covariance condition、无 spatial anomaly GT、无 paired HD-VG real，以及当前 S_t/component baseline。

## 11. Conclusion

`WEAK_OR_GENERATOR_SPECIFIC_FAKE_RESPONSE`
