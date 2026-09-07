# V7 explicit geometry runtime and structure probe

Status: completed runtime falsification; explicit frontend usable for the bounded V7 probe, component representation requires refinement.

## Scope

- Branch: `v7-dynamic-structure`.
- Population: real-only. The fixed V6 temporal-probe manifest supplied 8 pilot windows and the first 32 lexicographically ordered `real_val` windows for expansion.
- No fake video, label, authenticity score, old R7/V5 code, model, feature, cache, or experiment result was read or used.
- This report distinguishes runtime facts from research inferences. It is not a claim that the frontend is metric-accurate on arbitrary video.

## Runtime frontend

- Tracking: official TAPNet source commit `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`, official causal BootsTAPIR checkpoint SHA-256 `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.
- Depth: Apple Depth Pro source commit `9efe5c1def37a26c5367a71df664b18e1306c708`, checkpoint SHA-256 `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`.
- Pose: Open3D 0.19 sequential legacy RGB-D odometry; the Open3D target-from-source result is inverted while accumulating `world_from_camera`, with world fixed to camera zero.
- The implementation interprets the official Depth Pro output as optical-axis z-depth for back-projection. This is an explicit runtime interpretation, not an independent metric-accuracy validation.
- Intrinsics use the first frame's Depth Pro focal estimate for every frame in the window. This is a causal fixed-camera assumption; raw per-frame focal estimates remain in output metadata. Principal point is centered and distortion is zero by explicit experimental assumptions.
- RGB-D odometry uses a resized pose raster (maximum side 320) with pixel-center-aware intrinsics scaling. Final UV, depth sampling, and lifting use the original decoded raster.
- Outputs preserve decoder source frame indices and PTS timestamps. Invalid geometry remains NaN with explicit masks; there is no zero fill, interpolation, semantic track identity, label, or handcrafted motion input.

## Causality and repeatability

- BootsTAPIR is called online in frame order with causal context; Depth Pro is invoked independently per frame; RGB-D transforms are accumulated only from preceding adjacent pairs.
- A two-window prefix test compared an 8-frame output to the first 8 frames of the corresponding 16-frame output. Across 1,024 common valid XYZ positions and 1,024 UV observations, all distances were exactly zero; frame indices and masks were exactly equal.
- Two independently executed 8-window runs were exactly identical across 7,963 common valid XYZ positions and UV observations. Their frame indices and masks were exactly equal.
- The implementation never revises already-emitted history. This validates implementation causality for this bounded execution, not future provider behavior.

## Stage results

| Stage | Windows | Frames | Final XYZ coverage | Adjacent pose result |
| --- | ---: | ---: | ---: | --- |
| Stage 0 | 2 real | 16 | 1.000, 1.000 | both 15/15 valid |
| Pilot A | 8 real | 16 | median 1.000; range 0.875-1.000 | 8/8 complete |
| Pilot B | same 8 | 16 | repeat run | exact repeatability |
| Expansion | 32 real | 16 | median 0.9765625 | pair rate 1.000; 32/32 complete |

- The lower-vs-upper quartile same-track step diagnostic produced a median ratio of 0.0168474. It is explicitly a nonsemantic runtime proxy, not static-ground-truth identification: lower-quartile median 0.0044616 m; upper-quartile median 0.2648222 m.
- The expansion's largest observed same-track step was 51.7974 m. It is retained as an outlier/scale-sensitivity warning, not silently removed.
- GPU execution completed on the RTX 4090 D. Per-window geometry elapsed time totaled 155.695 s for the 32-window expansion; external model load and online tracking are reported separately by runtime execution.

## Part A classification

`EXPLICIT_FRONTEND_USABLE_FOR_V7_PROBE`

This classification is limited to the stated real-only probe: causal history was immutable, repeated execution was exact, complete pose was a majority (indeed 8/8 in pilot), median pilot coverage was at least 60%, no runtime coordinate-scale break was observed, the nonsemantic jitter proxy was below observed motion, and the 4090 D executed the pipeline. Depth scale accuracy and calibration assumptions remain unresolved research limitations.

## Part B dynamic structure

- Candidate components use only valid geometry, initial 3-D proximity and robust relative-motion coherence. The graph is not a Part hierarchy and does not encode labels, semantic classes, identity embeddings, or authenticity.
- The deterministic probe configuration was initial distance <= 1.0 m, median relative change <= 0.05 m, minimum component size 3, and at least 8 overlapping frames.
- 31/32 expansion windows produced at least one component. Component count median was 2 (range 0-5); median component particle fraction was 0.859375, but p10 was 0.1578125 and one window had no component.
- For the 31 nonempty windows, median-of-window-medians was 0.0217941 for structure-state change, 0.619492 for timestamp-aware first-order change, and 24.9179 for second-order change.

## Falsification outcome

The explicit frontend did not trigger a tracking, pose, depth, or runtime scale hard blocker for the bounded probe. However, the component representation is not stable enough to serve as a normality-training target: it has one empty window, strongly variable particle coverage, a scale-sensitive fixed threshold, and large geometric outliers. Therefore no V7 normality model was trained and no fake scoring was run.

Required next action: `REFINE_COMPONENT_REPRESENTATION`.

## Reproduction and artifact boundary

- Runtime artifacts: `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_explicit_geometry_probe_v1`.
- External source and checkpoints: `/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v7_explicit_geometry`.
- Numerical ParticleSequence artifacts are NPZ plus JSON metadata outside Git. They were saved and loaded with `allow_pickle=False` and validated by the repository validator.
- The isolated frontend environment is `/root/autodl-tmp/envs/v7-explicit-geometry`; it uses Python 3.12.3, torch 2.12.1+cu130, NumPy 1.26.4, SciPy 1.13.1, Open3D 0.19.0, timm 1.0.20, PyAV 18.1.0, and OpenCV 4.11.0.86. It was created with `--system-site-packages`, so it is reproducible by its recorded versions but not fully hermetic.
