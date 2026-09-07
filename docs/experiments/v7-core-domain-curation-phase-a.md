# V7 Core Domain Curation — Phase A

Status: HUMAN_REVIEW_REQUIRED

## 1. Research Question

Does the instability observed in the broad real-validation population arise partly from a heterogeneous scene/domain distribution rather than from component discovery alone?

This phase creates an independent human-review population. It does not claim that dataset mismatch has been proved.

## 2. Experimental Motivation

The preceding V7 runtime probe used the first 32 lexicographically ordered `real_val` windows from the frozen temporal-learnability manifest. That population was not filtered for persistent structured motion. Its component variability therefore could not separate domain mismatch from component-discovery failure.

This phase does not modify `motion_coherent_components()`, its 1.0 m and 0.05 m baseline thresholds, minimum size 3, or minimum overlap 8.

## 3. Core Domain Phase-2 Operational Definition

A candidate is eligible for manual consideration when the original visual window appears to contain:

- one dominant dynamic subject;
- persistent observability through the analysis window;
- primarily rigid or quasi-rigid motion, including translation, rotation, or simple rigid-body motion;
- limited occlusion;
- no dominant multi-subject interaction;
- no dominant complex articulated motion;
- no liquid, smoke, fire, large fragmentation, merge/split, or strong topology change;
- more than an essentially static scene;
- visible changes not completely dominated by camera motion.

This is an operational Phase-2 experimental protocol, not a permanent V7 boundary or an automatic classifier.

## 4. Population Protocol

The source was:

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/temporal_learnability_probe_v1/pilot_manifest.json`

Selection was deterministic and independent of all V7 outputs:

- `real_train`: source_video_id ascending, first 64;
- `real_val`: source_video_id ascending, first 32;
- exact manifest `frame_indices` reused;
- no random supplementation;
- no fake candidate;
- no component count/fraction, XYZ outlier, tracking quality, depth quality, score, or detector output read for selection.

Actual candidate count: 96 windows (64 train, 32 val). Every selected window contained exactly 16 manifest-defined frame indices.

## 5. Human Review Protocol

Review is based on the original RGB content shown in generated materials. A decision is not an authenticity label and is not an anomaly annotation.

The generated CSV is:

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_core_domain_review_v1/review_template.csv`

All 96 `decision` and `reason_code` cells are intentionally blank. Allowed decisions are `IN_DOMAIN`, `OUT_OF_DOMAIN`, and `UNCERTAIN`. Allowed reason codes are recorded in `review_schema.json`.

`IN_DOMAIN` requires the reviewer to select `SINGLE_STRUCTURED_MOTION`. `UNCERTAIN` is excluded by the deferred finalizer.

## 6. Leakage Control

This phase uses no fake video, authenticity label, component metric, XYZ quality, tracking metric, depth metric, detector score, or model output to select or pre-label a window. The builder only consumes manifest identity, split, source path, and fixed frame indices.

## 7. Generated Review Materials

Review root:

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_core_domain_review_v1/`

Generated successfully:

- 96 raw-preview MP4 files under `raw_previews/`;
- 96 4×4 PNG contact sheets under `contact_sheets/`;
- 96-row `review_template.csv`;
- `review_manifest.json`;
- `review_schema.json`;
- `population_summary.json`.

Generation success: 96/96. Generation failure: 0. A read-back check decoded one preview as 16 RGB frames at 480×640 and read one contact sheet as 1280×1008 RGB.

Contact-sheet text is limited to source_video_id, split, frame index, and timestamp. It contains no component, particle, depth, XYZ, tracking, fake score, or model-quality display.

Example train sheets:

- `contact_sheets/0_30_s_academic_v0_1_0005-6134155761.png`
- `contact_sheets/0_30_s_academic_v0_1_0006-2612683715.png`
- `contact_sheets/0_30_s_academic_v0_1_0008-2403134475.png`

Example validation sheets:

- `contact_sheets/0_30_s_academic_v0_1_0002-10192494165.png`
- `contact_sheets/0_30_s_academic_v0_1_0013-6480018109.png`
- `contact_sheets/0_30_s_academic_v0_1_0029-3018394896.png`

## 8. Deferred Finalization

`scripts/finalize_v7_core_domain_manifest.py` is implemented but was not run. After manual review, it will:

- validate the review schema;
- include only `IN_DOMAIN`;
- require `SINGLE_STRUCTURED_MOTION`;
- preserve exact frame indices and generated timestamps;
- record the review CSV SHA-256;
- exclude `UNCERTAIN` by default.

No final core-domain manifest exists from this phase.

## 9. Planned Paper Metrics After Review

After manual review, compare broad and core populations using:

- windows with a component;
- component count per window;
- component particle fraction (median, IQR, p10–p90);
- component-size distribution;
- no-component window rate;
- within-window component persistence;
- particle XYZ outlier rate;
- valid-frame rate of `S_t`;
- bootstrap 95% intervals and paired/stratified comparisons where supported.

These metrics are planned, not reported as a result of this curation phase.

## 10. Current Status and Boundaries

Current status: `HUMAN_REVIEW_REQUIRED`.

No component algorithm or baseline threshold was changed. No new clustering, `S_t`, `ΔS_t`, `Δ²S_t`, normality model, AUROC, or fake analysis was run. No new dataset was downloaded. The V7 formal design contract was not modified.

The implementation uses the existing video decoder and true PTS timestamps. Generated MP4/PNG/CSV/JSON artifacts remain outside Git under the review root.
