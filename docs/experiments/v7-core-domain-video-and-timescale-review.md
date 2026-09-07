# V7 Video-Level Domain and Multi-Timescale Window Review

Status: `VIDEO_LEVEL_HUMAN_REVIEW_REQUIRED`

## 1. Research questions

**RQ1 — Video-level Core Domain eligibility.**  Among the fixed real-video
population, how many complete videos satisfy the V7 Persistent Structured
Motion Core Domain definition?

**RQ2 — Temporal observation adequacy.**  For videos that pass RQ1, which
physical observation durations provide enough continuous dynamic evolution for
later `S_t`, `ΔS_t`, and `Δ²S_t` review?

These are separate labels.  A weak short window does not imply that the
complete video is out of domain, and a dynamic complete video does not make
every short window adequate.

## 2. Motivation

The historical V6 fixed 16-frame material is approximately 0.5 seconds for
the relevant videos.  A complete approximately 5-second video can contain a
clear action while one fixed window shows only a very small visible change.
Therefore the old window-level observation cannot be used as a video-level
physical-domain decision.

The historical protocol is retained for provenance but marked
`HISTORICAL_WINDOW_LEVEL_PROTOCOL_SUPERSEDED`: it conflated complete-video
domain eligibility with adequacy of one fixed analysis window.

## 3. Fixed population

The v2 review reuses the Phase-A `review_manifest.json` identities exactly;
it does not reselect from the pilot manifest.  The population is 96 original
real videos: 64 `train` and 32 `val`.  Fake candidates are not included
(`fake_count = 0`).  Original relative paths and split assignments are
preserved, while the v2 CSV records each resolved absolute source-video path.

Population material:

- `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_core_domain_review_v1/review_manifest.json`
- `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_core_domain_review_v2/video_review/full_video_manifest.json`
- `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_core_domain_review_v2/video_review/full_video_review.csv`

## 4. Video-level domain protocol

The reviewer labels the complete original RGB video with one of:

- `IN_DOMAIN`
- `OUT_OF_DOMAIN`
- `UNCERTAIN`

The complete-video inclusion criterion is one dominant persistent structured
motion process, such as rigid or quasi-rigid translation/rotation or simple
object manipulation, with limited occlusion and camera motion.  The video
must contain a real dynamic process.  Dominant complex independent
multi-subject interaction, complex articulation, severe occlusion, fluid,
smoke, fire, fragmentation, merge/split, or severe topology change are
exclusion or uncertainty reasons.  Visual complexity alone is not an
exclusion criterion.

Allowed reason codes are `SINGLE_STRUCTURED_MOTION`,
`MULTI_SUBJECT_COMPLEX`, `ARTICULATED_COMPLEX`, `STATIC_VIDEO`,
`STRONG_OCCLUSION`, `TOPOLOGY_CHANGE`, `FLUID_SMOKE_FIRE`,
`CAMERA_MOTION_DOMINANT`, `NO_PERSISTENT_STRUCTURE`, `SUBJECT_TOO_SMALL`,
and `OTHER`.  `STATIC_VIDEO` applies only when the complete video is nearly
static, never merely because one 0.5-second window is weak.

The v2 full-video contact sheet samples 20 unique source frames uniformly
from the first through the last decoded frame in a 4×5 layout.  Each cell
contains only the source frame index and true PTS timestamp.  No component,
XYZ, tracking, depth, pose, score, or fake information is shown.

## 5. Temporal window protocol

Window adequacy is reviewed only after a video is manually marked
`IN_DOMAIN`.  The implementation uses true decoded PTS timestamps and the
following physical durations:

- 0.5 s
- 1.0 s
- 2.0 s

For each duration, deterministic centered anchors are 25%, 50%, and 75% of
the full decoded timestamp span.  A requested interval is clipped at the
video boundaries; it never wraps and never receives duplicate-frame padding.
If the complete span is shorter than a requested duration, the row records
`WINDOW_DURATION_UNAVAILABLE` and the available frames without artificial
padding.

The future window CSV contains only source identity, split, timestamp/frame
boundaries, review paths, and manual review fields.  It has no model-output
columns.  The window generator rejects blank/invalid video decisions and
skips `OUT_OF_DOMAIN` and `UNCERTAIN`; only `IN_DOMAIN` rows can produce the
9 (3 durations × 3 anchors) candidate windows.  Preview/contact-sheet
generation is available behind an explicit CLI flag and was not run in this
phase because video-level decisions are still blank.

## 6. Leakage and cherry-picking control

Selection and material generation read only the frozen Phase-A identities,
split, original path, and original RGB/PTS timeline.  They do not read
component counts, XYZ, tracking/depth/pose quality, scores, fake videos,
labels, or detector outputs.  Fixed anchors prevent selecting the visually
largest motion event after watching the entire video.  Any future
event-centered protocol must be registered separately.

## 7. Separate human review

Video-level review answers whether the complete physical process belongs to
the Core Domain.  Window-level review answers whether one particular duration
and anchor expose enough continuous dynamic change for later structural
analysis.  Video finalization includes only explicit `IN_DOMAIN` rows and
excludes `UNCERTAIN` by default.  Window finalization accepts only videos in
that finalized video manifest and excludes `UNCERTAIN` windows by default.

No finalizer was executed in this phase.

## 8. Planned paper-facing metrics

After manual video review, report `N_total`, `N_IN`, `N_OUT`, `N_UNCERTAIN`,
retention, train/validation stratification, and exclusion-reason
distribution, with a later predeclared Wilson or bootstrap 95% interval.

After window review, report `DYNAMIC_ADEQUATE` rates for 0.5/1.0/2.0 seconds,
stratified by 25/50/75% anchor, together with too-weak, partial, occlusion,
camera-motion, unavailable, and uncertain counts.  A final duration is not
predeclared: it must balance adequate-rate, usable-population retention,
multiple observable evolution states, and runtime cost.

## 9. Implementation and status

The implementation is limited to:

- `src/sparse3d_forgery/experiments/v7_core_domain_timescale.py`
- `scripts/build_v7_full_video_review.py`
- `scripts/build_v7_window_review.py`
- `scripts/finalize_v7_video_domain_review.py`
- `scripts/finalize_v7_window_review.py`
- `tests/experiments/test_v7_core_domain_timescale.py`

Generated v2 materials were built successfully for 96/96 videos (64 train,
32 val), with 96/96 full-duration contact sheets and no automatic decisions.
The complete v2 review root is:

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_core_domain_review_v2/`

Current status remains:

`VIDEO_LEVEL_HUMAN_REVIEW_REQUIRED`

This report does not claim domain mismatch, component validity, `Δ²S`
validity, or fake detectability.

## 10. Boundaries

This phase did not modify the component algorithm or its 1 m / 0.05 m probe
thresholds, run a frontend, use fake videos, select by model result, train a
normality model, calculate AUROC, change the V7 formal design contract, or
enter window finalization.  No old R7/V5 repository was accessed.
