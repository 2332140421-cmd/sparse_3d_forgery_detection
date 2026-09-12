# V7 boundary/pooling review display fix

## Scope

This change corrects the offline human-review page for the already completed
boundary/pooling pilot. It reads the saved 289-point `ParticleSequence`, H/B
grouping, triplet support, screenshots and serialized forward responses. It
does not rerun tracking, depth, pose, segmentation, model fitting or OOF
scoring. The review-only refresh was run with:

```bash
PYTHONPATH=src .venv/bin/python -m \
  research_tools.v7.boundary_pooling_probe.review_and_forward \
  --root /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1 \
  --rebuild-review-only
```

The refresh covered all 192 frozen windows and reused the serialized
`model_response` values. The default page still shows the 16 deterministic
real/fake MANIP/CTRL cases; other windows remain selectable in the index.

## What was corrected

- `frame_records` now carry both `slot` and `track_id`. Triplet common members
  and pairs are explicitly mapped from track identity to array slot; no
  `track_id == slot` assumption is used.
- H and B triplets carry `owner_group_ids`, `common_member_slots`,
  `pair_member_slots`, `members_considered` and `track_ids_considered`. The
  dropdown only lists triplets owned by the currently selected organization
  and local group. No owner produces `NO_VALID_TRIPLET` rather than a fallback.
- `parent_component_id`, H/B `local_group_id`, triplet `component_index` and
  model-response `component_index` are displayed separately. The saved model
  response is an organization-local-group component response; it is a
  classifier logit contribution, not a calibrated probability or localization
  truth.
- Browser media time is matched to the saved PTS timeline with a finite
  tolerance. Outside the saved observation interval, or inside a saved gap,
  the canvas is cleared and no points, groups or edges are drawn. The page
  reports playback time, matched PTS, delta, source frame and `model-used`.
- A window change is the only operation that changes the video source. H/B,
  triplet, model and display-color changes reuse the loaded video and keep its
  current time. Each side has independent source/window selectors. A window
  change uses a token to discard stale asynchronous detail loads.
- The page provides observation-start and triplet `t0/t1/t2` seek buttons,
  uses `requestVideoFrameCallback` when available (with event fallback), and
  offers deterministic group colors separate from visible/geometry-valid/
  triplet-common status colors. Edges are only the selected triplet pairs.
- Screenshots are generated from exact source frame indices and saved PTS;
  their overlay is a frozen track-identity display. The segmentation
  assignment remains explicitly first-frame-only.

## Time-origin check

For representative `0AGCS`, fake MANIP window
`0003_MANIP_25::fake`, the source stream reports `time_base=1/12288` and
`start_time=0`. PyAV decoded source frames 446, 447 and 469 at
18.5833333333, 18.625 and 19.5416666667 seconds, respectively, exactly
matching the saved sequence PTS (`frame_indices` 446–469). No FPS-derived
offset or timestamp fallback is used. A browser-specific visual check remains
pending because this server has no installed Chromium/Firefox executable.

## Representative audit inputs

The representative source is `0AGCS`, fake MANIP, with the earliest frozen
window `0003_MANIP_25::fake` (and its independently selectable real pair).
The refreshed detail reports:

- saved observation: 24 frames, source frames 446–469, PTS
  18.5833333333–19.5416666667 s;
- H/B support is read from the saved feature JSON;
- H triplet 3 maps to H local group 1 through its member slots, while its
  support `component_index=1` is not confused with the original
  `parent_component_id=0`;
- model responses remain serialized and are shown with their component scope.

The unchanged upstream inputs were fingerprinted for this audit:

```text
scores/oof_window_scores.csv  badb4f8d947251a2312322478c9002932fae5cd9c004703631827b5873a4d142
models/fold_models.json       5f0e2b9d75e091b46de5487d0ee6a72122c5cb9817225f2a82276bf9cd44745a
```

## Verification and limitations

The targeted behavior suite passes (`11 passed`). It covers PTS range and
tolerance matching, rejection of saved gaps, explicit track-to-slot mapping,
local-group ownership and page controls. The complete V7 browser acceptance
is not claimed: no supported browser binary was available on the server, so
`browser_visual_verification` remains `false`. To inspect the page locally,
serve the review directory without a CDN:

```bash
cd /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1/review
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/` and check one MANIP, one CTRL, a window with no
triplet support, and a time outside the saved observation interval. Exact
source-frame screenshots remain the authoritative frame check; player seek is
only a convenience.

This is a `DISPLAY_ONLY` correction. It does not create pixel-level masks,
spatial ground truth, calibrated probabilities or a new detection method.
