# V7 04LAX ROI evidence review

This bounded case review prepares manual, frame-specific source-pixel ROI
confirmation for the existing 04LAX fake local-observation recovery case. The
entry point reads the saved O/R arrays and R H structure, decodes only frames
487, 488, 503, 506, 509, 512 and 515, and writes a self-contained review page.
It does not rerun a frontend or calculate model scores.

Run from the repository with the project environment:

    .venv/bin/python -m research_tools.v7.local_observation_recovery_probe.roi_evidence

The page is written under the existing case data directory at
roi_evidence_review_v1/review/index.html. Its exported rectangles are
frame-specific source-pixel coordinates and remain a user download; the page
does not write to the server. Until a user supplies the downloaded
roi_annotations_04LAX.json, the output status remains WAITING_FOR_ROI.

## Second-stage confirmed-ROI analysis

The supplied annotation export is validated against the case manifest, the
video hash, the 480x360 source-pixel coordinate convention, and the saved
frame PTS values. Only entries with `human_confirmation_status=CONFIRMED`
are used; `SKIP`, `UNCERTAIN`, and `PENDING` remain excluded. Run the bounded
read-only analysis with:

    .venv/bin/python -m research_tools.v7.local_observation_recovery_probe.roi_evidence_phase2 \
      --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_observation_recovery_probe_v1/roi_evidence_review_v1 \
      --annotations /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_observation_recovery_probe_v1/roi_evidence_review_v1/roi_annotations_user.json

The phase-2 output contains frame-level ROI observation counts, point
membership records, confirmed-ROI overlays, and `report.md`. The current
observation-support feature manifest has no 04LAX row, so this case is
reported as `NOT_AVAILABLE` for the earlier ROI phase; that phase did not
silently treat the missing formal manifest row as a model input.

## Final bounded frozen five-time readout

The final case-specific step reconstructs the current five-time input from the
verified R geometry cache, rather than wrapping the old sliding triplets. It
calls the current `build_local_groups`, `build_five_time_unit`, and
observation-support Q builder with the saved H history, target PTS and fixed
minimum-common-member rule. The output is independent of the formal feature
manifest:

    .venv/bin/python -m research_tools.v7.local_observation_recovery_probe.frozen_five_time_readout

The output is under
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_observation_recovery_probe_v1/frozen_five_time_readout_v1/`.
The R cache yields 44 retained historical groups, 32 valid five-time units and
12 retained groups rejected because fewer than three members are geometry-valid
at all five target frames; five additional groups were below the historical
minimum size. Targets are frames 503, 506, 509, 512 and 515, with history
frames 488--502 and Q reference frame 502. All six frozen readout models
(STRUCTURE_ONLY and STRUCTURE_SUPPORT, three seeds each) reproduce an existing
legal saved window logit within `1e-5`, and produce finite per-unit and window
logits for this diagnostic case. The final status is
`COMPLETE_WITH_MODEL_READOUT`.

The old R sliding triplets are retained only as an identity/cross-check. Their
stored history scale was defined on the retained raw group, while the current
five-time builder defines the scale on the five-time common members; therefore
their numeric S values are not reused as current model input. `R_requery.npz`
and `R_geometry.npz` agree on frame/PTS/track/UV/visibility identity, while the
former has no finite XYZ and is not used for 3D construction. 04LAX is absent
from the frozen model training and validation source lists, but this remains a
single developed diagnostic and is not a sealed-test result. No new ROI audit,
frontend, tracking, depth, pose, segmentation or training was added.
