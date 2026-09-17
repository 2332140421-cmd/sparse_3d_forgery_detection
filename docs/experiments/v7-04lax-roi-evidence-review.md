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
reported as `NOT_AVAILABLE` for model unit/window logits rather than using a
different model or fabricating a readout. The resulting status is
`COMPLETE_NO_MODEL_READOUT` (ROI evidence complete, model readout unavailable).
