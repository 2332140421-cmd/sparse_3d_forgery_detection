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
