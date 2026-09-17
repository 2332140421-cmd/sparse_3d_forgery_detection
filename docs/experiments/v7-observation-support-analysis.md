# V7 observation-support pilot — finite read-only analysis

This note records the small, read-only follow-up for
`v7_activityforensics_observation_support_pilot_v1`.  It consumes the saved
three-seed logits, Q feature arrays, feature manifest, and support records.
It does not rerun a frontend, tracking, depth, pose, segmentation, training,
or model forward pass.

The generated tables and report are kept outside Git at:

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_observation_support_pilot_v1/analysis_v1/`

The entry point is
`research_tools/v7/observation_support_pilot/analyze.py`.  It verifies the
83-window / 14-source / 42-real / 41-fake validation population, reproduces
the saved mean-logit metrics, reports source/seed differences, separates
target-stage Q changes from history-only changes, and summarizes threshold
error transitions.  Generated CSVs are analysis artifacts, not new model
inputs or evaluation definitions.

The pilot point estimates are positive for both requested comparisons, but
the source-paired bootstrap intervals cross zero and seed/source directions
are not uniform.  The analysis therefore recommends no expansion from this
development set; an eventual follow-up would need a previously unused frozen
source/time population.  Q remains an observation-support diagnostic rather
than spatial forgery ground truth, and the unformed-group count remains an
uncovered measurement region.
