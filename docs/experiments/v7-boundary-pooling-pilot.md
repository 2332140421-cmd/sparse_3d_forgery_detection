# V7 boundary partition and local-pooling matched pilot

This document records the executable protocol. The numerical result is written
by the independent runner to
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1/`.

The pilot uses the frozen 289-point density-matched `ParticleSequence`
artifacts, 16 sources and 192 windows. `H` is the historical local grouping
from the preceding pilot. `B` partitions each H group with a first-frame
YOLO26m-seg instance boundary; it never merges H parents or reassigns points
in later frames. A valid point outside all masks is retained as background;
invalid or out-of-raster UV is explicit `UNASSIGNED`. Instance category,
confidence and mask area are not model inputs.

The predeclared matrix is `H/B × MEAN/MAX × A/C/D`. A, C and D are the
existing `UNORDERED_STATE`, `ORDERED_SECOND` and `PERMUTED_SECOND` arms. H and
B are restricted to their common valid window intersection for matched
training and source-disjoint LOSO evaluation. The encoder and linear head are
unchanged; MEAN reproduces the existing local-group mean, while MAX applies
the same head to each local-group representation and takes the maximum logit.
CTRL windows remain descriptive and are not training labels.

The runner has no formal `src` changes, does not rerun tracking/depth/pose, and
does not use the dense branch, ROI labels, quality features or source identity.
It writes progress, failure tracebacks, coverage, OOF scores, fold/model
audit, source metrics, paired source bootstrap and fixed sparse visualizations.
It is a development pilot, not a sealed-test or pixel-localization result.

Run after the committed code is available:

```bash
research_tools/v7/boundary_pooling_probe/run_all.sh
```

Resume with the same command. Inspect `progress.json`, `run_summary.json` and
`evaluation/summary.json` under the data artifact root.
