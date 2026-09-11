# V7 dense candidate points and boundary-constrained local structure pilot

Status: **EXPLORATORY_PILOT — frontend gate blocked**

This pilot uses the frozen 16-source/192-window manifest, official
`yolo26m-depth` and `yolo26m-seg`, and the existing causal BootsTAPIR
checkpoint.  The declared analysis grid is 384×384 with 3-pixel spacing,
centred at 1.5, 4.5, … (16,384 initialized queries).  Queries are grouped
only at the start frame by fixed 24×24 blocks intersected with instance masks;
background is kept per block.  Within each group, at most eight nearest
start-coordinate neighbours form frozen undirected edges.  No confidence,
quality, old component, source, label, or point-count feature is used.

The first fixed window was attempted with the project CUDA environment.  The
BootsTAPIR causal state allocation for all 16,384 queries raised CUDA
out-of-memory on the RTX 4090 D before depth, segmentation, or geometry could
run.  The protocol does not permit silently reducing the grid or reusing the
old 256 analysis, so the 192-window frontend, fixed-edge features, and three
supervised arms are **NOT_RUN**.  The independent output directory contains a
reproducible `protocol.json`, `frontend_summary.json`, `summary.json`, and a
small `review_bundle.zip`; no dense arrays, weights, or videos are committed.

Reproduction command:

```bash
/root/autodl-tmp/projects/sparse_3d_forgery_detection/.venv/bin/python \
  -m research_tools.v7.dense_local_structure_pilot.run_pipeline all --resume
```

The exact runtime result and traceback are in the data-disk output directory
`derived/v7_activityforensics_dense_local_structure_pilot_v1/`.
