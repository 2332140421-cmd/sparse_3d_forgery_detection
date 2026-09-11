# V7 dense local-structure pilot: GPU resume run

The resumable runner audits the two prior dense-pilot output roots, runs one
fixed-window CUDA batch benchmark (128/256/512/1024 queries on the same 1024
queries), and freezes the numerically compatible faster batch. It then reuses
only validated prior arrays or computes missing windows with the unchanged
384x384, 3-pixel grid and 16,384 query identities. The output directory is

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_dense_local_structure_pilot_gpu_resume_v1/`

Start or resume independently with:

```bash
research_tools/v7/dense_local_structure_pilot/run_all.sh
```

Progress is in `progress.json`, the finite benchmark is in
`performance_benchmark.json`, and the final state is in `run_status.json`.
The runner writes `FINAL_SUCCESS`, `FINAL_BUDGET_EXHAUSTED`,
`FINAL_PARTIAL`, or `FINAL_FAILED`; completed artifacts remain resumable with
the same command. The run is an exploratory measurement and is not a formal
detector or a pixel-level localization result.
