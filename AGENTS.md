# Repository Instructions

## Academic research code boundary

This repository serves an academic thesis and scientific hypothesis
evaluation. It is not intended to become a general-purpose or
production software platform.

All implementation decisions must use the smallest design that can:

1. implement the frozen research method;
2. support the current experiment or ablation;
3. preserve correctness and reproducibility;
4. prevent invalid data from corrupting experimental conclusions.

Do not introduce generalized frameworks, plugin systems, registries,
services, databases, production observability, distributed
orchestration, broad provider factories, or speculative abstractions
unless the user has explicitly approved them for a concrete current
experiment.

Tests should protect research-critical contracts and regressions.
They should not attempt to cover production-scale adversarial input
spaces without a demonstrated experimental need.

Before adding a new module or abstraction, state:

- the thesis question or experiment it supports;
- its immediate caller and output;
- why a simpler implementation is insufficient.

If these cannot be identified, do not add the complexity.

Research reproducibility remains mandatory. This boundary does not
permit silent data repair, uncontrolled randomness, unverifiable
results, or inconsistency between the paper and implementation.

## Authority and scope

- Before every task, read `docs/design_contract.md` completely.
- Treat `docs/design_contract.md` as the repository's authoritative method baseline under the priority order stated there.
- Do not copy or automatically migrate code, configuration, schemas, tests, artifacts, or documentation from the old repository.
- Do not infer this repository's layout or responsibilities from the old repository's directories.
- Access the old repository only for a user-authorized, read-only engineering audit.

## Method boundaries

- Do not restore old models, detection logic, residuals, feature schemas, fusion, or training routes.
- Do not introduce explicit or implicit Part, Block, Region, or Surface hierarchies or discovery.
- Do not use handcrafted motion inputs such as velocity, acceleration, jerk, direction, curvature, or motion classes.
- Do not treat masks as ordinary continuous features or concatenate masks with XYZ as ordinary numeric input.
- Do not let a model read provider-private artifacts. Provider outputs must first become a canonical `ParticleSequence`.
- Do not let authenticity labels, dataset split identity, paths, provider identity, or other provenance enter model numeric input.
- Do not treat an unfrozen question in the design contract as settled.
- Missing and invalid observations must remain explicit. Silent repair, implicit fallback, and concealed zero filling are prohibited.

## Change discipline

- Implementations must agree with `docs/design_contract.md`.
- Every code change requires matching tests.
- Any change to the underlying method design must, in the same change:
  - update `docs/design_contract.md`;
  - add or update an ADR under `docs/decisions/`;
  - review and update this `AGENTS.md` when necessary;
  - state which earlier wording or decision is superseded.
- Never delete an accepted ADR. Supersede it with a new numbered ADR.
- Do not download weights, install large dependencies, or run long GPU jobs without explicit user authorization.

## V8 scope and entry points

- For V8 work, start at `docs/v8/current_status.md` and `docs/v8/design_contract.md`,
  together with the applicable experiment contract/ADR. Implementations live in
  `research_tools/v8`; the early V7 restrictions above do not override the explicit
  V8 exceptions in root design contract section 14 (dense fields, components,
  motion observations and validity state inputs).
- New-server setup and localization preparation: `docs/v8/new_server_setup.md`;
  proposed spatial supervision: `docs/v8/spatial_localization_plan.md`, ADR-0021.
- Distinguish historical results from local trained assets and new execution evidence.
  A random interface model must never be called COMPONENT_ADAPTED.
- Joint edit-region localization: ADR-0022 and `docs/v8/joint_spatial_contract.md`.
  User authorizes selected full-video frontend extraction and two paired-initialized
  CELL_LOC/COMPONENT_LOC models. Keep both 256 geometry and 512 localization paths,
  all-frame temporal gradients, and edit-region labels distinct from old visible-anomaly labels.
- Head decoupling: ADR-0023 and `docs/v8/head_decoupling_contract.md` authorize
  COUPLED/SEPARATE_HEADS on full COMPONENT_LOC, original monitor16 training/dev32,
  original train96 statistics, read-only source caches, and exactly30 epochs.
  Separate detection head reads GRU hidden; never gate pixel masks by detection.
- Pretrained spatial/context comparison: ADR-0024 and
  `docs/v8/pretrained_spatial_context_contract.md` authorize the two specified
  conditions, full ANAKIN48 train/16 dev, frozen official AINPAINT spatial backbone
  including trained LoRA, and spatial-only decoder training. The context condition
  adds full dense geometry/component/attention/GRU through zero-initialized residual
  projection. Source caches/statistics remain read-only; no old trained weights.
- Fixed decoder BN comparison: ADR-0025 and
  `docs/v8/spatial_context_frozen_bn_contract.md` authorize the same two full models
  from official initialization with fixed official BN buffers but trainable affine
  parameters. Keep Dropout's original modes and the default option False. Reference
  existing visual/frontend caches; no new features, downloads, or old-model fine-tuning.
- Spatial LoRA adaptation: ADR-0026 and `docs/experiments/v8-spatial-lora-adaptation.md`
  authorize one online-visual condition with original DINO frozen, existing LoRA and
  decoder trainable, backbone eval, and fixed official decoder BN buffers. Preserve
  full-frame pair weighting; old patch caches are parity references only, never
  inputs to the adapted model's training or final scores.
- Spatial supervision expansion: ADR-0027 and `docs/experiments/v8-spatial-supervision-expansion.md`
  authorize two matched visual conditions, a source-isolated enlarged research subset,
  visual-only RGB/GT caches, and at most four full-pair compute probes. Preserve original
  roles, all frames and loss; never score or extract DINO for test_reserved. Record
  experiment completion before publication; Git/network failures must not overwrite it.
- Matched temporal detection: ADR-0028 and `docs/experiments/v8-3d-temporal-detection-core.md`
  authorize only STRUCTURE_2D/STRUCTURE_3D, three paired seeds, fresh detection
  initialization and AF48/20/20 source pairs. Reuse attention/GRU/full-video gradients;
  2D grouping/normalization/model inputs must never read XYZ-derived fields.
  Preserve all content cells when geometry/history is missing. Test best only after
  all six models finish; no ANAKIN reserved-test access or extra ablations.

## PVST independent method

- ADR-0029 and docs/pvst/design_v02.md define the user-authorized new method.
- Use research_tools/pvst; never substitute V8 cell means/components or V9 query partitions.
- Preserve native content pixels, point128/fine256/coarse512, both three-layer causal
  scales and point-to-pixel bypass. UV must not read XYZ-derived inputs.
- Exactly two conditions, one seed, AF8/4 and ANAKIN8/4 source pairs. If equivalent
  execution cannot fit hardware, measure and report RESOURCE_BLOCKED; do not shrink.
- PVST v0.3: ADR-0030 and docs/pvst/design_v03.md explicitly supersede only
  v0.3's XYZ voxel scale, full-video update/coverage and training budget from v0.2.
  Select the finest authorized training-coordinate-budget scale; never cap nodes.
  Preserve every valid point and the native point bypass; only one epoch per condition.
  Use the separate v03 entry, shared read-only caches, frozen clips and complete-video
  evaluation. Resource failure stops training; retain v0.2 evidence unchanged.

## Depth-slice relation design only

- ADR-0031 and `docs/pvst/depth_slice_relation_design.md` register an independent
  design-stage correction on `pvst-depth-slice-design`: viewing-depth slabs and
  native point/local features. ADR-0032 supersedes the real-only mainline with
  supervised temporal/depth pair relations, and authorizes only the small CPU audit;
  real-only prediction remains an alternative, not a concurrent implementation.
- Root contract section16 governs this proposal; existing PVST cube contracts and
  resource evidence remain unchanged. No model implementation/training is authorized
  by the design document; unresolved axes/units/alignment and parameters stay explicit.

## RGB-D adaptive set-to-set diagnostic

- ADR-0036 and root contract16.4 authorize RGBDSetSlices on pvst-set-to-set-relations:
  overlapping learned role sets, induced states, joint interactions and relation-only
  frame GRU. Keep native RGB-D/SlabPool, absolute depth IDs and actual causal PTS.
  Only two specified training pairs, four24-frame clips, max80 updates/1800s worker.
  No formal training, dev/test scoring, new frontend or old checkpoint fine-tuning.
  Old A/B/NORM paths and user JSON stay unchanged; bounded diagnostic is independent.
- ADR-0037 and root contract16.5 authorize only the matched RGBD and
  RGB_DEPTH_PARTITION two-pair fit on `pvst-rgb-partition-compare`. In the new
  condition true depth is used for slice assignment and slice position only;
  same encoder/score shapes receive fixed constants in geometry channels.
  Both arms start from the same initialization, 80 updates/1800s each.
  Preserve old entry/results; no frontend extraction or dev/test scoring.
- ADR-0033 and `docs/pvst/depth_slice_pilot_config.json` supersede the design-only
  execution restriction for one DEPTH_SLICE_RELATION pilot. Implementations live
  in `research_tools/depth_slice`; preserve learned native-pixel slab encoding,
  explicit temporal/depth relations and local pixel bypass. Reuse registered
  ANAKIN train8/dev4 pairs, no new scene filter, no old voxel model or forensic
  checkpoint. One seed, max30/min10/patience8; no extra conditions or tests access.
- ADR-0034 authorizes exactly RGBD_LOCAL/RGBD_SLICE_RELATION on branch
  pvst-rgbd-slice-relations. Use research_tools/rgbd_slice and the pinned original
  DFormer-S RGB-D weights; train the encoder with fixed BN statistics. Preserve
  multiscale256 local decoding/native RGB detail, 4x128 learned slab pooling and
  two temporal/depth relation blocks with zero-initialized residual projections.
  Reuse train8/dev4 and fixed24-frame clips; only their independent MoGe<=768
  cache and essential official dependencies are authorized. Fit check must pass
  before formal max30/min10/patience8 training. Both conditions use depth.
- ADR-0035 authorizes only RGBD_SLICE_RELATION_NORM with one shared affine
  LayerNorm512 after final relations and before both zero-initialized residual
  projections. Keep all ADR-0034 data/model/training settings, old A/B defaults
  and strict checkpoint compatibility. Reuse caches/initialization; no frontend,
  repeated pair fitting or old-condition training. Use the existing train loop
  via research_tools.rgbd_slice.context_norm; record detached scalar context and
  preclip-gradient diagnostics, then compare saved old results and new best/last.
- ADR-0038 and root contract16.6 authorize only the source-isolated ANAKIN
  train8/dev4 RGBD/RGB_DEPTH_PARTITION comparison: existing 48 clips, matched
  initial state and epoch orders, fixed12 epochs/576 updates per arm, four-hour
  independent worker cap. Reuse high-resolution geometry and build/update/evaluate;
  main epoch12 and separate best, no frontend, reserved test or model changes.

## Frozen slice relation readout

- ADR-0039/root16.7 authorize only CURRENT_ONLY/CURRENT_HISTORY on
  pvst-slice-relation-readout: original frozen RGB_DEPTH_PARTITION last C512,
  train8/48 clips and dev4, exact paired initialization/orders, 576 updates each.
  New readout uses current non-self slabs, optionally all same-window past slabs;
  do not fine-tune encoder/pool or use old sets/GRU/head predictions as inputs.
  Preserve all slices and actual PTS; training GT only defines slice BCE targets.
  Slice AP and valid-pixel backfill AP are separate from precise pixel localization.

## CPU localization oracle

- ADR-0040/root16.8 authorize only original dev four fake/462 frames, SLAB versus
  one fixed native SLAB_GRID, CPU GT-driven exact whole-unit overlap limits.
  No model construction, weights, GPU, frontend, training or new data. Preserve
  old inputs/results; this oracle is neither model performance nor grid-model authorization.

## Matched slab/grid localization

- ADR-0041/root16.9 authorize SLAB_CURRENT/SLAB_GRID_CURRENT, original frozen
  RGB-only local/pool weights and train8/dev4. Encode once/frame, preserve all
  valid pixels/units, same 512/8head current-only readout and initialization.
  Fixed native grid only; query-block SDPA preserves all non-self keys and gradients.
  Shared count-weighted pixel BCE, original orders, 576 updates per arm after
  real maximum-clip resource gate. No history, frontend or reserved test access.
  Keep older models, entry defaults and results untouched.

## Compact XYZ temporal pilot

- ADR-0043/root17 authorize research_tools/xyz_pilot on pvst-xyz-temporal-pilot-v1:
  user-fixed train7/dev4, single-scale D128 XYZ cubes, train-only global scale/h,
  frozen generic DFormer-S (never forensic last/common-C initialization), learned
  geometry modulation/pooling and spatial/current-or-past interactions. Native
  all-pixel final evaluation, same576 steps/seed/init; exclude fake40y1x9 frames6–9
  as supervised targets. One real resource gate, then A/B and one B history reversal.
  No shared-key requirement, test access or unapproved extra experiments.

## Joint XYZ window pilot

- ADR-0044/root17.1 authorize only XYZ_JOINT_CURRENT/XYZ_JOINT_TEMPORAL:
  shared generic caches and train7/dev4; C0 before old space, two128/4head
  window layers replace both old relations, same-frame mask versus all-window.
  Keep every valid block, cutoff-t windows, per-block R/pixel bypass and shared
  log-mean-exp evidence readout. New paired initialization and14700 updates;
  train56 monitor and dev824 at prescribed steps. One real update gate then
  train both; no new frontend, forensic initialization, extra seeds or search.
  Prior defaults/results and user changes remain untouched.

## Same-source localization probe

- ADR-0045/root17.2 authorize IMAGE_ONLY/XYZ_CURRENT/XYZ_TEMPORAL on only the
  original seven train sources, shared 70/30 PTS boundary and partition-contained
  histories. Train-only scale, fixed h=.5, cached generic observations; no frontend.
  Pixel BCE+Dice only, common initial weights, ten complete no-replacement epochs,
  checkpoints/evaluation at3/10, fixed train monitor28. No frame-head metrics,
  original dev4 or test access. Preserve all historical protocols/results.

## Frozen temporal readout analysis

- ADR-0046/root17.3 authorize only frozen A_CURRENT/B_FULL/B_CURRENT on original
  insource444 targets, with H-strata, fixed99 probability thresholds and original42
  previews. No training/model changes/frontend. Same B weights/C0 for its two
  paths, no A/B encoding sharing. Preserve native AP, partition history and old
  outputs; threshold bounds are evaluation-label diagnostics, never deployment
  threshold selection or new main results.

## XYZ localization readout adaptation

- ADR-0047/root17.4 authorize original B epoch10 independent full model/AdamW/RNG
  restoration for READOUT_ONLY (four registered decoder modules) and FULL_CONTINUE.
  Both explicitly use XYZ_TEMPORAL history; append original epochs11–13, 3078 updates
  each, same orders/loss/data. Frozen parameters keep grad=None and optimizer state.
  Baseline replay once, monitor28 per epoch, final444 and fixed42 previews. No new
  frontend, threshold fitting, best selection or old-output writes.

## Frozen independent-source localization evaluation

- ADR-0048/root17.5 authorize strict I/A/B localization epoch10 evaluation on
  original dev4 full824 frames, plus B full-continue13338 as a separate appendix.
  Keep original train scale and history modes; no frontend/training/model changes.
  Prepare bounded unreviewed local train/unassigned candidates, pending human
  semantics; never move dev/test into train or use scores to select candidates.

## Full-source coverage comparison

- ADR-0049/root17.6 authorize TRAIN7_FULL versus original7 plus the four specified
  technically aligned sources under current batch permission. Explicit per-ID
  negative mask reviews override it. Full-video train targets, original dev4;
  only missing new-source generic observations may be extracted. Fixed original
  scale/model/loss and fresh shared random initial.pt/XYZ; U=10*Nexp per arm.
  Same seed+cycle shuffle algorithm, distinct orders and exposure records. Check
  one maximum new window then reset; evaluate U/2 and U, no best or extra budget.

## Real DFormer final-stage adaptation

- ADR-0050/root17.7 authorize one DFORMER_STAGE4_TRAINABLE group: actual official
  stage4 RGB/depth downsamplers and two blocks; prefix frozen, all backbone eval.
  Cache raw RGB/depth stage3 state, reuse low3 features/geometry, no output adapter.
  Recompute every current/history final-stage observation with gradients each
  update. Original expanded-order first12845, common random PVST initialization,
  official stage4 and fresh AdamW at1e-5/1e-4; final824 evaluation, fixed24 previews.
  No MoGe, reference retraining, extra dev budgets or changed data/relations/loss.

## Generic-initialized AINPAINT spatial baseline

- ADR-0051/root17.8 authorize one official DINOv2 ViT-L/14 registers + official
  HighResContextHead baseline, only generic initialization, train LoRA4/8/.25
  and head. Original expanded-order first12845, batch1/T1, full-work-frame pad14
  then crop logits before native recovery. No XYZ/history/RAFT/forensic checkpoint.
  Keep raw DINO frozen; train-only BN100 per epoch counted separately; final824
  and fixed24 previews. No best, parameter search, extra conditions or old writes.

## Fixed-weight spatial BN diagnosis

- ADR-0052/root17.9 authorize only original spatial final12845, full dev824,
  RUNNING_BN versus five-head-BN functional IMAGE_BN. Entire model eval/no_grad,
  dropout off; preserve affine/eps and all parameters/buffers/counts. No optimizer,
  training/calibration, train GT, XYZ/history or normalization combination search.
  Reuse original metrics/24 previews, retain baseline and same-input replay evidence.

## Native local-detail matched input probe

- ADR-0053/root17.10 authorize only NATIVE_PATCH/LOWDETAIL_PATCH on original11/4,
  original-order12845 updates each, shared native288 crop/GT plan and new generic
  LoRA/head initialization. LOW degrades whole frame via old384 AREA then native
  bilinear before same crop. Native padding masked from loss; model pad14 cropped.
  Keep running-BN protocol500 extra forwards, no XYZ/history or new frontend.
  Final824 blind288/144 tiles, tail aligned, uniform FP32-logit fusion, all pixels.
  No third arm/search/best; old full-frame results remain historical references.

## Expanded full-frame spatial training

- ADR-0054/root17.11 authorize one expanded spatial condition, technical PASS
  source-group train/dev draft plus original11/4; explicit human mask negatives
  and all protected test identities remain excluded. Individual unknown human
  semantics/camera do not block this user-authorized trial.
- Reuse official generic DINO/LoRA/head and whole-frame384 path; ten complete
  epochs, train-only BN100 each, dev/monitor at1/3/5/10, epoch10 primary and
  additional-dev-only best supplement. No XYZ/history/frontends or new branch.

## Full-video direct prototype

- ADR-0055/root17.12 authorize the independent fullvideo_direct_model/probe only:
  frozen adapted epoch01 DINO/LoRA/head observations, all XYZ blocks, two direct
  bidirectional SDPA layers, all-frame zero-initialized localization correction.
  No relay, token pruning, causal cutoff or new MoGe. Codex runs only the complete
  low-token case, two discardable updates per mode; user launches medium/high
  terminal jobs. Preserve old entries/results; no formal training or polling.

## Full-video matched localization handoff

- ADR-0056/root17.13 authorize the frozen89/19 two-arm three-epoch terminal
  pipeline, shared adapted-DINO/own-MoGe observations, all-frame supervision,
  exact OOM-only Q1024 fallback, per-video atomic resume and original metrics.
  Codex only runs metadata/CPU checks and one bounded cached-video interface
  smoke. User launches full prepare, queue extremes and training; no polling.


## Official MumPy ANAKIN transfer

- ADR-0057/root17.14 authorize one official full MumPy RGB baseline, frozen89/19
  source split and official generic initializer. Keep all three views/DWTI/DCT/
  multi-pyramid decoder; never substitute the spatial baseline or PVST relations.
- Three centered RGB frames with explicit boundary clamping; exclude40y1x9 fake6–9
  only as targets. Ten full target epochs, official SGD/polynomial/focal+IoU,
  effective batch12 (Davis collate actual clips) and native source-macro metrics; best additional-dev and last.
- Codex runs bounded discardable smoke only. User starts full terminal/tmux job;
  no long-task monitoring, new data/frontends, test access or old-result writes.
