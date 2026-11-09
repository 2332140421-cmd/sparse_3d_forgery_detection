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
