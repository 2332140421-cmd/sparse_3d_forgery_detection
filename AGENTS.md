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
