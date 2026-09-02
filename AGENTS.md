# Repository Instructions

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
