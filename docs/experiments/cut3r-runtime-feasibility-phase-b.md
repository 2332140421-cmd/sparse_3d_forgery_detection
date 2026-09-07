# CUT3R runtime feasibility — Phase B

## Scope and outcome

Source V6 HEAD: `7d93be5b7364dee80628a5ef308f17e63ce92c33`.

This phase asked whether CUT3R's persistent-state geometry is a materially more
stable causal measurement frontend than the current VGGT construction. The
mandatory license gate did not permit acquisition or execution, so this is a
blocked audit rather than a runtime result.

Evidence labels in this document are:

- **OFFICIAL SOURCE FACT**: stated by the official CUT3R distribution.
- **CURRENT V6 FACT**: recorded by the current repository or environment.
- **RUNTIME RESULT**: produced by this phase's runtime protocol.
- **DESIGN INFERENCE**: consequence for V6, not an upstream guarantee.

Final classification: **CUT3R_RUNTIME_BLOCKED_BY_LICENSE**.

Decision: **MOVE_TO_EXPLICIT_GEOMETRY_FRONTEND_AUDIT**.

## Git and fixed candidate

**CURRENT V6 FACT.** Preflight passed on `main` at
`7d93be5b7364dee80628a5ef308f17e63ce92c33`, with a clean workspace and
`HEAD...origin/main = 0/0`.

The fixed candidate remains the official CUT3R initial revision:

- canonical repository: [`CUT3R/CUT3R`](https://github.com/CUT3R/CUT3R);
- Phase-A repository identifier: `MultiPath/ProgressiveDust3R`, a GitHub fork
  of the canonical repository;
- fixed full SHA:
  [`a5433c13848e56fd5acd36a741cccb7d76614c76`](https://github.com/CUT3R/CUT3R/commit/a5433c13848e56fd5acd36a741cccb7d76614c76);
- official checkpoint preference: `cut3r_224_linear_4.pth`.

**OFFICIAL SOURCE FACT.** The fixed SHA is present in the canonical official
repository as its initial commit. It is not silently replaced by a later
revision in this audit.

## License gate

Classification: **LICENSE_UNRESOLVED**.

### Evidence

1. **OFFICIAL SOURCE FACT.** The tree at fixed commit
   `a5433c13848e56fd5acd36a741cccb7d76614c76` has no root project-level
   `LICENSE`. Its vendored `src/croco/LICENSE` covers CroCo and is not treated as
   a CUT3R license.
2. **OFFICIAL SOURCE FACT.** The current canonical `CUT3R/CUT3R` repository has
   subsequently added a six-line root
   [`LICENSE`](https://github.com/CUT3R/CUT3R/blob/main/LICENSE) stating that “CUT3R is licensed
   under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
   License.” This is clear evidence for noncommercial use of the currently
   distributed CUT3R project, subject to attribution and share-alike terms.
3. **OFFICIAL SOURCE FACT.** The current
   [README](https://github.com/CUT3R/CUT3R/blob/main/README.md) distributes
   `cut3r_224_linear_4.pth` and `cut3r_512_dpt_4_64.pth` through Google Drive and
   calls the former an intermediate checkpoint and the latter the final
   checkpoint. The README and Drive pages inspected here do not separately
   state that the repository license covers those checkpoint files.
4. **OFFICIAL SOURCE FACT.** The [project page](https://cut3r.github.io/) and
   [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.pdf)
   describe online,
   common-coordinate, metric-scale pointmaps and both static and dynamic
   content. They do not provide code or checkpoint usage terms.
5. No release-specific license or official model card was found that explicitly
   binds the later root license to both the fixed historical source revision and
   the externally hosted checkpoint.

### Gate interpretation

**DESIGN INFERENCE.** The later blanket project license is encouraging, but it
is not enough to infer checkpoint rights or retroactively rewrite the contents
of the fixed commit. Public downloadability, publication, and the CroCo license
are not substitutes for explicit CUT3R checkpoint terms. Because runtime
necessarily loads an official checkpoint, source-only permission cannot clear
the complete runtime gate.

Accordingly:

- fixed-revision source use: **not established with sufficient scope for this
  gated acquisition**;
- checkpoint use: **unresolved**;
- academic runtime acquisition: **not authorized by the available evidence**.

The gate can be cleared by an official statement from the CUT3R maintainers
that the CC BY-NC-SA 4.0 terms cover the fixed code revision and the named
official checkpoint, or by separate explicit terms covering both.

## Assets and environment

No source, checkpoint, or environment was created.

| Field | Value |
| --- | --- |
| source SHA | `a5433c13848e56fd5acd36a741cccb7d76614c76` |
| checkpoint filename | planned `cut3r_224_linear_4.pth`, not acquired |
| checkpoint exact bytes | `NOT_AVAILABLE_LICENSE_BLOCKED` |
| checkpoint SHA-256 | `NOT_AVAILABLE_LICENSE_BLOCKED` |
| checkpoint authentication | official Drive link appeared public; this does not establish usage rights |
| isolated environment | `NOT_CREATED_LICENSE_BLOCKED` |
| Python / torch / torchvision / CUDA | `NOT_INSTALLED_LICENSE_BLOCKED` |
| pip check | `NOT_RUN_NO_ENVIRONMENT` |

The current V6 `.venv`, system Python, CUDA installation, and Git configuration
were not changed.

## Runtime and geometry results

There are no **RUNTIME RESULT** values. The gate was evaluated before source
clone, checkpoint download, dependency installation, sample materialization,
or GPU execution as required.

| Required result | Status |
| --- | --- |
| fixed 8-real-video manifest | `NOT_MATERIALIZED_LICENSE_BLOCKED`; intended source is the accepted per-horizon audit's stable real-primary center-window selection |
| successful / failed windows | `0 / 0`; no attempt |
| peak GPU memory | `NOT_RUN` |
| runtime | `NOT_RUN` |
| selected XYZ representation | `NOT_SELECTED_BY_RUNTIME`; source candidate is `pts3d_in_other_view` |
| coordinate semantics | source claims online common-coordinate metric-scale pointmaps; not runtime-verified for this revision/checkpoint |
| handedness / axes | `NOT_RUNTIME_CONFIRMED` |
| length unit / scale | official project claim is metric scale; fixed runtime behavior not verified |
| same-input repeatability | `NOT_RUN` |
| history future-invariance | source-predicted by stored emission semantics, but `NOT_RUN` |
| retrospective revision | `NOT_APPLICABLE_NO_RUNTIME` |
| gauge or scale drift | `NOT_MEASURED` |
| same-UV coverage | `NOT_MEASURED` |
| per-horizon motion scale | `NOT_MEASURED` |

No `Q_target` or replacement `Q` value is manufactured. No fake sample,
annotation, detector score, or AUROC was accessed or computed.

## Comparison with current VGGT

**CURRENT V6 FACT.** VGGT requires separate prefix runs plus a history-only
proper Sim(3). Its fixed B1 repeatability sample had exactly zero repeat-run
target disagreement. Real-primary median `Q_target` was 1.6056, 0.9326, and
0.4278 for h=1/2/4. Short-horizon context change therefore remains material.

**DESIGN INFERENCE.** CUT3R's recurrent emitted-output path addresses this
mechanism at source level because an emitted history value need not be
recomputed when future frames arrive. This phase cannot establish that the
checkpoint is repeatable, that `pts3d_in_other_view` retains a directly
subtractable coordinate gauge, that scale does not drift, that same-UV coverage
is adequate, or that it fits the 4090 D. Therefore CUT3R has not demonstrated a
runtime advantage and cannot replace VGGT.

| Criterion | Current VGGT | CUT3R Phase B |
| --- | --- | --- |
| causal history | accepted prefix + Sim(3) construction | promising source semantics; runtime blocked |
| coordinate stability | measured prefix/context limitation | not measured |
| same-UV coverage | existing artifacts; exact unified subset not rerun | not measured |
| resource use | known runnable on current 4090 D | not measured |
| directly resolves context instability | no | source-level hypothesis only |

## Explicit Geometry Frontend Fallback

The next candidate is specified, not implemented, as:

```text
Video
  -> causal or fixed 2D tracking
  -> per-frame depth
  -> camera intrinsics
  -> causal camera pose
  -> explicit back-projection
  -> camera-motion-compensated XYZ
  -> ParticleSequence
```

The next task should be an **Explicit Geometry Frontend Feasibility Audit**
designed from the current V6 measurement question. It should separately test
depth stability, pose drift, back-projection correctness, causal information
flow, common-coordinate displacement validity, same-UV coverage, and
reprojection only as a frontend diagnostic.

Explicit geometry may measure XYZ; it must not define fake anomaly. It must not
restore handcrafted residuals, velocity, acceleration, curvature, occlusion
anomaly rules, or any prior implementation. Provider-private values must still
be converted into validated `ParticleSequence`, with invalid values represented
by NaN plus explicit masks and with xyz as the only continuous model input.

This fallback specification does not select depth, tracking, or pose providers,
does not alter the formal V6 design, and is not an implementation authorization.

## Boundaries

No detector was trained; no fake data was used; no AUROC was computed; the
formal design contract and ADRs were not changed; CUT3R was not declared the
formal frontend; no old explicit anomaly rule was introduced; no source,
checkpoint, dependency, environment, or dataset was downloaded or installed;
and no GPU inference was run.
