# 3D frontend replacement feasibility — Phase A

## Scope and evidence labels

Source V6 HEAD: `1accc2772323306b6939886842e6896113676242`.

This is a source- and asset-level feasibility audit. No candidate runtime was
run because neither candidate had the required local official source,
checkpoint, and isolated environment. Statements below use these labels:

- **OFFICIAL SOURCE FACT**: directly supported by the linked official repository.
- **CURRENT V6 FACT**: recorded by the current repository or environment.
- **EXPERIMENT RESULT**: measured in an already accepted V6 experiment.
- **DESIGN INFERENCE**: consequence for the fixed V6 frontend requirements; it
  is not an upstream guarantee.

The fixed requirements are: RGB-to-per-frame geometry; same tracked UV to XYZ;
history independent of images after the cutoff; target at `t+h` dependent on at
most `I_<=t+h`; an explicit common coordinate relation; no future-point
alignment; dynamic-video applicability; explicit invalidity; conversion to
`ParticleSequence`; and preservation of the xyz-only continuous model input.

## Current environment and local assets

**CURRENT V6 FACT.** The project environment is Python 3.12.3, PyTorch
2.3.1+cu121, torchvision 0.18.1+cu121, and NumPy 1.26.1. The GPU is an NVIDIA
GeForce RTX 4090 D with 24,564 MiB, driver 595.71.05.

A bounded search of `/root/autodl-tmp/projects`, `/root/autodl-tmp/models`,
`/root/autodl-tmp/checkpoints`, and `/root/.cache/huggingface` found no CUT3R or
VGGT-Ω official source tree, checkpoint, or isolated environment. The latter
three candidate-asset locations did not exist. The project `.venv` was not
modified. A shallow source clone attempt into a temporary directory failed with
`gnutls_handshake() failed: The TLS connection was non-properly terminated`;
official GitHub source pages were therefore inspected read-only. No TLS or Git
configuration was changed.

## Current VGGT baseline

**CURRENT V6 FACT.** The frontend pins official VGGT source revision
`a288dd0f14786c93483e45524328726ab7b1b4ce` and `facebook/VGGT-1B` weight
revision `860abec7937da0a4c03c41d3c269c366e82abdf9` (ADR 0008). The accepted
construction uses history-anchored causal windows and a global proper Sim(3).

**EXPERIMENT RESULT.** Eight fixed B1 windows produced exactly zero repeat-run
target disagreement. On the 160-window real primary population, median
`Q_target` was 1.6056, 0.9326, and 0.4278 for h=1, h=2, and h=4. T0
prefix-target-error/frontend-residual Pearson correlation was 0.6967, 0.7833,
0.7355, and 0.7883 for h=1/2/4/8. These measurements did not use fake AUROC to
choose a provider.

**CURRENT V6 FACT.** Its key limitation is joint-sequence geometry: different
prefixes can redefine past geometry and gauge, requiring history-only Sim(3)
alignment. Short-horizon context disagreement is material relative to observed
motion. The current disposition remains `INCONCLUSIVE`.

## CUT3R source audit

Official repository: [`MultiPath/ProgressiveDust3R`](https://github.com/MultiPath/ProgressiveDust3R),
default branch `main`, inspected commit
[`a5433c13848e56fd5acd36a741cccb7d76614c76`](https://github.com/MultiPath/ProgressiveDust3R/commit/a5433c13848e56fd5acd36a741cccb7d76614c76).
GitHub identifies this repository as a fork of `CUT3R/CUT3R`.

### Distribution and environment

- **OFFICIAL SOURCE FACT.** The repository root has no project-level license
  file; a CroCo license exists only inside the vendored dependency. Therefore
  CUT3R project-license compatibility is **not confirmed**.
- **OFFICIAL SOURCE FACT.** The README specifies Python 3.11, CMake 3.14,
  PyTorch/torchvision with CUDA 12.1, and the
  [requirements file](https://github.com/MultiPath/ProgressiveDust3R/blob/a5433c13848e56fd5acd36a741cccb7d76614c76/requirements.txt)
  pins NumPy 1.26.4 and Pillow 10.3.0 and includes scipy, opencv-python,
  transformers, accelerate, h5py, and vendored CroCo/DUSt3R code. An optional
  CUDA RoPE extension is documented; xformers, flash-attn, and pytorch3d are not
  declared. `gsplat` is described for training, not core inference.
- **OFFICIAL SOURCE FACT.** The README names Google Drive checkpoints
  `cut3r_224_linear_4.pth` and `cut3r_512_dpt_4_64.pth`. Their exact sizes and
  checkpoint-specific license/access terms are not stated on the inspected
  official pages, so both are **not confirmed** rather than estimated.
- **DESIGN INFERENCE.** Use a separate Python 3.11 environment. Do not alter the
  V6 `.venv`, even though its CUDA/PyTorch generation is broadly close.

### Information flow and V6 mapping

- **OFFICIAL SOURCE FACT.** In
  [`model.py`](https://github.com/MultiPath/ProgressiveDust3R/blob/a5433c13848e56fd5acd36a741cccb7d76614c76/src/dust3r/model.py),
  `_init_state` initializes persistent state, `_recurrent_rollout` consumes the
  prior state and current-frame features, and `_forward_impl` appends each
  frame's result before advancing state and memory. `inference_step` exposes the
  stateful single-step path. Previously appended outputs are not rewritten by
  that loop.
- **DESIGN INFERENCE.** If the adapter stores the result emitted at each step
  and never reruns or replaces it, the stored history satisfies
  `X_t=f(I_<=t)` by construction. Prefix-history re-alignment is then not
  required for those stored outputs; runtime must still verify this exact API
  usage and determinism.
- **OFFICIAL SOURCE FACT.** The prediction head exposes
  `pts3d_in_self_view`, `pts3d_in_other_view`, confidence, optional camera pose,
  and depth-related output. The recurrent inference API returns persistent
  state and per-frame results.
- **DESIGN INFERENCE.** `pts3d_in_other_view` is the promising common-state
  representation, but the inspected source alone does not prove a stable
  metric/world gauge across a long dynamic sequence. Coordinate convention,
  scale drift, and whether different timesteps are directly subtractable are
  mandatory runtime checks.
- **OFFICIAL SOURCE FACT.** The input/training surface includes dynamic-video
  datasets and contains no static-scene-only gate. It does not provide an
  official guarantee of dynamic-object reconstruction accuracy.
- **DESIGN INFERENCE.** Dense pointmaps and confidence permit sampling the
  existing V6 same-track UV observations, after explicit provider-to-original
  UV mapping. Invalid/nonfinite or low-confidence samples can remain NaN with
  explicit masks, so conversion to `ParticleSequence` is source-feasible without
  exposing provider-private tensors or changing the xyz-only model boundary.

Classification: **SOURCE_FEASIBLE_RUNTIME_BLOCKED**. Runtime is
`NOT_RUN_ASSET_BLOCKED`; local source, checkpoint, and isolated environment are
absent, and project-level license terms also require confirmation.

## VGGT-Ω source audit

Official repository: [`facebookresearch/vggt-omega`](https://github.com/facebookresearch/vggt-omega),
default branch `main`, inspected commit
[`282ec70363edeff59424bf43731658092fba3d37`](https://github.com/facebookresearch/vggt-omega/commit/282ec70363edeff59424bf43731658092fba3d37).

### Distribution and environment

- **OFFICIAL SOURCE FACT.** The
  [FAIR Noncommercial Research License](https://github.com/facebookresearch/vggt-omega/blob/282ec70363edeff59424bf43731658092fba3d37/LICENSE)
  permits noncommercial research subject to its restrictions and attribution
  terms. Compatibility here is conditional on the thesis use remaining within
  those terms.
- **OFFICIAL SOURCE FACT.** `pyproject.toml` requires Python >=3.10. Core
  requirements include torch >=2.3, torchvision >=0.18, NumPy <2, Pillow,
  einops, safetensors, and opencv-python. No xformers, flash-attn, pytorch3d, or
  custom CUDA extension is declared.
- **OFFICIAL SOURCE FACT.** Official Hugging Face checkpoints are
  `VGGT-Omega-1B-512` and `VGGT-Omega-1B-256-Text-Alignment`, both requiring
  access approval. Exact checkpoint byte sizes were not stated on the inspected
  official pages and are **not confirmed**.
- **DESIGN INFERENCE.** Core version constraints overlap the V6 environment,
  but an isolated environment is still required to protect the accepted V6
  stack and accommodate missing packages/checkpoint access.

### Information flow and V6 mapping

- **OFFICIAL SOURCE FACT.** `VGGTOmega.forward(images)` consumes all frames at
  once. Its aggregator alternates per-frame and inter-frame attention; the
  global inter-frame path attends over tokens from all supplied frames. The
  inspected model exposes no causal mask, streaming entrypoint, persistent
  state, or KV-cache interface.
- **DESIGN INFERENCE.** A frame's representation can depend on later supplied
  frames. Different prefix runs may therefore redefine past geometry. V6 would
  still require target-prefix inference plus history-only proper Sim(3)
  alignment; future target points must remain outside the fit. VGGT-Ω does not
  resolve this causality issue by source semantics alone.
- **OFFICIAL SOURCE FACT.** The model returns pose encodings, depth and depth
  confidence, camera/register tokens, and optional text embeddings; it does not
  return tracks. Pose utilities interpret extrinsics as camera-from-world
  (`world_to_camera`) in OpenCV coordinates and invert them for camera-to-world
  use.
- **DESIGN INFERENCE.** Joint inference provides a common sequence gauge inside
  one run, but separately inferred prefixes may have different gauges. Current
  independent V6 UV tracks can be retained and sampled from dense depth/world
  geometry after explicit resize/crop mapping. This supports a
  `ParticleSequence` adapter with NaN/mask semantics and no change to the
  xyz-only boundary.
- **OFFICIAL SOURCE FACT.** No static-scene rejection is present, but the model
  has no tracking head and the inspected source provides no explicit guarantee
  for dynamic-object geometry.
- **DESIGN INFERENCE.** Dynamic-video input is technically accepted, while
  dynamic same-particle quality remains a runtime question.

Classification: **SOURCE_FEASIBLE_RUNTIME_BLOCKED**. Runtime is
`NOT_RUN_ASSET_BLOCKED`; local source, approved checkpoint, and isolated
environment are absent.

## Dependency compatibility

| Item | Current V6 | CUT3R requires | VGGT-Ω requires | Assessment |
| --- | --- | --- | --- | --- |
| Python | 3.12.3 | README: 3.11 | >=3.10 | CUT3R differs; isolate both |
| torch | 2.3.1+cu121 | torch, CUDA 12.1 install path | >=2.3 | version-level overlap |
| torchvision | 0.18.1+cu121 | torchvision | >=0.18 | version-level overlap |
| numpy | 1.26.1 | ==1.26.4 | <2 | CUT3R pin differs |
| xformers | not required by V6 audit | not declared | not declared | no declared conflict |
| flash-attn | not required by V6 audit | not declared | not declared | no declared conflict |
| pytorch3d | not required by V6 audit | not declared | not declared | no declared conflict |
| CroCo / DUSt3R | not a V6 dependency | vendored/required | not declared | CUT3R-specific stack |
| scipy | present status not relied upon | required | demo/utility use | candidate-specific |
| opencv | present status not relied upon | required | required | candidate-specific |
| custom CUDA | none for this audit | optional CUDA RoPE | none declared | CUT3R build risk |

## Unified comparison matrix

`NOT_RUN` means no number was fabricated in the absence of authorized assets.

| Criterion | Current VGGT | CUT3R | VGGT-Ω |
| --- | --- | --- | --- |
| official implementation inspected | yes, existing pin | yes | yes |
| exact source SHA | `a288dd0...b1b4ce` | `a5433c...614c76` | `282ec7...a3d37` |
| license compatible | existing accepted baseline | not confirmed: no root license | conditional noncommercial research |
| local checkpoint available | yes | no | no |
| current env compatible | yes | not accepted as-is | constraints overlap, unverified |
| isolated env required | no | yes | yes |
| dynamic-scene support | accepts video; measured limitations remain | accepted by source surface; quality unverified | accepts multi-frame input; quality unverified |
| online/causal history semantics | no; prefix construction required | recurrent stored-output path | no explicit causal/streaming mode |
| persistent common coordinate | joint-run gauge; prefix gauge changes | promising state-frame output; runtime unverified | joint-run gauge; prefix gauge changes possible |
| needs history-prefix re-alignment | yes | no by stored-output construction; verify runtime | yes |
| same-UV XYZ sampling feasible | yes | source-feasible | source-feasible with external tracker |
| ParticleSequence mapping feasible | implemented | source-feasible | source-feasible |
| repeatability | exact zero disagreement on 8 B1 windows | NOT_RUN | NOT_RUN |
| prefix stability | context-sensitive; history-only Sim(3) used | NOT_RUN | NOT_RUN |
| Q_target | medians h1/h2/h4: 1.6056/.9326/.4278 | NOT_RUN | NOT_RUN |
| valid coverage | existing accepted artifacts; unified 8-window value not rerun | NOT_RUN | NOT_RUN |
| GPU memory | existing frontend known runnable on 24 GiB; no new measurement | NOT_RUN | NOT_RUN |
| runtime | existing accepted experiments | NOT_RUN_ASSET_BLOCKED | NOT_RUN_ASSET_BLOCKED |

## Recommendation and next authorization

Recommendation: **AUTHORIZE_CUT3R_RUNTIME_EVAL_NEXT**.

**DESIGN INFERENCE.** CUT3R is the only candidate whose inspected recurrent API
offers a materially different, naturally history-frozen information flow. That
directly targets V6's current short-horizon context/gauge limitation. VGGT-Ω is
source-feasible and may improve geometry, but its joint global attention retains
the same causal-prefix class of problem, so it is lower priority rather than
rejected. Neither candidate is selected as the formal paper frontend.

A subsequent, separately authorized CUT3R runtime phase would need:

1. a small source download pinned to
   `a5433c13848e56fd5acd36a741cccb7d76614c76`;
2. one official inference checkpoint download (prefer the documented final
   `cut3r_512_dpt_4_64.pth`), after confirming its exact size, hash, access terms,
   and project-level license; estimated size is **unknown from official evidence**;
3. a new isolated Python 3.11 environment with the official inference
   dependencies, without changing the project `.venv`; and
4. the fixed eight-real-window, same-UV protocol measuring repeatability,
   stored-output history invariance, target context sensitivity, valid coverage,
   time, and peak memory.

VGGT-Ω evaluation would separately require the pinned source, approval for the
gated official checkpoint, confirmation of its exact size/hash/license terms,
and an isolated environment. It should be authorized only if a second
joint-feed-forward geometry baseline is desired after CUT3R.

No detector was trained; no fake AUROC was used; the design contract and ADRs
were not changed; no provider was promoted to the formal paper frontend; no
candidate checkpoint, dataset, or demo asset was downloaded; and the current
project environment was not modified.
