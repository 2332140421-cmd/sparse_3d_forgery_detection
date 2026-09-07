# V7 Phase 1A: Explicit geometry frontend candidate audit

## 1. Objective

This static audit asks which explicit provider combination is suitable for a
bounded V7 runtime falsification experiment. It does not select the paper
frontend and it does not claim that any candidate produces sufficiently stable
geometry.

The required information flow is:

```text
video -> causal/fixed 2D tracks -> single-frame depth and intrinsics
      -> causal incremental camera pose -> explicit back-projection
      -> camera-motion-compensated XYZ -> ParticleSequence
```

**CURRENT V7 FACT.** The audit started on `v7-dynamic-structure` at
`55f609cb08c9c539af2599339c37e00e48bc2105`, with a clean worktree and
`HEAD...origin/v7-dynamic-structure = 0/0`. The existing environment is Python
3.12.3, torch 2.3.1+cu121, torchvision 0.18.1+cu121, and NumPy 1.26.1.

**CURRENT V7 FACT.** V6's real-primary VGGT `Q_target` medians are 1.6056,
0.9326, and 0.4278 at h=1/2/4. They motivate a causal explicit measurement
path; they are not provider-selection scores.

## 2. V7 requirements

The runtime candidate must preserve a track identity across frames, emit no
history that is silently revised using future images, map provider coordinates
back to original decoded UV, and yield XYZ in one recorded gauge. Missing
observations remain NaN with separate visibility and geometry-validity masks.
Appearance may be used privately by the tracker but cannot enter downstream
continuous features. Geometry, reprojection, occlusion, confidence, or provider
failure cannot become an authenticity rule.

For pixel `(u,v)` and optical-axis depth `z`, the intended explicit operation is

```text
p_c,t = z_t K_t^-1 [u, v, 1]^T
p_0,t = T_0<-t p_c,t
```

where every transform direction and inversion is tested rather than inferred
from a variable name. If a provider outputs ray distance instead of z-depth,
an explicit documented ray-to-z conversion is mandatory before this equation.

## 3. Candidate search methodology

Candidates were limited to official repositories, documentation, papers, model
cards, and license files. The search covered mature, arbitrary-video point
trackers, single-frame metric/camera-aware depth, and sequential VO/RGB-D
odometry. Whole-video VGGT and CUT3R were not reconsidered: current V7 records
already establish their role as historical baselines.

Evidence labels used below are:

- **OFFICIAL SOURCE FACT**: explicitly stated or exposed by an official source.
- **CURRENT V7 FACT**: read from this branch or its accepted experiment record.
- **STATIC COMPATIBILITY INFERENCE**: a consequence to test, not an upstream guarantee.
- **UNRESOLVED**: the inspected official evidence does not settle the matter.

No repository, checkpoint, package, or model was downloaded. GitHub HTTPS
revision queries were unavailable through the server's existing GnuTLS path;
therefore mutable `main` is never treated as a runtime pin. Phase 1B must resolve
a full source SHA before acquiring each selected provider.

## 4. Tracking candidates

### Online BootsTAPIR — primary tracking candidate

- **OFFICIAL SOURCE FACT.** The official
  [TAPNet repository](https://github.com/google-deepmind/tapnet) describes
  Online TAPIR/BootsTAPIR as sequential causal point tracking and links both
  JAX and PyTorch Online BootsTAPIR checkpoints. It predicts point tracks,
  occlusion, and expected distance for arbitrary queried image points.
- **OFFICIAL SOURCE FACT.** The repository states that all linked pretrained
  checkpoints, including BootsTAPIR, are Apache-2.0 along with the relevant
  software. No authentication requirement is stated for the linked checkpoint.
- **STATIC COMPATIBILITY INFERENCE.** A point query creates a persistent track;
  occlusion becomes `visibility=False`, and the V7 adapter must retain NaN rather
  than interpolating occluded UV. Birth is an explicit later query; death is the
  final observed state. Appearance remains tracker-private.
- **STATIC COMPATIBILITY INFERENCE.** The provider operates at 256/512
  resolutions. Its exact pixel convention and inverse resize mapping must be
  tested against original decoded UV. The emitted-state API must also be tested
  for bitwise/numeric repeatability and absence of retrospective replacement.
- Source pin: **UNRESOLVED** full SHA; mandatory before Phase 1B acquisition.
  The exact runtime checkpoint is the official PyTorch
  `causal_bootstapir_checkpoint.pt`; its bytes and SHA-256 are not established
  by this static audit.

### CoTracker3 online — secondary tracking candidate

- **OFFICIAL SOURCE FACT.** The official
  [CoTracker repository](https://github.com/facebookresearch/co-tracker) tracks
  arbitrary pixels and returns tracks plus visibility. Its online API is a
  sliding-window/multi-window implementation, not an offline single-window run.
- **STATIC COMPATIBILITY INFERENCE.** Because the example supplies overlapping
  chunks, “online” does not by itself prove zero-lookahead output or immutable
  per-frame history. Phase 1B would need an explicit output-finalization delay
  and a prefix-extension test. Backward smoothing is not permitted.
- **OFFICIAL SOURCE FACT.** Source is CC-BY-NC-4.0; the official
  [CoTracker3 model card](https://huggingface.co/facebook/cotracker3) identifies
  the checkpoint license as CC-BY-NC-4.0. It is usable for the current
  noncommercial academic study but commercially restricted, with attribution
  obligations. Public downloads do not require a stated approval gate.
- Source pin: **UNRESOLVED** full SHA; mandatory before Phase 1B acquisition.
  The exact online checkpoint variant (baseline or scaled), bytes, and SHA-256
  remain intentionally unselected until the primary path is falsified or a
  secondary run is authorized.

### TAPNext / TAPNext++

- **OFFICIAL SOURCE FACT.** The TAPNet repository distributes TAPNext/TAPNext++
  implementations and checkpoints under Apache-2.0.
- **UNRESOLVED.** The inspected material did not establish a simpler,
  history-immutable online contract than Online BootsTAPIR for this experiment.
  It is retained as a research candidate, not a Phase 1B combination.

## 5. Depth and intrinsics candidates

### Apple Depth Pro — primary depth/intrinsics candidate

- **OFFICIAL SOURCE FACT.** The official
  [Depth Pro repository](https://github.com/apple/ml-depth-pro) describes
  single-image zero-shot metric depth with absolute scale and no required input
  camera metadata. `infer` returns `depth` in metres and `focallength_px`; its
  source resizes internally and restores depth to input image resolution.
- **OFFICIAL SOURCE FACT.** Its README expressly releases both sample code and
  model weights under the repository
  [LICENSE](https://github.com/apple/ml-depth-pro/blob/main/LICENSE). The grant
  permits use, reproduction, modification, and redistribution subject to its
  notice and disclaimer terms; no academic-only, noncommercial, authentication,
  or separate checkpoint restriction is stated.
- **STATIC COMPATIBILITY INFERENCE.** Single-image inference is causal and cannot
  use future RGB. It does not guarantee temporal consistency, dynamic-object
  accuracy, or calibrated uncertainty. Those are runtime measurements.
- **UNRESOLVED.** The API returns focal length, not a complete calibrated K.
  Phase 1B must record the centered-principal-point/no-distortion assumption (or
  stop if source evidence contradicts it), establish whether output samples are
  optical-axis z-depth or ray distance, and verify resize/pixel-center mapping.
- Source pin: **UNRESOLVED** full SHA; checkpoint filename and SHA-256 must be
  fixed before runtime. Official setup recommends Python 3.9, so isolation is
  required. `VRAM_NOT_OFFICIALLY_SPECIFIED`.

### UniDepth V2

- **OFFICIAL SOURCE FACT.** The official
  [UniDepth repository](https://github.com/lpiccinelli-eth/unidepth) exposes
  per-image metric depth, camera-coordinate points, intrinsics, and V2
  confidence. This is the cleanest complete-K interface in the depth set.
- **OFFICIAL SOURCE FACT.** Repository software is CC-BY-NC-4.0.
- **UNRESOLVED.** The inspected official model pages did not separately bind the
  named V2 checkpoint weights to explicit terms. Therefore it is
  `LICENSE_UNRESOLVED` and cannot be a runtime primary despite its attractive
  geometry interface.

### Depth Anything V2 metric

- **OFFICIAL SOURCE FACT.** The official
  [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)
  provides single-image metric variants. Small is Apache-2.0; Base/Large/Giant
  are CC-BY-NC-4.0 according to the official README.
- **STATIC COMPATIBILITY INFERENCE.** It does not supply a complete camera K;
  indoor/outdoor metric heads and maximum-depth settings add domain choices.
  It is causal but does not close the intrinsics path and has no official
  temporal-consistency guarantee. It is not shortlisted.

### Metric3D v2

- **OFFICIAL SOURCE FACT.** The official
  [Metric3D repository](https://github.com/YvanYin/Metric3D) publishes metric
  depth and confidence code under BSD-2-Clause and documents the importance of
  focal length/canonical-camera conversion.
- **UNRESOLVED.** Separate terms for the required hosted weights were not found
  in the inspected official evidence. Unknown internet-video focal length can
  distort lifted geometry. It is not shortlisted.

## 6. Pose candidates

### Open3D 0.19 sequential RGB-D odometry — primary pose candidate

- **OFFICIAL SOURCE FACT.** Open3D 0.19 documents pairwise
  [RGB-D odometry](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html)
  from source and target RGB-D frames with a camera intrinsic matrix. The
  returned transformation maps source to target in its documented example.
- **OFFICIAL SOURCE FACT.** Open3D is [MIT licensed](https://github.com/isl-org/Open3D/blob/main/LICENSE),
  has no model checkpoint, and 0.19 provides Python wheels. Release tag
  `v0.19.0` is the candidate version; its full tag commit must be recorded before
  installation.
- **STATIC COMPATIBILITY INFERENCE.** Calling only adjacent `(t-1,t)` pairs and
  composing stored increments is causal. No pose-graph optimization, loop
  closure, global BA, ICP against future frames, or retrospective rewrite is
  allowed. Feeding Depth Pro's metre-scale depth to both back-projection and
  RGB-D odometry closes the translation-scale loop, subject to the depth and K
  semantic checks above.
- **STATIC COMPATIBILITY INFERENCE.** It is a rigid/dominant-static-scene
  estimator. A large moving foreground can corrupt pose. Failure must invalidate
  camera compensation, never silently substitute identity.

### DPVO

- **OFFICIAL SOURCE FACT.** The official
  [DPVO repository](https://github.com/princeton-vl/DPVO) implements learned
  monocular visual odometry with CUDA-dependent components and distributed
  checkpoints.
- **STATIC COMPATIBILITY INFERENCE.** Monocular translation is only determined
  up to scale. Pairing it directly with metre-scale depth does not yield metre
  world XYZ. An independently justified causal scale bridge would be required;
  per-window arbitrary Sim(3) is disallowed. `COMMON_COORDINATE_SCALE_BLOCKER`.
- **UNRESOLVED.** Checkpoint terms must be confirmed separately from source
  terms before use. Not shortlisted.

### DROID-SLAM

- **OFFICIAL SOURCE FACT.** The official
  [DROID-SLAM repository](https://github.com/princeton-vl/DROID-SLAM) is a deep
  monocular SLAM system with custom CUDA correlation components and learned
  weights.
- **STATIC COMPATIBILITY INFERENCE.** Monocular translation has an arbitrary
  scale, and its iterative/global optimization can revise historical poses.
  Without a causal fixed-lag contract and metric scale bridge it is blocked for
  V7 primary use. It is not shortlisted.

### Essential-matrix/PnP sequential VO

- **STATIC COMPATIBILITY INFERENCE.** Two-view essential-matrix translation is
  up to scale. A PnP/RGB-D formulation using metric back-projected points can
  close scale, but at that point it is the same design class as sequential
  RGB-D odometry. A bespoke implementation is unnecessary before testing the
  mature Open3D path.

## 7. License audit

Only final-combination dependencies are license-gated here.

| Candidate | Source license | Checkpoint/weight terms | Access | Restrictions | Classification |
| --- | --- | --- | --- | --- | --- |
| Online BootsTAPIR | Apache-2.0 | Official README explicitly says all linked checkpoints are Apache-2.0 | public link; no stated auth | notice/patent terms; no NC restriction | `LICENSE_CLEAR_FOR_CURRENT_ACADEMIC_RESEARCH` |
| CoTracker3 online | CC-BY-NC-4.0 | official model card: CC-BY-NC-4.0 | public HF; no stated auth | attribution, noncommercial; redistribution under license terms | `LICENSE_RESTRICTED_BUT_USABLE_FOR_CURRENT_ACADEMIC_RESEARCH` |
| Apple Depth Pro | Apple repository LICENSE | README expressly places model weights under same LICENSE | official script; no stated auth | preserve notices/disclaimer; no stated NC limit | `LICENSE_CLEAR_FOR_CURRENT_ACADEMIC_RESEARCH` |
| Open3D 0.19 | MIT | no weights | public package/source | copyright/license notice | `LICENSE_CLEAR_FOR_CURRENT_ACADEMIC_RESEARCH` |

Exact source SHAs, checkpoint filenames, byte sizes, SHA-256 hashes, and copies
of the applicable terms remain pre-download gates for Phase 1B. This audit does
not equate public accessibility with permission.

## 8. Environment compatibility

| Candidate | Official/runtime surface | Current V7 overlap | Isolation | VRAM/runtime |
| --- | --- | --- | --- | --- |
| Online BootsTAPIR PyTorch | official PyTorch checkpoint/demo; PyTorch stack | plausible but not statically proven on Python 3.12/torch 2.3 | required | `VRAM_NOT_OFFICIALLY_SPECIFIED`; learned tracker GPU cost |
| CoTracker3 online | PyTorch + torchvision, CUDA recommended | declared core stack overlaps | required | `VRAM_NOT_OFFICIALLY_SPECIFIED`; sliding-window GPU cost |
| Depth Pro | official setup recommends Python 3.9; torch/torchvision/timm family | Python recommendation differs; dependencies not present as a pinned provider env | required | `VRAM_NOT_OFFICIALLY_SPECIFIED`; official report is model-specific, not this server |
| Open3D 0.19 | cp312 wheels documented by release/package distribution; no CUDA required for legacy RGB-D API | Python 3.12 feasible | install with the combination's isolated env | CPU incremental optimization; no model VRAM |

**STATIC COMPATIBILITY INFERENCE.** One isolated environment per complete
combination is safer than modifying the accepted project `.venv`. No official
source inspected establishes that the entire primary combination already works
together on torch 2.3.1/Python 3.12. The 24-GiB GPU is therefore a runtime
question, not an environment blocker at audit time.

## 9. Coordinate and scale compatibility

The primary scale chain is explicit:

1. Depth Pro emits depth in metres and focal length in input-image pixels.
2. After confirming depth convention and constructing the recorded K, the same
   depth/K pair back-projects tracked UV and builds each RGB-D odometry pair.
3. Open3D estimates each adjacent rigid transform from those metre-coordinate
   RGB-D points. Its translation therefore inherits the same depth scale.
4. Each accepted transform is inverted/composed only after a synthetic
   direction test, producing `T_0<-t` without future optimization.
5. Original UV is mapped explicitly through the tracker's resize and samples
   Depth Pro output restored to original resolution. No hidden crop is allowed.

This is a **STATIC COMPATIBILITY INFERENCE**, not proof of calibrated metric
accuracy. Depth Pro's complete-K limitation, exact depth convention, monocular
scale bias, and accumulated odometry drift are Phase 1B falsification targets.

By contrast, `metric depth + DPVO/DROID monocular translation` has mismatched
scale gauges. It remains `COMMON_COORDINATE_SCALE_BLOCKER` unless a causal,
history-fixed scale bridge is separately specified and validated.

## 10. Dynamic-scene pose risks

RGB-D odometry estimates camera motion from a rigid or dominant-static image
component. A moving person, animal, vehicle, or manipulated object occupying a
large fraction of the frame can bias the estimate; the resulting error would
contaminate every compensated particle.

Phase 1B may compare label-blind robust correspondence selection, RANSAC, or a
background/static-region candidate mask. Such a mask must be based only on
geometry/consensus, not authenticity, anomaly score, semantic identity, or
generator. Foreground residual is a frontend diagnostic only. A failed or
degenerate pose makes compensated geometry invalid; identity pose is not a
fallback. No mitigation is selected in Phase 1A.

## 11. ParticleSequence compatibility

Both proposed combinations can map to the current logical contract:

| Field | Source and rule |
| --- | --- |
| frame index / timestamp | existing decoder source index and real PTS |
| track_id | stable query identity assigned by adapter; never a continuous feature |
| uv | tracker output mapped to original frame; occluded/unavailable is NaN |
| visibility | tracker visibility/occlusion under one preregistered policy |
| xyz | explicit back-projection followed by accepted camera transform |
| geometry_validity | visible AND finite depth/K/transform/XYZ; independent mask |

There is no zero-fill, invented interpolation, or provider-confidence feature.
Tracker appearance and depth/pose private tensors do not enter the downstream
model. Track birth, death, and re-observation remain explicit mask patterns.

## 12. Candidate combination matrix

`PASS` below means source-level compatibility with a runtime test, not measured
scientific success.

| Dimension | Primary: Online BootsTAPIR + Depth Pro + Open3D RGB-D | Secondary: CoTracker3 online + Depth Pro + Open3D RGB-D | Depth Pro + monocular DPVO/DROID |
| --- | --- | --- | --- |
| causality | `LIKELY_PASS` | `UNCERTAIN` sliding-window finalization | pose may be online; scale still blocked |
| history immutability | `LIKELY_PASS`, must test emitted states | `UNCERTAIN`, must define delayed final output | `UNCERTAIN` optimization may revise |
| cross-frame identity | `PASS` | `PASS` | tracker-dependent |
| depth scale semantics | metre output; z/ray detail `UNRESOLVED` | same | same |
| pose scale semantics | inherits metric RGB-D scale | inherits metric RGB-D scale | arbitrary monocular translation |
| common coordinate feasibility | `LIKELY_PASS` | `LIKELY_PASS` | `BLOCKED` |
| dynamic-scene suitability | `UNCERTAIN`, rigid-background risk | same | `UNCERTAIN` |
| same-UV XYZ feasibility | `LIKELY_PASS` after mapping test | `LIKELY_PASS` after mapping test | scale blocked |
| ParticleSequence compatibility | `PASS` | `PASS` | `BLOCKED` for comparable XYZ |
| license | all clear | tracker noncommercial-restricted, usable | weight terms not all cleared |
| environment compatibility | `UNCERTAIN`, isolated env | `UNCERTAIN`, isolated env | custom CUDA/high integration risk |
| runtime complexity | tracker GPU + depth GPU + CPU odometry | same class, sliding-window tracker | custom CUDA VO plus depth |
| major blocker | calibration/depth convention and dynamic-pose runtime quality | tracker lookahead/history finalization | `COMMON_COORDINATE_SCALE_BLOCKER` |

## 13. Primary runtime combination

`PRIMARY_RUNTIME_COMBINATION`:

- Tracker: Online BootsTAPIR, PyTorch causal checkpoint.
- Depth: Apple Depth Pro, one independent inference per decoded frame.
- Intrinsics: Depth Pro focal length in original-input pixels plus an explicitly
  recorded centered-principal-point and zero-distortion assumption, accepted
  only after source and synthetic mapping verification.
- Pose: Open3D 0.19 adjacent-frame RGB-D odometry using the same Depth Pro
  depth/K; forward-only composition, no pose graph/BA/future refinement.

This is primary because its tracker has an explicit causal source claim and
clear source/checkpoint terms, while the depth/pose path shares one metric scale.
It is ready for falsification, not adoption. Failure to establish z/ray depth
semantics or a defensible K stops the runtime before XYZ claims.

## 14. Secondary runtime combination

`SECONDARY_RUNTIME_COMBINATION`:

- Tracker: CoTracker3 online checkpoint.
- Depth/intrinsics: the same Depth Pro path.
- Pose: the same forward-only Open3D 0.19 RGB-D odometry path.

The shared depth/K/pose semantics make the coordinate chain identical. It is
secondary because CC-BY-NC limits use and the online sliding-window API has a
less direct history-immutability story. Phase 1B may run it only after defining
which delayed outputs are final and proving prefix extension cannot change
stored history.

## 15. Blockers

- Pin full source SHAs and checkpoint identity/bytes/SHA-256 before download.
- Confirm Depth Pro depth is z-depth or implement the documented ray-to-z
  conversion; verify the full K assumption and original-pixel mapping.
- Confirm Open3D transform direction through a known-pose synthetic example.
- Show tracker emissions are causal and never retrospectively replaced.
- Quantify dynamic-foreground contamination and odometry failure/degeneracy.
- Verify packages in an isolated environment and measure actual memory/runtime.
- Establish that depth temporal jitter plus pose drift is below the structural
  signal of interest. A schema-valid but all-invalid output is failure.

These are runtime falsification gates. At audit time they do not create a known
license, scale, or environment impossibility for the primary combination.

## 16. Phase 1B exact runtime experiment proposal

Phase 1B is `V7 Phase 1B — Explicit Geometry Runtime Falsification`.

- Population: **REAL ONLY**, the already fixed real-primary sample/window used
  by the V6 per-horizon audit; begin with its stable center window(s), without
  selecting on geometry outcome. Fake data is excluded.
- Assets: one pinned primary source/checkpoint set, recorded licenses, bytes,
  hashes, environment manifest, exact decoded frame indices and PTS.
- Execution: run each input twice; run history prefix and longer prefix while
  preserving already emitted tracker/depth/pose outputs; process depth per
  frame and pose only from adjacent past/current RGB-D.
- Geometry checks: synthetic K/back-projection, pixel-center and resize mapping,
  transform direction, units, handedness, axes, z/ray convention, and one fixed
  reference gauge.
- Measurements: same-input repeatability; causal history invariance; track
  persistence/coverage and birth/death; depth temporal jitter; incremental pose
  drift and failure rate; same-track XYZ temporal stability; static/background
  geometry stability; moving-particle signal versus frontend noise; same-UV
  finite XYZ coverage; joint history/future valid length; elapsed time and peak
  VRAM.
- Comparison: report against the existing VGGT h1/h2/h4 `Q_target` values
  without rerunning VGGT or training a detector. Determine whether the explicit
  path is a more credible measurement basis for later `S_t`, `ΔS_t`, and
  `Δ²S_t`; do not implement those objects in Phase 1B.
- Stop conditions: unresolved license/pin, causal revision, unusable K/depth
  semantics, inconsistent metric gauge, invalid UV mapping, dominant pose
  failure, inadequate finite coverage, or resource excess. No threshold may be
  selected using fake performance.

Final Phase 1A decision:

`PROCEED_TO_EXPLICIT_GEOMETRY_RUNTIME_PHASE_1B`

This authorizes only a separately controlled real-only falsification of the
primary combination after its asset and semantic pre-download gates pass.
