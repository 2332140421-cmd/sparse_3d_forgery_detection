# ADR 0010: V7 dynamic structure evolution normality

## Status

Accepted on branch `v7-dynamic-structure` as a candidate research direction.

## Context

V6 tested deterministic per-particle future 3D displacement prediction. Its bounded evidence motivates, but does not prove, a revised hypothesis:

- temporal-only real learnability was very weak: the GRU reduced real-validation RMSE over zero displacement by only 1.36%, while its mean and median L2 error were worse;
- the spatial-dependency probe was `FAIL`: the tested dense soft-relation model was worse than temporal baselines on real validation and did not support the bounded V6 hypothesis;
- the multi-window probe was `MIXED_INCONCLUSIVE`; scanning more windows did not establish learned normality;
- prediction scores were highly correlated with frontend alignment instability, including real T0 correlation `0.8522` in the multi-window audit;
- per-horizon prefix targets only partially reduced instability: real median `Q_target` remained 1.6056/0.9326/0.4278 for h=1/2/4;
- CUT3R runtime evaluation was blocked because license coverage for the fixed source/checkpoint combination remained unresolved.

These results neither establish that V7 is correct nor disprove all future-displacement or spatial models. They show that the current evidence does not justify treating deterministic pointwise outcome prediction as the primary research task.

## Decision

- V7 studies real-only normality of continuous three-dimensional dynamic-structure evolution in the Persistent Structured Motion Domain.
- Its candidate chain is explicit 3D measurement, persistent particles, motion-coherent dynamic components, intra/inter-component relation state `S_t`, first-order structural change `ΔS_t`, second-order structural evolution `Δ²S_t`, and multi-level anomaly evidence.
- `ΔS_t` is structural-state change, not a handcrafted velocity feature. `Δ²S_t` is change of structural evolution, not a claim of physical acceleration.
- Motion-Coherent Dynamic Components are correspondence- and motion-coherence structures, not semantic Parts, objects, rigid-body truth, segmentation classes or fake regions.
- Component discovery is label-blind and separated from anomaly detection.
- The first implementation gate is a stable explicit 3D frontend feasibility audit. Explicit geometry measures XYZ and cannot itself define authenticity.
- Component algorithm, state representation, distance, normality model and score aggregation remain unfrozen.

## Domain boundary

The initial domain covers persistent structured motions such as rigid, articulated, ordinary-object and moderate non-rigid motion with limited interaction. Processes dominated by fluid/smoke/flame behavior, fragmentation, explosion, severe merge/split or topology birth/death are deferred because persistent correspondence is not an adequate initial observation model. This is a physical observability boundary, not a semantic-scene restriction.

## Inherited and non-inherited V6 assets

V7 retains video decoding, true timestamps, ParticleSequence foundations, track identity and validity semantics, lineage, reproducibility infrastructure and V6 experiments as baseline evidence.

V7 does not adopt the deterministic future-displacement predictor, T0/T1/S architectures, dense topology, fixed V6 probe horizons, loss or aggregation as its formal method.

## Consequences

- `docs/design_contract.md` remains an unchanged V6 historical contract; `docs/v7/design_contract.md` governs this V7 research branch.
- No component, state, model or detector may be implemented before the Phase-1 frontend measurement gate is tested.
- No fixed rule may map large or small second-order change to authenticity.
- Reprojection, depth consistency and pose residuals may be frontend diagnostics only.
- V7 is a falsifiable candidate direction, not a validated method or claimed improvement.

## Supersedes

For branch `v7-dynamic-structure`, this ADR supersedes the V6 choice of deterministic direct multi-horizon particle displacement prediction as the primary research task and permits non-semantic Motion-Coherent Dynamic Components. It does not rewrite ADR 0001–0009 or the V6 design contract, and it does not authorize restoration of any prior model, feature, residual or anomaly rule.
