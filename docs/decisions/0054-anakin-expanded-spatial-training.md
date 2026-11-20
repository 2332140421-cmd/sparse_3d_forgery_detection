# ADR-0054: Expanded source-isolated spatial training

Status: accepted for the user-authorized development experiment.

The user explicitly removes individual human-review approval as a prerequisite.
Use official annotations, established technical alignment and original-video
isolation. Keep all existing human evidence verbatim; explicit negative mask
reviews, unresolved mapping, role conflicts and protected test sources remain
excluded. This supersedes only the old candidate-report waiting requirement.

Reuse the existing generic DINOv2 ViT-L/14 reg4 + LoRA4/8/.25 + official
HighResContextHead, full-frame384/pad14, pixel BCE + soft Dice and running-BN
evaluation. No XYZ/history or architectural changes. Freeze the93 PASS candidates
according to prior source-group drafts and merge original11 train/4 dev.
One fresh seed20260909 condition, ten no-replacement epochs, U=10*Ntrain;
100 predetermined train-only BN-statistics forwards per epoch. Evaluate full dev
and fixed training monitors at1/3/5/10. Epoch10 is primary; additional-dev AP picks
only a separately disclosed development checkpoint, tie to earlier epoch.

The existing11-source model is a read-only reference on the same dev set; its
budget differs, so this is not a pure sample-count causal comparison. Missing
observations are limited to working RGB/GT caches. Test remains metadata-only.
Old entries, cached observations, roles, manual records and results stay intact.
