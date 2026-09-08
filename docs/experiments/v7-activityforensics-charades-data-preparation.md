# V7 ActivityForensics + Charades data preparation

Status for this run: **PARTIAL_TARGETED_REVIEW_READY**

This is a data-preparation report, not a V7 suitability conclusion. The run
stopped after the official endpoints proved reachable but their current server
throughput made complete media acquisition non-viable. Resumable partial files
were retained.

## Objective

Prepare an auditable candidate population for later human review of persistent
structured motion. No V7 frontend, particle extraction, component discovery,
normality model, training, or fake detection was run.

## Official Sources

- ActivityForensics dataset: `ActivityForensics/ActivityForensics`, revision
  `a34d4b7b04b0f3f3e26ba900adc367218667c581`,
  [official Hugging Face dataset card](https://huggingface.co/datasets/ActivityForensics/ActivityForensics).
- ActivityForensics code and policy: [official repository](https://github.com/ActivityForensics/activityforensics),
  [DATA_POLICY.md](https://github.com/ActivityForensics/activityforensics/blob/main/DATA_POLICY.md).
  The saved policy states CC BY-NC 4.0 research/non-commercial use, attribution,
  and no redistribution.
- Charades: [official project page](https://prior.allenai.org/projects/charades),
  [official license](https://prior.allenai.org/projects/data/charades/license.txt),
  and the official original archive URL
  `https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip`.
  The official page does not provide an archive checksum; this is recorded as
  `not_provided`.

## License / Access

The public metadata, annotation archive, official Charades metadata archive,
and the two official media endpoints were accessible without login, token, or
manual license acceptance. No access control was bypassed. The downloaded
policy/license snapshots and request headers are under the data root's
`source/*/source_info/` directories.

## Download

Expected ActivityForensics media population: 5,389 MP4 files and
118,549,962,852 declared bytes. The fixed Hugging Face revision, CSV splits,
`annot.zip`, dataset card, code README, policy, per-file declared SHA-256
metadata, and an auditable download manifest were saved. Only 32 MP4 files
(176,563,066 bytes) completed before the transfer was stopped; 5,357 remain
missing.

Expected `Charades_v1.zip`: 58,807,341,227 bytes. The resumable S3 transfer
retained 20,873,216 bytes and is not a usable archive. No Charades MP4 was
extracted. `acquisition/download_manifest.json`, `checksums.json`, and
`run_summary.json` record expected/local bytes, local hashes where available,
and the stop reason.

## ActivityForensics Population

The official metadata contains 3,824 train rows and 1,565 test rows. Across all
segments the generator counts are `fcvg=1,559`, `ltx=1,058`, `scifi=1,152`,
`vace-1.3B=1,061`, `vidu=265`, and `wan=1,437`; operations are
`add=1,674` and `delete=4,858`. Edit multiplicity is preserved (4,406
single-edit, 846 two-edit, 114 three-edit, and 23 four-edit files).

## Charades Mapping

The parser does not infer a pair from a directory name. It cross-checks every
`charades@<split>_<operation>@<source_id>` filename segment and the official
annotation archive against `Charades_v1_train.csv` and `Charades_v1_test.csv`.

- ActivityForensics fake files with exact Charades lineage: **1,438**.
- Unique exact Charades source IDs: **976**.
- `EXACT`: **1,438** files; `AMBIGUOUS`: **0**; `UNRESOLVED`: **3,951**
  (the latter are the non-Charades source lineage and are deliberately not
  paired).
- Exact-file generator counts: `fcvg=697`, `wan=606`, `vidu=135`.
- Exact-file operation counts: `delete=971`, `add=392`, and 75 files contain
  both operations; segment counts are `delete=1,270`, `add=493`.

The complete per-file mapping, including every official interval and evidence,
is `paired/manifests/activityforensics_source_mapping.json`; the summary is
`validation/lineage_validation.json`.

## Media Validation

PyAV-only validation was run over the exact mapping population. All 32
completed ActivityForensics files were `MEDIA_VALID` (container open, video
stream, RGB frame decode, finite monotonic PTS, dimensions); the remaining
2,382 checked items (missing ActivityForensics files and all 976 missing
Charades originals) were `MEDIA_MISSING`. No media failure was converted into
an accepted pair, and no fallback timestamp or visual quality rule was used.

## Human Review Package

Because no Charades original was available and no complete pair passed media
availability, `review/review_manifest.csv` and `.json` contain zero rows and
`review/batch_001/` contains no symlinked pair. `review/README.md` defines
the human fields and explicitly leaves `V7_CORE_DECISION` to the user; no
ACCEPT/REJECT/UNCERTAIN value was auto-filled.

## Targeted Review Attempt

The follow-up targeted materialization froze 100 primary and 30 reserve
sources by metadata-only round-robin over `(generator, operation)` cells. The
primary coverage was 25 each for `fcvg/delete`, `vidu/add`, `wan/add`, and
`wan/delete`; each source had one fake variant.

Thirty-seven checksum-valid targeted fake files were reused (33 primary and 4
reserve). Ninety-three targeted fake files were not downloaded because the
official payload transfer timed out. The official Charades endpoint passed a
1 KiB `HTTP 206` probe and its ZIP64 central directory was read successfully:
9,849 entries, including 9,848 MP4 members. Exact member `Charades_v1/00ZCA.mp4`
was located, but its 63,880,942-byte compressed payload timed out; no Charades
original was materialized and no valid pair was formed.

Targeted audit files are under `acquisition/targeted_review_v1/`, including
`candidate_sources.json`, `selected_sources.json`, `reserve_sources.json`,
`charades_range_index.json`, `activityforensics_downloads.json`,
`targeted_media_validation.json`, and `run_summary.json`.


## Network Path Diagnostic

Diagnostic status: **NETWORK_PATH_DIAGNOSED_BUT_DOWNLOAD_BLOCKED**.

The follow-up bring-up was limited to one exact source (`00ZCA`) and did not
expand the review population. The environment inherited
`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` (and lowercase equivalents) pointing to
the redacted local proxy `http://127.0.0.1:17890`. Git global/system proxy
settings were absent; no curl proxy file was present, and `/etc/wgetrc` had no
active proxy directive. No credential-bearing value was recorded.

The same range protocol was used for every request: `HTTP 206`, exact
`Content-Range`, and exact body length were required. Temporary bodies were
deleted after each measurement. Representative measurements were:

| path/mode | 1 MiB | 8 MiB | 16 MiB | 32 MiB | 64 MiB |
| --- | --- | --- | --- | --- | --- |
| S3, current environment (first run) | 2.32 s / 0.43 MiB/s | 3.00 s / 2.65 MiB/s | 4.67 s / 3.41 MiB/s | 6.96 s / 4.59 MiB/s | 10.31 s / 6.19 MiB/s |
| S3, current environment (repeat) | 22.06 s / 0.045 MiB/s | timeout at 180 s, 1.44 MiB received | not run after stop | not run | not run |
| S3, no proxy | 43.56 s / 0.023 MiB/s | timeout at 70 s, 2.0 MiB received | timeout at 110 s, 3.7 MiB received | stopped after lower-size failures | stopped after lower-size failures |
| S3, explicit local proxy | not separately repeated | timeout at 90 s, 5.1 MiB received | not run | not run | not run |

For the fixed unresolved fake
`video/02_wan/0YOXV+0.00=5.00=charades@train_delete@0YOXV@577@wan.mp4`,
the Hugging Face resolve URL returned exact `206` ranges through the current
environment for 1/8/16 MiB at approximately 0.27/1.73/3.36 MiB/s. Its metadata
resolve response was `302` to the redacted `us.aws.cdn.hf.co/xet-bridge-us`
host, with `x-linked-size=29481372` and the fixed revision, followed by a
final `206` range response; signed query credentials were not recorded. The
no-proxy Hugging Face run was not continued after the direct S3 path had
already shown low throughput; no full fake was downloaded.

The primary diagnosis is **BOTH_PATHS_LOW_THROUGHPUT**, with secondary
`PROXY_PATH_DEGRADED`, `DIRECT_PATH_DEGRADED`, and
`LONG_RANGE_UNSTABLE`. A 1 MiB explicit-proxy probe succeeded 4/5 times, but
one request timed out after receiving only 365 KiB. A resumable 1 MiB attempt
for `00ZCA` completed only chunk 0/61 after six retries; the compressed
`.part` and identity state were retained for a possible later resume. The
logical compressed payload is 63,880,942 bytes, but no final MP4 was written,
decompressed, or CRC-accepted. No stable 8/16 MiB strategy was demonstrated,
so no reusable downloader was added and no single pair was created.

Single-pair state at stop:

- source: `00ZCA`, exact lineage, fake `fcvg/delete`;
- fake: existing checksum-valid `MEDIA_VALID` file;
- real: no completed media, therefore `MEDIA_MISSING`;
- review pair: not created;
- persistent new media bytes: 0; only one 1,048,576-byte compressed range is
  valid in the retained resumable state (the `.part` file is preallocated).

This diagnostic does not establish ActivityForensics suitability for V7 Core.
It only establishes that both currently available network paths are too
unstable for bounded single-pair materialization.

## Targeted Download Resume

The follow-up resume used the already selected and fixed proxy environment
(NODE3); no node benchmark or node switch was performed. Existing checksum-
valid, `MEDIA_VALID` ActivityForensics fakes were reused. The frozen primary
order was resumed without re-sampling: 31 sources were attempted, 30 primary
sources formed valid pairs, and no reserve substitution was needed. Thirty
fake files were reused and no new fake payload was downloaded. Thirty exact
Charades originals were range-materialized and validated; 143,156,781 bytes
of compressed Charades ranges were newly transferred and 143,781,190 bytes of
validated MP4 output were written. The pre-existing `00ZCA` partial/state was
retained; its identity did not match the current cached member index and it
was recorded as one source failure, not deleted or restarted.

The resulting pairs are `01KML` through `5D85P` in frozen completion order.
The generator distribution is `vidu=17`, `wan=9`, `fcvg=4`; the operation
distribution is `add=23`, `delete=9`. No reserve source was used. The current
review package contains 30 valid `EXACT + MEDIA_VALID` pairs under
`review/batch_001/`, with symlinks and per-pair `INFO.json`; the CSV and JSON
manifests contain the same 30 rows and all human-review fields remain blank.
No contact sheets were generated. The resume reached
`PARTIAL_TARGETED_REVIEW_READY`; this is a review-materialization state only,
not a V7 Core accept/reject or feasibility conclusion.

## Paths

- ActivityForensics metadata/partial fake media: `source/activityforensics/`
- Charades metadata and partial archive: `source/charades/`
- Exact lineage: `paired/manifests/activityforensics_source_mapping.json`
- Media validation: `validation/media_validation.{csv,json}`
- Acquisition audit: `acquisition/`
- Review placeholders: `review/`

## Boundaries

This run did not modify old data, access the old R7 repository, modify formal
`src/sparse3d_forgery/`, run a frontend, tracking, depth, pose, GPU, particle
builder, component algorithm, `S_t`, `ΔS_t`, `Δ²S_t`, normality model,
training, fake detection, or AUROC. No train/validation/test split for V7 was
created. The next action is to resume official transfers after network
throughput is restored, then validate both media sides and build the 100-source
human-review batch; stop again for human review before any feasibility work.
