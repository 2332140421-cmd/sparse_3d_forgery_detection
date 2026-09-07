# V7 GenVidBench selective real-pilot materialization

Status: `GENVIDBENCH_SELECTIVE_MATERIALIZATION_BLOCKED`

## 1. Objective

本轮只尝试从官方 Hugging Face archive shards 定向物化已经冻结的
32 个 train real 与 16 个 test real source。没有重新抽样、没有下载 fake，
也没有进入 frontend、component 或 normality 实验。

## 2. Archive access protocol

- Dataset: `jian-0/GenVidBench`
- Fixed revision: `701cafb6f999d7ea0cbf3c354df6177311a4d824`
- Inventory: `materialization/hf_file_inventory.json`
- Target map: `materialization/target_archive_map.json`
- Plan: `materialization/materialization_plan.json`

官方 API 实际返回：

| archive | format | size |
| --- | --- | ---: |
| `GenVidBench/Pair1/vript.rar` | RAR5, uncompressed members | 49,190,751,379 B |
| `GenVidBench/Pair2/hd_vg_130m.7z.001` | 7z volume | 21,474,836,480 B |
| `.002` | 7z volume | 21,474,836,480 B |
| `.003` | 7z volume | 21,474,836,480 B |
| `.004` | 7z volume | 15,594,648,596 B |

Vript 的 bounded HTTP range probe 能读取 RAR5 member headers，并确认
archive 内部使用 `vript/` 前缀。HD-VG-130M 的多卷 7z member table 未能在
不取得完整 volume set 的前提下独立确认。

## 3. Target resolution

48 个冻结 target identity 未改变：

- Vript train：32/32 `RESOLVED`。依据 official `Pair1_verify.txt`
  的唯一 filename 与 RAR member-prefix probe。
- HD-VG-130M test：0/16 `RESOLVED`，16/16 `UNRESOLVED`。官方
  `HD_VG_130M_verify.txt` 只保留去掉第一个点之后的 basename，且多卷
  7z member table 未能无完整下载读取；没有进行 fuzzy 猜配。
- `AMBIGUOUS`：0。

未解析的 test target 没有被任取一个 archive member。所有 map 行都记录
`source_id`、role、source、expected identity、archive shard、member
状态、resolution method 和空的 `video_sha256`。

## 4. Download bound

完整 48 target 所需 unique shards 为 5 个，总大小：

- 129,209,909,415 bytes（约 120.34 GiB）

即使按任务规定的最低 reduced tranche（16 train + 8 test），仍需完整
Vript RAR 和全部四个 HD-VG 7z volumes：

- 129,209,909,415 bytes（约 120.34 GiB）
- 下载 gate：20 GiB
- 磁盘剩余空间：约 996 GiB
- 判定：`SELECTIVE_MATERIALIZATION_TOO_EXPENSIVE`

因此没有下载任何完整 archive，没有提取视频。外部 HF cache 只保留约
79,231,692 bytes 的 HTTP range/格式探测片段；archive payload download
为 0 bytes。

## 5. Media validation and review material

- materialized real videos：0/48
- access failures：48
- true media decode failures：0
- media validation：未进入 decode 阶段，不能把 ACCESS_FAILURE 记为
  `MEDIA_DECODE_FAILURE`
- `materialized_real/`：未创建目标视频文件
- contact sheets：0
- `real_review/review_core_materialized.csv`：
  `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_genvidbench_core_pilot_v1/real_review/review_core_materialized.csv`

该 CSV 保留上一轮 48 个 source identity，所有 `video_decision`、
`video_reason_code` 和人工 notes 仍为空；原始
`real_review/review_core.csv` 未覆盖。

由于没有真实可打开视频，本轮不能提供 materialized train/test example
paths；逻辑 target examples 仍由上一轮 manifest 保留。

## 6. Reproducibility

已保留：

- fixed HF revision；
- archive path、size、OID；
- target source ID 与 role；
- official verify-path resolution method；
- archive member（仅已确认的 Vript target）；
- materialization plan；
- `video_sha256 = null`，因为没有 extracted video。

## 7. Tests

新增 research-tool 测试覆盖：

- 冻结 target identity；
- deterministic target mapping；
- ambiguous mapping rejection；
- unresolved 7z member 不猜配；
- 20 GiB size gate；
- fake 不进入 real target map；
- requested-member-only extraction；
- path traversal rejection；
- SHA lineage 初始为空；
- materialized review 决策保持空值。

本轮测试和完整验证结果将在提交前报告。

## 8. Current status and next step

当前状态是 `GENVIDBENCH_SELECTIVE_MATERIALIZATION_BLOCKED`，原因是官方
archive 分卷的最小可行下载成本超过本轮 20 GiB gate，而不是媒体解码失败。
不扩大预算、不更换 target、不进行新的数据集审计。

若用户后续明确提高下载预算或提供可访问的 individual MP4/解包后的 bounded
subset，只需按现有 `target_archive_map.json` 继续 materialize real，验证
媒体并完成人工审核；随后直接进入：

`finalize Core real pilot → Explicit 3D → unchanged component baseline
→ S_t / ΔS_t / Δ²S_t → real-only normality training`。

本轮未修改 `src/sparse3d_forgery/`、V7 formal design contract、component、
frontend、训练或 fake detector；未下载 fake、全量 dataset，也未访问旧 R7/V5。
