# V7 实验数据流与缓存复用核查

本次核查只读取已完成 V7 pilot 的数组、manifest、模型记录和小规模 CPU 前向；未重跑 frontend、tracking、depth、pose、segmentation 或训练，未访问旧 R7/V5。完整小型证据保存在数据盘：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_dataflow_cache_audit_v1/`

## 结论先行

在每个指定实验中按稳定规则抽取一个 real 和一个 fake（pair 使用 b=0、其余优先非零偏移）后，未发现“名义条件 B 实际消费条件 A 特征或旧模型”的实际证据。保存特征由原始粒子数组重建后与缓存逐元素一致；抽样模型前向与保存分数的最大误差均在 `1e-5` 容差内。这个结果是有限抽样核查，不等于所有历史缓存均有可追溯指纹。

| 实验 | 抽样窗口 | 条件/模型核对 | 结果 |
|---|---:|---|---|
| periodic_requery | 1 real + 1 fake，O/R，共 12 条特征/分数证据 | O/R 粒子、query cohort、父片段身份；SET/RAW/MULTI | 全部 PASS |
| multi_order_sequence | 1 real + 1 fake，共 8 条 | SET_A、RAW_SEQ、MULTI_ORDER_SEQ、SHUFFLED_MULTI_ORDER | 全部 PASS；合成非退化输入的多阶/打乱差异最大约 294.49 |
| pair_trajectory | 1 real + 1 fake，共 6 条 | 三条件模型前向；固定关系与控制变换 | 全部 PASS；2401 个局部单元的距离多重集合保持 |
| attention_pooling | 1 real + 1 fake，共 4 条 | `MeanBaselineModel`/`AttentionPoolModel`；保存分数 | 全部 PASS；扰动 attention 参数后最大输出变化约 0.01009 |
| source128_extension | 1 real + 1 fake，3 个正式 seed | 128 名义 source、实际特征、标准化、验证分数 | 全部 PASS；122 个有效训练 source、718 个训练窗口，验证无训练 source 重叠 |

## 各项核对

- periodic requery：零偏移 O/R 共用初始化是协议允许的；非零偏移 R 的 query cohort、粒子 lineage 和 source/role/parent/window 身份均在消费缓存前核对。抽样 O/R 特征和分数均可复现。
- multi-order：四种条件从对应数组生成 batch；合成样本确认多阶输入与打乱输入不是同一数组。实际抽样分数可复现。
- pair trajectory：消费关系数组前核对视频、source、role、parent、query cohort 和 b=0/非零偏移规则；ID shuffle 的变化比例约 0.956（轨迹约 1.000），time shuffle 的窗口变化比例为 1，时间间隔保持。
- attention pooling：模型类确实分流，注意力参数连接到输出；均匀初始化退化检查和参数扰动检查均通过。
- source128：名义 128 source 中实际有效训练 source 为 122，缺失 source 为 `2544C, 54XD1, 7FNZZ, 869NM, AYXFY, DU7H1`；这是覆盖记录，不是 A/B 错用。三个正式记录均为 200 epoch 模型，验证训练 source 交集为空，重算标准化与记录最大差为 0。

## 已修复的最小风险

1. 五个 pilot 的模型恢复不再仅凭 seed/source/condition/status 等键跳过；新记录保存 `input_identity`，恢复时核对支持/关系/特征、训练窗口集合、条件、留出 source 以及必要协议哈希。缺少旧指纹的记录会写入 `state/unverified_model_records.json`，不能静默复用；旧记录仍保留在文件中。
2. periodic 的非零偏移 R 粒子在特征消费边界核对 source/video、role、parent、pair、window 和 query cohort；错误 cohort 或错视频不会通过。
3. source128 支持行现在要求 `parent_id` 与冻结窗口一致。历史支持行若没有该证据，不会在未来恢复中被冒充为已验证缓存。
4. README 补充当前 V7 research-tools 入口及外部数据/权重依赖，未把研究工具描述成正式检测链。

## 可追溯性边界与是否重跑

历史模型记录普遍早于 `input_identity`，历史 periodic support 也缺少 `parent_id`；因此历史运行时是否曾经发生过错误复用，不能仅靠旧记录完全证明。当前抽样重建没有发现污染，也没有理由据此重跑昂贵前端或训练。若未来恢复某个 pilot，应仅对缺少身份证据的受影响 fold/condition 重新生成；不需要全量重置。

自有工作流入口、runner、测试和 `scripts/run_v7_source128_screen.sh` 均已纳入 Git。视频、ParticleSequence、外部跟踪/深度/姿态代码与权重、特征/模型/分数等实验资产仍在数据盘，未提交 Git。

证据文件：`summary.json`、`evidence.csv`、`source_code_inventory.csv` 和本目录的 `report.md`。本次定向测试覆盖身份拒绝、条件/输入指纹不匹配拒绝及合法缓存复用；未运行全仓库训练或数据重算。
