# V7 R 新增可观测窗口训练补充 pilot

本补充建立在既有 periodic re-query pilot 之上，不覆盖或改写原实验及其主报告。
既有数据盘产物位于：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_requery_training_support_pilot_v1/`。

## 资格核对

`runner.py:763` 中的 `paired_eligible` 同时要求本模式和另一模式都有有效五时刻结构支撑，且两行标签均为 0/1；它不是单纯的标签资格。独立依据父片段标注区间、实际窗口起点和固定 1 秒窗口重算后，96 个窗口全部标签合格：48 个 `REAL_NEGATIVE`、48 个 `FAKE_MANIPULATION`，没有 `BOUNDARY_MIXED` 或 `OUTSIDE`。标签映射与 manifest 完全一致。

六个 R-only 窗口均为标签合格且 R 支撑有效，但 O 无共同支撑，因此旧 `paired_eligible=False`、旧 R_SET 训练纳入数为 0/6；旧 held-out 模型已有这六个窗口的评分。资格清单和逐条原因见数据盘 `audit/eligibility_audit.csv` 与 `audit/r_only_audit.csv`。

## 补充条件

新增条件为 `R_SET_ALL_VALID_TRAIN`：每个原有 15-source LOSO fold 使用训练 source 内全部“标签合格且 R 有效”的窗口，其他模型、SET_A 表示、权重、fold 标准化、Adam、200 epochs 和三个 seed 均保持不变。正式训练完成 45 个模型（15×3），使用 CUDA；1 epoch smoke 模型未混入正式记录。每折新增窗口、旧/新权重和 fold 身份见 `training_window_manifest.csv`。

主比较仍固定在原共同支持、标签合格且双类别的 14 source、76 window 集合。旧 R_SET 与新条件的 source-macro AUROC 分别为 0.787698 与 0.843254，差值 0.055556，source bootstrap 95% CI 为 [-0.007937, 0.158730]，区间跨 0。pooled AUROC 分别为 0.597367 与 0.641026；这些只是开发性补充结果，不是独立确认性测试。

全部 85 个标签合格且 R 有效窗口上的新模型评分仅作扩展描述；六个 R-only 窗口不混入主要配对指标。当前仍没有空间真值，R 结构支撑不能证明观测单元位于失真区域。

完整数值、逐 source/seed 结果、R-only 分数、状态和成本见数据盘 `report.md`、`evaluation/summary.json` 及相关 CSV。代码入口为 `research_tools/v7/periodic_requery_probe/requery_training_support_pilot.py`。
