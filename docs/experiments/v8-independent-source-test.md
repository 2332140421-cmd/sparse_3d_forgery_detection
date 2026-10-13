# V8 按source隔离的冻结测试

起点55905ae7a7d6e4724bdce300e5618354dc5d7a3c；分支v8-independent-source-test。
独立输出`derived/v8_independent_source_test_v1`。本次无训练、拟合或阈值搜索。

## 资格和冻结

纠正此前准备文档的保守排除：review_manifest字段及README只记录媒体/溯源，
人工作业字段为空，不能凭登记排除。只查直接相关V7测量/评分及当前V8输入清单，
不做全服务器历史审计。19个本地官方test source中，0DVVD/0FM93/0KTWY/0PU21
有实际历史测量或检测分数，排除。0HV07曾运行前端但没有有效结构/模型评分，
periodic训练按paired_eligible筛选不纳入它；保留，不把缺失占位算性能使用。
历史模型未保存完整输入指纹的范围不能完全追溯，因此不称严格从未接触。

全部15个合格本地source纳入，30视频；hash seed20260909排序，fake官方记录
稳定排序选唯一可用变体。清单在评分前保存，预计19607元数据帧；不补到20对，
不按分数、几何成功率或fake观感筛选。官方Charades scene/actions支持室内自然
活动范围，未声称人工核验了全部动作。视频内容SHA与source/role共同标识，不使用
pair_id单键。名单和资格证据保留数据目录。

Vidu15个fake，训练只见wan/fcvg。这是同数据族独立source与未见生成器的联合迁移，
不是跨数据集，不能独立分解两个因素；小子集不代表完整benchmark。

## 数值路径

一次原生全视频前端，MoGe/CoTracker/ResNet、standardizer/grouping与原连续帧
实现不变。无抽帧、截片或未来上下文；全部非padding格（包括无几何/无历史）参与。
直接复用`full_video_global_context.runner.process`，只参数化总视频计数与禁用
已通过的重复大smoke，额外按CUDA events记录各模型设备耗时；原入口默认行为保留。

明确九份权重：COMPONENT_ADAPTED=GLOBAL_COMPONENT epoch5×3seed；
COMPONENT_WINDOW=GLOBAL_COMPONENT epoch100×3seed；CELL_WINDOW=GLOBAL_CELL
epoch100×3seed。strict加载并核对condition/seed/epoch/hash，eval冻结。
真实全帧attention和原多前驱GRU，hidden跨64帧块；各seed独立完整面积LSE，
随后平均最终video logit，阈值0。无模型平台或新前端。

主要结果为ADAPTED video AUROC及自身95%CI；辅助AP、FPR、Recall、Balanced
Accuracy、P/F1/Accuracy/混淆矩阵及逐seed。仅两配对：ADAPTED−WINDOW与
COMPONENT_WINDOW−CELL_WINDOW；前者适配影响，后者同训练阶段分区对照。
source簇bootstrap10000次/seed20260909，同次采样保留真假并用于全部组。
不将适配−CELL称匹配消融，不做无真值的帧/像素定位。

## 验证与执行

3项本轮定向测试通过；旧梯度、padding、连续状态验证只读引用，不重跑。
正式入口另做两帧九模型加载与连续/块输出核对，容差2e-5。

```bash
PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$PWD .venv/bin/python -u -m research_tools.v8.full_video_global_context.independent_test --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_independent_source_test_v1 --resume
```

唯一任务，输出目录flock防重；TERM在当前64帧块保存后停止。成功视频不重复处理。
实际结果以全部30视频、九权重评分、评价及report完成后的final_status为准。
当前仅冻结与实现完成，尚不标实验COMPLETE；执行结果随后补入本文。
