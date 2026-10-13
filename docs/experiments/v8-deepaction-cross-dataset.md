# V8 DeepAction小规模跨数据集冻结测试

起点`d5237f1c06ff8911baf02fb115eabcb01d715fd7`；分支
`v8-deepaction-cross-dataset`。无训练、微调、统计拟合、阈值校准或模型变动。
独立输出`derived/v8_deepaction_cross_dataset_v1`。

## 官方来源与选择

[作者数据](https://huggingface.co/datasets/faridlab/deepaction_v1)通过官方API固定
revision `476932c3369424ca59dc1f8fb758cb3c13fe63fe`；保留文件清单、README及captions。
[论文](https://arxiv.org/abs/2412.00526)：Bohacek & Farid，Human Action CLIPs。
Pexels为真实媒体；RunwayML/CogVideoX5B为生成媒体，依官方目录赋标签，不按观感。
生成媒体CC BY 4.0，真实媒体Pexels许可；原视频不提交Git。未安装datasets旧版、
未执行trust_remote_code、未下载整包或整个数据集。

评分前依据caption语义冻结52类室内日常活动候选池，seed20260909稳定SHA排序。
每类按官方revision文件字典序取第一份媒体。只审核real：12个均匀原生帧的
简短视觉检查，不宣称逐帧完整切镜/相机运动审计。原先20类中：73海边室外明确
排除，32/19/59无法确认室内标为UNCERTAIN；后续69镜头明显拉远，按相机条件
排除。依原顺序补审75/69/23/25/22，得到20个准入类。fake未被查看或评分来筛选。
共获取25份real审核候选，仅20份准入real进入前端；另获取20份所选fake。

准入动作ID顺序：76,47,18,29,26,68,82,62,67,78,49,71,80,35,17,70,75,23,25,22。
fake按准入序号交替RunwayML/CogVideoX5B，各10。40视频元数据帧总数12302，
小于30000预算，全部保留，无截视频/抽帧。若超预算的既定规则为最大可容纳的
原顺序类别前缀，最低10类；不按视频长短重排。本轮未触发缩减。

每份媒体以官方路径、revision和内容SHA赋唯一video/source身份，共40个媒体身份；
action_id只是语义匹配，不构造同源派生关系。bootstrap按20个action簇重采样，
每次保留该类real/fake，10000次、seed20260909，所有模型组同次抽样。

## 复用与冻结

`deepaction.py`只增加官方获取、scene材料、manifest、预算前缀与报告适配；
`independent_test.py`原主循环增加显式`--deepaction`，提取现有评分/成本汇总函数，
bootstrap仅增加明确的action_id键。没有数据集注册器或第二套前端。
原ActivityForensics入口默认仍按source簇，旧产物不修改或重新评价。

九权重：COMPONENT_ADAPTED epoch5×3seed、COMPONENT_WINDOW epoch100×3seed、
CELL_WINDOW epoch100×3seed。strict加载、condition/seed/epoch/hash核验，eval冻结。
全部正常解码帧，只运行一次MoGe/CoTracker/ResNet前端；旧standardizer/grouping
冻结。256画布、基础格、动态component/CELL singleton、过去上下文、真实delta_t、
多前驱、全帧attention、GRU及跨64帧状态不变；所有真实内容格（包括缺失）参与，
padding排除。每seed整视频面积LSE，再平均最终video logit，阈值0。

主结果ADAPTED video AUROC及自身action-cluster CI；辅助AP、FPR、Recall、Balanced
Accuracy、P/F1/Accuracy、混淆矩阵及逐seed。两配对仅ADAPTED−COMPONENT_WINDOW、
COMPONENT_WINDOW−CELL_WINDOW；不把适配−CELL称匹配消融。
这是受限人体日常动作场景下的跨数据集及生成方式联合迁移，不代表完整benchmark，
不验证帧/像素定位，不直接对照不同子集/训练协议的论文数字。

6项定向检查通过（5项整组及新增manifest检查）；正式入口仅做两帧九权重加载与连续/块输出检查，不重跑大smoke
或梯度测试。正式代码在运行前固定，连续前端复用现有`runner.process`。

## 获取问题与成本口径

Pexels/18、17、22曾连接截断，保留失败与部分下载；仅定向官方HTTP Range续传，
完整LFS SHA和字节数校验后才接受，无替代镜像或绕过访问控制。
初期Pexels/17一次失败计时被旧单记录覆盖，无法精确恢复，后续失败改独立记录。
下载耗时只报告已记录下界（并行存在重叠），冗余失败流量UNKNOWN；成功媒体总字节
精确记录，不把未知成本填0。正式前端、推理和评价耗时独立实测。

## 执行

冻结清单已保存，尚未标实验COMPLETE。正式命令（已独立screen执行时勿重复启动）：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
export PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$PWD
.venv/bin/python -u -m research_tools.v8.full_video_global_context.independent_test --deepaction --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_deepaction_cross_dataset_v1 --resume
```

全部40视频、九权重评分、评价和report完成才写COMPLETE。结果随后补充，不预填数字。
