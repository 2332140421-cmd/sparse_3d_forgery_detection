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

冻结清单已完成执行与验收，最终状态COMPLETE、退出码0。复现/恢复命令（已有任务运行时勿重复启动）：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
export PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$PWD
.venv/bin/python -u -m research_tools.v8.full_video_global_context.independent_test --deepaction --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_deepaction_cross_dataset_v1 --resume
```

## 实际结果与验收

40/40视频、20动作类别、12302/12302原生帧（real9252/fake3050），没有失败、抽帧或截断。
九权重各覆盖全部40视频；360条seed分数+120条集成分数，键唯一、有限值、三seed
平均video logit核对通过。每视频解码/前端/每模型响应帧数一致；帧数与metadata无差异。
两帧九权重加载路径检查最大logit差2.521e-8。运行前6项定向检查及收尾新增下载缺失语义测试通过，无训练或统计重拟合。

三seed各完成整视频面积LSE后平均logit；阈值0，n_real=n_fake=20：

|条件|AUROC|AP|FPR|Recall|Balanced ACC|Precision|F1|ACC|TN/FP/FN/TP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|COMPONENT_ADAPTED|0.7400|0.745633|0.05|0.30|0.625|0.857143|0.444444|0.625|19/1/14/6|
|COMPONENT_WINDOW|0.6625|0.703869|0.60|0.70|0.550|0.538462|0.608696|0.550|8/12/6/14|
|CELL_WINDOW|0.7250|0.659867|0.40|0.75|0.675|0.652174|0.697674|0.675|12/8/5/15|

ADAPTED自身action-cluster95%CI为[0.585,0.875]。主要预声明配对：
ADAPTED−COMPONENT_WINDOW=+0.0775，CI[-0.035,0.2075]；
COMPONENT_WINDOW−CELL_WINDOW=-0.0625，CI[-0.2700,0.1375]。
均为10000次相同类别抽样、seed20260909；不是同源source bootstrap。

|seed|ADAPTED AUROC|COMPONENT_WINDOW AUROC|CELL_WINDOW AUROC|适配−窗口|组件窗口−CELL|
|---|---:|---:|---:|---:|---:|
|20260909|0.4950|0.5675|0.6825|-0.0725|-0.1150|
|20260910|0.3200|0.4575|0.7875|-0.1375|-0.3300|
|20260911|0.7650|0.8125|0.6575|-0.0475|+0.1550|

集成适配差为正，但三个单seed适配AUROC均下降；集成AUROC不是单seed AUROC均值。
相对窗口模型FP从12降到1，同时FN从6增到14，Recall下降0.40；不能只凭低误报称成功。
RunwayML子集ADAPTED AUROC0.89，CogVideoX5B子集0.60，每项只含10fake及对应10动作real，
仅描述，动作子集与生成器同时不同，不能认定差异纯粹来自生成器。

真实FPS23.976–30（中位25）、时长8.16–39.24s（中位14.7646）、204–981帧；
fake分别8–24（中位16）、6.125–10.6667s（中位8.3958）、49–256帧。
原分辨率(H×W)：real3840×2160(6)、2160×4096(3)、1440×2732(2)、1080×1920(2)、
2160×3840(5)、1920×1080(2)；fake Runway768×1280(10)、Cog480×720(10)。
这些类别相关的数据域差异仅披露，不据此重新采样/调参。动作簇也不保证人物/场地独立。

正式执行5293.74s（88.23min），其中视频处理5259.13s、评价31.27s；模型组CUDA时间
ADAPTED225.06s、COMPONENT_WINDOW209.03s、CELL208.30s（包含在总执行中，不能重复相加）。
前端几何429.61s、跟踪3218.22s、RGB35.58s、紧凑特征368.23s。
CUDA峰值allocated2.119GB/reserved3.217GB；实验输出约0.668GB，媒体0.694GB，
合计约1.36GB另加小型metadata。下载已记录逐文件尝试1325.60s，并行重叠且一条失败耗时
缺失，不能当总墙钟；完整媒体字节694414010，额外失败流量未知。总下载秒数/流量字段
置null，避免沿用本地source入口的0默认值。无关模型/评分数值未改动。

**判断：** 当前主方法可作为受限跨数据集迁移基线保留，但尚无稳定适配增益证据；
集成排序保留信号，seed不稳且固定阈值漏报明显，不宜宣称解决跨数据集泛化。
动态component在相同训练阶段也未显示稳定优于CELL。无定位真值，不支持空间定位结论。
如考虑进一步外部验证，应先正视本次seed与数据域差异，而不是自动扩测或训练；本轮已停止。

完整报告、指标、分数、成本和冻结清单在上述独立输出目录；执行源码为1d320f7，
后续改动仅报告与下载成本缺失语义，不改变已执行模型路径。独立screen已退出，无残留实验进程。
