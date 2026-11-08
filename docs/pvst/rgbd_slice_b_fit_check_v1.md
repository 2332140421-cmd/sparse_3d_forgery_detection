# 完整B固定pair拟合诊断

状态：准备独立运行；还没有本次B拟合结论。

只补齐原A-only拟合检查的证据缺口。入口`python -m research_tools.rgbd_slice.b_fit_check`，复用make(B)、optimizer、update、evaluate和grad_info，不改变正式训练默认行为。
固定原an:7ogg04，fake cb41b3d004b6640f6c3b5bb1、real c26e3fc1fdc6aabda308ae84，两段均[0,24)、实际PTS0–0.92秒，fake→real交替。
使用原共同初始化和高分辨率几何；正式目录只读，不加载正式best/last或拟合A权重，不重提几何/预训练特征。
150次更新/约2700秒、每10次评价；主目标Dice≥.8且real FPR≤.05；次级Dice/召回均≥.6且FPR≤.05。模型、loss、学习率、BN与Dropout均保持。

2项必要身份/完整B构造路由检查通过。独立目录`derived/pvst_rgbd_slice_b_fit_check_v1`保存选段/PTS、配置、梯度与残差标量、曲线、样例、真实退出码和报告。诊断日志不作为新的论文成绩。

正式history初读：B监测Dice第11轮.7654→第12轮.1263，第12轮恰为开发空间best；B历史开发frame AUROC最高在第6轮.6539，而非空间best第12轮.4717。只读汇总将保留全部相关轮次，不能用检测最佳轮次替换正式checkpoint。监测只覆盖一个pair的两条训练片段。

完成后本页由独立入口更新并普通commit/push，本轮不启动重训或扩量。
