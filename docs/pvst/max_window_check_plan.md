# PVST v0.3 最大窗口单次900秒资源检查

2026-11-04用户明确授权：只在独立资源入口绕过旧fine/coarse计数预算，不改变正式v0.3入口、协议或XYZ=null。
沿用ADR-0030模型/损失，不增加方法变化或正式训练；本次1/8仅为候选检查。

唯一XYZ_PVST窗口：782426ea0330c7fea684953c，ANAKIN real，an:4wdo37，720×1280；历史[104,120)、目标[120,128)。
统一局部XYZ h=1/8、coarse2h，seed20260909、BF16、attention512/QKV8192/FFN8192/point32768，CPU卸载仅现有外层时空checkpoint边界。
复用实际官方初始化、Video、PVST、AdamW参数组和v03.clip_terms。逐目标帧frame BCE均值及真实AN real全零空间GT的pixel BCE/原softDice均值；不使用完整视频分母，不计算方法分数。
先核对GPU重新加载fine2,970,629/coarse1,488,042及实际PTS；不一致即FAILED，绝不换样本。

现有资源worker有两次更新和多微批，本轮需要精确一次更新及父进程截止，因此新增research_tools/pvst/max_window_check.py薄入口，直接供本轮CLI调用，输出独立阶段记录、结果/报告和坐标预览。复用已有模型、数据、损失和trace，不复制训练循环或改模型。

父进程Popen(start_new_session=True)，自启动计900秒；每约5秒采GPU信息。截止仅SIGTERM本worker进程组，等待最多5秒，再SIGKILL同组并wait；不操作共享GPU其他PID。
PASS/OOM/TIMEOUT/FAILED分开，保留真实returncode/信号、最后阶段与采样峰值；不自动重试。CUDA stage边界同步计时，节点内无新增同步/写文件；独立线程每5秒仅采计数器与既有shape trace，便于超时保留证据。
一次Adam更新后记录optimizer状态与关键梯度存在性/有限性/范数。首步零初始化的上游零梯度不判失败；一次更新不代表完整学习路径验证。权重丢弃，不保存正式checkpoint。

坐标统计在GPU结束后逐帧流式读取相同24帧；占据成员数优先复用本次GPU路由的计数/父子关系，仅加载未完成时才补缺失帧计数。
XYZ范围/有效数精确；分位数固定seed每帧至多8192点、明确近似。每帧和24帧累加分别记录（不混为跨时空间去重集合）。
固定120/124/127原图、统一logZ显示、抽样fine中心XY投影与每块点数图；显示裁剪不改变模型输入。不加人体分割，不把近处当人体。

只测试父进程限时/进程组隔离、原退出码、singleton/成员合计等直接合同，并运行已有PVST相关回归。
代码提交后才启动，运行期间不改源码。工作区中用户已有无关历史检测结果JSON改动保留，不提交；“干净源码”指本任务执行源码及配置均已提交并有SHA身份，未掩盖整个工作区有用户改动。

```bash
PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages:. \
TORCH_HOME=/media/njupt/8FCD3869E2FAB06F/chenyh/cache/torch \
XFORMERS_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.pvst.max_window_check
```

输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_v03_max_window_check_v1/`。
包括identity、launch、stages/live、worker_result、result/small、GPU samples、coordinates、index.html、报告和publication。原目录只读；同身份已结束结果复用，无结果的历史launch要求核查，不自动再运行。
后台入口独立完成超时、CPU坐标审计、报告和结果文档普通commit/push；发布失败不改变计算结果。本轮结束后不追加资源检查或训练。
