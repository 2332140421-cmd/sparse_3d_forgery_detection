# V8 完整视频弱监督微调

分支`v8-full-video-training`；起点`211fba5be2b22c6e4b7524ef509a22060f89967e`。
合同见`docs/v8/full_video_training_contract.md`及ADR-0018。

## 协议与实现

两阶段方法：匹配condition/seed的窗口epoch100预训练→完整视频epoch20微调。
GLOBAL_COMPONENT/GLOBAL_CELL各三seed；319969参数，真实全帧attention与多前驱GRU均保留。
冻结的244训练视频/122source、28验证视频/14source，共234284实际连续帧。
无前端重跑，无标准化/分组重拟合，无帧或块标签。完整视频面积LSE后一次BCE，
同source real/fake各半权重，统一clip/step。checkpoint跨块梯度不截断。

新增`model.py`为可微适配，不能复用inference_mode接口；`data.py`只读并核验
原紧凑缓存；`run.py`完成必要资源测试、原分数复现、训练恢复、评价及报告。
未引入通用训练框架；旧方法和结果不变。

## 已实测资源

完整1LARL real/fake各914帧；最长训练视频X7R9N fake为1829帧。
二者均完成完整前向/反向，未截断梯度或缩短输入。

|执行块|1LARL pair读入+前向+反向秒|最大allocated GiB|最大reserved GiB|
|---|---:|---:|---:|
|32|8.6406704504|0.944141865|1.214843750|
|64|7.6019999869|1.853305340|2.111328125|

资源指标选择chunk64；按该pair粗估六模型×20epochs约30.91小时，
不是已完成训练成本，也未计最终评价。各视频时长不同，实际成本由模型记录给出。
GPU为RTX4090D；FP32，未启用AMP或TF32。

## 验证与运行状态

定向测试检查两条件/多chunk冻结推理一致、padding隔离、缺失内容有限输出、
checkpoint开关loss/梯度一致、跨块hidden梯度、直接LSE和optimizer/RNG/order重载。
真实28验证视频六模型分数复现记录在输出目录，完成后才进入正式训练。
正式模型与smoke分离；最终指标以运行产物为准，本文不预填结果。

2026-10-13 验收：5项定向测试通过。28验证视频×6模型=168分数及ensemble复现通过，
最大绝对误差`1.8516453089034712e-5`，固定容差`2e-5`。首个正式source pair
完整反向与optimizer更新后已原子保存，next_pair_index=1，optimizer有35个参数状态。
安全TERM后退出；随后用同一恢复入口继续，不丢弃或重复该pair。
实际训练循环的定向恢复测试确认分段恢复与不中断的最终参数及loss一致。

当前为RUNNING交付记录，不是6/6完成报告。完整六模型训练、评分及CI仍需实际执行，
不能将资源smoke或168分数复现算作本轮微调结果。

独立启动（只能存在一个任务，内部运行锁防重）：

```bash
screen -dmS v8_full_video_training bash -c 'cd /root/autodl-tmp/projects/sparse_3d_forgery_detection && exec bash scripts/run_v8_full_video_training.sh >> /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_v1/run.log 2>&1'
```

安全停止向`progress.json`中本轮真实PID发送TERM，完成当前source pair并原子保存后退出。
恢复执行同一脚本。恢复模型/optimizer/RNG/source顺序和pair索引，不重做已提交pair。
本服务器已验证screen独立运行，恢复日志从pair=2开始，并继续推进到pair=3。
未采用未能存活的nohup启动。查看：`screen -ls`及输出目录`progress.json`/`run.log`。

产物根：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_v1/`。
`final_status.json`才表示整轮状态；`report.md`及`evaluation/summary.json`由六模型及
全部训练/验证评分完成后生成。主比较pooled video AUROC；辅助微调−冻结。
source配对bootstrap10000次，seed20260909；阈值0；三seed先各自全视频LSE再均值。

仅视频级弱监督开发评价。未验证帧/像素定位、独立source泛化；不会自动追加实验。
