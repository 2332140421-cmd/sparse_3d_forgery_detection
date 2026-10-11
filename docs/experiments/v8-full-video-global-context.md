# V8 全视频全帧关系冻结评价

本轮是**窗口训练模型的全视频连续帧冻结评价**，不是全视频训练。
合同ADR-0017；起始HEAD abce88ae440d13509d46ff272adbd9fbe149b0fd。

独立分支v8-full-video-global-context，冻结六个GLOBAL_COMPONENT/GLOBAL_CELL epoch100
模型。272视频/136 source、预计234284元数据帧；实际数量以顺序解码EOF为准。
validation 14 source；主指标video pooled AUROC，source配对bootstrap10000/20260909，
阈值0。每seed独立完成视频面积LSE后平均video logit，不平均cell再LSE。

64帧NPZ无损保存float32格输入/mu/关联/状态及packbits原像素mask；不持久化新dense
XYZ/RGB/depth。调用原GlobalContextModel.spatial/region_context，因果GRU跨块延续。
旧两视频合法输入只读复用，其他视频共享一次前端供六模型评分。

输出：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_global_context_v1/`。
`progress.json`及`final_status.json`记录真实状态，仅全量评分/评价/报告完成后COMPLETE。
`window_auxiliary_bootstrap.json`补充旧窗口辅助指标，不修改旧macro主结论。

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
bash research_tools/v8/full_video_global_context/run_all.sh --resume \
 --baseline /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_identityfix_v1 \
 --reference /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_local_geometry_encoding_pilot_v1 \
 --models /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_component_global_context_pilot_v1 \
 --legacy /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_streaming_pilot_v1 \
 --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_global_context_v1
```

运行锁防重复；TERM在当前块原子保存后停止。恢复核对输入/模型/统计hash、视频身份与
PTS，不重算合法块；不使用V7数值资产。完整结果和实测资源由数据目录report.md生成，
本页不预填未完成的指标。

## 本轮实际验收

11项直接相关测试通过。两条件冻结权重原16时刻forward/连续接口比较最大cell误差
1.90735e-6；真实原生连续64帧、chunk7/16/31、序列化恢复、未来扰动、面积LSE及
紧凑缓存重建均通过既有atol/rtol=2e-5。真实attention调用已核验。
1LARL两视频1828帧六模型完整评分；紧凑输入84,331,303 bytes，约46,133 bytes/帧。
按该样本估算全队列输入及响应约12.78GB（11.91GiB），不是最终实测。
已继续00ZCA新增真实前端，未在两视频阶段停止；全部结果仍在运行，不预填最终指标。

上一轮窗口辅助source配对bootstrap（10000/20260909）补充：

| 指标差 COMPONENT−CELL | 点估计 | 95% CI |
|---|---:|---|
| pooled AUROC | +0.134727 | [-0.006240, 0.248548] |
| AP | +0.197246 | [0.029463, 0.293576] |
| F1 | +0.137769 | [0.031254, 0.252348] |
| Accuracy | +0.156627 | [0.047619, 0.273810] |

这些是旧窗口辅助结果，不替换原macro主比较−0.007937。新视频结果另算。
run_all成功后仅发布本轮小型报告：核对272视频/六模型/帧数与冻结资产不变，保护
用户工作区及暂存区，仅暂存指定实验文档、普通commit/push。失败不重启计算；
交付状态另存git_delivery.json，不能用代码已推送冒充实验完成。
