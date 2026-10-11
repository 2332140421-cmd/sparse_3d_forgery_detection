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
