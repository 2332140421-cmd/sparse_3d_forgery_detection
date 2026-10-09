# V8 全视频连续帧工程 pilot

本轮为工程验收，不训练模型、不计算新的检测性能。独立分支 `v8-full-video-streaming`，数值运行代码提交 `e962aab38b74d6a7d782bd31dede2671ca68941d`。

## 完成范围

冻结清单含272个独立视频；按预声明规则选择训练source 1LARL的一对real/fake。两视频均完整解码914帧，共1828帧；前端和逐帧响应均1828/1828。保留原生PTS，不时间采样。未处理其余270个视频。

冻结读出为既有局部几何pilot的MEAN_REPLICATED、seed 20260909、最后epoch 100。无新增学习参数；hidden跨处理块延续，不在块边界重置。帧和视频采用真实内容面积加权LSE；padding面积为0。追踪使用当前及有限过去帧，不读未来帧；不是逐像素持久跟踪。

处理画布256×256，1920×1080原视频内容缩放为256×144；有效基础格160个。64帧身份一致的旧V8几何缓存复用，其余1764帧新算MoGe；RGB和连续上下文追踪重新计算。旧模型与统计等25个冻结文件内容hash未改变。

## 验证与成本

旧窗口冻结读出复现最大cell logit误差9.54e-7。相同连续64帧的chunk 7/16/31与序列化恢复检查最大cell误差9.54e-7；未来扰动不改变前缀。两级LSE与直接汇总误差1.78e-15。保持2e-5既定容差；检测器关闭TF32以通过原路径与单帧路径比较，前端保持原配置。

完整调用866.12秒；fake/real分别431.96/404.92秒。RTX4090D峰值allocated 1.962GiB、reserved 3.016GiB；新增磁盘约2.081GiB。结果为COMPLETE，范围仅two-video engineering pilot，不是全数据集完成。实际生产运行无中断；恢复能力经过独立保存重载测试，不冒称发生过真实故障恢复。

全部272视频的元数据帧数为234284；按本案例估算约29–31小时、266.7GiB磁盘，测量时可用约103.2GiB。该外推未执行，受视频复杂度和缓存命中影响。

动态component数min/P25/median/P75/max为56/123/138/149/160，最大4基础格；不是固定网格编号变化的替代统计。全内容区域与无前驱区域保留有限输出。几何有效不等于几何准确，局部响应不等于有真值验证的定位。

## 产物与恢复

数据根目录：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_streaming_pilot_v1/`。完整中文报告为`review_summary.md`，另有`report.md`、`final_status.json`、`frame_responses.csv`、`coverage_cost.json`和`validation/`。模型、大数组、帧图与日志不提交Git。

仅在确需恢复同一配置时使用：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
bash research_tools/v8/full_video_streaming/run_all.sh \
  --baseline /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_identityfix_v1 \
  --reference /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_local_geometry_encoding_pilot_v1 \
  --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_streaming_pilot_v1 --resume
```

完成包装器只统一最终progress与JSON布尔状态，不运行模型，不改变数值缓存。服务器重启后确认最终状态已持久化，未重复前端或推理。

本轮结束；没有正式训练、AUROC、定位评价或全量扩展。
