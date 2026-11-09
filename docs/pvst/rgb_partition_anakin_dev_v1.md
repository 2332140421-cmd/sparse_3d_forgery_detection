# ANAKIN来源隔离RGB分片对照

执行准备完成，尚无本实验指标。ADR-0038；基于725486ffd065cb0ec9867d6ba2b282bb943bd1d0。
原8/4 pair、48段、12轮×48更新，RGBD与RGB_DEPTH_PARTITION顺序执行，分别从相同初始化开始。
主比较epoch12，best单列，不早停；每组4小时。训练/开发来源、视频及高分辨率缓存将逐项冻结核对。
已有RGB-only隔离等4项测试及新协议3项测试通过；真实输入前向在启动器中预检，通过才训练。

独立产物：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1/`

入口：`python -u -m research_tools.rgbd_slice.rgb_partition_anakin_dev`。
父进程自动管理两worker时限、保存真实退出码，最终报告和文档提交推送；无自动重试。
计算状态见独立status.json及分条件progress.json，发布状态另见publication.json。
