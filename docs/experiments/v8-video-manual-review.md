# V8 DeepAction原视频人工观看材料

从干净的`116b0b28f23698c74234393c61a05f935d2d4fbf`创建`v8-video-manual-review`。
仅一个标准库生成入口和静态HTML模板；调用既有PyAV读取编码头部，不解码全片、不运行GPU。
唯一测试清单为`v8_deepaction_cross_dataset_v1/video_manifest.json`的rows。

输出：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_deepaction_manual_review_v1/`。
40/40测试视频、20action、20real/20fake，Runway/Cog各10fake，清单帧数12302；
原文件与复制件SHA全部匹配。稳定video ID编号V001–V040、seed20260909哈希播放顺序，
不按分数/观感排序。40条媒体541124138字节；此前694MB含额外场景审核候选，不全属于测试。
训练比较4视频按冻结训练source排序：Wan1LARL、FCVG1O6PH，独立T001–T004，不计入40条。
训练官方区间从精确source+视频路径匹配的已有官方映射读取，Wan[3.6,10.2]s、FCVG[5.9,14.5]s，
仅来源揭示后可见；没有空间真值。训练样例为原视频及派生版本，DeepAction仅动作语义匹配。

原媒体合计611290167字节（约611MB/583MiB），无软链接，未生成预览。
42个H.264、2个MPEG-4（T002/T004）；浏览器兼容性待用户实际播放确认，不默认批量转码。
如确有不能播放的文件，保留原件，仅针对该编号制作注明“仅供观看”的完整H.264预览，
记录原SHA/预览SHA映射，不进入模型、不改正式输入。

下载整个输出目录到Windows（VSCode远程资源管理器或SFTP，含videos），双击index.html。
不需要ZIP/框架/依赖。默认不展示标签、生成器、路径或模型分数；来源按钮用于第二遍查看。
页面没有模型分数/预测数据，不声称防止主动查看源码或CSV的密码学盲测。
一次一个视频、原比例、自然速度、无自动播放；人工项全部空白，完整看过须用户确认。
CSV/JSON导出并保存本机，支持导入并核对编号/video ID/SHA。浏览器localStorage只是临时备份，
不会自动把人工记录写入服务器；可把导出文件手动放到观看包records/。

备用服务器命令（本轮检查8765空闲，未启动HTTP服务）：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_deepaction_manual_review_v1
```

VSCode SSH Ports转发8765，Windows打开`http://127.0.0.1:8765/`；用完Ctrl+C。
若届时占用，换空闲端口并同步转发。脚本复现命令：

```bash
.venv/bin/python -m research_tools.v8.full_video_global_context.manual_review
```

状态为`MATERIALS_READY_HUMAN_REVIEW_PENDING`，不是人工审核完成。
包内README用一页区分身份错误、场景不匹配、任务分布不同、方法能力不足；四者可并存，
不预设效果不足来自数据。未见明显异常不等于真实，也不覆盖官方标签。
不据人工记录自动删fake或算新AUROC；后续子集分析只能称事后探索，不替代原测试。
没有修改原实验文件/分数、训练/前端/评分/阈值，媒体与人工记录留数据目录，不入Git。
