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

## 人工审核对应既有预测（事后诊断）

用户附件为完整44条导出JSON，package_id匹配；观看包内未发现独立导出，直接读取附件文件，
不是从聊天叙述重建。原文副本留`derived/v8_deepaction_manual_diagnosis_v1/manual_review_input.json`，
后续导出可放观看包`records/manual_review.json`。按video_id连接、核对SHA、官方标签，T001–T004
审核为空、不纳入测试。40条保留，三组原混淆矩阵及最终video-logit集成复现，不算筛选AUROC。

16条明确异常全部为官方fake：ADAPTED检出4、漏检12。室内、连续、相机稳定/轻微且主体清楚的
9条与用户所列一致：V002/V007/V012/V019/V021/V024/V028/V030/V035，其中漏检7条（除V002/V021）。
12条漏检中Runway/Cog各6；基本稳定6、轻微1、明显运动5；主体清楚10、遮挡/离开2。
因此不支持“漏检仅由场景超约束造成”。WINDOW检出其中8，CELL检出9，既有权重不改。
9条中V002/V028/V030的ADAPTED三seed正负分歧；其余漏检有5条三seed全部负。
V019“不是全生成式”为人工观感原文，官方Cog身份保留。逆光/泛光无逐视频字段，不统计。

仅按编号选V007/V012读取已保存逐帧响应：均49帧、PTS0–6秒，用户异常区间也0–6秒，
无区间外对照，不能判定异常时段相对上升。三seed平均逐帧响应分别在[-6.8605,-5.1730]、
[-6.9489,-5.8533]，全部低于0；不重做视频聚合。没有bbox，未生成定位结论或补图。
详细对应CSV、`diagnosis.md`及98条逐帧描述位于上述独立诊断目录。
两项定向测试、40条连接/标签/SHA/有限数值/集成与混淆检查通过；没有模型前向、训练或前端。
这说明存在条件合适但仍未识别的人工异常，不能据此独断编码、聚合、前端或分布差异的责任。

```bash
.venv/bin/python -m research_tools.v8.full_video_global_context.manual_diagnosis --review <原始人工导出JSON或CSV路径>
```

## 固定9条的已有空间/时间响应图

`derived/v8_deepaction_response_visualization_v1/index.html`提供27张四栏SVG对照图及9张时间曲线。
原画面、ADAPTED/WINDOW/CELL各三seed平均格响应；每视频按人工区间与原生PTS交集均匀选3时刻，
最近实际保存帧，不按峰值选图。编号固定不增删。帧号/PTS/备注及原保存frame/video logit见图。
根据summary.models索引，不硬编码模型轴；格按行优先16×16，用content/area排除padding，
按缓存letterbox实际content_hw/pad_xy裁回原比例。热格最近邻显示，无像素级平滑或相对排名替代。
27图共用绝对色阶[-19,+19]，0中点，蓝负红正；三seed逐帧细线/平均粗线，人工区间明确非GT。

8/9视频的所选三个时刻，ADAPTED所有内容格均为负；V021所选格最大约+0.474。
这是固定时刻结果，不等于整视频没有正响应。V002等的正式视频集成可以为正，而所选格均值为负：
视频先每seed整视频LSE再平均，不等于显示用先平均seed格/帧响应再聚合，不能相互替代。
已查看9条中间固定时刻的缩略对照：多个主体和背景同时呈负色，V021的WINDOW/CELL有较大片暖色；
没有空间bbox，无法可靠判定是否准确对应备注中的手/物体，不能把暖色或相对较高负值称准确定位。
没有定位AP/IoU、重评分或模型前向。原清单/分数/汇总/report hash未变。

CPU生成墙钟8.28s、累计CPU12.08s（含有限多线程），GPU未使用；三项定向映射/模型轴/全负色阶检查。
生成脚本为`response_visualization.py`，不导入训练/前端入口。SVG含内嵌画面，整个输出目录可离线查看。
这是事后定性材料，不替代原40视频结果，不证明因果或精确空间依据。
