# ADR-0038：ANAKIN来源隔离RGB-D／仅深度分片匹配开发对照

状态：已接受，2026-11-09用户授权。基于725486ffd065cb0ec9867d6ba2b282bb943bd1d0，独立分支pvst-rgb-partition-anakin-dev。

问题：ADR-0037中去掉片内真实深度／XYZ的定位优势能否延续至未参与训练的ANAKIN来源？
仅本新实验扩展ADR-0037两对拟合的样本及预算；旧入口、像素消息干预、权重与结果不变。

复用pvst_rgbd_slice_relations_v1的原train8/dev4 pair、48个首中尾24帧片段、共同初始化与高分辨率geometry。
以source_id、原URL规范化和媒体SHA核对两侧隔离；完整媒体帧数／尺寸／实际PTS、官方GT和缓存身份逐一验证，出错不替换。
训练帧次48×24=1152/轮，独立帧按清单并集另记；开发8个完整视频，实际帧数重新核对。
不访问保留测试，不调用前端、不下载权重。

两组RGBDSetSlices完全沿用ADR-0037，像素消息开启，不增加网络。
RGBD读真实深度及XYZ；RGB_DEPTH_PARTITION相同通道固定0，真实深度仅决定valid、分片及绝对片号位置编码。
同共同初始化与seed20260909，逐张量hash一致；每轮seed+epoch相同顺序；各固定12轮×48更新=576。
AdamW编码器1e-5/其他1e-4，wd.01，clip1，固定编码器BN、原Dropout、BF16及完整24帧梯度；frame BCE+pixel BCE+softDice等权。

epoch0及每轮完整开发评价复用原24帧因果窗口与尾段左上下文；每原生帧恰好评分一次，阈值logit≥0。
主比较last epoch12；best只在epoch1–12按fake视频宏Dice选择，严格大于才替换，平分取早，不早停。
pixel AP最终last及best计算，保持正GT帧→fake视频两级平均；辅助检测不选权重，FP32帧BCE独立从保存scores/labels计算。

同一GPU顺序执行，两独立worker各训练及评价最多4小时；父进程只终止本worker进程组。
OOM、非有限值、身份错误及时限不自动重试或降级；最近完整epoch、部分更新及真实退出码保留。
任一组未完成不报主对照胜负。计算完成与Git发布分开记录。

新增薄入口research_tools.rgbd_slice.rgb_partition_anakin_dev：调用现有build/update_once/load_clip/evaluate，负责冻结、固定循环和报告。
直接输出独立derived/pvst_rgb_partition_anakin_dev_v1；数据权重不提交。针对性测试保护隔离、覆盖及排序，不修改旧模型实现。
仅小规模开发证据，不作正式测试或深度分片自身增益结论；本轮结束不追加条件或训练。
