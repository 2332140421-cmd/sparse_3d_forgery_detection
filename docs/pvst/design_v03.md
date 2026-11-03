# PVST v0.3 设计与单轮贯通协议

2026-11-03；依据[ADR-0030](../decisions/0030-pvst-v03-voxel-scale-clip-updates.md)。
起点29205c78c85ad5a3e23237b7cf8aae4690fbbc70，新分支pvst-point-voxel-v03。
旧v0.2及其三次失败、紧凑执行失败记录只读保留。

## 方法变化与保持项

仅改变XYZ结构尺度和全视频→片段更新，不称等价优化。XYZ候选1/32、1/16、1/8，训练窗口fine最大≤200,000、coarse最大≤50,000的最细候选冻结为全队列同一值。没有候选则停止。不用开发标签/分数选尺度。UV维持1/32。
配置voxel_sizes由geometry.voxel_size统一供Video和PVST使用，fine中心/9维相对坐标/coarse父子关系来自同一h，元数据含h，不允许尺度不一致；旧无此配置时仍为UV1/32、XYZ1/64。

全部原生像素有输出，全部有效几何点参与geo128与VFE；每个点恰好归一个fine块。点局部表示g与视觉a、所属时空块z输入原像素解码器；无效几何点仍有视觉路径。fine256/coarse512、双尺度各三层、全局context、真实PTS、候选邻接与softmax范围保持。不丢点、不设块数截断、不减宽度、8目标及16历史不变。
统一调粗会降低块级结构分辨率；保留点旁路不能保证效果不下降。XYZ来源仍dense256实际场裁padding/回映原生，不把插值称独立高分辨率观测。

norm+QKV投影整体按8192节点执行并checkpoint，FP32归一化仅在块内；参数、epsilon、state_dict键保持。沿用紧凑query512、FFN8192及块内残差；CPU仅在GPU失败后作外层checkpoint卸载后备。不重建已完成的拓扑优化或新稀疏平台。

## 训练与评价

旧manifest/scale原样复制或引用：48视频22,074帧，训练32/开发16，source隔离不变。RGB/冻结无LoRA448 tokens来自旧cache_root；XYZ/valid仍读取manifest指定dense_source；run/output只存v0.3产物。缺缓存则报错，不提取/下载。几何路由只留当前窗，身份含h，不保存可学习特征跨更新复用。

每训练视频最多4个真实不同stride8目标区间，尾段允许不足8帧。按稳定video-id顺序、seed20260909：real和AN fake按时间排序分层随机选；AF fake优先2个含修改段+2个纯正常段，不足从剩余真实候选分层补足，记录fallback。AN逐帧标签取实际mask是否非空。保存source、范围、输入实际PTS、标签来源及两条件共享顺序。不按预测选片段。

AF/AN片段分别shuffle配对，一次更新依次前向/反向，对参与片段数取平均，不同时保留两图；不足按真实段数均值。片段内frame BCE目标帧均值；AN在每帧原生全画面算pixel BCE和原epsilon=1 softDice再目标帧均值，AF无空间标签。历史不监督，无旧完整视频分母、类别/source额外权重。

官方decoder/公共初始化、固定BN、BF16、学习率1e-5/1e-4、AdamW wd1e-4、clip_norm1保持。资源模型丢弃，正式从规定seed初始化。两条件各一轮，不选best不判断收敛。每更新checkpoint含model/optimizer/RNG/片段顺序/完成位置，协议严格匹配才能恢复；更新日志记录三项loss、学习率、梯度、时间、峰值、片段/目标帧数。

epoch0/epoch1都完整遍历开发16视频及固定训练4视频（每库按source排序首对），输出逐帧响应和既有标准视频/帧/AN空间指标，不读测试集、不调阈值。每行同checkpoint；monitor不冒充全训练集。固定样例为每库开发fake/real的首个manifest身份，25/50/75%实际PTS，统一概率0–1，不按输出挑图。

## 顺序与停止条件

必要测试→训练坐标筛尺度→冻结后开发坐标计数→最大节点完整历史/旧最大视频/高原生像素代表去重≤3→两条件真实损失、累积梯度、2次Adam更新及eval→重新初始化两条件各一轮→完整评价→中文报告/普通push。
已通过的case按严格身份复用，不因一个失败重测全部。若当前代表OOM，仅该代表尝试CPU后备；仍失败停止一轮训练。计数无候选停止，不擅自扩大尺度搜索。
已有测试覆盖邻接/因果/PTS/BN/重载等；新增测试只补尺度贯通、真实点旁路梯度、norm+QKV输出/梯度、片段选择与监督归一化。资源probe不是正式训练，输出无权重。

## 独立运行

源码先commit及push，运行期间不改源码。仓库根目录：

```bash
PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages:. \
TORCH_HOME=/media/njupt/8FCD3869E2FAB06F/chenyh/cache/torch \
XFORMERS_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.pvst.v03
```

输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_point_voxel_v03/`。
progress记录计算，publication单列发布，exit_code保留真实退出；run.log为后台stdout。独立入口自动依序完成，不依赖交互轮询。
最终报告results_v03.md/JSON由脚本生成并普通提交推送，大型缓存/权重/图片留数据目录。若结果仍运行如实报RUNNING，不预填成绩，不自动30轮。
