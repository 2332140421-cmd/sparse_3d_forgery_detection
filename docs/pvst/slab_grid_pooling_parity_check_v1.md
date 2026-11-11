# an:4z1827/fake冻结整片特征复现诊断

**诊断COMPLETE，真实退出码0；72/72帧均未复现超差。** 没有首个失败帧，不将任何最坏误差帧冒称原进程的失败帧。原实验仍FAILED/退出1，不能宣称已修复，暂不进入资源验收。

## 输入、顺序及三方比较

唯一视频`03940cfef98b2a38304cdd0e`，`an:4z1827/fake/train`，原生540×960。
沿冻结片段[0,24)、[48,72)、[96,120)，按原提取顺序升序检查全部72帧。每帧局部视觉编码一次，合计72次；同一个GPU local张量供A/B使用。

- A：现有SlabPool原实现，eval/no_grad，原BF16 autocast，输出FP32。
- B：现有frozen_pool分块实现，仅配置原整片归属，8192像素分块；未改变代码或归约规则。
- C：pvst_slice_relation_readout_v1/cache/该视频.pt，核对cache_identity.json中的SHA后只读映射，未重算。

帧身份、实际PTS、绝对片号顺序、张量形状、valid像素计数与官方GT投影计数均逐帧匹配。核对冻结媒体、GT、高分辨率几何、manifest、权重及相关源码hash；没有使用其他视频作输入。

|比较|检查帧|累计元素|超差元素|最大绝对误差|最坏帧/PTS(s)|片号/通道|该元素允许误差|
|---|---|---|---|---|---|---|---|
|A−B|72|1602048|0|6.19888305664e-05|63 / 2.102100000|9 / 423|0.000225931568146|
|A−C|72|1602048|0|8.48770141602e-05|51 / 1.701700000|8 / 423|0.000226121110916|
|B−C|72|1602048|0|0.000141620635986|51 / 1.701700000|8 / 423|0.000226121110916|

上表列的是每种比较最大绝对误差所在元素。实际判定使用全部元素的原始`torch.isclose(atol=2e-5,rtol=3e-5)`，允许误差为`2e-5+3e-5*abs(reference)`。因此绝对差大于2e-5不自动表示超差。各帧详细计数及最坏元素见comparisons.csv和scan.jsonl，没有只取一个最大值代替全量核验。

|比较|实际值|参考值|有符号差值|
|---|---|---|---|
|A−B|6.86432361603|6.86438560486|-6.19888305664e-05|
|A−C|6.87061882019|6.8707036972|-8.48770141602e-05|
|B−C|6.87056207657|6.8707036972|-0.000141620635986|

## 归因结论与放行判断

本次同local的新旧池化A−B全部通过，A−C与B−C也全部通过。未触发用户规定的失败分支，因此没有阶段采集、额外旧路径复跑或额外编码；没有扫描其他视频。

**原超差原因仍未确认。** 本次B只配置整片归属，且没有重放首次整队列提取在前1008帧后的进程状态；未复现不能证明原失败不存在，也不能直接归因于BF16、编码不确定性或浮点归约。首次记录的87/23552元素超差、最大绝对差0.0003304481506347656及退出码1仍原样保留。

本轮不能据此放行资源验收：只说明这72帧在当前限定三方对照中满足原容差，没有修复或解释首次超差。后续如继续定位，应先取得可复现的原失败帧及调用环境证据；本轮停止，不重复抽取、不改池化、不补提缓存、不运行最大窗口或训练。

## 保护、测试与成本

4项必要小图测试通过：原isclose容差与片号/通道记录、全元素计数、分块partial维度映射、只读实际池化阶段采集。测试命令：`PYTHONPATH=<rgbd-packages>:. <v8-python> -m unittest discover -s tests -p test_pooling_parity_check.py`。正式model.py/grid_localization.py及验收口径未修改。

冻结模型参数和缓冲区前后state hash相同、无梯度；原权重SHA一致。首次失败日志/状态/退出码文件SHA一致，14份1008帧缓存的文件清单、大小及mtime前后相同；本脚本只向新目录写入诊断证据。原失败记录不被本次COMPLETE覆盖。

墙钟7.848秒；扫描阶段GPU峰值allocated/reserved 0.301/0.381GiB（计数在模型准备后重置，不包括模型构造峰值）；进程峰值RSS 2.000GiB。资源验收0次、优化器更新0次、额外旧路径复跑0次，原模型源代码与权重不变。

执行HEAD `26779f6aa9d28b30545b8db01d178b06bd03998d`；基准5c42f4be512f440e8bb24b3a1db0d27a7737a002；分支pvst-slab-grid-localization。

独立产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_slab_grid_pooling_parity_check_v1/`：report.md、summary.json、identity.json、selected_identity.json、original_preservation.json、scan.jsonl、comparisons.csv、tests.json、execution.log、exit_code.txt及status.json。没有失败帧，故不生成first_failure.json或attribution.json；不落盘稠密local、像素权重或新特征缓存。Git发布与计算状态分列publication.json。
