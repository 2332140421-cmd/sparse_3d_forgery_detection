# 冻结方格C的受控时序读出核查

当前帧一致性通过；有序历史未能比较（NO_COMMON_KEYS），没有证据支持或否定扰动得分更高。

|时刻|H|当前格数|两版C最大绝对差|逐格分数最大绝对差|四帧共有键/覆盖率|顺序三项/总分|扰动三项/总分|扰动−顺序|
|---|---|---|---|---|---|---|---|---|
|3|3|735|0|0|0 / 0%|不可比较|不可比较|不可比较|
|6|3|737|0|0|0 / 0%|不可比较|不可比较|不可比较|

## 固定资产和执行

用户人工原话已补记：“运动连续可变；只改变局部”。本组已人工确认；不代表三维异常真值。

仅源帧0–6的无损RGB输入，每帧MoGe一次、RGB编码与原池化一次，共7帧；两版按清单引用同一份缓存。没有使用alpha、GT、差分、版本标签作为模型输入。背景和对象是程序生成，不称自然视频。

编码权重为pvst_rgb_partition_anakin_dev_v1/RGB_DEPTH_PARTITION/last.pt，condition=RGB_DEPTH_PARTITION、epoch12/576；读出为pvst_slab_grid_localization_v1/SLAB_GRID_CURRENT/final.pt，epoch12/576。严格state_dict加载，原配置、manifest、checkpoint身份及SHA核对，完整路径和哈希见identity.json。原BF16编码/池化、FP32 C、FP32当前关系读出保持，模型参数/缓冲未变。

无已有兼容几何，新增7帧原MoGe-2-vits-normal推理，num_tokens及输入大小由原geometry_shape计算，force_projection=True/apply_mask=False/use_fp16=True；固定训练尺度1.9888148307800293，h=.125、d0=0；原生360×640、L=23。provider revision和权重SHA详见identity.json。没有逐帧重新归一化或拟合深度校正。

## 共有键缺口

|源帧|格数|深度片号范围|有效像素/230400|
|0|636|40–46|230391|
|1|710|39–47|230395|
|2|741|39–49|230392|
|3|735|46–55|230382|
|4|819|56–69|230370|
|5|802|53–65|230064|
|6|737|51–60|230380|

主窗口源3,4,5,6：源3深度号46–55，源4为56–69，连深度号交集都没有，因此四帧完整(绝对深度号,格行,格列)交集为空。对照窗口源0,1,2,3的完整键交集也为空。零范数额外剔除数均为0，阻塞发生在对齐之前。每帧共有键比例和像素覆盖率均为0%，详见summary.json。不能改用格号近邻、最近深度或每帧重编号补齐。

固定计算定义：四帧共有完整键，剔除任一帧零范数键；相邻帧逐键余弦距离均值，三项求和。这里只将CPU距离归约使用float64，未改变模型精度。无共有键时三项、总分、差值均为null，不填0。t=3的源帧/输入确实逐帧相同，但不存在可定义的历史分数，不能用预期零差值代填。

## 状态与判断

首次提取全部7帧后，因为诊断入口错误地对不可比较的null差值断言等于0，进程真实退出1。error.log、first_failure_status.json及execution.log保留。随后只修正无共有键状态记录和从已完成缓存汇总的薄入口，未改变特征、对齐口径或容差，未重复MoGe/编码/池化。汇总进程退出0，COMPLETE表示核查交付完成，不表示历史比较通过。首轮运行没有在异常前保存阶段计时/全程GPU峰值，因此这些数值未记录，不以汇总峰值冒充全程峰值。

1项针对性检查通过：完整键对齐、零范数剔除、同序结果一致、交换改变固定测试向量总距离、无共有键返回null。未拟合阈值或训练。

下一步优先解决本受控样例在原逐帧几何下的键可比性，再谈有序历史对照。当前结果不能说明C丢失了顺序信息，也不能说明历史读出有效；自然伪造收益、学习式时序模型及深度独立贡献均未检验。本轮不实施任何几何修正或追加样例。

## 复跑与产物

入口：PYTHONPATH=<原rgbd-packages>:. <v8 Python> -m research_tools.rgbd_slice.synthetic_temporal_readout

已有完整结果拒绝重复提取；仅复核已有缓存并汇总用--cached-finalize。缓存几何身份、RGB哈希、分组ids/counts核对后读取C。每帧geometry_XX.pt/features_XX.pt及身份只在独立产物目录，不加入Git。

结果目录：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_synthetic_temporal_readout_v1

执行提取HEAD b4d9e0f86dcfbfd9fb9324a8a3a350afdbc98079；汇总HEAD f6c753e3553577be394a35b47ed6f8d94b50737b。计算与发布分开。无本任务残留进程，其他用户GPU任务未干预。
