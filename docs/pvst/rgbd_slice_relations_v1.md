# 预训练RGB-D编码与深度切片关系匹配实验

状态：PREPARING；尚无新训练成绩。分支pvst-rgbd-slice-relations，从0570938aa600072a0cc780a897dc8f8eedef51c3开始。

[ADR-0034](../decisions/0034-rgbd-slice-relations.md)与[配置](rgbd_slice_config_v1.json)定义两个条件。两者均使用深度；比较切片时空分支的增量，不是纯二维/三维。

|条件|总参数|可训练参数|新增模块|
|---|---:|---:|---|
|RGBD_LOCAL|20,402,578|20,402,578|共同DFormer-S、多尺度256局部融合、原生卷积细化与独立帧头|
|RGBD_SLICE_RELATION|33,264,311|33,264,311|4×128学习式片池化、两组512维8head时间→深度显式关系及零初始化残差|

预训练：官方原版DFormer-S，源码e3273009b759b578945483828ff315d560be94c9。
官方权重229183347字节，SHA256 8a086a3fb4e7ebda0806172938dd199bb7d8394593a8176640ed79735e35a0c0，state_dict epoch297。
保留编码器530键strict加载，RGB/depth stem各35键、交互270键，无missing/unexpected。分类头及末端仅供分类的深度更新按官方分割结构不用，前向四尺度一致性检查已通过。

5项本轮CPU检查＋6项直接相关既有检查通过：原生全点学习式pool与稠密参考输出/梯度一致（含8192尾块）、官方分类尾部不影响四尺度输出、零残差共同输出一致、BN统计固定/仿射可训练、BF16两步上游关系与编码器梯度；负深度/边界/空片与因果检查沿用既有测试。
依赖采用隔离rgbd-packages中的mmcv-lite2.1.0/mmengine0.10.7；未升级PyTorch、驱动或CUDA，不使用其可选deformable算子。

沿用ANAKIN训练8pair/16视频/2138帧（1065正帧）与开发4pair/8视频/924帧（462正帧），48条固定24帧训练片段。没有场景过滤或新数据下载。
新几何仅为这些视频：MoGe-2输入完整画面长边≤768、小视频不放大；num_tokens=round(input_h*input_w/14²)，记录实际内部token网格和输出尺寸。不把旧256场放大冒充新观测。

独立入口自动运行：几何缺项准备→B真实24帧两次更新/初始化匹配→A首pair小样本拟合（150更新/45分钟上限）→丢弃检查权重→共同初始权重重新训练两条件→同best完整开发评价→中文报告/普通commit/push。
拟合目标Dice.8且real FPR≤.05；达到预算但Dice/GT召回均≥.6且FPR≤.05时单列明确拟合通过，否则FIT_BLOCKED并停止正式训练。判据预先固定，不按开发结果调参。

运行目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/`。
实时状态见progress.json与run.log；完成后report.md、result_summary.json、各条件history/best_dev/samples及publication.json。
当前没有best/结束轮次或新开发指标，不将必要检查当作检测效果。相机运动/深度漂移与有限因果窗口的局限保留，旧实验不改。
