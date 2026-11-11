# 局部编辑三维关系候选：下载与人工预览

状态：固定规则后执行资产准备，不训练、不推理、不提取深度，不修改正式划分或分数。
起点6ba9597b0e5f0c5e752b1da01da631bd5102c67d，独立分支pvst-geometry-dataset-screen。

GRIP只取02 Fast Car、04 Rolling Can、05 Falling Can、10 Tennis Ball、06 Walnuts、14 Wall Frame的VIDEO_ORI、VIDEO_FORG_rigid、GT_rigid。作者论文明确GT同时包含原复制区与克隆区；已抽查MAT只有GT二值数组，不能直接把它当编辑目标像素真值。保留定性参考，不从像素差补目标mask。

AINPAINT固定HF revision 3dfae9aca06b058aef7996349a2df83d2ec5b717，作者代码/val清单revision753ae1b700b054c97e15569718690211e36cdf53。只用CoCoCo原始432x240、对应input_frames与resized_432x240 mask；不读保留测试媒体。原片与mask允许24条val初筛，只为最终最多4条提取fake。PNG本身无PTS，25fps展示时钟不冒充官方时标。

按作者clip ID去重后排序，Random(20260909).shuffle取前24：boat、incident_curve、airplane_1、D06_2、airplane_11_bis、panda_3、D09_3、mugello_1、airplane_3、D09_1、traffic_3、D08_2、traffic_4、safari1_6、spin、airplane_5、airplane_2、traffic_1、drift_2、D09_2、safari1_1、mugello_2、airplane、D06_1。不可读/标注未对齐不静默换样本。作者未提供原始长视频URL映射，不将不同同系列clip ID强称为独立长源。

机位预筛在原片5个均匀帧做外围背景ORB/Hamming匹配与部分仿射RANSAC。可靠要求匹配>=20、内点>=15、内点率>=.5、内点xy覆盖跨度各>=.1；较稳定候选要求四段都可靠且最大角点移动<=画面对角线3%、旋转<=3°、尺度.97–1.03。匹配不可靠=不确定，不宣称移动或切镜。不是人工机位确认。

稳定且标注核实的候选按mask质心跨度/画面对角线排序：较少位移前2作参照，剩余末2作较多位移候选，seed顺序破同分。该相对分组不是异常标签；不足4保留实际数量，不换池。全部选择在提取fake之前保存。

逐文件清点计划18,235,396,886字节（GRIP六对及AINPAINT三个必需包），总新增下载预算20,000,000,000字节。只解包允许source/variant的成员，不提取其他压缩/后处理版本或测试媒体。所有原件保留，兼容MP4只是预览副本。

单入口：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_geometry_screen.py`；同入口`--download`可复算已保存download_plan.json的固定下载列表。四项针对性检查通过：archive源/版本过滤、mask位移及时间段、低纹理不确定、同帧纹理背景RANSAC。

输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1/`。所有候选仍待人工确认，不能当作下一轮冻结协议或三维异常真值。
