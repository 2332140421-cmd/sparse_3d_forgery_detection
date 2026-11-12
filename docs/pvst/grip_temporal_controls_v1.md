# GRIP同源普通复制与局部时间反转：数据核验

执行前范围：原片/rigid/GT_rigid复用，只新增04 Rolling Can、10 Tennis Ball、02 Fast Car的flipTemp AVI与GT_flipTemp。rigid和flipTemp都是伪造，不能作为真/伪标签对照。不运行模型、深度或训练，不改变任何既有实验。

人工口述原样按一个批次登记：共4条，3固定、1轻微移动；编辑为局部物体移除或额外复制原视频中的运动主体；4条看起来协调、肉眼难辨。没有找到能映射具体ID的导出或用户标记，四条编号和轻微移动编号待补，不按顺序或自动机位推断。保留为自然编辑样本，不标为已确认三维时序异常。

核对原始AVI的codec/pixel format、原生尺寸、完整帧数及PTS；固定0/25/50/75/100%五时刻。用两个官方GT并集排除已有复制区域后，仅统计剩余背景的差异，不从帧差生成GT。源/目标未分离时不能用作实际目标像素监督。作者标称局部克隆内容反转，不直接倒置整条视频充作新资产；逐目标时钟映射缺失时明确限制，不以反转是否明显淘汰样本。

入口：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/grip_temporal_controls.py`；仅指定缺失文件下载使用`--download-only`。三项CPU检查保护GT外比较、完整PTS核对和未核实操作不误报严格通过。
