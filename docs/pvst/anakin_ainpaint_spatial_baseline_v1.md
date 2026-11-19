# ANAKIN官方AINPAINT空间结构基线

执行前冻结协议见ADR-0051。仅通用DINOv2 ViT-L/14 registers＋r4/alpha8/dropout.25 LoRA＋官方HighResContextHead。
不使用作者取证成品、不接PVST/XYZ/历史。原11来源2569目标、订单前12845项、每目标五次；原4开发824帧。
右下pad14后输出再裁回工作网格；统一原生计分。每轮前100训练目标额外BN统计，500前向单列。
关键测试4项通过。独立入口research_tools.xyz_pilot.ainpaint_spatial_experiment；完成后补入实测结果。
