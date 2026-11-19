# 官方DFormer真实末级适配

执行前协议，ADR-0050：一组12845更新，参照原扩充组同预算；数据、PVST结构及loss不改。
真实RGB/depth末级10670080参数可训练，原PVST631170参数；前缀及运行统计冻结。
入口：`PYTHONPATH=<原rgbd-packages>:. <原v8 Python> -u -m research_tools.xyz_pilot.dformer_stage4_experiment`。
完成后补入实测报告；数据保存derived/pvst_xyz_dformer_stage4_adaptation_v1。
