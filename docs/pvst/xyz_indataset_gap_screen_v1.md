# PVST独立原视频开发评价及第三批来源审阅

执行前协议，ADR-0048。I10/A10/B10同10260步，训练仅原7来源前70%；本轮dev4全视频824帧。
B13仅继续训练附表。零训练、零前端，原scale/h/精度/指标不变。独立入口：
`python -m research_tools.xyz_pilot.indataset_gap_screen`；CPU候选：`CUDA_VISIBLE_DEVICES='' python scripts/anakin_candidate_review_v3.py`。
完成后补入实际结果、唯一后续建议、候选表与预览路径。
