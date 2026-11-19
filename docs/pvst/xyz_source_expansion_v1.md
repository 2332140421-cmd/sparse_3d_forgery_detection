# PVST全视频训练来源扩充配对

执行前冻结协议，ADR-0049。两组原7全视频对照原7＋第三批指定4来源；原dev4完整视频不变。
共同原随机initial.pt/XYZ、全新AdamW、完整历史，scale12.880994319915771/h.5不变。
U=10*Nexp训练前固定，U/2和U评价，不按开发改预算。新增缺失观测按原官方流程一次准备。
入口：`PYTHONPATH=. /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.xyz_pilot.source_expansion`。
正式完成后补入实测结果与单项下一步判断；大资产留derived/pvst_xyz_source_expansion_v1。
