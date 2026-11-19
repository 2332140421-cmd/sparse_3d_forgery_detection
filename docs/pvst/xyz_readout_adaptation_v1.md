# PVST定位读出适配

执行前协议，ADR-0047。共同B epoch10完整恢复；READOUT_ONLY四读出模块对照全部继续训练。
保留原数据与完整历史，追加第11–13轮，每组3078更新，末次13338。入口：
`PYTHONPATH=. /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.xyz_pilot.readout_adaptation`。
正式完成后补入实测结果，资源与身份见独立产物pvst_xyz_readout_adaptation_v1。
