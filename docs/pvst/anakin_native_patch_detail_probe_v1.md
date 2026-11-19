# ANAKIN原生局部细节匹配输入对照

执行前冻结协议ADR-0053：两条件NATIVE_PATCH/LOWDETAIL_PATCH，仅输入退化不同。
原11训练来源、原4开发、共享原订单前12845与288原生坐标crop/GT；新通用LoRA/head共同初值。
原生整帧单次解码；LOW按原AREA缩384再LINEAR恢复整帧后裁同块。原生GT不退化。
各12845更新、BN额外500。只末次824原生盲滑窗288/144，FP32 logit均匀融合，全覆盖，不依GT选tile。
五项关键测试通过；少量真实检查后直接训练两组。完成后补实测结果，不改旧实验。
入口research_tools.xyz_pilot.native_patch_experiment。
