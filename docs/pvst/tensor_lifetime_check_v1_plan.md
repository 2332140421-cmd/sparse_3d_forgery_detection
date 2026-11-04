# PVST无用view清理：训练前限定验收

只在最后使用后清理SparseLayer投影循环xb/ab/args、已cat的旧values、head投影最后part、FFN尾部part/x。不加永久清理开关，不删除autograd保存，不用detach/no_grad/empty_cache/gc。保留全部当前执行优化、模型/参数/梯度/数据/监督。8项直接相关CPU检查通过，包括固定父提交b7ab0b8同权重/输入/seed、256/512、FP32/BF16、dropout/内外checkpoint/尾块，以及元数据观察器不改变梯度。

直接观察器仅SparseLayer和sparse_head调用：metadata(shape/dtype/device/storage_id/storage_bytes/grad_fn)及allocated/RSS，去重共享storage，不把view.numel当存储。不持有Tensor/grad_fn对象；ToCopy节点前/后hook闭包只存元数据，记录pending未完成节点，不修改梯度。覆盖QKV精度、稀疏结果精度、head GPU→CPU、attended CPU→GPU；不重写checkpoint恢复或拷贝反向。复制日志限1024事件，生命周期另限256条，当前pending持续落盘；不建监控框架或全程profiler。

独立入口python -m research_tools.pvst.tensor_lifetime_check：四组必要CUDA正确性→一组project_block存活隔离检查（无outer checkpoint展示autograd合法保存；有outer checkpoint核对解除view后的实际allocated）。131072×512的attended=x*1.5仅隔离所有权，不是单层性能基准，不报检测效果。

通过后原worker同一模型/AdamW、同一最大窗口最多两次正常zero_grad/新图更新；总900秒。第一次失败立即停止第二次；每次成功更新独立落盘，保留第二次失败时的第一次成功，不伪装全部失败。原默认worker仍只一次；窗口身份/预期节点/原始PTS从冻结manifest校验，update loss/梯度/optimizer.step次数/参数变化/资源逐次记录。

唯一最大窗口：XYZ video782426ea0330c7fea684953c，ANAKIN/train/real/an:4wdo37，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4，节点2970629/1488042。原seed20260909、attention512/QKV8192/FFN8192/point32768/tile256、BF16/逐head FP32、外层卸载、CPU head、逐headQKV、node_split均保持。正式XYZ仍null，正式模型0/2，权重不保存。

仅最大窗口两次通过后，检查本轮严格匹配的非空GT证据；当前新目录无此证据，旧资源仅真实零GT/旧执行版本不匹配。按冻结clips稳定综合身份首合格；若原v03因预算停止没有clips.json，明确记录使用冻结ANAKIN train队列的稳定身份/stride8首个完整16历史+8目标且至少一帧非空mask的片段，不冻结新训练划分、不按模型选择。相同片段XYZ/UV各规定初始化一次更新，各300秒；UV只UV输入/1/32，禁止native_dense读取XYZ。最大窗口未通过则不选择/执行补充case。

新产物derived/pvst_tensor_lifetime_check_v1；每GPU运行前已提交源码，运行期间不改。父进程只管理本worker组、保留真实退出码/信号；OOM优先CPU证据，异常退出与计算状态分别记录；Git发布单列。不重复原实现完整窗口，不训练，不追加优化或数据/前端任务。旧资产/协议/错误/退出历史/config及用户JSON保留。共享GPU当前占用与过去可能不同，记录起始资源，不把整体差异都归于del。
