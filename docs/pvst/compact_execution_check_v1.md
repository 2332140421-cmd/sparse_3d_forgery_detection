# PVST紧凑组织与限定资源验收

本轮范围：只验证既定UV/XYZ完整设计在4090上的资源与吞吐，不调用正式30轮训练入口。
原运行保持RESOURCE_BLOCKED、正式0/2；新验收输出独立于原失败目录。

## 执行优化

- 每尺度只保留原整数坐标、排序键、全局顺序与帧边界；query块内以同一searchsorted规则生成准确81候选并立即attention。
- 三层dilation1/2/4、同帧/t-d/t-2d、27空间偏移、真实PTS差、空邻居mask及节点顺序不变。
- 相对关系缩为每帧每层[T,81,4]；bias MLP按共享帧模板计算，参数梯度从所有查询累加。
- 不在checkpoint外构造全量图；确定性索引在query checkpoint重计算时再生成。完整K/V只共享引用，不复制给每个query。
- FFN块内执行residual+FFN(norm2(residual))，仅拼接最终D维输出；不做最终全节点额外加法。
- 不保存或跨更新复用decoder/点/块/时空可学习输出；没有detach历史特征。
- CPU卸载仅作为实测GPU仍受限的后备：在每时空层外层checkpoint边界卸载该层输入一次；不在小query中重复卸载完整K/V。

不改变初始化、数据、分辨率、全部像素/有效点/体素/邻接、128/256/512、target8+halo16、损失或评价定义。
没有永久新增几何缓存；已有滑窗仅复用无参数路由。需要的全队列资源计数仅存每窗口少量数字。

## 必要正确性

旧展开实现只留在小规模测试参考，不保留第二套正式推理/训练路径。
31项相关测试通过：负坐标、空帧/邻居、非均匀PTS、跨帧query及尾块的邻接和关系严格相同。
多帧三种dilation对旧公式的最大前向误差：FP32约5.36e-7、BF16为0；输入梯度约1.05e-9/1.53e-5，参数梯度约6.98e-9/6.10e-5。
固定新分块下，dropout开启的checkpoint/RNG与梯度一致性测试通过；卸载边界测试只捕获每层一次输入，无重复K/V。
原模型参数及state_dict键的实测核验随资源worker执行，不用测试替代实际GPU验收。

## 独立入口与顺序

本新增入口的直接调用者是本轮CLI，输出资源case/result_summary/中文报告。
旧train.main会进入完整正式训练，因此不适合作为本轮调用入口；只复用现有Video、模型、损失和优化器参数组。

```bash
PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages:. \
TORCH_HOME=/media/njupt/8FCD3869E2FAB06F/chenyh/cache/torch \
XFORMERS_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.pvst.compact_check
```

先直接XYZ失败窗口[0,8)；GPU不够才尝试一次限定卸载后备。失败则停止长验收。
通过后扫描原48个train/dev缓存的窗口节点数；选择已知失败、完整历史最大fine/coarse、高原生分辨率，去重最多4窗。
两条件每窗含两次微批累积/两次Adam更新、所有目标像素损失、关键梯度检查及eval路径。
这些是窗口资源更新，不是正式完整视频更新；权重丢弃，不保存模型checkpoint。
attention512、FFN8192起步；同一代表最多一次更大attention2048短测；不搜索其他组合。
一个高分辨率代表做一次短profiler，独立于非profiler吞吐测量，不记录全队列trace。

输出绝对目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_compact_execution_check_v1/`

progress.json、case_*.json/log、resource_windows.json、result_summary.json、report_zh.md；只有完成对应短测才产生profiler_summary.json。
记录code/config/manifest/scale/缓存metadata身份。原失败目录不改状态、协议或退出码。

最终结果见[资源与吞吐报告](compact_execution_results_v1.md)。限定验收已结束，RESOURCE_BLOCKED、退出码1，正式0/2。原失败首8帧窗口通过，最大完整历史窗口GPU和CPU卸载均在全量LayerNorm转换OOM；因此未执行2048短测及profiler，不重新启动入口或正式训练。
