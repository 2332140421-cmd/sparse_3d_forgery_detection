# PVST v0.3 XYZ统一尺度CPU节点预算筛选

**状态：NO_CANDIDATE；仅CPU计数，正式XYZ仍null，正式模型0/2。**

执行HEAD：`e4192beccfc1bb37ce28aa250d0a7e744530b000`；全局scale=1.9888148307800293，单位为归一化坐标单位，不解释为米。
复用native_dense原生回映、有效性及全局scale；逐帧占据求和，不跨帧空间去重。原生点/像素覆盖不变，没有模型构建、CUDA调用或前端提取。
固定帧104/116/127逐层直接floor与整数归并的坐标及原始点成员数完全比较；负数用floor。成员分位数为24帧全部占据块的精确线性分位数（每块等权），不把一个细块视为一个点。
1/8锚点复用原计数并核对manifest/scale/video SHA和区间：2,970,629/1,488,042。

## 固定高负载窗口[104,128)
|fine/coarse h|有效点|fine/coarse节点|预算fine/coarse|节点减少fine/coarse|单份BF16特征载荷GiB|
|---|---:|---:|---|---|---:|
|0.25/0.5|16446812|1488042/613039|False/False|49.91%/58.80%|1.2942|
|0.5/1.0|16446812|613039/198398|False/False|79.36%/86.67%|0.4815|
|1.0/2.0|16446812|198398/55153|True/False|93.32%/96.29%|0.1472|

载荷仅为2×(256×Nfine+512×Ncoarse)字节，两尺度各一份BF16特征，不是训练峰值；本轮不测边数或推测完整显存。

h=0.25逐帧节点min/median/max：{'fine': {'min': 58334, 'median': 61970.5, 'max': 65142}, 'coarse': {'min': 23806, 'median': 25518.0, 'max': 27143}}；块内原始点median/P90/max/singleton：{'fine': {'blocks': 1488042, 'median': 2.0, 'P90': 11.0, 'max': 9085, 'singleton_fraction': 0.4366711423467886, 'valid_point_members': 16446812}, 'coarse': {'blocks': 613039, 'median': 3.0, 'P90': 26.0, 'max': 21543, 'singleton_fraction': 0.2883845889086991, 'valid_point_members': 16446812}}
h=0.5逐帧节点min/median/max：{'fine': {'min': 23806, 'median': 25518.0, 'max': 27143}, 'coarse': {'min': 7708, 'median': 8213.0, 'max': 8902}}；块内原始点median/P90/max/singleton：{'fine': {'blocks': 613039, 'median': 3.0, 'P90': 26.0, 'max': 21543, 'singleton_fraction': 0.2883845889086991, 'valid_point_members': 16446812}, 'coarse': {'blocks': 198398, 'median': 7.0, 'P90': 79.0, 'max': 63681, 'singleton_fraction': 0.11458280829443845, 'valid_point_members': 16446812}}
h=1.0逐帧节点min/median/max：{'fine': {'min': 7708, 'median': 8213.0, 'max': 8902}, 'coarse': {'min': 2145, 'median': 2296.0, 'max': 2491}}；块内原始点median/P90/max/singleton：{'fine': {'blocks': 198398, 'median': 7.0, 'P90': 79.0, 'max': 63681, 'singleton_fraction': 0.11458280829443845, 'valid_point_members': 16446812}, 'coarse': {'blocks': 55153, 'median': 24.0, 'P90': 293.0, 'max': 135781, 'singleton_fraction': 0.043624100230268525, 'valid_point_members': 16446812}}

## 训练队列确认
原队列核对：32训练视频/15351帧/1932候选窗口；窗口含原历史边界与尾段，不只取选中训练片段。
阶段一无合格候选，按规则停止，阶段二未执行。
最接近预算h=1.0，fine/coarse分别为预算的0.99199/1.10306倍；不得因均值较低而放行。
推荐CPU节点预算合格候选：None。这不是冻结正式尺度或4090训练通过。
统一更粗尺度会降低结构空间分辨率；边长相对1/8分别扩大2/4/8倍，全部点旁路和128/256/512宽度保持。
计数worker时间1.878934926353395秒；父进程墙时3.3185685370117426秒；真实退出码0，终止信号None。
产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_xyz_scale_budget_screen_v1`；逐帧/逐窗口JSON留derived，不提交大数组。未读开发/测试坐标或标签，不改旧结果/config/ADR/节点预算。

