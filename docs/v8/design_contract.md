# V8 Full-Coverage 3D Observation Field 设计合同

## 目的与隔离

V8 是 V7 分支上的独立开发分支。它只复用原始视频、官方标签、source/split
身份和窗口定位信息；不读取 V7 的粒子、XYZ、轨迹、component、S/Q、特征、
standardizer 或模型。所有前端输出由原始视频重新生成。

## 全覆盖与缺失

256×256 等比例 letterbox 画布中的每个非 padding 16×16 基础格始终存在，
包括几何完全缺失的格。几何/visibility mask 只标记某项数值是否可计算，不能
删除区域、改变标签或自动制造异常。缺失使用有限占位加独立状态通道进入模型，
并在 provenance 中保存原始非有限值和 mask。

## 观测前端

每个窗口按 PTS 取 16 个目标时刻，保留实际帧索引/PTS/重复帧。每帧从 MoGe-2
官方 `Ruicheng/moge-2-vits-normal` 获取 dense point map、depth、原生 mask、
intrinsics；短时对应使用官方 CoTracker3 接口，在当前帧重新放置 32×32（每
基础格 2×2）查询，仅看 `[t−2,t−1,t]`，接口所需的重复端点单独记录。输出被
规范化为 V8 前端 schema，不暴露 provider 私有张量给模型。

## 区域与关系

每帧 16×16 基础格通过确定性二维四邻接 union-find 动态合并，每 component 最多
4 格。空间代价只使用当前有效 XYZ 和可用近期观测运动；信息不足时保留基础格。
保存 pixel→cell→component、面积、成员、未合并边、几何/观测状态和跨帧多对多
关联计数。component ID 不是模型语义特征，不能要求跨帧固定。

## 模型

FULL 使用冻结 ImageNet ResNet18 feature map 的格级 RGB 特征、dense 几何和
观测状态；动态 component 内置置换不变 pooling，同帧邻接关系 MLP，跨帧只从
此前 component 的关联 hidden 经过 GRUCell 更新，显式接收真实 `delta_t`。所有
格/时刻输出 logit，窗口用 beta=1 的面积加权 log-sum-exp 聚合。RGB_2D 使用同
一 RGB/基础格/二维对应和时间模块，不读取几何分支及几何 component。监督是
窗口级真假标签，不复制成像素标签。

## 证据边界

报告仅使用窗口级分类指标和统一尺度证据图。热图是弱监督模型响应，不能称像素
级检测或伪造 mask；相机坐标 apparent motion 不是世界物理速度。V8 队列沿用开发
实验窗口，不能称独立 sealed test。
