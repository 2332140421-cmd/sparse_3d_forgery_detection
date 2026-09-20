# ADR-0013: V8 全覆盖三维观测场独立方法边界

状态：accepted（V8 分支范围内）

## Context

用户授权在当前项目中验证一种与 V7 稀疏粒子方法隔离的全覆盖观测场：每个
真实图像内容基础格始终保留，几何、短时二维对应和缺失状态在区域级模型中
共同使用。V7 的粒子、查询、component、S/Q、特征和模型缓存不能成为输入。

## Decision

在 `v8-full-coverage-observation-field` 分支中使用独立的
`research_tools/v8/full_coverage_observation_field` 包和独立数据目录。V8
重新从原始视频运行 MoGe-2 与官方 CoTracker3；只复用窗口级身份和官方标签。
所有非 padding 基础格进入模型，缺失通过显式状态和有限占位表示，不能将缺失
区域删除或改标为 fake。每帧动态 component 与仅历史的多对多关联保存在前端
产物中，模型先做空间/关系和跨帧递归，再做面积加权窗口聚合。

正式条件固定为 `FULL` 与 `RGB_2D`，各三个 seed、100 epochs、AdamW、lr
`1e-3`、weight decay `1e-4`、有效窗口 batch 8、梯度裁剪 1；不导入旧模型
或 standardizer。RGB_2D 使用同一基础格和二维短时对应，但不读取 XYZ、深度、
几何 mask 或几何 component。

## Consequences

V8 的“全覆盖”只表示处理分辨率上的区域覆盖，不是原始每个像素独立测量，也
不是空间真值。V8 的前端可靠性、世界坐标物理速度和热图定位能力均不在本轮
被证明。V8 队列沿用开发实验窗口，不能称独立 sealed test。

## Supersedes

不取代 V7 或其 ADR；本 ADR 只在新 V8 分支和新产物目录内建立隔离合同。
