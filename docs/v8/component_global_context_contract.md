# V8 全帧区域关系匹配对照合同

分支 `v8-component-global-context`；ADR-0016。唯一主比较
GLOBAL_COMPONENT−GLOBAL_CELL，检验区域成员分区，不是attention对无attention。

同801窗口/16时刻：718训练/122 source、83验证/14 source。只读identityfix与
local_geometry的合法V8输入、content mask、有效mask、shared数据和训练统计。
MEAN_REPLICATED无需重新读取dense XYZ；从同mu与原有效像素mask计算完全相同patch。

141维输入→128维cell，保留固定格邻接；区域节点133维：真实面积加权hidden128、
有效点数加权XYZ3（训练绝对XYZ统计归一化，无有效点时显式0占位）、区域面积比1、
成员数/4。两条件唯一分区差异。Linear133→128+ReLU，单层MHA128/4/dropout0，
residual+LayerNorm；区域context广播→原256→128→128 residual一次→cell GRU。
同帧全域交互不读取未来；区域编号无连续输入或位置embedding。padding不能作为节点。

保留原多前驱、真实delta_t、no-history、状态和头。frame面积LSE与直接时空面积LSE
一致。全缺失内容区域保留，纯padding排除；不将响应称为真值定位。

两条件参数量相同、同seed初始化、同数据顺序。AdamW lr1e-3/wd1e-4，clip1，
有效batch8，100 epochs，最后epoch；seed20260909/10/11。只在窗口级使用原
source/class weighted BCE。允许物理batch减小且按同8窗口归一化累积，不降样本。

三seed先平均logit；阈值0。source配对bootstrap10000/seed20260909，宏平均与pooled
同总体。独立输出，不重训旧模型，不跑前端，不更改streaming接口。
