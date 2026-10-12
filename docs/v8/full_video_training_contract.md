# V8 完整视频级弱监督微调合同

ADR-0018；两阶段训练：既有同condition/seed窗口epoch100权重→完整视频epoch20。
条件GLOBAL_COMPONENT/GLOBAL_CELL，seed20260909/10/11，共六个模型；每个319969参数。
保持MEAN_REPLICATED、77+64输入、hidden128、4头region attention、原residual与多前驱GRU。
全部网络参数可训练，前端、标准化与grouping冻结，不读取V7数值资产。

只读完整272视频/234284实际帧紧凑缓存。244训练视频/122source；28验证视频/14source。
不跳帧、不抽样，所有真实内容格包含无几何/无历史区域；padding排除。
CPU逐视频读取，GPU当前块计算；CPU输入保留至完整反向完成。

z=logsumexp(cell_logit+log(area))−log(sum(area))，跨所有帧/真实内容格。
一个视频一个BCE；同source real/fake各乘1/2，完成pair统一clip/step。
只有真正视频首帧初始化hidden；其余帧使用真实dt、前驱及active。
非重入checkpoint重计算，hidden和累计LSE不detach；不引入块/帧/格标签。

资源smoke完整1LARL pair及最长训练视频；比较chunk32/64，必要时减小执行块，
不改变数据、监督或梯度。正式前冻结资源配置，并复现28验证视频六模型原始分数。

20epochs，AdamW lr1e-4/wd1e-4，clip1，FP32/dropout0，无验证选best或阈值搜索。
每epoch全部122pair，同seed两条件相同source顺序；新optimizer，不载入旧optimizer。
原子覆盖恢复文件含模型/optimizer/RNG/顺序/epoch/pair索引/输入身份，最终epoch20另存。

主比较微调COMPONENT−CELL的pooled video AUROC；辅助各条件微调−冻结。
每seed先完整视频LSE，随后平均三个video logit。阈值0；source配对bootstrap
10000次/seed20260909。训练/验证及各seed指标分别报告，不作帧或空间定位结论。

## 用户授权的固定5轮范围调整（ADR-0019）

原六模型20轮目录暂停保留；新目录`v8_full_video_training_budget5_v1`仅包含
GLOBAL_COMPONENT、原三seed各5epochs，最后epoch5。上文全部数据与计算路径不变，
chunk固定64。原合法checkpoint在5轮以内可显式迁移，保留optimizer/RNG/顺序/
epoch/pair/loss/耗时，记录原hash和新protocol身份，不能绕过身份检查。
主比较改为微调COMPONENT−冻结COMPONENT；旧CELL只作历史冻结参考。
已有资源和168分数复现记录只读引用，不重复执行。没有验证选轮或追加训练。
