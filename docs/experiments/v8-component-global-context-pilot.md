# V8 动态component全帧关系匹配对照

独立分支 `v8-component-global-context`，起始HEAD
`151e300549ab2f238e46b184da3647abdbe50647`。合同见
`docs/v8/component_global_context_contract.md`、ADR-0016。

两条件GLOBAL_COMPONENT与GLOBAL_CELL共享MEAN_REPLICATED、141维cell输入、
原固定格消息、区域节点属性/单层4头同帧attention、cell多前驱GRU与面积LSE。
唯一条件差异为动态component成员分区或active singleton格。旧component residual
不额外调用。每条件319,969参数，配对初始化、数据顺序、标准化完全相同。

只读identityfix和local_geometry合法V8缓存，801窗口身份逐项核对。
训练718窗口/122 source，验证83窗口/14 source（42 real/41 fake），每窗16时刻。
不运行任何前端、不读取V7数值资产、不改全视频接口、不用旧模型初始化。

AdamW lr=1e-3/wd=1e-4、clip1、有效batch8/物理batch8、原source/class weighted
窗口BCE，100 epochs/最后epoch，三个seed20260909/10/11。主评价先平均三seed
logit，再计算source-macro AUROC；配对source bootstrap10000/seed20260909，阈值0。

正式训练前19项本轮及直接相关测试通过。真实1LARL real/fake smoke完成训练更新、
optimizer保存恢复、评分与严格JSON评价落盘，重载误差0。GPU resident输入约2.73GiB，
smoke有效batch8 step=0.113885秒，allocated峰值6.447GiB；首步估计六模型约102.5分钟，
不是最终训练实测。未重复dense XYZ副本；仅预载mean/mask/shared。

输出目录：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_component_global_context_pilot_v1/`。

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
bash research_tools/v8/component_global_context_pilot/run_all.sh --resume \
  --baseline /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_identityfix_v1 \
  --reference /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_local_geometry_encoding_pilot_v1 \
  --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_component_global_context_pilot_v1
```

COMPLETE只表示六模型/评分/报告完成，不预设假设成立。历史局部几何与identityfix
分数仅作独立背景，不是新attention的匹配消融。结果完成后在此补充实测表。
