# ANAKIN独立来源扩充：等待人工审阅

本轮只做CPU媒体/官方mask核对与页面生成，神经模型、深度、特征提取、训练均为0。未改变旧数据角色与成绩。

官方版本 `978fe26e78ecaaf6562aa249be43c2c690484d0a`；metadata总记录1023，带mask记录425；规范化原视频URL得到425个独立来源。只计可核验URL，不以不同剪辑文件当新增来源。带mask官方记录split按官方loader seed47复现，297/43/85。

历史manifest/protocol及官方test按整个原视频隔离，测试保护94来源：仅身份元数据，未下载/解码/展示。历史角色证据逐条保存在metadata_inventory.json，train/dev冲突保留并阻止进入新训练队列。当前4开发继续是历史开发，不称最终测试。

## 实际准备

|项目|数量|
|---|---:|
|本轮新候选来源|120|
|可打开的预览来源|120|
|新候选技术PASS|93|
|新候选技术待核|27|
|预备失败|0|
|自动固定倾向（非人工确认）|51|
|新增逐ID人工确认|0|
|新获取必要资产字节|744643641|
|下载与CPU准备秒数|1025.7|

本批复用45个既有文件（13套完整triplet，其余为部分已有资产），其余仅获取缺失指定资产。旧资产文件数{'trimmed': 109, 'edited': 94, 'masks': 94}，带mask记录中实际现有文件{'trimmed': 210, 'edited': 201, 'masks': 201}（新取得315个必要文件）。逐ID人工mask原话整理{'unconfirmed': 402, 'corresponding': 21, 'not_corresponding': 2}；不把有相机/编辑描述就算mask已确认。旧技术记录分布{'PASS': 32, 'UNRESOLVED': 10}。新来源角色{'candidate_unassigned': 112, 'dev': 5, 'train': 3}；官方描述预筛覆盖{'addition_copy': 70, 'removal': 40, 'replacement_local': 10}。类别仅是元数据提示，不把“splicing”全当运动对象添加，不虚构人工描述。进一步按描述单列{'appearance_background_reference': 31, 'object_edit_pending_human': 89}；技术PASS为{'appearance_background_reference': 20, 'object_edit_pending_human': 73}。图表、屏幕、配饰和背景替换保留为较低优先级参照，不能拿它们凑运动对象主组。

固定机位主组尚不能最终计数：新人工确认0。技术PASS且传统背景匹配稳定的待审组与其他空间补充/机位待核组分别为{'PVST_camera_prescreen_only': 30, 'technical_pending': 27, 'spatial_moving_prescreen': 27, 'camera_uncertain': 16, 'appearance_background_spatial_reference': 20}。移动机位不自动纳入PVST固定机位主组；匹配不足记不确定，不当作移动结论。

## 推荐审阅顺序与限制

先看技术PASS的添加/复制来源，弥补旧训练以移除为主的覆盖；其次看不同场景的移除；其余局部修改作参照。页面按ID展示、不按旧模型成绩排序，可过滤角色/编辑类型/技术状态。协调自然的编辑同样保留。

有编号和尺寸明确的mask才生成叠加/时间轴，未明确者只显示官方原掩膜。mask的PTS由所对应fake帧承接，没有独立标注时钟。真伪联动是片内时间联动，不承诺像素配准。mask非空变化不等于真实编辑起止。原40y1x9_fake[6,10)监督排除保持原样。

新候选由原120来源清单冻结后准备，无GT面积/预测分数筛选。选定120是批量人工候选规模，技术问题和真实机位仍会减少最终数量；不能写成已获得120个合格训练来源。未复审的旧问题没有官方映射新证据，继续待核，见technical_pending.json。

人工证据保留在human_evidence.json，逐ID只引用明确ID记录；未映射四条自然编辑批次及第三批共同意见均保留原范围，不扩展成每ID mask通过。新增四字段默认未填写，支持选中ID批量写入及逐项例外修正，导出具体覆盖ID与原人工证据。

## 划分与下一轮草案（未冻结、未执行）

partition_draft.json保留已有train/dev/test，未分配来源参考官方train/validation提出train_draft/dev_draft；94个测试保护来源仅作元数据预留，语义质量未确认，不是已验收测试集。其中未有历史角色的74个官方测试来源中，固定seed提出20个fresh_test_draft.json，只有身份元数据，不作合格测试集宣称。人工确认前不自动纳入训练。全部真假/剪辑/编辑版本依原视频分组，现有开发不回流训练。

50–100独立训练来源是后续组织目标，当前可扩充规模上界为本批技术PASS、人工尚待确认的来源，加既有可用train；本批新增技术PASS按草案角色为{'train_draft': 75, 'dev_draft': 10, 'train': 3, 'dev': 5}，其中对象编辑待审且技术PASS的草案角色为{'train_draft': 58, 'dev_draft': 8, 'train': 3, 'dev': 4}。以上仅是人工确认前上界；真实可达规模取决于机位/对象/掩膜语义。添加/复制官方描述覆盖增加，但尚不能承诺获得多少运动对象添加实例。素材获取之外的主要限制是映射与人工语义/机位确认。

唯一下一步：人工确认后冻结来源隔离清单，先固定现有空间基线，按无放回完整遍历（候选10轮，需结合实际帧量与资源事先定稿）验证多来源定位；不机械套12845步。随后图像/XYZ同帧/XYZ时空共用数据和视觉底座；独立测试不调参。本轮不选择新解码器、BN组合或时序结构。

## 位置与执行

审阅页：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_source_pool_v1/index.html`

全量清单：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_source_pool_v1/source_pool_manifest.csv`

下载逐文件记录：downloads.json；完整decode PTS及官方mask检查：各an_ID/identity.json。原始素材优先引用既有ANAKIN目录；新增必要trimmed/edited/masks在本目录assets/。新增字节为取得文件的逻辑大小，不含协议开销或重传。

入口：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=.../rgbd-packages:. python -m scripts.anakin_source_scaleup`，一次准备后可用`--finalize`复用已有预览重建小型汇总；缺少待核RGB固定图时才补解码。完整命令见同目录commands.txt。计算状态COMPLETE，退出码0；Git发布状态单独登记于publication.json。用户人工审阅为本轮终点。

## 验证与计算终态

起始/媒体准备基准HEAD：`a8df5aef41622d63289bea3393b0f6486c8aca94`。实际工作区为`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen`，分支`pvst-anakin-source-scaleup-v1`。新入口由本轮实现；提交包含最终入口、测试及本报告，不改变旧入口。

7项针对性单元检查通过。实际页面120个独立来源，2505个本地引用均存在，2段JavaScript语法检查通过；用既有系统JavaScript引擎和最小DOM核对默认空字段、仅两个选中ID的批量写入/导出、单条“不对应”修正和其他ID不变均通过。未启动浏览器或网页服务，播放和语义最终仍待人工查看。240个原片/编辑片均解码为H.264。

媒体下载/处理进程退出0；最后页面整理退出0，无后续模型任务。完整机器记录见delivery_checks.json、summary.json及publication.json。仓库轻量全量清单为docs/pvst/anakin_source_pool_manifest_v1.csv；大资产只留数据目录。
