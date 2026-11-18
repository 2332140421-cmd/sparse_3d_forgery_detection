# ANAKIN第二批定向候选与首批人工登记

**交付已完成，停在人工审阅。** 本轮新增12个登记独立来源的预览，其中8对通过媒体/mask技术对应核验，4对明确待核；新增候选均未获人工场景/语义批准。没有下载、模型、深度、特征提取或训练。

## 首批人工结论登记

实际读取 `/media/njupt/8FCD3869E2FAB06F/chenyh/reports/human_review.json`，原文件SHA与逐字副本见human_review_original.json / human_review_registered.json。原口述中的笔误保留在raw字段，没有假称已读取不存在的文件；分组为整理意见，不是三维时序异常真值。

|首批pair|保留角色|原机位原文|mask原文|整理组别|

|---|---|---|---|---|

|an:40y1x9|train|近固定，镜头有晃动|对应|main_candidate|

|an:4rvro9|dev|固定|对应|attached_object_reference|

|an:4wdo37|train|移动镜头，非固定|对应|moving_camera_reference|

|an:4wgm37|train|移动镜头|对应|moving_camera_reference|

|an:4wgx37|dev|移动镜头|对应|moving_camera_reference|

|an:7e2e57|dev|固定镜头|对应|main_candidate|

|an:7ogg04|train|移动镜头|对应|moving_camera_reference|

|an:7ogo04|train|固定|对应|pending_verification|

|an:95q8g9|dev|固定|对应|main_candidate|

|an:95qlg9|train|移动视角|不对应|pending_verification|

|an:95qvg9|train|固定|对应|main_candidate|

|an:9l8w54|train|固定|对应|main_candidate|

主候选5对：40y1x9、95qvg9、9l8w54（保留训练角色）；7e2e57、95q8g9（保留开发角色）。**仅3个训练来源和2个开发来源，不能视为规模充分。** 4rvro9为附着对象参照，不因帽子随人物协调运动而排除。7ogo04的官方/人工内容冲突、95qlg9的人工mask不对应列为待核，不阻塞本轮找其他来源。其他移动机位样本保留参照，原开发4wgx37仍留在历史开发结果中。

总计固定6、近固定1、移动5；人工mask对应11、不对应1。原始编辑描述逐字保存在登记JSON/CSV，不自动改写官方文字。首批12对不要求重新填写。另一个未明确ID的四条自然编辑批次仍保留原记录，不映射到这12对。

## 定向来源范围与选择

先查本地94条edited文件、固定revision官方metadata.csv、下载文件索引及已有manifest的来源角色。只读来源/split元数据，未读取模型成绩或旧数值特征。官方has-masks保留test来源及已登记test来源排除、不读取媒体；原20来源排除；已查train/dev角色冲突排除；同URL编辑去重。对本地齐备的对象增删类素材进行预筛，最终完整本地候选池34个source。没有因素材不足申请其他库、全库下载或扩大本批上限。

背景预筛复用原camera_check：5个均匀原片帧，外围ORB匹配和部分仿射RANSAC；稳定阈值及可靠条件沿用原脚本。选12前先保存候选和规则。优先稳定倾向，同层使用排序ID后seed20260909打乱；最多9个train/unassigned＋3个dev以补充训练方向并兼顾开发多样性。未根据可见异常、GT面积或模型表现选择；自动稳定倾向不等于人工合格。完整34条预筛及未选22条在selection.json，排除理由在metadata_inventory.json。

保留角色只来自已有source清单，不从旧分支迁移模型或训练路线；具体role_provenance列出清单路径。train/dev是既有来源约束，本轮没有冻结新的PVST训练协议。无既定角色者记candidate_unassigned，不自动放训练。

## 本轮12个新来源

|pair|角色|官方编辑描述|自动机位|技术对应|语义/场景|

|---|---|---|---|---|---|

|an:4wdn37|train|remove running guy|固定倾向，待人工|PASS|待人工|

|an:9mygy9|train|remove a person|固定倾向，待人工|PASS|待人工|

|an:76rjm7|train|Remove the guy who goes with the phone|固定倾向，待人工|PASS|待人工|

|an:4ym5n4|train|remove a passing man in black in the background|固定倾向，待人工|PASS|待人工|

|an:4qwgw4|train|Remove the man walking on the left side|固定倾向，待人工|PASS|待人工|

|an:40ypx9|candidate_unassigned|Remove the woman in the pink T-shirt on the right|固定倾向，待人工|UNRESOLVED|待人工|

|an:9lpgg7|candidate_unassigned|remove people passing on the left|固定倾向，待人工|UNRESOLVED|待人工|

|an:9momg4|candidate_unassigned|remove the woman crossing the road|固定倾向，待人工|UNRESOLVED|待人工|

|an:95qnq9|candidate_unassigned|Delete player in jersey #26|固定倾向，待人工|UNRESOLVED|待人工|

|an:7e83j7|dev|add people sailing on a small boat along the coast|固定倾向，待人工|PASS|待人工|

|an:4xd3r9|dev|remove any 5 people|固定倾向，待人工|PASS|待人工|

|an:7ogeg4|dev|Remove the woman|固定倾向，待人工|PASS|待人工|

全部12个source URL与原20不重合；12条real的SHA彼此互异且与原20条real不重合。源URL、官方revision、文件路径/官方LFS校验、原生尺寸、完整解码PTS与role依据保存在inventory.json及每对identity.json。只表示已核实来源身份与媒体无字节重复，不证明不同URL不可能来自相同拍摄场景。

原角色构成：5个train、3个dev、4个candidate_unassigned。8对技术通过=5个train＋3个dev；4个未分配来源均有技术缺口。本轮新增独立来源12，新增完整技术核验对8；新增人工确认主组为0。最终训练/开发探针规模仍待人工批准及独立来源分组。

## 资产与mask对应缺口

|pair|视频/标注问题|本轮处理|

|---|---|---|

|an:40ypx9|fake 119帧；官方mask 120张|仅RGB/独立源mask展示；不叠加、不补帧或缩放修复，不作为已核验空间监督|

|an:9lpgg7|fake 180帧；官方mask 181张|仅RGB/独立源mask展示；不叠加、不补帧或缩放修复，不作为已核验空间监督|

|an:9momg4|fake 299帧；官方mask 300张|仅RGB/独立源mask展示；不叠加、不补帧或缩放修复，不作为已核验空间监督|

|an:95qnq9|fake 150帧；官方mask 150张；('an:95qnq9', 0, 'MASK_SIZE')；fake原生[[720, 1280]]|仅RGB/独立源mask展示；不叠加、不补帧或缩放修复，不作为已核验空间监督|

8对通过者：官方原片/伪造片/ZIP在固定revision978fe26e78ecaaf6562aa249be43c2c690484d0a的公开校验值匹配，逐帧mask数量/数字编号/原生尺寸通过。mask没有独立PTS，只能按官方帧号映射到实际解码PTS；语义是否覆盖真实编辑仍待人工。8对均为全程mask非空，**本轮没有增加已核验的部分时段编辑起止样本**；4对未对齐者时间状态未知，不填空mask、也不当全程编辑。

技术不对应的4对页面不允许切换不存在的叠加视频，提供官方ZIP及固定首/中/末源标注独立显示，不把它们假定为fake同帧GT。原始数据/掩膜不改写，没有用真假差分生成标签。

## 40y1x9边界专项

独立页面：40y1x9_boundary/index.html。展示零基3–12共10帧，每帧有fake RGB、官方mask及叠加，文字列原生帧号、实际PTS和mask像素数；另有这10帧的缩放叠加短片。帧6 PTS0.200000秒、帧10 PTS0.333333秒，空mask区间[6,10)已复核，帧5与帧10均非空；这10帧中6帧非空。

状态固定记为“mask状态变化，内容边界待确认”。请人工检查空档时编辑对象/填补背景是否仍存在、mask开关是否对应画面；未确认前不作为编辑起止真值。首批整体mask对应判断不替代此次精确边界核查。

## 预览及后续可扩充范围

index.html只要求本轮12对填写相机、实际对象及运动/遮挡/形态、mask和时间范围三项；首批已登记信息以链接提供，无需重填。同步按已核对片内时标联动；长度/PTS网格不同组明确提示，不视作像素配准。原媒体以链接复用，完整叠加/mask视频只用于预览缩放，重解码核对PTS。图像没有模型热图或成绩。

本轮大部分为行走、奔跑或背景人物移除，另有7e83j7的小船上人物添加。提供了主组同类新来源，但动物、对象替换、明确局部形态改变，以及可靠内容编辑起止样本仍不足。现有主组仍5对，不能把自动稳定的12对直接相加称为17对合格数据。自然协调的编辑同样保留；不要求人工先看到物理异常。

## 研究意图与执行边界

当前用户研究意图为在明确重建坐标系中划分等尺寸XYZ块、联合视觉/几何块编码、学习跨位置和因果时间关系并融合局部细节。单目可见表面不等于完整场景，无标定不称米制。允许对象跨块移动，不要求所有时刻存在相同非空键。此为后续尚未冻结的设计，本轮不选块边长、宽度、层数或时间窗，也不把手工历史距离探针当完整PVST。旧替代实现、资源失败、历史成绩和划分不改写。

## 运行与发布

CPU入口scripts/anakin_candidate_review_v2.py，3项针对性检查通过：source角色冲突/test排除、元数据对象类型；真实ZIP成员计数；mask尺寸失败保持未知不修补。新增下载0字节；未安装依赖、未下载权重，无深度/特征/模型调用。

首次严格检查在40ypx9的官方帧数缺口退出1；后续记录分支先遇ZIP导入遗漏，再在另一尺寸问题的错误上下文中暴露缺pair_id，均保留失败日志与真实退出1。只修正诊断控制流/错误记录并续完原12清单，未修数据、换样本或重复已完成预览。最终计算/预览COMPLETE、退出0，与标注8通过/4待核分列。first_failure.log、second_failure.log、third_failure.log和additional_checks.json保留证据。

完整34源预筛与已完成预览分次复用；最终续跑实测9.092秒（不含已保存的前序预筛/预览，不冒充整轮总耗时）。页面资源和新增/原20媒体身份另行核对通过。执行初版492d506，完成核验73faffd91895bd4783d486728de0494698f0d485，页面待核展示a9914e0。

计算/预览完成与Git发布分开记录。只提交脚本、必要测试和小型报告；媒体、mask、预览及人工原JSON副本留服务器。

预览绝对路径：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_candidate_review_v2/index.html

边界绝对路径：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_candidate_review_v2/40y1x9_boundary/index.html

本轮到此停止，等待人工确认，不自动提深度、特征、推理、训练或再扩来源。
