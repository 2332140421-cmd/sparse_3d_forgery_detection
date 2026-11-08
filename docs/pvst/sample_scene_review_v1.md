# 训练样本与优先场景：限定核验

基于 `pvst-set-to-set-relations` / `c9fc932518fed83793e02005660879b7fe27b8d8`。
本轮只做CPU媒体/标签/已有几何预览，未运行模型、前端、下载或训练；旧实验与报告不改。

打开：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_sample_scene_review_v1/preview/index.html`

## 结论与建议

**当前两对不是都明显偏离研究内容，也不能把数据不匹配当作上轮失败的确定原因。**
`7ogg04`有连续步行及运动人物移除，但固定机位条件存在疑问；`728qx7`机位较稳定、局部纹身随身体变化，但首段活动幅度小。
自然协调的编辑仍可保留，纹身、移除的类别本身不能证明有或没有关系信息。

本轮看过固定六帧及完整real等间隔八帧序列图，**没有播放全段确认**。
完整原视频及连续GT视频已供人工播放；下述全部为初审，不是“人工确认合格”。

|pair|拍摄与活动初审（A）|官方编辑与可观察内容（B）|场景建议|
|---|---|---|---|
|an:7ogg04|两人向镜头方向连续步行；近景长椅逐渐进入画面，疑似移动机位|inpainting：remove girl without umbrella；六帧GT覆盖被移除女孩原位置及填补区|活动/编辑内容相关；机位待确认，暂不作为固定机位优先项，不永久排除|
|an:728qx7|门框/墙/毛巾位置较稳定；首段可见头、口、上身小幅变化；后续手势不属于首段训练|splicing：Give a guy a tattoo on the right side of his chest；细图案GT随身体位置变化|初步可保留局部人体外观/形态编辑；需确认微动作是否符合目标|
|an:4wdo37|多名运动者在草地跑动；后段建筑相对画面移动，有跟拍/平移倾向；具体运动项目未核实|inpainting：Remove the guy with the ball in his hands who is running away from everyone；GT覆盖局部被移除运动者|机位与多人活动复杂程度有偏离倾向，暂不优先；不是因为移除或伪造是否明显|
|an:95qvg9|机位基本固定/轻微晃动倾向；前景说话、背景两人步行，有正常车流，不是全景静止|inpainting：Remove a woman and a man walking on the background；GT为包含行人及周围背景的宽矩形|初步优先，待人工确认正常背景运动与宽编辑区是否可接受|

**建议暂保留 `an:728qx7`、`an:95qvg9` 作为下一次拟合的待确认候选。**
依据是相对固定的视角倾向、可观察简单变化及已核验局部编辑GT，不依据模型表现。
这不冻结新训练协议；人工若不认可，应保留不合格/待定结论，本轮不再补位。

## 时长与标签事实

以下均为完整发布的trimmed/edited短视频，不是Vimeo原始长视频。
原片全部H.264/yuv420p，直接链接，无完整视频转码。时长为实际PTS加末帧间隔；不是简单用首末PTS差冒充时长。

|pair|原生H×W|real/fake总帧数|完整短片约秒数|首24帧真实时长real/fake（秒）|fake正GT帧|fake平均GT面积|
|---|---|---|---|---|---|---|
|7ogg04|360×640|75/75|3.000/3.000|0.960000/0.960000|24/24|5.901602%|
|728qx7|720×1280|150/150|5.005005/5.005000|0.800800801/0.800800|24/24|0.182794%|
|4wdo37|720×1280|300/300|5.005005/5.005000|0.400400400/0.400400|24/24|0.562988%|
|95qvg9|540×960|100/100|4.000/4.000|0.960000/0.960000|24/24|8.753086%|

real首段均为合法零GT，不使用其manifest中继承的pair级fake mask路径给real套标签。
`7ogg04`首末PTS为0–0.92秒，实际包含末帧后时长0.96秒；`728qx7`首末约0–0.767434秒。
各role独立PTS、全部24个帧索引及时间均在HTML展开表和facts JSON中，未宣称real/fake精确逐像素配准，不计算像素差分。
官方metadata的start-time/end-time是源长视频裁剪位置，不作为本短片内伪造区间。

两段首片确有可观察变化，但均不足1秒；新增4wdo37约59.94fps，24帧仅0.4秒。
因此本批能描述短时姿态/步态变化，不能证明已覆盖完整动作阶段或有足够关系差异可学。
短时间跨度、局部图案/背景人物等编辑范围能解释“探针实际检查了什么”，不能单独解释关系帧头的学习结果。
真假可分性、关系单元的使用/同质化原因和泛化收益（C）仍需后续训练或中间表征分析，本轮不判断。

## 选择和核对范围

固定先处理7ogg04、728qx7。因前者机位待确认、后者短段活动有限，最多补两对。
只在现有RGB-D**训练**清单中，按官方描述含running/walking的人物/物体局部编辑形成候选池：40y1x9、4wdo37、95qvg9。
ID排序后seed20260909打乱，取前两条为4wdo37、95qvg9；`selection.json`先于候选媒体处理保存，含元数据任务原文及清单hash。
不读取任何候选模型分数/预测来选样，不替换不理想候选。
四对均只检查原[0,24)连续片段；候选首段已有活动，没有检查或挑选中/尾24帧。
完整real的等间隔八帧图仅供镜头初审，不是另选训练片段或完整播放的替代。

8条源视频SHA重算匹配冻结manifest；完整CPU解码共1250帧，帧数/原生尺寸/逐帧PTS均与登记一致。
当前两对还与上轮selected_identity、GT_identity_supplement逐项核对。
fake ZIP SHA匹配；复用既有numeric_members/alignment/mask_array，按数字帧号排序、基准及帧数核对，当前协议阈值≥0.5、原生尺寸GT不修补。
六固定帧目视检查支持GT覆盖所描述编辑位置；24帧都有标签不等于24帧编辑都肉眼显著，也不等于复杂时间边界监督。
95qvg9的官方矩形GT含周围背景，不改成精确人体轮廓或“可见异常区域”。

高分辨率几何明确来自 `derived/pvst_rgbd_slice_relations_v1/geometry/`，不是配置旧cache_root的256场。
360×640视频几何为360×640，其余本批视频为432×768，按既有RGBD Video.frame回映原生网格、只除固定scale一次。
深度图使用同pair两role首24帧共同有效范围、signed log1p/TURBO，无分位截断，无效区域黑色；仅渲染已有深度，不作质量或主体评分。
插值回映不是独立测量，原始几何与模型输入未改。旧fit JPEG混有预测，本页没有引用。

## 文件与人工确认

原片：`secondary_datasets/anakin/trimmed/{id}.mp4`、`edited/{id}.mp4`，官方mask在`masks/{id}.zip`。
以上均位于 `/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/` 下，没有下载新文件。
本轮目录 `derived/pvst_sample_scene_review_v1/media/an_{id}/`：`real_full.mp4/fake_full.mp4`为原片软链接；
`real_clip.mp4/fake_clip.mp4/fake_GT_overlay.mp4`为本轮CPU生成的无音轨预览副本（宽≤640，仅预览缩放，独立PTS核验≤1ms）。
固定0/4/8/12/16/23帧的PNG和两类序列图也在该目录。复制到别处需携带媒体并解引用原片链接。

|pair|人工确认问题1|人工确认问题2|
|---|---|---|
|7ogg04|全3秒是否明显后退/跟拍而非轻微振动？|首0.96秒的主体移除及GT覆盖是否符合希望研究的活动/遮挡内容？|
|728qx7|完整约5秒是否基本固定且无切镜？|首0.8008秒的微动作及附着纹身编辑是否符合优先活动内容？|
|4wdo37|全段平移是否超出允许机位运动？|首0.4004秒多人跑动是否过短或复杂？|
|95qvg9|全4秒机位及正常车流/背景步行是否可接受？|包含行人与周围背景的官方矩形编辑GT是否符合监督范围？|

`sample_review.csv`区分事实字段、画面初审和人工待确认，所有human_confirmed均为false。
CSV中的GT平均面积为0–1比例，上表换算为百分比；它不是场景评分或准入阈值。
完整事实/路径/SHA/PTS在`facts/*.json`，主观描述在`visual_notes.json`，没有数字场景评分。

媒体处理累计**8.742秒**（两个CPU阶段3.834＋4.908，不含编写、静态图审阅和发布），远低于约10分钟范围；下载0字节，模型/前端/GPU调用0。
4项必要检查通过：实际PTS片段时长、mask数值排序与缺失区分、叠加不扩展GT、候选稳定排序；所有预览视频已完整解码并与原片相对PTS核对。
只提交本轮脚本、检查、新报告和小型CSV；视频、PNG、mask、几何及用户原有JSON不入Git。

复算/重建入口：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_pvst_scenes.py --add-candidates --reason 'current camera/activity awaiting confirmation; fixed metadata/seed candidates'`。
已有合法媒体/事实会复用；只重建HTML可用`--render-only`。本轮到此停止，不启动任何拟合或成对更新。
