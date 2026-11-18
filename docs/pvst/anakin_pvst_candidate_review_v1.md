# ANAKIN：PVST候选数据人工审阅准备

状态：CPU媒体与mask核验完成，首批12对预览已就绪；新探针资格与规模停在人工确认。没有运行三维前端、特征、模型、训练或评分，没有改变原划分与成绩。

## 来源并集与身份

来源去重后20个独立登记source URL、20对、40视频：训练16对（原8＋新增8），开发仍4对。source_id与原始URL/媒体SHA复核；20个real媒体SHA亦互异，同源real/fake角色不跨train/dev。总计完整解码4965帧，fake及官方mask2482帧。40条原片/伪造片均H264，全部已有，新增下载0字节。此处“原片”是作者发布的trimmed来源短片，不是完整Vimeo长视频；URL身份不代表跨来源物理场景绝不重复。

实际分支pvst-geometry-dataset-screen，起点65726e50abf325b2bcbae573723a552e82765d21。所列docs/pvst/rgb_partition_anakin_scale_v1.md在该分支不存在，本轮只读derived/pvst_rgb_partition_anakin_scale_v1/report.md及其manifest/inventory身份；未从其他分支拼接代码。原8/开发清单与扩量清单重复项逐字段核对后去重。

## 标注与时间边界

20个fake的官方mask ZIP SHA、逐帧数量、数字编号、原生尺寸均通过；逐帧完整解码PTS与冻结manifest一致。mask按既有mask_array灰度uint8/255≥0.5口径统计，原标注不修改。编号为官方顺序映射，没有独立mask PTS；因此核实的是文件索引、尺寸和视频时标对应，具体mask是否覆盖实际编辑内容仍交人工检查。缺失标注不填空mask。

19对全程mask非空，1对有空/非空变化：**an:40y1x9**，150帧，非空区间[0,6)、[10,150)，空区间[6,10)。帧6 PTS=0.200000s、帧10 PTS=0.333333s。可用于人工检查mask存在状态转换的候选数=1；已独立核实内容编辑开始/结束真值的数量=0。两处状态切换不自动等于真实编辑撤销/恢复，也可能与对象进出或标注范围相关。应优先检查该段画面和mask，不能仅凭空mask解释为正常。原开发四fake仍全部全程非空。

14对real/fake完整帧数和PTS网格相同；6对存在数量或时间网格差异，表中列明。即使时标相同，也没有据此宣布真伪像素内容精确配准；不同步组页面明确限制，只按片内相对时间联动，独立身份PTS均保留，未计算真假像素差分。

## 首批清单及其余初筛

先对20对全部核验，再按已登记规则选8训练＋全部4开发。训练排序：部分时段mask优先、明确运动对象增删类优先、原片自动稳定候选优先、pair稳定排序。未读取模型成绩、预测热图、深度或异常分数。首批含有移动机位样本供人工排查，不把“入预览”当成“符合固定机位”。较低优先级样本仍在完整清单中；外观/配饰不等于无关系信息。

|pair|原角色/来源组|官方描述|机位依据|标注时间型|首批|真伪PTS网格相同|

|---|---|---|---|---|---|---|

|an:40y1x9|原train8|remove a dog running towards a person|可见全局运动迹象（非切镜定论）|partial|是|True|

|an:48o2l9|新增train8|add a tie over the vest|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|True|

|an:4dvw24|新增train8|add glasses to the face|不确定：可靠背景匹配不足|all_edited|否|False|

|an:4nyz37|新增train8|add the crown on the head|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|True|

|an:4rvro9|原dev4|add a hat to your head|固定（明确人工记录）|all_edited|是|True|

|an:4wdo37|原train8|Remove the guy with the ball in his hands  who is running away from everyone.|可见全局运动迹象（非切镜定论）|all_edited|是|False|

|an:4wgm37|新增train8|remove 5-7 people in the reflection on the sculpture|可见全局运动迹象（非切镜定论）|all_edited|是|True|

|an:4wgx37|原dev4|remove the guy on the bike|明显跟随主体（明确人工记录）|all_edited|是|True|

|an:4z1827|原train8|add a spider to the hat|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|True|

|an:728qx7|原train8|Give a guy a tattoo on the right side of his chest|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|False|

|an:7e23j7|新增train8|Add any brooch to the blouse on the right side|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|True|

|an:7e2557|原train8|add a brooch on top of a shirt pocket|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|True|

|an:7e2e57|原dev4|Add a cat walking across in front of a woman with a dog|固定（明确人工记录）|all_edited|是|True|

|an:7ogg04|原train8|remove girl without umbrella|可见全局运动迹象（非切镜定论）|all_edited|是|True|

|an:7ogo04|新增train8|Remove the man in the white jacket|不确定：可靠背景匹配不足|all_edited|是|False|

|an:7p8r14|原train8|add panama hat|较稳定候选（传统背景匹配，待人工确认）|all_edited|否|False|

|an:95q8g9|原dev4|remove the rider in the white hat along with the horse|固定（明确人工记录）|all_edited|是|True|

|an:95qlg9|新增train8|remove the person on the blue ball|可见全局运动迹象（非切镜定论）|all_edited|是|True|

|an:95qvg9|原train8|Remove a woman and a man walking on the background|较稳定候选（传统背景匹配，待人工确认）|all_edited|是|True|

|an:9l8w54|新增train8|Remove the walking man in the blue shirt on the left side|较稳定候选（传统背景匹配，待人工确认）|all_edited|是|False|

## 机位及内容的证据边界

已有明确人工记录优先使用：7e2e57固定、加入猫；4rvro9固定、加帽；4wgx37跟随主体、移除滑板男子和滑板；95q8g9固定、移除左侧骑马男子。来源为用户此前按pair明确口述；这些是人工可见线索，不是三维异常真值。官方4wgx37描述“bike”和人工“滑板”、95q8g9官方含马匹移除与人工描述差异，原文并列保留，待本次人工确认，不自动更改标签。

其他组使用已有camera_check：原片5个均匀帧、外围背景ORB/Hamming、部分仿射RANSAC。排除中心区域并不能保证全是静态背景，自动固定倾向不是人工批准；匹配不足记待确认，不宣称移动。可见全局变换也不等于已确定跟拍或切镜；页面保留原指标和限制，不把相机变化视作伪造证据。未声称人工播放确认了全部视频。

另有四条自然编辑只知道“3固定、1轻微移动、均协调难辨”的批次记录仍保存在../temporal_controls_v1/human_review_batch.json，本轮没有把它按数量、排序或自动机位结果分配到任何ANAKIN ID。不能与上面四条明确ANAKIN口述混为一批。

建议先查看40y1x9的mask空档及运动对象进出，再看95qvg9、9l8w54的固定倾向运动人物移除，以及开发7e2e57、95q8g9。理由来自编辑描述、标注状态及机位初筛，不来自模型成败。其他首批样本仍完整保留。

## 预览与人工确认

index.html按pair ID展示12对完整real/fake短片，支持同步播放、fake RGB/叠加mask/仅mask切换，编辑mask存在时间轴和固定首/中/末帧；有状态切换时额外列切换帧。原始视频用符号链接引用已有文件；叠加和mask预览仅缩放至宽≤640，原生数据不变，所有新预览编码后重新完整解码并核对每帧PTS。无模型成绩或热图。

人工仅填三项：相机是否符合固定或近固定；具体对象和运动/遮挡/形态；mask/时间轴是否对应画面。表单保存在当前浏览器并可导出JSON。已有人工作为旁注，三项本轮确认不自动预填通过。页面资源相对链接核对通过；需在原服务器访问，链接原片不是独立打包副本。

当前用户提出的等尺寸XYZ块、联合块内编码和学习式空间/因果时间关系是后续实现待定路线，本轮数据整理不宣称这些已经实现，也不冻结块边长、维度、层数或窗长。旧点体素、深度整片、方格、集合与读出结果全部保持原样。

## 执行与产物

执行入口scripts/anakin_candidate_review.py；1项针对性CPU测试通过（并集去重/来源划分冲突、mask缺号不修补、编号口径和区间）。媒体/标注核验与预览耗时40.194秒，退出0；后续页面资源链接/来源说明单独补齐，没有重跑模型或前端。无下载、GPU、三维计算、特征提取或训练。

执行核验HEAD a5fdd34，页面/来源说明HEAD d5bc327。完整原PTS、官方revision、源URL、视频/ZIP路径与SHA在inventory.json和各预览identity.json；inventory.csv含全部20对，selection.json含12对及其余8对，protocol.json含清单SHA和固定排序规则。未新增数据到任何正式训练/开发协议。

输出绝对目录：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_candidate_review_v1

完成后停止。计算/预览状态与Git发布状态分开。
