# 候选伪造数据小批预览（2026-11-08）

本轮是素材核验，不改变 RGB-D 切片方法，不运行模型、前端、训练或评分。
基于 `pvst-rgbd-slice-relations` 的 `15e925125f680b517b0eb7982f99849a88c50202`。
原有无关 JSON 修改保留。只新增一个下载／解码／静态预览入口及直接相关检查。

数据根目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/action_forgery_preview_v1/`

打开其 `preview/index.html`；`samples.csv`、`results.json` 保留身份、SHA256、实际帧数、尺寸、FPS、完整 PTS 与对应限制。
媒体均未裁剪或转码；AF 使用相对页面路径指向目录内软链接，其目标为现有本机原文件。
复制预览目录到其他机器时需连同媒体并解引用这些链接；这批并非独立复制包。

## 取得的样本

| 数据集 | fake | real／官方指向的源片 | 结果 |
|---|---:|---:|---|
| FakeParts | 4 | 4 | 3 ProPainter inpainting＋1 Framer interpolation；完整解码通过 |
| HFV | 0 | 0 | 官方镜像及 HF 大文件下载中断；未取得可解码媒体或包内 CSV |
| TT-DF | 3 | 3 | 三种方法各一条 Match，统一 c40、不同原始 ID；完整解码通过 |
| ActivityForensics | 2 | 2 | 复用本地官方 fake 和准确对应的 Charades 原片；完整解码通过 |
| 合计 | 9 | 9 | **18 段**；不凑足约24段目标 |

选择规则先于媒体解码固定：各类别按官方 ID／路径排序，用 seed20260909 打乱。
FakeParts 固定3条 ProPainter、1条 Framer；其 metadata 没有人物类别或动作描述，无法按官方语义筛出3条人物局部编辑，不假装满足这个子目标。
TT-DF 固定 Match，每种方法按该顺序取首个未用源 ID；不把同一来源的压缩版本重复计样本。
AF 使用官方 train.csv 的划分及现有可读 Charades 原片，按 video_id 排序打乱，取不同来源2条；文件名中的 Charades train 标签不作为 AF 划分。
没有按模型分数、fake 异常程度或相机运动更换样本，也未顺延补位。

## FakeParts：编辑类别存在，所选像素 GT 未发布到可取资产

固定 HF revision `2ee8f98a94ef41e25cf548b6763570d125ce0f5d`；
[官方入口](https://huggingface.co/datasets/hi-paris/FakeParts/tree/2ee8f98a94ef41e25cf548b6763570d125ce0f5d)。
读取完整 `metadata.jsonl`（85,115行，SHA256核对通过），及 ProPainter、DiffuEraser、ROVI、Framer manifests。
所选 metadata 与 manifest 的文件名、帧数、FPS一致。

| fake ID | 官方指向的源片 | 实际 fake 帧／尺寸 | 来源限制 |
|---|---|---|---|
| propainter:97ba838008 | youtube_vos_2019/97ba838008 | 36／1280×720 | 同名源，manifest 标记 unresolved |
| propainter:8b523bdfd6 | youtube_vos_2019/8b523bdfd6 | 30／1280×720 | 同名源，manifest 标记 unresolved |
| propainter:a378053b20 | youtube_vos_2019/a378053b20 | 29／1270×720 | 同名源，manifest 标记 unresolved |
| framer:7042 | tenk_youtube/7042 | 46／512×320 | partial；生成帧选取未核验 |

三个 inpainting 源与 fake 实际帧数及尺寸相同，**这不能证明编辑区域对应**。
Framer 源为151帧、640×360、约5.038秒，fake为46帧、约6.571秒，裁剪／缩放／时间对应无可核实映射。
其公开时间标签是零起始、两端包含：real帧[0,15]、[30,45]，generated帧[16,29]；不是空间标注。

本轮查看的 ProPainter 2244、DiffuEraser 2243、ROVI 9090条 manifest 的 `mask` 均为 null。
作者把 YouTube-VOS 实例标注目录列为候选来源，但明确写明实际移除目标／生成mask未核验。
固定 revision 下 API 清点 `assets` 只有 images、prompts、readme、videos；README 图示的 `assets/masks`、`assets/frames`、`assets/provenance` 查询均404。
因此本批**连可读取的源实例mask也未取得**，目标实例ID未知，不存在将所有非零实例合并或插值成GT的操作。
这些证据针对所查 revision 和样本，不推断论文或未来发布资产必然没有mask。

固定50%源图观察：97ba838008为水池／前景观众头部；8b523bdfd6为滑板人物；a378053b20为猫头鹰；7042可见手持工具作业。
没有因为出现非人体内容而替换样本。仅滑板片明显含完整人体活动，手部作业也是人体交互线索；本批不足以完成“三条人体局部动作修改”目标。
实际移除对象未公开，不能把 inpainting 类别写成人体动作局部修改。

## HFV：公开定义是视频类别，本轮包内标注未核验

HF revision `7a567eef47636ed2d3729d20ad7551bc73b79616`；
[作者仓库](https://github.com/dejian-lc/humansam/tree/46dd9d8f780d6b2e61d0927c0c19d7318dcdeaaf)
指向官方 ModelScope 镜像 DawnOfDark/HFV。
训练 ZIP 1,614,774,372字节（约1.50GiB），低于5GiB限制，未下载其他评测包或权重。
ModelScope文件 revision `6cb8e3e664af174708ce912006769bcdf6ad8cc1`，其SHA与HF一致。

[HumanSAM 官方论文 §3.1](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_HumanSAM_Classifying_Human-centric_Forgery_Videos_in_Human_Spatial_Appearance_and_ICCV_2025_paper.pdf)
定义0＝spatial anomaly、1＝appearance anomaly、2＝motion anomaly、3＝real。
README说明 `_4.csv` 为四分类、`_2.csv` 为二分类，列为文件名、标签及可选置信度；作者数据读取代码也已只读核对。
但**包内实际CSV尚未取得，不能宣称已核验其逐行标签或完成2空间＋2运动＋2real抽样**。
官方介绍的生成视频与real通常独立，不构造同源配对。
本轮没有取得空间mask或异常起止区间；也不能因下载失败而断言ZIP内部绝无这些文件。

## TT-DF：人体动画，c40包中只有视频

[官方仓库](https://github.com/HashTAG00002/TT-DF/tree/6010d790fd6bb99992adb91ab9857f96d5253e05)
链接的 [Drive目录](https://drive.google.com/drive/folders/1O1ZwHpSgTLZDjMFpmX7_IfNGTuVU_9s6)。
仅下载 c40.zip，Drive ID `1eI8iRCPQb0Z2jmnIMUi1qeBRi8ih1jAn`，386,697,630字节。
Drive不提供本次可固定的内容revision，另记录包SHA256和精确成员路径；raw、c23包均未下载。
清点完整中央目录2390项，非目录成员全部为mp4，无mask、逐像素标签、独立时间区间或额外身份映射文件。
只提取所选6个媒体成员，并校验其CRC；不解码或抽取包中其他视频。

| 方法／配置 | source／身份参考来源／动作来源 | fake／原片实际帧数 | fake／原片持续时间 |
|---|---|---|---|
| MagicDance／Match | 00247／00247／00247 | 265／265 | 8.833／8.833秒 |
| MagicAnimate／Match | 00139／00139／00139 | 123／123 | 4.920／4.100秒 |
| AnimateAnyone／Match | 00076／00076／00076 | 53／53 | 1.767／1.767秒 |

均512×512，原片指包内 c40/original，不冒称未压缩raw。
MagicAnimate虽同帧数但fake为25fps、原片30fps，不能按相同PTS直接配准。
本批没有选Mismatch，因此不猜测双ID文件名中哪个是身份、哪个是动作驱动。
这些是基于人体参考／动作驱动的生成，不是带官方局部编辑mask的动作修改样本。
Match身份可追溯也不代表已经提供精确像素对应或独立伪造区间。

## ActivityForensics：活动改变与时间监督

固定官方revision `a34d4b7b04b0f3f3e26ba900adc367218667c581`。
原始train/test CSV及annot.zip核对官方checksum；两条fake核对官方文件checksum；两条源片核对本机Charades官方归档的尺寸与CRC。

| source／fake ID | 原始动作（Charades描述） | 修改及官方区间（秒） |
|---|---|---|
| NK9RX／35e67923a90fe0a63cd080d1 | 沙发上使用笔记本电脑、从杯子饮用 | activity deletion；[5.1,10.5]、[13.2,28.2] |
| QNKDI／9c722cd7fd373d43a0b2da29 | 床边看电视、读书 | activity deletion；[18.3,32.21] |

具体删去哪项动作的自然语言prompt未出现在现有官方标注中，不自行补写。
区间是官方秒标注，未公开明确端点包含约定，不擅自改写成逐帧闭区间。
NK9RX：fake与原片均739帧，fake1920×1080、源480×270；QNKDI：fake968帧、源967帧，均480×360。
不把源/fake自动当作像素级配准。
重新清点官方非MP4清单及annot.zip，公开标注为时间文本，无空间标注文件；未读取feat.zip或旧模型/特征/分数。

## 成本、检查与判断

TT-DF包下载629.577秒；FakeParts八个媒体成功下载14,834,093字节。
成功媒体／压缩包合计401,531,723字节（约382.93MiB）；metadata＋四份manifest共45,242,544字节。
HFV两次HTTP失败已记录写入字节合计28,311,552；探索请求及最初metadata失败响应没有完整计量，因此这些是可核查下载量下限，不冒充全部网络流量的精确总值。
ModelScope请求约153.968秒超时；HF HTTP约357.293秒中断；最后使用已有huggingface-hub精确单文件下载，600秒无媒体字节落盘，由timeout终止，真实退出码124。
没有继续重试或另取非官方视频。逐请求记录和失败原因保留在数据目录。
18条媒体的两遍CPU解码／固定截图合计约4.71秒，均有单调实际PTS，未运行GPU。
时长按最后PTS加末帧间隔估算；完整原始PTS/ticks另存，不将容器声明帧数当实际帧数。

4项小型检查通过：选择不受目录顺序影响、缺mask不变空GT、现有成员CRC核验、HTML相对路径与缺GT展示。
18条真实媒体完整解码是本轮集成核验；没有对整库做媒体验收。

**当前没有一条可直接用于像素伪造定位监督的已核验GT。**
本批中AF最接近“真实活动被改变”的研究内容，可继续人工审阅已有两对的活动变化，但它只有时间监督。
TT-DF适合审阅人体驱动生成，不能与局部动作编辑等同；FakeParts应先向作者核实最终编辑mask、目标实例与帧映射，再谈空间监督；HFV需先完成官方媒体包获取及CSV核对。
不因mask承诺、画面显眼或人体出现就推断适合三维机制。正式方法、配置、模型结果及用户JSON均未改动。

复算入口（已有各官方metadata／小包，缺失HFV时保持缺口）：

```bash
PYTHONPATH=. /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python scripts/prepare_action_forgery_preview.py \
  --root /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/action_forgery_preview_v1 \
  --base /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection
```

入口不会自动下载HFV/TT-DF大包，先人工核对包大小及官方入口；不会补齐缺失GT。
单纯重建页面可加 `--stage render`，有效媒体及已核验解码记录会复用。
