# V8 第二数据集资产核验

2026-10-26；只核验数据，不冻结联合训练方法、规模或新正式split。本轮无训练、评分、
MoGe/CoTracker提取或新模型安装。核验结论：**优先继续ANAKIN小规模人工确认；TVIL空间资产仍待核实。**

## ANAKIN规模与选择

[固定官方revision](https://huggingface.co/datasets/AlexBlck/ANAKIN/tree/978fe26e78ecaaf6562aa249be43c2c690484d0a)
`978fe26e78ecaaf6562aa249be43c2c690484d0a`。实际metadata为1023行/1023唯一video-id，
425个has-masks、974个full-available。1013个不同原始URL，8个重复URL组共多出10个编辑版本；
完整URL/裁剪位置/编辑关系在外部metadata_audit.json，不能按video-id假定source隔离。
本批20个候选的原始URL彼此不同。

|配置|train|validation|test|解释|
|---|---:|---:|---:|---|
|all|681|98|195|先过滤full-available|
|no-full|716|102|205|全部1023条，不取完整长视频|
|has-masks|297|43|85|只过滤has-masks，不取完整长视频|
|full-masks|297|43|85|同时过滤full-available和has-masks|

官方loader对过滤后的CSV顺序以Python random seed47打乱，再按int(0.7N)、int(0.8N)切分。
配置之间重新打乱，因此同video-id在不同配置的split可能不同；选择清单记录各配置split。
has-masks/full-masks在此revision包含相同425条，不能相加为850个视频。
[ICCV论文4.1](https://openaccess.thecvf.com/content/ICCV2023/papers/Black_VADER_Video_Alignment_Differencing_and_Retrieval_ICCV_2023_paper.pdf)
写1042条（352 VideoSham + 690 Vimeo）及80:20实验划分；当前官方card/metadata为1023条
（card写352+671），现loader为70:10:20。差异原因未确认，不将论文规模/协议冒充发布规模/划分。

全量编辑类型：swap272、splicing228、inpainting190、frame-level172、audio161。
带mask部分：swap161、splicing150、inpainting114。
本轮先冻结选择规则再下载：has-masks官方train中，按描述筛人物、肢体/穿戴或交互相关
splicing/inpainting，排除文字、图案、调色、背景整体替换、水体/烟雾等词；只作为下载前过滤，
不代表画面合格。74条候选按video-id排序后，以独立seed20260909随机取10 splicing和10 inpainting。
精确规则、正则、URL、metadata/loader SHA、revision及60个相对路径见
[evidence/anakin_selection.json](evidence/anakin_selection.json)。未看fake挑明显异常，未换样本凑数。
当前是资产核验池，尽管来源split可准确复现，也不是新冻结的正式训练集。

## 实际媒体、mask和空间对应

实际下载20对trimmed/edited及20个mask ZIP，共60文件，104,507,965字节，均通过固定revision
文件大小与官方LFS SHA（或Git blob SHA）校验，另记本地SHA。没有下载完整长视频或全部数据。
全部40段实际解码成功；外部media_audit.json逐段保留实际帧数、分辨率、平均帧率、time_base、
逐帧PTS及逐ZIP成员数量、帧号范围、原始编码/值域。不是只读容器声明帧数。

20套mask均为RGB PNG，实际含软边界和非0/255值。原始ZIP不改。
统一解释为官方所定义的**编辑区域**：RGB按PIL标准亮度转L，再/255，以>=0.5二值化，
白色为编辑前景；抽样原图/edited/GT叠加支持该含义。未侵蚀、膨胀、平滑，也未用real/fake差造GT。
部分GT呈粗矩形，尤其添加帽子、移除人物；不能宣称为精细轮廓或可见异常区域真值。
预览缩小仅用于显示，核验/GT均保留原尺寸；没有必要的GT空间变换，原生到fake坐标为identity。
已有V8最近邻mask变换/回映仅做合成用例验证，不生成训练标签或dense特征。

17/20套mask有完整连续0起始编号，数量与edited帧数完全一致、原生尺寸相同；固定时刻叠加
通过位置检查。此结论是结构对应及抽帧检查，不保证人工标注逐像素准确或整段无瞬间偏差。
3套保持alignment_unresolved：

|ID|trimmed帧|edited帧|mask数/编号|未解决问题|
|---|---:|---:|---|---|
|4qwe54|150|149|150，0..149|mask比edited多1帧|
|4vo387|150|144|144，6..149|数量相同但编号起点6，缺明确映射，不能自动减6|
|9g2gk7|150|149|150，0..149|mask比edited多1帧|

不截断、不复制、不插值，不将缺失标注当正常空mask。对应未解决项不叠加GT到fake。
另4dvw24的trimmed180帧、edited179帧、mask179帧：mask与edited可对应，不能把real帧序号
误当严格同时间像素对应。预览明确记录两段各自PTS，原图对照按解码序号展示，不声称像素差真值。

## 时间标签含义

start-time/end-time是原视频裁剪位置，不是edited内的伪造时间范围。
只有上述17个结构对应成立的视频才导出每帧mask有无前景：16个全程非空、1个空/非空混合
（4q26w4）、0个全空；其余3个未解决，不计作负帧。
连续正帧的首末解码序号和真实PTS在各视频JSON中；区间末端明确是最后一个正帧的采样PTS，
不臆测连续时间终点。这里只说明编辑区域是否存在，不将edited中空mask帧称原始真实视频。
唯一有空/非空转换的样本又不符合本轮相机场景，因此不能称已获得复杂时间边界监督。

## 场景预审与人工入口

实际查看原始trimmed固定时刻帧序列；另看real/edited/GT叠加核对标签语义。
13对初审符合、4对待播放确认、3对不符合。13对中有2对mask对应未解决，
故**11对同时初审符合场景且mask结构可对应**。这是预审，不是最终入选训练结论。
稀疏画面不能证明整段无切镜；索引保留全段播放器，仍需人工确认相机运动和瞬间切换。

不符合：4q26w4体育场明显平移；4qwe54街景视角明显变化；4wge37大面积瀑布主导。
待确认：4qwn54滑板跟随幅度、4xeqd7河岸手持及水体影响、4ymrn4室内视角变化、
4z1827大量前景枝叶遮挡及相机运动。均依据原始画面，未按fake篡改形变排除。
正常人体舞蹈、跑步、讲话和肢体非刚体运动本身不作排除条件。
逐项人体、相机、切镜观察与待确认事项见
[evidence/anakin_scene_review.json](evidence/anakin_scene_review.json)。

预览绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/anakin/preview/index.html`

每对有两个播放器、编辑描述、固定25%/50%/75%帧及首个非空GT帧（明确按GT选取）的
原图/edited/GT叠加，未解决项不画错位GT。没有任何模型热图。媒体、预览和缓存不进Git。

## 与ActivityForensics的差别及判断

本批为10个splicing、10个inpainting；包含眼镜/帽子/胸针/纹身等添加及人物移除。
较多稳定说话人物镜头符合固定相机场景，但局部外观装饰、粗矩形GT不等于人体结构异常。
ActivityForensics为六种生成器的修改视频，提供真假与时间区间；公开资产未提供可直接
使用的空间mask/框。ANAKIN提供公开编辑区域GT，即使编辑自然也仍为正标签。
两者监督含义与编辑分布不能无条件等同，更不能将本轮结果写成三维方法有效性的证据。
建议先人工确认11对结构可对应且初审符合的候选，再决定目标一致性及是否开展联合实验；
当前不冻结模型、loss、训练规模、新split或样本资格，不扩展本轮20对下载。

## TVIL公开资产核查

[官方UMMAFormer仓库](https://github.com/ymhzyj/UMMAFormer)说明原始来源为YouTube-VOS2018，
XMem生成修复mask，E2FGVI/FGT/FuseFormer/STTN制作修复视频；这是制作过程，不能证明mask已公开。
[官方配置](https://github.com/ymhzyj/UMMAFormer/blob/main/configs/UMMAFormer/tvil_tsn.yaml)
实际引用`./data/TVIL/annotations/metadata.json`及`feats/tsn`的rgb/flow `.npy`，
能确认代码支持时间定位特征/JSON路线，但本轮没有取得该时间JSON实际内容，不能列出实测字段。

官方网盘入口：
- Baidu：<https://pan.baidu.com/s/1h3sHu56z3slJnPCH47QRsg?pwd=8tj1>，提取码8tj1；本机TLS失败，外部读取也不可用。
- OneDrive：<https://1drv.ms/f/s!AgVmq0AY0Su8grknz_DKYQN2IbA63Q?e=hi5mbj>；HTTP200并重定向
  到personal/BC2BD11840AB6605目录。实际412113字节为网页应用壳，无可清点文件列表；
  公开shares/root/children接口另遇TLS失败，外部网页工具也无法读取目录。

因此原始real/最终fake是否都可下载、时间JSON是否有明确区间、是否实际公开逐帧空间mask、
mask到最终fake的帧/尺寸对应、包体量，**均尚未核实**。不能断言只有特征，也不能断言已有空间GT。
最短后续步骤是在浏览器打开上述官方共享目录，清点文件名和大小；如需登录，用本人账号在
网页完成，再提供目录清单或少量公开样例的本地路径。本轮不索取密码/cookies，不下载完整包，
不跑XMem、不把YouTube-VOS标签当fake真值。TVIL尚不足以替代ANAKIN的已实际取得样例。
访问响应和失败类型保留外部references/TVIL_access.json与TVIL_directory_attempts.json。

## 复现、成本及提交边界

使用已验证V8环境，未新增依赖。脚本只用现有huggingface-hub、CSV、PyAV、PIL及NumPy。

```bash
V8_WORK=/media/njupt/8FCD3869E2FAB06F/chenyh
V8_REVIEW="$V8_WORK/data/sparse_3d_forgery_detection/secondary_datasets/anakin"
"$V8_WORK/envs/v8/bin/python" -m scripts.review_anakin select --root "$V8_REVIEW" --manifest docs/v8/evidence/anakin_selection.json
# 下载使用官方客户端；本机CDN TLS失败后，仅本进程改镜像端点，仍按官方固定revision SHA校验。
(
  export HF_HOME="$V8_WORK/cache/huggingface" HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
  "$V8_WORK/envs/v8/bin/python" -m scripts.review_anakin download --root "$V8_REVIEW" --manifest docs/v8/evidence/anakin_selection.json
)
"$V8_WORK/envs/v8/bin/python" -m scripts.review_anakin review --root "$V8_REVIEW" --manifest docs/v8/evidence/anakin_selection.json
```

references保留固定metadata、ANAKIN.py和文件API清单；select会校核所选metadata/loader SHA记录，
不执行旧loader、不改变torch。已完整文件先复核再复用，不下载full目录。
下载最终镜像续传阶段14.55秒（复用此前已下部分文件，不能算60文件全新下载吞吐）；
首次CDN失败及等待未持久计时，记未知而不是0。首次完整核验/预览21.41秒；
后续为加入场景备注及PTS统计进行一次完整重建，实际21.56秒。
媒体/ZIP共104.51MB；整个外部目录含网页证据/预览在检查点为111,061,736字节；少量审计文件重写可能改变字节数。
小型完整性及逐样本证据见[evidence/anakin_asset_review.json](evidence/anakin_asset_review.json)。
7项定向测试通过，覆盖数值排序、缺帧/空mask区别、mask阈值、回映和真实PTS范围。

此前ActivityForensics恢复任务自然结束，状态文件分别记录v1.2 val28/28、train16/16目标恢复，
本轮未改动其数据、日志或进程；其全库最终统一验收仍另列，不抢占本轮第二数据集任务。
