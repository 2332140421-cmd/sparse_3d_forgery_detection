# 新服务器 V8 准备与运行

后续资产任务状态见[data_assets.md](data_assets.md)。以下环境和8帧验证是上一轮记录；本轮不重复GPU验证、不执行试标或训练。官方补充材料1.4节已确认hcv1为HC-STVG v1，取代上一版Hollywood误归因。

2026-10-23实际验证；分支`v8-spatial-localization`，基线远端
`v8-temporal-supervision-pilot` / `9a9b6b8efc41a0c752b070c440ab3c6a69a4a663`。
fetch后该远端HEAD未变；原工作区干净。本轮不迁移旧服务器数值资产。

## 环境与目录

个人工作区`/media/njupt/8FCD3869E2FAB06F/chenyh`，沿用其`projects`仓库，新增
`envs/v8`、`external/{MoGe,co-tracker,utils3d}`、`checkpoints`、`cache`、
`data/sparse_3d_forgery_detection/{metadata,media,smoke}`；媒体和环境均在Git之外。
磁盘初查12TB可用。GPU0 RTX4090 24564MiB，驱动580.178.04，初查977MiB占用；
未结束其他用户进程或改共享驱动。沙箱不暴露GPU设备，GPU命令须在获准的主机执行环境运行。

Python3.12.7来自已有Anaconda基础解释器，独立venv不读取其他研究环境。
PyTorch2.5.1+cu124 / Torchvision0.20.1，CUDA runtime12.4。
完整实际依赖锁见[requirements-new-server.txt](requirements-new-server.txt)。
没有安装根pyproject中的VGGT。首次默认清华镜像TLS失败，改官方PyPI；
PyAV14.4.0无cp312 wheel，改用16.0.1二进制包，未编译系统FFmpeg。
未安装MoGe演示/训练、CoTracker Kubric或空间评价库。

```bash
export V8_WORK=/media/njupt/8FCD3869E2FAB06F/chenyh
export HF_HOME="$V8_WORK/cache/huggingface"
export TORCH_HOME="$V8_WORK/cache/torch"
export PIP_CACHE_DIR="$V8_WORK/cache/pip"
export XFORMERS_DISABLED=1
source "$V8_WORK/envs/v8/bin/activate"
cd "$V8_WORK/projects/sparse_3d_forgery_detection"
```

新建同环境：`/home/njupt/anaconda3/bin/python -m venv "$V8_WORK/envs/v8"`，然后
`python -m pip install --index-url https://pypi.org/simple -r docs/v8/requirements-new-server.txt`。
官方代码用git clone，分别checkout以下完整revision，再执行：

```bash
python -m pip install --no-deps --no-build-isolation "$V8_WORK/external/MoGe" "$V8_WORK/external/co-tracker" "$V8_WORK/external/utils3d"
```

这里刻意不装上游声明的demo/训练依赖；不是通用MoGe CLI环境。验证的是v2.infer和
CoTrackerOnlinePredictor实际运行路径。系统`python3`为3.10，不能替代此venv。

## 已准备资产与版本

|工具|官方代码revision|本地权重|
|---|---|---|
|MoGe-2|07444410f1e33f402353b99d6ccd26bd31e469e8|checkpoints/moge-2-vits-normal.pt|
|CoTracker3 online|82e02e8029753ad4ef13cf06be7f4fc5facdda4d|checkpoints/scaled_online.pth|
|utils3d|3fab839f0be9931dac7c8488eb0e1600c236e183|无|
|Torchvision ResNet18|torchvision0.20.1 / IMAGENET1K_V1|cache/torch/hub/checkpoints/resnet18-f37072fd.pth|

MoGe新下载默认代码是v3，已切回最后一段v2历史；未找到旧服务器精确revision，
不声称逐位一致。接口保持`from moge.model.v2 import MoGeModel`，1200 tokens、fp16。
CoTracker保持online/window_len16/32×32 queries，短历史与重复端点显式记录；没有offline替代。
ResNet18保持conv1至layer1冻结，224预处理后插值16×16、64维输出。

下载来源：[MoGe权重](https://huggingface.co/Ruicheng/moge-2-vits-normal)、
[online权重](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth)、
[ResNet18权重](https://download.pytorch.org/models/resnet18-f37072fd.pth)。
权重仓库读取revision及每份完整SHA见[evidence/new_server_smoke.json](evidence/new_server_smoke.json)；
下载时使用main，随后读取的revision是审计时revision，内容身份以已落盘SHA为准。
没有旧V8 checkpoint、标准化统计、分组统计或旧前端缓存；不伪造这些文件。

## 数据与空间标注核查

官方HF dataset revision `a34d4b7b04b0f3f3e26ba900adc367218667c581`。
实际下载dataset信息/文件清单、README、train/test CSV、144584字节annot.zip，
以及官方Charades小型标注包。全清单统计见[evidence/official_inventory.json](evidence/official_inventory.json)：
5389 fake，3730唯一(dataset,source)；train3824/test1565；6生成器；
Charades1438、ActivityNet(acnet)1832、HC-STVG v1(hcv1)2119。这里是来源数据族，
不是5389独立场景或3730拍摄地点。AF仓库本身没有real媒体；不能将fake当原视频。

每条CSV精确匹配官方媒体路径，再与annot中的split/generator/完整文件名匹配。
从所有编辑片段检查dataset/source/generator一致；支持带或不带数字编辑ID的两种官方命名。
pair_id不参与唯一索引；本地video_id由source+role+完整媒体路径派生，另保留实际视频SHA。
`annot.zip`每行是文件名、时长秒、`start=end+...`时间区间；这不是空间框或mask。
官方HF全5395文件清单除5389媒体外只有README/CSV/annot/feat等6文件，未见空间标注资产；
GitHub说明也未给空间标注下载。论文的grounding构造与图示不能证明空间GT已发布。
这仅是公开入口核查结论，不声称作者内部没有mask；空间标注分辨率/帧索引/坐标规范不适用。
官方时间区间描述被操纵活动的时间范围，不是人工逐像素可见异常真值。
依据：[官方数据入口](https://huggingface.co/datasets/ActivityForensics/ActivityForensics)、
[官方说明](https://github.com/ActivityForensics/activityforensics)、
[论文方法](https://arxiv.org/html/2604.03819v1)。未下载feat.zip作为前端替代。

本地仅一对1LARL：real来自官方
`https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip`
中的`Charades_v1_480/1LARL.mp4`。通过HTTP Range读取central directory及该member，
校验ZIP CRC，不下载16,339,546,533字节全包。real实际1,862,447字节；
fake为官方固定revision下`video/02_wan/1LARL+3.60=10.20=charades@train_delete@1LARL@214@wan.mp4`，
35,212,917字节，时间区间[3.60,10.20]秒。官方Charades train CSV确认source1LARL、
scene=Living room、subject=2RTW；同一source只作开发入口，绝非新增独立测试。
其他来源的真实视频应分别回到原始ActivityNet/Hollywood数据获取，不猜测或拼接real。

全视频仅做CPU顺序解码清点：双方各914帧，共1828帧，全原生PTS，未跑1828帧前端。
real270×480、time_base1/30000；fake1080×1920、1/11988；两者PTS并非逐位相同，
不能按相同帧号做像素差真值。媒体hash、元数据与最后PTS见执行证据。

## 已执行的小规模贯通与命令

```bash
python -m research_tools.v8.spatial_localization.prepare --metadata "$V8_WORK/data/sparse_3d_forgery_detection/metadata" --output /tmp/v8-inventory.json
python -m pytest -q tests/test_v8_spatial_localization.py tests/research_tools/v8/test_component_global_context.py
python -m research_tools.v8.spatial_localization.smoke --root "$V8_WORK" --output "$V8_WORK/data/sparse_3d_forgery_detection/smoke/another_run"
```

输出目录必须新建，不能覆盖已有导出。最终实测目录`smoke/localization_final`；
另有首次同样8帧调通目录`localization_v1`，两次均无训练，无额外媒体/全库提取。
从4秒开始连续4帧，各为120—123；保持原始PTS ticks和time_base。
完成letterbox、denseXYZ/validity、online tracking、cell/动态component、
GLOBAL_CELL/GLOBAL_COMPONENT两种319969参数随机模型前向/反向、padding扰动不变性、
响应裁padding并映回原图。输入x为[1,4,256,77]，XYZ[1,4,256,16,16,3]；
每帧160个内容格、36864内容像素，component数量45—52。8帧几何有效覆盖都是36864，
不代表其他视频也可靠。model梯度有限且非零；10项定向测试通过。
模型数值输出不作语义或性能解释。smoke使用identity标准化、grouping的1/1/1常量，
明确不是旧拟合参数，也不是为正式实验选定的分组阈值。未生成训练checkpoint。

8张原图+JSON构成人工标注入口；所有状态为unannotated、mask_path=null。
转换实现与回映测试已执行；没有真实mask，不能声称空间GT实际往返验证完成。
人工目标与导入规则、两条件架构及公平初始化见[spatial_localization_plan.md](spatial_localization_plan.md)。

## 新机成本：小样本测量和外推边界

最终8帧端到端5.598秒，含模型载入、导出等；GPU allocated峰值2,117,948,928字节(1.97GiB)。
前端各段计时和2.769秒，约0.346秒/帧（2.89fps，含dense写盘，不含完整原视频解码）。
若机械外推旧234284帧约22.5小时；短样本/初始化/视频分辨率/I/O差异使其只能作粗预算，
不是已执行全量耗时。旧服务器0.332秒/帧只作历史参照，不替代新机实测。

两条件各4帧的forward+backward+第二次padding前向合计0.142秒，16个condition-frame，
约0.00885秒/condition-frame。以此线性比例套用206779训练帧约30.5分钟/条件/轮，
但它包含额外检查且缺少完整视频反向/optimizer/重计算，不能当正式训练资源验收；
像素解码器尚不存在，其训练耗时和显存目前无法实测。正式预算须另做完整pair smoke。

当前单次dense/图像导出14,605,912字节，约1.83MB/帧；若按此留全量约428GB(398GiB)，
并非紧凑缓存成本。旧缓存51,161字节/帧约12GB只是格均值/掩码数据，
不能替代新像素分支所需的dense局部XYZ。原媒体约37.1MB，三权重约289MB；
环境/CUDA wheel、pip缓存另计。两次smoke导出各约14.6MB，未提交Git。
不能把随机小序列成本或历史视频弱监督成本声称为定位训练成本。
