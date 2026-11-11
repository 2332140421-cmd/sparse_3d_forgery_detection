# 冻结局部像素特征读出诊断

状态：COMPLETE，真实退出码0。本轮只拟合一个训练像素线性读出；主模型不训练、不改权重/缓冲，不重提前端，不访问保留测试集。

## 开发pixel AP主表

| pair | 正GT帧数 | EXISTING_ZERO | LINEAR_LOCAL | 线性−原头 |
|---|---:|---:|---:|---:|
| an:7e2e57 | 120 | 0.036868 | 0.029659 | -0.007209 |
| an:4rvro9 | 150 | 0.869735 | 0.709064 | -0.160671 |
| an:4wgx37 | 120 | 0.257167 | 0.326748 | +0.069581 |
| an:95q8g9 | 72 | 0.063102 | 0.100318 | +0.037216 |
| **4视频宏平均** | **462** | **0.306718** | **0.291447** | **-0.015271** |

## 直接判断

两条重点漏检没有同时改善：an:95q8g9的AP由0.063102升至0.100318（+0.037216），说明冻结局部特征中存在本次可线性读出的部分编辑排序线索，但绝对AP仍低；an:7e2e57由0.036868降至0.029659（−0.007209），没有提供“只换读出即可改善”的证据。不能给两条视频统一归因。

an:4wgx37改善+0.069581；an:4rvro9下降−0.160671。共两升两降。宏平均下降主要受4rvro9驱动；其余三条平均差为+0.033196，仅用于说明平均值来源，主结果始终包含全部4条。线性读出没有形成跨视频一致优势。

**唯一优先下一步：针对an:7e2e57核查局部特征对实际编辑内容的表达。** 本次没有读出稳定线性证据，不代表特征中完全没有信息；95q8g9的读出改善也不足以证明原像素头是全部漏检的主因。本轮不追加方格、重训、阈值校准或其他读出。

## 冻结身份、层位置和一次双读出

基于a94e1ce3e8e30cb2f6783775a7481e44b03d3cec。执行分支pvst-local-pixel-readout-probe，执行HEAD `414af1e9840f7c64a218e301a9a75aa0e3cbeb89`。

唯一权重：原pvst_rgb_partition_anakin_dev_v1/RGB_DEPTH_PARTITION/last.pt；condition=RGB_DEPTH_PARTITION、epoch12、updates576；SHA256 `3aa4fc6c9ca4b6c8f7422c08f49ac056bffb63b2557f9c46e938d197936cca7b`。冻结清单SHA256 `cf8bfc66a6d4f5590eb3157358ee40271ae3638a3ef0fa4339c0590300fd4272`。原配置、code哈希、train/dev媒体与缓存、官方mask身份均核对，source_id隔离。共同初始化hash核对后严格加载state_dict。

源码Local.decode先把局部RGB多尺度特征与原图细节融合，经Local.native得到[1,32,H,W]，随后才加context，再调用local.final。只读native forward hook取上述实际32维原生像素特征，不换层、不读XYZ/深度/身份作为线性输入。模型仍完整运行分片、集合关系与GRU。

EXISTING_ZERO复用pixel_context(C+M)完整投影后zeros_like（含bias），保留局部视觉和原像素头。LINEAR_LOCAL从同一份native特征FP32打分。一次开发前向同时获取两者，无额外开发前向。训练和开发投影交付非零元素都为0；原主模型参数及缓冲state_dict hash前后相同，所有grad为None。

## 训练采样和唯一拟合

原train8对/16视频/48个固定24帧片段，1152唯一帧，每视频72帧。实际正GT帧572，空GT帧580。空GT包含576个real帧和4个fake帧，没有将所有fake帧强行当正帧。

每个唯一帧GT正/负区各最多128像素、无放回，不足全取、空类别不造样本；seed20260909，按SHA256(seed,video_id,frame_index)派生固定随机流。负区包括real及fake非编辑区，不按valid、模型分数、类别或面积筛选。实际正像素73,131、负像素147,456，共220,587；采样32维FP32特征约26.927MiB，仅运行中暂存，未保存整库特征。

不足128的正类：an:40y1x9帧6、7、8、9为空GT；an:7e2557帧90只有43个GT像素，全部保留。其他正GT帧各128。逐帧采样索引hash、PTS、GT计数见training_sampling.csv；逐视频覆盖与类计数见sampling_summary.json。

StandardScaler只用训练采样拟合；L2 LogisticRegression固定C=1、solver=lbfgs、max_iter=1000、class_weight=balanced，其余沿sklearn默认，版本见identity.json。迭代168次收敛，无ConvergenceWarning，未搜索其他设置。标准化和分类器均未接触开发数据。

将标准化折入线性系数后，以小数组和真实训练采样前1024点核验。真实折入最大绝对差2.68166849e-05，通过预设逐元素atol=1e-5、rtol=1e-5（允许误差为1e-5+1e-5×|参考值|，不声称纯绝对误差小于1e-5）。主模型保持BF16原路径，只有线性乘加局部关闭autocast用FP32。

## 评价与复现

原dev4对8完整视频924唯一计分帧，462正/462负。960帧次输入中的36个尾段左上下文帧不重复计分。记录视频ID、源帧号、实际PTS、GT像素数和native形状；与原冻结清单逐帧核对。所有分数有限。

pixel AP完全复用原sklearn average_precision_score：正GT帧在全原生画面评价，不采样、不按深度valid裁掉像素；先视频内帧均值，再4条fake等权。无编辑GT的帧AP保持缺失，不补0。

EXISTING_ZERO宏平均0.306718133002424，逐视频也与上一轮PIXEL_CONTEXT_ZERO完整精度相同，基线差均为0。native hook在小图真实解码路径中精确不改变输出，真实开发基线亦复现。

两种读出不比较logit0下Dice、召回或FPR；线性训练用了类别平衡，阈值不能直接视为等价。本轮pixel AP不验证real视频误报，也不替代检测或正式测试结论。

## 检查与成本

4项必要测试通过：训练专属/可复现/无放回采样及原生GT索引；折入FP32等价；尾段只计分一次与冻结训练边界；实际Local.native被动hook前后decode输出精确一致。真实运行再次核对1152训练帧、924开发帧、GT/PTS及冻结状态。

总墙钟226.002秒；训练采样/前向104.974秒，线性标准化+拟合+折入核验1.133秒，开发一次双读出前向及评价110.735秒。其余时间为身份校验/模型构造及冻结检查，不冒充训练时间。

GPU peak allocated 1.834GiB，reserved 21.574GiB；reserved包含分配器缓存，不等于同时活动张量。进程峰值RSS 5.578GiB。退出码0，无本任务残留进程。计算状态与Git发布状态另记。

## 产物与复算

产物目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_local_pixel_readout_probe_v1`。report.md、summary.json、dev_video_AP.csv、dev_frame_AP.csv、training_sampling.csv、sampling_summary.json、linear_coefficients.json、identity.json、execution.log、exit_code.txt、command.txt。线性系数/StandardScaler统计留在独立产物目录，无checkpoint、全量像素权重或特征图入Git。

实际命令：

```bash
PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/rgbd-packages:. MPLCONFIGDIR=/tmp/pvst-local-readout-mpl /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m research_tools.rgbd_slice.local_pixel_readout_probe
```

小型线性读出身份：训练采样特征与标签hash `e00c24fa365ef037114f4c8cafa97dc5001aec43b8355fba5d6c3ac677b32592`。原始媒体、GT、高分辨率几何和旧结果均只读，用户JSON修改保留。
