# V8 完整视频弱监督微调

分支`v8-full-video-training`；起点`211fba5be2b22c6e4b7524ef509a22060f89967e`。
合同见`docs/v8/full_video_training_contract.md`及ADR-0018。

## 协议与实现

两阶段方法：匹配condition/seed的窗口epoch100预训练→完整视频epoch20微调。
GLOBAL_COMPONENT/GLOBAL_CELL各三seed；319969参数，真实全帧attention与多前驱GRU均保留。
冻结的244训练视频/122source、28验证视频/14source，共234284实际连续帧。
无前端重跑，无标准化/分组重拟合，无帧或块标签。完整视频面积LSE后一次BCE，
同source real/fake各半权重，统一clip/step。checkpoint跨块梯度不截断。

新增`model.py`为可微适配，不能复用inference_mode接口；`data.py`只读并核验
原紧凑缓存；`run.py`完成必要资源测试、原分数复现、训练恢复、评价及报告。
未引入通用训练框架；旧方法和结果不变。

## 已实测资源

完整1LARL real/fake各914帧；最长训练视频X7R9N fake为1829帧。
二者均完成完整前向/反向，未截断梯度或缩短输入。

|执行块|1LARL pair读入+前向+反向秒|最大allocated GiB|最大reserved GiB|
|---|---:|---:|---:|
|32|8.6406704504|0.944141865|1.214843750|
|64|7.6019999869|1.853305340|2.111328125|

资源指标选择chunk64；按该pair粗估六模型×20epochs约30.91小时，
不是已完成训练成本，也未计最终评价。各视频时长不同，实际成本由模型记录给出。
GPU为RTX4090D；FP32，未启用AMP或TF32。

## 验证与运行状态

定向测试检查两条件/多chunk冻结推理一致、padding隔离、缺失内容有限输出、
checkpoint开关loss/梯度一致、跨块hidden梯度、直接LSE和optimizer/RNG/order重载。
真实28验证视频六模型分数复现记录在输出目录，完成后才进入正式训练。
正式模型与smoke分离；最终指标以运行产物为准，本文不预填结果。

2026-10-13 验收：5项定向测试通过。28验证视频×6模型=168分数及ensemble复现通过，
最大绝对误差`1.8516453089034712e-5`，固定容差`2e-5`。首个正式source pair
完整反向与optimizer更新后已原子保存，next_pair_index=1，optimizer有35个参数状态。
安全TERM后退出；随后用同一恢复入口继续，不丢弃或重复该pair。
实际训练循环的定向恢复测试确认分段恢复与不中断的最终参数及loss一致。

以上为旧20轮任务的RUNNING交付历史；该任务现为PAUSED_SCOPE_CHANGE，不是6/6完成报告。完整六模型训练、评分及CI未执行，
不能将资源smoke或168分数复现算作本轮微调结果。

独立启动（只能存在一个任务，内部运行锁防重）：

```bash
screen -dmS v8_full_video_training bash -c 'cd /root/autodl-tmp/projects/sparse_3d_forgery_detection && exec bash scripts/run_v8_full_video_training.sh >> /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_v1/run.log 2>&1'
```

安全停止向`progress.json`中本轮真实PID发送TERM，完成当前source pair并原子保存后退出。
恢复执行同一脚本。恢复模型/optimizer/RNG/source顺序和pair索引，不重做已提交pair。
本服务器已验证screen独立运行，恢复日志从pair=2开始，并继续推进到pair=3。
未采用未能存活的nohup启动。查看：`screen -ls`及输出目录`progress.json`/`run.log`。

产物根：`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_v1/`。
`final_status.json`才表示整轮状态；`report.md`及`evaluation/summary.json`由六模型及
全部训练/验证评分完成后生成。主比较pooled video AUROC；辅助微调−冻结。
source配对bootstrap10000次，seed20260909；阈值0；三seed先各自全视频LSE再均值。

仅视频级弱监督开发评价。未验证帧/像素定位、独立source泛化；不会自动追加实验。

## 固定5轮主方法范围调整

用户要求优先降低时间成本。旧任务已安全TERM：最终checkpoint实际为完整epoch1、
epoch2前44个source pair，累计训练实测1226.2690557092428秒（含此前迁入首pair）。
旧目录标为PAUSED_SCOPE_CHANGE，20轮六模型protocol及checkpoint不变。

新目录`v8_full_video_training_budget5_v1`只训练GLOBAL_COMPONENT三seed各5轮，
迁移后从epoch2第45个pair继续，完整数据/帧、跨块梯度和全部数值协议不变。
不是两条件匹配微调；主要比较微调COMPONENT−原冻结COMPONENT。
代码只参数化已有runner与评价条件，显式保存scope_migration.json，已有验证不重跑。

```bash
bash scripts/run_v8_full_video_training.sh --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_budget5_v1 --conditions GLOBAL_COMPONENT --epochs 5 --migrate-from /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_training_v1
```

三模型各5轮、全部评分、bootstrap及report.md均已完成，最终状态COMPLETE。

### 固定5轮实际结果

244训练视频/122 source、28验证视频/14 source，共234284实际连续帧不变。
三个seed各610次source pair更新（5×122），checkpoint均为epoch5、pair_index=0，
保存optimizer、RNG与5条有限loss。共816条seed视频分数及272条均值分数，无重复键。
阈值0，三个seed先分别完成全视频面积LSE，再平均最终video logit。

|集合|COMPONENT版本|pooled video AUROC|AP|Precision|Recall|F1|Accuracy|TN/FP/FN/TP|
|---|---|---:|---:|---:|---:|---:|---:|---|
|train|原冻结|0.971446|0.962609|0.567442|1.000000|0.724036|0.618852|29/93/0/122|
|train|5轮适配|0.993617|0.992842|0.960630|1.000000|0.979920|0.979508|117/5/0/122|
|validation|原冻结|0.826531|0.857998|0.565217|0.928571|0.702703|0.607143|4/10/1/13|
|validation|5轮适配|0.872449|0.904242|0.833333|0.714286|0.769231|0.785714|12/2/4/10|

主要差值为验证pooled video AUROC +0.04591836734693877，source配对bootstrap
10000次、seed20260909，95%CI [-0.010204081632653184, 0.11224489795918369]。
AP差+0.046244，F1差+0.066528。三个seed的AUROC（冻结→适配）分别为
0.714286→0.846939、0.816327→0.882653、0.826531→0.836735，方向均正。
每source仅一real/一fake：三seed均值后的source内排序两版均14/14正确，
故逐source pair AUROC提高/持平/下降=0/14/0；这与pooled跨source排序改善并不矛盾，
不能描述成14个source均获排序提升。

固定阈值FP减少8个，但FN增加3个；Precision提高而Recall从0.928571降至0.714286。
支持保留为固定低成本适配的初步证据，不支持宣称稳定泛化收益或无代价减少误报。
不追加轮数、CELL或其他候选。训练集拟合提升不是独立泛化证据。

实测总训练13319.505秒（3.70小时），含迁入1226.269秒；本次新增训练12093.236秒，
评分/评价1046.475秒。本次进程墙钟13141.220秒，不能与含迁入的训练时间直接相加。
GPU峰值allocated 2.689249 GiB、reserved 3.013672 GiB；报告生成时新增存储25097497字节，
旧连续帧缓存未复制。原恢复checkpoint SHA迁移前后保持不变。
本轮新增范围测试3项通过；旧5项关键测试、168分数复现和资源smoke只读复用，未重跑。

完整中文报告及小表位于`v8_full_video_training_budget5_v1/report.md`、
`evaluation/summary.json`、`evaluation/metrics.csv`、`evaluation/per_source_differences.csv`；
模型、视频与缓存未提交Git。实验进程及screen均已正常结束。

## 下一阶段独立验证准备（只读清点，未启动）

后续资格核对修正见`v8-independent-source-test.md`：准备性review登记不自动排除，
缺失分数占位不等于已性能使用。下文保留当时保守准备记录，不作为最终测试资格结论。

核查起点`b250813a2a4757eb72898aab6855521d042146d9`，分支仍为
`v8-full-video-training`，起始工作区干净、与origin同步。主方法固定为
GLOBAL_COMPONENT、三seed、窗口epoch100后完整视频epoch5。以下不改变模型或旧结果。

### 实际数据范围

依据`v8_full_video_training_budget5_v1/input_manifest.json`中唯一video_id与
`actual_frames`，不是初始video_manifest中的pending字段；不重复计算窗口和seed。

|用途|唯一视频|real/fake视频|source|real帧|fake帧|总实际帧|
|---|---:|---|---:|---:|---:|---:|
|训练|244|122/122|122|103365|103414|206779|
|验证|28|14/14|14|13748|13757|27505|
|合计|272|136/136|136|117113|117171|234284|

与窗口预训练`v8_component_global_context_pilot_v1/input_manifest.json`逐split
比较(source_id, role, video_path, video_sha256)集合完全一致：训练244、验证28视频。
窗口预训练实际718训练窗口、83验证窗口；微调改变时间覆盖与监督聚合，不增加来源。
帧数是唯一缓存帧数，不是训练5轮累计呈现次数。

### 是否明显训练不足

|seed|五轮已保存loss|train/validation AUROC|差距|train/validation AP|train/validation F1|
|---|---|---|---:|---|---|
|20260909|0.664323→0.278223→0.228391→0.177287→0.180735|0.985555/0.846939|0.138616|0.984991/0.881556|0.960317/0.740741|
|20260910|0.961491→0.454920→0.305887→0.230287→0.197647|0.993013/0.882653|0.110360|0.991845/0.910964|0.979920/0.800000|
|20260911|0.442349→0.312114→0.245433→0.227857→0.211170|0.984547/0.836735|0.147812|0.981243/0.892152|0.952756/0.733333|
|三seed均值logit|不平均loss冒充新训练曲线|0.993617/0.872449|0.121168|0.992842/0.904242|0.979920/0.769231|

三seed的训练拟合较好，未发现明显欠拟合证据；不能据此证明充分收敛或独立泛化。
两个seed末轮loss仍下降，一个略回升，不据此追加epochs。现有记录不能唯一分辨
source覆盖、表示局限、前端分布与泛化问题；目前优先冻结方法做独立验证，而非
默认训练数据已足够或立即扩训练集。

### 能确认的来源多样性

使用本地官方Charades_v1_train/test.csv的scene/actions/subject字段；生成方法除
已有source mapping外，核对官方`annot.zip`内train@wan.txt、train@fcvg.txt的精确
文件记录（136个fake均匹配），不从文件名猜测。仅报告metadata粒度：

|用途|生成方法与fake视频数|官方scene类别数|官方action ID覆盖|subject ID数|
|---|---|---:|---:|---:|
|训练|wan 46、fcvg 76（2种）|16|136|53|
|验证|wan 8、fcvg 6（2种）|10|53|14|

训练scene分布：Kitchen26、Living room16、Bedroom13、Bathroom13、Laundry room11、
Home Office/Study8、Hallway7、Entryway6、Stairs5、Recreation room5、Dining room4、
Closet2、Garage2、Pantry2、Basement1、Other1。
scene是场景类别，不是独立物理地点；122 source不是122种场景。独立房屋/拍摄地点数量
未知。action ID是原真实视频标注覆盖，不表示每个生成修改均准确实现该动作。
subject ID不是已验证的不同家庭或摄像设备身份；没有这些额外字段的可靠统计。

### 本地独立测试资格

数据根datasets当前只有ActivityForensics/Charades这一已登记配对数据族。
本地官方mapping共5389条，逐项只检查媒体文件存在性：174个source具备174份fake及
对应174份real；其中136 source属于当前队列，额外38 source/76视频。
额外30 source已在`review/review_manifest.json`；另外8 source中，0S9KN/1E1VE为
旧source-learning-curve验证source，2544C/54XD1/7FNZZ为旧训练池，
869NM/AYXFY/DU7H1为旧source128扩容训练计划。计划或review接触不等于证实每个都
完成训练，但足以禁止宣称“从未参与历史开发”。本次未清点所有历史数值产物。

额外38个中19 source/38视频属于官方test（vidu17、wan1、fcvg1），全部已有review
记录；可作已接触数据的描述性冻结评价，不能作为本轮推荐的严格独立验证。
未确认有本地可直接使用、且可证明未开发使用的成对source。
raw/extracted另有deeptrace_reward目录，但未发现本任务范围内的独立双类视频/
source/split/lineage清单证据，不读取旧路线内容，不把目录名或孤立视频算作可用测试集。
没有已核验的跨数据集双类测试集；不跨数据集拼real与fake。

### 唯一优先方案：官方test中未接触source的冻结视频级验证

优先使用已有官方ActivityForensics test metadata及对应Charades真实来源，回答
同数据族内独立source泛化，不称跨数据集验证。官方test映射的Charades子集为429
fake记录、351 source；排除当前来源、已查review与历史训练计划后剩332 source，
均能匹配官方Charades test metadata。这只是metadata候选数，不是已下载或已证实
从未使用的数量。不得直接将332标作sealed test。

建议下一次授权后仅选20 source、每source一real/一fake，共40视频（计划额度，不是
实际冻结清单）。先用现有历史使用记录核对source及全部派生lineage，未能排除开发
接触的身份不称独立；匹配双方视频级标签、官方split、来源和可用路径。以固定
selection seed20260909的稳定source哈希排序，fake变体以官方记录稳定排序选一份，
不看模型分数、前端成功率或伪造观感。场景范围依据官方室内日常活动scene定义预先
确定；不按结构异常筛掉fake。若无法确认20个合法身份，先报告缺口，不凑数。
本轮没有生成最终清单、下载或查看候选模型分数。

尚缺：候选20对的最终身份及完整历史使用声明、实际媒体/内容hash、完整解码帧数、
真实文件字节数。已有官方metadata支持真假/来源核对，但不能代替这些使用历史证据。
外部数据集另需同一数据集内真假两类、source/原始来源/生成方法、标签及许可、
split与派生lineage；当前不足以选定具体跨数据集数据。

成本按旧同硬件实测估算：234284帧完整流水线墙钟77724.895秒（约0.332秒/帧，
含旧六模型及I/O，不是纯GPU前端时间）；紧凑缓存11986210941字节（约51161字节/帧）。
本次三模型已缓存评分/统计1046.475秒，约0.00447秒/帧。若40视频长度接近当前
272视频均值，预计34454帧：完整处理约3.18小时、紧凑缓存约1.64GiB，
三模型评分/统计约154秒。上述均为估计，视频长度、解码和缓存策略会改变成本；
不重新跑资源smoke。原视频下载时间/空间和临时dense前端存储未知，不能把1.64GiB
当总磁盘需求。可按实际计划总帧数F用0.332F秒与51161F字节更新。

冻结当前三个epoch5 checkpoint及各自初始化/标准化/grouping hash，chunk64、FP32、
阈值0和全视频面积LSE不变。无训练、统计拟合、阈值搜索、best选择。各seed先完整
视频LSE，再平均最终video logit。只评价video AUROC、classification AP、阈值0的
FPR=FP/(FP+TN)、Recall及Balanced Accuracy=(Recall+TNR)/2；同时列混淆矩阵及
逐seed结果。CI以source为簇配对重采样10000次、seed20260909，保留同源真假及派生
视频，不把帧当独立样本。任何执行缺失单列，不填0或按效果换source。

下一步仅此验证；不扩大训练池、不恢复旧20轮/CELL、不新增候选。本轮仅更新本文，
未运行模型前向、前端或GPU实验，也未改动既有模型与评分。
