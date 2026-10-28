# V8预训练空间定位基础与三维时序上下文增强

日期2026-10-29。当前状态RUNNING：独立入口已启动，尚无完整正式评价或最终结论。
基于审阅提交`1cafc51e01f3d0b668b99680522fa655f7320f82`建立
`v8-pretrained-spatial-context`，旧实验不变。

研究问题：同一预训练视觉特征、编辑区域监督和训练规则下，dense三维几何+
动态component全帧关系与因果时序上下文是否改善像素定位？
只有SPATIAL_TRANSFER和SPATIAL_3D_CONTEXT，见
[合同](../v8/pretrained_spatial_context_contract.md)与[配置](../v8/pretrained_spatial_context_config.json)。

## 已完成准备

- 原joint_manifest全部ANAKIN身份冻结：[小型清单](../v8/pretrained_spatial_context_manifest.json)。
  train48视频/6178帧，dev16视频/1806帧，共32pair/7984帧；无重选。
- 原96训练统计/grouping、canonical缓存和已核验官方mask只读；未复制130GiB或重跑前端。
- 官方AINPAINT代码revision `753ae1b700b054c97e15569718690211e36cdf53`；
  官方DINOv2代码revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
  前者外部目录`workspace/external/AINPAINT-CVIU26`，后者`workspace/external/dinov2-ainpaint`。
- 官方[权重目录](https://drive.google.com/drive/folders/12p7chrWu7fY3_aObwd8_MfgmiAk0CIbX)
  中仅取[空间checkpoint](https://drive.google.com/file/d/1Wz4IZ046XTsikE6wPUc7_VDWxZJbzs9d/view)，
  文件`spatial_opn_sttn_noreal_loconly_v1.pt`，1,266,224,146字节，SHA256
  `6f6210b10a9b09aabd2dbb139e8797cd68e786a7591440af6d7dca4ba9e58cc7`。
  成功续传阶段1527.9秒；此前连通性尝试和13.6MiB部分下载未合并，不能作为全新下载吞吐。
- 真实checkpoint源epoch6，575定位必需键完整匹配，仅排除4分类头键。
  backbone305,952,048参数（其中训练过的LoRA1,579,312），decoder2,094,017。
  全骨干及LoRA冻结、eval、与新optimizer参数集合无交集。
- 横屏368×640/竖屏640×360中帧核验：原视频解码按原mapping到512与旧RGB缓存逐像素相同，
  目视无裁边/翻转；512补右下6至518后保留全部内容。
  同一次官方token重解码差0，重复CUDA骨干编码logit最大差0.000162/0.000295；
  FP16往返量化mean abs 0.00000135/0.00000149，max abs0.000077/0.000112，阈值翻转0。
  A/B零修正最大logit差0。实际落盘缓存还将再次核对。
- 原V8环境保持，隔离overlay `workspace/envs/v8-ainpaint-packages`：
  peft0.13.2/transformers4.44.2/accelerate0.34.2/huggingface-hub0.24.6/
  tokenizers0.19.1/safetensors0.4.5/regex2024.11.6，torch2.5.1+cu124不变。
  官方DINOv2本地结构可构建，无xFormers，未下载原始DINOv2权重。
- 27项相关定向测试通过（新路径10项、旧输出分离7项、旧联合路径10项）。
  覆盖严格键检查、边缘/真实尺寸mapping、GT不入模型、两次更新梯度、BN重算、
  跨块梯度/状态重置、恢复同序同参数、epoch0 best和报告口径。

## 已通过的真实资源smoke

[小型执行证据](../v8/evidence/pretrained_spatial_context_preflight.json)保留完整权重加载、
横竖屏real/edited四条实际缓存量化检查及两个条件更新证据。
64视频7984帧共享特征累计579.99秒、0.072645秒/帧，22,384,844,800字节（20.85GiB）。
未复制原缓存。新增权重与特征成本分开记录。

原首训练pair `an:95qlg9`，real90帧/fake90帧，两个条件各两次完整pair更新：

|条件|可训练参数|总参数（含冻结视觉）|pair更新秒（两次）|常规评价秒|精确AP评价秒|峰值allocated MiB|
|---|---:|---:|---|---:|---:|---:|
|SPATIAL_TRANSFER|2,094,017|308,046,065|1.461 / 0.975|0.481|1.834|357.8|
|SPATIAL_3D_CONTEXT|2,569,665|308,530,034|2.052 / 1.865|0.763|2.148|629.5|

增强新增支路483,969参数，其中8,321为不参与optimizer的旧检测head；
两条件冻结视觉均305,952,048参数，增强总冻结305,960,369。
decoder和P实际更新；第一步几何/GRU梯度0，第二步梯度绝对和0.002791/0.025179，
P末层第二步2.736927（裁剪后）。smoke权重已丢弃，正式重新构建配对官方decoder和零P。

按单pair实测外推最大30轮：A训练1254秒、常规评价639秒、额外dev AP约81秒；
B训练2017秒、常规评价1015秒、额外dev AP约83秒，合计约85分钟。
仅为单pair线性外推，不含最终完整best/last AP、epoch0 AP、样例导出与每pair checkpoint I/O；
原生分辨率、视频长度和共享服务器负载可影响耗时，早停可缩短。

## 执行及交付位置

独立输出绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_context_pilot_v1/`

单入口：

```bash
PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages \
OMP_NUM_THREADS=4 \
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u \
  -m research_tools.v8.pretrained_spatial_context.run \
  --config docs/v8/pretrained_spatial_context_config.json --resume
```

正式运行前需先通过完整定位权重加载、横竖屏官方forward/量化/零修正一致性、
两条件完整ANAKIN pair两次真实更新及评价smoke，再提交推送和独立tmux启动。
权重不能完整恢复则资产阻塞，不用随机decoder替代。

### 本次启动

上述前置验证已全部通过。执行代码提交`d53fb5a914769ac1cfecb1d66157eb5d67d8d1d1`
已先普通push。北京时间2026-10-29 04:20:38独立启动：

```bash
tmux attach -t v8_pretrained_spatial_context_pilot_v1
```

日志绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_context_pilot_v1/run.log`

`tee`同时输出到tmux画面和日志，会话设置remain-on-exit，退出后保留画面。
同目录`launch.json`保存代码提交/命令/启动时刻，`progress.json`为当前执行状态。
启动先验证资产和已存在特征SHA，复用smoke证据，再完整epoch0评价、训练A/B、最终评价/报告。
RUNNING不表示两模型成绩已完成；实际早停、指标与最终存储需完成后集中审阅。

Python最终自动写`report_zh.md`、`result_summary.json`、`index.html`和固定原始F32 NPZ；
best/last完整train/dev Dice/IoU/AP/real空间FP、epoch0、参数/成本分列。
固定样例原生2像素GT边界带仅用于描述覆盖/背景/边界错误，不修改GT或训练loss。
训练中不逐轮Codex轮询，不追加模型或解冻LoRA。
