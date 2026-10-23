# V8 数据资产准备

本轮仅资产准备：不试标、不训练、不全量前端、不评分。完整发布集与历史探针队列分开。

## 目录与入口

个人数据根：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection`。
沿用已有media、metadata、smoke；新增`official/<revision>`、`real/<source_dataset>`、
`real_archives`和`asset_preparation`。先前1LARL real继续原路径，不复制；fake在官方
snapshot初建时使用hardlink复用，HF后续若更换目录项，以完整性核验结果为准。
一个247字节官方basename加HF的.metadata后缀超255字节；已用官方hash blob cache
单独下载再hardlink解决，官方文件名和相对路径不改，已完成文件不重下。
所有媒体、下载缓存、完整JSONL索引、凭据都在Git仓库外。
已有前端环境与三权重SHA未变，未重复GPU验证。仅新增轻量下载工具yt-dlp==2026.10.21，
未升级PyTorch、Torchvision或MoGe/CoTracker。

```bash
export V8_WORK=/media/njupt/8FCD3869E2FAB06F/chenyh
export HF_HOME="$V8_WORK/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="$V8_WORK/cache/pip"
export V8_DATA="$V8_WORK/data/sparse_3d_forgery_detection"
source "$V8_WORK/envs/v8/bin/activate"
cd "$V8_WORK/projects/sparse_3d_forgery_detection"
python -m scripts.prepare_v8_assets official --root "$V8_DATA" --revision a34d4b7b04b0f3f3e26ba900adc367218667c581
python -m scripts.prepare_v8_assets charades-download --root "$V8_DATA"
python -m scripts.prepare_v8_assets charades-extract --root "$V8_DATA"
```

本机后续原视频恢复需在独立子shell中使用已实测的传输路径，不写.bashrc或全局代理配置：

```bash
(
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DOWNLOAD_TIMEOUT=120
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
  python -m scripts.download_v8_archive_sources --root "$V8_DATA" --dataset hcv1 --workers 2
  python -m scripts.prepare_v8_assets activitynet-download --root "$V8_DATA"
  python -m scripts.download_v8_archive_sources --root "$V8_DATA" --dataset acnet --workers 2
  python -m scripts.prepare_v8_assets activitynet-tar --archive-workers 4 --root "$V8_DATA"
)
python -m scripts.prepare_v8_assets index --root "$V8_DATA"
```

只恢复未完成阶段；已经COMPLETE的阶段不重复运行。镜像原始清单与精确source映射保存在
asset_preparation/references，重启无需临时调试文件，缺失时应报错，不能创建空manifest跳过。
命令分别顺序执行，官方snapshot可复用已完成文件及官方incomplete缓存；Charades下载
支持按实际部分字节恢复，必须服务器honor Range，否则拒绝追加。既有固定revision
manifest不会随main更新。需要重试失败文件时先查看`integrity_failures.json`，仅对
确认损坏的目标用官方`hf_hub_download(...,force_download=True)`补取，不能忽略失败项。
检查CPU容器不等于全帧解码；只对每个来源×生成器首个可读样本解码开始、中间、末尾。
SHA与官方大小不符的LFS pointer不会被当作媒体。

## Git认证

已按本仓库Deploy key、Allow write access获得授权；仓库局部core.sshCommand与
push URL使用该专用身份，fetch仍为原HTTPS。未更改全局Git/SSH配置，不使用其他账户密钥。
工作盘无法chmod（尝试600后仍777），所以本轮新私钥移至个人Linux文件系统
`/home/njupt/.local/share/chenyh/github/v8_deploy_key`，实际权限600、父目录700。
本轮生成后发现权限不可收紧的工作盘私钥副本已删除。公钥/独立known_hosts在个人工作区credentials目录。
GitHub主机Ed25519指纹与官方文档一致。实际普通push已成功并确认远端HEAD为8c3ba1e；
本轮收尾还需提交修改并再次push，最终以远端完整hash为准。

## 官方资产目标与状态

固定revision `a34d4b7b04b0f3f3e26ba900adc367218667c581`，5395正式文件，
120480401906字节（120.48GB）；其中5389 fake，train3824/test1565。
官方train/test有122个(dataset,source)交集；这是发布划分统计，不是历史探针训练/开发交集。
其中ActivityNet92、Charades30；后续研究必须另按source分组冻结划分，不能直接把官方split解释为source隔离。
其余是.gitattributes、README、2份CSV、annot.zip、feat.zip（约1.93GB）。
feat.zip保留为官方发布资产，不能替代V8原始媒体前端。
初查可用磁盘约12TB，足以覆盖发布集、Charades16.34GB包与关联real；后两来源实际
可下载体量尚未确认，不以0计成本，也不以数据齐全为由扩大训练。

官方HF发布集已验收 **COMPLETE**：5395/5395文件、5389/5389 fake通过；
对应real仍在恢复，不能据此写整个real/fake数据集COMPLETE。当前计数见`asset_preparation/asset_summary.json`。
下载成功标志仅为DOWNLOADED_PENDING_VERIFICATION，所有目标通过官方大小/校验、
媒体容器和分层抽样检查后才可写COMPLETE。`download_status.json`持久保存开始时间、
实际下载耗时与失败；不是启动进程即完成。

|生成器|fake目标|
|---|---:|
|Wan|1059|
|FCVG|1166|
|SciFi|780|
|VACE-1.3B|1061|
|LTX|1058|
|Vidu|265|

## 关联真实来源

|来源|fake数量|唯一real目标|入口与边界|
|---|---:|---:|---|
|Charades-STA/Charades|1438|976|官方Charades_v1_480.zip；只解出关联source，包中额外媒体不作为实验队列|
|ActivityNet Captions|1832|842|842/842精确匹配官方ID；直链镜像覆盖675，旧missing-files ZIP另覆盖83，84仍无可取媒体|
|HC-STVG v1|2119|1912|官方补充材料1.4节确认；不是Hollywood，原v1 OneDrive访问失败|

同source多fake共享一份real。索引保留(dataset,source)、完整fake相对路径、generator、
官方split、local_path、real_path、字节数、校验类型与SHA、容器信息、时间标注关联。
不能用pair_id或无数据集前缀source做唯一键。

ActivityNet下载实试：`v_---9CpRcKoU`与`v_cyXWvxVt8qE`要求Sign in确认非机器人，
`v_mSyfGQigb8U`报告unavailable。仅这3个是逐视频实测，其余不能声称已逐一失效。
进一步核查公开镜像：TornadoLabs/activitynet revision
`bc9c567ea67be43daf443cddb832ed983e5ccae5`的metadata.jsonl按video_id和youtube_url
逐一匹配675个目标（17,303,875,891字节）；下载核对其size_bytes并记录HTTP ETag和本地SHA。
YimuWang/ActivityNet revision `1a20776262cc1fe3761588843cad26b62c2b7125`的
missing_files.zip目录另精确匹配83个目标（1,929,017,817字节），只Range提取相关成员并核CRC。
这两组来源互不重复；原生MKV保留，不改后缀冒充MP4。镜像metadata固定revision不意味着
外部bucket媒体具有官方不可变SHA，因此必须保留下载当时本地SHA与来源，不声称逐位复现旧机。
剩余84个ID未在这些直链/ZIP镜像找到，继续检查同镜像219,976,627,189字节的
`v1-3_train_val.tar.gz.*`历史分段包。只用固定revision取得这些段。初次代理路径遇CDN超时；普通snapshot还可能在API失败时
返回已有不完整local_dir，本轮清单校验明确拒绝了该结果。TAR段改用官方hf_hub_download
逐目标枚举，HF_HUB_DOWNLOAD_TIMEOUT=120。直连huggingface.co测试失败，而直连hf-mirror.com
的实际1,872,819,189字节分段在111.6秒完成，LFS SHA完全匹配；HC实际成员CRC也匹配。
随后在本任务进程设置HF_ENDPOINT并清空代理变量，经专用tmux恢复，不改全局配置。
官方Xet测试未观察到payload进展，停止专用测试，未填0耗时。逐段LFS SHA
通过后用cat管道+stdlib tarfile流式扫描，只写目标84个source，不生成220GB合并副本，
不解出原库其他视频、不解码成图。段下载和扫描均有独立RUNNING/PARTIAL/完成状态；
包内实际视频数与最终命中数待实际扫描，不把包头或目标数当作已恢复。
不将全部842个都写成已失败或全部恢复。
官方恢复途径是[数据访问申请](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform)，
由用户填写后取得7天访问的官方Drive/Baidu目录；不索取聊天密码、token或cookies。

HC-STVG原v1入口为[官方README的v1 OneDrive](https://github.com/tzhhhh123/HC-STVG)。
本机多次TLS失败，外部读取也502；不能据此断言链接已永久撤销。用户可在网页打开v1链接，
若可访问则下载官方包到`real_archives`，再按source清单解出目标；不要用v2或相似视频替代。
辅助镜像核查：k-nick/HC-STVG API返回401。CircleRadon/hc-stvg固定revision
`4d26c230bbdba17e57a2e9965d15b584cb69a33e`的三个分段组成ZIP。通过HTTP Range
只读central directory，1912/1912个AF source有唯一精确同名原生成员（1174 MP4、738 MKV）。
仅提取这些13,072,940,637字节成员，不下载115GB全镜像或改容器后缀。每个文件由
zipfile解压并校验成员CRC，计算本地SHA，并保存镜像revision/member/CRC/SHA sidecar。
这是第三方镜像身份/完整性证据，不宣称匹配官方原始文件SHA；官方未提供逐文件SHA，
整个分段包的LFS校验也不能冒充已下载子成员的官方原始校验。
官方入口访问历史在`real_access_issues.json`，其中HC原入口错误不代表镜像最终缺失。
最终尚缺source及预期文件以`missing_real.json`和`asset_summary.json`为准，不以占位文件补齐。

## 标注审计

已下载并检查annot.zip全部11个TXT（5389行）、全部CSV、完整官方文件清单、
官方GitHub代码/说明、项目页、论文及709769字节补充材料。AF另一个官方OneDrive入口
本机TLS失败，无法清点其内部是否有额外资产；此访问限制也保留，不推断未查看内容。GitHub审计HEAD
`8717fdbe20ef973c9ef079542a027cf15ef9773a`；标注读取器是video/duration/times、N×2区间。
`annotation_audit.json`保存每个文件行数、完整样例和补充材料SHA。
每行字段为fake完整basename、duration秒、一个或多个`start=end`区间。
例如1LARL fake的duration与[3.60,10.20]是时间范围，不描述像素位置。

论文4.2.1的tIoU是预测时间区间与真实修改时间区间的交并比，AP阈值0.75/0.85/0.95。
HC-STVG原人物框描述真实source中的目标人轨迹（bbox[x,y,w,h]、st_frame从1开始、
原width/height），不等于fake伪造框。补充材料提及生成时编辑mask，但未找到已发布
mask文件及到最终fake的逐帧坐标/身份映射；即使取得编辑mask也不能自动当可见异常mask。
论文示意图框、时间区间、原人物框、编辑mask、最终fake伪造mask、人工可见异常mask分开。

结论：**当前检查到的官方公开资产未提供可直接使用的空间mask/框；空间定位监督仍未准备好。**
不推断作者内部从未制作空间标注；本轮不造伪标签，不发起人工标注。

## 历史探针身份

目标保持244训练视频/122source、28开发验证视频/14source。已查当前仓库、所有本地
Git分支历史中的JSON/JSONL/manifest路径及V8实验文档；没有提交完整V8身份manifest。
可确认14个验证source：00ZCA、0RJKT、0YOXV、129SP、13MWT、14TJT、170OQ、19ZJB、
1CITG、1G9R7、1JGRO、1K0SU、1L376、1TW98；官方对应有23个fake候选，不能擅选14个。
训练122source完整名称和精确fake变体、split映射尚缺；本地无法实际计算完整source交集。
历史报告声明无交集，不替代新机身份验收。未读取旧数值特征、统计或checkpoint。

需要旧V8 `v8_full_video_training_budget5_v1/input_manifest.json`、
`v8_component_global_context_pilot_v1/input_manifest.json`，或等价video/data manifest。
完整缺口与候选列表见`historical_probe_recovery.json`。下载全库不会生成新的替代训练队列。
