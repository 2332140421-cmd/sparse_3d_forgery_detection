# 预训练RGB-D编码与深度切片关系匹配实验

当前状态：RUNNING。必要检查与A固定小样本拟合均通过，独立入口已继续正式阶段。尚无新最终开发成绩。
执行源码b3e11381aaeebc149cc576083b120e1c3d1f0a29，分支pvst-rgbd-slice-relations。
协议：[ADR-0034](../decisions/0034-rgbd-slice-relations.md)、[固定配置](rgbd_slice_config_v1.json)。两个条件均使用深度，不是纯二维/三维对照。

|条件|总参数|可训练参数|方法|
|---|---:|---:|---|
|RGBD_LOCAL|20,402,578|20,402,578|共同DFormer-S、多尺度256局部融合、原生卷积细化和独立帧头|
|RGBD_SLICE_RELATION|33,264,311|33,264,311|共同路径＋4×128学习式片池化、两组512维8head时间→深度关系和零初始化像素/帧残差|

官方原版DFormer-S源码e3273009b759b578945483828ff315d560be94c9，Small RGB-D ImageNet权重229183347字节，SHA256=8a086a3fb4e7ebda0806172938dd199bb7d8394593a8176640ed79735e35a0c0。保留编码器530键strict加载；RGB/depth stem各35键、交互270键，无missing/unexpected。仅略去分类pred和官方分割不使用的最后深度更新，四尺度输出一致性已通过检查。编码器微调，BN统计固定、仿射可训练；依赖隔离，不改PyTorch或驱动。

复用ANAKIN训练8pair/16视频/2138帧/1065正帧，开发4pair/8视频/924帧/462正帧。片段严格沿用48条首/中/尾24原生帧清单，无场景筛选或新数据。
24视频、3062帧新MoGe几何已全部完成，新增前端耗时82.6809秒；完整画面长边≤768，小视频不放大。例：720×1280→432×768，1693 tokens，实际内部434×770；360×640保持尺寸，1176 tokens。内部patch取整和输出尺寸逐视频登记，不是旧256场放大。encoder_depth和固定slice_depth=Z/1.9888148307800293分开，h.125,d0=0不改。

11项CPU相关检查及新增1项CUDA设备传输回归通过。真实an:7ogg04、[0,24)检查：
- A/B共同初始化逐张量核验，零残差像素/帧输出最大差均为0；BF16预设atol/rtol=.002。
- B两次真实AdamW更新通过，分别8.506秒/6.553秒；梯度存在且有限，关系上游在第二步取得非零梯度，共同BN缓冲未变。
- 检查片段allocated峰值分别0.925/1.113GiB，进程峰值RSS约12.129GiB；不是完整训练峰值。

A固定第一个训练pair首24帧real/fake拟合检查（不是开发成绩）：

|指标|初始|30次片段更新后|
|---|---:|---:|
|fake Dice|0.098097|0.809245|
|real像素FPR|0.624738|0.00003147|
|fake GT召回|—|0.956399|

拟合耗时51.234秒，达到预设Dice≥.8且real FPR≤.05目标。检查/拟合权重均丢弃，正式A/B从同一共同初始化重新开始。这个结果说明固定训练片段上具备定位学习能力，不代替完整训练或泛化证据。

首次启动在完成新几何后，因CPU暂存上下文未回传GPU而退出1；已补齐可微设备拷贝并完成CUDA/BF16反向核验。旧日志、退出码与82.6809秒准备成本保存于execution_history/00_cpu_context_device_copy。新执行复用全部几何和相同初始化，新增前端调用0；不是OOM，不改模型或损失。

独立流程将自动按最多30轮/最少10轮/Dice耐心8完成两个条件，从各自同一best评价全部开发帧、生成B−A和逐fake差值、固定样例、中文报告以及普通commit/push。当前尚无best/最终指标，不将检查样本当开发效果；本页将在结束时由报告替换。

运行根目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/`。
查看progress.json/run.log、后续各条件history.csv；完成后report.md、result_summary.json、best_dev.json、samples/index.html与publication.json。
相机运动/深度漂移、窗口边界和单seed小队列限制仍在。开发4fake均全程编辑，不能验证复杂伪造切换边界；不自动追加条件或训练。
