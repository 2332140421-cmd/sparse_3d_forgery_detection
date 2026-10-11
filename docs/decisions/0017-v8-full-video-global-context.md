# ADR-0017：冻结全帧关系模型的全视频连续评价

状态：用户授权；独立分支v8-full-video-global-context，不替代旧实验结果。

复用GLOBAL_COMPONENT/GLOBAL_CELL六个epoch100模型，对冻结272视频全部正常解码帧
评分，无训练或统计拟合。新增小型连续适配调用现有GlobalContextModel.spatial，再
复用旧接口GRU更新。旧接口仅提取原空间方法，默认行为不变；不能绕过全帧attention。
调用者为本轮runner，输出为格/帧/视频响应；普通窗口forward不能跨块保存状态。

每64帧NPZ保存float32格输入、mu、面积、分区、关联、dt及无损packbits像素mask。
不长期保存新dense XYZ/depth/RGB；不以冻结patch embedding替代可训练输入。
前端只增加关闭原始大缓存写入的选项，默认旧行为不变。旧两视频输入核验后只读复用。
无通用缓存框架、调度器或新训练系统。AGENTS无需改写，适用独立V8边界。

每模型分别连续hidden与面积LSE累计，三seed平均最终video logit。主指标validation
pooled video AUROC，配对source bootstrap10000/20260909。旧窗口辅助CI独立补充，
不修改旧macro主指标。不展开帧/像素标签，不称新sealed test或全视频训练。
