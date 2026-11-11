# 冻结局部像素特征读出：协议

本诊断从a94e1ce3e8e30cb2f6783775a7481e44b03d3cec派生，不改变主模型或正式训练协议。
直接调用方为独立CLI，输出采样身份、一个线性读出、开发pixel AP。现有evaluate不提供训练像素采样和双读出，因此仅新增薄hook入口。

唯一权重：RGB_DEPTH_PARTITION last epoch12/576。Local.native输出[1,32,H,W]，融合局部RGB与原图细节后、加上下文前。原头pixel_context完整投影含bias清零。每次前向被动hook，不改输出。
只用冻结train48段1152唯一帧，全原生GT正/负各至多128像素，无放回；固定seed20260909，通过SHA256(seed,video,frame)派生每帧固定随机流。real及fake非编辑区域均为负，不用valid过滤。不足全取、空类别不造样本。先StandardScaler再唯一L2逻辑回归C1/lbfgs/max_iter1000/balanced；训练和标准化不读开发。未收敛停止，不搜索。
开发原4对8完整视频924唯一帧；沿evaluate原24帧窗口及尾段左上下文，历史不重复评分。一遍开发前向同时取得原ZERO与线性AP。线性折入FP32，先以atol=rtol=1e-5验证等价。pixel AP用全原生像素，正GT帧→视频均值→4fake宏平均；不比较阈值指标、不校准，不以AP证明real低误报。
检查：原生索引/GT、可复现训练采样、尾段去重、折入等价、实际native hook前后输出不变、冻结参数和缓冲不变。不开训练主模型或前端，不读保留测试。计算及发布分开记录。
