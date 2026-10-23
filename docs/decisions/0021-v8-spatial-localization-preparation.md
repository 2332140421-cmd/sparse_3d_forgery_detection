# ADR-0021: V8新服务器空间定位准备

状态：接受准备范围；定位模型训练设计仍为提案。
日期：2026-10-23。

用户授权独立环境、预训练前端、官方metadata和一对媒体的小规模验证；分支
`v8-spatial-localization`以远端`v8-temporal-supervision-pilot`的
`9a9b6b8efc41a0c752b070c440ab3c6a69a4a663`为基线。

复用现有V8前端、native PTS、content mask、component和GlobalContextModel。
新增模块的直接调用者是本轮prepare/smoke命令及人工标注导入；输出是清点、
随机模型接口检查和原尺寸帧/JSON标注清单。已有历史runner要求旧缓存与训练统计，
不能用空文件或虚构checkpoint绕过；故新增一个独立、上限16帧的smoke入口。
不增加训练框架，不改历史数值路径。

新机没有旧训练资产。接口使用明确标注的identity scaling及固定grouping常量和
随机检测器，仅验证计算/梯度；不形成模型性能、分组效果或历史复现结论。
已下载官方标注是时间区间，未发现发布的空间mask/框；人工目标先定义为
visible_forgery_anomaly，显式区分未标注、不确定、正常空、异常mask。

`docs/v8/spatial_localization_plan.md`给出后续公平两条件设计，但本ADR不冻结
定位网络/训练超参，不启动全量提取或训练。正式方法改变另行更新契约与ADR。

仅取代`docs/v8/current_status.md`中作为“当前下一步”的过时局部几何/窗口范围
描述；不取代ADR-0014至0020的实验协议、历史结果或V7方法。
