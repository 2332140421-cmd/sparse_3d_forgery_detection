# AINPAINT空间基线固定权重BN诊断

ADR-0052：原final.pt、原dev4/824帧，RUNNING_BN/IMAGE_BN各一次。
整体eval/no_grad/dropout关闭，仅五个head BN函数式当前图统计，不更新任何权重/缓冲区/计数。
预定少量回放/重复、原生指标和24固定预览/统一差分；不训练、校准或调阈值，不替换旧报告。
关键测试4项通过；执行后补入实际结果。入口research_tools.xyz_pilot.ainpaint_bn_diagnosis。
