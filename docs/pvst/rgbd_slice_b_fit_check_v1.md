# 完整B固定pair拟合诊断与正式history审阅

计算状态：COMPLETE，退出码0。拟合判断：MAIN_GOAL_REACHED。

固定an:7ogg04；fake cb41b3d004b6640f6c3b5bb1、real c26e3fc1fdc6aabda308ae84，两段均[0,24)，PTS 0–0.92秒。fake→real交替；原make(B)+common_initial，不加载正式best/last或拟合A，不重提几何。

原A实际协议：150次/约2700秒上限、每10次评价；主要目标Dice≥.8且real FPR≤.05；次级判据Dice≥.6、GT召回≥.6且FPR≤.05，分别报告。像素AP原A未计算，本轮不补。所有分项损失/优化器/BN/Dropout保持原实现。

|固定片段诊断|更新数|fake Dice|IoU|GT召回|real FPR|预测前景面积(fake)|

|---|---:|---:|---:|---:|---:|---:|

|原A|30|0.8092452021647429|0.6803536755940982|0.9563991715528423|3.146701388888888e-05|0.08042914496527777|

|完整B|40|0.9002224460203717|0.8204980870757247|0.9141084057310431|5.063657407407407e-06|0.06081344039351851|

A拟合用时51.23427032865584秒；B分阶段用时：{"model_construct_s": 1.3046331005170941, "data_load_s": 0.18496590852737427, "initial_eval_s": 2.776920821517706, "updates_s": 266.02741361409426, "intermediate_eval_s": 9.001982989721, "final_eval_s": 2.2749230870977044}。此为不同模型的诊断成本，不是等更新数速度比。

|分支|第1次梯度范数|第2次|最后一次采样|采样梯度存在且有限|

|---|---:|---:|---:|---|

|rgb_stem|5.591930389404297|8.079171180725098|19.7989444732666|True|

|depth_stem|15.804208755493164|33.69154739379883|91.45805358886719|True|

|rgbd_interaction|4.466722011566162|6.354456901550293|12.014492988586426|True|

|decoder|3.859707832336426|1.5828216075897217|0.09202856570482254|True|

|pixel|2.3140196800231934|2.3007407188415527|0.72405606508255|True|

|frame|3.7871127128601074|5.158471584320068|5.638445854187012|True|

|pool|0.0|0.018317775800824165|0.04226480424404144|True|

|time|0.0|0.04276378080248833|0.08243044465780258|True|

|depth|0.0|0.28965258598327637|0.5712469220161438|True|

|pixel_context|23.93328285217285|14.573226928710938|0.761539101600647|True|

|frame_context|59.9830436706543|88.61906433105469|64.3051986694336|True|

最终两个残差投影权重范数：{"pixel_context": 0.10225188732147217, "frame_context": 0.004416364245116711}

第1次训练投影输出：{"pixel_context": {"mean_absolute": 0.0, "elements": 433344, "calls": 24, "scope": "occupied-slice projection elements, before native broadcast"}, "frame_context": {"mean_absolute": 0.0, "elements": 24, "calls": 24, "scope": "occupied-slice projection elements, before native broadcast"}}

第2次训练投影输出：{"pixel_context": {"mean_absolute": 0.209951402389374, "elements": 448640, "calls": 24, "scope": "occupied-slice projection elements, before native broadcast"}, "frame_context": {"mean_absolute": 0.24894205729166666, "elements": 24, "calls": 24, "scope": "occupied-slice projection elements, before native broadcast"}}

最终eval投影输出：{"pixel_context": {"mean_absolute": 0.9349889941402707, "elements": 881984, "calls": 48, "scope": "occupied-slice projection elements, before native broadcast"}, "frame_context": {"mean_absolute": 0.4119466145833333, "elements": 48, "calls": 48, "scope": "occupied-slice projection elements, before native broadcast"}}

## 正式RGBD_LOCAL（原记录只读）

三项在线训练loss：{"frame_BCE": {"first": 0.8565425590301553, "last": 0.4056059338375538, "minimum": 0.3805999918161736}, "pixel_BCE": {"first": 0.30939300327251357, "last": 0.1037569713468353, "minimum": 0.1037569713468353}, "dice_loss": {"first": 0.965132067600886, "last": 0.8979525423298279, "minimum": 0.8827870016296705}}

监测最高：epoch13，Dice=0.7893566461700915

历史开发帧AUROC最高（不替换正式checkpoint）：epoch6，AUROC=0.5244185828601413，该轮开发Dice=0.12918600445484857

正式空间best轮次5，该轮监测Dice=0.6436738624544424、开发帧AUROC=0.5143527107812822。

## 正式RGBD_SLICE_RELATION（原记录只读）

三项在线训练loss：{"frame_BCE": {"first": 1.099535101481403, "last": 0.4026162618577776, "minimum": 0.3304107658338277}, "pixel_BCE": {"first": 0.2983110562587778, "last": 0.08151423457699518, "minimum": 0.0789628386652718}, "dice_loss": {"first": 0.9668966718018055, "last": 0.8655783090119561, "minimum": 0.8655783090119561}}

监测最高：epoch11，Dice=0.7653515831072791

历史开发帧AUROC最高（不替换正式checkpoint）：epoch6，AUROC=0.653932647439141，该轮开发Dice=0.14465063010995155

正式空间best轮次12，该轮监测Dice=0.12626882518543334、开发帧AUROC=0.47167875414628657。

B监测Dice第11轮.7654→第12轮.1263→第13轮0，后期反复低值；不是始终不能拟合，也非单调下降。三项在线loss总体下降不代表这个固定pair定位稳定。监测只有1pair/两条24帧片段，不代表全训练集。

B在线frame BCE从1.0995降至.4026，开发frame AUROC第6轮.6539高于空间best第12轮.4717，随后未保持；A最高开发frame AUROC第6轮.5244，空间best是第5轮.5144。监测帧AUROC两者早期达到1，但固定阈值检测仍可能偏向同一类。历史检测与定位最优不同步，不能替换best或拼成绩。

正式history未保存逐轮关系残差/分支梯度，能确认随训练的定位波动，不能仅凭时间顺序认定关系支路导致退化。

## 判断

完整B达到与原A相同的主要拟合目标，排除了“完整B没有基本小样本拟合能力”的解释。结合正式记录，当前证据最支持多pair训练中空间优化不稳定，而不是直接归因于相机运动或三维无效。

唯一优先下一步：定位多pair训练第11→12轮监测定位骤降的更新干扰，先查空间优化与残差变化，不扩量。

正式文件未变：True；本轮执行源码：dcbc576a513d4a8b59568cc175da27be26fad99f

产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_b_fit_check_v1`，固定样例samples/index.html，机器摘要result.json、formal_history_review.json。检查权重不进入正式模型，本轮到此停止。
