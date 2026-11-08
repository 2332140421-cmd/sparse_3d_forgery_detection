# 自适应集合对集合：两个训练pair限定拟合诊断
状态：COMPLETE；真实worker退出码0；已完成更新80/80。
只用训练an:7ogg04、an:728qx7各fake/real的原始[0,24)，四段共96帧；不是开发/测试性能。四段按pair1 fake→real→pair2 fake→real循环，每段一次更新，最多1800秒。
方法与张量见ADR-0036及[设计](set_to_set_design_v1.md)。完整DFormer-S/原生解码/SlabPool复用；8个关系单元×A/B×4诱导状态512；两块8head联合交互；关系readout→512因果GRU→帧头，无旧global分类捷径；像素路径C+关系消息→零初始化512→32投影。
各次确定性评价（logit≥0；空GT不填假标；正GT帧Dice/IoU/召回，real像素FPR）：

|更新|pair|role|Dice|IoU|GT召回|real FPR|预测面积|frame BCE|
|---|---|---|---|---|---|---|---|---|
|0|an:7ogg04|fake|0.09809653029491043|0.05162514084883744|0.5588332684803268|None|0.6147734013310185|0.7081705729166666|
|0|an:7ogg04|real|None|None|None|0.6247381365740741|0.6247381365740741|0.6793619791666666|
|0|an:728qx7|fake|0.0015980032208004734|0.0007996427566261739|0.3231448494392886|None|0.7372883662471065|0.76513671875|
|0|an:728qx7|real|None|None|None|0.7262794325086804|0.7262794325086804|0.6295572916666666|
|10|an:7ogg04|fake|0.0|0.0|0.0|None|0.0026255063657407408|0.5623372395833334|
|10|an:7ogg04|real|None|None|None|0.0026394314236111113|0.0026394314236111113|0.8453776041666666|
|10|an:728qx7|fake|0.0|0.0|0.0|None|3.987630208333334e-05|0.568359375|
|10|an:728qx7|real|None|None|None|3.996672453703704e-05|3.996672453703704e-05|0.8352864583333334|
|20|an:7ogg04|fake|0.0|0.0|0.0|None|0.0004944299768518518|0.599609375|
|20|an:7ogg04|real|None|None|None|0.0006212022569444444|0.0006212022569444444|0.79541015625|
|20|an:728qx7|fake|0.0|0.0|0.0|None|0.0|0.5970052083333334|
|20|an:728qx7|real|None|None|None|0.0|0.0|0.80029296875|
|30|an:7ogg04|fake|0.0|0.0|0.0|None|1.3563368055555555e-05|0.64013671875|
|30|an:7ogg04|real|None|None|None|1.8265335648148146e-05|1.8265335648148146e-05|0.74853515625|
|30|an:728qx7|fake|0.0|0.0|0.0|None|0.0|0.6363932291666666|
|30|an:728qx7|real|None|None|None|0.0|0.0|0.7535807291666666|
|40|an:7ogg04|fake|0.6337037904971609|0.4677129422164639|0.4861351421619287|None|0.03133029513888889|0.6886393229166666|
|40|an:7ogg04|real|None|None|None|5.787037037037037e-06|5.787037037037037e-06|0.697265625|
|40|an:728qx7|fake|0.14706006125486215|0.0794406450976925|0.08134231923200108|None|0.00019115306712962966|0.6739908854166666|
|40|an:728qx7|real|None|None|None|0.0|0.0|0.7117513020833334|
|50|an:7ogg04|fake|0.8169463921162494|0.6913908516093468|0.968278970825847|None|0.08085322627314816|0.8040364583333334|
|50|an:7ogg04|real|None|None|None|1.8084490740740741e-06|1.8084490740740741e-06|0.5927734375|
|50|an:728qx7|fake|0.4224222940016095|0.2678235067332644|0.9436819799978816|None|0.006341462311921296|0.7752278645833334|
|50|an:728qx7|real|None|None|None|0.0|0.0|0.6162109375|
|60|an:7ogg04|fake|0.8874022829905689|0.7996759598210493|0.866623936210691|None|0.05623444733796296|1.0748697916666667|
|60|an:7ogg04|real|None|None|None|2.350983796296296e-06|2.350983796296296e-06|0.416259765625|
|60|an:728qx7|fake|0.42060537665101716|0.26635303997682297|0.9460268546803631|None|0.006396032262731481|0.9886067708333334|
|60|an:728qx7|real|None|None|None|0.0|0.0|0.4646809895833333|
|70|an:7ogg04|fake|0.8346474478687709|0.7170558257386576|0.9895387127172093|None|0.08088342737268518|1.5146484375|
|70|an:7ogg04|real|None|None|None|1.4467592592592592e-06|1.4467592592592592e-06|0.2462158203125|
|70|an:728qx7|fake|0.4958271009713224|0.32965750592814386|0.8755491058809991|None|0.004627278645833333|1.3577473958333333|
|70|an:728qx7|real|None|None|None|0.0|0.0|0.297607421875|
|80|an:7ogg04|fake|0.8643025884155996|0.762088374958342|0.9898769811964598|None|0.07614131221064814|1.7132161458333333|
|80|an:7ogg04|real|None|None|None|0.0|0.0|0.19584147135416666|
|80|an:728qx7|fake|0.40926195102345364|0.2573025213709413|0.9552891745576986|None|0.006705910011574074|1.5358072916666667|
|80|an:728qx7|real|None|None|None|0.0|0.0|0.24239095052083334|

资源与停止记录：
```json
{
  "status": "COMPLETE",
  "updates": 80,
  "exit_code": 0,
  "model_s": 0.7765688160434365,
  "load_s": 1.900408323854208,
  "stop_reason": "80_UPDATES",
  "BN_unchanged": true,
  "optimizer_states": 631,
  "wall_s": 1037.8580716913566,
  "peak_allocated": 3133802496,
  "peak_reserved": 12821987328,
  "peak_rss": 17069850624,
  "actual_worker_exit_code": 0
}
```
首／末更新三项loss（交替样本不同，不能直接当同一片段下降）：[[0.7056519985198975, 0.7198401689529419, 0.8952390551567078], [0.2514188289642334, 0.1437646448612213, 0.9999918937683105]]
最后更新剪裁前分支梯度：{"local.encoder": {"present": true, "finite": true, "norm": 0.060964521020650864}, "local.refine": {"present": true, "finite": true, "norm": 0.05046120285987854}, "pool": {"present": true, "finite": true, "norm": 0.0355101078748703}, "sets.participation.0": {"present": true, "finite": true, "norm": 0.00031881436007097363}, "sets.participation.1": {"present": true, "finite": true, "norm": 0.002430214313790202}, "sets.read_members": {"present": true, "finite": true, "norm": 0.0890025645494461}, "sets.interaction": {"present": true, "finite": true, "norm": 0.13383689522743225}, "sets.unit_readout": {"present": true, "finite": true, "norm": 0.3084075450897217}, "sets.current_readout": {"present": true, "finite": true, "norm": 0.0012688656570389867}, "sets.global_readout": {"present": true, "finite": true, "norm": 0.5879164934158325}, "sets.gru": {"present": true, "finite": true, "norm": 0.7883502840995789}, "sets.frame_head": {"present": true, "finite": true, "norm": 2.1372337341308594}, "pixel_context": {"present": true, "finite": true, "norm": 0.04178108647465706}}
代表性首／末前向、反向、AdamW及检查耗时：[{"forward_s": 1.7918529510498047, "backward_s": 3.086148609407246, "adamw_and_checks_s": 0.06081304978579283, "seconds": 4.938814610242844}, {"forward_s": 5.266058010980487, "backward_s": 13.951996530406177, "adamw_and_checks_s": 0.04048350267112255, "seconds": 19.258538044057786}]
固定关系消息干预：{
  "scope": "first fixed training fake clip, same final weights/eval; zero returned relation message and relation g only",
  "frame_logit_mean_abs_difference": 1.6796875,
  "pixel_logit_mean_abs_difference": 0.3806793689727783,
  "pixel_probability_mean_abs_difference": 0.05273805186152458
}
参与与状态记录：participation.json；独立sigmoid允许重叠，是否实际退化需结合该记录，不能由容量宣称已学到关系。
固定样例：samples_80/index.html；RGB | GT | 概率0–1 | 二值预测。
只判断本次执行和两个训练pair拟合进展；关系被消费不证明其泛化增益，Dice改善也不能单独证明关系有效。旧A/B/NORM的样本/预算不同，不作胜负比较。
完整产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_set_to_set_fit_check_v1`。本轮结束，不追加正式训练、其他seed、消融或数据。
