# PVST v0.3 目标帧2＋2＋2＋2资源验收

**状态：OOM；真实更新0次；正式模型0/2，不启动训练。**

源码：`b7b2f46a00c0bb03e3c34981ae3bcbf4706ad756`；产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_target_microbatch_2_gradient_check_v1`。
沿用原模型/初始化/AdamW/Dropout和BF16外层、逐head FP32、CSR桥接、CPU卸载、head暂存、即时QKV、node_split/view清理。attention512/QKV8192/FFN8192/point32768/tile256不变。
三层因果最大回溯14帧加visual两帧=16；同帧context、固定BN、逐位置LayerNorm、冻结全局坐标scale、真实PTS保持。
frame BCE逐帧标量、pixel BCE逐帧原生像素均值、softDice逐帧全像素sum/epsilon1；clip_terms按目标帧均值。此AN real全部合法零GT，分项均以原8目标帧为分母，权重按目标数/8计算；缺GT语义、loss权重不变。以后AF/AN配对外层权重仍需保留。
每组反向后释放图和Video GPU缓存，只保留梯度；所有组成功前不clip、不step、不zero_grad。未完成组不会计作更新。
计划每更新72帧次、24唯一输入、8唯一目标；重复历史重新计算。失败时实际加载组数见下，不能把计划量当作完成量。
小图沿用同非零诊断头、全部Dropout关闭、训练/checkpoint路径与既定容差；真实资源恢复原Dropout，随机轨迹随分组变化，不称训练逐步一致。

## 语义核验
```json
{
  "status": "PASS",
  "comparisons": [
    {
      "status": "PASS",
      "gate": "grouped_reference",
      "dtype": "torch.float32",
      "tolerances": {
        "grad_atol": 2e-05,
        "grad_rtol": 3e-05,
        "output_atol": 2e-05,
        "output_rtol": 1e-05
      },
      "comparison_rule": "actual first, reference second; abs(actual-reference)<=atol+rtol*abs(reference); nonfinite/missing mismatch fails",
      "checks": {
        "A_Bsum": false,
        "Bsum_Csum": true,
        "Cactual_Csum": true,
        "Cactual_Bsum": true,
        "A_Cactual": false,
        "B_C_group1": true,
        "B_C_group2": true,
        "B_C_group3": true,
        "B_C_group4": true
      },
      "output_loss_checks": {
        "A_B_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5441737510087714,
          "reference_norm": 0.5441737510087714
        },
        "A_C_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 2.9802322387695312e-08,
          "max_abs_over_reference_max": 1.087907494790555e-07,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5441737510087714,
          "reference_norm": 0.5441737542460482
        },
        "B_C_group1_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.31169567025266726,
          "reference_norm": 0.31169567025266726
        },
        "B_C_group2_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 2.9802322387695312e-08,
          "max_abs_over_reference_max": 1.087907494790555e-07,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.31472543664536917,
          "reference_norm": 0.31472546258575085
        },
        "B_C_group3_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.20685698456760174,
          "reference_norm": 0.20685698456760174
        },
        "B_C_group4_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 2.9802322387695312e-08,
          "max_abs_over_reference_max": 1.3872462544524537e-07,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.2390166686586472,
          "reference_norm": 0.2390166418719899
        },
        "A_B_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 1.3371631810375122,
          "reference_norm": 1.3371631810375122
        },
        "A_C_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 2.771615982055664e-06,
          "max_abs_over_reference_max": 9.208114071305259e-06,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 1.3371631810375122,
          "reference_norm": 1.337162767242467
        },
        "B_C_group1_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 4.470348358154297e-08,
          "max_abs_over_reference_max": 1.6257432762780537e-07,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.6813658075360534,
          "reference_norm": 0.6813658354981343
        },
        "B_C_group2_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 2.771615982055664e-06,
          "max_abs_over_reference_max": 1.018764435919443e-05,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.6000768955244756,
          "reference_norm": 0.600075714171644
        },
        "B_C_group3_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 5.960464477539063e-08,
          "max_abs_over_reference_max": 1.9802395852269374e-07,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.6652898501052877,
          "reference_norm": 0.6652898476305067
        },
        "B_C_group4_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 4.172325134277344e-07,
          "max_abs_over_reference_max": 1.4838392942690735e-06,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.7218331828391461,
          "reference_norm": 0.7218333742745827
        },
        "A_C_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 8.940696716308594e-08,
          "max_abs_over_reference_max": 4.593086593231888e-08,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 2.250007416698488,
          "reference_norm": 2.250007502955616
        },
        "A_B_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 5.960464477539063e-08,
          "max_abs_over_reference_max": 3.062057775702247e-08,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 2.250007416698488,
          "reference_norm": 2.250007465096654
        },
        "B_C_group1_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5566709151646213,
          "reference_norm": 0.5566709151646213
        },
        "B_C_group2_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 2.9802322387695312e-08,
          "max_abs_over_reference_max": 6.1267084173254e-08,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5623210580982827,
          "reference_norm": 0.5623210920811653
        },
        "B_C_group3_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.568228106635231,
          "reference_norm": 0.568228106635231
        },
        "B_C_group4_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 1.4901161193847656e-08,
          "max_abs_over_reference_max": 3.0601148546788174e-08,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5628330156646913,
          "reference_norm": 0.5628330197472574
        }
      },
      "all_original_mismatches_explained": true,
      "retained_gradient_tensors_on_disk": false,
      "A_loss_components": [
        1.9465551376342773,
        0.6052652597427368,
        0.6090775728225708,
        0.7322123646736145
      ],
      "B_weighted_loss_components": [
        [
          0.4815022647380829,
          0.14735689759254456,
          0.1513015329837799,
          0.18284383416175842
        ],
        [
          0.486432820558548,
          0.1480250358581543,
          0.15476855635643005,
          0.18363922834396362
        ],
        [
          0.4916723370552063,
          0.1556803584098816,
          0.15283319354057312,
          0.1831587702035904
        ],
        [
          0.48694777488708496,
          0.15420293807983398,
          0.15017428994178772,
          0.18257054686546326
        ]
      ],
      "C_weighted_loss_components": [
        [
          0.4815022647380829,
          0.14735689759254456,
          0.1513015329837799,
          0.18284383416175842
        ],
        [
          0.48643285036087036,
          0.1480250358581543,
          0.15476858615875244,
          0.18363922834396362
        ],
        [
          0.4916723370552063,
          0.1556803584098816,
          0.15283319354057312,
          0.1831587702035904
        ],
        [
          0.48694777488708496,
          0.15420295298099518,
          0.15017428994178772,
          0.18257054686546326
        ]
      ],
      "gradient_pair_summary": {
        "A_Bsum": {
          "mismatched_parameters": 1,
          "mismatched_elements": 3,
          "max_abs": 3.4477561712265015e-05
        },
        "Bsum_Csum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 4.425644874572754e-06
        },
        "Cactual_Csum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 8.940696716308594e-08
        },
        "Cactual_Bsum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 4.421919584274292e-06
        },
        "A_Cactual": {
          "mismatched_parameters": 1,
          "mismatched_elements": 2,
          "max_abs": 3.0055642127990723e-05
        },
        "B_C_group1": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 3.900378942489624e-06
        },
        "B_C_group2": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 2.339482307434082e-06
        },
        "B_C_group3": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 3.7178397178649902e-06
        },
        "B_C_group4": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 6.172340363264084e-07
        }
      },
      "exception_elements": [
        {
          "parameter": "decoder.final_conv_block.op.bias",
          "index": [
            6
          ],
          "A": -0.12205001711845398,
          "B_sum": -0.12208449468016624,
          "C_sum": -0.12208006903529167,
          "C_actual": -0.12208007276058197,
          "B_groups": [
            -0.02734094299376011,
            -0.020472751930356026,
            -0.0335364043712616,
            -0.04073439538478851
          ],
          "C_groups": [
            -0.02733989804983139,
            -0.020472832024097443,
            -0.03353268653154373,
            -0.04073465242981911
          ],
          "allowed_by_comparison": {
            "A_Bsum": 2.3662534840404988e-05,
            "Bsum_Csum": 2.366240207105875e-05,
            "Cactual_Csum": 2.366240207105875e-05,
            "Cactual_Bsum": 2.3662534840404988e-05,
            "A_Cactual": 2.366240218281746e-05,
            "B_C_group1": 2.0820196941494942e-05,
            "B_C_group2": 2.0614184960722925e-05,
            "B_C_group3": 2.1005980595946314e-05,
            "B_C_group4": 2.1222039572894574e-05
          },
          "original_C_minus_A": -3.0055642127990723e-05,
          "full_graph_B_minus_A": -3.4477561712265015e-05,
          "residual_C_minus_B": 4.421919584274292e-06,
          "residual_allowed": 2.3662534840404988e-05,
          "explained": true
        },
        {
          "parameter": "decoder.final_conv_block.op.bias",
          "index": [
            9
          ],
          "A": -0.13404925167560577,
          "B_sum": -0.13407491333782673,
          "C_sum": -0.13407296873629093,
          "C_actual": -0.13407295942306519,
          "B_groups": [
            -0.0251972284168005,
            -0.030325815081596375,
            -0.04273814707994461,
            -0.035813722759485245
          ],
          "C_groups": [
            -0.02519639953970909,
            -0.03032550774514675,
            -0.04273730516433716,
            -0.03581375628709793
          ],
          "allowed_by_comparison": {
            "A_Bsum": 2.4022247400134804e-05,
            "Bsum_Csum": 2.402218906208873e-05,
            "Cactual_Csum": 2.402218906208873e-05,
            "Cactual_Bsum": 2.4022247400134804e-05,
            "A_Cactual": 2.4022188782691958e-05,
            "B_C_group1": 2.0755891986191273e-05,
            "B_C_group2": 2.0909765232354404e-05,
            "B_C_group3": 2.1282119154930115e-05,
            "B_C_group4": 2.107441268861294e-05
          },
          "original_C_minus_A": -2.370774745941162e-05,
          "full_graph_B_minus_A": -2.5661662220954895e-05,
          "residual_C_minus_B": 1.953914761543274e-06,
          "residual_allowed": 2.4022247400134804e-05,
          "explained": true
        },
        {
          "parameter": "decoder.final_conv_block.op.bias",
          "index": [
            19
          ],
          "A": 0.09173545986413956,
          "B_sum": 0.09170827083289623,
          "C_sum": 0.09171100333333015,
          "C_actual": 0.09171099960803986,
          "B_groups": [
            0.024110691621899605,
            0.02360749989748001,
            0.022979848086833954,
            0.021010231226682663
          ],
          "C_groups": [
            0.02411312237381935,
            0.02360784262418747,
            0.0229797400534153,
            0.021010298281908035
          ],
          "allowed_by_comparison": {
            "A_Bsum": 2.275124812498689e-05,
            "Bsum_Csum": 2.2751330099999906e-05,
            "Cactual_Csum": 2.2751330099999906e-05,
            "Cactual_Bsum": 2.275124812498689e-05,
            "A_Cactual": 2.2751329988241196e-05,
            "B_C_group1": 2.0723393671214582e-05,
            "B_C_group2": 2.0708235278725626e-05,
            "B_C_group3": 2.068939220160246e-05,
            "B_C_group4": 2.063030894845724e-05
          },
          "original_C_minus_A": -2.4460256099700928e-05,
          "full_graph_B_minus_A": -2.718903124332428e-05,
          "residual_C_minus_B": 2.728775143623352e-06,
          "residual_allowed": 2.275124812498689e-05,
          "explained": true
        }
      ]
    },
    {
      "status": "PASS",
      "gate": "original_reference",
      "dtype": "torch.bfloat16",
      "tolerances": {
        "grad_atol": 0.2,
        "grad_rtol": 0.03,
        "output_atol": 0.02,
        "output_rtol": 0.02
      },
      "comparison_rule": "actual first, reference second; abs(actual-reference)<=atol+rtol*abs(reference); nonfinite/missing mismatch fails",
      "checks": {
        "A_Bsum": true,
        "Bsum_Csum": true,
        "Cactual_Csum": true,
        "Cactual_Bsum": true,
        "A_Cactual": true,
        "B_C_group1": true,
        "B_C_group2": true,
        "B_C_group3": true,
        "B_C_group4": true
      },
      "output_loss_checks": {
        "A_B_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5399039515094038,
          "reference_norm": 0.5399039515094038
        },
        "A_C_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001953125,
          "max_abs_over_reference_max": 0.007142857142857143,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5399039515094038,
          "reference_norm": 0.5401511877836265
        },
        "B_C_group1_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001953125,
          "max_abs_over_reference_max": 0.00823045267489712,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.30834796630228306,
          "reference_norm": 0.30984129011646455
        },
        "B_C_group2_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001953125,
          "max_abs_over_reference_max": 0.007142857142857143,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.3151937730396784,
          "reference_norm": 0.31444857576290025
        },
        "B_C_group3_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001953125,
          "max_abs_over_reference_max": 0.013245033112582781,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.20451466492392498,
          "reference_norm": 0.20309213225669356
        },
        "B_C_group4_frame_logit": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001953125,
          "max_abs_over_reference_max": 0.009174311926605505,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.2350413704027043,
          "reference_norm": 0.23587572918624042
        },
        "A_B_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0,
          "max_abs_over_reference_max": 0.0,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 1.3403222539961137,
          "reference_norm": 1.3403222539961137
        },
        "A_C_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001708984375,
          "max_abs_over_reference_max": 0.005627009646302251,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 1.3403222539961137,
          "reference_norm": 1.341091171408126
        },
        "B_C_group1_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.001708984375,
          "max_abs_over_reference_max": 0.006183745583038869,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.681298533814884,
          "reference_norm": 0.6821251036726644
        },
        "B_C_group2_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.00146484375,
          "max_abs_over_reference_max": 0.005319148936170213,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.602206050005568,
          "reference_norm": 0.6023920488483754
        },
        "B_C_group3_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.00146484375,
          "max_abs_over_reference_max": 0.00482315112540193,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.6689262005916311,
          "reference_norm": 0.6706056995700896
        },
        "B_C_group4_pixel_logits": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.00146484375,
          "max_abs_over_reference_max": 0.0051963050330007445,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.7226213834904474,
          "reference_norm": 0.7215557415451305
        },
        "A_C_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.00016486644744873047,
          "max_abs_over_reference_max": 8.467513005714254e-05,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 2.250749526753173,
          "reference_norm": 2.2505619208292384
        },
        "A_B_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 5.960464477539063e-08,
          "max_abs_over_reference_max": 3.061025242040623e-08,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 2.250749526753173,
          "reference_norm": 2.2507495783194904
        },
        "B_C_group1_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.0001386404037475586,
          "max_abs_over_reference_max": 0.00028785055333885605,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5569838290925038,
          "reference_norm": 0.5568266292073927
        },
        "B_C_group2_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 9.775161743164062e-06,
          "max_abs_over_reference_max": 2.010000980488283e-05,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5622094032226778,
          "reference_norm": 0.5621983193046145
        },
        "B_C_group3_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.00011302530765533447,
          "max_abs_over_reference_max": 0.00022982391458270765,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5683132526001963,
          "reference_norm": 0.56836224779869
        },
        "B_C_group4_loss": {
          "pass": true,
          "mismatches": 0,
          "max_abs": 0.00011593103408813477,
          "max_abs_over_reference_max": 0.00023790995432373515,
          "actual_zero": false,
          "reference_zero": false,
          "actual_norm": 0.5632904770478521,
          "reference_norm": 0.5632225246014415
        }
      },
      "all_original_mismatches_explained": true,
      "retained_gradient_tensors_on_disk": false,
      "A_loss_components": [
        1.947211742401123,
        0.6061400175094604,
        0.6089023947715759,
        0.7321693897247314
      ],
      "B_weighted_loss_components": [
        [
          0.4817788600921631,
          0.14760874211788177,
          0.15132202208042145,
          0.18284808099269867
        ],
        [
          0.48633599281311035,
          0.14804057776927948,
          0.1546768993139267,
          0.18361851572990417
        ],
        [
          0.4917476177215576,
          0.1558733582496643,
          0.152739018201828,
          0.1831352561712265
        ],
        [
          0.48734933137893677,
          0.1546173393726349,
          0.15016445517539978,
          0.1825675368309021
        ]
      ],
      "C_weighted_loss_components": [
        [
          0.4816402196884155,
          0.14750102162361145,
          0.1512969732284546,
          0.1828421950340271
        ],
        [
          0.4863262176513672,
          0.14803332090377808,
          0.1546749472618103,
          0.18361796438694
        ],
        [
          0.49179089069366455,
          0.15598638355731964,
          0.15268254280090332,
          0.1831219643354416
        ],
        [
          0.48728954792022705,
          0.15450140833854675,
          0.1502099633216858,
          0.1825781762599945
        ]
      ],
      "gradient_pair_summary": {
        "A_Bsum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.0087890625
        },
        "Bsum_Csum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.0078125
        },
        "Cactual_Csum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 2.7939677238464355e-09
        },
        "Cactual_Bsum": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.0078125
        },
        "A_Cactual": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.01171875
        },
        "B_C_group1": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.001953125
        },
        "B_C_group2": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.00390625
        },
        "B_C_group3": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.00390625
        },
        "B_C_group4": {
          "mismatched_parameters": 0,
          "mismatched_elements": 0,
          "max_abs": 0.001953125
        }
      },
      "exception_elements": [
        {
          "parameter": "decoder.final_conv_block.op.bias",
          "index": [
            6
          ],
          "A": -0.1240234375,
          "B_sum": -0.1243896484375,
          "C_sum": -0.1241455078125,
          "C_actual": -0.1241455078125,
          "B_groups": [
            -0.0286865234375,
            -0.01904296875,
            -0.03271484375,
            -0.0439453125
          ],
          "C_groups": [
            -0.0289306640625,
            -0.018798828125,
            -0.03271484375,
            -0.043701171875
          ],
          "allowed_by_comparison": {
            "A_Bsum": 0.203731689453125,
            "Bsum_Csum": 0.203724365234375,
            "Cactual_Csum": 0.203724365234375,
            "Cactual_Bsum": 0.203731689453125,
            "A_Cactual": 0.203724365234375,
            "B_C_group1": 0.200867919921875,
            "B_C_group2": 0.20056396484375,
            "B_C_group3": 0.2009814453125,
            "B_C_group4": 0.20131103515625
          },
          "original_C_minus_A": -0.0001220703125,
          "full_graph_B_minus_A": -0.0003662109375,
          "residual_C_minus_B": 0.000244140625,
          "residual_allowed": 0.203731689453125,
          "explained": true
        },
        {
          "parameter": "decoder.final_conv_block.op.bias",
          "index": [
            19
          ],
          "A": 0.0966796875,
          "B_sum": 0.0963134765625,
          "C_sum": 0.0960693359375,
          "C_actual": 0.0960693359375,
          "B_groups": [
            0.0244140625,
            0.025390625,
            0.023193359375,
            0.0233154296875
          ],
          "C_groups": [
            0.025146484375,
            0.025146484375,
            0.023193359375,
            0.0225830078125
          ],
          "allowed_by_comparison": {
            "A_Bsum": 0.20288940429687502,
            "Bsum_Csum": 0.202882080078125,
            "Cactual_Csum": 0.202882080078125,
            "Cactual_Bsum": 0.20288940429687502,
            "A_Cactual": 0.202882080078125,
            "B_C_group1": 0.20075439453125002,
            "B_C_group2": 0.20075439453125002,
            "B_C_group3": 0.20069580078125002,
            "B_C_group4": 0.200677490234375
          },
          "original_C_minus_A": -0.0006103515625,
          "full_graph_B_minus_A": -0.0003662109375,
          "residual_C_minus_B": -0.000244140625,
          "residual_allowed": 0.20288940429687502,
          "explained": true
        }
      ]
    }
  ]
}
```

## 实测阶段
|组/阶段|秒|allocated/reserved GiB|峰值A/R GiB|RSS GiB|
|---|---:|---:|---:|---:|
|None/worker_start|None|0.000/0.000|0.000/0.000|0.516|
|None/model_built|0.8260494312271476|0.066/0.080|0.066/0.080|0.678|
|1/before_microbatch_load|None|0.066/0.080|0.066/0.080|0.679|
|1/data_loaded|0.5152803482487798|1.216/1.320|1.252/1.320|1.203|
|1/forward_complete|17.766844197176397|4.943/11.756|9.491/11.756|14.429|

## 节点/边
|组/层|N|E|拓扑字节|
|---|---:|---:|---:|
|1/fine_time.0|2240579|27586977|459316272|
|1/fine_time.1|2240579|22869132|383830752|
|1/fine_time.2|2240579|21004368|353994528|
|1/coarse_time.0|1123151|20138619|331203120|
|1/coarse_time.1|1123151|16770405|277311696|
|1/coarse_time.2|1123151|14977876|248631232|

结果：成功step=0；优化器状态参数=0。
失败：['CUDA out of memory. Tried to allocate 548.00 MiB. GPU 0 has a total capacity of 23.51 GiB of which 445.75 MiB is free. Process 290469 has 507.28 MiB memory in use. Including non-PyTorch memory, this process has 21.86 GiB memory in use. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 808.53 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)']；阶段=backward；组=1
进程：{'pid': 900051, 'returncode': 2, 'termination_signal': None, 'timed_out': False, 'wall_s': 56.04233952611685, 'timeout_s': 900}
更新记录：[]
主机内存：{'available_bytes': 36813422592, 'rss_bytes': 15786573824, 'peak_rss_bytes': 18088882176}
完整逐组记录/资源采样/原生错误保存在max_window。未完成的反向与完整更新时间不填0，不用旧4＋4失败耗时计算加速比。
仅规定窗口资源检查，XYZ正式配置仍null，不改变节点门槛/模型完成数；不重跑前端、旧24帧/4＋4或额外样本。本轮停止，不重试、不追加逐帧微批或算子修改。
