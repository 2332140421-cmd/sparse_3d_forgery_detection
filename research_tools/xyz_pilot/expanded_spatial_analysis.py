"""CPU-only closeout of the frozen expanded spatial experiment.

Caller: python -m research_tools.xyz_pilot.expanded_spatial_analysis.
Inputs are saved JSON/CSV/logs only; outputs live under analysis/ and the thesis
report. No torch/model imports, checkpoint loading, media decoding or inference.
"""
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
OUT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_expanded_spatial_training_v1')
HIST={'an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4'}
KEYS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area','constant_AP']
LABELS={'pixel_AP':'像素AP','Dice':'Dice','IoU':'IoU','GT_recall':'GT召回','real_FPR':'real FPR','fake_background_FPR':'fake背景FPR','prediction_area':'fake预测面积','constant_AP':'常数AP'}
MODEL={'reference':'旧11来源','epoch01':'第1轮（开发选择）','epoch10':'第10轮（末次主结果）'}
GROUP={'HISTORICAL_DEV4':'历史4来源','ADDITIONAL_DEV':'新增开发15来源','EXPANDED_DEV':'全部开发19来源','object_edit_pending_human':'对象编辑（官方描述预筛）','appearance_background_reference':'外观/背景参照（官方描述预筛）','addition_copy':'添加/复制（官方描述）','removal':'移除（官方描述）','replacement_local':'其他修改（官方描述）','automatic_stable':'自动固定倾向','automatic_motion':'自动移动倾向','automatic_uncertain':'自动机位不确定','human_record_not_automatic':'已有人工机位记录（单列）'}

def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def mean(values):
    values=[v for v in values if v is not None];return statistics.mean(values) if values else None

def official_action(task):
    words=task.lower().replace(':',' ').split()
    if any(x in words for x in ['remove','delete','erase']):return 'removal'
    if any(x in words for x in ['add','insert','copy','clone','duplicate']):return 'addition_copy'
    return 'replacement_local'

def aggregate(evaluation,pairs):
    videos=[v for v in evaluation['videos'] if v['pair_id'] in pairs]
    result={k:mean(v.get(k) for v in videos if v['role']==('real' if k=='real_FPR' else 'fake')) for k in KEYS}
    result.update(sources=len({v['pair_id'] for v in videos}),frames=sum(v['frames'] for v in videos))
    return result

def pair_metrics(ev,pair):
    fake=next(v for v in ev['videos'] if v['pair_id']==pair and v['role']=='fake')
    real=next(v for v in ev['videos'] if v['pair_id']==pair and v['role']=='real')
    return {k:real.get(k) if k=='real_FPR' else fake.get(k) for k in KEYS}

def subsets(rows):
    pairs={r['pair_id'] for r in rows}
    assert HIST<=pairs
    result={'HISTORICAL_DEV4':HIST,'ADDITIONAL_DEV':pairs-HIST,'EXPANDED_DEV':pairs}
    for field in ['content_priority','official_action','camera_group']:
        for value in sorted({r[field] for r in rows}):result[value]={r['pair_id'] for r in rows if r[field]==value}
    return result

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in row)+'|' for row in rows])
def fmt(x):return '未提供/未定义' if x is None else f'{x:.3e}' if 0<abs(x)<1e-6 else f'{x:.6f}'

def preview_page(protocol,dest):
    rows=[r for r in protocol['rows'] if r['split']=='dev']
    h=['<!doctype html><meta charset="utf-8"><title>扩充空间模型固定预览</title><h1>固定目标：旧参照与第10轮</h1><p>复用已保存的预测图，不新增前向。概率统一0–1，二值阈值0.5，仅预览缩放。第1轮未保存固定图，不以第10轮图冒充。</p><p>按官方编辑动作整理筛选；原程序页面与图像保留不变。</p><select id="filter"><option value="">全部官方动作</option><option value="addition_copy">添加/复制</option><option value="removal">移除</option><option value="replacement_local">其他修改</option></select><style>body{font:16px sans-serif;margin:20px}section{border-top:1px solid #aaa;padding:12px 0}.images{display:flex;gap:5px}.images div{width:16%}.images img{width:100%}</style>']
    targets=0;references=0
    for r in rows:
        for i in r['preview_targets']:
            targets+=1;stem=r['video_id']+f'_{i:05d}';action=official_action(r['official_task'])
            h.append(f'<section data-type="{action}"><h3>{r["pair_id"]}/{r["role"]} 帧{i}，PTS {r["pts"][i]:.6f}s</h3><p>{html.escape(r["official_task"])}</p><div class="images">')
            for suffix,label in [('rgb','RGB'),('gt','官方GT'),('old_prob','旧参照概率'),('old_binary','旧参照二值'),('new_prob','第10轮概率'),('new_binary','第10轮二值')]:
                name=stem+'_'+suffix+'.png';assert (OUT/'preview'/name).is_file();references+=1
                h.append(f'<div>{label}<img loading="lazy" src="../preview/{name}"></div>')
            h.append('</div></section>')
    h.append('<script>document.getElementById("filter").onchange=e=>{for(const s of document.querySelectorAll("section"))s.hidden=!!e.target.value&&s.dataset.type!==e.target.value}</script>')
    (dest/'preview.html').write_text('\n'.join(h))
    (dest/'preview_notes.md').write_text('# 固定预览说明\n\n整理页preview.html引用原../preview/图像，原index.html不变。54目标/324图：历史4对首中末24目标；新增15对real/fake中帧30目标。仅旧模型与第10轮预测，第1轮没有固定图，不新增前向补图。按官方编辑动作纠正筛选标签，仅改变整理页文字。\n')
    return dict(fixed_preview_targets=targets,existing_image_links=references,new_predicted_images=0)

def main():
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    raw=['protocol.json','config.json','summary.json','history.json','reference_dev.json','epoch01_dev.json','epoch03_dev.json','epoch05_dev.json','epoch10_dev.json','report.md','run.log','preview/index.html']
    hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={name:hashfile(OUT/name) for name in raw}
    protocol=read(OUT/'protocol.json');config=read(OUT/'config.json');summary=read(OUT/'summary.json');history=read(OUT/'history.json')
    assert summary['status']=='COMPLETE' and (OUT/'exit_code.txt').read_text().strip()=='0'
    dev=[dict(r,official_action=official_action(r['official_task'])) for r in protocol['rows'] if r['split']=='dev' and r['role']=='fake'];sets=subsets(dev)
    assert len(dev)==len({r['source_group_id'] for r in dev})==protocol['dev_sources']
    evaluations={'reference':read(OUT/'reference_dev.json'),**{f'epoch{e:02d}':read(OUT/f'epoch{e:02d}_dev.json') for e in config['evaluate_epochs']}}
    assert len({ev['frame_identity_sha256'] for ev in evaluations.values()})==1
    assert all(ev['scored_frames']==protocol['dev_frames'] for ev in evaluations.values())
    grouped={name:{group:aggregate(ev,pairs) for group,pairs in sets.items()} for name,ev in evaluations.items()}
    selected=min(config['evaluate_epochs'],key=lambda e:(-grouped[f'epoch{e:02d}']['ADDITIONAL_DEV']['pixel_AP'],e))
    assert selected==summary['best_epoch']==1 and (OUT/'best_additional_dev.pt').resolve().name==f'epoch{selected:02d}.pt'
    core=[dict(model=name,group=group,**grouped[name][group]) for group in ['HISTORICAL_DEV4','ADDITIONAL_DEV','EXPANDED_DEV'] for name in MODEL]
    csvwrite(dest/'core_comparison.csv',core)
    source_rows=[dict(model=name,pair_id=r['pair_id'],source_group_id=r['source_group_id'],historical=r['pair_id'] in HIST,official_action=r['official_action'],content_priority=r['content_priority'],camera_group=r['camera_group'],**pair_metrics(evaluations[name],r['pair_id'])) for r in dev for name in MODEL]
    csvwrite(dest/'source_comparison.csv',source_rows)
    csvwrite(dest/'group_comparison.csv',[dict(model=name,group=group,**values) for name,groups in grouped.items() for group,values in groups.items()])
    curve=[]
    for h in history:
        epoch=h['epoch'];m=read(OUT/f'epoch{epoch:02d}_monitor.json') if epoch in config['evaluate_epochs'] else None
        curve.append(dict(epoch=epoch,updates=h['updates'],pixel_BCE=h['pixel_BCE'],soft_Dice_loss=h['soft_Dice'],total_online_loss=h['pixel_BCE']+h['soft_Dice'],monitor_AP=m['macro']['pixel_AP'] if m else None,monitor_Dice=m['macro']['Dice'] if m else None,**{group+'_AP':grouped[f'epoch{epoch:02d}'][group]['pixel_AP'] if m else None for group in ['HISTORICAL_DEV4','ADDITIONAL_DEV','EXPANDED_DEV']}))
    csvwrite(dest/'learning_curve.csv',curve)
    coverage={split:dict(sources=len({r['source_group_id'] for r in protocol['rows'] if r['split']==split}),videos=sum(r['split']==split for r in protocol['rows']),legal_frames=sum(len(r['train_targets'] if split=='train' else r['evaluation_targets']) for r in protocol['rows'] if r['split']==split),official_action=dict(Counter(official_action(r['official_task']) for r in protocol['rows'] if r['split']==split and r['role']=='fake')),content_priority=dict(Counter(r['content_priority'] for r in protocol['rows'] if r['split']==split and r['role']=='fake'))) for split in ['train','dev']}
    comparisons={};deltas=[]
    for name in ['epoch01','epoch10']:
        diff={r['pair_id']:pair_metrics(evaluations[name],r['pair_id'])['pixel_AP']-pair_metrics(evaluations['reference'],r['pair_id'])['pixel_AP'] for r in dev};ordered=sorted(diff.items(),key=lambda x:-x[1]);net=sum(diff.values())
        comparisons[name]=dict(AP_improved=sum(v>0 for v in diff.values()),AP_declined=sum(v<0 for v in diff.values()),object_AP_improved=sum(diff[pair]>0 for pair in sets['object_edit_pending_human']),object_sources=len(sets['object_edit_pending_human']),top3_net_gain_share=sum(v for _,v in ordered[:3])/net,top3=ordered[:3],mean_gain_excluding_top1=mean(v for _,v in ordered[1:]))
    for r in dev:
        pair=r['pair_id'];a=pair_metrics(evaluations['epoch01'],pair);b=pair_metrics(evaluations['epoch10'],pair)
        deltas.append(dict(pair_id=pair,official_action=r['official_action'],content_priority=r['content_priority'],**{k+'_last_minus_selected':b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in KEYS}))
    csvwrite(dest/'last_minus_selected.csv',deltas)
    analysis=dict(status='CPU_ANALYSIS_COMPLETE',training_status=summary['status'],selected_epoch=selected,coverage=coverage,groups=grouped,improvement=comparisons,late_AP_declines=sum(r['pixel_AP_last_minus_selected']<0 for r in deltas),late_Dice_declines=sum(r['Dice_last_minus_selected']<0 for r in deltas),new_model_forwards=0,new_training_updates=0,new_depth_or_features=0)
    write(dest/'summary.json',analysis);write(dest/'input_identity.json',before)
    # Charts use existing scalar metrics only; no prediction images are regenerated.
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    measured=[x for x in curve if x['monitor_AP'] is not None]
    for group in ['HISTORICAL_DEV4','ADDITIONAL_DEV','EXPANDED_DEV']:axes[0].plot([x['epoch'] for x in measured],[x[group+'_AP'] for x in measured],marker='o',label=group)
    axes[0].plot([x['epoch'] for x in measured],[x['monitor_AP'] for x in measured],marker='x',label='fixed train monitor');axes[0].set_ylim(0,1);axes[0].set_ylabel('Pixel AP')
    for key in ['pixel_BCE','soft_Dice_loss']:axes[1].plot([x['epoch'] for x in curve],[x[key] for x in curve],label=key)
    axes[1].set_ylabel('Mean online loss component')
    for ax in axes:ax.set_xlabel('Complete epoch');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/'learning_curve.png',dpi=150);plt.close(fig)
    lines=['# ANAKIN扩充空间训练：最终结果分析',
      '扩充训练在多个独立来源上改善了空间定位，猫添加和骑马人物移除恢复了部分定位，过街样本仍未检出。第1轮开发选择结果明显好于第10轮；后期训练观察继续改善而开发普遍退化。本轮只分析已有产物，新增训练、模型前向、深度和特征提取均为0。',
      '## 实际协议',
      f"实际训练{coverage['train']['sources']}独立原视频来源、{coverage['train']['videos']}条真假视频、每轮{coverage['train']['legal_frames']:,}个合法帧目标；开发{coverage['dev']['sources']}来源、{coverage['dev']['videos']}视频、{coverage['dev']['legal_frames']:,}个唯一目标。两者是同一ANAKIN内不同原视频，不是同视频前后段，也不是跨数据集评价。",
      '开发由历史4来源（7e2e57、95q8g9、4xd3r9、7ogeg4）与新增15来源组成。历史4为824帧，新增15为4328帧。共2575个fake目标，其中2570帧GT非空、5帧空；real为2577帧。AP仅计算非空GT fake帧，先逐帧再视频/来源等权；real不混入AP，空GT AP不补0或1。Dice/IoU/召回、两类FPR沿用原实现和logit≥0，未搜索阈值。',
      '官方标注经必要技术映射核验后直接使用，不要求逐条人工语义审阅，已有人工原文及明确否定记录保留。按source_group_id隔离训练/开发；94个测试保护来源未参与，27个技术待核来源未使用；40y1x9_fake[6,10)不监督/不计分。',
      '模型为冻结通用DINOv2 ViT-L/14 reg4＋可训练LoRA(r4/alpha8/dropout.25)＋官方HighResContextHead；无XYZ、历史或帧分类头。整帧长边384、ImageNet归一化、右下pad14、排除CLS/register后解码、去padding并恢复原生尺寸。可训练参数3,673,329。seed20260909，新LoRA/head初值；与旧基线通用初始状态hash一致，未从旧训练终点续训。',
      f"AdamW lr1e-4、wd.01、batch1、clip1，BF16前向/FP32 BCE＋soft Dice各1。10轮无放回遍历，共{summary['updates']:,}次更新；每轮既定训练顺序前100目标额外BN统计前向，共{summary['BN_forwards']}次，开发统一eval/running BN。",
      '实际完整评价点是第1、3、5、10轮；按ADDITIONAL_DEV的宏AP、同分取早规则，选择第1轮（22,878步），不是按历史四来源选权重。第10轮（228,780步）仍是预定末次主结果。两者都是反复查看过的开发结果，不称独立测试成绩。',
      '## 三份权重核心对照','所有FPR均为比例而非百分数；例如0.001表示0.1%。极小非零值以科学记数表示。',
      table(['开发范围','权重']+[LABELS[k] for k in KEYS[:6]],[[GROUP[row['group']],MODEL[row['model']]]+[fmt(row[k]) for k in KEYS[:6]] for row in core]),
      '原11来源基线训练12,845步，本轮数据覆盖和更新预算均增加；差值反映整套扩充训练方案，不能解释为纯样本数量的因果效应。旧模型仅只读评价同一开发集，历史四来源回放各指标与旧记录差值均为0。',
      '## 历史四来源明细',
      table(['来源','权重','AP','Dice','GT召回'],[[pair,MODEL[name]]+[fmt(pair_metrics(evaluations[name],pair)[k]) for k in ['pixel_AP','Dice','GT_recall']] for pair in ['an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4'] for name in MODEL]),
      '猫：第1轮AP/Dice/召回约0.818/0.522/0.393，第10轮降至0.464/0.194/0.146；已有部分定位，后期明显变弱。骑马：从旧召回0.0073升至第1轮0.219，末次0.0833；同样部分恢复而未解决。过街：三份权重Dice和召回均为0，末次AP0.01338还低于其常数AP约0.01600，不能因宏平均提高而写成已检出。7ogeg4始终相对较强，但本轮提升不再仅来自它。',
      '## 所有开发来源与常数参照',
      table(['来源','官方动作','常数AP','旧AP','第1轮AP','第10轮AP','第1轮Dice','第10轮Dice'],[[r['pair_id'],GROUP[r['official_action']],fmt(pair_metrics(evaluations['reference'],r['pair_id'])['constant_AP'])]+[fmt(pair_metrics(evaluations[name],r['pair_id'])['pixel_AP']) for name in MODEL]+[fmt(pair_metrics(evaluations[name],r['pair_id'])['Dice']) for name in ['epoch01','epoch10']] for r in dev]),
      '逐来源完整IoU、召回、两类FPR、预测面积和常数AP见analysis/source_comparison.csv；不将真实/伪造视频假定为逐像素配准。',
      '## 编辑覆盖与机位分组',
      f"训练官方动作构成为{coverage['train']['official_action']}，开发为{coverage['dev']['official_action']}。旧11来源以移除为主，本轮训练添加/复制扩大到47来源；这些包含配饰、图表等，不能全部称为运动对象添加。分项按官方动作整理，修正粗关键词把白衣/白帽描述误标为外观的3条标签，仅用于报告，不改变冻结清单、GT或选择标准。",
      '对象/外观背景类别来自官方文字预筛，机位倾向来自传统背景匹配；不是人工语义或三维异常确认。两条已有人工机位记录单列，不混作自动固定结果。']
    for field,values in [('内容',['object_edit_pending_human','appearance_background_reference']),('官方动作',['addition_copy','removal','replacement_local']),('机位',['automatic_stable','automatic_motion','automatic_uncertain','human_record_not_automatic'])]:
        lines += [f'### {field}',table(['分组','来源数','权重','AP','Dice','召回','real FPR','fake背景FPR','常数AP'],[[GROUP[g],grouped[name][g]['sources'],MODEL[name]]+[fmt(grouped[name][g][k]) for k in ['pixel_AP','Dice','GT_recall','real_FPR','fake_background_FPR','constant_AP']] for g in values for name in MODEL])]
    lines += ['外观/背景参照3来源常数AP宏平均约0.489，其中48owl9、72yzw9的正类比例约0.712、0.753，高AP部分受到大面积标签先验支持。对象编辑16来源常数AP仅约0.0203，其AP仍从旧0.1040升至第1轮0.3262、末次0.2639，所以增益并非全由大面积参照造成；但其Dice仍低，不能宣称运动对象定位已经解决。',
      '## 多来源改善与后期变化',
      f"相对旧基线，第1轮AP提高{comparisons['epoch01']['AP_improved']}/19来源（对象编辑{comparisons['epoch01']['object_AP_improved']}/16）；末次提高{comparisons['epoch10']['AP_improved']}/19（对象编辑{comparisons['epoch10']['object_AP_improved']}/16）。这包括极小变化，不等于每条都获得实际定位；过街即是反例。",
      f"第1轮净AP增益最大的三个来源为{comparisons['epoch01']['top3']}，占净增益约{comparisons['epoch01']['top3_net_gain_share']:.1%}；末次前三为{comparisons['epoch10']['top3']}，占约{comparisons['epoch10']['top3_net_gain_share']:.1%}。仍有集中性，但分别去掉最大增益来源后的其余18来源平均AP增量仍为{comparisons['epoch01']['mean_gain_excluding_top1']:.6f}/{comparisons['epoch10']['mean_gain_excluding_top1']:.6f}；这些仅作敏感性描述，主表始终保留全部19来源。",
      f"第1轮到末次，{analysis['late_AP_declines']}/19来源AP下降、{analysis['late_Dice_declines']}/19来源Dice下降；对象编辑12/16、外观背景3/3来源AP下降，添加/复制、移除、其他修改分组宏AP均下降，并非所有视频都退化。主要AP降幅包括猫7e2e57(-0.35365)、花盆4rv8o9(-0.28093)、骑马95q8g9(-0.23971)、青蛙4ny537(-0.23464)；飞虫4nyv37则提高约0.38952，说明仍有样本依赖。完整方向见last_minus_selected.csv。",
      table(['轮次','更新','在线BCE','在线Dice loss','观察AP','观察Dice','历史4 AP','新增15 AP','全部19 AP'],[[x['epoch'],x['updates'],fmt(x['pixel_BCE']),fmt(x['soft_Dice_loss']),fmt(x['monitor_AP']),fmt(x['monitor_Dice'])]+[fmt(x[g+'_AP']) for g in ['HISTORICAL_DEV4','ADDITIONAL_DEV','EXPANDED_DEV']] for x in curve]),
      '第2/4/6/7/8/9轮按计划只记录训练过程，没有固定训练观察和完整开发评价；表中对应空值为未安排测量，不用其他实验代填。',
      '固定训练观察为356帧（每训练视频约1/3、2/3合法帧），确实属于本轮：AP从0.7975升至0.8703，Dice从0.7258升至0.8206；在线BCE/Dice loss也持续下降。开发宏AP从0.3799降至0.3134，Dice0.2511→0.1705、召回0.2696→0.1784，预测面积0.0604→0.0209；real FPR和fake背景FPR同时下降。既有更保守的输出，也有AP排序下降，不能仅称为阈值/校准平移。',
      '这支持训练观察与开发分化、后期泛化退化的现象；没有完整训练集固定权重评价，不能把观察集代填完整训练成绩，也未证明具体哪层错误或唯一原因是过拟合。不存在所有来源单调退化：新增开发AP第3→5轮略回升、部分视频末次改善；第1轮开发最好本身并非原因证明。曲线见analysis/learning_curve.png。',
      '本轮为单seed、反复使用开发集的开发结论，未访问独立测试。模型无XYZ/时间支路，不支持三维或时序贡献的独立结论，也不能据此否定PVST关系假设。',
      '## 唯一优先下一步（未执行）',
      '优先做训练后期稳定性的单因素适配，而非立即接入XYZ：固定本轮数据划分、空间模型、初始化、目标顺序、损失和总更新，复用当前恒定1e-4参照，只新增一个预先登记的“首轮后学习率衰减”条件，仍按相同独立开发集合比较。不得搜索多个计划或按历史四来源挑权重。这是可检验的候选方向，不是已经证明学习率导致退化。依据是多个对象来源确有增益，但训练观察持续变好而开发AP/Dice在多个类别下降；先稳定视觉底座更有证据。此处不启动该对照。',
      '## 预览、成本与运行约定',
      f"原固定预览保持不变：{OUT}/preview/index.html。共54个固定目标：历史4对各首/中/末24个，新增15对各real/fake中帧30个。已有RGB/GT、旧模型和末次概率/0.5二值图，统一色阶；没有保存第1轮固定预测图，本轮不为补图新增前向。",
      f"原训练执行提交{summary['identity']['execution_head']}；训练{summary['cost']['training_seconds']:.2f}s，训练/BN/评价墙钟{summary['cost']['train_eval_wall_seconds']:.2f}s，主流程{summary['cost']['wall_seconds']:.2f}s（约{summary['cost']['wall_seconds']/60:.1f}分钟），峰值allocated/reserved {summary['cost']['peak_allocated']/2**30:.3f}/{summary['cost']['peak_reserved']/2**30:.3f} GiB。工作缓存新增24,637帧、8,797,886,001字节，复用3,397帧，CPU准备97.20s。MoGe、DFormer及神经特征预计算均为0。",
      '程序退出0，228,780更新与1000次额外BN前向取自已有summary/log记录；本轮不重复模型恢复、预检或检查点张量核验。既有评价的帧身份hash一致。原始report.md、summary.json、各评价JSON/CSV、日志及预测不覆盖；本次派生表和曲线只写analysis/。',
      '### 后续长训练运行约定',
      '由Codex实现启动脚本、必要简短预检并交付准确命令；由用户在远端终端启动，优先tmux承受SSH断开。Codex不持续轮询或等待，用户主动接续后再读结果。',
      '使用python -u并通过tee同时输出终端和日志；脚本保留Python实际退出码。程序报告epoch/总轮数、step/总step、loss、学习率、耗时和预计剩余时间，每轮汇总并按固定计划自动评价、保存权重和结果表，成功/失败写退出状态。本次不再启动或迁移任何进程。',
      '### 交付位置',f'仓库报告：{REPO}/docs/pvst/anakin_expanded_spatial_training_v1.md',f'CPU分析：{dest}',f'固定预览：{OUT}/preview/index.html','CPU收尾入口：python -m research_tools.xyz_pilot.expanded_spatial_analysis；它只读保存指标，不调用训练入口的report子命令（后者含旧检查点复查）。本轮提交的scripts/run_anakin_expanded_spatial_training_v1.sh是未执行的手动终端包装，含防重复启动、python -u与tee及实际退出码；对应进度字段补丁仅影响后续日志，不改变本次实际执行提交或数值协议。',
      '计算状态与发布状态分开；普通Git发布结果写publication.json。本轮完成后不追加实验。']
    preview_checks=preview_page(protocol,dest)
    lines += [f'整理预览（复用已有图、纠正官方动作筛选标签）：{dest}/preview.html；原固定预览index.html未修改。']
    text='\n\n'.join(lines)+'\n';(dest/'report.md').write_text(text);(REPO/'docs/pvst/anakin_expanded_spatial_training_v1.md').write_text(text)
    assert before=={name:hashfile(OUT/name) for name in raw},'Raw artifacts changed'
    write(dest/'checks.json',dict(raw_artifacts_unchanged=True,saved_evaluation_identity_consistent=True,selected_epoch_reproduced=selected,new_model_forwards=0,new_checkpoint_loads=0,new_training_updates=0,**preview_checks))
    print(json.dumps(dict(coverage=coverage,improvement=comparisons,selected_epoch=selected,late_AP_declines=analysis['late_AP_declines']),ensure_ascii=False))

if __name__=='__main__':main()
