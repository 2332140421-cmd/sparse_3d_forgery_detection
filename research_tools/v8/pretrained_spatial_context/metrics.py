"""Spatial-only native metrics and reports; no detection aggregation."""
import json
from pathlib import Path
import time

import numpy as np
import cv2
from sklearn.metrics import average_precision_score
import torch

from research_tools.v8.spatial_localization.data import to_source
from research_tools.v8.spatial_localization.metrics import spatial_frame
from research_tools.v8.spatial_localization.head_decoupling import sample_index
from research_tools.v8.spatial_localization.joint_data import save
from .assets import CachedVideo
from .model import pixel_sums,CONDITIONS


@torch.no_grad()
def evaluate(model,config,out,rows,fit,pixel_ap=False,samples=None):
    model.eval();device=next(model.parameters()).device;records=[];started=time.monotonic()
    for r in rows:
        v=CachedVideo(config,out,r,fit,model.context);h=torch.zeros(256,128,device=device);active=torch.zeros(256,dtype=torch.bool,device=device)
        sums=np.zeros(3);ious=[];dices=[];aps=[];empty=[];cm=np.zeros((2,2),np.int64)
        keep={int(r['actual_frames']*f) for f in [.25,.5,.75]};export=samples is not None and (r['split']=='dev' or r.get('train_monitor'))
        if export:(Path(samples)/r['split']).mkdir(parents=True,exist_ok=True)
        try:
            for start in range(0,r['actual_frames'],config['chunk_frames']):
                a=v.chunk(start,min(start+config['chunk_frames'],r['actual_frames']))
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['amp']):
                    h,active,z=model.chunk(a,h,active,start==0,r['mapping256'],r['mapping512'])
                    losses=pixel_sums(z,a['gt'].to(device),a['content512'].to(device))
                sums+=np.array([float(x) for x in losses])
                for j,canvas in enumerate(z.float().cpu().numpy()):
                    fi=start+j;gt=v.native_gt(fi);native=to_source(canvas,r['mapping512']);prediction=native>=0
                    spatial=spatial_frame(prediction,gt);cm+=np.asarray(spatial['confusion_matrix'])
                    if gt.any():
                        ious.append(spatial['iou']);dices.append(spatial['dice'])
                        if pixel_ap:aps.append(float(average_precision_score(gt.reshape(-1),native.reshape(-1))))
                    else:empty.append(spatial['empty_fpr'])
                    if export and fi in keep:
                        rgb=np.asarray(v.base.arrays['rgb512'][fi]);metadata={k:r[k] for k in ['video_id','source_id','role','split','mapping512','mapping256']}
                        metadata.update(extra_padding_right_bottom=[6,6],target='edit_region')
                        np.savez_compressed(Path(samples)/r['split']/f'{r["video_id"]}_{fi}.npz',
                            logits=canvas.astype(np.float32),gt512=a['gt'][j].numpy(),native_gt=gt,
                            rgb512=rgb,content512=a['content512'][j].numpy(),frame_index=np.array(fi),
                            pts_s=np.array(r['pts'][fi]['pts_s']),metadata=np.array(json.dumps(metadata)))
            records.append({**{k:r[k] for k in ['video_id','pair_id','source_id','role','split','generator']},
                'frames':r['actual_frames'],'loss':sums[0]/r['actual_frames'],'bce':sums[1]/r['actual_frames'],'dice_loss':sums[2]/r['actual_frames'],
                'spatial_dice':float(np.mean(dices)) if dices else None,'spatial_iou':float(np.mean(ious)) if ious else None,
                'pixel_ap':float(np.mean(aps)) if aps else None,'pixel_ap_reason':None if aps else 'not_scheduled_or_no_positive_GT',
                'positive_gt_frames':len(dices),'empty_gt_frames':len(empty),
                'empty_spatial_fpr':float(np.mean(empty)) if empty else None,'pixel_confusion_matrix':cm.tolist()})
        finally:v.close()
    mean=lambda k:float(np.mean([r[k] for r in records if r[k] is not None])) if any(r[k] is not None for r in records) else None
    real=[r for r in records if r['role']=='real']
    metrics={k:mean(k) for k in ['loss','bce','dice_loss','spatial_dice','spatial_iou','pixel_ap']}
    metrics['empty_spatial_fpr']=mean('empty_spatial_fpr')
    metrics['real_spatial_fpr']=float(np.mean([r['empty_spatial_fpr'] for r in real]))
    return {'metrics':metrics,'records':records,'seconds':time.monotonic()-started,'pixel_ap_scheduled':pixel_ap}


def boundary_samples(out,rows):
    """Descriptive fixed-sample errors in native 2-pixel GT bands; no label editing."""
    records=[];kernel=np.ones((5,5),np.uint8)
    for c in CONDITIONS:
        for r in rows:
            if r['split']!='dev' or r['role']!='fake':continue
            for fi in [int(r['actual_frames']*f) for f in [.25,.5,.75]]:
                with np.load(out/c/'samples'/'best'/'dev'/f'{r["video_id"]}_{fi}.npz') as z:
                    gt=z['native_gt'].astype(bool);pred=to_source(z['logits'],r['mapping512'])>=0
                inside=cv2.erode(gt.astype(np.uint8),kernel).astype(bool)
                outside=~cv2.dilate(gt.astype(np.uint8),kernel).astype(bool)
                regions={'interior_recall':inside,'GT_boundary_recall':gt&~inside,
                    'near_boundary_BG_FPR':~gt&~outside,'far_BG_FPR':outside}
                records.append({'condition':c,'video_id':r['video_id'],'frame_index':fi,
                    **{k:float(pred[m].mean()) if m.any() else None for k,m in regions.items()}})
    aggregates={}
    for c in CONDITIONS:
        aggregates[c]={}
        for key in ['interior_recall','GT_boundary_recall','near_boundary_BG_FPR','far_BG_FPR']:
            means=[]
            for vid in {r['video_id'] for r in records if r['condition']==c}:
                values=[r[key] for r in records if r['condition']==c and r['video_id']==vid and r[key] is not None]
                if values:means.append(float(np.mean(values)))
            aggregates[c][key]=float(np.mean(means)) if means else None
    save(out/'fixed_boundary_statistics.json',{'scope':'best dev fake fixed25/50/75%; native2px bands; video means; GT unchanged',
        'aggregates':aggregates,'records':records})
    return aggregates


def curves(out,condition,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8));epochs=[r['epoch'] for r in history]
    for ax,k in zip(axes.flat[:3],['loss','bce','dice_loss']):
        for split in ['train','dev']:ax.plot(epochs,[r[split][k] for r in history],label=split)
        online='dice' if k=='dice_loss' else k
        ax.plot(epochs,[r['online'][online] for r in history],label='online',linestyle=':');ax.set_title(k)
    for ax,k in zip(axes.flat[3:],['spatial_dice','spatial_iou','real_spatial_fpr']):
        for split in ['train','dev']:ax.plot(epochs,[r[split][k] for r in history],label=split)
        ax.set_title(k)
    for ax in axes.flat:ax.set_xlabel('epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/condition/'curves.png');plt.close(fig)


def report(out,rows):
    summaries={c:json.loads((out/c/'summary.json').read_text()) for c in CONDITIONS}
    epoch0=json.loads((out/'epoch0.json').read_text())
    chosen=[r for r in rows if r['split']=='dev' or r.get('train_monitor')]
    sample_index(out,chosen,conditions=CONDITIONS,title='AINPAINT迁移与完整三维时序增强：各自best/last')
    boundary=boundary_samples(out,rows)
    fixed=(out/'reference_reuse.json').exists()
    lines=['# V8预训练空间定位基础与三维时序增强'+('（FIXED_BN）' if fixed else ''),'',
        'AINPAINT官方spatial模型在ANAKIN上的统一迁移，不是论文原设置复现。只有空间BCE+Dice，无检测损失/门控。','',
        '|条件|checkpoint/epoch|集合|Dice|IoU|pixel AP|real空间FP|','|---|---|---|---|---|---|---|']
    fmt=lambda x:'未计算' if x is None else f'{x:.6f}'
    for split in ['train','dev']:
        m=epoch0[split]['metrics'];lines.append(f'|共同官方权重|epoch0/0|{split}|'+ '|'.join(fmt(m[k]) for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr'])+'|')
    for checkpoint in ['best','last']:
        for c,s in summaries.items():
            for split in ['train','dev']:
                m=s[checkpoint][split]['metrics']
                lines.append(f'|{c}|{checkpoint}/{s[checkpoint+"_epoch"]}|{split}|'+ '|'.join(fmt(m[k]) for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr'])+'|')
    a,b=[summaries[c] for c in CONDITIONS];ad=a['best']['dev']['metrics']['spatial_dice'];bd=b['best']['dev']['metrics']['spatial_dice'];base=epoch0['dev']['metrics']['spatial_dice']
    lines.extend(['',f'各自best开发Dice增强−迁移={bd-ad:+.6f}；迁移−epoch0={ad-base:+.6f}。'])
    if a['best_epoch']==b['best_epoch']==0:next_step='下一步只检查小样本适配与过拟合：两者微调未超过epoch0，不自动解冻LoRA。'
    elif bd>ad:next_step='下一步优先做去掉新增dense局部几何分支、保留其余上下文的匹配消融，区分整个增强模块的收益来源。'
    else:next_step='下一步保留预训练定位基础，重新判断上下文接入方式；若两者仍弱，先核查监督目标/编辑类型/覆盖，不继续堆模块。'
    if not fixed:lines.append(next_step)
    lines.append('差值仅作单seed描述，不作显著性判断；整个模块包含几何与时序，不能归因于纯XYZ或component单独贡献。')
    for c,s in summaries.items():
        lines.append(f'{c}：实际{s["last_epoch"]}轮/{s["updates"]}更新，停止{s["stop_reason"]}；参数/耗时/显存：'+json.dumps(s['costs'],ensure_ascii=False))
        rr=[r for r in s['best']['dev']['records'] if r['role']=='fake'];cm=np.sum([r['pixel_confusion_matrix'] for r in rr],axis=0)
        tn,fp,fn,tp=cm.reshape(-1)
        lines.append(f'best开发fake全帧pooled像素GT召回={tp/max(1,tp+fn):.6f}，背景FP={fp/max(1,fp+tn):.6f}（与主表视频均值口径不同）。')
        lines.append('固定开发fake边界诊断（原生2px，仅固定时刻，GT未修改）：'+json.dumps(boundary[c],ensure_ascii=False))
        for generator in sorted({r['generator'] for r in rr}):
            values=[r['spatial_dice'] for r in rr if r['generator']==generator and r['spatial_dice'] is not None]
            lines.append(f'开发编辑类型{generator}：{len(values)}视频，Dice均值={np.mean(values):.6f}。')
        for r in sorted(rr,key=lambda r:r['spatial_dice'])[:3]:
            lines.append(f'开发最低Dice：{r["pair_id"]}/{r["video_id"]}，{r["generator"]}，Dice={r["spatial_dice"]:.6f}。')
    refs=json.loads((out/'source_references.json').read_text());historical=Path(refs['source_run'])/'COMPONENT_LOC'/'summary.json'
    if historical.exists():
        old=json.loads(historical.read_text())['best_metrics']['datasets']['ANAKIN']
        lines.append(f'历史随机COMPONENT_LOC最佳dev Dice={old["spatial_dice"]:.6f}、AP={old["pixel_ap"]:.6f}；本轮epoch0 Dice={base:.6f}、AP={epoch0["dev"]["metrics"]["pixel_ap"]:.6f}。仅描述同开发队列的位置选择性；训练库、监督和模型不同，不是单独预训练的匹配消融。')
    for filename in ['asset_download.json','feature_costs.json']:
        p=out/filename
        if p.exists():lines.append(filename+'：'+json.dumps(json.loads(p.read_text()),ensure_ascii=False))
    storage=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
    contents='checkpoint、固定样例及日志；已有视觉特征与前端缓存只读引用，无新增特征' if fixed else '新增视觉特征、checkpoint、固定样例及日志；不含仓库外官方权重/原130GiB缓存'
    lines.append(f'报告生成前独立输出目录总存储={storage}字节（{contents}）。')
    lines.extend(['固定时刻全部开发与原monitor训练对照见index.html；全部视频指标见summary，pixel AP以正帧均值后视频均值统计。',
        'GT覆盖/背景FP依据完整帧计数；边界仅依据固定样例原生2px带，不外推为完整队列边界精度。',
        '下载、视觉特征提取与训练成本分别见asset_download.json、feature_costs.json、各条件summary；原缓存只读。',
        '本轮两个条件结束，不自动追加数据集、seed、模型、LoRA微调或参数搜索。'])
    save(out/'result_summary.json',{'epoch0':{s:epoch0[s]['metrics'] for s in ['train','dev']},'conditions':summaries})
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')
    if fixed:fixed_bn_comparison(out,summaries)


def fixed_bn_comparison(out,current):
    """Compare saved original results/history only; never execute the old models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    refs=json.loads((out/'reference_reuse.json').read_text());reference=Path(refs['reference_run'])
    original={c:json.loads((reference/c/'summary.json').read_text()) for c in CONDITIONS}
    fmt=lambda v:f'{v:.6f}'
    lines=['','## 原BN策略与FIXED_BN主比较','',
        '各自按同一规则选择best，同一行来自同一checkpoint；原策略仅引用已有评价。','',
        '|BN策略|模型|best/结束轮次|训练Dice|开发Dice|IoU|pixel AP|real空间FP|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for label,s in [('原策略',original[c]),('FIXED_BN',current[c])]:
            m=s['best']['dev']['metrics'];values=[s['best']['train']['metrics']['spatial_dice']]+[m[k] for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr']]
            lines.append('|'+ '|'.join([label,c,f'{s["best_epoch"]}/{s["last_epoch"]}']+[fmt(v) for v in values])+'|')
    deltas={c:{k:current[c]['best']['dev']['metrics'][k]-original[c]['best']['dev']['metrics'][k]
        for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr']} for c in CONDITIONS}
    extra={k:current[CONDITIONS[1]]['best']['dev']['metrics'][k]-current[CONDITIONS[0]]['best']['dev']['metrics'][k]
        for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr']}
    for c in CONDITIONS:
        lines.append(c+' 固定BN−原策略：'+json.dumps(deltas[c],ensure_ascii=False))
        old,new=original[c]['best'],current[c]['best']
        train_delta=new['train']['metrics']['spatial_dice']-old['train']['metrics']['spatial_dice']
        gap=new['train']['metrics']['spatial_dice']-new['dev']['metrics']['spatial_dice']
        lines.append(f'{c}：训练Dice变化={train_delta:+.6f}，开发Dice变化={deltas[c]["spatial_dice"]:+.6f}，固定BN best训练−开发差={gap:+.6f}；是否开发改善据开发重叠指标判断，不以训练提升或real误报下降代替。')
    lines.append('FIXED_BN内增强−迁移：'+json.dumps(extra,ensure_ascii=False))
    maps={(tag,c):{r['video_id']:r for r in ss[c]['best']['dev']['records'] if r['role']=='fake'}
        for tag,ss in [('original',original),('fixed',current)] for c in CONDITIONS}
    keys=set(maps['original',CONDITIONS[0]])
    if len(keys)!=8 or any(set(m)!=keys for m in maps.values()):raise ValueError('MATCHED_DEV_FAKE_IDENTITIES')
    paired=[]
    for vid,r in maps['original',CONDITIONS[0]].items():
        a,b=[maps['original',c][vid]['spatial_dice'] for c in CONDITIONS]
        x,y=[maps['fixed',c][vid]['spatial_dice'] for c in CONDITIONS]
        paired.append({'video_id':vid,'pair_id':r['pair_id'],'generator':r['generator'],'original_transfer':a,'original_context':b,
            'fixed_transfer':x,'fixed_context':y,'delta_bn_transfer':x-a,'delta_bn_context':y-b,'delta_fixed_context':y-x})
    lines.extend(['','|开发fake|类型|原迁移|固定迁移|BN差|原增强|固定增强|BN差|固定增强−迁移|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in paired:
        lines.append('|'+ '|'.join([r['pair_id'],r['generator']]+[fmt(r[k]) for k in ['original_transfer','fixed_transfer','delta_bn_transfer',
            'original_context','fixed_context','delta_bn_context','delta_fixed_context']])+'|')
    groups={}
    for label,rr in [('全部8条',paired),('旧弱6条',[r for r in paired if r['pair_id'] not in {'an:4rvro9','an:4wgx37'}])]:
        groups[label]={}
        for k in ['delta_bn_transfer','delta_bn_context','delta_fixed_context']:
            d=[r[k] for r in rr];groups[label][k]={'mean':float(np.mean(d)),'up':sum(x>0 for x in d),'tie':sum(x==0 for x in d),'down':sum(x<0 for x in d)}
        lines.append(label+'：'+json.dumps(groups[label],ensure_ascii=False))
    lines.append('旧弱6条预先固定为除4rvro9/4wgx37外的原开发fake；仅作描述，不更换样本或主指标。')
    fig,axes=plt.subplots(2,3,figsize=(15,8));fluctuation={}
    for ri,c in enumerate(CONDITIONS):
        oldh=json.loads((reference/c/'history.json').read_text());newh=json.loads((out/c/'history.json').read_text());common=min(len(oldh),len(newh))
        fluctuation[c]={'common_epochs':common}
        for label,h in [('原策略',oldh),('FIXED_BN',newh)]:
            values=[r['dev']['spatial_dice'] for r in h[:common]]
            fluctuation[c][label]=float(np.abs(np.diff(values)).mean()) if common>1 else None
        for ci,key in enumerate(['spatial_dice','spatial_iou','real_spatial_fpr']):
            ax=axes[ri,ci]
            for label,h in [('ORIGINAL_BN',oldh),('FIXED_BN',newh)]:
                ax.plot([r['epoch'] for r in h],[r['dev'][key] for r in h],label=label+' dev')
                if ci==0:ax.plot([r['epoch'] for r in h],[r['train'][key] for r in h],label=label+' train',linestyle=':')
            ax.set_title(c+' '+key);ax.set_xlabel('epoch');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'bn_strategy_curves.png');plt.close(fig)
    lines.append('同轮曲线见bn_strategy_curves.png；共同epoch范围内dev Dice相邻轮绝对差均值（描述性波动，受学习趋势影响）：'+json.dumps(fluctuation,ensure_ascii=False))
    old_boundary=json.loads((reference/'fixed_boundary_statistics.json').read_text())['aggregates']
    new_boundary=json.loads((out/'fixed_boundary_statistics.json').read_text())['aggregates']
    coverage={}
    for c in CONDITIONS:
        coverage[c]={}
        for label,ss in [('original',original),('fixed',current)]:
            cm=np.sum([r['pixel_confusion_matrix'] for r in ss[c]['best']['dev']['records'] if r['role']=='fake'],axis=0)
            tn,fp,fn,tp=cm.reshape(-1);coverage[c][label]={'GT_recall':float(tp/max(1,tp+fn)),'BG_FPR':float(fp/max(1,fp+tn))}
        lines.append(c+' 完整dev fake pooled覆盖/背景：'+json.dumps(coverage[c],ensure_ascii=False))
        lines.append(c+' 固定样例边界（仅原生2px带，非全队列边界成绩）：'+json.dumps({'original':old_boundary[c],'fixed':new_boundary[c]},ensure_ascii=False))
    if deltas[CONDITIONS[0]]['spatial_dice']>0 and extra['spatial_dice']<=0:
        decision='唯一下一步：停止继续修补当前融合方式，保留稳定迁移基线，将后续成本转向空间样本覆盖和视觉适配。'
    elif extra['spatial_dice']>0:
        decision='唯一下一步：保留固定BN基础，设计分离dense几何与结构时序贡献的最小匹配消融。'
    else:
        decision='唯一下一步：如实保留固定BN未改善的结果，转向空间样本覆盖与视觉适配，不继续搜索BN策略。'
    lines.extend([decision,'该建议仅据本轮单seed描述性差异，不证明显著性；训练与开发差距、GT覆盖、real FP和边界须联合审阅，不能以背景FP降低代替定位改善。',
        'epoch0复用来源与固定样本一致性见epoch0_reuse.json；无新增特征提取或大缓存复制。',
        '原结果与缓存未覆盖的元信息核对见reference_reuse.json。本轮两个条件结束，不自动执行后续建议。'])
    comparison={'deltas_fixed_minus_original':deltas,'context_minus_transfer_fixed':extra,'paired_dev_fake':paired,
        'groups':groups,'fluctuation_common_epochs':fluctuation,'coverage':coverage,'boundary':{'original':old_boundary,'fixed':new_boundary},'recommendation':decision}
    save(out/'bn_strategy_comparison.json',comparison)
    with (out/'report_zh.md').open('a') as f:f.write('\n'.join(lines)+'\n')
