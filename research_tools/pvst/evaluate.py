"""Native standard metrics; old positive-GT frame/video aggregation retained."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .data import Video,save
from research_tools.v8.spatial_localization.metrics import classification,spatial_frame


@torch.no_grad()
def evaluate(model,rows,out,config,scale,*,pixel_ap=True,samples=None,progress=None):
    started=time.monotonic();model.eval();device=next(model.parameters()).device;records=[];voxels=[]
    for ri,r in enumerate(rows):
        print('EVALUATE_VIDEO',model.condition,r['dataset'],r['split'],r['video_id'],r['actual_frames'],flush=True)
        if progress:progress('evaluation',condition=model.condition,video_index=ri,video=r['video_id'],dataset=r['dataset'],split=r['split'])
        v=Video(out,r,model.condition,scale);scores=[];yy=[];dice=[];iou=[];ap=[];empty=[];recall=[];confusion=np.zeros((2,2),np.int64)
        times=np.asarray([p['pts_s'] for p in r['pts']]);selected={int(np.argmin(abs(times-(times[0]+x*(times[-1]-times[0]))))) for x in [.25,.5,.75]}
        for start in range(0,r['actual_frames'],config['target_frames']):
            if progress and start%80==0:progress('evaluation',condition=model.condition,video=r['video_id'],frames=start,total=r['actual_frames'],dataset=r['dataset'],split=r['split'])
            end=min(r['actual_frames'],start+config['target_frames']);lo=max(0,start-16)
            fs=v.frames(lo,end,device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=config['amp'] and device.type=='cuda'):pred=model(fs,start-lo)
            for j,p in enumerate(pred):
                i=start+j;z=p['pixel_logits'].float().cpu().numpy();gt=v.gt(i);score=float(p['frame_logit']);scores.append(score);yy.append(v.label(i))
                voxels.append({'video_id':r['video_id'],'frame':i,'fine':p['voxels_fine'],'coarse':p['voxels_coarse'],'valid_pixels':p['valid_pixels'],'native_pixels':r['native_pixels']})
                if gt is not None:
                    m=spatial_frame(z>=0,gt);confusion+=np.array(m['confusion_matrix'])
                    if gt.any():
                        dice.append(m['dice']);iou.append(m['iou']);recall.append(np.count_nonzero((z>=0)&gt)/np.count_nonzero(gt))
                        if pixel_ap:ap.append(float(average_precision_score(gt.reshape(-1),z.reshape(-1))))
                    else:empty.append(m['empty_fpr'])
                if samples and i in selected:
                    samples.mkdir(parents=True,exist_ok=True)
                    image=np.array(v.rgb[i]);gtimg=np.zeros_like(image) if gt is None else np.repeat((gt*255).astype(np.uint8)[...,None],3,-1)
                    probability=1/(1+np.exp(-np.clip(z,-60,60)));heat=cv2.applyColorMap((probability*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
                    binary=np.repeat(((z>=0)*255).astype(np.uint8)[...,None],3,-1)
                    preview=np.concatenate((image,gtimg,heat,binary),axis=1)
                    cv2.putText(preview,f'RGB | '+('GT missing' if gt is None else 'GT edit region')+f' | P 0..1 | P>=0.5 ; frame logit={score:.4f}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
                    cv2.imwrite(str(samples/f'{r["video_id"]}_{i}.jpg'),preview[:,:,::-1])
            del pred,fs
        v.close();z=float(np.logaddexp.reduce(np.asarray(scores)))-np.log(len(scores))
        mean=lambda a:float(np.mean(a)) if a else None
        records.append({**{k:r[k] for k in ['video_id','source_id','pair_id','dataset','split','role','label','generator']},
            'logit':z,'frames':len(scores),'frame_scores':scores,'frame_labels':yy,
            'spatial_dice':mean(dice),'spatial_iou':mean(iou),'pixel_ap':mean(ap),'GT_recall':mean(recall),
            'empty_spatial_fpr':mean(empty),'positive_gt_frames':len(dice),'empty_gt_frames':len(empty),'pixel_confusion_matrix':confusion.tolist() if gt is not None else None})
    if samples:
        samples.mkdir(parents=True,exist_ok=True)
        (samples/'index.html').write_text('<meta charset="utf-8"><title>PVST fixed samples</title><p>身份固定；25/50/75%。原图 | GT（AF缺失） | 概率0–1 | 概率≥0.5；帧分数标在图内。</p>'+''.join('<p>'+p.name+'</p><img style="max-width:100%" src="'+p.name+'">' for p in sorted(samples.glob('*.jpg'))))
    datasets={}
    for dataset in ['ActivityForensics','ANAKIN']:
        rr=[r for r in records if r['dataset']==dataset]
        if not rr:continue
        mean=lambda key,source:float(np.mean([r[key] for r in source if r[key] is not None])) if any(r[key] is not None for r in source) else None
        datasets[dataset]={'video':classification([r['label'] for r in rr],[r['logit'] for r in rr]),
            'frame':classification([y for r in rr for y in r['frame_labels']],[z for r in rr for z in r['frame_scores']]),
            **{k:mean(k,rr) for k in ['spatial_dice','spatial_iou','pixel_ap','GT_recall']},
            'real_spatial_fpr':mean('empty_spatial_fpr',[r for r in rr if r['role']=='real'])}
    return {'datasets':datasets,'records':records,'elapsed_s':time.monotonic()-started,'voxel_summary':{
        k:{'min':int(min(r[k] for r in voxels)),'median':float(np.median([r[k] for r in voxels])),'max':int(max(r[k] for r in voxels))} for k in ['fine','coarse','valid_pixels']},
        'spatial_aggregation':'positive GT frame mean per video then video mean; AP only positive-GT frames; empty/real FPR separate; all native pixels'}


def report(out,config,rows,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# PVST-Forensics v0.2 完整小样本探针','', '两条件单seed；AF dev视频AUROC唯一选best；epoch0/best/last同checkpoint全任务评价。',
        '实际数据：'+json.dumps({f'{d}/{s}':{'pairs':len([r for r in rows if r['dataset']==d and r['split']==s])//2,
          'videos':len([r for r in rows if r['dataset']==d and r['split']==s]),'real':sum(r['role']=='real' for r in rows if r['dataset']==d and r['split']==s),'fake':sum(r['role']=='fake' for r in rows if r['dataset']==d and r['split']==s),'frames':sum(r['actual_frames'] for r in rows if r['dataset']==d and r['split']==s),
          'resolutions_hw':sorted({tuple(r['source_hw']) for r in rows if r['dataset']==d and r['split']==s})} for d in ['ActivityForensics','ANAKIN'] for s in ['train','dev']},ensure_ascii=False),
        '固定张量：native N×9→N×128，细M1×256、粗M2×512；两尺度各三层d1/2/4；原生N×512融合分tile计算，帧描述符1024；全像素/全帧覆盖。',
        '', '|条件|checkpoint|epoch|集合|AF视频AUROC|AP|ACC|P|R|F1|混淆矩阵|AF帧AUROC|AP|F1|AN Dice|IoU|pixel AP|real FPR|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for summary in summaries:
        c=summary['condition'];folder=out/c
        for name in ['epoch0','best','last']:
            for split in ['train','dev']:
                result=json.loads((folder/f'{name}_{split}.json').read_text());af=result['datasets']['ActivityForensics'];an=result['datasets']['ANAKIN'];v=af['video'];f=af['frame']
                row=[c,name,str(result['epoch']),split,*[str(v[k]) for k in ['auroc','ap','acc','precision','recall','f1','confusion_matrix']],*[str(f[k]) for k in ['auroc','ap','f1']],*[str(an[k]) for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr']]]
                lines.append('|'+ '|'.join(row)+'|')
        history=json.loads((folder/'history.json').read_text());fig,axes=plt.subplots(1,3,figsize=(13,3));x=[r['epoch'] for r in history]
        axes[0].plot(x,[r['loss'] for r in history]);axes[0].set_title('Full-video normalized online loss')
        for split in ['monitor','dev']:
            axes[1].plot(x,[r[split]['ActivityForensics']['video']['auroc'] for r in history],label=split)
            axes[2].plot(x,[r[split]['ANAKIN']['spatial_dice'] for r in history],label=split)
        axes[1].set_title('AF video AUROC');axes[2].set_title('ANAKIN Dice');axes[1].legend();axes[2].legend();fig.tight_layout();fig.savefig(folder/'curves.png');plt.close(fig)
        lines+=['',c+'成本及更新：'+json.dumps(summary,ensure_ascii=False)]
        for split in ['train','dev']:
            value=json.loads((folder/f'best_{split}.json').read_text());lines.append(c+' best '+split+'全帧体素数量分布：'+json.dumps(value['voxel_summary'],ensure_ascii=False))
        scores={(n,sp):json.loads((folder/f'{n}_{sp}.json').read_text())['datasets'] for n in ['epoch0','best','last'] for sp in ['train','dev']}
        for dataset,task,key in [('ActivityForensics','video','auroc'),('ANAKIN',None,'spatial_dice')]:
            get=lambda n,sp:scores[n,sp][dataset][task][key] if task else scores[n,sp][dataset][key]
            lines.append(f'{dataset} {key}: dev epoch0={get("epoch0","dev"):.6f}, best={get("best","dev"):.6f}, last={get("last","dev"):.6f}; best train−dev={get("best","train")-get("best","dev"):+.6f}。')
        an=scores['best','dev']['ANAKIN'];lines.append('best ANAKIN GT召回与real像素FPR：'+str(an['GT_recall'])+' / '+str(an['real_spatial_fpr']))
    diff={}
    for split in ['train','dev']:
        a=json.loads((out/'UV_PVST'/f'best_{split}.json').read_text())['datasets'];b=json.loads((out/'XYZ_PVST'/f'best_{split}.json').read_text())['datasets']
        diff[split]={'AF_video_'+k:b['ActivityForensics']['video'][k]-a['ActivityForensics']['video'][k] for k in ['auroc','ap','f1']}
        diff[split].update({'AN_'+k:b['ANAKIN'][k]-a['ANAKIN'][k] for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr']})
    if (out/'execution_revision.json').exists():
        revision=json.loads((out/'execution_revision.json').read_text());archive=out/revision['previous_archive']
        lines+=['','## FFN执行修复与成本分账',
            'norm2与完整FFN按节点分块，ffn_chunk='+str(config['ffn_chunk'])+'；逐块non-reentrant checkpoint并保存RNG。模块参数/state_dict、全部输入覆盖与监督未变。改变dropout分块调用不保证旧执行逐位相同，不改变表达能力。',
            '旧失败执行及退出码1保留：'+json.dumps(revision,ensure_ascii=False),
            '原失败版本准备成本：'+(archive/'preparation_cost.json').read_text(),
            '本次复用缓存核查成本：'+(out/'preparation_cost.json').read_text(),
            '正确性测试：'+(out/'ffn_correctness.txt').read_text()]
        scan=json.loads((out/'resource_windows.json').read_text());lines.append('全缓存坐标计数耗时秒：'+str(scan['seconds']))
        for condition in config['conditions']:
            r=json.loads((out/f'resource_{condition}_{config["execution_revision"]}.json').read_text())
            lines.append(condition+'代表窗口验收：'+json.dumps(r,ensure_ascii=False))
        lines.append('完整pair验收成本与正式训练成本分别列于下文；所有验收权重丢弃，正式模型从原初始化开始。')
    lines+=['','前端成本：'+(out/'preparation_cost.json').read_text(),
        '输出存储（报告前）：逻辑'+str(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))+'字节；文件系统已分配'+str(sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file()))+'字节；未实测完整FLOPs。',
        '完整pair两次更新与路径梯度检查：'+(out/'full_pair_check.json').read_text(),
        '最大真实块与体素/邻居资源证据：'+str([p.name for p in out.glob('resource_*.json')]),
        '','3D−2D直接差值：'+json.dumps(diff,ensure_ascii=False),
        '结合曲线、完整训练/开发差距、best后变化、GT召回与real误报评价学习情况；降低背景误报不自动等于覆盖改善。',
        '单seed小样本仅用于可学习性与初步差异，不作最终总体性能或稳定多seed提升结论。MoGe密集输出经回映不是原生像素独立测量。',
        '本轮到此停止，不追加条件、seed、阈值或架构搜索。']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n');save(out/'result_summary.json',{'status':'COMPLETE','summaries':summaries,'differences':diff})
