"""Native-resolution semantic metrics and fixed-sample reports for this pilot."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_recall_fscore_support,accuracy_score

from .data import to_source
from .joint_data import Video,save
from .model import loss_sums


def classification(labels,scores):
    y=np.asarray(labels,int);z=np.asarray(scores,float)
    if not len(y):return {'reason':'no_labels'}
    if not np.isfinite(z).all():raise ValueError('NONFINITE_EVALUATION')
    pred=z>=0;p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    return {'auroc':float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None,
        'ap':float(average_precision_score(y,z)) if y.any() else None,
        'acc':float(accuracy_score(y,pred)),'precision':float(p),'recall':float(r),'f1':float(f),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'undefined_reason':None if len(np.unique(y))==2 else 'one_class; AUROC undefined, AP undefined without positives'}


def spatial_frame(prediction,gt):
    if gt is None:return {'iou':None,'dice':None,'empty_fpr':None,'confusion_matrix':None}
    tp=int(np.count_nonzero(prediction&gt));pred=int(np.count_nonzero(prediction));positive=int(np.count_nonzero(gt))
    fp,fn=pred-tp,positive-tp;tn=gt.size-tp-fp-fn
    return {'iou':tp/(pred+positive-tp) if positive else None,'dice':2*tp/(pred+positive) if positive else None,
        'empty_fpr':pred/gt.size if not positive else None,'confusion_matrix':[[tn,fp],[fn,tp]]}


@torch.no_grad()
def evaluate_video(model,video,chunk,amp=False,pixel_ap=False,sample_dir=None,save_logits=False):
    started=time.monotonic();r=video.row;device=next(model.parameters()).device
    previous=torch.zeros(256,128,device=device);active=torch.zeros(256,dtype=torch.bool,device=device)
    numerator=torch.tensor(-torch.inf,device=device,dtype=torch.float64);area=0.;fl=0.;pl=0.;known=0;pb=0.;pd=0.
    labels=[];scores=[];ious=[];dices=[];aps=[];empty_fps=[];samples={int(r['actual_frames']*f) for f in [.25,.5,.75]}
    pixel_cm=np.zeros((2,2),np.int64)
    if sample_dir:Path(sample_dir).mkdir(parents=True,exist_ok=True)
    for start in range(0,r['actual_frames'],chunk):
        a=video.chunk(start,min(start+chunk,r['actual_frames']))
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
            outputs=model.chunk(a,previous,active,start==0);previous,active,logits=outputs[:3]
        loss_inputs={k:a[k].to(device) for k in ['content512','gt','pixel_known','frame_label']}
        if len(outputs)==4:loss_inputs['area']=a['area'].to(device)
        n,ar,f,p,k,frame,b,d=loss_sums(logits,loss_inputs,outputs[3] if len(outputs)==4 else None,True)
        numerator=torch.logaddexp(numerator,n);area+=float(ar);fl+=float(f);pl+=float(p);known+=int(k)
        pb+=float(b);pd+=float(d)
        labels.extend(a['frame_label'].tolist());scores.extend(frame.cpu().tolist())
        for j,z in enumerate(logits.float().cpu().numpy()):
            fi=start+j;gt=video.native_gt(fi);native=to_source(z,r['mapping512']);prediction=native>=0
            if gt is not None:
                spatial=spatial_frame(prediction,gt);pixel_cm+=np.asarray(spatial['confusion_matrix'])
                if gt.any():
                    ious.append(spatial['iou']);dices.append(spatial['dice'])
                    if pixel_ap:aps.append(float(average_precision_score(gt.reshape(-1),native.reshape(-1))))
                else:empty_fps.append(spatial['empty_fpr'])
            if sample_dir and fi in samples:
                import cv2
                rgb=a['rgb512'][j].numpy();image=to_source_rgb(rgb,r['mapping512'])
                gt_rgb=np.zeros_like(image) if gt is None else np.repeat((gt*255).astype(np.uint8)[...,None],3,-1)
                pred_rgb=np.repeat((prediction*255).astype(np.uint8)[...,None],3,-1)
                preview=np.concatenate((image,gt_rgb,pred_rgb),axis=1)
                cv2.putText(preview,'RGB | '+('GT MISSING' if gt is None else 'GT edit region')+' | prediction logit>=0',
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,0),2)
                cv2.imwrite(str(Path(sample_dir)/f'{r["video_id"]}_{fi}.jpg'),preview[:,:,::-1])
                if save_logits and r['dataset']=='ANAKIN':
                    np.savez_compressed(Path(sample_dir)/f'{r["video_id"]}_{fi}.npz',
                        logits=z.astype(np.float32),gt512=a['gt'][j].numpy(),native_gt=gt,
                        content512=a['content512'][j].numpy(),rgb512=rgb,
                        pts_s=np.array(r['pts'][fi]['pts_s']),frame_index=np.array(fi),
                        metadata=np.array(json.dumps({'video_id':r['video_id'],'source_id':r['source_id'],
                            'split':r['split'],'role':r['role'],'mapping512':r['mapping512'],
                            'output_mode':getattr(model,'output_mode','COUPLED'),'spatial_target':'edit_region'})))
    z=float(numerator)-np.log(area);vl=float(np.logaddexp(0,z)-r['label']*z)
    result={k:r[k] for k in ['video_id','pair_id','source_id','dataset','split','label','role','generator']}
    result.update(logit=z,frame_labels=labels,frame_scores=scores,
        loss_video=vl,loss_frame=fl/r['actual_frames'],loss_pixel=pl/max(known,1),
        loss_pixel_bce=pb/max(known,1),loss_pixel_dice=pd/max(known,1),
        loss=vl+.5*fl/r['actual_frames']+pl/max(known,1),
        spatial_iou=float(np.mean(ious)) if ious else None,spatial_dice=float(np.mean(dices)) if dices else None,
        pixel_ap=float(np.mean(aps)) if aps else None,pixel_ap_reason=None if aps else 'not_scheduled_or_no_positive_GT',
        positive_gt_frames=len(dices),empty_gt_frames=len(empty_fps),
        pixel_confusion_matrix=pixel_cm.tolist() if dices or empty_fps else None,
        empty_spatial_fpr=float(np.mean(empty_fps)) if empty_fps else None,
        frames=r['actual_frames'],elapsed_s=time.monotonic()-started)
    if sample_dir:
        save(Path(sample_dir)/(r['video_id']+'_time.json'),{'pts':[p['pts_s'] for p in r['pts']],
            'frame_logits':scores,'frame_labels':labels,'sample_indices':sorted(samples),
            'columns':['original','GT edit_region (missing explicitly noted)','prediction logit>=0'],
            'spatial_status':r['spatial_status']})
    return result


def to_source_rgb(rgb,mapping):
    import cv2
    x,y=mapping['pad_xy'];h,w=mapping['content_hw'];sh,sw=mapping['source_hw']
    return cv2.resize(rgb[y:y+h,x:x+w],(sw,sh),interpolation=cv2.INTER_LINEAR)


def aggregate(records):
    result={}
    for dataset in ['ActivityForensics','ANAKIN']:
        rr=[r for r in records if r['dataset']==dataset]
        if not rr:continue
        mean=lambda k:float(np.mean([r[k] for r in rr if r.get(k) is not None])) if any(r.get(k) is not None for r in rr) else None
        real=[r for r in rr if r['role']=='real']
        result[dataset]={'video':classification([r['label'] for r in rr],[r['logit'] for r in rr]),
            'frame':classification([y for r in rr for y in r['frame_labels']],[z for r in rr for z in r['frame_scores']]),
            'real_frame_fpr':float(np.mean([np.mean(np.asarray(r['frame_scores'])>=0) for r in real])) if real else None,
            'real_spatial_fpr':float(np.mean([r['empty_spatial_fpr'] for r in real])) if real else None,
            'empty_mask_spatial_fpr':mean('empty_spatial_fpr'),
            'spatial_reason':None if mean('spatial_dice') is not None else 'no_positive_spatial_GT',
            'pixel_ap_reason':None if mean('pixel_ap') is not None else 'not_scheduled_or_no_positive_spatial_GT',
            'pixel_confusion_matrix':np.sum([r['pixel_confusion_matrix'] for r in rr if r.get('pixel_confusion_matrix') is not None],axis=0).tolist() if any(r.get('pixel_confusion_matrix') is not None for r in rr) else None,
            'pixel_confusion_scope':'pooled counts of known native GT pixels; does not replace positive-video mean Dice',
            **{k:mean(k) for k in ['loss','loss_video','loss_frame','loss_pixel','loss_pixel_bce','loss_pixel_dice','spatial_iou','spatial_dice','pixel_ap']}}
    result['dataset_macro']={k:(float(np.mean([v['video'][k] for v in result.values() if v['video'][k] is not None]))
                              if any(v['video'][k] is not None for v in result.values()) else None)
                              for k in ['auroc','ap','acc','precision','recall','f1']}
    for k in ['auroc','ap']:
        values=[v['frame'][k] for d,v in result.items() if d!='dataset_macro' and v['frame'][k] is not None]
        result['dataset_macro']['frame_'+k]=float(np.mean(values)) if values else None
    for k in ['loss','loss_video','loss_frame','loss_pixel']:
        values=[v[k] for d,v in result.items() if d!='dataset_macro' and v[k] is not None]
        result['dataset_macro'][k]=float(np.mean(values)) if values else None
    return result


def evaluate(model,out,rows,fit,chunk,amp=False,pixel_ap=False,sample_dir=None,*,cache_root=None,read_only=False,save_logits=False):
    started=time.monotonic();model.eval();records=[]
    for r in rows:
        v=Video(out,r,fit['statistics'],fit['grouping'],cache_root=cache_root,read_only=read_only)
        try:records.append(evaluate_video(model,v,chunk,amp,pixel_ap,sample_dir,save_logits))
        finally:v.close()
    return {'datasets':aggregate(records),'records':records,'elapsed_s':time.monotonic()-started,
        'pixel_ap_scheduled':pixel_ap,'spatial_aggregation':'positive GT frames mean within video, then video mean; empty backgrounds excluded from Dice/IoU'}


def history_csv(path,history):
    # One condition/epoch per row; scalar task metrics have readable CSV columns.
    if not history:return
    def flatten(row,prefix=''):
        result={}
        for k,v in row.items():
            key=prefix+k
            if isinstance(v,dict):result.update(flatten(v,key+'_'))
            else:result[key]=json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v
        return result
    rows=[flatten(r) for r in history];fields=list(rows[0]);tmp=Path(str(path)+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[]
    for condition in ['CELL_LOC','COMPONENT_LOC']:
        p=out/condition/'history.json'
        if not p.exists():continue
        history=json.loads(p.read_text());summary=json.loads((out/condition/'summary.json').read_text()) if (out/condition/'summary.json').exists() else {'status':'RUNNING'}
        summaries.append({'condition':condition,**summary})
        fig,axes=plt.subplots(2,3,figsize=(14,8));epochs=[r['epoch'] for r in history]
        for split in ['train_monitor','dev']:
            for dataset in ['ANAKIN','ActivityForensics']:
                label=split+'/'+dataset
                axes[0,0].plot(epochs,[r[split][dataset]['loss'] for r in history],label=label)
                axes[0,2].plot(epochs,[r[split][dataset]['video']['auroc'] for r in history],label=label)
                axes[1,0].plot(epochs,[r[split][dataset]['video']['ap'] for r in history],label=label)
                axes[1,1].plot(epochs,[r[split][dataset]['frame']['auroc'] for r in history],label=label)
                axes[1,2].plot(epochs,[r[split][dataset]['frame']['ap'] for r in history],label=label)
            axes[0,1].plot(epochs,[r[split]['ANAKIN']['spatial_dice'] for r in history],label=split)
        axes[0,0].plot(epochs,[r['online_train']['loss'] for r in history],label='online_train',linestyle=':')
        for ax,title in zip(axes.flat,['Video-normalized loss','ANAKIN positive-GT Dice','Video AUROC','Video AP','Frame AUROC','Frame AP']):ax.set_title(title);ax.set_xlabel('epoch');ax.legend(fontsize=6);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/condition/'curves.png');plt.close(fig)
        for path in (out/condition/'samples').glob('*_time.json'):
            data=json.loads(path.read_text());fig,ax=plt.subplots(figsize=(8,3))
            ax.plot(data['pts'],data['frame_logits'],label='frame logit');ax.step(data['pts'],data['frame_labels'],label='frame GT',where='post')
            ax.axhline(0,color='gray',linewidth=.5);ax.set_xlabel('native PTS seconds');ax.legend();fig.tight_layout();fig.savefig(path.with_suffix('.png'));plt.close(fig)
    save(out/'result_summary.json',summaries)
    lines=['# V8联合定位结果','', '两条件为完整256结构/512像素分支，只有成员分区不同。','']
    lines.extend(['|条件|checkpoint/epoch|数据集|video AUROC/AP|frame AUROC/AP|IoU/Dice|pixel AP|real空间FP|',
        '|---|---|---|---|---|---|---|---|'])
    fmt=lambda x:'未计算' if x is None else f'{x:.6f}'
    for s in summaries:
        if s.get('status')!='COMPLETE':continue
        for checkpoint in ['best','last']:
            for dataset,v in s[checkpoint+'_metrics']['datasets'].items():
                if dataset=='dataset_macro':continue
                lines.append(f'|{s["condition"]}|{checkpoint}/{s[checkpoint+"_epoch"]}|{dataset}|{fmt(v["video"]["auroc"])}/{fmt(v["video"]["ap"])}|{fmt(v["frame"]["auroc"])}/{fmt(v["frame"]["ap"])}|{fmt(v["spatial_iou"])}/{fmt(v["spatial_dice"])}|{fmt(v["pixel_ap"])}|{fmt(v["real_spatial_fpr"])}|')
    for r in summaries:
        lines.extend([f'## {r["condition"]}',f'状态：{r.get("status")}；best epoch：{r.get("best_epoch")}；停止：{r.get("stop_reason")}。',
            f'参数量：{r.get("parameters")}；训练/开发视频：{r.get("train_videos")}/{r.get("dev_videos")}；完整任务指标、混淆矩阵及训练开发差距见该条件summary.json。',''])
    if len(summaries)==2 and all(r.get('status')=='COMPLETE' for r in summaries):
        a,b=summaries
        da=a['best_metrics']['datasets']['ANAKIN']['spatial_dice'];db=b['best_metrics']['datasets']['ANAKIN']['spatial_dice']
        lines.append(f'COMPONENT_LOC减CELL_LOC的开发Dice：{db-da:+.6f}。单seed小队列结果；按各自best同epoch评价，不拼接任务最佳值。')
        for s in summaries:
            records=s['best_metrics']['records'];groups={}
            for r in records:
                key=r['dataset']+'/'+r['generator'];groups.setdefault(key,[]).append(r)
            lines.append(s['condition']+'按编辑类别的正确率/漏检和真实误报：')
            for key,rr in groups.items():
                fake=[r for r in rr if r['label']];real=[r for r in rr if not r['label']]
                lines.append(f'- {key}：fake检出{sum(r["logit"]>=0 for r in fake)}/{len(fake)}；real误报{sum(r["logit"]>=0 for r in real)}/{len(real)}。')
            gap=s['best_train_monitor']['datasets']['ANAKIN']['spatial_dice']-s['best_metrics']['datasets']['ANAKIN']['spatial_dice']
            lines.append(f'ANAKIN train_monitor减dev Dice={gap:+.6f}；'+('存在明显训练开发差距，应先检查过拟合。' if gap>.15 else '本轮未见超过预设0.15的明显Dice差距，不等于证明泛化。'))
        lines.append('前端耗时与缓存字节见costs.json；训练/评价实测见各轮history及summary。后续不自动扩大数据或追加模型。')
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')
