"""Frozen same-source temporal readout diagnosis (ADR-0046).

Caller: python -m research_tools.xyz_pilot.temporal_readout_diagnosis.
Output: CSV/count curves/fixed preview, no model changes or optimization. Native
logits are released per target; B reuses only its own fixed-weight C0 history.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from .data import REPO, read
from .insource_probe import OUT as SOURCE, PartitionCache, partition_history
from .insource_model import XYZLocalization
from .joint_experiment import save,digest
from .experiment import csvsave
from research_tools.rgbd_slice.experiment import state_hash
from research_tools.v8.spatial_localization.metrics import spatial_frame
from scripts.review_anakin import numeric_members,mask_array,decode

OUT=SOURCE.parent/'pvst_xyz_temporal_readout_diagnosis_v1'
CONDITIONS=['A_CURRENT','B_FULL','B_CURRENT']
PROBABILITIES=np.arange(1,100,dtype=np.float64)/100
LOGIT_THRESHOLDS=np.log(PROBABILITIES)-np.log1p(-PROBABILITIES)
QUANTILES=[.05,.25,.5,.75,.95]
METRICS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area']
GROUPS=['ALL','H0','H1_6','H7']


def history_info(row,i):
    slots=partition_history(row,i,'evaluation');past=[j for j in slots[:-1] if j is not None]
    h=len(past)
    return dict(H=h,unique_history=len(set(past)),history_group='H0' if h==0 else 'H7' if h==7 else 'H1_6')


def read_frames(path,rows):
    byid={r['video_id']:r for r in rows};result=[]
    with path.open() as f:
        for raw in csv.DictReader(f):
            item={}
            for key,value in raw.items():
                if value=='':item[key]=None
                elif key in ['frame_index','native_pixels','GT_pixels']:item[key]=int(value)
                elif key in ['pts',*METRICS,'constant_AP']:item[key]=float(value)
                else:item[key]=value
            item.update(history_info(byid[item['video_id']],item['frame_index']));result.append(item)
    return result


def average(rows,key):
    values=[x[key] for x in rows if x.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate(records):
    """Per-frame -> video -> fake-source macro, with paired real FPR separate."""
    videos=[];sources=[]
    for vid in dict.fromkeys(x['video_id'] for x in records):
        selected=[x for x in records if x['video_id']==vid];first=selected[0]
        values={k:average(selected,k) for k in METRICS}
        videos.append(dict(video_id=vid,pair_id=first['pair_id'],role=first['role'],frames=len(selected),
                           mean_unique_history=average(selected,'unique_history'),**values))
    for fake in [v for v in videos if v['role']=='fake']:
        real=next((v for v in videos if v['pair_id']==fake['pair_id'] and v['role']=='real'),None)
        sources.append(dict(pair_id=fake['pair_id'],fake_frames=fake['frames'],real_frames=real['frames'] if real else 0,
            **{k:fake[k] for k in METRICS if k!='real_FPR'},real_FPR=real['real_FPR'] if real else None))
    macro={k:average(sources,k) for k in METRICS}
    return dict(macro=macro,sources=sources,videos=videos,frames=len(records),source_count=len(sources))


def grouped(records):
    return {g:aggregate(records if g=='ALL' else [x for x in records if x['history_group']==g]) for g in GROUPS}


def region_statistics(sorted_values):
    n=len(sorted_values)
    if not n:return dict(count=0,**{f'logit_p{int(q*100):02d}':None for q in QUANTILES})
    loc=(n-1)*np.asarray(QUANTILES);lo=np.floor(loc).astype(int);hi=np.ceil(loc).astype(int);a=loc-lo
    values=sorted_values[lo].astype(np.float64)*(1-a)+sorted_values[hi].astype(np.float64)*a
    return dict(count=n,**{f'logit_p{int(q*100):02d}':float(v) for q,v in zip(QUANTILES,values)})


def counts_and_regions(logit,gt):
    # Copies are deliberate: sorting must never reorder a preview's pixels.
    pos=np.sort(logit[gt].copy());neg=np.sort(logit[~gt].copy())
    tp=len(pos)-np.searchsorted(pos,LOGIT_THRESHOLDS,side='left')
    fp=len(neg)-np.searchsorted(neg,LOGIT_THRESHOLDS,side='left')
    counts=np.stack((tp,fp,len(pos)-tp,len(neg)-fp),axis=1).astype(np.int64)
    return counts,region_statistics(pos),region_statistics(neg)


def curve_values(counts):
    tp,fp,fn,tn=counts.T.astype(np.float64);gt=tp+fn;negative=fp+tn;pred=tp+fp
    precision=np.divide(tp,pred,out=np.zeros_like(tp),where=pred>0)
    recall=np.divide(tp,gt,out=np.zeros_like(tp),where=gt>0)
    dice=np.divide(2*tp,gt+pred,out=np.zeros_like(tp),where=(gt+pred)>0)
    bg=np.divide(fp,negative,out=np.zeros_like(tp),where=negative>0)
    return dict(precision=precision,GT_recall=recall,Dice=dice,FPR=bg,prediction_area=pred/(gt+negative))


def summarize_curves(records,counts):
    byvideo=[]
    for vid in dict.fromkeys(x['video_id'] for x in records):
        ids=[i for i,r in enumerate(records) if r['video_id']==vid];first=records[ids[0]]
        selected=[curve_values(counts[i]) for i in ids]
        byvideo.append(dict(video_id=vid,pair_id=first['pair_id'],role=first['role'],frames=len(ids),
            **{key:np.mean([v[key] for v in selected],axis=0) for key in selected[0]}))
    sources=[]
    for fake in [x for x in byvideo if x['role']=='fake']:
        real=next(x for x in byvideo if x['pair_id']==fake['pair_id'] and x['role']=='real')
        sources.append(dict(pair_id=fake['pair_id'],fake_frames=fake['frames'],real_frames=real['frames'],
            precision=fake['precision'],GT_recall=fake['GT_recall'],Dice=fake['Dice'],prediction_area=fake['prediction_area'],
            fake_background_FPR=fake['FPR'],real_FPR=real['FPR']))
    keys=['precision','GT_recall','Dice','prediction_area','fake_background_FPR','real_FPR']
    return dict(macro={k:np.mean([x[k] for x in sources],axis=0) for k in keys},sources=sources)


def store_preview(condition,row,i,native,gt):
    if i not in row['preview_targets']:return
    h,w=native.shape;factor=min(1,480/w);size=(round(w*factor),round(h*factor))
    prob=1/(1+np.exp(-np.clip(native,-40,40)))
    folder=OUT/'preview'/'scores';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{condition}_{row["video_id"]}_{i:05d}.npz',
        probability=cv2.resize(prob,size,interpolation=cv2.INTER_LINEAR),
        binary=cv2.resize((native>=0).astype(np.uint8),size,interpolation=cv2.INTER_NEAREST))


def measure(condition,row,i,native,gt):
    assert native.dtype==np.float32 and np.isfinite(native).all()
    overlap=spatial_frame(native>=0,gt);tp=int(((native>=0)&gt).sum());neg=gt.size-int(gt.sum());fp=int(((native>=0)&~gt).sum())
    record=dict(condition=condition,video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],frame_index=i,
        pts=row['pts'][i],native_pixels=gt.size,GT_pixels=int(gt.sum()),**history_info(row,i),
        pixel_AP=float(average_precision_score(gt.ravel(),native.ravel())) if gt.any() else None,
        Dice=overlap['dice'],IoU=overlap['iou'],GT_recall=tp/int(gt.sum()) if gt.any() else None,
        real_FPR=fp/neg if row['role']=='real' else None,
        fake_background_FPR=fp/neg if row['role']=='fake' and gt.any() and neg else None,
        prediction_area=float((native>=0).mean()),GT_area=float(gt.mean()))
    counts,pos,bg=counts_and_regions(native,gt)
    assert counts[49].tolist()==[tp,fp,int(gt.sum())-tp,neg-fp]
    regions=[]
    for name,stats in ([('fake_GT',pos),('fake_background',bg)] if row['role']=='fake' else [('real_all',bg)]):
        regions.append(dict(condition=condition,video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],
            frame_index=i,pts=row['pts'][i],region=name,**stats,
            region_positive_fraction=(tp/pos['count'] if name=='fake_GT' and pos['count'] else fp/bg['count'] if name!='fake_GT' and bg['count'] else None),
            prediction_area=record['prediction_area']))
    store_preview(condition,row,i,native,gt)
    return record,counts,regions


def replay_check(records,old):
    assert [(r['video_id'],r['frame_index'],r['pts'],r['native_pixels'],r['GT_pixels']) for r in records]==[(r['video_id'],r['frame_index'],r['pts'],r['native_pixels'],r['GT_pixels']) for r in old]
    diffs={k:max(abs(a[k]-b[k]) for a,b in zip(records,old) if a.get(k) is not None and b.get(k) is not None) for k in METRICS if k!='prediction_area'}
    new,ref=aggregate(records),aggregate(old)
    macro={k:new['macro'][k]-ref['macro'][k] for k in diffs}
    source={a['pair_id']:{k:a[k]-b[k] for k in diffs} for a,b in zip(new['sources'],ref['sources'])}
    maximum=max(abs(x) for v in source.values() for x in v.values())
    # Predeclared replay guard, not a scientific significance threshold.
    assert maximum<=.001,'REPLAY_MISMATCH: inspect evaluation path before attribution'
    return dict(macro_delta=macro,source_delta=source,max_abs_frame_delta=diffs,source_macro_guard=.001,passed=True)


@torch.no_grad()
def infer(model,rows,cache,is_b):
    labels=['B_FULL','B_CURRENT'] if is_b else ['A_CURRENT']
    records={n:[] for n in labels};counts={n:[] for n in labels};regions=[];h0=[]
    for row in rows:
        ring={};z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        members=numeric_members(z.namelist()) if z else None
        for i in row['evaluation_targets']:
            f=cache.get(row,i,'evaluation')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                current=model.encode(f)
                if is_b:
                    ring[i]={k:current[k].cpu() for k in ['C','position']}
                    ids=[j for j in partition_history(row,i,'evaluation') if j is not None]
                    contexts=[{k:v.cuda() for k,v in ring[j].items()} for j in ids]
                    full=model.relate_window(contexts,[row['pts'][j]-row['pts'][i] for j in ids],ids,True)
                    now=model.relate_window([current],[0],[i],False)
                    outputs=[model.decode(f,current,full),model.decode(f,current,now)]
                else:
                    now=model.relate_window([current],[0],[i],False);outputs=[model.decode(f,current,now)]
            if is_b and history_info(row,i)['H']==0:
                delta=float((outputs[0].float()-outputs[1].float()).abs().max());h0.append(dict(video_id=row['video_id'],frame_index=i,max_abs_work_logit_difference=delta))
                assert delta==0,'H0_INTERVENTION_INCONSISTENT'
            gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool)
            assert list(gt.shape)==row['native_hw'] and int(gt.sum())==f['native_GT_pixels']
            for name,pixel in zip(labels,outputs):
                native=F.interpolate(pixel.float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                r,c,q=measure(name,row,i,native,gt);records[name].append(r);counts[name].append(c);regions.extend(q)
            # Enough old C0 for the exact earliest slot, without retaining maps.
            earliest=[j for j in partition_history(row,i,'evaluation') if j is not None]
            for j in list(ring):
                if j<min(earliest):del ring[j]
            del outputs,current,f,now
            if is_b:del full,contexts
        if z:z.close()
        print('EVALUATED',labels,row['video_id'],len(row['evaluation_targets']),flush=True)
    return records,{k:np.stack(v) for k,v in counts.items()},regions,h0


def region_sources(regions):
    output=[]
    for condition in CONDITIONS:
        for vid in dict.fromkeys(x['video_id'] for x in regions if x['condition']==condition):
            for region in ['fake_GT','fake_background','real_all']:
                selected=[x for x in regions if x['condition']==condition and x['video_id']==vid and x['region']==region]
                if not selected:continue
                output.append(dict(condition=condition,video_id=vid,pair_id=selected[0]['pair_id'],region=region,frames=len(selected),
                    total_pixels=sum(x['count'] for x in selected),**{k:average(selected,k) for k in [*[f'logit_p{int(q*100):02d}' for q in QUANTILES],'region_positive_fraction','prediction_area']}))
    return output


def curves_and_preview(rows,records,counts):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    macro_csv=[];source_csv=[];curves={}
    for condition in CONDITIONS:
        curves[condition]={}
        for group in GROUPS:
            ids=[i for i,x in enumerate(records[condition]) if group=='ALL' or x['history_group']==group]
            result=summarize_curves([records[condition][i] for i in ids],counts[condition][ids]);curves[condition][group]=result
            for j,prob in enumerate(PROBABILITIES):
                macro_csv.append(dict(condition=condition,history_group=group,probability=float(prob),**{k:float(v[j]) for k,v in result['macro'].items()}))
                for row in result['sources']:
                    source_csv.append(dict(condition=condition,history_group=group,pair_id=row['pair_id'],probability=float(prob),**{k:float(row[k][j]) for k in result['macro']}))
    csvsave(OUT/'threshold_macro.csv',macro_csv);csvsave(OUT/'threshold_sources.csv',source_csv)
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for condition in CONDITIONS:
        c=curves[condition]['ALL']['macro'];axes[0,0].plot(c['GT_recall'],c['precision'],label=condition)
        axes[0,0].scatter([c['GT_recall'][49]],[c['precision'][49]],s=14)
        axes[0,1].plot(PROBABILITIES,c['Dice'],label=condition)
        axes[1,0].plot(PROBABILITIES,c['real_FPR'],label=condition)
        axes[1,1].plot(PROBABILITIES,c['fake_background_FPR'],label=condition)
    for ax,title in zip(axes.ravel(),['Macro precision / recall','Macro Dice / probability threshold','Real pixel FPR / threshold','Fake background FPR / threshold']):ax.set_title(title);ax.legend(fontsize=7);ax.grid(alpha=.2)
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.axvline(.5,color='gray',linestyle='--',linewidth=.7)
    fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for pair,ax in zip(['an:9l8w54','an:9mygy9'],axes):
        for condition in CONDITIONS:
            row=next(r for r in curves[condition]['ALL']['sources'] if r['pair_id']==pair);ax.plot(PROBABILITIES,row['Dice'],label=condition)
        ax.set_title(pair+' Dice / threshold');ax.set_xlabel('Probability threshold');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'focus_curves.png',dpi=160);plt.close(fig)
    folder=OUT/'preview';images=folder/'images';images.mkdir(exist_ok=True)
    html=['<!doctype html><meta charset="utf-8"><h1>固定42目标：时间读出对照</h1><p>概率统一0–1灰度；二值均为0.5；概率差统一[-1,1]蓝白红。仅预览缩放，不按效果挑图。</p>']
    for row in sorted(rows,key=lambda x:x['video_id']):
        _,rgb=decode(Path(row['video_path']),set(row['preview_targets']));z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None;members=numeric_members(z.namelist()) if z else None
        for i in row['preview_targets']:
            key=f'{row["video_id"]}_{i:05d}';values=[np.load(folder/'scores'/f'{n}_{key}.npz') for n in CONDITIONS]
            h,w=values[0]['probability'].shape;gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool)
            Image.fromarray(np.asarray(rgb[i])).resize((w,h)).save(images/f'{key}_rgb.png');Image.fromarray((gt*255).astype(np.uint8)).resize((w,h),resample=Image.Resampling.NEAREST).save(images/f'{key}_gt.png')
            info=history_info(row,i);html.append(f'<h2>{row["pair_id"]}/{row["role"]} 帧{i}, PTS {row["pts"][i]:.6f}, H={info["H"]}, 唯一历史={info["unique_history"]}</h2><img width="32%" src="images/{key}_rgb.png"><img width="32%" src="images/{key}_gt.png">')
            for label,array in [('probability',None),('binary',None)]:
                html.append('<p>'+('概率：' if label=='probability' else '0.5二值：')+' / '.join(CONDITIONS)+'</p>')
                for condition,value in zip(CONDITIONS,values):
                    path=f'{key}_{condition}_{label}.png';Image.fromarray((value[label]*255).astype(np.uint8)).save(images/path);html.append(f'<img width="32%" src="images/{path}">')
            for name,x,y in [('B_FULL_minus_B_CURRENT',1,2),('B_FULL_minus_A_CURRENT',1,0)]:
                diff=values[x]['probability']-values[y]['probability'];rgba=plt.get_cmap('coolwarm')((diff+1)/2)
                path=f'{key}_{name}.png';Image.fromarray((rgba[:,:,:3]*255).astype(np.uint8)).save(images/path);html.append(f'<p>{name}，概率差[-1,1]</p><img width="48%" src="images/{path}">')
            for x in values:x.close()
        if z:z.close()
    (folder/'index.html').write_text('\n'.join(html))
    diagnostic={}
    for condition in CONDITIONS:
        diagnostic[condition]={}
        for row in curves[condition]['ALL']['sources']:
            at=int(np.argmax(row['Dice']))
            diagnostic[condition][row['pair_id']]=dict(diagnostic_max_Dice=float(row['Dice'][at]),label='Evaluation-label diagnostic only; not selected operating point')
    return diagnostic


def run():
    started=time.monotonic();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    c=read(SOURCE/'config.json');protocol=read(SOURCE/'protocol.json');rows=protocol['rows'];source_identity=read(SOURCE/'run_identity.json')
    assert digest(SOURCE/'config.json')==source_identity['config_sha256'] and digest(SOURCE/'protocol.json')==source_identity['protocol_sha256']
    config=dict(probability_thresholds=PROBABILITIES.tolist(),logit_thresholds=LOGIT_THRESHOLDS.tolist(),quantiles=QUANTILES,preview_targets={r['video_id']:r['preview_targets'] for r in rows},history_groups=GROUPS,source_config=c,source_protocol_sha256=digest(SOURCE/'protocol.json'),replay_source_macro_atol=.001,source_quantile_aggregation='mean of corresponding per-frame quantile; not pooled pixels')
    save(OUT/'config.json',config)
    old={n:read_frames(SOURCE/f'{name}_epoch10_evaluation_frames.csv',rows) for n,name in [('A_CURRENT','XYZ_CURRENT'),('B_FULL','XYZ_TEMPORAL')]}
    cpu={n:grouped(v) for n,v in old.items()};save(OUT/'old_history_summary.json',cpu)
    csvsave(OUT/'old_history_sources.csv',[dict(condition=n,history_group=g,**v) for n,data in cpu.items() for g,result in data.items() for v in result['sources']])
    csvsave(OUT/'history_targets.csv',[dict(video_id=r['video_id'],frame_index=i,pts=r['pts'][i],**history_info(r,i)) for r in rows for i in r['evaluation_targets']])
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_identity=source_identity,source_config_sha256=digest(SOURCE/'config.json'),source_protocol_sha256=digest(SOURCE/'protocol.json'),diagnostic_source_sha256=digest(Path(__file__)),weights={})
    for label,name in [('A_CURRENT','XYZ_CURRENT'),('B_FULL','XYZ_TEMPORAL')]:
        path=SOURCE/f'{name}_epoch10.pt';identity['weights'][label]=dict(path=str(path),sha256=digest(path))
    identity['weights']['B_CURRENT']=identity['weights']['B_FULL'].copy();save(OUT/'run_identity.json',identity)
    all_records={};all_counts={};regions=[];replay={};cost={};h0=[];cache=PartitionCache(c)
    for is_b,name in [(False,'XYZ_CURRENT'),(True,'XYZ_TEMPORAL')]:
        checkpoint=torch.load(SOURCE/f'{name}_epoch10.pt',weights_only=False,map_location='cpu')
        assert checkpoint['condition']==name and checkpoint['updates']==10260 and checkpoint['identity']==source_identity
        model=XYZLocalization(c['h']);model.load_state_dict(checkpoint['state_dict'],strict=True);model=model.cuda().eval().requires_grad_(False)
        before=state_hash(model.state_dict());torch.cuda.reset_peak_memory_stats();begin=time.monotonic();save(OUT/'status.json',dict(status='RUNNING',stage=name))
        records,counts,qs,empty=infer(model,rows,cache,is_b)
        assert state_hash(model.state_dict())==before;assert all(len(v)==444 for v in records.values())
        normal='B_FULL' if is_b else 'A_CURRENT';replay[normal]=replay_check(records[normal],old[normal]);save(OUT/'replay.json',replay)
        all_records.update(records);all_counts.update(counts);regions+=qs;h0+=empty
        for condition,rs in records.items():csvsave(OUT/f'{condition}_frames.csv',rs)
        cost[name]=dict(seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),state_unchanged=True)
        del model,checkpoint;torch.cuda.empty_cache()
    for name in CONDITIONS:
        assert [(r['video_id'],r['frame_index'],r['pts']) for r in all_records[name]]==[(r['video_id'],r['frame_index'],r['pts']) for r in all_records['A_CURRENT']]
    groups={n:grouped(v) for n,v in all_records.items()};save(OUT/'grouped_summary.json',groups)
    csvsave(OUT/'history_sources.csv',[dict(condition=n,history_group=g,**v) for n,data in groups.items() for g,result in data.items() for v in result['sources']])
    csvsave(OUT/'sources.csv',[dict(condition=n,**v) for n,data in groups.items() for v in data['ALL']['sources']]);csvsave(OUT/'regions_frames.csv',regions);csvsave(OUT/'regions_sources.csv',region_sources(regions))
    np.savez_compressed(OUT/'threshold_counts.npz',probabilities=PROBABILITIES,**all_counts)
    save(OUT/'H0_consistency.json',dict(targets=len(h0),max_work_logit_difference=max(x['max_abs_work_logit_difference'] for x in h0),records=h0))
    diagnostic=curves_and_preview(rows,all_records,all_counts)
    result=dict(status='COMPLETE',exit_code=0,identity=identity,replay=replay,groups=groups,old_history_summary=cpu,cost=cost,H0_targets=len(h0),H0_max_difference=max(x['max_abs_work_logit_difference'] for x in h0),diagnostic_curve_bounds=diagnostic,wall_seconds=time.monotonic()-started,peak_host_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_geometry_frames=0,new_backbone_frames=0,training_updates=0)
    save(OUT/'summary.json',result);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


if __name__=='__main__':
    if (OUT/'run_identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':
            summary=read(OUT/'summary.json');assert all(digest(v['path'])==v['sha256'] for v in summary['identity']['weights'].values());print('EXISTING_COMPLETE_REUSED');raise SystemExit(0)
        raise SystemExit('Existing incomplete run preserved; no duplicate')
    try:run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error)));raise
