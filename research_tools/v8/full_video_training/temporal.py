"""One-seed temporal-supervision pilot, reusing the complete-video trainer."""
import argparse
import fcntl
import json
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from . import run as training
from .data import load_video
from .model import VideoTraining
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file, write_npz_atomic
from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.local_geometry_pilot.runner import csv_rows, seed_all
from research_tools.v8.local_geometry_pilot.data import digest

ROOT=Path('/root/autodl-tmp/data/sparse_3d_forgery_detection')
BASE=ROOT/'derived/v8_component_global_context_pilot_v1'
OLD=ROOT/'derived/v8_full_video_training_budget5_v1'
OUT=ROOT/'derived/v8_temporal_supervision_pilot_v1'
MAPPING=ROOT/'datasets/v7_core_candidates/activityforensics_charades_v1/paired/manifests/activityforensics_source_mapping.json'
SEED=20260909


def interval_labels(pts, intervals):
    y=np.zeros(len(pts),bool)
    for v in intervals:
        lo,hi=float(v['start_s']),float(v['end_s'])
        if not np.isfinite([lo,hi]).all() or lo<0 or hi<lo:raise ValueError('INVALID_ANNOTATION_INTERVAL')
        y|=(pts>=lo)&(pts<=hi)
    return y


def exact_annotation(row, mapping):
    if row['role']=='real':return None
    raw=ROOT/'datasets/v7_core_candidates/activityforensics_charades_v1/source/activityforensics/raw'
    matches=[m for m in mapping if m['charades_source_id']==row['source_id'] and str(raw/m['activityforensics_file'])==row['video_path']]
    if len(matches)!=1 or matches[0]['lineage_status']!='EXACT':raise ValueError(f'ANNOTATION_IDENTITY:{row["video_id"]}')
    if not matches[0]['manipulation_intervals']:raise ValueError('MISSING_FAKE_INTERVALS')
    return matches[0]


def frame_logits(cells, area):
    area=area.to(cells.device,dtype=torch.float64)
    weights=torch.where(area>0,area.clamp_min(1).log(),-torch.inf)
    if not torch.all(area.sum(1)>0):raise ValueError('FRAME_WITHOUT_CONTENT')
    return torch.logsumexp(cells.double()+weights,dim=1)-area.sum(1).log()


def frame_loss(z, labels):
    y=torch.as_tensor(labels,device=z.device,dtype=z.dtype)
    losses=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    return torch.stack([losses[y==c].mean() for c in (0,1) if (y==c).any()]).mean()


def prepare(out):
    manifest=read_json(OLD/'input_manifest.json');rows=manifest['rows'];old_protocol=read_json(OLD/'protocol.json')
    if len(rows)!=272 or {r['source_id'] for r in rows if r['split']=='train'}&{r['source_id'] for r in rows if r['split']=='validation'}:raise ValueError('INPUT_SPLIT')
    for name,key in [('standardizer.json','standardizer_sha256'),('grouping_config.json','grouping_sha256'),('models/fold_models.json','pretrained_models_sha256')]:
        if sha256_file(BASE/name)!=old_protocol[key]:raise ValueError('FROZEN_STATISTICS_OR_PRETRAINED_IDENTITY')
    old=next(r for r in read_json(OLD/'models/fold_models.json')['records'] if r['seed']==SEED and r['condition']=='GLOBAL_COMPONENT')
    if old['epoch']!=5 or old['status']!='COMPLETE' or sha256_file(old['checkpoint'])!=old['checkpoint_sha256']:raise ValueError('OLD_MODEL_IDENTITY')
    init=next(r for r in read_json(BASE/'models/fold_models.json')['records'] if r['seed']==SEED and r['condition']=='GLOBAL_COMPONENT')
    if old['identity']['initial_sha256']!=init['checkpoint_sha256']:raise ValueError('INITIAL_CHECKPOINT_MISMATCH')
    annotations=read_json(MAPPING);labels={};timing={};coverage=[]
    for row in rows:
        metadata=[]
        for b in row['blocks']:
            m=read_json(Path(b['path']).with_suffix('.json'))
            if m['identity']!=old_protocol['cache_identity'] or m['video_sha256']!=row['video_sha256']:raise ValueError('CACHE_METADATA_IDENTITY')
            metadata+=m['frames']
        pts=np.array([m['pts_s'] for m in metadata]);indices=[m['frame_index'] for m in metadata]
        if indices!=list(range(row['actual_frames'])) or not np.isfinite(pts).all() or np.any(np.diff(pts)<0):raise ValueError('FRAME_PTS_IDENTITY')
        if any(abs(m['absolute_pts_s']-m['time_origin_s']-m['pts_s'])>1e-8 or m['time_origin_s']!=0 for m in metadata):raise ValueError('ANNOTATION_TIME_ORIGIN_UNRESOLVED')
        annotation=exact_annotation(row,annotations);intervals=annotation['manipulation_intervals'] if annotation else []
        if intervals and max(v['end_s'] for v in intervals)>row['duration_s']:raise ValueError('INTERVAL_EXCEEDS_VIDEO_DURATION')
        y=interval_labels(pts,intervals)
        if row['role']=='fake' and not y.any():raise ValueError('NO_POSITIVE_FRAME_FROM_OFFICIAL_INTERVAL')
        labels[row['video_id']]=y;timing[row['video_id']]=pts
        coverage.append({'video_id':row['video_id'],'source':row['source_id'],'split':row['split'],'role':row['role'],
            'frames':len(pts),'positive':int(y.sum()),'negative':int((~y).sum()),'first_pts':float(pts[0]),'last_pts':float(pts[-1]),
            'intervals':intervals,'annotation_evidence':annotation['lineage_evidence'] if annotation else [],
            'video_sha256':row['video_sha256'],'endpoint':'closed [start,end]; union; native video-relative seconds'})
    csv_rows(out/'annotation_coverage.csv',coverage)
    config={**old_protocol,'seeds':[SEED],'loss':'video BCE + 0.5 class-balanced video-normalized frame BCE; real/fake half per pair',
        'annotation_mapping_sha256':sha256_file(MAPPING),'endpoint':'closed [start,end] union','old_comparator_sha256':old['checkpoint_sha256'],
        'chunk':64,'parameters':319969,'entry_sha256':sha256_file(__file__),'training_loop_sha256':sha256_file(training.__file__)}
    if (out/'protocol.json').exists() and read_json(out/'protocol.json')!=config:raise ValueError('PROTOCOL_CHANGED')
    save_json(out/'protocol.json',config);save_json(out/'input_manifest.json',manifest)
    return rows,labels,timing,config,old,coverage


def temporal_step(model,row,identity,chunk,labels):
    cpu,read_s=load_video(row,identity);z,cells=VideoTraining(model).video(cpu,chunk,return_cells=True)
    frames=frame_logits(cells,cpu['area']);lv=F.binary_cross_entropy_with_logits(z,z.new_tensor(float(row['video_label'])))
    lf=frame_loss(frames,labels[row['video_id']]);loss=.5*(lv+.5*lf);loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('NONFINITE_LOSS_OR_GRADIENT')
    return {'loss':float(loss.detach()),'video_loss':float(lv.detach()),'frame_loss':float(lf.detach()),'read_s':read_s}


def smoke(out,rows,labels,config):
    seed_all(SEED);model,_=training.load_model(BASE,'GLOBAL_COMPONENT',SEED);model.train()
    pair=sorted([r for r in rows if r['source_id']=='1LARL' and r['split']=='train'],key=lambda r:r['role'])
    if len(pair)!=2:raise ValueError('SMOKE_PAIR_MISSING')
    cpu,_=load_video(pair[0],config['cache_identity']);short={k:v[:70].clone() for k,v in cpu.items()};del cpu
    # A real sequence crosses the 64-frame boundary; gradients to first-chunk inputs must survive.
    short['x'].requires_grad_();z,c=VideoTraining(model).video(short,64,return_cells=True)
    f=frame_logits(c,short['area']);frame_loss(f[-1:],np.array([False])).backward()
    gradient=float(short['x'].grad[:64].abs().max());assert gradient>0 and np.isfinite(gradient)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        a=VideoTraining(model).video(short,64,use_checkpoint=False)
        b,cc=VideoTraining(model).video(short,64,use_checkpoint=False,return_cells=True)
        ff=frame_logits(cc,short['area']);weights=short['area'].sum(1).to('cuda',dtype=torch.float64)
        equivalent=torch.logsumexp(ff+weights.log(),0)-weights.sum().log()
        error=max(float(abs(a-b)),float(abs(b-equivalent)));assert error<1e-7
    del short,z,c,f,cc,ff,model
    seed_all(SEED);model,_=training.load_model(BASE,'GLOBAL_COMPONENT',SEED);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);opt.zero_grad(set_to_none=True)
    details=[temporal_step(model,r,config['cache_identity'],64,labels) for r in pair]
    torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    save_json(out/'smoke.json',{'status':'PASS','source':'1LARL','real_fake_full_pair':True,'last_frame_to_first_chunk_gradient':gradient,
                              'zero_auxiliary_forward_and_area_lse_error':error,'details':details,'formal_model_reinitialized':True})
    del opt,model;torch.cuda.empty_cache()


def metrics(rows,predictions,labels):
    y=np.array([r['video_label'] for r in rows]);z=np.array([predictions[r['video_id']]['video_logit'] for r in rows])
    video=training._metric(y,z);yy=np.concatenate([labels[r['video_id']] for r in rows]);zz=np.concatenate([predictions[r['video_id']]['frame_logits'] for r in rows]);pred=zz>=0
    tp=int((pred&yy).sum());fp=int((pred&~yy).sum());fn=int((~pred&yy).sum())
    eligible=[r for r in rows if r['role']=='fake' and len(np.unique(labels[r['video_id']]))==2]
    real=[r for r in rows if r['role']=='real']
    frame={'pooled_ap':float(average_precision_score(yy,zz)),'pooled_auroc':float(roc_auc_score(yy,zz)),
        'fake_video_macro_ap':float(np.mean([average_precision_score(labels[r['video_id']],predictions[r['video_id']]['frame_logits']) for r in eligible])) if eligible else None,
        'eligible_fake_videos':len(eligible),'excluded_single_class_fake_videos':sum(r['role']=='fake' for r in rows)-len(eligible),
        'real_video_macro_fpr':float(np.mean([np.mean(predictions[r['video_id']]['frame_logits']>=0) for r in real])),
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else None,'tp':tp,'fp':fp,'fn':fn}
    return {'video':video,'frame':frame}


def evaluate(out,rows,labels,timing,config,new_record,old_record,progress):
    val=[r for r in rows if r['split']=='validation'];results={};all_scores=[]
    for name,record in [('VIDEO_ONLY',old_record),('TEMPORAL',new_record)]:
        model,_=training.load_model(BASE,'GLOBAL_COMPONENT',SEED)
        if sha256_file(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('EVALUATION_CHECKPOINT_HASH')
        cp=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        if cp['epoch']!=5 or cp['identity']!=record['identity']:raise ValueError('EVALUATION_CHECKPOINT_IDENTITY')
        model.load_state_dict(cp['model'],strict=True);model.eval();predictions={}
        saved=out/'responses'/name;saved.mkdir(parents=True,exist_ok=True)
        old_scores=read_json(OLD/'scores.json')
        for i,r in enumerate(val):
            p=saved/f'{r["video_id"]}.npz';meta=p.with_suffix('.json')
            if p.exists():
                m=read_json(meta)
                if m!={'checkpoint_sha256':record['checkpoint_sha256'],'video_sha256':r['video_sha256'],'sha256':sha256_file(p)}:raise ValueError('SCORE_RESUME_IDENTITY')
                with np.load(p,allow_pickle=False) as v:predictions[r['video_id']]={k:v[k].copy() for k in ('video_logit','frame_logits')}
            else:
                cpu,_=load_video(r,config['cache_identity'])
                with torch.no_grad():z,cells=VideoTraining(model).video(cpu,64,use_checkpoint=False,return_cells=True);f=frame_logits(cells,cpu['area'])
                predictions[r['video_id']]={'video_logit':np.array(float(z)),'frame_logits':f.cpu().numpy()}
                write_npz_atomic(p,**predictions[r['video_id']],pts_s=timing[r['video_id']],labels=labels[r['video_id']])
                save_json(meta,{'checkpoint_sha256':record['checkpoint_sha256'],'video_sha256':r['video_sha256'],'sha256':sha256_file(p)});del cpu,z,cells,f
            if name=='VIDEO_ONLY':
                expected=old_scores[r['video_id']]['video_logits']['GLOBAL_COMPONENT__seed20260909']
                if abs(float(predictions[r['video_id']]['video_logit'])-expected)>2e-5:raise ValueError('OLD_SAVED_VIDEO_SCORE_NOT_REPRODUCED')
            all_scores.append({'condition':name,'video_id':r['video_id'],'source':r['source_id'],'role':r['role'],'logit':float(predictions[r['video_id']]['video_logit'])})
            progress('score',i+1,len(val),condition=name)
        results[name]=metrics(val,predictions,labels);del model;torch.cuda.empty_cache()
    csv_rows(out/'evaluation/video_scores.csv',all_scores);save_json(out/'evaluation/summary.json',results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed first three validation sources, not selected by scores.
    for source in sorted({r['source_id'] for r in val})[:3]:
        r=next(r for r in val if r['source_id']==source and r['role']=='fake');fig,ax=plt.subplots(figsize=(9,3))
        for name in results:
            with np.load(out/'responses'/name/f'{r["video_id"]}.npz',allow_pickle=False) as z:ax.plot(z['pts_s'],z['frame_logits'],label=name)
        for v in read_json(out/'annotation_details.json')[r['video_id']]['intervals']:ax.axvspan(v['start_s'],v['end_s'],alpha=.2,color='gold')
        ax.axhline(0,ls='--',color='black');ax.set(xlabel='Native video-relative PTS (s)',ylabel='Frame logit',title=source+' official intervals (not pixel truth)');ax.legend();fig.tight_layout();fig.savefig(out/f'{source}_curve.svg');plt.close(fig)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGTERM,training.stop);signal.signal(signal.SIGINT,training.stop);started=time.monotonic();phase='prepare'
    def progress(stage,completed,total,**extra):
        nonlocal phase
        phase=stage;state.update(stage=stage,completed=completed,total=total,**extra)
        save_json(out/'progress.json',{'status':'RUNNING','pid':os.getpid(),'updated_unix':time.time(),**state});print(state,flush=True)
    state={}
    try:
        rows,labels,timing,config,old,coverage=prepare(out)
        save_json(out/'annotation_details.json',{r['video_id']:r for r in coverage})
        for split in ('train','validation'):
            cc=[r for r in coverage if r['split']==split];print('ANNOTATION_COVERAGE',split,len(cc),sum(r['frames'] for r in cc),sum(r['positive'] for r in cc),flush=True)
        if a.prepare_only:return
        save_json(out/'final_status.json',{'status':'RUNNING','pid':os.getpid(),'phase':'smoke'})
        print('CUDA',torch.cuda.get_device_name(),flush=True)
        if not (out/'smoke.json').exists():smoke(out,rows,labels,config)
        terms=read_json(out/'training_terms.json') if (out/'training_terms.json').exists() else {}
        def step(model,row,identity,chunk):
            detail=temporal_step(model,row,identity,chunk,labels)
            terms[f'{state["epoch"]}:{row["video_id"]}']={'epoch':state['epoch'],'video_id':row['video_id'],**detail}
            save_json(out/'training_terms.json',terms);return detail
        train_started=time.monotonic()
        records=training.train(out,BASE,rows,config['cache_identity'],64,digest(config),progress,a.resume,
            conditions=('GLOBAL_COMPONENT',),epochs=5,seeds=(SEED,),train_step=step)
        train_s=time.monotonic()-train_started;eval_started=time.monotonic()
        result=evaluate(out,rows,labels,timing,config,records[0],old,progress);evaluation_s=time.monotonic()-eval_started
        loss=[]
        for epoch in range(1,6):
            rr=[r for r in terms.values() if r['epoch']==epoch]
            if len(rr)!=244:raise ValueError('EPOCH_LOSS_TERMS_INCOMPLETE')
            loss.append({'epoch':epoch,'video_loss':float(np.mean([r['video_loss'] for r in rr])),
                         'frame_loss':float(np.mean([r['frame_loss'] for r in rr]))})
        csv_rows(out/'loss_components.csv',loss)
        lines=['# V8 官方时间区间辅助监督单seed pilot','', 'GLOBAL_COMPONENT seed20260909，窗口epoch100初始化，固定5轮，最后权重；仅新增0.5帧BCE，无新增参数。',
            '训练244视频/122source/206779帧；验证28视频/14source/27505帧。闭区间并集、原生PTS，real全负；监督仅frame面积LSE，不给cell逐格标签。',
            '正帧训练32181、验证4046，无全正fake；全部帧/内容格参与。完整跨chunk梯度，FP32参数，原双精度LSE，chunk64，AdamW1e-4/wd1e-4、clip1。',
            '旧对照同seed、同初始checkpoint、统计、输入与5轮，旧视频分数前向复现通过。',
            '', '## 视频与帧结果',json.dumps(result,ensure_ascii=False,indent=2),'','## 训练分项',json.dumps(loss,ensure_ascii=False),
            '','## 边界与决策','帧AP不是视频AP或时间检测mAP；无空间监督/评价，不独立证明三维或component贡献。开发验证反复使用，单seed机制探索，不称稳定增益。',
            '是否补另外两seed在真实结果验收后讨论，本轮不执行追加训练。',
            f'训练本次调用{train_s:.3f}s；模型累计{records[0]["training_elapsed_s"]:.3f}s；评分评价{evaluation_s:.3f}s；调用总耗时{time.monotonic()-started:.3f}s。',
            f'GPU峰值allocated {torch.cuda.max_memory_allocated()/2**30:.4f}GiB，reserved {torch.cuda.max_memory_reserved()/2**30:.4f}GiB；新增存储{sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}bytes。']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        save_json(out/'final_status.json',{'status':'COMPLETE','models':1,'seed':SEED,'epochs':5,'training_s':records[0]['training_elapsed_s'],'evaluation_s':evaluation_s,'elapsed_s':time.monotonic()-started})
        save_json(out/'progress.json',{'status':'COMPLETE','stage':'report','completed':1,'total':1})
    except BaseException as e:
        save_json(out/'final_status.json',{'status':'STOPPED_SAFE' if isinstance(e,InterruptedError) else 'FAILED','phase':phase,'error':str(e),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
