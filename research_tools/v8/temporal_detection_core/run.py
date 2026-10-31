"""One independent Python entry: shared preparation, six matched runs, final test/report.

Reuses V8 VideoTraining and temporal losses. No generic job orchestration or
new score/calibration branch. Publication errors never overwrite COMPLETE.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_recall_fscore_support,confusion_matrix

from scripts.prepare_v8_assets import digest
from research_tools.v8.local_geometry_pilot.runner import seed_all,atomic_torch,rng_state,restore_rng
from research_tools.v8.full_video_training.model import VideoTraining
from research_tools.v8.full_video_training.temporal import frame_logits,frame_loss
from research_tools.v8.spatial_localization.run import pairs,parameters_sha
from .model import StructureModel,CONDITIONS,INPUT_FIELDS
from .data import save,freeze,extract,prepare_structures,load_video,validate_manifest

STOP=False

def stop(*_):
    global STOP
    STOP=True


def classification(y,z):
    y=np.asarray(y,dtype=int);z=np.asarray(z,dtype=float)
    if not np.isfinite(z).all():raise ValueError('NONFINITE_PREDICTION')
    p=z>=0;precision,recall,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'AUROC':float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None,
        'AP':float(average_precision_score(y,z)) if y.any() else None,'ACC':float(accuracy_score(y,p)),
        'F1':float(f1),'Precision':float(precision),'Recall':float(recall),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}


def labels(r):
    pts=np.array([p['pts_s'] for p in r['pts']]);y=np.zeros(len(pts),bool)
    for lo,hi in r['intervals_s']:
        if not np.isfinite([lo,hi]).all() or not 0<=lo<=hi:raise ValueError('INVALID_OFFICIAL_INTERVAL')
        y|=(pts>=lo)&(pts<=hi)
    if r['role']=='real' and y.any():raise ValueError('REAL_LABEL')
    if r['role']=='fake' and not y.any():raise ValueError('MISSING_POSITIVE_OFFICIAL_FRAME')
    return y


def forward(model,cpu,config,checkpoint):
    return VideoTraining(model).video(cpu,config['chunk_frames'],use_checkpoint=checkpoint,return_cells=True)


def update(model,opt,pair,out,condition,config):
    t0=time.monotonic();opt.zero_grad(set_to_none=True);read_s=0.;losses=[];vls=[];fls=[]
    for r in sorted(pair,key=lambda r:r['role']):
        cpu,rs=load_video(out,r,condition);read_s+=rs
        z,cells=forward(model,cpu,config,config['use_checkpoint']);f=frame_logits(cells,cpu['area'])
        lv=F.binary_cross_entropy_with_logits(z,z.new_tensor(float(r['label'])))
        lf=frame_loss(f,labels(r));loss=.5*(lv+config['frame_loss_weight']*lf)
        if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_LOSS')
        loss.backward();losses.append(float(loss.detach()));vls.append(float(lv.detach())*.5);fls.append(float(lf.detach())*.5)
        del cpu,z,cells,f,loss,lv,lf
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True));opt.step();torch.cuda.synchronize()
    return {'loss':sum(losses),'video_loss':sum(vls),'frame_loss':sum(fls),'read_s':read_s,'seconds':time.monotonic()-t0,'gradient_norm':norm}


def evaluate(model,rows,out,condition,config,response_dir=None):
    t0=time.monotonic();model.eval();vy=[];vz=[];fy=[];fz=[];records=[];read_s=0.
    with torch.inference_mode():
        for r in rows:
            cpu,rs=load_video(out,r,condition);read_s+=rs;z,cells=forward(model,cpu,config,False)
            f=frame_logits(cells,cpu['area']).cpu().numpy();y=labels(r);logit=float(z)
            vy.append(r['label']);vz.append(logit);fy.append(y);fz.append(f)
            records.append({'video_id':r['video_id'],'source_id':r['source_id'],'role':r['role'],'generator':r['generator'],
                            'video_label':r['label'],'video_logit':logit,'frames':len(y),'positive_frames':int(y.sum())})
            if response_dir:
                response_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(response_dir/(r['video_id']+'.npz'),frame_logits=f,labels=y,pts_s=np.array([p['pts_s'] for p in r['pts']]),video_logit=logit)
            del cpu,z,cells,f
    torch.cuda.synchronize()
    fm=classification(np.concatenate(fy),np.concatenate(fz))
    return {'video':classification(vy,vz),'frame':{k:fm[k] for k in ['AUROC','AP','F1']},
            'videos':records,'seconds':time.monotonic()-t0,'read_s':read_s}


def preflight(config,out,rows,progress):
    target=out/'preflight.json'
    if target.exists():return json.loads(target.read_text())
    pair=next(iter(pairs(rows,'train').values()));extract(config,out,pair,progress)
    dest=out/'preflight';dest.mkdir(exist_ok=True)
    if not (dest/'observations').exists():(dest/'observations').symlink_to(out/'observations',target_is_directory=True)
    fitted=prepare_structures(config,dest,pair,progress);records=[];common=None
    for c in CONDITIONS:
        seed_all(config['seeds'][0]);m=StructureModel(fitted[c]['statistics']).cuda();sha=parameters_sha(m)
        if common and sha!=common:raise ValueError('COMMON_INITIALIZATION_MISMATCH')
        common=sha;opt=torch.optim.AdamW(m.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        torch.cuda.reset_peak_memory_stats();m.train();before=parameters_sha(m)
        detail=update(m,opt,pair,dest,c,config)
        if parameters_sha(m)==before:raise ValueError('NO_PARAMETER_UPDATE')
        result=evaluate(m,pair,dest,c,config)
        cp=dest/(c+'.pt');atomic_torch(cp,{'model':m.state_dict()});m2=StructureModel(fitted[c]['statistics']).cuda()
        m2.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);again=evaluate(m2,pair,dest,c,config)
        error=max(abs(a['video_logit']-b['video_logit']) for a,b in zip(result['videos'],again['videos']))
        if error>1e-6:raise ValueError('SAVE_RELOAD_PARITY')
        # Last-frame objective must reach first-chunk input through recurrent state.
        cpu,_=load_video(dest,pair[0],c);n=min(len(cpu['x']),config['chunk_frames']+3)
        short={k:v[:n].clone() for k,v in cpu.items()};short['x'].requires_grad_();m.zero_grad(set_to_none=True)
        _,cc=forward(m,short,config,True);ff=frame_logits(cc,short['area']);ff[-1].backward()
        gradient=float(short['x'].grad[:config['chunk_frames']].abs().max())
        if not gradient>0:raise ValueError('CROSS_CHUNK_GRADIENT')
        records.append({'condition':c,**detail,'evaluation_s':result['seconds'],'reload_max_error':error,
            'last_to_first_chunk_gradient':gradient,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'parameters':sum(p.numel() for p in m.parameters())})
        del m,m2,opt,cpu,short,cc,ff;torch.cuda.empty_cache()
    frames=sum(r['actual_frames'] for r in pair);allframes=sum(r['actual_frames'] for r in rows)
    trainframes=sum(r['actual_frames'] for r in rows if r['split']=='train');devframes=sum(r['actual_frames'] for r in rows if r['split']=='dev')
    monitor_sources=sorted({r['source_id'] for r in rows if r['split']=='train'})[:4]
    monitorframes=sum(r['actual_frames'] for r in rows if r['source_id'] in monitor_sources)
    maxtrain=sum(r['seconds']/frames*trainframes*config['epochs']*3 for r in records)
    maxeval=sum(r['evaluation_s']/frames*(devframes+monitorframes)*config['epochs']*3 for r in records)
    finaleval=sum(r['evaluation_s']/frames*allframes*3*2 for r in records)
    observed=[json.loads((out/'observations'/r['video_id']/'complete.json').read_text())['cost'] for r in pair]
    result={'status':'PASS','chunk_frames':config['chunk_frames'],'checkpoint':config['use_checkpoint'],'records':records,
        'common_parameter_sha256':common,'full_pair_frames':frames,'monitor_sources':monitor_sources,
        'budget_seconds':{'preparation_upper_linear_reference':sum(v['elapsed_s'] for v in observed)/frames*allframes,
            'six_models_max30_updates':maxtrain,'six_models_max30_monitor_dev':maxeval,'final_best_last_upper_reference':finaleval},
        'projection_limit':'one complete training pair, linear estimate; early stopping, IO and shared GPU change actual costs',
        'smoke_weights':'discarded; formal initialization recreated by seed; temporary diagnostic checkpoints only'}
    save(target,result);return result


def train_one(config,out,rows,fitted,condition,seed,identity,progress,resume=False):
    dest=out/condition/str(seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():
        s=json.loads((dest/'summary.json').read_text())
        if s['identity']!=identity:raise ValueError('SUMMARY_IDENTITY')
        return s
    seed_all(seed);m=StructureModel(fitted[condition]['statistics']).cuda();initial=parameters_sha(m)
    opt=torch.optim.AdamW(m.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);order_rng=random.Random(seed)
    trainpairs=pairs(rows,'train');dev=[r for r in rows if r['split']=='dev'];monitor_ids=json.loads((out/'preflight.json').read_text())['monitor_sources']
    monitor=[r for r in rows if r['split']=='train' and r['source_id'] in monitor_ids]
    history=[];epoch=1;position=0;order=[];terms=[];best=-float('inf');best_epoch=0;stale=0;updates=0;train_s=0.;eval_s=0.;peak_a=peak_r=0
    cp_path=dest/'resume.pt'
    if cp_path.exists():
        if not resume:raise ValueError('EXISTING_RUN_REQUIRES_EXPLICIT_RESUME')
        p=torch.load(cp_path,map_location='cpu',weights_only=False)
        if p['identity']!=identity or p['condition']!=condition or p['seed']!=seed or p['initial_sha']!=initial:raise ValueError('RESUME_IDENTITY')
        m.load_state_dict(p['model']);opt.load_state_dict(p['optimizer']);restore_rng(p['rng']);order_rng.setstate(p['order_rng'])
        epoch,position,order,terms,history,best,best_epoch,stale,updates,train_s,eval_s,peak_a,peak_r=(p[k] for k in ['epoch','position','order','terms','history','best','best_epoch','stale','updates','train_s','eval_s','peak_a','peak_r'])
    def payload(next_epoch,next_position,next_order,next_terms):
        return {'identity':identity,'condition':condition,'seed':seed,'model':m.state_dict(),'optimizer':opt.state_dict(),
            'rng':rng_state(),'order_rng':order_rng.getstate(),'epoch':next_epoch,'position':next_position,'order':next_order,'terms':next_terms,
            'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,'updates':updates,'train_s':train_s,'eval_s':eval_s,
            'peak_a':peak_a,'peak_r':peak_r,'initial_sha':initial}
    torch.cuda.reset_peak_memory_stats()
    while epoch<=config['epochs'] and stale<config['patience']:
        if not order:order=sorted(trainpairs);order_rng.shuffle(order);terms=[];position=0
        m.train()
        for pi in range(position,len(order)):
            if STOP:
                atomic_torch(cp_path,payload(epoch,pi,order,terms));raise InterruptedError('SAFE_PAIR_BOUNDARY_STOP')
            progress('training',condition=condition,seed=seed,epoch=epoch,pair=pi+1,pairs=len(order))
            term=update(m,opt,trainpairs[order[pi]],out,condition,config);terms.append(term);train_s+=term['seconds'];updates+=1
            peak_a=max(peak_a,torch.cuda.max_memory_allocated());peak_r=max(peak_r,torch.cuda.max_memory_reserved())
            atomic_torch(cp_path,payload(epoch,pi+1,order,terms))
        monitor_result=evaluate(m,monitor,out,condition,config);dev_result=evaluate(m,dev,out,condition,config)
        eval_s+=monitor_result['seconds']+dev_result['seconds'];score=dev_result['video']['AUROC'];improved=score>best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        record={'epoch':epoch,'online_loss':float(np.mean([t['loss'] for t in terms])),
            'video_loss':float(np.mean([t['video_loss'] for t in terms])),'frame_loss':float(np.mean([t['frame_loss'] for t in terms])),
            'monitor':{k:monitor_result[k] for k in ['video','frame']},'dev':{k:dev_result[k] for k in ['video','frame']},
            'learning_rate':opt.param_groups[0]['lr'],'epoch_updates_s':sum(t['seconds'] for t in terms),
            'read_s':sum(t['read_s'] for t in terms),'evaluation_s':monitor_result['seconds']+dev_result['seconds'],
            'peak_allocated_bytes':peak_a,'peak_reserved_bytes':peak_r,'best_epoch':best_epoch,'updates':updates}
        history.append(record);save(dest/'history.json',history)
        completed=payload(epoch+1,0,[],[]);completed['checkpoint_epoch']=epoch
        atomic_torch(dest/'last.pt',completed)
        if improved:atomic_torch(dest/'best.pt',completed)
        atomic_torch(cp_path,completed)
        print('EPOCH_COMPLETE',condition,seed,epoch,score,'best',best_epoch,flush=True)
        epoch+=1;position=0;order=[];terms=[]
    summary={'identity':identity,'condition':condition,'seed':seed,'initial_parameter_sha256':initial,'best_epoch':best_epoch,
             'last_epoch':epoch-1,'updates':updates,'train_updates_s':train_s,'routine_evaluation_s':eval_s,
             'peak_allocated_bytes':peak_a,'peak_reserved_bytes':peak_r,'parameters':sum(p.numel() for p in m.parameters())}
    save(dest/'summary.json',summary);del opt,m;torch.cuda.empty_cache();return summary


def aggregate(records):
    result={}
    for c in CONDITIONS:
        rr=[r for r in records if r['condition']==c];result[c]={}
        for level,fields in [('video',['AUROC','AP','ACC','F1','Precision','Recall']),('frame',['AUROC','AP','F1'])]:
            result[c][level]={k:{'mean':float(np.mean([r['best']['test'][level][k] for r in rr])),
                                   'std':float(np.std([r['best']['test'][level][k] for r in rr],ddof=1))} for k in fields}
    return result


def finish(config,out,rows,fitted,summaries,identity,progress):
    records=[];start=time.monotonic()
    # ALL six models must have finished before any test scoring.
    if len(summaries)!=6:raise ValueError('SIX_MODELS_REQUIRED_BEFORE_TEST')
    for s in summaries:
        c,seed=s['condition'],s['seed'];dest=out/c/str(seed);record={**s}
        for name in ['best','last']:
            cp=dest/(name+'.pt');checkpoint_sha=digest(cp);m=StructureModel(fitted[c]['statistics']).cuda()
            p=torch.load(cp,map_location='cpu',weights_only=False)
            if p['identity']!=identity:raise ValueError('FINAL_CHECKPOINT_IDENTITY')
            m.load_state_dict(p['model']);result={'epoch':p['checkpoint_epoch'],'checkpoint_sha256':checkpoint_sha}
            # Best alone is scored on test exactly once. Last train/dev are descriptive.
            for split in (['train','dev','test'] if name=='best' else ['train','dev']):
                report_path=dest/f'{name}_{split}.json'
                if report_path.exists():
                    result[split]=json.loads(report_path.read_text())
                    if result[split]['checkpoint_sha256']!=checkpoint_sha:raise ValueError('SAVED_SCORE_CHECKPOINT_MISMATCH')
                else:
                    progress('final_evaluation',condition=c,seed=seed,checkpoint=name,split=split)
                    scored=evaluate(m,[r for r in rows if r['split']==split],out,c,config,dest/'responses'/name/split)
                    scored['checkpoint_sha256']=checkpoint_sha;save(report_path,scored);result[split]=scored
            record[name]=result;del m;torch.cuda.empty_cache()
        records.append(record)
    result={'status':'COMPLETE','identity':identity,'models':records,'test_mean_std':aggregate(records),
        'seed_differences':[{ 'seed':seed,**{level:{k:next(r for r in records if r['condition']==CONDITIONS[1] and r['seed']==seed)['best']['test'][level][k]-next(r for r in records if r['condition']==CONDITIONS[0] and r['seed']==seed)['best']['test'][level][k] for k in fields} for level,fields in [('video',['AUROC','AP','ACC','F1','Precision','Recall']),('frame',['AUROC','AP','F1'])]}} for seed in config['seeds']],
        'final_evaluation_s':time.monotonic()-start,'preparation':json.loads((out/'preparation_cost.json').read_text()),
        'new_output_bytes_before_report':sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())}
    save(out/'result_summary.json',result);report(config,out,rows,result)
    save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'unix_time':time.time()})
    return result


def report(config,out,rows,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# V8二维／三维时序结构匹配检测实验','', 'COMPLETE；新队列从头训练；不复现历史0.872449，不含空间定位成绩。',
        '测试各seed独立评分，mean±sample std(ddof=1)，不集成；阈值logit≥0，帧指标不是区间mAP。',
        '', '实际source pair：'+str({s:len(pairs(rows,s)) for s in ['train','dev','test']}),
        '全部实际帧：'+str({s:sum(r['actual_frames'] for r in rows if r['split']==s) for s in ['train','dev','test']}),
        '', '|条件|seed|best/结束轮|更新|test视频AUROC|AP|ACC|F1|Precision|Recall|TN/FP/FN/TP|帧AUROC|帧AP|帧F1|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in result['models']:
        v=r['best']['test']['video'];f=r['best']['test']['frame']
        lines.append('|'+ '|'.join([r['condition'],str(r['seed']),f"{r['best_epoch']}/{r['last_epoch']}",str(r['updates']),*[f'{v[k]:.6f}' for k in ['AUROC','AP','ACC','F1','Precision','Recall']],'/'.join(str(v[k]) for k in ['TN','FP','FN','TP']),*[f'{f[k]:.6f}' for k in ['AUROC','AP','F1']]])+'|')
        h=json.loads((out/r['condition']/str(r['seed'])/'history.json').read_text());x=[v['epoch'] for v in h]
        fig,axes=plt.subplots(1,3,figsize=(12,3))
        axes[0].plot(x,[v['online_loss'] for v in h]);axes[0].set_title('Online pair loss')
        for split in ['monitor','dev']:
            axes[1].plot(x,[v[split]['video']['AUROC'] for v in h],label=split);axes[2].plot(x,[v[split]['frame']['AP'] for v in h],label=split)
        axes[1].set_title('Video AUROC');axes[2].set_title('Frame AP')
        for ax in axes:ax.set_xlabel('Epoch')
        axes[1].legend();axes[2].legend();fig.tight_layout();fig.savefig(out/r['condition']/str(r['seed'])/'curves.png');plt.close(fig)
    lines+=['','## 三seed汇总',json.dumps(result['test_mean_std'],ensure_ascii=False,indent=2),
        '', '## 同seed 3D−2D',json.dumps(result['seed_differences'],ensure_ascii=False,indent=2),
        '', '## 结论与边界']
    diffs=[d['video']['AUROC'] for d in result['seed_differences']]
    lines.append(f'视频AUROC的3D−2D三个配对差值为{diffs}；提高{sum(d>0 for d in diffs)}、持平{sum(d==0 for d in diffs)}、下降{sum(d<0 for d in diffs)}。不挑最好seed。')
    for level,fields in [('video',['AUROC','AP','ACC','F1']),('frame',['AUROC','AP','F1'])]:
        for metric in fields:
            a=result['test_mean_std'][CONDITIONS[0]][level][metric];b=result['test_mean_std'][CONDITIONS[1]][level][metric]
            ds=[v[level][metric] for v in result['seed_differences']]
            lines.append(f"{level} {metric}: 2D {a['mean']:.6f}±{a['std']:.6f}; 3D {b['mean']:.6f}±{b['std']:.6f}; 均值差{b['mean']-a['mean']:+.6f}，逐seed提高{sum(d>0 for d in ds)}/3。")
    lines+=['只有空间表示及派生分组/关系不同；不把差异进一步归因于XYZ、分组或某层。绝对检测效果、帧级是否同步改善应依据上表和各seed标准指标解释。',
        '场景仅原片固定三时刻预审，不能证明未采样时刻绝无短暂镜头变化。旧完整训练source清单缺失；测试采用官方test-only且排除已知使用source，以历史官方train协议支持独立性，而非宣称恢复完整历史交集。',
        'best/last完整train/dev标准指标及逐视频分数见各模型JSON；测试只评价best。monitor固定训练4source，不代表完整训练指标。',
        '', '## 成本', '每模型训练/评价/显存：'+json.dumps([{k:r[k] for k in ['condition','seed','train_updates_s','routine_evaluation_s','peak_allocated_bytes','peak_reserved_bytes']} for r in result['models']],ensure_ascii=False),
        '共享前端总耗时（逐视频实测和）：'+str(sum(r['elapsed_s'] for r in result['preparation']['videos']))+'秒；旧RGB/XYZ只读复用，缺失逐点跟踪才补算。',
        '最终评价：'+str(result['final_evaluation_s'])+'秒；报告前输出：'+str(result['new_output_bytes_before_report'])+'字节。',
        '本轮到此停止，不自动追加实验。']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')


def publish(out):
    repo=Path(__file__).resolve().parents[3];branch='v8-3d-temporal-detection-core'
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    if git('branch','--show-current')!=branch or git('status','--porcelain'):raise ValueError('PUBLICATION_REQUIRES_CLEAN_TASK_BRANCH')
    doc=repo/'docs/experiments/v8-3d-temporal-detection-core.md';base=doc.read_text().split('\n## 自动完成结果')[0].replace('准备及真实完整pair实测已通过，尚无正式检测成绩。','六模型训练、最终测试及报告已完成。')
    doc.write_text(base+'\n## 自动完成结果\n\n'+(out/'report_zh.md').read_text())
    small=repo/'docs/experiments/v8-3d-temporal-detection-core-results.json'
    result=json.loads((out/'result_summary.json').read_text())
    small.write_text(json.dumps({**{k:result[k] for k in ['status','identity','test_mean_std','seed_differences','final_evaluation_s','new_output_bytes_before_report']},
        'models':[{**{k:r[k] for k in ['condition','seed','best_epoch','last_epoch','updates','train_updates_s','routine_evaluation_s','peak_allocated_bytes','peak_reserved_bytes']},
                  **{name:{'epoch':r[name]['epoch'],**{split:{level:r[name][split][level] for level in ['video','frame']} for split in (['train','dev','test'] if name=='best' else ['train','dev'])}} for name in ['best','last']}} for r in result['models']]},ensure_ascii=False,indent=2)+'\n')
    status=repo/'docs/v8/current_status.md';text=status.read_text();text=text.replace('本轮二维／三维检测实验：PREPARED，尚无正式结果。','本轮二维／三维检测实验：COMPLETE，六模型最终测试与报告完成；见对应实验文档。');status.write_text(text)
    paths=[doc,small,status]
    for c in CONDITIONS:
        for seed in [20260909,20260910,20260911]:
            target=repo/'docs/experiments'/f'v8-detection-{c.lower()}-{seed}.png';shutil.copyfile(out/c/str(seed)/'curves.png',target);paths.append(target)
    subprocess.run(['git','add','--',*[str(p.relative_to(repo)) for p in paths]],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Record completed matched 2D 3D temporal detection results'],cwd=repo,check=True)
    head=git('rev-parse','HEAD');save(out/'publication.json',{'status':'COMMITTED','commit':head})
    subprocess.run(['git','-c','http.version=HTTP/1.1','push','origin',branch],cwd=repo,check=True)
    save(out/'publication.json',{'status':'PUSHED','commit':head})
    remote=git('-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch).split()[0]
    if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
    save(out/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('docs/v8/temporal_detection_core_config.json'));p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text()).get('status')=='COMPLETE':
        raise ValueError('EXPERIMENT_ALREADY_COMPLETE_DO_NOT_RERUN')
    reference=json.loads(Path('docs/v8/evidence/temporal_detection_core_manifest.json').read_text())
    if digest(out/'manifest.json')!=reference['full_manifest_sha256']:raise ValueError('FROZEN_MANIFEST_CHECKSUM')
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def progress(stage,**kw):
        if STOP:raise InterruptedError('SAFE_PREPARE_OR_PAIR_BOUNDARY_STOP')
        save(out/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'unix_time':time.time(),**kw})
    code=Path(__file__).parent
    identity={'config_sha256':digest(a.config),'manifest_sha256':digest(out/'manifest.json'),
              'code_sha256':{str(q):digest(q) for q in [*sorted(code.glob('*.py')),
                  Path('research_tools/v8/full_video_training/model.py'),Path('research_tools/v8/full_video_training/temporal.py'),
                  Path('research_tools/v8/component_global_context_pilot/model.py'),Path('research_tools/v8/local_geometry_pilot/model.py'),
                  Path('research_tools/v8/local_geometry_pilot/data.py'),Path('research_tools/v8/full_video_streaming/inputs.py'),
                  Path('research_tools/v8/full_coverage_observation_field/frontend.py'),Path('research_tools/v8/full_coverage_observation_field/pipeline.py')]},
              'frontend_weights':{n:digest(Path(config['workspace'])/p) for n,p in [('moge','checkpoints/moge-2-vits-normal.pt'),
                 ('cotracker','checkpoints/scaled_online.pth'),('resnet18','cache/torch/hub/checkpoints/resnet18-f37072fd.pth')]},
              'input_fields':list(INPUT_FIELDS)}
    frozen=out/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=identity:raise ValueError('FROZEN_PROTOCOL_CHANGED')
    # preflight is before final code freeze; formal protocol only written at launch.
    if not a.preflight_only:save(frozen,identity)
    try:
        rows=freeze(config,out);validate_manifest(rows)
        from scripts.prepare_v8_assets import identities
        official={x['media_path']:x for x in identities(Path(config['data_root']))}
        for r in rows:
            labels(r)
            if r['role']=='fake':
                key=Path(r['video_path']).relative_to(Path(config['data_root'])/'official'/r['official_revision']).as_posix()
                ref=official[key]
                if any(ref[k]!=r[k] for k in ['source_id','generator','official_split','intervals_s']):raise ValueError('OFFICIAL_COMPOSITE_ANNOTATION_IDENTITY')
        smoke=preflight(config,out,rows,progress)
        if a.preflight_only:
            save(out/'progress.json',{'status':'PREFLIGHT_COMPLETE','stage':'preflight_complete','pid':os.getpid()});print(json.dumps(smoke),flush=True);return
        extract(config,out,rows,progress);fit_started=time.monotonic();fitted=prepare_structures(config,out,rows,progress)
        save(out/'structure_preparation_cost.json',{'seconds':time.monotonic()-fit_started,'statistics_sha256':digest(out/'statistics.json')})
        shared=[*range(67),71,76,78,79]
        for key in ['cell_mean','cell_scale']:
            a0=np.array(fitted[CONDITIONS[0]]['statistics'][key]);a1=np.array(fitted[CONDITIONS[1]]['statistics'][key])
            if not np.array_equal(a0[shared],a1[shared]):raise ValueError('SHARED_STANDARDIZER_MISMATCH')
        # Compare parameters, not representation-specific fitted statistic buffers.
        for seed in config['seeds']:
            hashes=[]
            for c in CONDITIONS:
                seed_all(seed);m=StructureModel(fitted[c]['statistics']);hashes.append(parameters_sha(m));del m
            if len(set(hashes))!=1:raise ValueError('UNPAIRED_COMMON_INITIALIZATION')
        summaries=[]
        for seed in config['seeds']:
            for c in CONDITIONS:summaries.append(train_one(config,out,rows,fitted,c,seed,identity,progress,a.resume))
        finish(config,out,rows,fitted,summaries,identity,progress)
    except BaseException as e:
        save(out/'progress.json',{'status':'STOPPED_SAFE' if isinstance(e,InterruptedError) else 'FAILED','error':str(e),'traceback':traceback.format_exc(),'unix_time':time.time()});raise
    # Completion is irrevocable by publication/network failure; exit history stays real.
    try:publish(out)
    except BaseException as e:
        previous=json.loads((out/'publication.json').read_text()) if (out/'publication.json').exists() else {}
        save(out/'publication.json',{**previous,'status':'FAILED','error':str(e),'traceback':traceback.format_exc(),'experiment_status':'COMPLETE'});raise

if __name__=='__main__':
    exit_code=0
    try:main()
    except BaseException:
        exit_code=1
        raise
    finally:
        # This is an exit record, not a success assertion; preserve earlier attempts.
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',type=Path,default=Path('docs/v8/temporal_detection_core_config.json'))
        args,_=parser.parse_known_args();output=Path(json.loads(args.config.read_text())['output'])
        if output.exists():
            (output/'exit_code.txt').write_text(str(exit_code)+'\n')
            with (output/'exit_history.jsonl').open('a') as history:
                history.write(json.dumps({'pid':os.getpid(),'exit_code':exit_code,'unix_time':time.time()})+'\n')
