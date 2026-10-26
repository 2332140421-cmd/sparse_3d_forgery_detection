"""One bounded joint experiment: preparation, complete-pair smoke, train, report."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch

from research_tools.v8.local_geometry_pilot.runner import atomic_torch,seed_all,rng_state,restore_rng
from scripts.prepare_v8_assets import digest
from .joint_data import select,extract,fit,Video,save,SEED,SCHEMA
from .model import LocalizationModel,CONDITIONS,video_loss
from .metrics import evaluate,history_csv,report

STOP=False


def request_stop(signum,frame):
    global STOP
    STOP=True


def progress(out,stage,**fields):
    if STOP and stage.startswith('frontend'):raise InterruptedError('SAFE_FRONTEND_STOP; incomplete video restarts on resume')
    save(out/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'unix_time':time.time(),**fields})


def parameters_sha(model):
    h=hashlib.sha256()
    for k,v in model.named_parameters():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def pairs(rows,split):
    result={}
    for r in rows:
        if r['split']==split:result.setdefault(r['pair_id'],[]).append(r)
    for k,v in result.items():
        if len(v)!=2 or {r['role'] for r in v}!={'real','fake'}:raise ValueError('PAIR_CONTRACT:'+k)
    return result


def update_pair(model,optimizer,pair,out,fitted,config,scaler):
    started=time.monotonic();optimizer.zero_grad(set_to_none=True);total=0.;losses={k:0. for k in ['video','frame','pixel']}
    for r in sorted(pair,key=lambda r:r['role']):
        v=Video(out,r,fitted['statistics'],fitted['grouping'])
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                loss,parts=video_loss(model,v,config['chunk_frames'],True)
            if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_LOSS')
            scaler.scale(loss*.5).backward();total+=float(loss.detach())*.5
            for k in losses:losses[k]+=float(parts[k].detach())*.5
        finally:v.close()
    scaler.unscale_(optimizer)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
    return {'loss':total,**losses,'gradient_norm':norm,'elapsed_s':time.monotonic()-started,'frames':sum(r['actual_frames'] for r in pair)}


def smoke(config,out,rows):
    path=out/'resource_smoke.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('status')=='COMPLETE':return saved
    chosen=[]
    for dataset in ['ActivityForensics','ANAKIN']:
        chosen.extend(next(v for v in pairs(rows,'train').values() if v[0]['dataset']==dataset))
    extract(config,out,chosen,lambda stage,**kw:progress(out,stage,scope='two_pair_smoke',**kw))
    fitted=fit(out,chosen,out/'smoke_statistics.json');records=[];initial=None
    for c in CONDITIONS:
        seed_all(SEED);m=LocalizationModel(c,fitted['statistics']).cuda();sha=parameters_sha(m)
        if initial is not None and sha!=initial:raise ValueError('UNPAIRED_INITIALIZATION')
        initial=sha;opt=torch.optim.AdamW(m.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',enabled=False) # BF16 autocast has no scaling; state still recorded.
        for dataset in ['ActivityForensics','ANAKIN']:
            pair=[r for r in chosen if r['dataset']==dataset];torch.cuda.reset_peak_memory_stats();m.train()
            before=parameters_sha(m);progress(out,'smoke_update',condition=c,dataset=dataset)
            update=update_pair(m,opt,pair,out,fitted,config,scaler)
            if parameters_sha(m)==before:raise ValueError('NO_ACTUAL_UPDATE')
            scored=evaluate(m,out,pair,fitted,config['chunk_frames'],config['amp'])
            records.append({'condition':c,'dataset':dataset,**update,'evaluation_s':scored['elapsed_s'],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'parameters':sum(p.numel() for p in m.parameters()),'updated':True})
        del m,opt;torch.cuda.empty_cache()
    result={'status':'COMPLETE','records':records,'initial_parameter_sha256':initial,'smoke_video_ids':[r['video_id'] for r in chosen],
            'smoke_weights':'discarded; formal models start from paired frozen initialization'}
    save(path,result);return result


def freeze(config,out,rows,smoked):
    path=out/'joint_manifest.json'
    if path.exists():return json.loads(path.read_text())['rows']
    rates={d:np.mean([r['elapsed_s']/r['frames'] for r in smoked['records'] if r['dataset']==d]) for d in ['ActivityForensics','ANAKIN']}
    eval_rates={d:np.mean([r['evaluation_s']/r['frames'] for r in smoked['records'] if r['dataset']==d]) for d in rates}
    quota=24;decisions=[]
    for candidate in [24,16,12]:
        chosen=[]
        for dataset in rates:
            pp=[v for v in pairs(rows,'train').values() if v[0]['dataset']==dataset]
            chosen.extend(r for pair in pp[:candidate] for r in pair)
        evaluation=[r for r in rows if r['split']=='dev']
        for dataset in rates:
            evaluation.extend(r for pair in [p for p in pairs(chosen,'train').values() if p[0]['dataset']==dataset][:4] for r in pair)
        train_s=sum(r['actual_frames']*rates[r['dataset']] for r in chosen)*config['epochs']*2
        eval_s=sum(r['actual_frames']*eval_rates[r['dataset']] for r in evaluation)*config['epochs']*2
        estimate=train_s+eval_s
        decisions.append({'train_pairs_per_dataset':candidate,'projected_training_s_two_conditions':train_s,
            'projected_routine_evaluation_s_two_conditions':eval_s,'projected_total_excluding_exact_pixel_AP_s':estimate,
            'extra_exact_pixel_AP_s':None,'pixel_AP_cost_reason':'not measured in resource smoke; actual logged at scheduled epochs and final'})
        quota=candidate
        if estimate<=config['max_projected_training_hours']*3600:break
    final=[*chosen,*[r for r in rows if r['split']=='dev']]
    train_pairs=pairs(final,'train');monitor=[]
    for dataset in rates:
        monitor.extend(r['video_id'] for pair in [v for v in train_pairs.values() if v[0]['dataset']==dataset][:4] for r in pair)
    for r in final:r['train_monitor']=r['video_id'] in monitor
    save(out/'budget_decision.json',{'smoke_based_projections':decisions,'chosen_quota':quota,'threshold_hours':config['max_projected_training_hours'],
        'rule':'only reduce train pairs synchronously; dev8 per dataset unchanged; no architecture/frame reduction'})
    save(path,{'schema':SCHEMA,'seed':SEED,'rows':final,'counts':dict(Counter(r['dataset']+':'+r['split'] for r in final)),
               'train_pairs_per_dataset':quota,'dev_pairs_per_dataset':8,'not_historical_queue':True})
    save(out/'visualization_selection.json',{'rule':'fixed decode fractions 25%,50%,75%; no prediction/GT-based selection',
        'rows':[{'video_id':r['video_id'],'indices':[int(r['actual_frames']*f) for f in [.25,.5,.75]]} for r in final if r['split']=='dev']})
    return final


def checkpoint_payload(model,opt,scaler,identity,epoch,pair_index,history,online,updates,best,best_epoch,patience_ref,stale,elapsed):
    return {'identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'amp':scaler.state_dict(),'amp_dtype':'bfloat16; backbone FP32',
        'rng':rng_state(),'epoch':epoch,'pair_index':pair_index,'history':history,'online':online,'updates':updates,
        'best':best,'best_epoch':best_epoch,'patience_ref':patience_ref,'stale':stale,'elapsed_s':elapsed}


def train(config,out,rows,fitted,resume):
    identity={'schema':SCHEMA,'manifest_sha256':digest(out/'joint_manifest.json'),'statistics_sha256':digest(out/'statistics.json'),
        'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'code_sha256':{str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('joint_data.py'),Path(__file__).with_name('model.py'),Path(__file__).with_name('metrics.py'),Path('research_tools/v8/full_video_training/model.py')]}}
    train_pairs=pairs(rows,'train');dev=[r for r in rows if r['split']=='dev'];monitor=[r for r in rows if r.get('train_monitor')]
    initial_path=out/'paired_initialization.pt';initial_sha=None
    for c in CONDITIONS:
        directory=out/c;directory.mkdir(exist_ok=True);summary_path=directory/'summary.json'
        if summary_path.exists() and json.loads(summary_path.read_text())['status']=='COMPLETE':
            if json.loads(summary_path.read_text())['identity']!=identity:raise ValueError('COMPLETE_RUN_IDENTITY_CHANGED')
            continue
        seed_all(SEED);model=LocalizationModel(c,fitted['statistics']).cuda();sha=parameters_sha(model)
        if initial_sha is not None and sha!=initial_sha:raise ValueError('UNPAIRED_PARAMETERS')
        initial_sha=sha
        if initial_path.exists():
            init=torch.load(initial_path,map_location='cpu',weights_only=False)
            if init['sha256']!=sha:raise ValueError('INITIALIZATION_CHANGED')
            model.load_state_dict(init['model'],strict=True)
            if parameters_sha(model)!=sha:raise ValueError('INITIALIZATION_PAYLOAD_CHANGED')
        else:atomic_torch(initial_path,{'sha256':sha,'model':model.state_dict(),'statistics_sha256':identity['statistics_sha256']})
        opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=False)
        epoch=0;pair_index=0;history=[];online=[];updates=0;best=-float('inf');best_epoch=None;patience_ref=-float('inf');stale=0;elapsed=0.
        cp_path=directory/'last.pt'
        if cp_path.exists():
            if not resume:raise ValueError('EXISTING_RUN_REQUIRES_RESUME')
            cp=torch.load(cp_path,map_location='cpu',weights_only=False)
            if cp['identity']!={**identity,'condition':c}:raise ValueError('RESUME_IDENTITY')
            model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['amp']);restore_rng(cp['rng'])
            epoch,pair_index,history,online,updates,best,best_epoch,patience_ref,stale,elapsed=(cp[k] for k in ['epoch','pair_index','history','online','updates','best','best_epoch','patience_ref','stale','elapsed_s'])
            if epoch!=len(history) or pair_index!=len(online) or epoch>config['epochs']:raise ValueError('RECOVERY_POSITION_INCONSISTENT')
        begin=time.monotonic();stop_reason='max_epochs'
        while epoch<config['epochs'] and stale<config['patience']:
            epoch_start=time.monotonic();prior_pair_s=sum(r['elapsed_s'] for r in online);torch.cuda.reset_peak_memory_stats();model.train();ordered=[]
            by_dataset={d:sorted([key for key,pair in train_pairs.items() if pair[0]['dataset']==d]) for d in ['ActivityForensics','ANAKIN']}
            rng=random.Random(SEED+epoch)
            for keys in by_dataset.values():rng.shuffle(keys)
            for af,an in zip(by_dataset['ActivityForensics'],by_dataset['ANAKIN']):ordered.extend([af,an])
            if len(ordered)!=len(train_pairs):raise ValueError('UNEQUAL_DATASET_QUOTA')
            for j in range(pair_index,len(ordered)):
                progress(out,'train',condition=c,epoch=epoch+1,pair=j+1,total_pairs=len(ordered))
                online.append(update_pair(model,opt,train_pairs[ordered[j]],out,fitted,config,scaler));updates+=1
                payload=checkpoint_payload(model,opt,scaler,{**identity,'condition':c},epoch,j+1,history,online,updates,best,best_epoch,patience_ref,stale,elapsed+time.monotonic()-begin)
                atomic_torch(cp_path,payload)
                if STOP:raise InterruptedError('SAFE_STOP_AT_PAIR_BOUNDARY; resume with --resume')
            progress(out,'evaluate',condition=c,epoch=epoch+1)
            tm=evaluate(model,out,monitor,fitted,config['chunk_frames'],config['amp'])
            dv=evaluate(model,out,dev,fitted,config['chunk_frames'],config['amp'],pixel_ap=(epoch+1)%5==0)
            dice=dv['datasets']['ANAKIN']['spatial_dice']
            if dice is None:raise ValueError('DEV_NO_POSITIVE_SPATIAL_GT')
            record={'condition':c,'epoch':epoch+1,'online_train':{k:float(np.mean([r[k] for r in online])) for k in ['loss','video','frame','pixel']},
                'train_monitor':tm['datasets'],'dev':dv['datasets'],'lr':opt.param_groups[0]['lr'],
                'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in online])),'updates':updates,
                'epoch_s':time.monotonic()-epoch_start+prior_pair_s,'epoch_time_convention':'active wall; resumed epoch adds measured prior pair time, excludes downtime',
                'training_pair_s':sum(r['elapsed_s'] for r in online),
                'evaluation_s':tm['elapsed_s']+dv['elapsed_s'],'frames':sum(r['frames'] for r in online),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            history.append(record);save(directory/'history.json',history);history_csv(directory/'history.csv',history)
            improved=dice>best
            if improved:best=dice;best_epoch=epoch+1
            if dice>patience_ref+config['min_delta']:patience_ref=dice;stale=0
            else:stale+=1
            epoch+=1;pair_index=0;online=[]
            payload=checkpoint_payload(model,opt,scaler,{**identity,'condition':c},epoch,0,history,online,updates,best,best_epoch,patience_ref,stale,elapsed+time.monotonic()-begin)
            atomic_torch(cp_path,payload)
            if improved:atomic_torch(directory/'best.pt',payload)
            print(f'{c} epoch={epoch:02d} online={record["online_train"]["loss"]:.6f} dev_dice={dice:.6f} video_auc_AF={dv["datasets"]["ActivityForensics"]["video"]["auroc"]:.6f} video_auc_AN={dv["datasets"]["ANAKIN"]["video"]["auroc"]:.6f} best={best:.6f}@{best_epoch} seconds={record["epoch_s"]:.1f}',flush=True)
        if stale>=config['patience']:stop_reason='early_stop_dev_dice_patience8_delta0.002'
        progress(out,'final_metrics',condition=c)
        last_metrics=evaluate(model,out,dev,fitted,config['chunk_frames'],config['amp'],pixel_ap=True)
        last_monitor=evaluate(model,out,monitor,fitted,config['chunk_frames'],config['amp'])
        best_cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best_cp['model'])
        best_metrics=evaluate(model,out,dev,fitted,config['chunk_frames'],config['amp'],pixel_ap=True,sample_dir=directory/'samples')
        best_monitor=evaluate(model,out,monitor,fitted,config['chunk_frames'],config['amp'])
        save(summary_path,{'status':'COMPLETE','best_epoch':best_epoch,'last_epoch':epoch,'stop_reason':stop_reason,
            'best_metrics':best_metrics,'last_metrics':last_metrics,'best_train_monitor':best_monitor,'last_train_monitor':last_monitor,
            'parameters':sum(p.numel() for p in model.parameters()),'initial_parameters_sha256':sha,
            'elapsed_s':elapsed+time.monotonic()-begin,'train_videos':sum(r['split']=='train' for r in rows),'dev_videos':len(dev),
            'finite_training':True,'updates':updates,'gradient_norms':'history.csv','identity':identity})
        del model,opt;torch.cuda.empty_cache()
        if c==CONDITIONS[0]:report(out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=['select','smoke','all'],default='all');args=parser.parse_args()
    config=json.loads(args.config.read_text());out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    torch.set_num_threads(4);os.environ.setdefault('TORCH_HOME',str(Path(config['workspace'])/'cache/torch'))
    os.environ.setdefault('MPLCONFIGDIR',str(Path(config['workspace'])/'cache/matplotlib'))
    try:
        progress(out,'selection');rows=select(config,out)
        if args.phase=='select':
            save(out/'progress.json',{'status':'SELECTED','stage':'candidate_queue_complete','pid':os.getpid()})
            return
        smoked=smoke(config,out,rows)
        if args.phase=='smoke':
            save(out/'progress.json',{'status':'SMOKE_COMPLETE','stage':'resource_smoke_complete','pid':os.getpid()})
            return
        rows=freeze(config,out,rows,smoked)
        preparation_start=time.monotonic()
        extract(config,out,rows,lambda stage,**kw:progress(out,stage,**kw))
        frontend_wall=time.monotonic()-preparation_start;fit_start=time.monotonic();statistics_hit=(out/'statistics.json').exists()
        progress(out,'fit_training_statistics');fitted=fit(out,rows,out/'statistics.json')
        old_costs=json.loads((out/'costs.json').read_text()) if (out/'costs.json').exists() else {}
        calls=old_costs.get('preparation_calls',[])+[{'frontend_call_wall_s':frontend_wall,'statistics_call_wall_s':time.monotonic()-fit_start,'statistics_cache_hit':statistics_hit}]
        save(out/'costs.json',{'preparation_calls':calls,'frontend_timed_stages_s':sum(json.loads((out/'cache'/r['video_id']/'complete.json').read_text())['cost'][key] for r in rows for key in ['geometry_s','rgb_s','tracking_s','features_s']),
            'frontend_timing_scope':'synchronized MoGe/ResNet/CoTracker calls plus feature construction; excludes decoding, mmap IO, startup and cache hash verification',
            'cache_io_and_decode_s':None,'unmeasured_reason':'not separately timed; do not report as zero',
            'cache_bytes':sum(p.stat().st_size for p in (out/'cache').rglob('*') if p.is_file()),'frame_count':sum(r['actual_frames'] for r in rows)})
        train(config,out,rows,fitted,args.resume);report(out)
        costs=json.loads((out/'costs.json').read_text())
        costs['training_evaluation_active_wall_s']={c:json.loads((out/c/'summary.json').read_text())['elapsed_s'] for c in CONDITIONS}
        costs['cache_bytes']=sum(p.stat().st_size for p in (out/'cache').rglob('*') if p.is_file())
        costs['final_output_logical_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        save(out/'costs.json',costs)
        with (out/'report_zh.md').open('a') as f:
            f.write(f'\n前端同步推理/特征阶段累计{costs["frontend_timed_stages_s"]:.1f}秒；正式准备调用墙钟累计{sum(c["frontend_call_wall_s"] for c in costs["preparation_calls"]):.1f}秒。\n')
            f.write(f'缓存{costs["cache_bytes"]/2**30:.3f}GiB；输出计量检查点{costs["final_output_logical_bytes"]/2**30:.3f}GiB；训练与评价活跃墙钟秒：{costs["training_evaluation_active_wall_s"]}。\n')
        save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'unix_time':time.time()})
    except BaseException as e:
        save(out/'progress.json',{'status':'INTERRUPTED' if isinstance(e,InterruptedError) else 'FAILED','stage':'execution',
            'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'pid':os.getpid(),'unix_time':time.time()})
        raise

if __name__=='__main__':main()
