"""One complete-video fine-tuning experiment; existing caches remain read-only."""
import argparse
import csv
import fcntl
import gc
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research_tools.v8.component_global_context_pilot.model import GlobalContextModel, CONDITIONS
from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file
from research_tools.v8.full_video_global_context.evaluate import evaluate, paired_pooled, SEEDS
from research_tools.v8.full_coverage_observation_field.pipeline import _metric
from research_tools.v8.local_geometry_pilot.data import digest
from research_tools.v8.local_geometry_pilot.runner import atomic_torch, seed_all, rng_state, restore_rng, csv_rows
from .data import audit, load_video
from .model import VideoTraining

STOP=False


def stop(signum, frame):
    global STOP
    STOP=True


def load_model(base, condition, seed, device='cuda'):
    records=read_json(base/'models/fold_models.json')['records']
    rr=[r for r in records if r['condition']==condition and r['seed']==seed]
    if len(rr)!=1:raise ValueError('PRETRAINED_MODEL_KEY')
    r=rr[0];p=Path(r['checkpoint'])
    if r['epoch']!=100 or r['status']!='COMPLETE' or sha256_file(p)!=r['checkpoint_sha256']:
        raise ValueError('PRETRAINED_CHECKPOINT_IDENTITY')
    cp=torch.load(p,map_location='cpu',weights_only=False)
    if (cp['identity']!=r['identity'] or cp['epoch']!=100 or
        cp['identity']['condition']!=condition or cp['identity']['seed']!=seed or
        cp['identity']['statistics_sha256']!=sha256_file(base/'standardizer.json')):
        raise ValueError('PRETRAINED_PAYLOAD_IDENTITY')
    model=GlobalContextModel(condition,read_json(base/'standardizer.json')).to(device)
    model.load_state_dict(cp['model'],strict=True)
    if sum(p.numel() for p in model.parameters())!=319969:raise ValueError('PARAMETER_COUNT')
    for k,v in model.state_dict().items():
        if not torch.equal(v.cpu(),cp['model'][k].cpu()):raise ValueError('INITIAL_PARAMETER_MISMATCH')
    return model,r


def step_video(model,row,identity,chunk):
    cpu,read_s=load_video(row,identity)
    torch.cuda.synchronize();t=time.monotonic()
    z=VideoTraining(model).video(cpu,chunk)
    loss=F.binary_cross_entropy_with_logits(z,torch.tensor(float(row['video_label']),device='cuda',dtype=z.dtype))*.5
    torch.cuda.synchronize();forward=time.monotonic()-t;t=time.monotonic()
    loss.backward()
    torch.cuda.synchronize();backward=time.monotonic()-t
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('NONFINITE_VIDEO_LOSS_OR_GRADIENT')
    result={'video_id':row['video_id'],'source':row['source_id'],'role':row['role'],'frames':row['actual_frames'],
            'read_s':read_s,'forward_s':forward,'backward_s':backward,'loss':float(loss.detach()),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    del cpu,z,loss
    return result


def smoke(out,base,rows,identity,progress):
    train=[r for r in rows if r['split']=='train']
    pair=sorted([r for r in train if r['source_id']=='1LARL'],key=lambda r:r['role'])
    if len(pair)!=2:raise ValueError('SMOKE_SOURCE_PAIR')
    longest=max(train,key=lambda r:(r['actual_frames'],r['video_id']))
    samples=pair+([] if longest['video_id'] in {r['video_id'] for r in pair} else [longest])
    results=[]
    for chunk in (32,64):
        seed_all(SEEDS[0]);model,_=load_model(base,CONDITIONS[0],SEEDS[0]);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        try:
            records=[]
            for i,row in enumerate(samples):
                progress('resource_smoke',i,len(samples),chunk=chunk,video=row['video_id'])
                records.append(step_video(model,row,identity,chunk))
                if i==1 or i==len(samples)-1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            result={'chunk':chunk,'status':'COMPLETE','elapsed_s':time.monotonic()-t,'videos':records}
            result['pair_s']=sum(sum(r[k] for k in ('read_s','forward_s','backward_s')) for r in records[:2])
        except torch.cuda.OutOfMemoryError as e:
            result={'chunk':chunk,'status':'CUDA_OOM','error':str(e),'elapsed_s':time.monotonic()-t,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'traceback':traceback.format_exc()}
        results.append(result);save_json(out/'resource_smoke.json',{'records':results,'longest_training_video':longest['video_id']})
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
    good=[r for r in results if r['status']=='COMPLETE']
    if not good:
        # A smaller execution chunk is permitted, never detach or shorten the video.
        chunk=16;model,_=load_model(base,CONDITIONS[0],SEEDS[0]);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        try:
            records=[]
            for i,r in enumerate(samples):
                records.append(step_video(model,r,identity,chunk))
                if i==1 or i==len(samples)-1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            result={'chunk':chunk,'status':'COMPLETE','elapsed_s':time.monotonic()-t,'videos':records,
                    'pair_s':sum(sum(r[k] for k in ('read_s','forward_s','backward_s')) for r in records[:2])}
        except torch.cuda.OutOfMemoryError as e:
            result={'chunk':chunk,'status':'CUDA_OOM','elapsed_s':time.monotonic()-t,'error':str(e),
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'traceback':traceback.format_exc()}
        results.append(result);save_json(out/'resource_smoke.json',{'records':results,'longest_training_video':longest['video_id']})
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        good=[r for r in results if r['status']=='COMPLETE']
    if not good:raise RuntimeError('FULL_VIDEO_GRADIENT_RESOURCE_BLOCKED; see resource_smoke.json')
    chosen=min(good,key=lambda r:r['pair_s'])
    save_json(out/'execution_config.json',{'chunk':chosen['chunk'],'selection':'measured pair throughput; full longest-video backward passed',
               'projected_training_s':chosen['pair_s']*122*20*6,'resource_smoke':results})
    return chosen['chunk']


def reproduce(out,base,cache,rows,identity,chunk,progress):
    reference={}
    with (cache/'evaluation/video_scores.csv').open() as f:
        for r in csv.DictReader(f):reference[(r['condition'],str(r['seed']),r['video_id'])]=float(r['logit'])
    records=[];values={}
    val=[r for r in rows if r['split']=='validation']
    for c in CONDITIONS:
        for seed in SEEDS:
            model,record=load_model(base,c,seed);model.eval()
            for r in val:
                cpu,_=load_video(r,identity)
                with torch.no_grad():z=float(VideoTraining(model).video(cpu,chunk,use_checkpoint=False))
                expected=reference[(c,str(seed),r['video_id'])];error=abs(z-expected)
                records.append({'condition':c,'seed':seed,'video_id':r['video_id'],'expected':expected,'actual':z,'absolute_error':error})
                values[(c,seed,r['video_id'])]=z
                csv_rows(out/'initial_score_reproduction.csv',records)
                if error>2e-5:raise ValueError(f'INITIAL_SCORE_REPRODUCTION:{c}:{seed}:{r["video_id"]}:{error}')
                progress('initial_reproduction',len(records),168,condition=c,seed=seed,video=r['video_id'])
                del cpu
            del model
        for r in val:
            np.testing.assert_allclose(np.mean([values[(c,s,r['video_id'])] for s in SEEDS]),
                                       reference[(c,'MEAN_VIDEO_LOGIT',r['video_id'])],atol=2e-5,rtol=0)
    save_json(out/'initial_reproduction.json',{'status':'COMPLETE','models':6,'videos':28,'scores':168,
              'max_absolute_error':max(r['absolute_error'] for r in records),'tolerance':2e-5})


def train(out,base,rows,identity,chunk,run_identity,progress,resume,conditions=None,epochs=20):
    conditions=CONDITIONS if conditions is None else tuple(conditions)
    planned=len(conditions)*len(SEEDS)
    sources=sorted({r['source_id'] for r in rows if r['split']=='train'})
    pairs={s:sorted([r for r in rows if r['split']=='train' and r['source_id']==s],key=lambda r:r['role']) for s in sources}
    directory=out/'models';directory.mkdir(exist_ok=True);records=[]
    for c in conditions:
        for seed in SEEDS:
            key=f'{c}__seed{seed}';final=directory/f'{key}.pt';recovery=directory/f'{key}.recovery.pt'
            model,initial=load_model(base,c,seed);model.train();seed_all(seed)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            order_rng=torch.Generator().manual_seed(seed)
            start_epoch=0;pair_index=0;order=None;loss_history=[];epoch_loss=0.;elapsed=0.
            expected={'run_identity':run_identity,'condition':c,'seed':seed,'initial_sha256':initial['checkpoint_sha256']}
            if final.exists() or recovery.exists():
                if not resume:raise ValueError('EXISTING_TRAINING_REQUIRES_RESUME')
                cp=torch.load(final if final.exists() else recovery,map_location='cpu',weights_only=False)
                if cp['identity']!=expected:raise ValueError('TRAINING_RECOVERY_IDENTITY')
                model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer'])
                restore_rng(cp['rng']);order_rng.set_state(cp['order_rng']);start_epoch=cp['epoch'];pair_index=cp['pair_index']
                order=cp['order'];loss_history=cp['loss_history'];epoch_loss=cp['epoch_loss'];elapsed=cp['training_elapsed_s']
            clock=time.monotonic()
            if start_epoch>epochs or (start_epoch==epochs and pair_index):raise ValueError('CHECKPOINT_EXCEEDS_TARGET_EPOCH')
            for epoch in range(start_epoch,epochs):
                if pair_index==0:order=torch.randperm(len(sources),generator=order_rng).tolist()
                for j in range(pair_index,len(sources)):
                    source=sources[order[j]];optimizer.zero_grad(set_to_none=True);loss=0.
                    for row in pairs[source]:
                        progress('train',len(records),planned,condition=c,seed=seed,epoch=epoch+1,target_epochs=epochs,pair=j+1,pairs=len(sources),video=row['video_id'])
                        detail=step_video(model,row,identity,chunk);loss+=detail['loss']
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();epoch_loss+=loss
                    next_epoch=epoch;next_pair=j+1
                    if next_pair==len(sources):
                        loss_history.append(epoch_loss/len(sources));epoch_loss=0.;next_epoch=epoch+1;next_pair=0
                    payload={'identity':expected,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':next_epoch,
                             'pair_index':next_pair,'order':order,'loss_history':loss_history,'epoch_loss':epoch_loss,
                             'rng':rng_state(),'order_rng':order_rng.get_state(),'training_elapsed_s':elapsed+time.monotonic()-clock,
                             'parameters':319969,'chunk':chunk}
                    atomic_torch(recovery,payload)
                    save_json(directory/f'{key}.loss.json',{'loss_history':loss_history,'epoch':next_epoch,'pair_index':next_pair})
                    if STOP:raise InterruptedError('SAFE_TERM_AFTER_SOURCE_PAIR')
                pair_index=0
            payload=torch.load(final if final.exists() else recovery,map_location='cpu',weights_only=False)
            if payload['epoch']!=epochs or payload['pair_index']!=0 or len(payload['loss_history'])!=epochs:
                raise ValueError('FINAL_EPOCH_INCOMPLETE')
            atomic_torch(final,payload)
            records.append({'condition':c,'seed':seed,'status':'COMPLETE','epoch':epochs,'checkpoint':str(final),
                            'checkpoint_sha256':sha256_file(final),'training_elapsed_s':payload['training_elapsed_s'],
                            'loss_history':payload['loss_history'],'parameter_count':319969,'identity':expected})
            save_json(directory/'fold_models.json',{'records':records})
            del optimizer,model;gc.collect();torch.cuda.empty_cache()
    return records


def score(out,base,cache,rows,identity,chunk,records,progress):
    conditions=tuple(dict.fromkeys(r['condition'] for r in records))
    if {(r['condition'],r['seed']) for r in records}!={(c,s) for c in conditions for s in SEEDS}:
        raise ValueError('INCOMPLETE_SCORING_MODEL_SET')
    path=out/'scores.json';saved=read_json(path) if path.exists() else {r['video_id']:{'video_logits':{}} for r in rows}
    for record in records:
        c,seed=record['condition'],record['seed'];key=f'{c}__seed{seed}'
        model,_=load_model(base,c,seed);model.load_state_dict(torch.load(record['checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True);model.eval()
        for i,r in enumerate(rows):
            if key not in saved[r['video_id']]['video_logits']:
                cpu,_=load_video(r,identity)
                with torch.no_grad():z=float(VideoTraining(model).video(cpu,chunk,use_checkpoint=False))
                if not np.isfinite(z):raise ValueError('NONFINITE_FINAL_SCORE')
                saved[r['video_id']]['video_logits'][key]=z;save_json(path,saved);del cpu
            progress('score',i+1,len(rows),condition=c,seed=seed)
        del model
    result=evaluate(out,rows,[saved[r['video_id']] for r in rows],conditions=conditions)
    result['primary']=('pooled video AUROC; fine-tuned COMPONENT minus frozen COMPONENT' if len(conditions)==1 else
                       'pooled video AUROC; fine-tuned COMPONENT minus fine-tuned CELL')
    with (cache/'evaluation/video_scores.csv').open() as f:
        old_rows=list(csv.DictReader(f))
    old={(r['condition'],r['video_id']):float(r['logit']) for r in old_rows if r['seed']=='MEAN_VIDEO_LOGIT'}
    original={(r['condition'],r['seed'],r['video_id']):r for r in old_rows}
    for r in rows:
        for c in conditions:
            for seed in [*map(str,SEEDS),'MEAN_VIDEO_LOGIT']:
                old_row=original[(c,seed,r['video_id'])]
                if old_row['source']!=r['source_id'] or int(old_row['label'])!=r['video_label'] or old_row['split']!=r['split']:
                    raise ValueError('FROZEN_SCORE_VIDEO_IDENTITY')
    val=[r for r in rows if r['split']=='validation'];vv=[{**r,'label':r['video_label']} for r in val]
    result['post_minus_frozen']={c:paired_pooled(vv,np.array([np.mean([saved[r['video_id']]['video_logits'][f'{c}__seed{s}'] for s in SEEDS]) for r in val]),
                                                   np.array([old[(c,r['video_id'])] for r in val])) for c in conditions}
    result['frozen_metrics']=[]
    for c in conditions:
        for split in ('train','validation'):
            rr=[r for r in rows if r['split']==split];yy=np.array([r['video_label'] for r in rr])
            for seed in [*map(str,SEEDS),'MEAN_VIDEO_LOGIT']:
                zz=np.array([float(original[(c,seed,r['video_id'])]['logit']) for r in rr])
                result['frozen_metrics'].append({'condition':c,'split':split,'seed':seed,**_metric(yy,zz)})
    result['seed_differences']=[]
    for seed in SEEDS:
        mm={r['condition']:r for r in result['metrics'] if r['split']=='validation' and r['seed']==seed}
        comparator=(mm[conditions[1]] if len(conditions)==2 else next(r for r in result['frozen_metrics'] if
                    r['condition']==conditions[0] and r['split']=='validation' and r['seed']==str(seed)))
        result['seed_differences'].append({'seed':seed,**{k:mm[conditions[0]][k]-comparator[k] for k in ('auroc','ap','f1','accuracy')}})
    if len(conditions)==1:
        result['paired_comparison']=result['post_minus_frozen'][conditions[0]]
        per_source=[]
        for source in sorted({r['source_id'] for r in val}):
            pair=sorted([r for r in val if r['source_id']==source],key=lambda r:r['video_label'])
            z=np.array([np.mean([saved[r['video_id']]['video_logits'][f'{conditions[0]}__seed{s}'] for s in SEEDS]) for r in pair])
            oz=np.array([old[(conditions[0],r['video_id'])] for r in pair])
            a=float((z[1]>z[0])+.5*(z[1]==z[0]));b=float((oz[1]>oz[0])+.5*(oz[1]==oz[0]))
            per_source.append({'source':source,'fine_tuned_pair_auroc':a,'frozen_pair_auroc':b,'difference':a-b,'n_real':1,'n_fake':1})
        csv_rows(out/'evaluation/per_source_differences.csv',per_source)
        result['source_directions']={k:sum((r['difference']>0 if k=='positive' else r['difference']<0 if k=='negative' else r['difference']==0) for r in per_source) for k in ('positive','tie','negative')}
    save_json(out/'evaluation/summary.json',result)
    return result


def migrate_scope(source,out,protocol,base,conditions,epochs):
    """Explicitly change only model scope/target epochs, keeping committed training state."""
    previous=read_json(source/'protocol.json')
    if {k:v for k,v in previous.items() if k not in ('conditions','epochs')}!={k:v for k,v in protocol.items() if k not in ('conditions','epochs')}:
        raise ValueError('SCOPE_MIGRATION_CHANGED_COMPUTATION_OR_INPUT')
    if read_json(source/'execution_config.json')['chunk']!=64:raise ValueError('SCOPE_MIGRATION_CHUNK')
    if read_json(source/'final_status.json')['status']!='PAUSED_SCOPE_CHANGE':raise ValueError('SOURCE_NOT_PAUSED_FOR_SCOPE_CHANGE')
    directory=out/'models';directory.mkdir(exist_ok=True)
    if (out/'scope_migration.json').exists():
        saved=read_json(out/'scope_migration.json')
        if saved['old_protocol_sha256']!=sha256_file(source/'protocol.json') or saved['new_protocol_sha256']!=sha256_file(out/'protocol.json'):
            raise ValueError('MIGRATION_PROVENANCE_CHANGED')
        return
    migrated=[]
    for c in conditions:
        for seed in SEEDS:
            path=source/'models'/f'{c}__seed{seed}.recovery.pt'
            if not path.exists():continue
            old_hash=sha256_file(path);cp=torch.load(path,map_location='cpu',weights_only=False)
            _,initial=load_model(base,c,seed,device='cpu')
            expected={'run_identity':digest(previous),'condition':c,'seed':seed,'initial_sha256':initial['checkpoint_sha256']}
            if cp['identity']!=expected or cp['chunk']!=64 or cp['parameters']!=319969:
                raise ValueError('MIGRATION_CHECKPOINT_IDENTITY')
            if cp['epoch']>epochs or (cp['epoch']==epochs and cp['pair_index']):raise ValueError('MIGRATION_PROGRESS_EXCEEDS_TARGET')
            if len(cp['loss_history'])!=cp['epoch'] or not 0<=cp['pair_index']<122:raise ValueError('MIGRATION_PROGRESS_FIELDS')
            if sorted(cp['order'])!=list(range(122)) or not np.isfinite(cp['loss_history']+[cp['epoch_loss'],cp['training_elapsed_s']]).all():
                raise ValueError('MIGRATION_ORDER_OR_LOSS')
            for group in cp['optimizer']['param_groups']:
                if group['lr']!=protocol['lr'] or group['weight_decay']!=protocol['weight_decay'] or tuple(group['betas'])!=(.9,.999):
                    raise ValueError('MIGRATION_OPTIMIZER')
            cp['identity']={**expected,'run_identity':digest(protocol)}
            target=directory/path.name
            if target.exists():raise ValueError('UNRECORDED_MIGRATION_TARGET_EXISTS')
            atomic_torch(target,cp)
            migrated.append({'source':str(path),'source_sha256':old_hash,'target':str(target),'target_sha256':sha256_file(target),
                             'epoch':cp['epoch'],'next_pair_index':cp['pair_index'],'training_elapsed_s':cp['training_elapsed_s'],
                             'preserved':'model, optimizer, RNG, order_rng, order, loss_history, epoch_loss, epoch, pair_index, elapsed'})
    save_json(out/'scope_migration.json',{'source_directory':str(source),'old_protocol_sha256':sha256_file(source/'protocol.json'),
              'new_protocol_sha256':sha256_file(out/'protocol.json'),'changes':{'conditions':[previous['conditions'],list(conditions)],'epochs':[previous['epochs'],epochs]},
              'checkpoints':migrated,'migrated_training_elapsed_s':sum(r['training_elapsed_s'] for r in migrated)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--pretrained',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS));parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--migrate-from',type=Path);args=parser.parse_args()
    if (tuple(args.conditions),args.epochs) not in ((CONDITIONS,20),(('GLOBAL_COMPONENT',),5)):
        raise ValueError('ONLY_AUTHORIZED_ORIGINAL_OR_BUDGET5_SCOPE')
    planned=len(args.conditions)*len(SEEDS)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.time();phase='audit'
    def progress(stage,completed,total,**extra):
        nonlocal phase
        phase=stage;value={'status':'RUNNING','stage':stage,'completed':completed,'total':total,'updated_unix':time.time(),'pid':os.getpid(),**extra}
        save_json(out/'progress.json',value);print(value,flush=True)
    try:
        if (out/'final_status.json').exists():
            history=read_json(out/'run_history.json') if (out/'run_history.json').exists() else []
            history.append(read_json(out/'final_status.json'));save_json(out/'run_history.json',history)
        save_json(out/'final_status.json',{'status':'RUNNING','pid':os.getpid(),'started_unix':started})
        print('CUDA',torch.cuda.get_device_name(),flush=True)
        if args.migrate_from and not (out/'input_manifest.json').exists():
            save_json(out/'input_manifest.json',read_json(args.migrate_from/'input_manifest.json'))
            if sha256_file(out/'input_manifest.json')!=sha256_file(args.migrate_from/'input_manifest.json'):
                raise ValueError('MIGRATION_MANIFEST_CONTENT')
        if (out/'input_manifest.json').exists() and args.resume:
            saved=read_json(out/'input_manifest.json');rows=saved['rows'];cache_protocol=read_json(args.cache/'protocol.json')
            if saved['cache_protocol_sha256']!=sha256_file(args.cache/'protocol.json'):raise ValueError('CACHE_PROTOCOL_CHANGED')
        else:
            rows,cache_protocol=audit(args.cache)
            save_json(out/'input_manifest.json',{'rows':rows,'cache_protocol_sha256':sha256_file(args.cache/'protocol.json')})
        protocol={'cache_identity':cache_protocol['identity'],'input_manifest_sha256':sha256_file(out/'input_manifest.json'),
                  'standardizer_sha256':sha256_file(args.pretrained/'standardizer.json'),'grouping_sha256':sha256_file(args.pretrained/'grouping_config.json'),
                  'pretrained_models_sha256':sha256_file(args.pretrained/'models/fold_models.json'),
                  'conditions':args.conditions,'seeds':list(SEEDS),'epochs':args.epochs,'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,
                  'clip':1.,'effective_batch':2,'checkpoint':'non-reentrant; full cross-chunk gradients','threshold':0,'precision':'FP32; TF32 disabled',
                  'loss':'one full-video BCE per video; half real plus half fake per source pair; area LSE beta1'}
        if (out/'protocol.json').exists() and read_json(out/'protocol.json')!=protocol:raise ValueError('PROTOCOL_CHANGED')
        save_json(out/'protocol.json',protocol)
        identity=cache_protocol['identity'];run_identity=digest(protocol)
        for filename,expected_hash in cache_protocol['frozen_files'].items():
            p=Path(filename)
            if p.is_relative_to(args.pretrained) and sha256_file(p)!=expected_hash:
                raise ValueError(f'FROZEN_PRETRAINED_ASSET_CHANGED:{p}')
        if args.migrate_from:
            migrate_scope(args.migrate_from,out,protocol,args.pretrained,args.conditions,args.epochs)
            for name in ('execution_config.json','initial_reproduction.json'):
                if not (out/name).exists():
                    original=read_json(args.migrate_from/name)
                    if name=='execution_config.json':
                        original={**original,'original_projected_training_s':original.get('projected_training_s'),
                                  'projected_training_s':original['projected_training_s']*planned*args.epochs/(6*20)}
                    save_json(out/name,{**original,'reused_from':str(args.migrate_from/name),'reference_sha256':sha256_file(args.migrate_from/name),'rerun':False})
        chunk=read_json(out/'execution_config.json')['chunk'] if (out/'execution_config.json').exists() else smoke(out,args.pretrained,rows,identity,progress)
        if not (out/'initial_reproduction.json').exists():reproduce(out,args.pretrained,args.cache,rows,identity,chunk,progress)
        train_start=time.monotonic()
        records=train(out,args.pretrained,rows,identity,chunk,run_identity,progress,args.resume,conditions=args.conditions,epochs=args.epochs)
        training_call_s=time.monotonic()-train_start;evaluation_start=time.monotonic()
        result=score(out,args.pretrained,args.cache,rows,identity,chunk,records,progress)
        evaluation_s=time.monotonic()-evaluation_start
        lines=['# V8 全视频级弱监督微调','', f'两阶段训练：匹配窗口预训练 epoch100 → 完整连续视频微调 epoch{args.epochs}。',
               '',f'训练244视频/122 source；验证28视频/14 source；共234284实际连续帧。{len(records)}/{planned}正式模型完成，参数各319969。chunk={chunk}。',
               '', '三个 seed 先各自全视频面积 LSE，再平均视频 logit；阈值固定0。', '',
               '|集合|条件|AUROC|AP|Precision|Recall|F1|ACC|TN/FP/FN/TP|','|---|---|---|---|---|---|---|---|---|']
        for m in result['metrics']:
            if m['seed']=='MEAN_VIDEO_LOGIT':lines.append('|'+ '|'.join(str(m[k]) for k in ('split','condition','auroc','ap','precision','recall','f1','accuracy'))+f"|{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}|")
        lines+=['','主要配对（source bootstrap 10000次，seed20260909）：',str(result['paired_comparison']),
                '', '逐seed验证主要比较差值：',str(result['seed_differences']),
                '', '逐source真假排序方向：',str(result['source_directions']),
                '', '相对冻结模型：',str(result['post_minus_frozen']), '',
                '冻结验证参照：COMPONENT AUROC 0.826530612244898，AP 0.8579976104660978，TN/FP/FN/TP=4/10/1/13；'
                'CELL AUROC 0.7142857142857143，AP 0.7079237938207612，TN/FP/FN/TP=5/9/2/12。阈值保持0，不因误报调参。', '',
                '开发验证集重复使用；并非 sealed-test。仅视频级弱监督与评价，不证明帧级/空间定位或独立三维机制。全部合法内容格保留，未重跑前端。',
                '后续建议只作讨论：结合排序、误报/漏报与seed/source一致性决定是否值得独立source测试；不自动执行或追加消融。',
                '',f'本次调用耗时：{time.time()-started:.3f}秒；完整模型训练耗时：{sum(r["training_elapsed_s"] for r in records):.3f}秒。']
        lines+=['', '资源配置与实测：', str(read_json(out/'execution_config.json')),
                '', '初始化复现：',str(read_json(out/'initial_reproduction.json')),
                '', '完整逐seed指标见evaluation/metrics.csv，逐source见evaluation/per_source_metrics.csv。模型loss与恢复记录保留models/。']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        if len(records)!=planned or any(r['epoch']!=args.epochs for r in records):raise ValueError('FINAL_MODEL_SCOPE_INCOMPLETE')
        if args.epochs==5:
            migration=read_json(out/'scope_migration.json')
            lines+=['','## 范围收缩与进度复用',
                    '为降低时间成本，按用户预先指定范围仅适配主方法三个seed、固定5轮，不根据验证表现选择。原六模型20轮任务已暂停并保留。',
                    '未改变全部视频/帧、模型、视频BCE、source pair等权、跨块完整梯度、统计或数值设置。CELL未微调，只能列为历史冻结参考，不是匹配消融。',
                    str(migration),
                    f'总训练实测含迁入：{sum(r["training_elapsed_s"] for r in records):.3f}秒；迁入已用：{migration["migrated_training_elapsed_s"]:.3f}秒；'
                    f'迁移后新增训练：{sum(r["training_elapsed_s"] for r in records)-migration["migrated_training_elapsed_s"]:.3f}秒。',
                    '', '冻结与微调逐seed/训练验证指标：',str(result['frozen_metrics'])]
            post=next(m for m in result['metrics'] if m['split']=='validation' and m['seed']=='MEAN_VIDEO_LOGIT')
            old=next(m for m in result['frozen_metrics'] if m['split']=='validation' and m['seed']=='MEAN_VIDEO_LOGIT')
            lines+=['',f'固定阈值误报FP：{old["fp"]}→{post["fp"]}；漏报FN：{old["fn"]}→{post["fn"]}；Recall：{old["recall"]}→{post["recall"]}。',
                    '低成本适配是否值得保留需同时考虑排序、AP和FP/FN；性能下降也不延长训练。完整视频处理/微调不等于独立泛化验证。']
            lines+=['',f'本次训练调用墙钟：{training_call_s:.3f}秒；评分/评价：{evaluation_s:.3f}秒；'
                    f'本次进程GPU峰值allocated：{torch.cuda.max_memory_allocated()/2**30:.6f}GiB；reserved：{torch.cuda.max_memory_reserved()/2**30:.6f}GiB。',
                    f'本目录新增存储（报告生成时）：{sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}字节；旧连续帧缓存只读引用，没有复制。']
            (out/'report.md').write_text('\n'.join(lines)+'\n')
        save_json(out/'final_status.json',{'status':'COMPLETE','models':len(records),'planned_models':planned,'epochs':args.epochs,'videos':272,'frames':234284,
                  'training_elapsed_s':sum(r['training_elapsed_s'] for r in records),'training_call_s':training_call_s,'evaluation_s':evaluation_s,
                  'elapsed_s':time.time()-started,'updated_unix':time.time()})
        progress('report',1,1);save_json(out/'progress.json',{'status':'COMPLETE','stage':'report','completed':1,'total':1,'updated_unix':time.time()})
    except BaseException as e:
        status='STOPPED_SAFE' if isinstance(e,InterruptedError) else 'RESOURCE_BLOCKED' if 'RESOURCE_BLOCKED' in str(e) else 'FAILED'
        save_json(out/'final_status.json',{'status':status,'phase':phase,'error':str(e),'traceback':traceback.format_exc(),'elapsed_s':time.time()-started,'updated_unix':time.time()})
        save_json(out/'progress.json',{'status':status,'stage':phase,'updated_unix':time.time(),'error':str(e)})
        raise


if __name__=='__main__':main()
