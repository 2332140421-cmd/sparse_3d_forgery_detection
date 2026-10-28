"""One research run: pinned assets -> shared frozen features -> two spatial models."""
import argparse
import copy
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

from research_tools.v8.local_geometry_pilot.runner import seed_all,atomic_torch,restore_rng
from research_tools.v8.spatial_localization.run import checkpoint_payload
from research_tools.v8.spatial_localization.metrics import history_csv
from research_tools.v8.spatial_localization.joint_data import save,SEED
from .assets import prepare,official,extract,patch_features,CachedVideo,verify_source
from .model import SpatialModel,CONDITIONS,video_loss
from .metrics import evaluate,curves,report

STOP=False
LAST_STAGE=None


def stop(signum,frame):
    global STOP
    STOP=True


def progress(out,stage,**kw):
    global LAST_STAGE
    if STOP and stage=='features':raise InterruptedError('FEATURE_VIDEO_RESTART_ON_RESUME')
    save(out/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'unix_time':time.time(),**kw})
    if stage!=LAST_STAGE or (stage=='features' and kw.get('frames')==kw.get('total_frames')):
        print(json.dumps({'stage':stage,**kw}),flush=True)
    LAST_STAGE=stage


def pair_update(model,opt,pair,config,out,fit,scaler):
    opt.zero_grad(set_to_none=True);losses={k:0. for k in ['loss','bce','dice']};begin=time.monotonic()
    for r in sorted(pair,key=lambda x:x['role']):
        v=CachedVideo(config,out,r,fit,model.context)
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                loss,parts=video_loss(model,v,config['chunk_frames'],True)
            if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAIN_LOSS')
            scaler.scale(loss*.5).backward()
            for k in losses:losses[k]+=float(parts[k].detach())*.5
        finally:v.close()
    scaler.unscale_(opt);norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
    paths={}
    if model.context:
        for name,p in [('geometry',model.geo1.weight),('gru',model.structure.gru.weight_ih),('projection',model.projection[-1].weight)]:
            paths[name]=float(p.grad.abs().sum()) if p.grad is not None else None
    scaler.step(opt);scaler.update();torch.cuda.synchronize()
    return {**losses,'gradient_norm':norm,'branch_grad_abs_sum':paths,'seconds':time.monotonic()-begin,'frames':sum(r['actual_frames'] for r in pair)}


@torch.no_grad()
def verify_online(config,out,rows,fit,network,use_cache=True):
    """Official forward, no edge crop, cache quantization, and zero-correction parity."""
    candidates=[]
    for portrait in [False,True]:
        first=next(r for r in rows if (r['source_hw'][0]>r['source_hw'][1])==portrait)
        candidates.extend(r for r in rows if r['pair_id']==first['pair_id'])
    decoder=network.encoder.localization_head;results=[];seed_all(SEED)
    a=SpatialModel(CONDITIONS[0],decoder,fit['statistics']).cuda().eval()
    seed_all(SEED);b=SpatialModel(CONDITIONS[1],decoder,fit['statistics']).cuda().eval()
    frozen_ids={id(p) for p in network.backbone.parameters()}
    if network.backbone.training or any(p.requires_grad for p in network.backbone.parameters()):raise ValueError('VISUAL_BACKBONE_NOT_FROZEN_EVAL')
    if any(id(p) in frozen_ids for m in [a,b] for p in m.parameters() if p.requires_grad):raise ValueError('FROZEN_BACKBONE_IN_OPTIMIZER')
    for (ka,pa),(kb,pb) in zip(a.decoder.named_parameters(),b.decoder.named_parameters()):
        if ka!=kb or not torch.equal(pa,pb):raise ValueError('UNPAIRED_PRETRAINED_DECODER')
    for r in candidates:
        from research_tools.v8.spatial_localization.joint_data import Video
        v=CachedVideo(config,out,r,fit,True) if use_cache else Video(Path(config['source_run']),r,fit['statistics'],fit['grouping'],read_only=True)
        try:
            i=r['actual_frames']//2;cpu=v.chunk(i,i+1)
            base=v.base if use_cache else v
            rgb=torch.from_numpy(np.array(base.arrays['rgb512'][i:i+1],copy=True)).cuda()
            from .assets import normalize518
            captured=[]
            hook=decoder.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
            try:
                x=normalize518(rgb);full=network(x[:,None],return_localization=True)[2][:,0,0,:512,:512]
            finally:hook.remove()
            same_forward=decoder(captured[0],target_size=(518,518))[:,0,:512,:512]
            torch.testing.assert_close(full,same_forward,atol=1e-5,rtol=1e-5)
            features=patch_features(network,rgb);online=decoder(features,target_size=(518,518))[:,0,:512,:512]
            if not torch.isfinite(full).all():raise ValueError('NONFINITE_OFFICIAL_LOCALIZATION')
            # Repeated CUDA SDP forward has small FP32 roundoff; compare captured
            # official tokens separately so this tolerance cannot hide a wrong interface.
            torch.testing.assert_close(captured[0],features,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(full,online,atol=1e-3,rtol=1e-4)
            if not use_cache:cpu['features']=features.half().float().cpu()
            cached=decoder(cpu['features'].cuda(),target_size=(518,518))[:,0,:512,:512]
            content=cpu['content512'].cuda();error=(online-cached).abs()[content]
            flip=float(((online>=0)!=(cached>=0))[content].float().mean())
            if float(error.mean())>.01 or float(error.max())>.1 or flip>.005:raise ValueError('FP16_CACHE_QUANTIZATION_EXCEEDS_FROZEN_LIMITS')
            h=torch.zeros(256,128,device='cuda');active=torch.zeros(256,dtype=torch.bool,device='cuda')
            za=a.chunk(cpu,h,active,True,r['mapping256'],r['mapping512'])[2]
            zb=b.chunk(cpu,h,active,True,r['mapping256'],r['mapping512'])[2]
            torch.testing.assert_close(za,zb,atol=1e-6,rtol=1e-6)
            results.append({'video_id':r['video_id'],'source_hw':r['source_hw'],'quantization_mean_abs':float(error.mean()),
                'quantization_max_abs':float(error.max()),'threshold_flip_fraction':flip,'AB_max_abs':float((za-zb).abs().max()),
                'same_official_tokens_decode_max_abs':float((full-same_forward).abs().max()),
                'repeated_backbone_features_max_abs':float((captured[0]-features).abs().max()),
                'repeated_backbone_logits_max_abs':float((full-online).abs().max()),
                'official_forward_vs_feature_decode':'PASS','mapping256':r['mapping256'],'mapping512':r['mapping512']})
        finally:v.close()
    save(out/('online_cache_parity.json' if use_cache else 'official_forward_probe.json'),results);del a,b;torch.cuda.empty_cache()


def smoke(config,out,rows,fit,decoder,frozen_parameters):
    path=out/'resource_smoke.json'
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if path.exists():
        result=json.loads(path.read_text())
        if result['config_sha256']!=config_sha:raise ValueError('SMOKE_CONFIG_CHANGED')
        return result
    pairid=next(r['pair_id'] for r in rows if r['split']=='train');pair=[r for r in rows if r['pair_id']==pairid];records=[]
    for c in CONDITIONS:
        seed_all(SEED);m=SpatialModel(c,decoder,fit['statistics']).cuda();seed_all(SEED)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=False)
        before=m.decoder.state_dict();before={k:v.clone() for k,v in before.items()}
        projection=m.projection[-1].weight.detach().clone() if m.context else None
        torch.cuda.reset_peak_memory_stats();m.train();updates=[]
        for _ in range(2):updates.append(pair_update(m,opt,pair,config,out,fit,scaler))
        if not any(not torch.equal(before[k],v) for k,v in m.decoder.named_parameters()):raise ValueError('DECODER_DID_NOT_UPDATE')
        if m.context and torch.equal(projection,m.projection[-1].weight):raise ValueError('PROJECTION_DID_NOT_UPDATE')
        if m.context and any(not updates[-1]['branch_grad_abs_sum'].get(k) for k in ['geometry','gru','projection']):
            raise ValueError('CONTEXT_PATH_GRADIENT_MISSING_AFTER_TWO_UPDATES')
        progress(out,'smoke_evaluate',condition=c)
        normal=evaluate(m,config,out,pair,fit);ap=evaluate(m,config,out,pair,fit,pixel_ap=True)
        records.append({'condition':c,'pair_id':pairid,'updates':updates,'evaluation_s':normal['seconds'],'exact_AP_evaluation_s':ap['seconds'],
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
            'model_parameters':sum(p.numel() for p in m.parameters())+frozen_parameters,'frozen_backbone_parameters':frozen_parameters})
        del m,opt;torch.cuda.empty_cache()
    projections={}
    for r in records:
        rate=sum(x['seconds'] for x in r['updates'])/sum(x['frames'] for x in r['updates']);n=sum(x['actual_frames'] for x in pair)
        train_frames=sum(x['actual_frames'] for x in rows if x['split']=='train');all_frames=sum(x['actual_frames'] for x in rows)
        projections[r['condition']]={'train30_s':rate*train_frames*30,'routine_eval30_s':r['evaluation_s']/n*all_frames*30,
            'extra_dev_AP6_s':max(0,r['exact_AP_evaluation_s']-r['evaluation_s'])/n*sum(x['actual_frames'] for x in rows if x['split']=='dev')*6}
    result={'config_sha256':config_sha,'records':records,'projection_max30':projections,'smoke_weights':'discarded; recreate paired official decoder and zero projection',
        'limits':'one complete train pair; final full AP and exports excluded; early stopping may shorten'}
    save(path,result);return result


def train(config,out,rows,fit,decoder,frozen_parameters,resume=False):
    trainrows=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    pairs={key:[r for r in trainrows if r['pair_id']==key] for key in sorted({r['pair_id'] for r in trainrows})}
    identity={'config':config,'manifest':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
        'official_load':hashlib.sha256((out/'official_load.json').read_bytes()).hexdigest(),
        'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    shared=['component_global_context_pilot/model.py','full_video_training/model.py','local_geometry_pilot/model.py',
        'spatial_localization/joint_data.py','spatial_localization/model.py','spatial_localization/data.py','spatial_localization/metrics.py']
    identity['shared_code']={name:hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest() for name in shared}
    for c in CONDITIONS:
        folder=out/c;folder.mkdir(exist_ok=True);summary=folder/'summary.json'
        if summary.exists():
            if json.loads(summary.read_text())['identity']!=identity:raise ValueError('COMPLETED_IDENTITY_CHANGED')
            continue
        seed_all(SEED);model=SpatialModel(c,decoder,fit['statistics']).cuda();seed_all(SEED)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=False)
        epoch=0;position=0;history=[];online=[];updates=0;elapsed=0.;stale=0
        initial=json.loads((out/'epoch0.json').read_text());best=initial['dev']['metrics']['spatial_dice'];best_epoch=0;anchor=best
        last=folder/'last.pt';run_identity={**identity,'condition':c}
        if last.exists():
            if not resume:raise ValueError('RESUME_REQUIRED')
            cp=torch.load(last,map_location='cpu',weights_only=False)
            if cp['identity']!=run_identity:raise ValueError('RESUME_IDENTITY')
            model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['amp']);restore_rng(cp['rng'])
            epoch,position,history,online,updates,best,best_epoch,anchor,stale,elapsed=(cp[k] for k in ['epoch','pair_index','history','online','updates','best','best_epoch','patience_ref','stale','elapsed_s'])
            if epoch!=len(history) or position!=len(online):raise ValueError('RESUME_POSITION')
        else:
            atomic_torch(folder/'best.pt',checkpoint_payload(model,opt,scaler,run_identity,0,0,[],[],0,best,0,anchor,0,0.))
        begin=time.monotonic()
        while epoch<config['epochs'] and stale<8:
            keys=sorted(pairs);random.Random(SEED+epoch).shuffle(keys);model.train();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            for j in range(position,len(keys)):
                progress(out,'train',condition=c,epoch=epoch+1,pair=j+1,total_pairs=len(keys))
                online.append(pair_update(model,opt,pairs[keys[j]],config,out,fit,scaler));updates+=1
                atomic_torch(last,checkpoint_payload(model,opt,scaler,run_identity,epoch,j+1,history,online,updates,best,best_epoch,anchor,stale,elapsed+time.monotonic()-begin))
                if STOP:raise InterruptedError('SAFE_PAIR_BOUNDARY_STOP')
            progress(out,'evaluate',condition=c,epoch=epoch+1)
            tm=evaluate(model,config,out,trainrows,fit);dv=evaluate(model,config,out,dev,fit,pixel_ap=(epoch+1)%5==0)
            dice=dv['metrics']['spatial_dice'];improved=dice>best
            if improved:best=dice;best_epoch=epoch+1
            if dice>anchor+.002:anchor=dice;stale=0
            else:stale+=1
            record={'epoch':epoch+1,'online':{k:float(np.mean([r[k] for r in online])) for k in ['loss','bce','dice']},
                'train':tm['metrics'],'dev':dv['metrics'],'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in online])),
                'lr':1e-4,'updates':updates,'train_s':sum(r['seconds'] for r in online),'eval_s':tm['seconds']+dv['seconds'],
                'frames':sum(r['frames'] for r in online),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            save(folder/'epoch_metrics'/f'epoch_{epoch+1:02d}.json',{'train':tm,'dev':dv,'pair_order':keys,'updates':online})
            history.append(record);save(folder/'history.json',history);history_csv(folder/'history.csv',history);curves(out,c,history)
            epoch+=1;position=0;online=[]
            cp=checkpoint_payload(model,opt,scaler,run_identity,epoch,0,history,online,updates,best,best_epoch,anchor,stale,elapsed+time.monotonic()-begin)
            atomic_torch(last,cp)
            if improved:atomic_torch(folder/'best.pt',cp)
            print(f'{c} epoch={epoch:02d} online={record["online"]["loss"]:.6f} train_dice={tm["metrics"]["spatial_dice"]:.6f} dev_dice={dice:.6f} best={best:.6f}@{best_epoch} stale={stale}/8',flush=True)
        metrics={}
        for ck in ['last','best']:
            if ck=='best':model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
            progress(out,'final_metrics',condition=c,checkpoint=ck)
            metrics[ck]={split:evaluate(model,config,out,rr,fit,pixel_ap=True,samples=folder/'samples'/ck)
                for split,rr in [('train',trainrows),('dev',dev)]}
        costs={'active_seconds':elapsed+time.monotonic()-begin,'frozen_backbone_parameters':frozen_parameters,
            'frozen_parameters':frozen_parameters+sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters())+frozen_parameters,
            'additional_branch_parameters':sum(p.numel() for k,p in model.named_parameters() if not k.startswith('decoder.')),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in history)}
        save(summary,{'status':'COMPLETE','best_epoch':best_epoch,'last_epoch':epoch,'updates':updates,'identity':identity,
            'stop_reason':'patience8_delta0.002' if stale>=8 else 'max30','costs':costs,**metrics})
        del model,opt;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=['assets','smoke','all'],default='all');args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['seed']!=SEED or config['epochs']!=30:raise ValueError('FROZEN_SEED_AND_MAX_EPOCHS')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        progress(out,'assets');rows,fit=prepare(config,out)
        network=official(config,out).cuda();decoder=copy.deepcopy(network.encoder.localization_head).cpu()
        verify_online(config,out,rows,fit,network,use_cache=False)
        if args.phase=='assets':
            if (out/'feature_costs.json').exists():verify_online(config,out,rows,fit,network)
            save(out/'progress.json',{'status':'ASSETS_VERIFIED','pid':os.getpid()});return
        extract(config,out,rows,fit,network,lambda stage,**kw:progress(out,stage,**kw));verify_online(config,out,rows,fit,network)
        frozen=sum(p.numel() for p in network.backbone.parameters());del network;torch.cuda.empty_cache()
        smoke(config,out,rows,fit,decoder,frozen)
        if args.phase=='smoke':save(out/'progress.json',{'status':'SMOKE_COMPLETE','pid':os.getpid()});return
        if not (out/'epoch0.json').exists():
            progress(out,'epoch0');baseline=SpatialModel(CONDITIONS[0],decoder,fit['statistics']).cuda()
            save(out/'epoch0.json',{s:evaluate(baseline,config,out,[r for r in rows if r['split']==s],fit,pixel_ap=True,
                samples=out/'epoch0_samples') for s in ['train','dev']});del baseline;torch.cuda.empty_cache()
        train(config,out,rows,fit,decoder,frozen,args.resume);report(out,rows);verify_source(config,out)
        save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'unix_time':time.time()})
    except BaseException as e:
        save(out/'progress.json',{'status':'INTERRUPTED' if isinstance(e,InterruptedError) else 'FAILED','pid':os.getpid(),
            'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
