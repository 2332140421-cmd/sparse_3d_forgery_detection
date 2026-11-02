"""PVST single entry: prepare -> checks/resources -> train -> evaluate -> report.

Exactly two fixed conditions. Video-gradient accumulation uses complete-video
class/frame denominators; target chunks only change materialization, not loss.
"""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from scripts.prepare_v8_assets import digest
from research_tools.v8.local_geometry_pilot.runner import atomic_torch,seed_all,rng_state,restore_rng
from research_tools.v8.spatial_localization.metrics import history_csv
from .data import freeze,save,prepare,fit_scale,decoder_only,Video,preview_originals
from .model import PVST
from .probe import parameters,execution_signature,scan_windows,model_trace
from .evaluate import evaluate,report

STOP=False

def stop(*_):
    global STOP
    STOP=True


def frame_weights(y):
    y=np.asarray(y,bool);n=len(y);positive=int(y.sum());negative=n-positive
    if positive and negative:return np.where(y,.5/positive,.5/negative)
    return np.full(n,1/n)


def video_backward(model,row,out,config,scale,scaler,resource_log=None):
    started=time.monotonic();video=Video(out,row,model.condition,scale);labels=np.array([video.label(i) for i in range(row['actual_frames'])]);weight=frame_weights(labels);total=np.zeros(4);event={"video_id":row["video_id"]}
    try:
        for start in range(0,row['actual_frames'],config['target_frames']):
            if start%80==0:print('VIDEO_BACKWARD',model.condition,row['video_id'],start,'/',row['actual_frames'],flush=True)
            if STOP:raise InterruptedError('STOP_WITHOUT_COMMITTING_PARTIAL_UPDATE')
            if resource_log is not None:torch.cuda.reset_peak_memory_stats()
            block_started=time.monotonic();end=min(row['actual_frames'],start+config['target_frames']);left=max(0,start-16)
            event={'video_id':row['video_id'],'target_range':[start,end],'context_range':[left,start],'condition':model.condition}
            if resource_log is not None:
                with resource_log.open('a') as log:log.write(json.dumps({**event,'status':'STARTED'})+'\n')
            frames=video.frames(left,end,'cuda')
            if resource_log is not None:
                event.update(fine_nodes=sum(len(f['voxel_meta']['coords']) for f in frames),coarse_nodes=sum(len(f['voxel_meta']['coarse_coords']) for f in frames))
            ctx=torch.autograd.graph.save_on_cpu(pin_memory=True) if config.get('offload_saved_tensors') else nullcontext()
            with ctx,torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                outputs=model(frames,start-left);loss=outputs[0]['frame_logit'].new_zeros((),dtype=torch.float32);lf=loss.clone();lb=loss.clone();ld=loss.clone()
                for j,p in enumerate(outputs):
                    i=start+j;frame=F.binary_cross_entropy_with_logits(p['frame_logit'].float(),p['frame_logit'].new_tensor(float(labels[i])).float())*float(weight[i]);lf=lf+frame
                    gt=video.gt(i)
                    if gt is not None:
                        y=torch.as_tensor(gt,device='cuda',dtype=torch.float32);z=p['pixel_logits'].float();prob=z.sigmoid()
                        lb=lb+F.binary_cross_entropy_with_logits(z,y)/row['actual_frames']
                        ld=ld+(1-(2*(prob*y).sum()+config['dice_epsilon'])/(prob.sum()+y.sum()+config['dice_epsilon']))/row['actual_frames']
                loss=.5*(lf+lb+ld)
            if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_VIDEO_CHUNK')
            scaler.scale(loss).backward();total+=np.array([float(loss.detach()),float(lf.detach())*.5,float(lb.detach())*.5,float(ld.detach())*.5])
            if resource_log is not None:
                torch.cuda.synchronize()
                event.update(status='PASS',seconds=time.monotonic()-block_started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),trace=model_trace(model))
                with resource_log.open('a') as log:log.write(json.dumps(event)+'\n')
            del frames,outputs,loss,lf,lb,ld
    except BaseException as e:
        if resource_log is not None:
            event.update(status='RESOURCE_BLOCKED' if isinstance(e,torch.OutOfMemoryError) else 'FAILED',error=str(e),trace=model_trace(model),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with resource_log.open('a') as log:log.write(json.dumps(event)+'\n')
        raise
    finally:video.close()
    return {'loss':total[0],'frame_BCE':total[1],'pixel_BCE':total[2],'Dice_loss':total[3],'seconds':time.monotonic()-started}


def build(config,condition):
    seed_all(config['seed']);return PVST(condition,decoder_only(config),config).cuda()


def initialization_hash(model):
    h=hashlib.sha256()
    for k,v in model.named_parameters():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def smoke_pair(config,out,rows,scale,progress):
    """One existing ANAKIN full pair through both models, two effective updates.

    Each effective update additionally includes one actual AF video; full-video
    denominators are exactly the formal ones. Discard all resulting weights.
    """
    target=out/'full_pair_check.json'
    identity=json.loads((out/'protocol.json').read_text())
    if target.exists():
        previous=json.loads(target.read_text())
        if previous.get('identity')!=identity or previous.get('status')!='PASS':raise ValueError('FULL_PAIR_CHECK_IDENTITY')
        return previous
    af=next(r for r in rows if r['dataset']=='ActivityForensics' and r['split']=='train')
    an=next(r for r in rows if r['dataset']=='ANAKIN' and r['split']=='train')
    pair=[r for r in rows if r['source_id']==an['source_id']]
    result=[];initial=None
    # Check the previously failing XYZ path first; formal condition order stays UV/XYZ.
    for c in reversed(config['conditions']):
        check_started=time.monotonic();m=build(config,c).train();sha=initialization_hash(m)
        if initial and initial!=sha:raise ValueError('UNMATCHED_INITIAL_PARAMETERS')
        initial=sha;opt=torch.optim.AdamW(parameters(config,m),weight_decay=config['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=False);steps=[]
        for i,anrow in enumerate(pair):
            opt.zero_grad(set_to_none=True);progress('full_pair_check',condition=c,step=i+1)
            resource_log=out/f'full_pair_{c}_step{i+1}.jsonl'
            terms=[video_backward(m,r,out,config,scale,scaler,resource_log) for r in [af,anrow]]
            scaler.unscale_(opt);grad=float(m.geo[0].weight.grad.abs().sum()) if m.geo[0].weight.grad is not None else 0
            torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);scaler.step(opt);scaler.update();m.check_bn()
            paths={'geometry':m.geo[0].weight,'VFE':m.vfe[0].weight,'fine_attention':m.fine_time.layers[0].qkv.weight,'coarse_attention':m.coarse_time.layers[0].qkv.weight}
            gradients={key:float(value.grad.abs().sum()) if value.grad is not None else 0. for key,value in paths.items()}
            if i==1 and any(v<=0 for v in gradients.values()):raise ValueError('FULL_STRUCTURE_PATH_NOT_TRAINED:'+str(gradients))
            steps.append({'update':i+1,'geometry_gradient':grad,'path_gradients':gradients,'terms':terms,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        if steps[-1]['geometry_gradient']<=0:raise ValueError('GEOMETRY_NOT_TRAINED')
        torch.cuda.reset_peak_memory_stats();evaluation_start=time.monotonic()
        predicted=evaluate(m,[af,*pair],out,config,scale,pixel_ap=True,progress=progress)
        file=out/(c+'_smoke.pt');atomic_torch(file,{'model':m.state_dict()});n=build(config,c);n.load_state_dict(torch.load(file,map_location='cpu',weights_only=False)['model'])
        again=evaluate(n,[af,*pair],out,config,scale,pixel_ap=False,progress=progress)
        error=max(abs(x-y) for a,b in zip(predicted['records'],again['records']) for x,y in zip(a['frame_scores'],b['frame_scores']))
        for a,b in zip(predicted['records'],again['records']):
            for key in ['spatial_dice','spatial_iou','GT_recall','empty_spatial_fpr']:
                if a[key] is not None and abs(a[key]-b[key])>1e-8:raise ValueError('SPATIAL_RELOAD_PARITY:'+key)
        if error>1e-5:raise ValueError('CHECKPOINT_RELOAD_PARITY')
        entry={'condition':c,'initial_sha256':sha,'steps':steps,'reload_error':error,'native_mapping':'predictions and masks already native HxW','parameters':sum(p.numel() for p in m.parameters()),'seconds':time.monotonic()-check_started,'evaluation_and_reload_s':time.monotonic()-evaluation_start,'evaluation_reload_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'evaluation_reload_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'identity':identity}
        if entry['parameters']!=17821672:raise ValueError('PARAMETER_COUNT_CHANGED')
        blocks=[json.loads(line) for i in [1,2] for line in (out/f'full_pair_{c}_step{i}.jsonl').read_text().splitlines() if json.loads(line)['status']=='PASS']
        entry.update(peak_allocated_bytes=max(entry['evaluation_reload_peak_allocated_bytes'],max(b['peak_allocated_bytes'] for b in blocks),max(b['peak_allocated_bytes'] for b in steps)),peak_reserved_bytes=max(entry['evaluation_reload_peak_reserved_bytes'],max(b['peak_reserved_bytes'] for b in blocks),max(b['peak_reserved_bytes'] for b in steps)),peak_block=max(blocks,key=lambda b:b['peak_allocated_bytes']))
        save(out/f'full_pair_{c}.json',{'status':'PASS',**entry});result.append(entry)
        del n,m,opt;file.unlink();torch.cuda.empty_cache()
    save(target,{'status':'PASS','identity':identity,'records':result,'smoke_weights':'discarded; formal models reinitialized'});return result


def train_condition(config,out,rows,scale,c,identity,progress,resume):
    folder=out/c;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        previous=json.loads((folder/'summary.json').read_text())
        if previous['identity']!=identity:raise ValueError('COMPLETED_SUMMARY_IDENTITY')
        return previous
    model=build(config,c);opt=torch.optim.AdamW(parameters(config,model),weight_decay=config['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=False);rng=random.Random(config['seed'])
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    # Fixed two source pairs per dataset, same in both conditions.
    sources={d:sorted({r['source_id'] for r in train if r['dataset']==d})[:2] for d in ['ActivityForensics','ANAKIN']}
    monitor=[r for r in train if r['source_id'] in sources[r['dataset']]]
    af=sorted([r for r in train if r['dataset']=='ActivityForensics'],key=lambda r:(r['source_id'],r['role'],r['video_sha256']))
    an=sorted([r for r in train if r['dataset']=='ANAKIN'],key=lambda r:(r['source_id'],r['role'],r['video_sha256']))
    if len(af)!=16 or len(an)!=16:raise ValueError('EFFECTIVE_UPDATE_QUEUE')
    epoch=1;pos=0;order=[];terms=[];history=[];best=-float('inf');best_epoch=0;stale=0;updates=0;elapsed=0.;peak_a=peak_r=0
    checkpoint=folder/'resume.pt'
    def state(ep,position,ordering,losses):
        return {'identity':identity,'condition':c,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(),'order_rng':rng.getstate(),'epoch':ep,'position':position,'order':ordering,'terms':losses,
            'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,'updates':updates,'elapsed_s':elapsed,
            'peak_allocated':peak_a,'peak_reserved':peak_r}
    if checkpoint.exists():
        if not resume:raise ValueError('EXPLICIT_RESUME_REQUIRED')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['condition']!=c:raise ValueError('RESUME_INPUT_IDENTITY')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler']);restore_rng(cp['rng']);rng.setstate(cp['order_rng'])
        epoch,pos,order,terms,history,best,best_epoch,stale,updates,elapsed,peak_a,peak_r=(cp[k] for k in ['epoch','position','order','terms','history','best','best_epoch','stale','updates','elapsed_s','peak_allocated','peak_reserved'])
    else:
        atomic_torch(folder/'epoch0.pt',state(1,0,[],[]))
        for split,rr in [('train',train),('dev',dev)]:
            value=evaluate(model,rr,out,config,scale,samples=folder/'epoch0_samples'/split,progress=progress);value['epoch']=0;save(folder/f'epoch0_{split}.json',value)
            if split=='dev':best=value['datasets']['ActivityForensics']['video']['auroc']
        initial=state(1,0,[],[]);initial['checkpoint_epoch']=0
        atomic_torch(folder/'epoch0.pt',initial);atomic_torch(folder/'best.pt',initial);atomic_torch(checkpoint,initial)
        for split in ['train','dev']:
            value=json.loads((folder/f'epoch0_{split}.json').read_text());value['checkpoint_sha256']=digest(folder/'epoch0.pt');save(folder/f'epoch0_{split}.json',value)
    torch.cuda.reset_peak_memory_stats()
    while epoch<=config['epochs'] and stale<config['patience']:
        if not order:
            aa=list(range(16));bb=list(range(16));rng.shuffle(aa);rng.shuffle(bb);order=list(zip(aa,bb));terms=[];pos=0
        model.train();epstart=time.monotonic()
        for i in range(pos,16):
            if STOP:raise InterruptedError('SAFE_UPDATE_BOUNDARY_STOP')
            progress('training',condition=c,epoch=epoch,update=i+1);update_start=time.monotonic();opt.zero_grad(set_to_none=True)
            pair_terms=[video_backward(model,r,out,config,scale,scaler) for r in [af[order[i][0]],an[order[i][1]]]]
            scaler.unscale_(opt);norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
            scaler.step(opt);scaler.update();model.check_bn();torch.cuda.synchronize();updates+=1
            item={k:sum(t[k] for t in pair_terms) for k in ['loss','frame_BCE','pixel_BCE','Dice_loss','seconds']};item['gradient_norm']=norm;item['effective_update_s']=time.monotonic()-update_start
            terms.append(item);elapsed+=item['seconds'];peak_a=max(peak_a,torch.cuda.max_memory_allocated());peak_r=max(peak_r,torch.cuda.max_memory_reserved())
            atomic_torch(checkpoint,state(epoch,i+1,order,terms))
        mon=evaluate(model,monitor,out,config,scale,progress=progress);validation=evaluate(model,dev,out,config,scale,progress=progress)
        score=validation['datasets']['ActivityForensics']['video']['auroc']
        if score>best:best=score;best_epoch=epoch;stale=0;improved=True
        else:stale+=1;improved=False
        record={'epoch':epoch,'updates':updates,**{k:float(np.mean([t[k] for t in terms])) for k in ['loss','frame_BCE','pixel_BCE','Dice_loss']},
            'monitor':mon['datasets'],'monitor_scope':sources,'dev':validation['datasets'],'lr_decoder':opt.param_groups[0]['lr'],'lr_new':opt.param_groups[1]['lr'],
            'effective_updates_s':sum(t['effective_update_s'] for t in terms),'epoch_s':time.monotonic()-epstart,'cumulative_update_s':elapsed,'peak_allocated_bytes':peak_a,'peak_reserved_bytes':peak_r,'best_epoch':best_epoch,'early_stop_count':stale}
        history.append(record);save(folder/'history.json',history);history_csv(folder/'history.csv',history)
        cp=state(epoch+1,0,[],[]);cp['checkpoint_epoch']=epoch;atomic_torch(folder/'last.pt',cp)
        if improved:atomic_torch(folder/'best.pt',cp)
        atomic_torch(checkpoint,cp);print('EPOCH_COMPLETE',c,epoch,'AF_dev_AUROC',score,'best',best_epoch,flush=True)
        epoch+=1;pos=0;order=[];terms=[]
    summary={'condition':c,'best_epoch':best_epoch,'last_epoch':epoch-1,'updates':updates,'update_elapsed_s':elapsed,'effective_updates_s':sum(r['effective_updates_s'] for r in history),'training_and_routine_evaluation_s':sum(r['epoch_s'] for r in history),'peak_allocated_bytes':peak_a,
             'peak_reserved_bytes':peak_r,'parameters':sum(p.numel() for p in model.parameters()),'decoder_parameters':sum(p.numel() for p in model.decoder.parameters()),'frozen_encoder_parameters':config['frozen_encoder_parameters'],'identity':identity}
    final_start=time.monotonic()
    for name in ['best','last']:
        p=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False);model.load_state_dict(p['model']);model.check_bn()
        for split,rr in [('train',train),('dev',dev)]:
            value=evaluate(model,rr,out,config,scale,samples=folder/(name+'_samples')/split,progress=progress);value['epoch']=p['checkpoint_epoch'];value['checkpoint_sha256']=digest(folder/(name+'.pt'));save(folder/f'{name}_{split}.json',value)
    summary['final_evaluation_s']=time.monotonic()-final_start;summary['epoch0_evaluation_s']=sum(json.loads((folder/f'epoch0_{split}.json').read_text())['elapsed_s'] for split in ['train','dev'])
    save(folder/'summary.json',summary);del model,opt;torch.cuda.empty_cache();return summary


def publish(out):
    repo=Path(__file__).resolve().parents[2];branch='pvst-point-voxel-v02'
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    if git('branch','--show-current')!=branch or git('status','--porcelain'):raise ValueError('PUBLICATION_REQUIRES_CLEAN_BRANCH')
    doc=repo/'docs/pvst/results_v02.md';doc.write_text((out/'report_zh.md').read_text());paths=[doc]
    compact=json.loads((out/'result_summary.json').read_text());compact['metrics']={}
    for condition in ['UV_PVST','XYZ_PVST']:
        compact['metrics'][condition]={}
        for name in ['epoch0','best','last']:
            compact['metrics'][condition][name]={}
            for split in ['train','dev']:
                value=json.loads((out/condition/f'{name}_{split}.json').read_text());compact['metrics'][condition][name][split]={k:value[k] for k in ['epoch','checkpoint_sha256','datasets','voxel_summary']}
    small=repo/'docs/pvst/results_v02.json';small.write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n');paths.append(small)
    p=repo/'docs/pvst/current_status.md';s=p.read_text();s='\n'.join('状态COMPLETE（两模型训练与epoch0/best/last评价完成）' if line.startswith('状态') else line for line in s.split('\n'));s+='\n最终状态COMPLETE；两模型全部评价完成，见[结果](results_v02.md)。\n';p.write_text(s);paths.append(p)
    import shutil
    for c in ['UV_PVST','XYZ_PVST']:
        p=repo/'docs/pvst'/f'{c.lower()}_curves.png';shutil.copyfile(out/c/'curves.png',p);paths.append(p)
    subprocess.run(['git','add','--',*[str(p.relative_to(repo)) for p in paths]],cwd=repo,check=True);subprocess.run(['git','commit','-m','Record completed PVST v02 probe results'],cwd=repo,check=True)
    head=git('rev-parse','HEAD');save(out/'publication.json',{'status':'COMMITTED','commit':head})
    subprocess.run(['git','-c','http.version=HTTP/1.1','push','origin',branch],cwd=repo,check=True)
    remote=git('-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+branch).split()[0]
    if head!=remote:raise ValueError('REMOTE_HEAD_MISMATCH')
    save(out/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote})


def resource_passed(out,condition,config):
    path=out/f'resource_{condition}_{config["execution_revision"]}.json'
    if not path.exists():return False
    r=json.loads(path.read_text())
    if r.get('status')!='PASS' or r.get('condition')!=condition:return False
    if any(r.get(k)!=v for k,v in execution_signature(config).items()):return False
    if r.get('manifest_sha256')!=digest(out/'manifest.json') or r.get('resource_scale_sha256')!=digest(out/'scale.json'):return False
    if not (out/'resource_windows.json').exists() or r.get('windows_sha256')!=digest(out/'resource_windows.json'):return False
    windows=json.loads((out/'resource_windows.json').read_text())['representatives']
    return [(a['video_id'],a['target_range']) for a in r.get('records',[])]==[(a['video_id'],a['target_range']) for a in windows]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('docs/pvst/config_v02.json'));parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    c=json.loads(a.config.read_text());o=Path(c['output']);o.mkdir(parents=True,exist_ok=True);lock=(o/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (o/'progress.json').exists() and json.loads((o/'progress.json').read_text()).get('status')=='COMPLETE':
        print('ALREADY_COMPLETE: no training, evaluation or frontend rerun',flush=True);return
    torch.set_num_threads(4);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def progress(stage,**kw):
        if STOP:raise InterruptedError('SAFE_STAGE_BOUNDARY_STOP')
        save(o/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'unix_time':time.time(),'execution_revision':c['execution_revision'],**kw});print('PROGRESS',stage,kw,flush=True)
    try:
        if not (o/'protocol.json').exists():
            registration=json.loads((o/'execution_revision.json').read_text())
            if registration['execution_revision']!=c['execution_revision'] or registration['new_config_sha256']!=digest(a.config):raise ValueError('UNREGISTERED_EXECUTION_REVISION')
            if any(list((o/cond).glob('*.pt')) for cond in c['conditions']):raise ValueError('EXISTING_FORMAL_CHECKPOINT_WITHOUT_PROTOCOL')
        rows=freeze(c,o)
        frozen_manifest=json.loads(Path('docs/pvst/manifest_v02.json').read_text())
        if digest(o/'manifest.json')!=frozen_manifest['full_manifest_sha256']:raise ValueError('FROZEN_MANIFEST_CHANGED')
        if c['conditions']!=['UV_PVST','XYZ_PVST'] or c['seed']!=20260909:raise ValueError('FROZEN_TWO_CONDITION_SEED')
        if not (o/'original_preview/selection.json').exists():progress('original_preview');preview_originals(o,rows)
        progress('scale_fit');scale=fit_scale(c,o,rows)
        progress('prepare');prepare(c,o,rows,progress)
        identity={'execution_revision':c['execution_revision'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'config_sha256':digest(a.config),'manifest_sha256':digest(o/'manifest.json'),'scale_sha256':digest(o/'scale.json'),
            'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
        if (o/'protocol.json').exists() and json.loads((o/'protocol.json').read_text())!=identity:raise ValueError('PROTOCOL_CHANGED')
        save(o/'protocol.json',identity);smoke_pair(c,o,rows,scale,progress)
        progress('resource_windows');scan_windows(c,o,progress)
        for cond in c['conditions']:
            if resource_passed(o,cond,c):continue
            progress('representative_resources',condition=cond)
            result=subprocess.run([sys.executable,'-u','-m','research_tools.pvst.probe','--config',str(a.config),'--condition',cond])
            if result.returncode or not resource_passed(o,cond,c):raise RuntimeError('RESOURCE_GATE_NOT_PASSED:'+cond+'; see resource evidence')
        if identity['code_sha256']!={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')} or identity['config_sha256']!=digest(a.config):raise ValueError('EXECUTION_CHANGED_DURING_CHECKS')
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('FORMAL_TRAINING_REQUIRES_CLEAN_COMMIT')
        if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=identity['source_commit']:raise ValueError('SOURCE_COMMIT_CHANGED_DURING_CHECKS')
        summaries=[train_condition(c,o,rows,scale,cond,identity,progress,a.resume) for cond in c['conditions']]
        report(o,c,rows,summaries);save(o/'progress.json',{'status':'COMPLETE','stage':'report_complete','completed_models':2,'planned_models':2,'execution_revision':c['execution_revision'],'pid':os.getpid(),'unix_time':time.time()})
    except BaseException as e:
        save(o/'progress.json',{'status':'RESOURCE_BLOCKED' if 'RESOURCE_GATE' in str(e) or isinstance(e,torch.OutOfMemoryError) else 'STOPPED_SAFE' if isinstance(e,InterruptedError) else 'FAILED','execution_revision':c['execution_revision'],'error':str(e),'traceback':traceback.format_exc()});raise
    try:publish(o)
    except BaseException as e:
        previous=json.loads((o/'publication.json').read_text()) if (o/'publication.json').exists() else {}
        save(o/'publication.json',{**previous,'status':'FAILED','experiment_status':'COMPLETE','error':str(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    status=0
    try:main()
    except BaseException:status=1;raise
    finally:
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',type=Path,default=Path('docs/pvst/config_v02.json'));a,_=p.parse_known_args();o=Path(json.loads(a.config.read_text())['output'])
        if o.exists():
            (o/'exit_code.txt').write_text(str(status)+'\n')
            with (o/'exit_history.jsonl').open('a') as f:f.write(json.dumps({'pid':os.getpid(),'execution_revision':json.loads(a.config.read_text())['execution_revision'],'exit_code':status,'unix_time':time.time()})+'\n')
