"""ADR-0056: user-launched full-video matched localization experiment.

Default: freeze -> observations -> exact resource gate -> two three-epoch arms
-> evaluation/report. --plan-only is CPU metadata; --smoke is bounded and does
not enter the queue. Resume only at committed complete-video boundaries.
"""
import argparse
import collections
import copy
import csv
import gc
import fcntl
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from . import fullvideo_matched_data as data
from .fullvideo_direct_model import FullVideoDirect, DirectLayer, MODES, video_loss, prepare_frame
from .fullvideo_direct_probe import save, state_hash, seed, sync, memory
from research_tools.v8.spatial_localization.metrics import spatial_frame
from scripts.review_anakin import numeric_members,mask_array

METRICS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area','constant_AP']
HIST={'an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4'}


def csvsave(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix('.tmp').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    path.with_suffix('.tmp').replace(path)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path,model,opt,identity,mode,epoch,position,records):
    cp=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
            random_state=rng_state(),identity=identity,mode=mode,epoch=epoch,position=position,
            updates=len(records),supervised_presentations=sum(r['supervised_frames'] for r in records),records=records)
    data.atomic_torch(path,cp)
    return cp


def restore_checkpoint(path,model,opt,identity,mode):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['identity']==identity and cp['mode']==mode,'CHECKPOINT_IDENTITY'
    assert cp['updates']==len(cp['records'])
    assert cp['supervised_presentations']==sum(r['supervised_frames'] for r in cp['records'])
    model.load_state_dict(cp['state_dict'],strict=True);opt.load_state_dict(cp['optimizer'])
    assert state_hash(model.state_dict())==state_hash(cp['state_dict'])
    set_rng(cp['random_state'])
    return cp


def train_video(model,opt,frames,mode,emit):
    model.train();opt.zero_grad(set_to_none=True);sync();start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        outputs=model(frames,mode,progress=emit);loss,parts=video_loss(outputs,frames)
    assert torch.isfinite(loss);emit('loss_backward',T=len(frames),M=sum(len(f['keys']) for f in frames),loss=float(loss))
    loss.backward()
    for name,p in model.named_parameters():
        if p.grad is not None:assert torch.isfinite(p.grad).all(),name
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    emit('optimizer_update',lr=opt.param_groups[0]['lr']);opt.step();sync()
    return dict(loss=float(loss),pixel_BCE=float(parts[0]),soft_Dice=float(parts[1]),
                seconds=time.monotonic()-start,T=len(frames),M=sum(len(f['keys']) for f in frames),
                supervised_frames=sum(f['supervised'] for f in frames),gradient_norm=float(norm),**memory())


def native_mask(z,members,row,i):
    if row['role']=='real':return np.zeros(row['native_hw'],bool)
    gt=mask_array(z.read(members[i][1]))[1]
    assert list(gt.shape)==row['native_hw']
    return gt


def score_video(row,frames,outputs,wanted=None,capture=None,emit=None):
    wanted=set(i for i in range(row['actual_frames']) if data.supervised(row,i)) if wanted is None else set(wanted)
    assert all(0<=i<row['actual_frames'] and data.supervised(row,i) for i in wanted)
    z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
    members=numeric_members(z.namelist()) if z else None
    if z:assert len(members)==row['actual_frames'] and [m[0] for m in members]==list(range(row['index_base'],row['index_base']+row['actual_frames']))
    result=[]
    try:
        for i in sorted(wanted):
            gt=native_mask(z,members,row,i);assert int(gt.sum())==frames[i]['native_GT_pixels']
            native=F.interpolate(outputs[i].float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            assert np.isfinite(native).all();pred=native>=0;overlap=spatial_frame(pred,gt)
            neg=gt.size-int(gt.sum());fp=int((pred&~gt).sum())
            rec=dict(video_id=row['video_id'],pair_id=row['pair_id'],source_group_id=row['source_group_id'],role=row['role'],
                     frame_index=i,pts=row['pts'][i],native_pixels=gt.size,GT_pixels=int(gt.sum()),
                     pixel_AP=float(average_precision_score(gt.ravel(),native.ravel())) if gt.any() else None,
                     constant_AP=float(gt.mean()) if gt.any() else None,Dice=overlap['dice'],IoU=overlap['iou'],
                     GT_recall=float((pred&gt).sum()/gt.sum()) if gt.any() else None,
                     real_FPR=overlap['empty_fpr'] if row['role']=='real' else None,
                     fake_background_FPR=fp/neg if row['role']=='fake' and gt.any() and neg else None,
                     fake_empty_FPR=overlap['empty_fpr'] if row['role']=='fake' and not gt.any() else None,
                     prediction_area=float(pred.mean()) if row['role']=='fake' else None)
            result.append(rec)
            if emit and (len(result)%20==0 or len(result)==len(wanted)):
                emit('score_frame',video=row['video_id'],scored=len(result),total=len(wanted),frame=i)
            if capture and i in row['preview_targets']:capture(row,i,frames[i],gt,native)
    finally:
        if z:z.close()
    return result


def mean_metric(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def aggregate_records(records):
    videos=[]
    for vid in sorted({f['video_id'] for f in records}):
        fs=[f for f in records if f['video_id']==vid];first=fs[0]
        videos.append(dict(video_id=vid,pair_id=first['pair_id'],source_group_id=first['source_group_id'],role=first['role'],
                           frames=len(fs),positive_GT_frames=sum(f['GT_pixels']>0 for f in fs),
                           empty_GT_frames=sum(f['GT_pixels']==0 for f in fs),**{k:mean_metric(fs,k) for k in METRICS}))
    sources=[]
    for source in sorted({v['source_group_id'] for v in videos}):
        vs=[v for v in videos if v['source_group_id']==source]
        sources.append(dict(source_group_id=source,pair_id=vs[0]['pair_id'],
            **{k:mean_metric([v for v in vs if v['role']==('real' if k=='real_FPR' else 'fake')],k) for k in METRICS}))
    macro={k:mean_metric(sources,k) for k in METRICS}
    return dict(frames=records,videos=videos,sources=sources,macro=macro,scored_frames=len(records))


def groups(ev,categories):
    sources=ev['sources'];result={}
    choices={'全部19开发来源':set(categories),'原4开发来源':HIST,'新增15开发来源':set(categories)-HIST,
             '对象编辑16来源':{p for p,c in categories.items() if c['content_priority']=='object_edit_pending_human'},
             '外观背景参照':{p for p,c in categories.items() if c['content_priority']=='appearance_background_reference'}}
    for key in ['official_action','camera_group']:
        for value in sorted({c[key] for c in categories.values()}):choices[key+':'+value]={p for p,c in categories.items() if c[key]==value}
    for name,ids in choices.items():
        ss=[s for s in sources if s['pair_id'] in ids]
        result[name]=dict(sources=len(ss),frames=sum(f['pair_id'] in ids for f in ev['frames']),**{k:mean_metric(ss,k) for k in METRICS})
    return result


def persist_eval(path,ev):
    for kind in ['frames','videos','sources']:csvsave(path.with_name(path.stem+'_'+kind+'.csv'),ev[kind])
    save(path,ev)  # JSON completion marker follows the derived tables.


@torch.no_grad()
def evaluate(model,cache,rows,mode,emit,categories=None,capture=None):
    model.eval();records=[];start=time.monotonic()
    for vi,row in enumerate(rows):
        emit('load_evaluation_video',mode=mode,video=row['video_id'],position=vi+1,total=len(rows),T=row['actual_frames'])
        frames=cache.load(row)
        emit('evaluate_video',mode=mode,video=row['video_id'],position=vi+1,total=len(rows),T=len(frames),M=sum(len(f['keys']) for f in frames))
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(frames,mode)
        context=dict(mode=mode,video=row['video_id'],position=vi+1,total_videos=len(rows),T=len(frames),M=sum(len(f['keys']) for f in frames))
        def score_emit(stage,context=context,**kw):emit(stage,**(context|kw))
        records.extend(score_video(row,frames,outputs,capture=capture,emit=score_emit))
        del frames,outputs
    assert len({(f['video_id'],f['frame_index']) for f in records})==len(records)
    ev=aggregate_records(records);ev['seconds']=time.monotonic()-start
    if categories:ev['groups']=groups(ev,categories)
    return ev


def preview_writer(out,mode):
    folder=out/'preview';folder.mkdir(parents=True,exist_ok=True)
    def capture(row,i,f,gt,native):
        stem=row['video_id']+f'_{i:05d}';h,w=row['native_hw'];width=min(w,480);size=(width,round(h*width/w))
        def pred_images(value,name):
            p=1/(1+np.exp(-np.clip(value,-80,80)))
            Image.fromarray(np.round(255*p).astype(np.uint8)).resize(size).save(folder/(stem+'_'+name+'_prob.png'))
            Image.fromarray(((value>=0)*255).astype(np.uint8)).resize(size,Image.Resampling.NEAREST).save(folder/(stem+'_'+name+'_mask.png'))
        if not (folder/(stem+'_rgb.png')).exists():
            work=data.WorkVideo(row).get(i)
            Image.fromarray(work['rgb'].numpy()).resize(size).save(folder/(stem+'_rgb.png'))
            Image.fromarray((gt*255).astype(np.uint8)).resize(size,Image.Resampling.NEAREST).save(folder/(stem+'_gt.png'))
            z0=F.interpolate(f['z0'].float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].numpy()
            pred_images(z0,'FROZEN_EPOCH01')
        pred_images(native,mode)
    return capture


def reference(out,rows,categories):
    source=data.read(data.SOURCE/'epoch01_dev.json')
    byid={r['video_id']:r for r in rows}
    assert source['scored_frames']==5152
    records=[]
    for old in source['frames']:
        r=byid[old['video_id']];i=old['frame_index']
        assert i in r['evaluation_targets'] and old['pts']==r['pts'][i]
        assert old['native_pixels']==int(np.prod(r['native_hw']))
        records.append(dict(old,source_group_id=r['source_group_id']))
    assert len(records)==len({(f['video_id'],f['frame_index']) for f in records})==5152
    ev=aggregate_records(records);ev['groups']=groups(ev,categories)
    ev['origin']='read-only epoch01_dev.json; no GPU reference replay'
    persist_eval(out/'frozen_reference.json',ev)
    return ev


def query_equivalence():
    # Exact output and input/parameter gradient comparison, only CPU test data.
    torch.manual_seed(20260909);a=DirectLayer(0).double();b=DirectLayer(1024).double();b.load_state_dict(a.state_dict())
    x=torch.randn(1025,128,dtype=torch.double,requires_grad=True);p=torch.randn_like(x)
    ya=a.attend(x,p);ga=torch.autograd.grad(ya.square().sum(),(x,*a.parameters()))
    yb=b.attend(x,p);gb=torch.autograd.grad(yb.square().sum(),(x,*b.parameters()))
    torch.testing.assert_close(ya,yb,rtol=1e-8,atol=1e-9)
    for g,h in zip(ga,gb):torch.testing.assert_close(g,h,rtol=1e-7,atol=1e-8)


def extreme_once(out,p,identity,chunk,emit):
    stats=data.read(out/'queue_stats.json');cache=data.VideoCache(out);result=[]
    for split in ['train','dev']:
        largest=max((s for s in stats if s['split']==split),key=lambda s:(s['M'],s['T'],s['video_id']))
        row=next(r for r in p['rows'] if r['video_id']==largest['video_id'])
        previous=data.PROBE.parent/'cases'/row['video_id']/'result.json'
        if split=='train' and chunk==0 and previous.exists():
            old=data.read(previous)
            if old['status']=='PASS' and old['identity']['adaptation_sha256']==identity['adaptation_sha256'] and old['identity']['provider_config_sha256']==identity['provider_sha256'] and all(m['initial_sha256']==data.read(out/'initialization.json')['state_sha256'] and m['M']==largest['M'] and m['T']==largest['T'] and m['query_chunk']==0 for m in old['modes']):
                result.append(dict(split=split,video=row['video_id'],reused_probe=str(previous),M=largest['M'],T=largest['T']));continue
        save(out/'resource_current.json',dict(split=split,video=row['video_id'],T=largest['T'],M=largest['M'],stage='load'))
        frames=cache.load(row)
        for mode in MODES:
            seed();model=FullVideoDirect(chunk).cuda();model.load_state_dict(torch.load(out/'initial.pt',weights_only=True,map_location='cpu'),strict=True)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);torch.cuda.reset_peak_memory_stats()
            save(out/'resource_current.json',dict(split=split,video=row['video_id'],T=largest['T'],M=largest['M'],mode=mode,query_chunk=chunk))
            emit('resource_extreme',split=split,video=row['video_id'],T=largest['T'],M=largest['M'],mode=mode,query_chunk=chunk)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                if split=='train':
                    steps=[train_video(model,opt,frames,mode,emit) for _ in range(2)]
                else:
                    sync();start=time.monotonic()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):outputs=model.eval()(frames,mode)
                    assert len(outputs)==len(frames) and all(torch.isfinite(x).all() for x in outputs)
                    sync();steps=[dict(seconds=time.monotonic()-start)];del outputs
            result.append(dict(split=split,video=row['video_id'],mode=mode,T=largest['T'],M=largest['M'],steps=steps,
                               dispatch=sorted({x.key for x in prof.key_averages() if 'scaled_dot_product' in x.key}),**memory()))
            save(out/f'resource_chunk{chunk}_partial.json',result)
            del model,opt,prof;gc.collect();torch.cuda.empty_cache()
        del frames
    return result


def resource_gate(out,p,identity,emit):
    if (out/'compute.json').exists():
        c=data.read(out/'compute.json');assert c['identity']==identity;return c['query_chunk']
    for chunk in [0,1024]:
        gc.collect();torch.cuda.empty_cache()
        if chunk:query_equivalence()
        try:
            measured=extreme_once(out,p,identity,chunk,emit)
        except torch.cuda.OutOfMemoryError:
            save(out/f'resource_chunk{chunk}_OOM.json',dict(error=traceback.format_exc(),context=data.read(out/'resource_current.json'),**memory()))
            gc.collect();torch.cuda.empty_cache()
            if chunk==1024:raise
            emit('exact_OOM_fallback',query_chunk=1024);continue
        save(out/f'resource_chunk{chunk}_PASS.json',measured)
        save(out/'compute.json',dict(identity=identity,query_chunk=chunk,checkpoint='non-reentrant',resource='PASS'))
        return chunk
    raise RuntimeError('Resource gate failed')


def validate_cursor(cp,orders):
    assert 1<=cp['epoch']<=4
    assert 0<=cp['position']<=(len(orders[cp['epoch']-1]) if cp['epoch']<=3 else 0)
    prefix=[v for seq in orders[:cp['epoch']-1] for v in seq]
    if cp['epoch']<=3:prefix+=orders[cp['epoch']-1][:cp['position']]
    assert [r['video_id'] for r in cp['records']]==prefix,'CHECKPOINT_EXPOSURE_ORDER'
    assert cp['updates']==len(prefix)


def ensure_initial(out):
    seed();initial=FullVideoDirect().state_dict();sha=state_hash(initial)
    if (out/'initial.pt').exists():
        assert state_hash(torch.load(out/'initial.pt',map_location='cpu',weights_only=True))==sha
    else:data.atomic_torch(out/'initial.pt',initial)
    save(out/'initialization.json',dict(seed=20260909,state_sha256=sha,parameters=sum(v.numel() for v in initial.values())))


def run_mode(out,p,identity,mode,chunk,resume,emit):
    folder=out/mode;folder.mkdir(exist_ok=True);last=folder/'last.pt';initial=torch.load(out/'initial.pt',weights_only=True,map_location='cpu')
    seed();model=FullVideoDirect(chunk).cuda();model.load_state_dict(initial,strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    runtime_identity=dict(protocol=identity,query_chunk=chunk,initial_sha256=data.read(out/'initialization.json')['state_sha256'])
    if last.exists():
        assert resume,'Existing last checkpoint requires explicit --resume'
        cp=restore_checkpoint(last,model,opt,runtime_identity,mode)
    else:
        assert state_hash(model.state_dict())==state_hash(initial)
        cp=save_checkpoint(last,model,opt,runtime_identity,mode,1,0,[])
    orders=data.read(out/'orders.json');validate_cursor(cp,orders);rows={r['video_id']:r for r in p['rows']};cache=data.VideoCache(out)
    monitor=[rows[v] for v in data.read(out/'monitor.json')];dev=[r for r in p['rows'] if r['split']=='dev'];cats=data.read(out/'categories.json')
    records=cp['records'];begin=time.monotonic();base_updates=len(records)
    for epoch in range(cp['epoch'],4):
        sequence=orders[epoch-1];position=cp['position'] if epoch==cp['epoch'] else 0
        expected=[v for seq in orders[:epoch-1] for v in seq]+sequence[:position]
        assert [r['video_id'] for r in records]==expected,'CHECKPOINT_EXPOSURE_ORDER'
        for i in range(position,len(sequence)):
            row=rows[sequence[i]]
            emit('load_training_video',mode=mode,epoch=epoch,video=i+1,videos=len(sequence),video_id=row['video_id'],T=row['actual_frames'])
            frames=cache.load(row);torch.cuda.reset_peak_memory_stats()
            emit('train_video',mode=mode,epoch=epoch,epochs=3,video=i+1,videos=len(sequence),video_id=row['video_id'],T=len(frames),M=sum(len(f['keys']) for f in frames))
            context=dict(mode=mode,epoch=epoch,epochs=3,video=i+1,videos=len(sequence),video_id=row['video_id'],T=len(frames),M=sum(len(f['keys']) for f in frames))
            def video_emit(stage,context=context,**kw):emit(stage,**(context|kw))
            rec=train_video(model,opt,frames,mode,video_emit);del frames
            records.append(dict(rec,epoch=epoch,position=i+1,video_id=row['video_id'],source_group_id=row['source_group_id'],role=row['role']))
            cp=save_checkpoint(last,model,opt,runtime_identity,mode,epoch,i+1,records)
            csvsave(folder/'updates.csv',records)
            elapsed=time.monotonic()-begin;done=len(records)-base_updates
            emit('train_progress',mode=mode,epoch=epoch,epochs=3,video=i+1,videos=len(sequence),updates=len(records),U=534,
                 lr=opt.param_groups[0]['lr'],elapsed_seconds=elapsed,estimated_remaining_train_seconds=sum(x['seconds'] for x in records[-done:])/max(1,done)*(534-len(records)),**rec)
        model_sha=state_hash(model.state_dict())
        for label,selected in [('monitor',monitor),('dev',dev)]:
            path=folder/f'epoch{epoch:02d}_{label}.json'
            if path.exists():
                ev=data.read(path);assert ev['model_sha256']==model_sha
            else:
                ev=evaluate(model,cache,selected,mode,emit,cats if label=='dev' else None,
                            preview_writer(out,mode) if label=='dev' and epoch==3 else None)
                ev['model_sha256']=model_sha;persist_eval(path,ev)
            emit('epoch_evaluation',mode=mode,epoch=epoch,label=label,metrics=ev['macro'])
        ep=folder/f'epoch{epoch:02d}.pt'
        if not ep.exists():data.atomic_torch(ep,cp)
        epoch_loss={k:float(np.mean([x[k] for x in records if x['epoch']==epoch])) for k in ['loss','pixel_BCE','soft_Dice']}
        save(folder/f'epoch{epoch:02d}_loss.json',epoch_loss)
        emit('epoch_summary',mode=mode,epoch=epoch,epochs=3,updates=len(records),online_video_mean_loss=epoch_loss,
             train_observation=data.read(folder/f'epoch{epoch:02d}_monitor.json')['macro'],
             development_groups=data.read(folder/f'epoch{epoch:02d}_dev.json')['groups'])
        cp=save_checkpoint(last,model,opt,runtime_identity,mode,epoch+1,0,records)
        report(out)
    assert len(records)==534 and sum(r['supervised_frames'] for r in records)==68634
    counts=collections.Counter(r['video_id'] for r in records);assert set(counts.values())=={3} and len(counts)==178
    csvsave(folder/'exposure.csv',[dict(video_id=v,presentations=n,role=rows[v]['role'],source_group_id=rows[v]['source_group_id'],
                                      supervised_frames=n*len(rows[v]['train_targets'])) for v,n in sorted(counts.items())])
    best=max(range(1,4),key=lambda e:(data.read(folder/f'epoch{e:02d}_dev.json')['groups']['新增15开发来源']['pixel_AP'],-e))
    for name,e in [('final',3),('best_development',best)]:
        link=folder/(name+'.pt')
        if not link.exists():link.symlink_to(f'epoch{e:02d}.pt')
    save(folder/'complete.json',dict(status='COMPLETE',updates=len(records),supervised_presentations=68634,best_epoch=best,
                                    final_epoch=3,identity=runtime_identity))
    del model,opt,initial,cp,records;gc.collect();torch.cuda.empty_cache();report(out)


def report(out):
    table=[];history=[];complete={}
    for mode in MODES:
        if (out/mode/'complete.json').exists():complete[mode]=data.read(out/mode/'complete.json')
        for epoch in [1,2,3]:
            path=out/mode/f'epoch{epoch:02d}_dev.json'
            if path.exists():
                ev=data.read(path);loss=data.read(out/mode/f'epoch{epoch:02d}_loss.json') if (out/mode/f'epoch{epoch:02d}_loss.json').exists() else {}
                monpath=out/mode/f'epoch{epoch:02d}_monitor.json'
                mon=data.read(monpath)['macro'] if monpath.exists() else {}
                history.append(dict(mode=mode,epoch=epoch,updates=epoch*178,**loss,**{'train_observation_'+k:v for k,v in mon.items()},**{k:ev['macro'][k] for k in METRICS}))
                for group,m in ev['groups'].items():table.append(dict(condition=mode,epoch=epoch,group=group,**m))
    if (out/'frozen_reference.json').exists():
        for group,m in data.read(out/'frozen_reference.json')['groups'].items():table.append(dict(condition='FROZEN_EPOCH01',epoch=0,group=group,**m))
    csvsave(out/'history.csv',history);csvsave(out/'group_metrics.csv',table)
    status='COMPLETE' if len(complete)==2 else 'PARTIAL'
    calls=collections.Counter()
    if (out/'frontend_calls.jsonl').exists():
        for line in (out/'frontend_calls.jsonl').open():calls[json.loads(line)['kind']]+=1
    resources={}
    for mode in complete:
        if (out/mode/'updates.csv').exists():
            rs=list(csv.DictReader((out/mode/'updates.csv').open()))
            resources[mode]=dict(train_seconds=sum(float(r['seconds']) for r in rs),
                peak_allocated_bytes=max(int(r['peak_allocated_bytes']) for r in rs),
                peak_reserved_bytes=max(int(r['peak_reserved_bytes']) for r in rs))
    save(out/'summary.json',dict(status=status,new_frontend_calls=dict(calls),new_DFormer=0,extra_BN=0,resources=resources,complete_modes=complete,scope='19-source development localization, not final test',history=history))
    lines=['# PVST完整视频匹配定位','计算状态：'+status+'。FRAME_ONLY/FULL_VIDEO两组都用XYZ；只比较跨时刻连接增量。',
           '末次epoch3是主结果；各自新增15开发AP选择最佳只是开发选择附表。未完成两组时不输出完整对照。',
           '|条件|轮|范围|AP|Dice|IoU|召回|real FPR|fake背景FPR|预测面积|常数AP|',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        if r['group'] in ['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照']:
            lines.append('|'+r['condition']+'|'+str(r['epoch'])+'|'+r['group']+'|'+'|'.join('未提供' if r.get(k) is None else f'{r[k]:.8f}' for k in METRICS)+'|')
    if status=='COMPLETE':
        a=data.read(out/'FRAME_ONLY/epoch03_dev.json');b=data.read(out/'FULL_VIDEO/epoch03_dev.json')
        assert [(f['video_id'],f['frame_index'],f['pts'],f['GT_pixels']) for f in a['frames']]==[(f['video_id'],f['frame_index'],f['pts'],f['GT_pixels']) for f in b['frames']]
        delta=[]
        for x,y in zip(a['sources'],b['sources']):
            assert x['source_group_id']==y['source_group_id']
            delta.append(dict(pair_id=x['pair_id'],**{k:y[k]-x[k] if x[k] is not None and y[k] is not None else None for k in METRICS}))
        csvsave(out/'FULL_VIDEO_minus_FRAME_ONLY_sources.csv',delta)
        best=[]
        for mode,c in complete.items():
            ev=data.read(out/mode/f"epoch{c['best_epoch']:02d}_dev.json")
            best.append(dict(mode=mode,selected_epoch=c['best_epoch'],**ev['macro']))
        csvsave(out/'best_development_supplement.csv',best)
        primary=[r for r in table if r['condition']=='FROZEN_EPOCH01' or r['epoch']==3]
        for group in a['groups']:
            primary.append(dict(condition='FULL_VIDEO_minus_FRAME_ONLY',epoch=3,group=group,**{k:b['groups'][group][k]-a['groups'][group][k] if a['groups'][group].get(k) is not None and b['groups'][group].get(k) is not None else None for k in METRICS}))
        csvsave(out/'primary_epoch3.csv',primary)
        lines.append('\n## 预定末次主表（epoch3）')
        lines+=['|条件|范围|AP|Dice|召回|real FPR|fake背景FPR|','|---|---|---:|---:|---:|---:|---:|']
        for r in primary:
            if r['group'] in ['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源']:
                lines.append('|'+r['condition']+'|'+r['group']+'|'+'|'.join('未提供' if r.get(k) is None else f'{r[k]:.8f}' for k in ['pixel_AP','Dice','GT_recall','real_FPR','fake_background_FPR'])+'|')
        lines.append('\n## 各自开发选择附表')
        for r in best:lines.append(f"- {r['mode']}：新增15开发AP选择epoch{r['selected_epoch']}，全部19开发AP={r['pixel_AP']:.8f}，Dice={r['Dice']:.8f}。")
        macro_delta={k:b['macro'][k]-a['macro'][k] for k in METRICS if a['macro'].get(k) is not None and b['macro'].get(k) is not None}
        lines.append('\n末次宏平均FULL_VIDEO−FRAME_ONLY：'+str(macro_delta))
        if macro_delta.get('pixel_AP',0)>0 and (macro_delta.get('Dice',0)<0 or macro_delta.get('GT_recall',0)<0):
            lines.append('AP提高但Dice或召回下降：排序与固定阈值操作点存在取舍，不能写成全面改善。')
        if macro_delta.get('real_FPR',0)>0 or macro_delta.get('fake_background_FPR',0)>0:
            lines.append('至少一类误报率上升，定位收益须与误报代价并列解释。')
        gains=[d for d in delta if d.get('pixel_AP') is not None and d['pixel_AP']>0]
        if gains:
            largest=max(gains,key=lambda d:d['pixel_AP'])
            lines.append(f"最大正AP增量来自{largest['pair_id']}，占所有正增量的{largest['pixel_AP']/sum(d['pixel_AP'] for d in gains):.2%}；不能用它替代完整来源表。")
        better=sum(d['pixel_AP']>0 for d in delta if d['pixel_AP'] is not None)
        lines += [f'\n末次逐来源AP改善{better}/19；完整来源差值见FULL_VIDEO_minus_FRAME_ONLY_sources.csv。',
                  'AP/Dice/召回与两类误报并列解释；梯度、loss和通路不等于效果，开发选择非独立测试。不得自动追加结构、seed或预算。']
    lines.append('\n新增前端实际调用（含显式恢复期间的成功调用）：'+str(dict(calls))+'；DFormer=0，额外BN=0。')
    lines.append('各模式训练资源：'+str(resources)+'。缓存队列与极值见queue_stats.csv；预算和有效曝光见各模式complete.json/exposure.csv。')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if history:
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        for mode in MODES:
            hs=[r for r in history if r['mode']==mode]
            ax[0].plot([r['epoch'] for r in hs],[r['pixel_AP'] for r in hs],label=mode)
            if all('loss' in r for r in hs):ax[1].plot([r['epoch'] for r in hs],[r['loss'] for r in hs],label=mode)
        ax[0].set_ylabel('All19 dev pixel AP');ax[1].set_ylabel('Mean online video loss')
        for a in ax:
            a.set_xlabel('Epoch')
            if a.get_legend_handles_labels()[0]:a.legend()
        fig.tight_layout();fig.savefig(out/'curves.png');plt.close(fig)
    preview_index(out)


def preview_index(out):
    if not (out/'protocol.json').exists():return
    lines=['<!doctype html><meta charset="utf-8"><title>PVST固定开发预览</title>',
           '<style>img{max-width:240px}td{vertical-align:top}table{margin-bottom:32px}</style>',
           '<h1>固定54目标；概率色阶统一0–1，二值阈值0.5</h1><p>RGB为模型工作RGB展示；GT为官方原生标注缩略。未完成输出显示缺失。</p>']
    for row in data.read(out/'protocol.json')['rows']:
        if row['split']!='dev':continue
        for i in row['preview_targets']:
            stem=row['video_id']+f'_{i:05d}'
            lines.append(f"<h3>{row['video_id']} frame {i}, PTS {row['pts'][i]:.6f}</h3><table><tr>")
            for name in ['rgb','gt','FROZEN_EPOCH01_prob','FROZEN_EPOCH01_mask','FRAME_ONLY_prob','FRAME_ONLY_mask','FULL_VIDEO_prob','FULL_VIDEO_mask']:
                path=out/'preview'/(stem+'_'+name+'.png')
                lines.append('<td>'+name+('<br><img src="'+path.name+'">' if path.exists() else '<br>未完成')+'</td>')
            lines.append('</tr></table>')
    (out/'preview').mkdir(exist_ok=True);(out/'preview/index.html').write_text('\n'.join(lines))


def smoke(out,p,identity,emit):
    # One full existing training video, one disposable update and one full
    # context restored inference; only native frame0 is scored for this check.
    row=next(r for r in p['rows'] if r['video_id']=='4ym5n4_fake')
    old=data.PROBE/row['video_id'];meta=data.read(old/'identity.json')
    assert data.observation_compatible(row,meta,identity['adaptation_sha256'],identity['provider_sha256'])
    folder=out/'smoke';assert not (folder/'result.json').exists(),'Existing smoke preserved'
    folder.mkdir(exist_ok=True);interface=folder/'transient_interface';paths=[]
    for i in range(row['actual_frames']):
        path=old/f'{i:05d}.pt';f=torch.load(path,map_location='cpu',weights_only=True)
        assert f['frame_index']==i and f['pts']==row['pts'][i]
        routed=prepare_frame(f)
        g={k:routed[k] for k in ['xyz_provider','valid','points','keys','index']}
        g.update(frame_index=i,pts=row['pts'][i],scale=data.SCALE,h=.5)
        data.atomic_torch(interface/'geometry'/row['video_id']/f'{i:05d}.pt',g);paths.append(str(path))
    save(interface/'visual'/row['video_id']/'frames.json',paths)
    frames=data.VideoCache(interface).load(row)
    seed();model=FullVideoDirect().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    rec=train_video(model,opt,frames,'FULL_VIDEO',emit)
    smoke_identity=dict(scope='disposable interface smoke',protocol=identity)
    before=state_hash(model.state_dict());cp=save_checkpoint(folder/'discarded.pt',model,opt,smoke_identity,'FULL_VIDEO',1,1,[dict(rec,video_id=row['video_id'])])
    del model,opt;gc.collect();torch.cuda.empty_cache()
    model=FullVideoDirect().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    restored=restore_checkpoint(folder/'discarded.pt',model,opt,smoke_identity,'FULL_VIDEO');assert restored['updates']==1 and state_hash(model.state_dict())==before
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):outputs=model.eval()(frames,'FULL_VIDEO')
    ev=aggregate_records(score_video(row,frames,outputs,wanted=[0]))
    save(folder/'result.json',dict(status='PASS',execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),T=len(frames),M=sum(len(f['keys']) for f in frames),
         main_forward_calls=2,optimizer_updates=1,full_context_inference_calls=1,scored_native_frames=1,
         restored_state_exact=True,train_record=rec,metric_fields_checked=list(ev['macro']),
         formal_training_started=False,new_MoGe=0,new_frozen_visual=0,exit_code=0))
    (folder/'discarded.pt').unlink();shutil.rmtree(interface);del model,opt,cp,restored,outputs,frames
    gc.collect();torch.cuda.empty_cache()


def exclusive_lock(out):
    out.mkdir(parents=True,exist_ok=True)
    handle=(out/'run.lock').open('a+')
    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close();raise RuntimeError('This output already has an active invocation; no waiting or duplicate run')
    return handle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=data.OUT)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output;lock=exclusive_lock(out);phase='plan';start=time.monotonic()
    def emit(stage,**kw):
        value=dict(status='RUNNING',stage=stage,**kw);value['invocation_seconds']=time.monotonic()-start;save(out/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
    try:
        torch.set_num_threads(4);p,identity=data.plan(out,emit)
        save(out/('invocation_'+str(time.time_ns())+'.json'),dict(
            execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            arguments=vars(args)|{'output':str(out)},torch_version=str(torch.__version__)))
        if args.plan_only:
            save(out/'status.json',dict(status='PLAN_READY',formal_training_started=False,exit_code=0));return
        if (out/'summary.json').exists() and data.read(out/'summary.json')['status']=='COMPLETE' and not args.smoke:
            print('EXISTING_COMPLETE: no prepare/train/evaluation repeated',flush=True);return
        assert torch.cuda.is_available(),'CUDA unavailable; no waiting/downgrade'
        if args.smoke:
            phase='smoke';smoke(out,p,identity,emit)
            save(out/'status.json',dict(status='SMOKE_PASS',formal_training_started=False,exit_code=0));return
        if any((out/m/'last.pt').exists() for m in MODES) and not args.resume:
            raise RuntimeError('Existing training checkpoint: explicit --resume required')
        phase='prepare';data.prepare_all(out,p,identity,emit)
        if args.prepare_only:
            save(out/'status.json',dict(status='PREPARED',formal_training_started=False,exit_code=0));return
        ensure_initial(out)
        phase='resource_gate';chunk=resource_gate(out,p,identity,emit)
        phase='reference';reference(out,[r for r in p['rows'] if r['split']=='dev'],data.read(out/'categories.json'))
        for mode in MODES:
            phase=mode;run_mode(out,p,identity,mode,chunk,args.resume,emit)
        report(out);save(out/'status.json',dict(status='COMPLETE',exit_code=0,seconds=time.monotonic()-start))
    except BaseException as exc:
        failure=dict(status='FAILED',stage=phase,last_progress=data.read(out/'status.json') if (out/'status.json').exists() else None,traceback=traceback.format_exc(),seconds=time.monotonic()-start,exit_code=130 if isinstance(exc,KeyboardInterrupt) else 1)
        save(out/('failure_'+str(time.time_ns())+'.json'),failure);save(out/'status.json',failure)
        report(out);raise
    finally:
        lock.close()


if __name__=='__main__':main()
