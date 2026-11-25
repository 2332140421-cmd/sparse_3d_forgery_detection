"""User-launched single official MumPy transfer baseline, no formal run by Codex.

Immediate caller is scripts/run_pvst_anakin_mumpy_baseline_v1.sh. Reuses existing
source-separated rows/native metrics; outputs self-contained epoch states,
original-grid metrics, fixed previews and report. No old experiment run/prepare.
"""
import argparse,collections,fcntl,gc,html,json,math,os,random,subprocess,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from . import data
from .model import MumPy,OFFICIAL,REVISION,WEIGHT,optimizers,losses
from research_tools.xyz_pilot.fullvideo_matched import aggregate_records,groups,persist_eval,csvsave,native_mask,numeric_members,rng_state,set_rng
from research_tools.xyz_pilot.fullvideo_context_diagnosis import native_record
from research_tools.xyz_pilot.fullvideo_direct_probe import save,digest,state_hash

REPO=Path(__file__).resolve().parents[2]
OUT=data.SOURCE.parent/'pvst_anakin_mumpy_baseline_v1'
CONFIG=REPO/'docs/pvst/anakin_mumpy_config_v1.json'


def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw),ensure_ascii=False),flush=True)

def seed():
    random.seed(20260909);np.random.seed(20260909);torch.manual_seed(20260909);torch.cuda.manual_seed_all(20260909)

def sync():torch.cuda.synchronize()

def memory():return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def identity(config):
    revision=subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()
    assert revision==REVISION
    assert not subprocess.check_output(['git','-C',str(OFFICIAL),'status','--porcelain'],text=True).strip(),'OFFICIAL_SOURCE_CHANGED'
    assert WEIGHT.is_file(),'Required author-linked initialization is missing; no random fallback'
    return dict(official_revision=revision,weight_sha256=digest(WEIGHT),weight_bytes=WEIGHT.stat().st_size,
        source_protocol_sha256=digest(data.SOURCE/'protocol.json'),source_categories_sha256=digest(data.SOURCE/'categories.json'),
        config=config,torch_version=str(torch.__version__),shared_helpers_sha256={name:digest(REPO/name) for name in ['research_tools/xyz_pilot/fullvideo_context_diagnosis.py','research_tools/xyz_pilot/fullvideo_matched.py','research_tools/xyz_pilot/fullvideo_matched_data.py','scripts/review_anakin.py']},code_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('data.py')]})


def freeze(out,config,ident):
    protocol=data.read(data.SOURCE/'protocol.json');rows=protocol['rows']
    train=data.targets(rows,'train');dev=data.targets(rows,'dev')
    assert len(train)==22878 and len(dev)==5152 and protocol['protected_test_sources']==94
    assert config['updates_per_epoch']==math.ceil(len(train)/config['effective_batch'])==1907
    assert config['total_updates']==config['epochs']*config['updates_per_epoch']==19070
    assert len({r['source_group_id'] for r in rows if r['split']=='train'})==89
    assert {r['source_group_id'] for r in rows if r['split']=='train'}.isdisjoint({r['source_group_id'] for r in rows if r['split']=='dev'})
    assert all(i not in range(6,10) for v,i in train if v=='40y1x9_fake')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():assert data.read(out/'identity.json')==ident,'PROTOCOL_OR_CODE_CHANGED'
    else:save(out/'identity.json',ident);save(out/'protocol.json',protocol);save(out/'config.json',config)
    orders=data.epoch_orders(rows);monitor=data.monitor_targets(rows)
    if (out/'orders.json').exists():assert data.read(out/'orders.json')==[[list(t) for t in order] for order in orders]
    else:save(out/'orders.json',orders)
    save(out/'monitor.json',monitor)
    counts=collections.Counter(v for order in orders for v,i in order)
    csvsave(out/'train_exposure.csv',[dict(video_id=r['video_id'],source_group_id=r['source_group_id'],role=r['role'],unique_targets=len(r['train_targets']),presentations=counts[r['video_id']]) for r in rows if r['split']=='train'])
    return protocol,orders,monitor


def prepare(root,rows):
    start=time.monotonic();cost=[data.prepare_video(root,r,emit) for r in rows]
    value=dict(new_RGB224_GT_frames=sum(c['new_frames'] for c in cost),reused_RGB224_GT_frames=sum(c['reused_frames'] for c in cost),
        new_decode_frames=0,new_MoGe=0,new_DINO=0,new_DFormer=0,seconds=time.monotonic()-start)
    save(root/f'preparation_{time.time_ns()}.json',value);return value


def batch_update(model,opts,scheds,dataset,ts,config,diagnostics=False):
    assert len(ts)<=config['effective_batch'];model.train();start=time.monotonic()
    for o in opts:o.zero_grad(set_to_none=True)
    sums=collections.Counter()
    for start_index in range(0,len(ts),config['micro_batch']):
        chosen=ts[start_index:start_index+config['micro_batch']];x,y=dataset.batch(chosen);x=x.cuda();y=y.cuda()
        z=model(x)
        loss,parts=losses(z,y)
        (loss*len(chosen)/len(ts)).backward()
        sums['loss']+=float(loss.detach())*len(chosen)/len(ts)
        for k,v in parts.items():sums[k]+=v*len(chosen)/len(ts)
    gradients={};families={
        'view1':lambda n:n.startswith('encoder.base.tokenize.project1'),
        'view2':lambda n:n.startswith('encoder.base.tokenize.project2'),
        'view3':lambda n:n.startswith('encoder.base.tokenize.project3'),
        'swin':lambda n:n.startswith('encoder.base.layers') and '.attn.' in n,
        'cva':lambda n:'.cva.' in n,
        'cva_internal':lambda n:'.cva.' in n and '.proj_out.' not in n,
        'global':lambda n:n.startswith('encoder.base.globalblocks'),
        'frequency_decoder':lambda n:n.startswith('decoder.decoder_frequency'),
        'localization_decoder':lambda n:n.startswith('decoder.') and not n.startswith('decoder.decoder_frequency')}
    all_grads=[p.grad for p in model.parameters() if p.grad is not None]
    norm=torch.stack(torch._foreach_norm(all_grads)).norm()
    assert torch.isfinite(norm),'NONFINITE_GRADIENT'
    for name,p in model.named_parameters() if diagnostics else []:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),('NONFINITE_GRADIENT',name)
            for family,belongs in families.items():
                if belongs(name):gradients[family]=gradients.get(family,0)+float(p.grad.float().square().sum())
    for o in opts:o.step()
    for s in scheds:s.step()
    sync()
    return dict(**sums,batch_targets=len(ts),gradient_norm=float(norm),gradient_L2={k:math.sqrt(v) for k,v in gradients.items()},seconds=time.monotonic()-start,**memory())


def checkpoint(path,model,opts,scheds,ident,epoch,updates,observations):
    value=dict(condition='MUMPY_ANAKIN',state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=[o.state_dict() for o in opts],scheduler=[s.state_dict() for s in scheds],mixed_precision_state=None,
        precision='official-style FP32; no scaler',identity=ident,epoch=epoch,next_epoch=epoch+1,updates=updates,
        target_presentations=observations,random_state=rng_state())
    value['model_sha256']=state_hash(value['state_dict']);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(value,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
    return value['model_sha256']


def restore(path,model,opts,scheds,ident,training=True):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['identity']==ident and cp['condition']=='MUMPY_ANAKIN'
    model.load_state_dict(cp['state_dict'],strict=True)
    assert state_hash(model.state_dict())==cp['model_sha256']
    if training:
        for o,state in zip(opts,cp['optimizer']):o.load_state_dict(state)
        for s,state in zip(scheds,cp['scheduler']):s.load_state_dict(state)
        set_rng(cp['random_state'])
    return {k:cp[k] for k in ['epoch','next_epoch','updates','target_presentations','model_sha256']}


def preview_frame(folder,row,i,gt,z,rgb,label):
    folder.mkdir(parents=True,exist_ok=True);stem=row['video_id']+f'_{i:05d}'
    h,w=gt.shape;size=(min(w,480),max(1,round(h*min(w,480)/w)))
    Image.fromarray(np.asarray(rgb)).resize(size).save(folder/(stem+'_rgb.png'))
    Image.fromarray((gt*255).astype(np.uint8)).resize(size,Image.Resampling.NEAREST).save(folder/(stem+'_gt.png'))
    prob=1/(1+np.exp(-np.clip(z,-80,80)))
    Image.fromarray(np.round(255*prob).astype(np.uint8)).resize(size).save(folder/(stem+'_'+label+'_prob.png'))
    Image.fromarray(((z>=0)*255).astype(np.uint8)).resize(size,Image.Resampling.NEAREST).save(folder/(stem+'_'+label+'_mask.png'))


@torch.no_grad()
def evaluate(model,cache,rows,ts,out,label,categories=None,preview=False):
    model.eval();start=time.monotonic();wanted=collections.defaultdict(list)
    for vid,i in ts:wanted[vid].append(i)
    records=[];byid={r['video_id']:r for r in rows}
    for vi,(vid,indices) in enumerate(wanted.items()):
        r=byid[vid];assert len(indices)==len(set(indices))
        archive=zipfile.ZipFile(r['mask_zip']) if r['role']=='fake' else None
        members=numeric_members(archive.namelist()) if archive else None
        if archive:assert len(members)==r['actual_frames']
        try:
            for bi in range(0,len(indices),2):
                selected=indices[bi:bi+2];x,_=cache.batch([(vid,i) for i in selected])
                z=model(x.cuda())
                for i,logit in zip(selected,z[:,0]):
                    gt=native_mask(archive,members,r,i);rec=native_record(r,i,gt,logit)
                    rec.update(input_frame_indices=data.clip_indices(i,r['actual_frames']),input_pts=[r['pts'][j] for j in data.clip_indices(i,r['actual_frames'])],target_slot=1)
                    counts=data.read(cache.root/'rgb224'/vid/'complete.json')['native_GT_pixels']
                    assert rec['GT_pixels']==counts[i]
                    records.append(rec)
                    if preview and i in r['preview_targets']:
                        native=F.interpolate(logit.float()[None,None],size=r['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                        rgb=cache.arrays_for(vid)[0][i]
                        preview_frame(out/'preview',r,i,gt,native,rgb,label)
                if bi%20==0 or bi+len(selected)==len(indices):emit('evaluate',label=label,video=vid,video_position=vi+1,videos=len(wanted),frames=min(bi+2,len(indices)),T=len(indices),seconds=time.monotonic()-start)
        finally:
            if archive:archive.close()
    assert len(records)==len(ts)==len({(f['video_id'],f['frame_index']) for f in records})
    ev=aggregate_records(records);ev['seconds']=time.monotonic()-start
    if categories:ev['groups']=groups(ev,categories)
    # Empty fake masks retain their error measurements, not a fabricated AP.
    ev['fake_empty']=dict(frames=sum(f['role']=='fake' and f['GT_pixels']==0 for f in records),
        FPR=float(np.mean([f['fake_empty_FPR'] for f in records if f['fake_empty_FPR'] is not None])) if any(f['fake_empty_FPR'] is not None for f in records) else None)
    persist_eval(out/(label+'.json'),ev)
    return ev


def report(out,rows):
    evs=[]
    for e in range(1,11):
        path=out/f'epoch{e:02d}_dev.json'
        if path.exists():evs.append((e,data.read(path)))
    lines=['# MumPy官方结构的ANAKIN迁移（程序报告）','',
        '本模型不是XYZ C。通用作者提供初始化，完整三视图Swin/可变形交互/全局时间Transformer/DCT/多金字塔Decoder；同源视频片段只输出中帧。',
        '训练89/178视频，开发19/38视频；每轮22878合法目标、5152开发帧。10完整轮，共228780目标呈现；正式有效batch12、micro2，每轮1907有效更新（末批6目标）、总19070。',
        'loss为官方WeightedFocal(alpha=[1,1],gamma2)+softIoU，非BCE/Dice。官方三组SGD/多项式调度；迁移采用官方可选SingleAugment和FP32，不是作者原数据/完整预算复现。',
        '', '|轮|全部19 AP|Dice|召回|real FPR|新增15 AP|','|---|---:|---:|---:|---:|---:|']
    for e,ev in evs:
        a=ev['macro'];lines.append(f"|{e}|{a['pixel_AP']:.8f}|{a['Dice']:.8f}|{a['GT_recall']:.8f}|{a['real_FPR']:.8f}|{ev['groups']['新增15开发来源']['pixel_AP']:.8f}|")
    if evs:
        selected=max(evs,key=lambda p:(p[1]['groups']['新增15开发来源']['pixel_AP'],-p[0]))
        save(out/'selection.json',dict(rule='ADDITIONAL_DEV source macro AP, earliest tie',epoch=selected[0],checkpoint=str(out/f'epoch{selected[0]:02d}.pt'),metrics=selected[1]['macro']))
        evaluation_complete=len(evs)==10
        fixed=[(r,i) for r in rows if r['split']=='dev' for i in r['preview_targets']]
        preview_complete=all((out/'preview'/(r['video_id']+f'_{i:05d}_'+label+'_'+kind+'.png')).exists() for r,i in fixed for label in ['last','selected'] for kind in ['prob','mask'])
        complete=evaluation_complete and all((out/f'epoch{e:02d}.pt').exists() for e in range(1,11)) and preview_complete
        summary=dict(status='COMPLETE' if complete else 'PARTIAL',epochs_evaluated=[e for e,_ in evs],selected_epoch=selected[0],
            selected=selected[1]['macro'],last=evs[-1][1]['macro'],updates=19070 if complete else None,
            formal_training_complete=complete,evaluation_complete=evaluation_complete,preview_complete=preview_complete,new_MoGe=0,new_DINO=0,new_DFormer=0)
        save(out/'summary.json',summary)
        lines+=['',f"开发选择轮次{selected[0]}；末次主结果为第10轮，选择结果不称独立测试成绩。"]
        lines+=['','|检查点/分组|AP|Dice|IoU|召回|real FPR|fake背景FPR|','|---|---:|---:|---:|---:|---:|---:|']
        for label,(e,ev) in [('selected',selected),('last_available',evs[-1])]:
            for group in ['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照']:
                if group in ev['groups']:
                    g=ev['groups'][group];lines.append('|'+label+'/epoch'+str(e)+'/'+group+'|'+'|'.join(f'{g[k]:.8f}' for k in ['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR'])+'|')
        lines+=['','|检查点|来源|AP|Dice|召回|real FPR|','|---|---|---:|---:|---:|---:|']
        for label,(e,ev) in [('selected',selected),('last_available',evs[-1])]:
            for r in ev.get('sources',[]):lines.append('|'+label+'/epoch'+str(e)+'|'+r['pair_id']+'|'+'|'.join(f'{r[k]:.8f}' if r[k] is not None else 'null' for k in ['pixel_AP','Dice','GT_recall','real_FPR'])+'|')
        csvsave(out/'group_metrics.csv',[dict(epoch=e,group=g,**v) for e,ev in evs for g,v in ev['groups'].items()])
    else:lines+=['','正式完整开发评价尚未产生；短预检不替代正式训练成绩。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    folder=out/'preview';folder.mkdir(exist_ok=True)
    page=['<!doctype html><meta charset="utf-8"><title>MumPy固定预览</title><h1>既定开发目标，统一概率0–1/阈值0.5</h1>']
    labels=['last','selected']
    for r in rows:
        if r['split']!='dev':continue
        for i in r['preview_targets']:
            stem=r['video_id']+f'_{i:05d}'
            page.append(f'<h3>{html.escape(r["video_id"])} frame {i} PTS {r["pts"][i]:.6f}</h3>')
            for suffix in ['rgb','gt']+[s+'_'+t for s in labels for t in ['prob','mask']]:
                path=folder/(stem+'_'+suffix+'.png')
                if path.exists():page.append(f'<label>{suffix}<img width="16%" src="{path.name}"></label>')
    (folder/'index.html').write_text('\n'.join(page))


def smoke(root,out,protocol,config,ident):
    """Two discard updates over true targets, real boundary/empty-mask cases."""
    out.mkdir(parents=True,exist_ok=True);rows=protocol['rows'];byid={r['video_id']:r for r in rows}
    chosen=[('4ym5n4_fake',0),('40y1x9_fake',10),('40y1x9_real',0)]
    selected=[byid[v] for v in dict(chosen)];cost=prepare(root,selected)
    seed();model=MumPy().cuda();opts,scheds=optimizers(model,config['total_updates']);cache=data.Clips(root,selected,training=True)
    initial=state_hash(model.state_dict());save(out/'initialization.json',dict(loading=model.loading,model_sha256=initial,parameters=sum(p.numel() for p in model.parameters())))
    # Passive stage-shape hooks, no capture mode or architecture change.
    shapes={};hooks=[]
    def capture(name):
        def hook(_m,_a,result):
            def shape(x):return list(x.shape) if isinstance(x,torch.Tensor) else [shape(y) for y in x]
            shapes[name]=shape(result)
        return hook
    for name,module in [('tokenize',model.encoder.base.tokenize),('encoder',model.encoder),('decoder',model.decoder)]:hooks.append(module.register_forward_hook(capture(name)))
    records=[];torch.cuda.reset_peak_memory_stats()
    for step in [1,2]:
        rec=batch_update(model,opts,scheds,cache,chosen*(config['effective_batch']//len(chosen)),config,diagnostics=True);records.append(dict(step=step,**rec));emit('smoke_update',step=step,**rec)
    for hook in hooks:hook.remove()
    assert all(records[-1]['gradient_L2'].get(k,0)>0 for k in ['view1','view2','view3','swin','cva','cva_internal','global','frequency_decoder','localization_decoder'])
    after=state_hash(model.state_dict());assert after!=initial
    path=out/'discarded.pt';checkpoint(path,model,opts,scheds,ident,0,2,2*config['effective_batch'])
    expected_rng=rng_state();loaded=restore(path,model,opts,scheds,ident)
    assert loaded['model_sha256']==after
    assert torch.equal(rng_state()['torch'],expected_rng['torch'])
    assert [s.last_epoch for s in scheds]==[2,2,2]
    ev=evaluate(model,data.Clips(root,selected),selected,chosen,out,'smoke_eval',preview=False)
    # Exercise real preview+JSON/CSV/report writes, using the predetermined smoke targets.
    model.eval()
    with torch.no_grad():
        for target in chosen:
            row=dict(byid[target[0]],split='dev',preview_targets=[target[1]])
            x,_=cache.get(target,augment=False)
            z=model(x.cuda()[None])[0,0]
            archive=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
            try:gt=native_mask(archive,numeric_members(archive.namelist()) if archive else None,row,target[1])
            finally:
                if archive:archive.close()
            native=F.interpolate(z.float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            preview_frame(out/'preview',row,target[1],gt,native,cache.arrays_for(target[0])[0][target[1]],'last')
    # The official user protocol excludes targets6-9 even if clip needs their RGB.
    assert data.clip_indices(10,byid['40y1x9_fake']['actual_frames'])==[9,10,11]
    with_mask_exclusion=False
    try:cache.get(('40y1x9_fake',6),augment=False)
    except AssertionError:with_mask_exclusion=True
    assert with_mask_exclusion
    report(out,[dict(byid[v],split='dev',preview_targets=[i]) for v,i in chosen])
    result=dict(status='SMOKE_PASS_FORMAL_NOT_STARTED',exit_code=0,updates=2,target_presentations=2*config['effective_batch'],steps=records,
        checkpoint_restore='PASS',initialization=model.loading,shapes=shapes,parameters=sum(p.numel() for p in model.parameters()),
        preparation=cost,metrics_for_interface_only=ev['macro'],new_MoGe=0,new_DINO=0,new_DFormer=0,
        coarse_training_seconds=10*22878/12*np.mean([r['seconds'] for r in records]),
        estimate_note='two cold updates only, excludes full dev/observation/native AP/checkpoints and sustained throughput',**memory())
    save(out/'result.json',result);save(root/'smoke_latest.json',dict(path=str(out),status=result['status']))
    emit('SMOKE_COMPLETE',status=result['status'],output=str(out),steps=2,**memory())
    del model,opts,scheds;gc.collect();torch.cuda.empty_cache()


def run(root,out,p,orders,monitor,config,ident,resume):
    rows=p['rows'];train=data.targets(rows,'train');dev=data.targets(rows,'dev');cats=data.read(data.SOURCE/'categories.json')
    if (out/'initialization.json').exists() or (out/'epoch01.pt').exists():assert resume,'Use --resume, never silently restart'
    if (out/'status.json').exists() and data.read(out/'status.json')['status']=='COMPLETE':
        report(out,rows);emit('already_complete',output=str(out));return
    prepare(root,rows);dataset=data.Clips(root,rows,training=True);evalset=data.Clips(root,rows)
    seed();model=MumPy(initialize=not (out/'epoch01.pt').exists()).cuda();opts,scheds=optimizers(model,config['total_updates'])
    if not (out/'initialization.json').exists():save(out/'initialization.json',dict(model_sha256=state_hash(model.state_dict()),loading=model.loading,parameters=sum(p.numel() for p in model.parameters())))
    updates=0;presentations=0;begin=time.monotonic();timings=[]
    for epoch,order in enumerate(orders,1):
        save(out/'status.json',dict(status='RUNNING',stage='epoch',epoch=epoch,updates=updates,target_presentations=presentations))
        ep=out/f'epoch{epoch:02d}.pt';evaluation=out/f'epoch{epoch:02d}_dev.json'
        if ep.exists():
            if evaluation.exists():
                updates=epoch*config['updates_per_epoch'];presentations=epoch*22878;continue
            meta=restore(ep,model,opts,scheds,ident)
            assert meta['epoch']==epoch and meta['updates']==epoch*config['updates_per_epoch']
            updates=epoch*config['updates_per_epoch'];presentations=epoch*22878
        else:
            if epoch>1:
                previous=out/f'epoch{epoch-1:02d}.pt';cp=restore(previous,model,opts,scheds,ident)
                assert cp['epoch']==epoch-1 and cp['updates']==updates and cp['target_presentations']==presentations
            losses_epoch=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            for j in range(0,len(order),config['effective_batch']):
                rec=batch_update(model,opts,scheds,dataset,order[j:j+config['effective_batch']],config);updates+=1;presentations+=len(order[j:j+config['effective_batch']]);timings.append(rec['seconds']);losses_epoch.append(rec)
                if updates%10==0 or j+config['effective_batch']>=len(order):emit('train',epoch=epoch,epochs=10,batch=j//config['effective_batch']+1,batches=config['updates_per_epoch'],update=updates,total_updates=config['total_updates'],
                    target_presentations=presentations,lr=[o.param_groups[0]['lr'] for o in opts],elapsed_seconds=time.monotonic()-begin,
                    estimated_remaining_train_seconds=np.mean(timings[-100:])*(config['total_updates']-updates),**rec)
            checkpoint(ep,model,opts,scheds,ident,epoch,updates,presentations)
            save(out/f'epoch{epoch:02d}_loss.json',dict(epoch=epoch,updates=updates,target_presentations=presentations,
                **{k:float(sum(r[k]*r['batch_targets'] for r in losses_epoch)/sum(r['batch_targets'] for r in losses_epoch)) for k in ['loss','weighted_focal','soft_IoU']},seconds=time.monotonic()-start,**memory()))
        observer=out/f'epoch{epoch:02d}_monitor.json'
        save(out/'status.json',dict(status='RUNNING',stage='epoch_evaluation',epoch=epoch,updates=updates,target_presentations=presentations))
        if not observer.exists():evaluate(model,evalset,rows,monitor,out,f'epoch{epoch:02d}_monitor')
        if not evaluation.exists():ev=evaluate(model,evalset,rows,dev,out,f'epoch{epoch:02d}_dev',cats)
        else:ev=data.read(evaluation)
        emit('epoch_summary',epoch=epoch,updates=updates,target_presentations=presentations,development=ev['macro'],training_observation=data.read(observer)['macro'])
        report(out,rows)
    assert updates==config['total_updates'] and presentations==10*len(train),'FINAL_BUDGET_COVERAGE'
    # Same54 fixed targets; predict only these for last/selected previews.
    report(out,rows);selected=data.read(out/'selection.json')['epoch']
    fixed=[(r['video_id'],i) for r in rows if r['split']=='dev' for i in r['preview_targets']]
    for name,e in [('last',10),('selected',selected)]:
        meta=restore(out/f'epoch{e:02d}.pt',model,opts,scheds,ident,training=False)
        assert meta['epoch']==e and meta['updates']==e*config['updates_per_epoch']
        evaluate(model,evalset,rows,fixed,out,'preview_'+name,preview=True)
        # Keep image labels short while preserving separate metrics.
        for image in (out/'preview').glob('*_preview_'+name+'_*png'):image.rename(image.with_name(image.name.replace('_preview_'+name+'_','_'+name+'_')))
    report(out,rows)
    save(out/'status.json',dict(status='COMPLETE',exit_code=0,updates=updates,target_presentations=presentations,seconds=time.monotonic()-begin))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true');parser.add_argument('--checkpoint',type=Path);parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True);out=(root/'smoke'/str(time.time_ns())) if args.smoke else root/'formal';out.mkdir(parents=True,exist_ok=True)
    stage='identity';started=time.monotonic();lock=None
    try:
        lock=(out/'invocation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.report_only:report(out,data.read(out/'protocol.json')['rows']);return
        config=data.read(CONFIG);ident=identity(config);p,orders,monitor=freeze(out,config,ident)
        if args.prepare_only:prepare(root,p['rows']);return
        assert torch.cuda.is_available(),'CUDA_UNAVAILABLE'
        torch.set_num_threads(4);stage='smoke' if args.smoke else 'formal'
        if args.smoke:smoke(root,out,p,config,ident)
        elif args.evaluate_only:
            assert args.checkpoint and args.checkpoint.is_file(),'Supply --checkpoint for evaluation-only'
            model=MumPy(initialize=False).cuda();opts,scheds=optimizers(model,config['total_updates'])
            meta=restore(args.checkpoint,model,opts,scheds,ident,training=False)
            assert 1<=meta['epoch']<=10 and meta['updates']==meta['epoch']*config['updates_per_epoch'],'FORMAL_CHECKPOINT_REQUIRED'
            evaluate(model,data.Clips(root,p['rows']),p['rows'],data.targets(p['rows'],'dev'),out,'independent_'+args.checkpoint.stem,data.read(data.SOURCE/'categories.json'))
        else:run(root,out,p,orders,monitor,config,ident,args.resume)
    except Exception as exc:
        value=dict(status='FAILED',stage=stage,error=str(exc),traceback=traceback.format_exc(),exit_code=1,seconds=time.monotonic()-started)
        save(out/f'failure_{time.time_ns()}.json',value)
        if not (out/'status.json').exists() or data.read(out/'status.json').get('status')!='COMPLETE':save(out/'status.json',value)
        print(value['traceback'],flush=True);raise SystemExit(1)
    finally:
        if lock:lock.close()


if __name__=='__main__':main()
