"""Frozen C extraction -> two matched slab relation readouts -> fixed final eval.

One experiment CLI, using original Video/forward/build. Cache only frame slabs;
no full-resolution features/maps, no old relation or prediction as input.
"""
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from .data import Video,frames,save
from .experiment import REPO,forward,git,state_hash,average_precision_score
from .set_fit_check import build
from .slice_relation_readout import SliceRelationReadout,CONDITIONS,project_gt,windows
from scripts.prepare_v8_assets import digest

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1')
OUT=BASE.parent/'pvst_slice_relation_readout_v1'
DOC='docs/pvst/slice_relation_readout_v1.md'
SEED=20260909


def read(p):return json.loads(Path(p).read_text())
def csvsave(p,rows):
    if rows:
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(a):return float(np.mean(a)) if a else None

def progress(stage,**extra):
    save(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'stage':stage,**extra});print(stage,json.dumps(extra),flush=True)


class StopBeforeOldRelations(Exception):pass


@torch.no_grad()
def extract(model,rows,clips,c):
    """Capture actual pool output and assert it is exactly the sets input.

    Abort at sets pre-hook, so old sets/GRU/heads are never executed. Frame-local
    pool is independent of window length; dev tail frames are extracted once.
    """
    cache={};records=[];started=time.monotonic();folder=OUT/'cache';folder.mkdir()
    for r in rows:
        wanted=sorted({i for start,end,_ in windows(r,clips if r['split']=='train' else None) for i in range(start,end)})
        v=Video(r,c);data={};pending=[];fs=[]
        def pool_post(module,args,output):pending.append(output)
        def sets_pre(module,args):
            encoded=args[0];assert len(encoded)==len(fs)==len(pending)
            for f,(feature,ids),(observed,oid) in zip(fs,encoded,pending,strict=True):
                assert torch.equal(feature,observed) and torch.equal(ids,oid)
                sid,index,counts,edits=project_gt(f['xyz'].to(ids.device),f['valid'].to(ids.device),f['gt'].to(ids.device))
                assert torch.equal(sid,ids) and feature.shape==(len(ids),512) and torch.isfinite(feature).all()
                i=f['frame_index'];assert i not in data and f['pts']==r['pts'][i]['pts_s']
                if r['role']=='real':assert not f['gt'].any()
                data[i]={'features':feature.float().cpu(),'ids':ids.cpu(),'pts':f['pts'],'counts':counts.cpu(),'edit_counts':edits.cpu(),'GT_pixels':int(f['gt'].sum()),'valid_GT_pixels':int(edits.sum()),'native_pixels':f['gt'].numel()}
            raise StopBeforeOldRelations()
        handles=[model.pool.register_forward_hook(pool_post),model.sets.register_forward_pre_hook(sets_pre)]
        try:
            for k in range(0,len(wanted),24):
                fs=[v.frame(i) for i in wanted[k:k+24]];pending.clear()
                try:forward(model,fs)
                except StopBeforeOldRelations:pass
                else:raise ValueError('OLD_RELATION_EXECUTED')
                pending.clear();fs=[]
        finally:
            for h in handles:h.remove()
            v.close()
        assert sorted(data)==wanted
        torch.save(data,folder/(r['video_id']+'.pt'));cache[r['video_id']]=data
        for i,f in data.items():records.append({'video_id':r['video_id'],'pair_id':r['pair_id'],'split':r['split'],'role':r['role'],'frame_index':i,'pts':f['pts'],'slabs':len(f['ids']),'positive_slabs':int((f['edit_counts']>0).sum()),'valid_pixels':int(f['counts'].sum()),'GT_pixels':f['GT_pixels'],'valid_GT_pixels':f['valid_GT_pixels']})
        progress('extract',video_id=r['video_id'],frames=len(data),slabs=sum(len(f['ids']) for f in data.values()))
    csvsave(OUT/'cache_frames.csv',records)
    save(OUT/'cache_identity.json',{'weight_sha256':digest(BASE/'RGB_DEPTH_PARTITION/last.pt'),'manifest_sha256':digest(BASE/'manifest.json'),'frames':len(records),'train_frames':sum(r['split']=='train' for r in records),'dev_frames':sum(r['split']=='dev' for r in records),'seconds':time.monotonic()-started,'files':{p.name:digest(p) for p in sorted(folder.glob('*.pt'))},'bytes':sum(p.stat().st_size for p in folder.glob('*.pt')),'feature':'exact SlabPool output == sets pre-hook input; stop before old relations; FP32 C512; all occupied IDs and all valid native pixels'})
    return cache


def window_frames(cache,video_id,lo,end):
    return [{**f,'features':f['features'].cuda(),'ids':f['ids'].cuda()} for i in range(lo,end) for f in [cache[video_id][i]]]


def new_model(initial):
    model=SliceRelationReadout().cuda();model.load_state_dict(initial,strict=True);return model


def update(model,opt,fs,pos_weight):
    model.train();opt.zero_grad(set_to_none=True);total=sum(len(f['ids']) for f in fs)
    if not total:raise ValueError('EMPTY_TRAIN_CLIP')
    loss_sum=0.;info={'queries':0,'candidate_edges':0,'empty_queries':0};start=time.monotonic()
    for t,f in enumerate(fs):
        z,a=model(fs,t,model.history);y=(f['edit_counts']>0).to(device='cuda',dtype=torch.float32)
        for k in info:info[k]+=a[k]
        if not len(z):continue
        loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=pos_weight,reduction='sum')/total
        if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
        loss.backward();loss_sum+=float(loss.detach())
    gs=[p.grad for p in model.parameters() if p.grad is not None]
    if not gs or not all(bool(torch.isfinite(g).all()) for g in gs):raise ValueError('NONFINITE_GRADIENT')
    norm=float(torch.sqrt(sum(g.float().square().sum() for g in gs)));opt.step();torch.cuda.synchronize()
    return {'loss':loss_sum,'gradient_norm':norm,'seconds':time.monotonic()-start,**info}


@torch.no_grad()
def evaluate_readout(model,cache,rows,clips=None):
    model.eval();records=[];scores={};frame_rows=[];started=time.monotonic()
    for r in rows:
        aps=[];positive=total=allpos=allneg=empty=0;seen=set();emptyq=edges=0
        for start,end,lo in windows(r,clips):
            fs=window_frames(cache,r['video_id'],lo,end)
            for i in range(start,end):
                if i in seen:raise ValueError('DUPLICATE_SCORED_FRAME')
                seen.add(i);f=fs[i-lo];z,a=model(fs,i-lo,model.history);z=z.float().cpu().numpy();y=(f['edit_counts']>0).numpy()
                assert np.isfinite(z).all();positive+=int(y.sum());total+=len(y);emptyq+=a['empty_queries'];edges+=a['candidate_edges'];ap=None
                if not len(y):empty+=1
                elif y.all():allpos+=1
                elif not y.any():allneg+=1
                else:ap=float(average_precision_score(y,z));aps.append(ap)
                scores[(r['video_id'],i)]=z
                frame_rows.append({'video_id':r['video_id'],'pair_id':r['pair_id'],'role':r['role'],'frame_index':i,'pts':f['pts'],'slabs':len(y),'positive_slabs':int(y.sum()),'slice_AP':ap})
            del fs
        assert seen=={i for s,e,_ in windows(r,clips) for i in range(s,e)}
        records.append({'video_id':r['video_id'],'pair_id':r['pair_id'],'role':r['role'],'frames':len(seen),'evaluable_frames':len(aps),'all_positive_frames':allpos,'all_negative_frames':allneg,'empty_frames':empty,'positive_slabs':positive,'slabs':total,'positive_fraction':positive/total if total else None,'slice_AP':mean(aps),'empty_queries':emptyq,'candidate_edges':edges})
    return {'slice_AP':mean([r['slice_AP'] for r in records if r['role']=='fake' and r['slice_AP'] is not None]),'records':records,'frames':len(frame_rows),'frame_records':frame_rows,'seconds':time.monotonic()-started},scores


@torch.no_grad()
def backfill(scores_by_condition,cache,rows,c,clips=None):
    perframe={k:[] for k in CONDITIONS};records={k:[] for k in CONDITIONS};started=time.monotonic()
    for r in rows:
        v=Video(r,c);aps={k:[] for k in CONDITIONS};coverage=[];gt_total=valid_gt=0
        try:
            for start,end,_ in windows(r,clips):
                for i in range(start,end):
                    f=v.frame(i);ids,index,counts,edits=project_gt(f['xyz'],f['valid'],f['gt']);stored=cache[r['video_id']][i]
                    assert torch.equal(ids,stored['ids']) and torch.equal(counts,stored['counts']) and torch.equal(edits,stored['edit_counts'])
                    gt=f['gt'][f['valid']].numpy();n=int(f['gt'].sum());g=int(gt.sum());gt_total+=n;valid_gt+=g
                    if n:coverage.append(g/n)
                    for condition in CONDITIONS:
                        z=scores_by_condition[condition][(r['video_id'],i)];assert len(z)==len(ids)
                        if g:
                            # Exactly same valid native pixels for both, full slab score shared.
                            ap=float(average_precision_score(gt,z[index.numpy()]));aps[condition].append(ap)
                        else:ap=None
                        perframe[condition].append({'video_id':r['video_id'],'frame_index':i,'valid_pixels':len(gt),'valid_GT_pixels':g,'backfill_pixel_AP':ap})
        finally:v.close()
        for condition in CONDITIONS:records[condition].append({'video_id':r['video_id'],'pair_id':r['pair_id'],'role':r['role'],'backfill_valid_pixel_AP':mean(aps[condition]),'positive_GT_frames':len(aps[condition]),'GT_coverage_mean':mean(coverage),'GT_pixels':gt_total,'valid_GT_pixels':valid_gt,'GT_coverage_pixel_weighted':valid_gt/gt_total if gt_total else None})
    return {k:{'macro_AP':mean([r['backfill_valid_pixel_AP'] for r in records[k] if r['role']=='fake' and r['backfill_valid_pixel_AP'] is not None]),'records':records[k],'seconds_shared':time.monotonic()-started} for k in CONDITIONS}


def main():
    assert git('branch','--show-current')=='pvst-slice-relation-readout'
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt';m=read(BASE/'manifest.json');identity=read(BASE/'identity.json');c=read(BASE/'config.json');orders=read(BASE/'orders.json')
    if OUT.exists():
        s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if s.get('status')=='COMPLETE' and s.get('weight_sha256')==digest(weight) and s.get('manifest_sha256')==digest(BASE/'manifest.json'):print('REUSE COMPLETE');return
        raise ValueError('EXISTING_OUTPUT_INSPECT_NO_RETRY')
    assert not git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice')
    OUT.mkdir();started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(SEED);train=[r for r in m['rows'] if r['split']=='train'];dev=[r for r in m['rows'] if r['split']=='dev'];clips=m['clips']
    assert len(train)==16 and len(dev)==8 and len(clips)==48
    assert len({(x['video_id'],i) for x in clips for i in range(x['start'],x['end'])})==1152 and sum(r['actual_frames'] for r in dev)==924
    assert {r['source_id'] for r in train}.isdisjoint({r['source_id'] for r in dev})
    for filename,key in [('manifest.json','frozen_manifest_sha256'),('config.json','config_sha256'),('orders.json','orders_sha256')]:assert digest(BASE/filename)==identity[key]
    for path,sha in identity['code'].items():assert digest(REPO/path)==sha
    for r in train+dev:
        assert digest(Path(r['video_path']))==r['video_sha256']
        if r['role']=='fake':assert digest(Path(r['mask_zip']))==r['mask_sha256']
        for path,sha in next(a for a in identity['assets'] if a['video_id']==r['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha
        v=Video(r,c);v.close()
    canonical=lambda x:sorted((a['video_id'],a['start'],a['end']) for a in x)
    for epoch in range(1,13):assert canonical(orders[str(epoch)])==canonical(clips)
    run_identity={'execution_head':git('rev-parse','HEAD'),'weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'orders_sha256':digest(BASE/'orders.json'),'baseline_identity':identity['identity'],'seed':SEED,'epochs':12,'updates':576,'feature':'frozen C512 pre-sets','position':'shared learned Linear1-128-GELU-512; absolute id*.125 depth, signed actual PTS delta','precision':'original frozen BF16 forward, FP32 cached C and readout','optimizer':{'AdamW_lr':1e-4,'weight_decay':.01,'gradient_clipping':None},'attention_dropout':0.,'loss':'weighted slab BCE sum / all valid slabs per clip; train neg/pos fixed; no other loss'}
    save(OUT/'identity.json',run_identity)
    for filename in ['manifest.json','orders.json']:shutil.copyfile(BASE/filename,OUT/filename)
    model=build(c,OUT,rgb_only=True).eval();cp=torch.load(weight,map_location='cpu',weights_only=True)
    assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==identity['identity']
    assert read(OUT/'initialization.json')['model_sha256']==cp['initial_model_sha256'];model.load_state_dict(cp['state_dict'],strict=True);del cp
    model.requires_grad_(False);before=state_hash(model.state_dict());cache=extract(model,train+dev,clips,c)
    assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters());del model;torch.cuda.empty_cache()
    positive=sum(int((f['edit_counts']>0).sum()) for r in train for f in cache[r['video_id']].values());total=sum(len(f['ids']) for r in train for f in cache[r['video_id']].values());assert 0<positive<total
    pos_weight=torch.tensor((total-positive)/positive,device='cuda');save(OUT/'supervision.json',{'train_positive_slabs':positive,'train_negative_slabs':total-positive,'pos_weight':float(pos_weight),'GT_positive_rule':'at least one valid edited pixel','GT_coverage_in_cache_frames_csv':True})
    torch.manual_seed(SEED);a=SliceRelationReadout();initial={k:v.clone() for k,v in a.state_dict().items()};initial_hash=state_hash(initial);params=sum(p.numel() for p in a.parameters());del a
    save(OUT/'readout_initialization.json',{'state_dict_sha256':initial_hash,'parameters':params,'conditions':list(CONDITIONS),'same_initialization':True})
    fs=window_frames(cache,clips[0]['video_id'],clips[0]['start'],clips[0]['end']);checks={}
    for condition in CONDITIONS:
        a=new_model(initial);a.history=condition=='CURRENT_HISTORY';assert state_hash(a.state_dict())==initial_hash;opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01);torch.cuda.reset_peak_memory_stats()
        record=update(a,opt,fs,pos_weight);assert state_hash(a.state_dict())!=initial_hash and len(opt.state)>0
        record.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),optimizer_states=len(opt.state));checks[condition]=record;del a,opt
    del fs;save(OUT/'real_clip_checks.json',{'clip':clips[0],'passed':True,'conditions':checks,'discarded_updates':True});torch.cuda.empty_cache()
    results={};history=[];dev_scores={};train_scores={}
    for condition in CONDITIONS:
        folder=OUT/condition;folder.mkdir();a=new_model(initial);a.history=condition=='CURRENT_HISTORY';assert state_hash(a.state_dict())==initial_hash;opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01);torch.cuda.reset_peak_memory_stats();condition_start=time.monotonic();updates=0;train_seconds=eval_seconds=0.
        with (folder/'updates.jsonl').open('w') as logfile:
            for epoch in range(1,13):
                losses=[];epoch_start=time.monotonic()
                for clip in orders[str(epoch)]:
                    fs=window_frames(cache,clip['video_id'],clip['start'],clip['end']);record=update(a,opt,fs,pos_weight);del fs;updates+=1;losses.append(record['loss']);train_seconds+=record['seconds'];record.update(epoch=epoch,update=updates,clip=clip);logfile.write(json.dumps(record)+'\n');logfile.flush()
                evaluation,scores=evaluate_readout(a,cache,dev);eval_seconds+=evaluation['seconds'];save(folder/f'epoch{epoch:02d}_dev.json',evaluation)
                h={'condition':condition,'epoch':epoch,'updates':updates,'train_loss':mean(losses),'dev_slice_AP':evaluation['slice_AP'],'epoch_seconds':time.monotonic()-epoch_start,'elapsed_seconds':time.monotonic()-condition_start};history.append(h);csvsave(OUT/'history.csv',history);progress('train',**h)
            assert updates==576
        torch.save({'condition':condition,'epoch':12,'updates':updates,'initial_sha256':initial_hash,'identity':run_identity,'state_dict':{k:v.cpu() for k,v in a.state_dict().items()}},folder/'final.pt')
        train_evaluation,t_scores=evaluate_readout(a,cache,train,clips);save(folder/'final_train.json',train_evaluation);save(folder/'final_dev.json',evaluation);dev_scores[condition]=scores;train_scores[condition]=t_scores
        results[condition]={'status':'COMPLETE','updates':updates,'initial_sha256':initial_hash,'train':train_evaluation,'dev':evaluation,'train_update_seconds':train_seconds,'epoch_dev_eval_seconds':eval_seconds,'seconds':time.monotonic()-condition_start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};del a,opt;torch.cuda.empty_cache()
    assert [(r['video_id'],r['frame_index'],r['slabs'],r['positive_slabs'],r['slice_AP'] is not None) for r in results[CONDITIONS[0]]['dev']['frame_records']]==[(r['video_id'],r['frame_index'],r['slabs'],r['positive_slabs'],r['slice_AP'] is not None) for r in results[CONDITIONS[1]]['dev']['frame_records']]
    dev_backfill=backfill(dev_scores,cache,dev,c);train_backfill=backfill(train_scores,cache,train,c,clips)
    for condition in CONDITIONS:
        results[condition]['dev_backfill']=dev_backfill[condition];results[condition]['train_backfill']=train_backfill[condition]
    comparison=[]
    for row in dev:
        if row['role']!='fake':continue
        entries=[next(x for x in results[k]['dev']['records'] if x['video_id']==row['video_id']) for k in CONDITIONS];pixels=[next(x for x in dev_backfill[k]['records'] if x['video_id']==row['video_id']) for k in CONDITIONS]
        comparison.append({'pair_id':row['pair_id'],'video_id':row['video_id'],'evaluable_frames':entries[0]['evaluable_frames'],'positive_fraction':entries[0]['positive_fraction'],'CURRENT_ONLY_slice_AP':entries[0]['slice_AP'],'CURRENT_HISTORY_slice_AP':entries[1]['slice_AP'],'slice_delta':entries[1]['slice_AP']-entries[0]['slice_AP'] if entries[0]['slice_AP'] is not None else None,'CURRENT_ONLY_backfill_AP':pixels[0]['backfill_valid_pixel_AP'],'CURRENT_HISTORY_backfill_AP':pixels[1]['backfill_valid_pixel_AP'],'backfill_delta':pixels[1]['backfill_valid_pixel_AP']-pixels[0]['backfill_valid_pixel_AP'] if pixels[0]['backfill_valid_pixel_AP'] is not None else None,'GT_coverage':pixels[0]['GT_coverage_mean']})
    csvsave(OUT/'dev_video_comparison.csv',comparison)
    summary={'status':'COMPLETE','exit_code':0,**run_identity,'results':results,'comparison':comparison,'wall_seconds':time.monotonic()-started,'cache':read(OUT/'cache_identity.json'),'supervision':read(OUT/'supervision.json'),'model_frozen_unchanged':True,'same_initialization':True,'same_orders':True}
    save(OUT/'summary.json',summary)
    from .pixel_message_dev_check import table
    lines=['# 冻结切片级关系异常定位对照','主比较固定12轮/576步；两组仅候选可见范围不同。未微调冻结模型，不使用旧sets/GRU/帧头/像素头输出。',table(['条件','训练slice AP','开发slice AP','训练有效像素回填AP','开发有效像素回填AP'],[[k,results[k]['train']['slice_AP'],results[k]['dev']['slice_AP'],train_backfill[k]['macro_AP'],dev_backfill[k]['macro_AP']] for k in CONDITIONS]),table(list(comparison[0]),[list(r.values()) for r in comparison]),'回填是切片分数原样映射到同一几何有效像素，并非精细定位，不能与此前全画面像素头AP直接比较。','资源及身份：'+json.dumps({k:{v:results[k][v] for v in ['updates','seconds','train_update_seconds','epoch_dev_eval_seconds','peak_allocated','peak_reserved','peak_RSS']} for k in CONDITIONS},ensure_ascii=False),'总墙钟'+str(summary['wall_seconds'])+'秒；执行HEAD '+run_identity['execution_head']+'；退出0。']
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);(REPO/DOC).write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for k in CONDITIONS:
        rr=[r for r in history if r['condition']==k];ax.plot([r['updates'] for r in rr],[r['dev_slice_AP'] for r in rr],label=k)
    ax.set(xlabel='Updates',ylabel='Dev fake-video-macro slice AP');ax.legend();fig.tight_layout();fig.savefig(OUT/'dev_slice_AP.png');plt.close(fig)
    save(OUT/'status.json',{'status':'COMPLETE','stage':'report_complete','exit_code':0})


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()})
        raise
