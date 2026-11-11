"""One matched SLAB/SLAB_GRID experiment, resource gate before both trainings.

Frozen RGB encode once/frame -> shared weighted pooling -> small C-only readout.
Reuses original data/build/order/windows; never calls old sets/GRU/pixel heads.
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
from .data import Video,save
from .experiment import REPO,git,state_hash
from .set_fit_check import build
from .slice_relation_readout import windows
from .grid_localization import CONDITIONS,CurrentReadout,grouping,frozen_pool,count_loss,count_metrics
from scripts.prepare_v8_assets import digest

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
BASE=ROOT/'pvst_rgb_partition_anakin_dev_v1';REF=ROOT/'pvst_slice_relation_readout_v1';ORACLE=ROOT/'pvst_slice_localization_oracle_v1'
OUT=ROOT/'pvst_slab_grid_localization_v1';DOC=REPO/'docs/pvst/slab_grid_localization_v1.md';SEED=20260909


def read(p):return json.loads(Path(p).read_text())
def mean(x):return float(np.mean(x)) if x else None
def csvsave(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def progress(stage,**extra):
    save(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'stage':stage,**extra});print(stage,json.dumps(extra),flush=True)
def table(headers,rows):
    fmt=lambda x:f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(fmt,r))+'|' for r in rows])


@torch.no_grad()
def extract(teacher,rows,clips,config):
    folder=OUT/'cache';folder.mkdir();records=[];started=time.monotonic();max_error=0.
    oracle={(r['video_id'],int(r['frame_index'])):r for r in csv.DictReader((ORACLE/'frames.csv').open())}
    for row in rows:
        v=Video(row,config);data={};reference=torch.load(REF/'cache'/f"{row['video_id']}.pt",map_location='cpu',weights_only=True,mmap=True)
        wanted=sorted({i for s,e,_ in windows(row,clips if row['split']=='train' else None) for i in range(s,e)})
        try:
            for i in wanted:
                f=v.frame(i);xyz=f['xyz'].cuda();valid=f['valid'].cuda();gt=f['gt'].cuda();points,groups,L=grouping(xyz,valid)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    # Same RGB-only condition as original: constant depth input.
                    local,_=teacher.local.encode(f['rgb'],torch.zeros_like(f['depth']))
                    features=frozen_pool(teacher.pool,local,points,groups,valid.shape)
                base=reference[i];assert f['pts']==base['pts']==row['pts'][i]['pts_s']
                torch.testing.assert_close(features[0].cpu(),base['features'],atol=2e-5,rtol=3e-5)
                if features[0].numel():max_error=max(max_error,float((features[0].cpu()-base['features']).abs().max()))
                entry={'pts':f['pts'],'GT_pixels':int(gt.sum()),'valid_GT_pixels':int((gt&valid).sum()),'valid_pixels':int(valid.sum())}
                stat={'video_id':row['video_id'],'pair_id':row['pair_id'],'split':row['split'],'role':row['role'],'frame_index':i,'pts':f['pts'],'L':L,**{k:entry[k] for k in ['GT_pixels','valid_GT_pixels','valid_pixels']}}
                for name,C,(ids,index) in zip(CONDITIONS,features,groups):
                    counts=torch.bincount(index,minlength=len(ids));edits=torch.bincount(index[gt[valid]],minlength=len(ids))
                    assert int(counts.sum())==entry['valid_pixels'] and int(edits.sum())==entry['valid_GT_pixels'] and torch.isfinite(C).all()
                    entry[name]={'features':C.cpu(),'ids':ids.cpu(),'counts':counts.cpu(),'edit_counts':edits.cpu()};stat[name+'_units']=len(ids)
                assert torch.equal(entry[CONDITIONS[0]]['ids'],base['ids']) and torch.equal(entry[CONDITIONS[0]]['counts'],base['counts']) and torch.equal(entry[CONDITIONS[0]]['edit_counts'],base['edit_counts'])
                assert entry['GT_pixels']==base['GT_pixels'] and entry['valid_GT_pixels']==base['valid_GT_pixels']
                if row['role']=='real':assert entry['GT_pixels']==0
                if (row['video_id'],i) in oracle:
                    a=oracle[(row['video_id'],i)];assert [stat[name+'_units'] for name in CONDITIONS]==[int(a['SLAB_units']),int(a['SLAB_GRID_units'])]
                    assert entry['valid_pixels']==int(a['valid_pixels']) and entry['valid_GT_pixels']==int(a['valid_GT_pixels']) and entry['GT_pixels']==int(a['GT_pixels'])
                data[i]=entry;records.append(stat)
                del local,features,f,xyz,valid,gt,points,groups
        finally:v.close()
        torch.save(data,folder/f"{row['video_id']}.pt");del data,reference
        progress('extract',video_id=row['video_id'],frames=len(wanted),completed_frames=len(records))
    csvsave(OUT/'cache_frames.csv',records)
    result={'frames':len(records),'train_frames':sum(r['split']=='train' for r in records),'dev_frames':sum(r['split']=='dev' for r in records),'seconds':time.monotonic()-started,'SLAB_reference_max_abs_error':max_error,'SLAB_atol':2e-5,'SLAB_rtol':3e-5,'bytes':sum(p.stat().st_size for p in folder.glob('*.pt')),'files':{p.name:digest(p) for p in folder.glob('*.pt')}}
    save(OUT/'extraction.json',result);return records,result


def load_cache(rows):
    return {r['video_id']:torch.load(OUT/'cache'/f"{r['video_id']}.pt",map_location='cpu',mmap=True,weights_only=True) for r in rows}


def make(initial):
    model=CurrentReadout(chunk=512,use_checkpoint=True).cuda();model.load_state_dict(initial,strict=True);return model


def update(model,opt,cache,clip,condition,positive_weight):
    model.train();opt.zero_grad(set_to_none=True);started=time.monotonic();entries=[cache[clip['video_id']][i][condition] for i in range(clip['start'],clip['end'])]
    denominator=sum(int(f['counts'].sum()) for f in entries);assert denominator>0;loss_sum=0.;events=[];nodes=[]
    for frame in entries:
        x=frame['features'].cuda();ids=frame['ids'].cuda();n=frame['counts'].cuda().float();g=frame['edit_counts'].cuda().float();nodes.append(len(ids))
        a,b,c=[torch.cuda.Event(enable_timing=True) for _ in range(3)];a.record();z=model(x,ids);loss=count_loss(z,n,g,positive_weight,denominator);b.record()
        if not bool(torch.isfinite(loss)):raise ValueError('NONFINITE_LOSS')
        loss.backward();c.record();events.append((a,b,c));loss_sum+=float(loss.detach());del x,ids,n,g,z,loss
    gradients={name:{'present':p.grad is not None,'finite':bool(torch.isfinite(p.grad).all()) if p.grad is not None else True,'norm':float(p.grad.norm()) if p.grad is not None else None} for name,p in model.named_parameters()}
    if not all(v['finite'] for v in gradients.values()):raise ValueError('NONFINITE_GRADIENT')
    before=model.compare[-1].weight.detach().clone();a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record();opt.step();b.record();torch.cuda.synchronize()
    if torch.equal(before,model.compare[-1].weight):raise ValueError('NO_PARAMETER_CHANGE')
    return {'loss':loss_sum,'forward_seconds':sum(a.elapsed_time(b) for a,b,c in events)/1000,'backward_seconds':sum(b.elapsed_time(c) for a,b,c in events)/1000,'adamw_seconds':a.elapsed_time(b)/1000,'seconds':time.monotonic()-started,'units_per_frame':nodes,'total_attention_edges':sum(n*(n-1) for n in nodes),'denominator':denominator,'optimizer_states':len(opt.state),'parameter_changed':True,'gradients':gradients}


@torch.no_grad()
def evaluate(model,cache,rows,condition,clips=None):
    started=time.monotonic();model.eval();records=[];frame_rows=[];seen=set()
    for row in rows:
        metrics=[];coverage=[];full=covered=0
        for start,end,_ in windows(row,clips):
            for i in range(start,end):
                key=(row['video_id'],i);assert key not in seen;seen.add(key);f=cache[row['video_id']][i];q=f[condition]
                z=model(q['features'].cuda(),q['ids'].cuda()).float().cpu().numpy();m=count_metrics(z,q['counts'].numpy(),q['edit_counts'].numpy());metrics.append(m)
                if f['GT_pixels']:coverage.append(f['valid_GT_pixels']/f['GT_pixels'])
                full+=f['GT_pixels'];covered+=f['valid_GT_pixels']
                frame_rows.append({'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frame_index':i,'pts':f['pts'],'units':len(z),**m})
        result={'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frames':len(metrics),'positive_valid_GT_frames':sum(m['valid_GT_pixels']>0 for m in metrics),'no_valid_GT_frames':sum(m['valid_GT_pixels']==0 for m in metrics),'GT_coverage':mean(coverage),'GT_pixels':full,'valid_GT_pixels':covered}
        for name in ['pixel_AP','Dice','IoU','GT_recall','prediction_area']:result[name]=mean([m[name] for m in metrics if m[name] is not None and m['valid_GT_pixels']>0])
        result['real_FPR']=mean([m['FPR'] for m in metrics if m['FPR'] is not None]) if row['role']=='real' else None
        records.append(result)
    fake=[r for r in records if r['role']=='fake'];real=[r for r in records if r['role']=='real']
    macro={name:mean([r[name] for r in fake if r[name] is not None]) for name in ['pixel_AP','Dice','IoU','GT_recall','prediction_area']};macro['real_FPR']=mean([r['real_FPR'] for r in real if r['real_FPR'] is not None])
    return {'macro':macro,'records':records,'frame_records':frame_rows,'frames':len(frame_rows),'seconds':time.monotonic()-started}


def write_report(summary):
    results=summary.get('results',{});lines=['# 整片与深度片内固定方格：匹配训练','计算状态：'+summary['status']+'；本轮GT上限不是模型成绩。']
    if len(results)==2:
        for split,label in [('train','原48段训练回测'),('dev','完整开发视频')]:
            rows=[]
            for k in CONDITIONS:rows.append([k]+[results[k][split]['macro'][m] for m in ['pixel_AP','Dice','IoU','GT_recall','real_FPR']])
            rows.append(['细分−整片']+[rows[1][i]-rows[0][i] for i in range(1,6)])
            lines += [label,table(['条件','pixel AP','Dice','IoU','GT召回','real FPR'],rows)]
        lines += [table(['pair','整片AP','细分AP','ΔAP','整片Dice','细分Dice','ΔDice'],[[r[k] for k in ['pair_id','SLAB_AP','GRID_AP','AP_delta','SLAB_Dice','GRID_Dice','Dice_delta']] for r in summary['comparison']])]
        improved=sum(r['AP_delta']>0 for r in summary['comparison']);delta=results[CONDITIONS[1]]['dev']['macro']['pixel_AP']-results[CONDITIONS[0]]['dev']['macro']['pixel_AP']
        lines += [f'开发AP提高视频{improved}/4，宏平均变化{delta:+.6f}。'+('本次支持固定深度片内细分改善当前空间定位。' if improved>=2 and delta>0 else '本次没有显示多视频同向的稳定定位收益。')]
    lines += ['GT上限（另列，非训练结果）：原整片Dice 0.446189，固定方格0.942438；不得把该差值当作模型训练增量。','同一有效原生像素范围计分，官方GT覆盖与旧记录逐帧核对；positive_GT_frames=0时不补AP。固定logit≥0，real FPR独立。','身份、资源验收、耗时及峰值详见summary.json/resource_check、extraction.json；独立产物 '+str(OUT),'执行HEAD '+summary['identity']['execution_head']+'；计算与发布分开，发布见publication.json。本轮不加历史或其他配置。']
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);DOC.write_text(report)


def main():
    started=time.monotonic();torch.set_num_threads(4);assert git('branch','--show-current')=='pvst-slab-grid-localization'
    identity={'execution_head':git('rev-parse','HEAD'),'weight_sha256':digest(BASE/'RGB_DEPTH_PARTITION/last.pt'),'manifest_sha256':digest(BASE/'manifest.json'),'config_sha256':digest(BASE/'config.json'),'orders_sha256':digest(BASE/'orders.json'),'readout_reference_sha256':digest(REF/'summary.json'),'oracle_reference_sha256':digest(ORACLE/'summary.json'),'conditions':list(CONDITIONS),'seed':SEED,'epochs':12,'updates':576,'query_chunk':512,'pool_pixel_chunk':8192,'readout_precision':'FP32 CUDA efficient SDPA only; checkpoint per query block','frozen_precision':'original BF16','optimizer':{'AdamW_lr':1e-4,'weight_decay':.01},'loss':'pixel-count weighted BCE / clip valid pixels; common train neg/pos','source_code':{p:digest(REPO/p) for p in ['research_tools/rgbd_slice/grid_localization.py','research_tools/rgbd_slice/slab_grid_experiment.py']}}
    if OUT.exists():
        existing=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if existing.get('status')=='COMPLETE' and all(existing['identity'][k]==identity[k] for k in identity if k!='execution_head'):print('REUSE COMPLETE');return
        raise ValueError('EXISTING_PARTIAL_OUTPUT_NO_RETRY')
    OUT.mkdir();save(OUT/'identity.json',identity);progress('verify_assets')
    m=read(BASE/'manifest.json');config=read(BASE/'config.json');orders=read(BASE/'orders.json');baseline=read(BASE/'identity.json');rows=m['rows'];train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];clips=m['clips']
    assert len(train)==16 and len(dev)==8 and len(clips)==48 and sum(r['actual_frames'] for r in dev)==924
    assert len({(x['video_id'],i) for x in clips for i in range(x['start'],x['end'])})==1152
    assert {r['source_id'] for r in train}.isdisjoint({r['source_id'] for r in dev})
    for file,key in [('manifest.json','frozen_manifest_sha256'),('config.json','config_sha256'),('orders.json','orders_sha256')]:assert digest(BASE/file)==baseline[key]
    for path,sha in baseline['code'].items():assert digest(REPO/path)==sha
    assert identity['weight_sha256']==read(REF/'identity.json')['weight_sha256'] and identity['manifest_sha256']==read(REF/'identity.json')['manifest_sha256']
    for row in rows:
        assert digest(Path(row['video_path']))==row['video_sha256']
        if row['role']=='fake':assert digest(Path(row['mask_zip']))==row['mask_sha256']
        for path,sha in next(a for a in baseline['assets'] if a['video_id']==row['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha
    canonical=lambda a:sorted((x['video_id'],x['start'],x['end']) for x in a)
    for epoch in range(1,13):assert canonical(orders[str(epoch)])==canonical(clips)
    for file in ['manifest.json','orders.json','config.json']:shutil.copyfile(BASE/file,OUT/file)
    save(OUT/'tests.json',{'passed':5,'FP32_atol':2e-5,'FP32_rtol':3e-5,'CUDA_output_max_abs':0.,'CUDA_input_gradient_max_abs':3.725290298461914e-9,'CUDA_parameter_gradient_max_abs':2.9802322387695312e-8,'scope':'CPU+CUDA old CURRENT_ONLY equivalence, self masking/empty, full-unit softmax/routing, pixel count BCE/gradient and exact tied AP'})
    teacher=build(config,OUT,rgb_only=True).eval();cp=torch.load(BASE/'RGB_DEPTH_PARTITION/last.pt',map_location='cpu',weights_only=True)
    assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==baseline['identity']
    assert read(OUT/'initialization.json')['model_sha256']==cp['initial_model_sha256'];teacher.load_state_dict(cp['state_dict'],strict=True);del cp
    teacher.requires_grad_(False);before=state_hash(teacher.state_dict());torch.cuda.reset_peak_memory_stats();stats,extraction=extract(teacher,train+dev,clips,config)
    assert state_hash(teacher.state_dict())==before and all(p.grad is None for p in teacher.parameters());extraction.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),teacher_state_unchanged=True);save(OUT/'extraction.json',extraction)
    del teacher;torch.cuda.empty_cache();cache=load_cache(rows)
    trainstats=[r for r in stats if r['split']=='train'];P=sum(r['valid_GT_pixels'] for r in trainstats);N=sum(r['valid_pixels'] for r in trainstats);assert 0<P<N
    pw=torch.tensor((N-P)/P,device='cuda');save(OUT/'supervision.json',{'train_valid_pixels':N,'train_valid_GT_pixels':P,'w_pos':float(pw),'same_between_conditions':True})
    torch.manual_seed(SEED);model=CurrentReadout();initial={k:v.clone() for k,v in model.state_dict().items()};initial_hash=state_hash(initial);params=sum(p.numel() for p in model.parameters());del model
    save(OUT/'readout_initialization.json',{'sha256':initial_hash,'parameters':params,'newly_seeded_not_old_readout_checkpoint':True})
    maximum=max(trainstats,key=lambda x:x['SLAB_GRID_CURRENT_units']);clip=next(c for c in clips if c['video_id']==maximum['video_id'] and c['start']<=maximum['frame_index']<c['end'])
    devmax=max((x for x in stats if x['split']=='dev'),key=lambda x:x['SLAB_GRID_CURRENT_units']);free,total=torch.cuda.mem_get_info();progress('resource_check',clip=clip,train_max_frame=maximum,dev_max_frame=devmax)
    a=make(initial);opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01);torch.cuda.reset_peak_memory_stats();check=update(a,opt,cache,clip,CONDITIONS[1],pw)
    check.update(clip=clip,train_max_frame=maximum,dev_max_frame=devmax,starting_free=free,GPU_name=torch.cuda.get_device_name(),GPU_total=total,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,passed=True,discarded_update=True)
    if devmax['SLAB_GRID_CURRENT_units']>maximum['SLAB_GRID_CURRENT_units']:
        a.eval();q=cache[devmax['video_id']][devmax['frame_index']][CONDITIONS[1]];t=time.monotonic()
        with torch.no_grad():z=a(q['features'].cuda(),q['ids'].cuda());assert torch.isfinite(z).all()
        torch.cuda.synchronize();check['larger_dev_forward_seconds']=time.monotonic()-t;del z
    check['peak_allocated']=torch.cuda.max_memory_allocated();check['peak_reserved']=torch.cuda.max_memory_reserved();check['conservative_576_max_clip_seconds']=check['seconds']*576;save(OUT/'resource_check.json',check);del a,opt;torch.cuda.empty_cache()
    # A predeclared operational gate, not a change in model/data/training budget.
    if check['conservative_576_max_clip_seconds']>4*3600:
        summary={'status':'RESOURCE_BLOCKED','identity':identity,'resource_check':check,'results':{},'reason':'even 576 maximum-clip updates estimated above four hours; no formal model trained','exit_code':0};save(OUT/'summary.json',summary);write_report(summary);save(OUT/'status.json',{'status':'RESOURCE_BLOCKED','stage':'report_complete','exit_code':0});return
    history=[];results={}
    for condition in CONDITIONS:
        a=make(initial);assert state_hash(a.state_dict())==initial_hash;opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01);updates=0;start=time.monotonic();train_s=eval_s=0.;torch.cuda.reset_peak_memory_stats();folder=OUT/condition;folder.mkdir()
        with (folder/'updates.jsonl').open('w') as log:
            for epoch in range(1,13):
                losses=[];epoch_started=time.monotonic()
                for clip in orders[str(epoch)]:
                    record=update(a,opt,cache,clip,condition,pw);updates+=1;train_s+=record['seconds'];losses.append(record['loss']);record.update(epoch=epoch,update=updates,clip=clip);log.write(json.dumps(record)+'\n');log.flush()
                evaluated=evaluate(a,cache,dev,condition);assert evaluated['frames']==924;eval_s+=evaluated['seconds'];save(folder/f'epoch{epoch:02d}_dev.json',evaluated)
                h={'condition':condition,'epoch':epoch,'updates':updates,'train_loss':mean(losses),'dev_pixel_AP':evaluated['macro']['pixel_AP'],'dev_Dice':evaluated['macro']['Dice'],'epoch_seconds':time.monotonic()-epoch_started,'elapsed_seconds':time.monotonic()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()};history.append(h);csvsave(OUT/'history.csv',history);progress('train',**h)
        assert updates==576
        torch.save({'condition':condition,'updates':updates,'epoch':12,'initial_sha256':initial_hash,'identity':identity,'state_dict':{k:v.cpu() for k,v in a.state_dict().items()}},folder/'final.pt')
        training=evaluate(a,cache,train,condition,clips);assert training['frames']==1152;save(folder/'final_train.json',training);save(folder/'final_dev.json',evaluated)
        for split,data in [('train',training),('dev',evaluated)]:csvsave(folder/f'{split}_videos.csv',data['records'])
        results[condition]={'status':'COMPLETE','updates':updates,'initial_sha256':initial_hash,'train':training,'dev':evaluated,'seconds':time.monotonic()-start,'train_update_seconds':train_s,'epoch_dev_eval_seconds':eval_s,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};del a,opt;torch.cuda.empty_cache()
    for split in ['train','dev']:
        names=['video_id','frame_index','pts','valid_pixels','valid_GT_pixels'];assert [[r[k] for k in names] for r in results[CONDITIONS[0]][split]['frame_records']]==[[r[k] for k in names] for r in results[CONDITIONS[1]][split]['frame_records']]
    comparison=[]
    for x,y in zip(results[CONDITIONS[0]]['dev']['records'],results[CONDITIONS[1]]['dev']['records']):
        assert x['video_id']==y['video_id']
        if x['role']=='fake':comparison.append({'pair_id':x['pair_id'],'video_id':x['video_id'],'SLAB_AP':x['pixel_AP'],'GRID_AP':y['pixel_AP'],'AP_delta':y['pixel_AP']-x['pixel_AP'],'SLAB_Dice':x['Dice'],'GRID_Dice':y['Dice'],'Dice_delta':y['Dice']-x['Dice'],'GT_coverage':x['GT_coverage']})
    csvsave(OUT/'dev_video_comparison.csv',comparison)
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'results':results,'comparison':comparison,'resource_check':check,'extraction':extraction,'wall_seconds':time.monotonic()-started,'GT_coverage_same':True,'frozen_parameters_and_buffers_unchanged':True}
    save(OUT/'summary.json',summary);write_report(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for k in CONDITIONS:
        rr=[r for r in history if r['condition']==k];ax.plot([r['updates'] for r in rr],[r['dev_pixel_AP'] for r in rr],label=k)
    ax.set(xlabel='Updates',ylabel='Dev valid-pixel AP');ax.legend();fig.tight_layout();fig.savefig(OUT/'dev_pixel_AP.png');plt.close(fig)
    save(OUT/'status.json',{'status':'COMPLETE','stage':'report_complete','exit_code':0})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():
            status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'FAILED';previous=read(OUT/'status.json') if (OUT/'status.json').exists() else {};summary={'status':status,'exit_code':1,'identity':read(OUT/'identity.json'),'results':{},'last_stage':previous,'error':traceback.format_exc()};save(OUT/'failure.json',summary);save(OUT/'summary.json',summary);write_report(summary);save(OUT/'status.json',{'status':status,'exit_code':1,'last_stage':previous,'error':traceback.format_exc()})
        raise
