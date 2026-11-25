"""Fixed epoch03 context diagnosis. User launches full queue; --smoke is one video.

Only new caller is run_pvst_fullvideo_context_diagnosis_v1.sh. Outputs are
streamed scalar statistics/metrics, never observations or trainable C/R caches.
The decoder adapter captures the exact original correction input; it does not
reimplement projection, interpolation or backfill. No optimizer/frontend exists.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from . import fullvideo_matched as original
from . import fullvideo_matched_data as data
from .fullvideo_direct_model import FullVideoDirect, MODES
from .fullvideo_direct_probe import digest, save, state_hash

ROOT = data.OUT.parent/'pvst_fullvideo_context_diagnosis_v1'
REGIONS = ('fake_GT', 'fake_background', 'real_all', 'fake_empty')
DIFFS = ('total', 'context', 'zero_remaining')
STATS = ('mean', 'mean_abs', 'frame_abs_p95', 'max_abs')


def emit(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize()


class DecodeAdapter:
    """Passive hook on original decode; zero the FULL already-backfilled context.

    Captured channels are detail32 followed by context32. The original decode
    remains the NORMAL implementation. Optional correction replay independently
    verifies this boundary without encoding or relating again.
    """
    def __init__(self, model, calls): self.model, self.calls = model, calls

    def pair(self, frame, relation, parity=False, repeat=False):
        captured=[]
        def capture(_module, args): captured.append(args[0])
        hook=self.model.correction.register_forward_pre_hook(capture)
        try:
            self.calls['normal_decode']+=1
            normal=self.model.decode(frame, relation)
        finally:
            hook.remove()
        assert len(captured)==1 and captured[0].shape[1]==64
        features=captured[0];detail,context=features.split(32,dim=1)
        alternative=torch.cat((detail,torch.zeros_like(context)),dim=1)
        assert torch.count_nonzero(alternative[:,32:])==0
        assert torch.equal(alternative[:,:32],detail)
        base=frame['z0'].to(normal.device).float()
        self.calls['zero_correction']+=1
        zero=base+self.model.correction(alternative)[0,0].float()
        check={}
        if parity:
            self.calls['adapter_parity_correction']+=1
            replay=base+self.model.correction(features)[0,0].float()
            check['adapter_normal_max_abs']=float((replay-normal).abs().max())
            torch.testing.assert_close(replay,normal,rtol=0,atol=0)
        if repeat:
            self.calls['repeat_normal_decode']+=1
            again=self.model.decode(frame,relation)
            check['repeat_normal_max_abs']=float((again-normal).abs().max())
        assert normal.shape==zero.shape==tuple(frame['work_hw'])
        assert torch.isfinite(normal).all() and torch.isfinite(zero).all()
        return normal,zero,detail,context,check


@torch.no_grad()
def encode_relations(model,frames,mode,calls,progress):
    """Exact FullVideoDirect.forward encode/position/relate; decode streamed later."""
    assert frames and all(b['pts']>a['pts'] for a,b in zip(frames,frames[1:]))
    cs=[];ps=[];counts=[len(f['keys']) for f in frames];device=next(model.parameters()).device
    for i,f in enumerate(frames):
        calls['frame_encode']+=1;cs.append(model.encode(f))
        centers=(f['keys'].to(device).float()+.5)*.5
        dt=centers.new_full((len(centers),1),f['pts']-frames[0]['pts'])
        ps.append(model.position(centers)+model.time(dt))
        if i%20==0 or i+1==len(frames):progress('encode',frame=i+1,T=len(frames))
    tokens=torch.cat(cs);calls['video_relation']+=1
    progress('relation_begin',T=len(frames),M=sum(counts))
    relations=model.relate(tokens,torch.cat(ps),counts,mode)
    assert relations.shape==tokens.shape
    return tokens,relations,counts


def native_record(row,i,gt,output):
    """Original score_video formula, shared native GT for both readouts."""
    native=F.interpolate(output.float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    assert np.isfinite(native).all();pred=native>=0;overlap=original.spatial_frame(pred,gt)
    positive=int(gt.sum());neg=gt.size-positive;fp=int((pred&~gt).sum())
    return dict(video_id=row['video_id'],pair_id=row['pair_id'],source_group_id=row['source_group_id'],role=row['role'],
        frame_index=i,pts=row['pts'][i],native_pixels=gt.size,GT_pixels=positive,
        pixel_AP=float(original.average_precision_score(gt.ravel(),native.ravel())) if positive else None,
        constant_AP=float(gt.mean()) if positive else None,Dice=overlap['dice'],IoU=overlap['iou'],
        GT_recall=float((pred&gt).sum()/positive) if positive else None,
        real_FPR=overlap['empty_fpr'] if row['role']=='real' else None,
        fake_background_FPR=fp/neg if row['role']=='fake' and positive and neg else None,
        fake_empty_FPR=overlap['empty_fpr'] if row['role']=='fake' and not positive else None,
        prediction_area=float(pred.mean()) if row['role']=='fake' else None)


def region_statistics(row,i,gt_work,normal,zero,base):
    arrays=[x.detach().float().cpu().numpy() for x in (normal,zero,base)]
    normal,zero,base=arrays;gt=np.asarray(gt_work,dtype=bool)
    assert gt.shape==normal.shape==zero.shape==base.shape
    assert all(np.isfinite(x).all() for x in arrays)
    empty=np.zeros_like(gt)
    masks=dict(fake_GT=gt if row['role']=='fake' else empty,
        fake_background=~gt if row['role']=='fake' and gt.any() else empty,
        real_all=np.ones_like(gt) if row['role']=='real' else empty,
        fake_empty=np.ones_like(gt) if row['role']=='fake' and not gt.any() else empty)
    results=[]
    for region,mask in masks.items():
        n=int(mask.sum())
        for name,left,right in [('total',normal,base),('context',normal,zero),('zero_remaining',zero,base)]:
            delta=(left-right)[mask];abs_delta=np.abs(delta)
            values=dict(mean=float(delta.mean()) if n else None,mean_abs=float(abs_delta.mean()) if n else None,
                frame_abs_p95=float(np.quantile(abs_delta,.95)) if n else None,max_abs=float(abs_delta.max()) if n else None)
            # Signed changes relative to z0 or CTX_ZERO, at fixed logit>=0.
            pn=int(((right>=0)&(left<0)&mask).sum()) if n and name!='zero_remaining' else None
            np_=int(((right<0)&(left>=0)&mask).sum()) if n and name!='zero_remaining' else None
            results.append(dict(video_id=row['video_id'],pair_id=row['pair_id'],source_group_id=row['source_group_id'],
                role=row['role'],frame_index=i,pts=row['pts'][i],region=region,difference=name,pixels=n,**values,
                positive_to_negative=pn,negative_to_positive=np_,
                positive_to_negative_ratio=pn/n if pn is not None else None,
                negative_to_positive_ratio=np_/n if np_ is not None else None))
    return results


def input_statistics(frame,detail,context):
    points=frame['points'].to(context.device);h,w=frame['work_hw']
    values=dict(valid_pixels=len(points),frame_pixels=h*w,coverage=len(points)/(h*w))
    for name,x in [('context',context),('detail',detail)]:
        v=x[0].permute(1,2,0).reshape(-1,32)[points].float()
        assert torch.isfinite(v).all()
        values[name+'_mean_abs']=float(v.abs().mean()) if v.numel() else None
        values[name+'_RMS']=float(v.square().mean().sqrt()) if v.numel() else None
    return values


def summarize_regions(records,level):
    groupkeys=['mode','region','difference']
    if level=='video':groupkeys+=['video_id','pair_id','source_group_id']
    elif level=='source':groupkeys+=['pair_id','source_group_id']
    elif level!='macro':raise ValueError(level)
    bins={}
    for r in records:bins.setdefault(tuple(r[k] for k in groupkeys),[]).append(r)
    output=[]
    for key,rs in bins.items():
        valid=[r for r in rs if r['pixels']>0];n=sum(r['pixels'] for r in rs)
        summary=dict(zip(groupkeys,key));summary.update(pixels=n,
            valid_frames=len(valid) if level=='video' else sum(r['valid_frames'] for r in rs),
            **{k:original.mean_metric(valid,k) for k in STATS})
        # max is maximum, not an averaged per-frame maximum; p95 is a mean of frame p95s.
        summary['max_abs']=max((r['max_abs'] for r in valid),default=None)
        for k in ['positive_to_negative','negative_to_positive']:
            counts=[r[k] for r in rs if r.get(k) is not None]
            summary[k]=sum(counts) if counts else None
            summary[k+'_ratio']=original.mean_metric(valid,k+'_ratio')
            summary[k+'_pixel_ratio']=summary[k]/n if summary[k] is not None and n else None
        output.append(summary)
    return output


def evaluation_difference(new,old):
    expected={(f['video_id'],f['frame_index']):f for f in old['frames']}
    actual={(f['video_id'],f['frame_index']):f for f in new['frames']}
    assert len(actual)==len(new['frames'])==len(expected) and set(actual)==set(expected)
    for key,f in actual.items():
        assert all(f[k]==expected[key][k] for k in ['pts','native_pixels','GT_pixels'])
    return {k:new['macro'][k]-old['macro'][k] if new['macro'][k] is not None and old['macro'][k] is not None else None for k in original.METRICS}


def source_identity(source):
    names=['protocol.json','config.json','compute.json','identity.json','initialization.json','monitor.json','categories.json','prepared.json']
    recorded=data.read(source/'identity.json')
    # Compare actual computation code, not current HEAD or the new thin entry.
    here=Path(__file__).parent
    for name,sha in recorded['code_sha256'].items():
        if (here/name).is_file():assert digest(here/name)==sha,('COMPUTATION_CODE_CHANGED',name)
    assert digest(source/'protocol.json')==recorded['protocol_sha256'],'PROTOCOL_FILE_IDENTITY'
    # Original categories_sha256 names its SOURCE CSV, not generated categories.json.
    assert digest(data.SOURCE/'analysis/source_comparison.csv')==recorded['categories_sha256'],'CATEGORIES_SOURCE_IDENTITY'
    assert data.read(source/'categories.json')==data.categories(),'CATEGORIES_CONTENT_IDENTITY'
    assert data.read(source/'config.json')==recorded['config'],'CONFIG_FILE_IDENTITY'
    assert data.read(source/'prepared.json')['identity']==recorded
    assert data.read(source/'compute.json')['identity']==recorded
    info=dict(source=str(source.resolve()),files={n:digest(source/n) for n in names},
        diagnosis_code_sha256=digest(Path(__file__)),models={})
    for mode in MODES:
        complete=data.read(source/mode/'complete.json');evaluation=data.read(source/mode/'epoch03_dev.json')
        assert complete['status']=='COMPLETE' and complete['updates']==534 and complete['final_epoch']==3
        info['models'][mode]=dict(checkpoint=str(source/mode/'epoch03.pt'),checkpoint_file_sha256=digest(source/mode/'epoch03.pt'),
            model_sha256=evaluation['model_sha256'],complete_sha256=digest(source/mode/'complete.json'),
            evaluation_sha256=digest(source/mode/'epoch03_dev.json'))
    return info


def load_model(source,mode,identity,device,calls):
    path=Path(identity['models'][mode]['checkpoint']);calls['checkpoint_load']+=1
    cp=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(protocol=data.read(source/'identity.json'),query_chunk=data.read(source/'compute.json')['query_chunk'],
        initial_sha256=data.read(source/'initialization.json')['state_sha256'])
    assert cp['identity']==expected and cp['mode']==mode and cp['epoch']==3
    assert cp['updates']==len(cp['records'])==534 and cp['position']==178
    assert state_hash(cp['state_dict'])==identity['models'][mode]['model_sha256']
    model=FullVideoDirect(expected['query_chunk']).to(device)
    model.load_state_dict(cp['state_dict'],strict=True);model.eval();model.requires_grad_(False)
    assert state_hash(model.state_dict())==identity['models'][mode]['model_sha256']
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    del cp  # optimizer and training RNG are not restored or used
    return model


@torch.inference_mode()
def diagnose_video(model,source,row,mode,calls,smoke=False):
    device=next(model.parameters()).device;start=time.monotonic()
    def progress(stage,**kw):emit(stage,mode=mode,video=row['video_id'],**kw)
    progress('load_video',T=row['actual_frames']);frames=data.VideoCache(source).load(row)
    assert len(frames)==row['actual_frames'];sync(device);loaded=time.monotonic()
    with amp(device):tokens,relations,counts=encode_relations(model,frames,mode,calls,progress)
    sync(device);related=time.monotonic();before=state_hash(dict(C=tokens,R=relations))
    versions=(tokens._version,relations._version) if not torch.is_inference(tokens) else None
    progress('relation_complete',T=len(frames),M=sum(counts),seconds=related-loaded)
    adapter=DecodeAdapter(model,calls);records={'NORMAL':[],'CTX_ZERO':[]};statistics=[];amplitudes=[];checks=[]
    decode_seconds=0.;stat_seconds=0.;score_seconds=0.
    repeats=set([0,len(frames)//2,len(frames)-1] if smoke else row['preview_targets'])
    z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
    members=original.numeric_members(z.namelist()) if z else None
    if z:assert len(members)==len(frames) and [m[0] for m in members]==list(range(row['index_base'],row['index_base']+len(frames)))
    try:
        for i,(frame,r) in enumerate(zip(frames,relations.split(counts))):
            tick=time.monotonic()
            with amp(device):normal,zero,detail,context,check=adapter.pair(frame,r,parity=i==0,repeat=i in repeats)
            sync(device);decode_seconds+=time.monotonic()-tick;tick=time.monotonic()
            checks.append(dict(frame_index=i,**check))
            statistics.extend(region_statistics(row,i,frame['gt_work'],normal,zero,frame['z0'].to(device)))
            amplitudes.append(dict(video_id=row['video_id'],pair_id=row['pair_id'],source_group_id=row['source_group_id'],frame_index=i,pts=row['pts'][i],**input_statistics(frame,detail,context)))
            stat_seconds+=time.monotonic()-tick;tick=time.monotonic()
            if data.supervised(row,i):
                gt=original.native_mask(z,members,row,i);assert int(gt.sum())==frame['native_GT_pixels']
                for label,output in [('NORMAL',normal),('CTX_ZERO',zero)]:records[label].append(native_record(row,i,gt,output))
            score_seconds+=time.monotonic()-tick
            if i%20==0 or i+1==len(frames):
                progress('decode_complete',frame=i+1,T=len(frames),cumulative_seconds=decode_seconds)
                progress('score_complete',frame=i+1,T=len(frames),cumulative_seconds=score_seconds,statistics_seconds=stat_seconds)
    finally:
        if z:z.close()
    assert before==state_hash(dict(C=tokens,R=relations)),'C_OR_R_MUTATED'
    if versions is not None:assert versions==(tokens._version,relations._version)
    assert all(p.grad is None for p in model.parameters())
    sync(device)
    return dict(status='COMPLETE',video_id=row['video_id'],T=len(frames),M=sum(counts),counts=counts,
        metrics={k:original.aggregate_records(v) for k,v in records.items()},region_frames=statistics,input_frames=amplitudes,
        checks=checks,C_R_unchanged=True,context_zero_verified=True,load_seconds=loaded-start,
        encode_relation_seconds=related-loaded,decode_seconds=decode_seconds,statistics_seconds=stat_seconds,native_score_seconds=score_seconds,decode_score_seconds=time.monotonic()-related,seconds=time.monotonic()-start)


def monitor_cpu(source,protocol,out):
    """CPU z0 restoration/GT scoring only: no neural modules or geometry loaded."""
    names=data.read(source/'monitor.json');assert len(names)==4 and set(names)=={'4dpnv4_fake','4dpnv4_real','4qw154_fake','4qw154_real'}
    byid={r['video_id']:r for r in protocol['rows']};records=[];start=time.monotonic()
    for name in names:
        row=byid[name];paths=data.read(source/'visual'/name/'frames.json');assert len(paths)==row['actual_frames']
        work=data.WorkVideo(row);z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        members=original.numeric_members(z.namelist()) if z else None
        try:
            for i,path in enumerate(paths):
                if not data.supervised(row,i):continue
                packet=torch.load(path,map_location='cpu',weights_only=True)
                assert packet['frame_index']==i and packet['pts']==row['pts'][i] and packet['work_hw']==row['work_hw']
                gt=original.native_mask(z,members,row,i);assert int(gt.sum())==work.get(i)['native_GT_pixels']
                records.append(native_record(row,i,gt,packet['z0']))
                if i%20==0 or i+1==len(paths):emit('monitor_z0_CPU',video=name,frame=i+1,T=len(paths))
        finally:
            if z:z.close()
    ev=original.aggregate_records(records);ev['seconds']=time.monotonic()-start
    ev['execution']='CPU FP32 bilinear restoration; no network forward; same formula as original GPU restoration, possible floating point differences'
    comparisons=[dict(condition='FROZEN_z0_CPU',epoch=0,**ev['macro'])]
    for mode in MODES:
        for epoch in [1,2,3]:
            old=data.read(source/mode/f'epoch{epoch:02d}_monitor.json')
            evaluation_difference(ev,old)  # identities/counts only, numerical values may differ
            comparisons.append(dict(condition=mode,epoch=epoch,**old['macro']))
    original.persist_eval(out/'monitor_z0_CPU.json',ev);original.csvsave(out/'monitor_comparison.csv',comparisons)
    return ev,comparisons


def report(out,summary):
    f=lambda x:'null' if x is None else f'{x:.8f}'
    lines=['# PVST固定权重context诊断（程序报告）','',f"状态：{summary['status']}；scope={summary['scope']}。",
        'z_normal=z0+correction(detail,context)，z_zero=z0+correction(detail,0)。投影bias与回填后全部context一起移除；C/R及detail/z0不变。这不是纯时间或纯几何消融。',
        '全局新增训练/MoGe/DINO提取均0。完整开发诊断仅在scope=development、状态COMPLETE时完成；smoke不能代替完整诊断。',
        '', '## 定位及回放','', '|模式/范围|读出|AP|Dice|IoU|召回|real FPR|fake背景FPR|面积|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary.get('primary',[]):lines.append('|'+row['mode']+'/'+row['group']+'|'+row['readout']+'|'+'|'.join(f(row[k]) for k in original.METRICS[:-1])+'|')
    lines+=['','### 全部来源（不筛选）','','|模式/读出|来源|AP|Dice|召回|real FPR|','|---|---|---:|---:|---:|---:|']
    for r in summary.get('source_rows',[]):lines.append('|'+r['mode']+'/'+r['readout']+'|'+r['pair_id']+'|'+'|'.join(f(r[k]) for k in ['pixel_AP','Dice','GT_recall','real_FPR'])+'|')
    lines+=['','NORMAL−CTX_ZERO、NORMAL−原epoch03逐来源和分组差值见comparison.csv/replay_sources.csv。回放差值不自动判某层失效；若明显不复现，应先检查权重/模式/输入/恢复规则，停止模型归因，不重跑寻找更近结果。',
        '', '## 三种修正与输入幅度','',
        'region_frames/videos/sources/macro.csv分区报告总修正、context敏感性、零context剩余修正；空区域null，有效帧数和像素数单列。mean/mean_abs及frame_abs_p95按帧→视频→来源等权，max_abs取最大值；不是全库像素合池p95。翻转计数求和，ratio按同层级均值，pixel_ratio为相应区域计数比。',
        'context/detail只在同一几何有效像素支持上给出幅度和覆盖率；幅度不等于信息含量或定位贡献。三种修正不是相互独立的因果分解；总修正小而context敏感性大可以来自路径补偿。',
        '', '|模式|区域|差值|有效帧/像素|mean|mean-abs|帧p95均值|max-abs|','|---|---|---|---|---:|---:|---:|---:|']
    for r in summary.get('region_macro',[]):lines.append('|'+r['mode']+'|'+r['region']+'|'+r['difference']+'|'+f"{r['valid_frames']}/{r['pixels']}"+'|'+'|'.join(f(r[k]) for k in STATS)+'|')
    lines+=['','## 同清单训练观察','', '|条件/轮|AP|Dice|召回|','|---|---:|---:|---:|']
    for r in summary.get('monitor_comparison',[]):lines.append('|'+r['condition']+'/'+str(r['epoch'])+'|'+'|'.join(f(r[k]) for k in ['pixel_AP','Dice','GT_recall'])+'|')
    lines+=['','仅使用monitor.json的四完整训练视频，来自4dpnv4和4qw154；非完整训练集，不是块特征单独能力。CPU z0原生恢复可能与旧GPU有微小差异，不与旧356帧观察比较。',
        '', '## 自动数据摘要与解释边界','']
    for v in summary.get('input_macro',[]):lines.append('context/detail幅度参照：'+json.dumps(v,ensure_ascii=False))
    if summary.get('monitor_z0_macro'):
        lines.append('相同观察清单z0指标：'+json.dumps(summary['monitor_z0_macro'],ensure_ascii=False))
        lines.append('各轮相对z0增量：'+json.dumps(summary['monitor_minus_z0'],ensure_ascii=False))
    for mode,v in summary.get('interpretation',{}).items():lines.append(f"- {mode}：{json.dumps(v,ensure_ascii=False)}")
    lines+=['', '上述幅度/翻转、重复误差、AP/Dice/FPR共同判断，不硬编码幅度阈值宣布故障。置零改变输入分布，性能下降不能证明三维/时间有效；无变化也不能证明C无可读信息。下一步仅建议：先依据本报告的回放、幅度与定位变化确认读出对context的实际依赖，再决定是否需要一个保留当前视觉定位的关系修正对照；不在本轮执行。',
        '',f"实际本次调用计数：{summary.get('invocation_calls',{})}；复用与新计算分列于summary.json。",
        f"耗时秒：{summary.get('seconds')}；退出码：{summary.get('exit_code')}。固定权重及C/R不变核验、每帧repeat参照均在逐视频result中。",'']
    (out/'report.md').write_text('\n'.join(lines))


def finalize(out,results,source,scope,calls,seconds,monitor=None):
    primary=[];source_rows=[];comparisons=[];replays=[];allregions=[];inputs=[];checks=[];interpretation={};evs={};input_macro=[]
    for mode in MODES:
        selected=[r for r in results if r['mode']==mode]
        assert selected
        evs[mode]={}
        for label in ['NORMAL','CTX_ZERO']:
            ev=original.aggregate_records([f for r in selected for f in r['metrics'][label]['frames']])
            if scope=='development':
                assert ev['scored_frames']==5152;ev['groups']=original.groups(ev,data.read(source/'categories.json'))
            else:ev['groups']={'完整小病例（不是开发集）':dict(sources=len(ev['sources']),frames=ev['scored_frames'],**ev['macro'])}
            original.persist_eval(out/mode/(label+'.json'),ev);evs[mode][label]=ev
            source_rows.extend(dict(mode=mode,readout=label,**s) for s in ev['sources'])
            for name,g in ev['groups'].items():primary.append(dict(mode=mode,readout=label,group=name,**g))
        normal,zero=evs[mode]['NORMAL'],evs[mode]['CTX_ZERO']
        for group in normal['groups']:
            comparisons.append(dict(mode=mode,group=group,comparison='NORMAL_minus_CTX_ZERO',**{k:normal['groups'][group][k]-zero['groups'][group][k] if normal['groups'][group][k] is not None and zero['groups'][group][k] is not None else None for k in original.METRICS}))
        for a,b in zip(normal['sources'],zero['sources']):
            assert a['source_group_id']==b['source_group_id']
            comparisons.append(dict(mode=mode,source_group_id=a['source_group_id'],pair_id=a['pair_id'],comparison='NORMAL_minus_CTX_ZERO',**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in original.METRICS}))
        if scope=='development':
            old=data.read(source/mode/'epoch03_dev.json');diff=evaluation_difference(normal,old)
            comparisons.append(dict(mode=mode,group='全部19开发来源',comparison='NORMAL_minus_original_epoch03',**diff))
            for group in normal['groups']:
                if group!='全部19开发来源':comparisons.append(dict(mode=mode,group=group,comparison='NORMAL_minus_original_epoch03',**{k:normal['groups'][group][k]-old['groups'][group][k] if normal['groups'][group][k] is not None and old['groups'][group][k] is not None else None for k in original.METRICS}))
            oldby={s['pair_id']:s for s in old['sources']}
            for s in normal['sources']:
                o=oldby[s['pair_id']];replays.append(dict(mode=mode,pair_id=s['pair_id'],**{k:s[k]-o[k] if s[k] is not None and o[k] is not None else None for k in original.METRICS}))
        for r in selected:
            allregions.extend(dict(mode=mode,**x) for x in r['region_frames'])
            inputs.extend(dict(mode=mode,**x) for x in r['input_frames'])
            checks.extend(dict(mode=mode,video_id=r['video_id'],**x) for x in r['checks'])
        interpretation[mode]=dict(source_directions={k:dict(up=sum(a[k] is not None and b[k] is not None and a[k]>b[k] for a,b in zip(normal['sources'],zero['sources'])),down=sum(a[k] is not None and b[k] is not None and a[k]<b[k] for a,b in zip(normal['sources'],zero['sources']))) for k in ['pixel_AP','Dice','GT_recall']},NORMAL_minus_CTX_ZERO_macro={k:normal['macro'][k]-zero['macro'][k] if normal['macro'][k] is not None and zero['macro'][k] is not None else None for k in original.METRICS},
            adapter_max_abs=max(x.get('adapter_normal_max_abs',0) for r in selected for x in r['checks']),
            repeat_decode_max_abs=max(x.get('repeat_normal_max_abs',0) for r in selected for x in r['checks']),
            NORMAL_AP=normal['macro']['pixel_AP'],CTX_ZERO_AP=zero['macro']['pixel_AP'])
    videos=summarize_regions(allregions,'video');sources=summarize_regions(videos,'source');macro=summarize_regions(sources,'macro')
    for name,rows in [('primary',primary),('source_metrics',source_rows),('comparison',comparisons),('replay_sources',replays),('region_frames',allregions),('region_videos',videos),('region_sources',sources),('region_macro',macro),('input_frames',inputs),('decode_checks',checks)]:original.csvsave(out/(name+'.csv'),rows)
    # Same pixel support for context/detail; frame -> video -> source summaries.
    inputvideos=[]
    for mode in MODES:
        for vid in sorted({r['video_id'] for r in inputs if r['mode']==mode}):
            rs=[r for r in inputs if r['mode']==mode and r['video_id']==vid]
            inputvideos.append(dict(mode=mode,video_id=vid,pair_id=rs[0]['pair_id'],source_group_id=rs[0]['source_group_id'],frames=len(rs),valid_pixels=sum(r['valid_pixels'] for r in rs),**{k:original.mean_metric(rs,k) for k in ['coverage','context_mean_abs','context_RMS','detail_mean_abs','detail_RMS']}))
    original.csvsave(out/'input_videos.csv',inputvideos)
    for level in ['source','macro']:
        bins={}
        for v in inputvideos:
            key=(v['mode'],v['source_group_id']) if level=='source' else (v['mode'],)
            bins.setdefault(key,[]).append(v)
        rows=[dict(mode=k[0],**({'source_group_id':k[1]} if level=='source' else {}),**{f:original.mean_metric(vs,f) for f in ['coverage','context_mean_abs','context_RMS','detail_mean_abs','detail_RMS']}) for k,vs in bins.items()]
        original.csvsave(out/('input_'+level+'s.csv'),rows)
        if level=='source':inputvideos=rows
        else:input_macro=rows
    totals=Counter()
    for r in results:totals.update(r['calls'])
    summary=dict(status='COMPLETE' if scope=='development' else 'SMOKE_PASS_FULL_NOT_STARTED',scope=scope,exit_code=0,seconds=seconds,
        primary=primary,source_rows=source_rows,interpretation=interpretation,region_macro=macro,input_macro=input_macro,monitor_comparison=monitor[1] if monitor else [],
        invocation_calls=dict(calls),completed_video_calls_including_reuse=dict(totals),new_training=0,new_MoGe=0,new_DINO=0)
    if monitor:
        summary['monitor_z0_macro']=monitor[0]['macro']
        summary['monitor_minus_z0']=[dict(condition=r['condition'],epoch=r['epoch'],**{k:r[k]-monitor[0]['macro'][k] if r[k] is not None and monitor[0]['macro'][k] is not None else None for k in original.METRICS}) for r in monitor[1][1:]]
    for mode in MODES:
        rs=[r for r in macro if r['mode']==mode and r['pixels']]
        interpretation[mode]['regional_mean_abs']={r['region']+'/'+r['difference']:r['mean_abs'] for r in rs}
        interpretation[mode]['threshold_flip_pixel_ratios']={r['region']+'/'+r['difference']:dict(positive_to_negative=r['positive_to_negative_pixel_ratio'],negative_to_positive=r['negative_to_positive_pixel_ratio']) for r in rs if r['difference']!='zero_remaining'}
    save(out/'summary.json',summary);report(out,summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=data.OUT);parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output/('smoke' if args.smoke else 'development')
    assert not args.output.resolve().is_relative_to(args.source.resolve()),'Do not write old experiment'
    out.mkdir(parents=True,exist_ok=True);calls=Counter();started=time.monotonic();stage='identity';lock=None
    try:
        lock=(out/'invocation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=source_identity(args.source);identity['scope']='smoke' if args.smoke else 'development'
        if (out/'identity.json').exists():assert data.read(out/'identity.json')==identity,'DIAGNOSIS_IDENTITY_CHANGED'
        else:save(out/'identity.json',identity)
        protocol=data.read(args.source/'protocol.json');rows=protocol['rows'];scope='smoke' if args.smoke else 'development'
        selected=[r for r in rows if r['video_id']=='4ym5n4_fake'] if args.smoke else [r for r in rows if r['split']=='dev']
        assert len(selected)==(1 if args.smoke else 38)
        if args.smoke:assert selected[0]['actual_frames']==96
        save(out/'configuration.json',dict(scope=scope,identity=identity,source_head='dc17fc8c7a6200378e988236b08069770d42c069',
            diagnosis_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),device='CUDA BF16 / FP32 native scoring',
            modes=list(MODES),readouts=['NORMAL','CTX_ZERO'],new_training=0,new_MoGe=0,new_DINO=0))
        monitor=None;stage='monitor_z0_CPU'
        if not args.smoke:
            if (out/'monitor_z0_CPU.json').exists():
                assert args.resume,'Existing results require --resume'
                monitor=(data.read(out/'monitor_z0_CPU.json'),[])
                # CSV is display only; reconstruct numeric rows rather than parsing strings.
                monitor=(monitor[0],[dict(condition='FROZEN_z0_CPU',epoch=0,**monitor[0]['macro'])]+[dict(condition=m,epoch=e,**data.read(args.source/m/f'epoch{e:02d}_monitor.json')['macro']) for m in MODES for e in [1,2,3]])
            else:monitor=monitor_cpu(args.source,protocol,out)
        results=[];stage='CUDA_environment'
        device=torch.device('cuda')
        for mode in MODES:
            pending=[]
            for row in selected:
                path=out/mode/'videos'/row['video_id']/'result.json'
                visual_identity=data.read(args.source/'visual'/row['video_id']/'identity.json')
                geometry_identity=data.read(args.source/'geometry'/row['video_id']/'identity.json')
                original_identity=data.read(args.source/'identity.json')
                expected_visual={k:row[k] for k in ['video_sha256','mask_sha256','pts','work_hw','native_hw']}
                expected_visual.update(adaptation_sha256=original_identity['adaptation_sha256'],preprocessing='SpatialBaseline epoch01 / BF16 eval / pad14')
                assert visual_identity==expected_visual,'VISUAL_INPUT_IDENTITY'
                for key in ['video_sha256','pts','native_hw','work_hw']:assert geometry_identity[key]==row[key],('GEOMETRY_INPUT_IDENTITY',key)
                assert geometry_identity['provider_sha256']==original_identity['provider_sha256']
                assert geometry_identity['scale']==data.SCALE and geometry_identity['h']==.5
                input_id=dict(diagnosis=json_hash(identity),row=json_hash(row),
                    visual_manifest=digest(args.source/'visual'/row['video_id']/'frames.json'),
                    visual_identity=digest(args.source/'visual'/row['video_id']/'identity.json'),
                    geometry_identity=digest(args.source/'geometry'/row['video_id']/'identity.json'))
                if path.exists():
                    assert args.resume,'Existing video result: use --resume'
                    r=data.read(path);assert r['identity']==input_id and r['status']=='COMPLETE'
                    results.append(r);emit('reuse_complete_video',mode=mode,video=row['video_id']);continue
                pending.append((row,path,input_id))
            if not pending:continue
            assert torch.cuda.is_available(),'CUDA_UNAVAILABLE'
            stage='load_checkpoint_'+mode;model=load_model(args.source,mode,identity,device,calls)
            before=state_hash(model.state_dict())
            for index,(row,path,input_id) in enumerate(pending):
                stage=mode+'/'+row['video_id'];emit('video_begin',mode=mode,video=row['video_id'],position=index+1,pending=len(pending),T=row['actual_frames'])
                previous=calls.copy()
                result=diagnose_video(model,args.source,row,mode,calls,smoke=args.smoke)
                assert state_hash(model.state_dict())==before,'PARAMETER_OR_BUFFER_MUTATED'
                result.update(mode=mode,identity=input_id,weights_buffers_unchanged=True,calls=dict(calls-previous))
                save(path,result);results.append(result);emit('video_complete',mode=mode,video=row['video_id'],seconds=result['seconds'],M=result['M'])
            del model;gc.collect();torch.cuda.empty_cache()
        stage='finalize';summary=finalize(out,results,args.source,scope,calls,time.monotonic()-started,monitor)
        save(out/'status.json',dict(status=summary['status'],exit_code=0,seconds=time.monotonic()-started,invocation_calls=dict(calls)))
        emit('finished',status=summary['status'],output=str(out),calls=dict(calls))
    except Exception as exc:
        failure=dict(status='FAILED',exit_code=1,stage=stage,error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started,calls=dict(calls))
        save(out/f'failure_{time.time_ns()}.json',failure)
        print(failure['traceback'],flush=True);raise SystemExit(1)
    finally:
        if lock:lock.close()


if __name__=='__main__':main()
