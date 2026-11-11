"""One-video frozen-pool reproduction diagnosis; no repair or training.

CLI caller: compare same-local old/new pooling against historical C. Existing
extraction stopped without frame identity; this bounded entry saves that evidence.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .data import Video,save
from .experiment import REPO,git,state_hash,forward
from .set_fit_check import build
from .grid_localization import frozen_pool
from research_tools.depth_slice.model import slab_ids
from scripts.prepare_v8_assets import digest

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
BASE=ROOT/'pvst_rgb_partition_anakin_dev_v1';REF=ROOT/'pvst_slice_relation_readout_v1';FAILED=ROOT/'pvst_slab_grid_localization_v1'
OUT=ROOT/'pvst_slab_grid_pooling_parity_check_v1';DOC=REPO/'docs/pvst/slab_grid_pooling_parity_check_v1.md'
ATOL=2e-5;RTOL=3e-5


def read(p):return json.loads(Path(p).read_text())
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def difference(a,b,row_ids=None,slab_axis=0,samples=32):
    a=a.detach().cpu().float();b=b.detach().cpu().float();assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    bad=~torch.isclose(a,b,atol=ATOL,rtol=RTOL);a=a.double();b=b.double()
    d=(a-b).abs();allowed=ATOL+RTOL*b.abs();flat=int(d.argmax()) if d.numel() else 0
    coords=list(np.unravel_index(flat,tuple(d.shape))) if d.numel() else []
    coords=[int(x) for x in coords]
    def element(ix):
        key=tuple(ix);r={'index':list(ix),'actual':float(a[key]),'reference':float(b[key]),'signed_difference':float(a[key]-b[key]),'allowed':float(allowed[key])}
        if row_ids is not None:r['slab_id']=int(row_ids[ix[slab_axis]])
        return r
    values=[element(x.tolist()) for x in bad.nonzero()[:samples]]
    result={'passed':not bool(bad.any()),'bitwise_equal':torch.equal(a,b),'shape':list(a.shape),'elements':a.numel(),'mismatches':int(bad.sum()),'max_absolute':float(d.max()) if d.numel() else 0.,'max_relative_with_denominator_floor_1e_minus_30':float((d/b.abs().clamp_min(1e-30)).max()) if d.numel() else 0.,'max_absolute_element':element(coords) if coords else None,'bad_elements_first':values}
    if row_ids is not None:
        summary=[]
        for j in range(a.shape[slab_axis]):
            mask=bad.select(slab_axis,j)
            if mask.any():summary.append({'slab_id':int(row_ids[j]),'mismatches':int(mask.sum()),'max_absolute':float(d.select(slab_axis,j).max())})
        result['affected_rows']=summary
    return result


def protected():
    logs={name:digest(FAILED/name) for name in ['failure.json','execution.log','exit_code.txt','status.json']}
    return {'logs_sha256':logs,'partial_cache':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (FAILED/'cache').glob('*.pt')}}


@torch.no_grad()
def trace_pool(pool,local,xyz,valid,points,ids,index,affected,new):
    """Passive observation of actual denominator/partials, related point values.

    Never patch the formal pool. No traced dense tensors are written to disk.
    """
    selected=torch.isin(index,affected);state={'score':[],'value':[],'partials':[],'points_equal':None,'routing_equal':None};offset={'score':0,'value':0}
    def hook(name):
        def record(module,args,output):
            lo=offset[name];mask=selected[lo:lo+len(output)];state[name].append(output[mask].detach().cpu());offset[name]+=len(output)
        return record
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}));op=str(func)
            if isinstance(out,torch.Tensor):
                if op=='aten.nonzero.default' and tuple(out.shape)==(len(points),1):state['points_equal']=torch.equal(out[:,0],points)
                if op=='aten.scatter_reduce.two' and tuple(out.shape)==(len(ids),4):state['maximum']=out.detach().cpu()
                if op=='aten.index_add.default':
                    if tuple(out.shape)==(len(ids),4):state['denominator']=out.detach().cpu();state['routing_equal']=torch.equal(args[2],index)
                    if tuple(out.shape)==(len(ids),512):state['partials'].append(out.detach().cpu())
                if op=='aten.div.Tensor' and tuple(out.shape)==(len(points),4) and 'denominator' in state:state['weights']=out[selected].detach().cpu()
            return out
    handles=[pool.score.register_forward_hook(hook('score')),pool.value.register_forward_hook(hook('value'))]
    try:
        with Observe(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=local.is_cuda):
            result=frozen_pool(pool,local,points,[(ids,index)],valid.shape)[0] if new else pool(local,xyz,valid)[0]
    finally:
        for h in handles:h.remove()
    assert offset=={'score':len(points),'value':len(points)}
    state['score']=torch.cat(state['score']);state['value']=torch.cat(state['value']);state['point_slab_ids']=ids[index][selected].cpu();state['output']=result.detach().cpu()
    if new:state['points_equal']=True  # Actual explicit argument, no internal nonzero.
    return state


@torch.no_grad()
def attribute(pool,local,xyz,valid,points,ids,index,A,B,comparisons):
    # Include every bad slice, even if examples were truncated.
    bad=(~torch.isclose(A.cpu(),B.cpu(),atol=ATOL,rtol=RTOL)).any(1);affected=torch.where(bad.to(index.device))[0]
    aa=trace_pool(pool,local,xyz,valid,points,ids,index,affected,False);bb=trace_pool(pool,local,xyz,valid,points,ids,index,affected,True)
    assert torch.equal(aa['point_slab_ids'],bb['point_slab_ids'])
    stages={'score':difference(aa['score'],bb['score']), 'value':difference(aa['value'],bb['value']), 'softmax_maximum':difference(aa['maximum'],bb['maximum'],ids.cpu()), 'softmax_denominator':difference(aa['denominator'],bb['denominator'],ids.cpu()), 'normalized_weights':difference(aa['weights'],bb['weights']), 'weighted_chunk_partials':difference(torch.stack(aa['partials']),torch.stack(bb['partials']),ids.cpu(),slab_axis=1), 'pooled_output':difference(aa['output'],bb['output'],ids.cpu())}
    checks={'A_capture_vs_original':difference(aa['output'],A,ids.cpu()),'B_capture_vs_original':difference(bb['output'],B,ids.cpu()),'A_actual_points_equal_B':aa['points_equal'] and bb['points_equal'],'A_B_routing_equal_expected':aa['routing_equal'] and bb['routing_equal']}
    # Direct controlled replay of exactly the same full partial tensors and shapes.
    partials=torch.stack(aa['partials']).cuda();tree=partials.sum(0);sequential=torch.zeros_like(tree)
    for x in partials:sequential.add_(x)
    replay={'old_sum_vs_A':difference(tree,A,ids.cpu()),'sequential_vs_B':difference(sequential,B,ids.cpu()),'same_partials_tree_vs_sequential':difference(tree,sequential,ids.cpu())}
    head_denominators=[{'slab_id':int(ids[j]),'head':h,'A':float(aa['denominator'][j,h]),'B':float(bb['denominator'][j,h]),'difference':float(aa['denominator'][j,h]-bb['denominator'][j,h])} for j in affected.cpu().tolist() for h in range(4)]
    earliest=next((k for k,v in stages.items() if not v['bitwise_equal']),None)
    return {'branch':'A_B_MISMATCH','affected_slab_ids':ids[affected].cpu().tolist(),'related_valid_pixels':int(torch.isin(index,affected).sum()),'stage_order':['point_indices','score','value','softmax_maximum','softmax_denominator','normalized_weights','weighted_chunk_partials','pooled_output'],'stages':stages,'head_denominators':head_denominators,'capture_checks':checks,'earliest_non_bitwise_stage':earliest,'same_partials_reduction_replay':replay,'trace_did_not_exceed_original_tolerance':all(checks[k]['passed'] for k in ['A_capture_vs_original','B_capture_vs_original'])}


class StopBeforeSets(Exception):pass


@torch.no_grad()
def legacy_once(model,f,local,A,B,C,ids):
    captured={}
    def pool_pre(module,args):captured['local']=args[0].detach().cpu().clone()
    def pool_post(module,args,out):captured['C']=out[0].detach().cpu();captured['ids']=out[1].cpu()
    def sets_pre(module,args):
        assert torch.equal(args[0][0][0].cpu(),captured['C']);raise StopBeforeSets()
    handles=[model.pool.register_forward_pre_hook(pool_pre),model.pool.register_forward_hook(pool_post),model.sets.register_forward_pre_hook(sets_pre)]
    try:
        try:forward(model,[f])
        except StopBeforeSets:pass
        else:raise AssertionError('OLD_SETS_SHOULD_NOT_EXECUTE')
    finally:
        for h in handles:h.remove()
    assert torch.equal(captured['ids'],ids.cpu())
    return {'branch':'CACHE_REPRODUCTION_CHAIN','old_path_single_frame_reruns':1,'legacy_minus_A':difference(captured['C'],A,ids.cpu()),'legacy_minus_B':difference(captured['C'],B,ids.cpu()),'legacy_minus_C':difference(captured['C'],C,ids.cpu()),'legacy_local_vs_initial_local':difference(captured['local'],local.cpu(),samples=4),'legacy_local_stride':list(captured['local'].stride()),'initial_local_stride':list(local.stride()),'conclusion':'差异位于与旧缓存的复现链路，具体原因未确认；不因BF16或余差量级自行归因。'}


@torch.no_grad()
def main():
    started=time.monotonic();torch.set_num_threads(4);assert git('branch','--show-current')=='pvst-slab-grid-localization'
    config=read(BASE/'config.json');manifest=read(BASE/'manifest.json');baseline=read(BASE/'identity.json');old=read(FAILED/'identity.json')
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt';row=next(r for r in manifest['rows'] if r['pair_id']=='an:4z1827' and r['role']=='fake' and r['split']=='train');clips=[x for x in manifest['clips'] if x['video_id']==row['video_id']];wanted=sorted({i for x in clips for i in range(x['start'],x['end'])})
    identity={'execution_head':git('rev-parse','HEAD'),'baseline_head':'5c42f4be512f440e8bb24b3a1db0d27a7737a002','weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'config_sha256':digest(BASE/'config.json'),'pair_id':row['pair_id'],'video_id':row['video_id'],'frames_in_order':wanted,'clips':clips,'atol':ATOL,'rtol':RTOL,'precision':'original BF16 autocast; pool result FP32; CPU-only statistics FP64','source_sha256':{p:digest(REPO/p) for p in ['research_tools/rgbd_slice/model.py','research_tools/rgbd_slice/grid_localization.py','research_tools/rgbd_slice/pooling_parity_check.py']}}
    if OUT.exists():
        previous=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if previous.get('status')=='COMPLETE' and all(previous['identity'][k]==identity[k] for k in identity if k!='execution_head'):print('REUSE COMPLETE');return
        raise ValueError('EXISTING_OUTPUT_INSPECT_NO_OVERWRITE')
    OUT.mkdir();save(OUT/'tests.json',{'passed':4,'scope':'unchanged isclose tolerance/coordinates, all-element summary, partial slab-axis mapping, passive stage capture on a small pool'});save(OUT/'identity.json',identity);save(OUT/'selected_identity.json',row);original=protected();save(OUT/'original_preservation.json',original)
    assert len(wanted)==72 and identity['weight_sha256']==old['weight_sha256'] and identity['manifest_sha256']==old['manifest_sha256']==baseline['frozen_manifest_sha256'] and identity['config_sha256']==old['config_sha256']
    for name,sha in old['source_code'].items():assert digest(REPO/name)==sha
    for name,sha in baseline['code'].items():assert digest(REPO/name)==sha
    assert digest(Path(row['video_path']))==row['video_sha256'] and digest(Path(row['mask_zip']))==row['mask_sha256']
    asset=next(x for x in baseline['assets'] if x['video_id']==row['video_id'])
    for name,sha in asset['cache_sha256'].items():assert digest(Path(name))==sha
    cache_path=REF/'cache'/f"{row['video_id']}.pt";assert digest(cache_path)==read(REF/'cache_identity.json')['files'][cache_path.name]
    identity['reference_cache_sha256']=digest(cache_path);save(OUT/'identity.json',identity)
    reference=torch.load(cache_path,map_location='cpu',weights_only=True,mmap=True);assert sorted(reference)==wanted
    model=build(config,OUT,rgb_only=True).eval();cp=torch.load(weight,map_location='cpu',weights_only=True)
    assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==baseline['identity'];assert read(OUT/'initialization.json')['model_sha256']==cp['initial_model_sha256'];model.load_state_dict(cp['state_dict'],strict=True);del cp
    model.requires_grad_(False);before=state_hash(model.state_dict());v=Video(row,config);records=[];first=None;attribution=None;torch.cuda.reset_peak_memory_stats()
    try:
        with (OUT/'scan.jsonl').open('w') as log:
            for i in wanted:
                save(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'stage':'compare','frame_index':i,'completed_frames':len(records)})
                f=v.frame(i);xyz=f['xyz'].cuda();valid=f['valid'].cuda();ids,index=slab_ids(xyz,valid);points=torch.where(valid.reshape(-1))[0];cached=reference[i]
                assert torch.equal(ids.cpu(),cached['ids']) and f['pts']==row['pts'][i]['pts_s']==cached['pts'];assert torch.equal(torch.bincount(index,minlength=len(ids)).cpu(),cached['counts'])
                assert torch.equal(torch.bincount(index[f['gt'].cuda()[valid]],minlength=len(ids)).cpu(),cached['edit_counts'])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    local,_=model.local.encode(f['rgb'],torch.zeros_like(f['depth']))
                    A,aid=model.pool(local,xyz,valid);B=frozen_pool(model.pool,local,points,[(ids,index)],valid.shape)[0]
                assert torch.equal(aid,ids);C=cached['features'];assert A.shape==B.shape==C.shape
                comparisons={'A_minus_B':difference(A,B,ids.cpu()),'A_minus_C':difference(A,C,ids.cpu()),'B_minus_C':difference(B,C,ids.cpu())}
                item={'frame_index':i,'pts':f['pts'],'shape':list(A.shape),'slab_ids':ids.cpu().tolist(),'comparisons':comparisons};records.append(item);log.write(json.dumps(item)+'\n');log.flush()
                print('frame',i,'pts',f['pts'],'mismatches',{k:d['mismatches'] for k,d in comparisons.items()},flush=True)
                if any(not d['passed'] for d in comparisons.values()):
                    first={**item,'pair_id':row['pair_id'],'video_id':row['video_id'],'native_hw':row['source_hw'],'RGB_sha256':tensor_sha(f['rgb']),'normalized_XYZ_sha256':tensor_sha(f['xyz']),'valid_sha256':tensor_sha(f['valid']),'local_dtype':str(local.dtype),'local_shape':list(local.shape),'encoder_calls_before_attribution':len(records)};save(OUT/'first_failure.json',first)
                    if not comparisons['A_minus_B']['passed']:attribution=attribute(model.pool,local,xyz,valid,points,ids,index,A,B,comparisons)
                    else:attribution=legacy_once(model,f,local,A,B,C,ids)
                    save(OUT/'attribution.json',attribution);break
                del f,xyz,valid,ids,index,points,local,A,B
    finally:v.close()
    assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters());assert protected()==original and digest(weight)==identity['weight_sha256']
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'checked_frames':len(records),'requested_frames':len(wanted),'first_failure':first,'attribution':attribution,'scan_worst':{k:max(x['comparisons'][k]['max_absolute'] for x in records) for k in ['A_minus_B','A_minus_C','B_minus_C']},'ready_for_resource_acceptance':False,'frozen_model_unchanged':True,'original_failure_and_partial_cache_unchanged':True,'formal_updates':0,'resource_check_runs':0,'seconds':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    save(OUT/'summary.json',summary)
    from .pixel_message_dev_check import table
    lines=['# an:4z1827/fake冻结特征三方复现诊断',f"诊断COMPLETE、退出0；扫描{len(records)}/{len(wanted)}冻结训练帧，未训练、未运行资源验收。"]
    if first:
        lines += [f"首个可复现超差帧：{first['frame_index']}，PTS={first['pts']:.9f}s，视频{row['video_id']}，形状{first['shape']}。不是猜测首次进程失败的帧号。",table(['比较','超差元素','最大绝对误差','通过'],[[k,x['mismatches'],x['max_absolute'],x['passed']] for k,x in first['comparisons'].items()]),'有限归因：'+json.dumps(attribution,ensure_ascii=False)]
    else:lines+=['该视频全部冻结帧未复现；保留首次失败事实，不宣称已修复，不扫描其他视频。']
    lines += ['原容差atol2e-5/rtol3e-5不变；逐元素允许误差=atol+rtol×abs(reference)。细节、片号/通道和超差示例在first_failure.json及scan.jsonl。','冻结权重/模型缓冲/首次失败日志和退出码1/1008帧缓存均保持不变。新诊断完成状态不能覆盖原实验FAILED。不能进入资源验收，本轮不修改正式池化或口径。',f"耗时{summary['seconds']:.3f}秒；峰值GPU allocated/reserved {summary['peak_allocated']/1024**3:.3f}/{summary['peak_reserved']/1024**3:.3f}GiB；峰值RSS {summary['peak_RSS']/1024**3:.3f}GiB。",'执行HEAD '+identity['execution_head']+'；产物 '+str(OUT)]
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);DOC.write_text(report);save(OUT/'status.json',{'status':'COMPLETE','stage':'report_complete','exit_code':0})


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()})
        raise
