"""Real maximum-native-size training block; exactly two update gradient check.

Immediate caller train.py before formal training. Child-process resource failures
are recorded, never replaced by reduced channels, sampled points or fake metrics.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch
from contextlib import nullcontext
import torch.nn.functional as F

from .data import decoder_only,Video,save
from .model import PVST
from research_tools.v8.local_geometry_pilot.runner import seed_all


def parameters(config,model):
    decoder=list(model.decoder.parameters());ids={id(p) for p in decoder}
    other=[p for p in model.parameters() if id(p) not in ids]
    return [{'params':decoder,'lr':config['lr_decoder']},{'params':other,'lr':config['lr_new']}]


def losses(pred,gt,label,epsilon):
    lf=F.binary_cross_entropy_with_logits(pred['frame_logit'].float(),torch.as_tensor(float(label),device=pred['frame_logit'].device))
    if gt is None:return lf,lf,lf.new_zeros(()),lf.new_zeros(())
    target=torch.as_tensor(gt,device=lf.device,dtype=torch.float32);z=pred['pixel_logits'].float()
    b=F.binary_cross_entropy_with_logits(z,target);p=z.sigmoid()
    d=1-(2*(p*target).sum()+epsilon)/(p.sum()+target.sum()+epsilon)
    return lf+b+d,lf,b,d


def execution_signature(config):
    """Only this implementation/config can authorize resource evidence reuse."""
    from scripts.prepare_v8_assets import digest
    import hashlib
    return {'execution':{k:config[k] for k in ['tile_size','attention_chunk','point_chunk','attention_backend','ffn_chunk','execution_revision']},
        'implementation_sha256':{n:digest(Path(__file__).parent/n) for n in ['temporal.py','model.py','probe.py','train.py','data.py']},
        'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}


def model_trace(model):
    return {name:{'active_layer':getattr(module,'active_layer',None),'layers':[m.last_execution for m in module.layers],
            'edge_stats':[getattr(m,'edge_stats',None) for m in module.layers],'head_storage':[getattr(m,'head_storage',None) for m in module.layers],'qkv_execution':[getattr(m,'qkv_execution',None) for m in module.layers]}
            for name,module in [('fine_time',model.fine_time),('coarse_time',model.coarse_time)]}


def scan_windows(config,out,progress,source=None):
    """Count exact native occupied coords only; no frontend or model inference.

    All48 cached videos, actual target8/left16 windows. Scalar injective keys
    avoid retaining per-pixel/node features; counts are a resource audit only.
    """
    import numpy as np
    from .data import native_dense
    from scripts.prepare_v8_assets import digest
    source=out if source is None else source
    path=out/'resource_windows.json';identity={'manifest_sha256':digest(source/'manifest.json'),'scale_sha256':digest(source/'scale.json'),'target_frames':config['target_frames'],'left_halo':16,'counter':'native_floor_exact_v1'}
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('RESOURCE_WINDOW_IDENTITY_CHANGED')
        return saved
    rows=json.loads((source/'manifest.json').read_text())['rows'];scale=json.loads((source/'scale.json').read_text())['s'];records=[];started=time.monotonic()
    def unique_count(coords):
        if not len(coords):return 0
        lo=coords.min(0);span=coords.max(0)-lo+1
        if int(span[0])*int(span[1])*int(span[2])>=2**62:raise ValueError('COUNT_KEY_OVERFLOW')
        c=coords-lo;key=(c[:,0]*span[1]+c[:,1])*span[2]+c[:,2]
        return len(np.unique(key))
    for ri,r in enumerate(rows):
        progress('resource_coordinate_count',video_index=ri,video=r['video_id'],total=r['actual_frames'])
        counts=[]
        for i in range(r['actual_frames']):
            q,valid=native_dense(r,i);q=q/scale
            v=np.floor(q[valid]/np.float32(1/64)).astype(np.int64)
            counts.append((unique_count(v),unique_count(v//2)))
        arr=np.asarray(counts,np.int64);prefix=np.concatenate((np.zeros((1,2),np.int64),arr.cumsum(0)))
        for start in range(0,r['actual_frames'],config['target_frames']):
            end=min(start+config['target_frames'],r['actual_frames']);left=max(0,start-16);total=prefix[end]-prefix[left]
            records.append({'video_id':r['video_id'],'dataset':r['dataset'],'split':r['split'],'target_range':[start,end],'context_range':[left,start],
                'input_range':[left,end],'fine_nodes':int(total[0]),'coarse_nodes':int(total[1]),'native_hw':r['source_hw']})
    # Keep high resolution, the previously failing video, and BOTH fine/coarse maxima.
    largest=max([r for r in rows if r['split']=='train'],key=lambda r:(r['native_pixels'],r['dataset']=='ANAKIN' and r['role']=='fake'))
    candidates=[('original_high_resolution',next(w for w in records if w['video_id']==largest['video_id'] and w['target_range'][0]==16)),
        ('known_failed_window',next(w for w in records if w['video_id']=='cb41b3d004b6640f6c3b5bb1' and w['target_range']==[0,8])),
        ('max_fine_nodes_full_history',max([w for w in records if w['context_range'][1]-w['context_range'][0]==16 and w['target_range'][1]-w['target_range'][0]==8],key=lambda w:w['fine_nodes'])),('max_coarse_nodes_full_history',max([w for w in records if w['context_range'][1]-w['context_range'][0]==16 and w['target_range'][1]-w['target_range'][0]==8],key=lambda w:w['coarse_nodes']))]
    chosen={}
    for reason,w in candidates:
        key=(w['video_id'],*w['target_range']);chosen.setdefault(key,{**w,'reasons':[]})['reasons'].append(reason)
    result={'identity':identity,'seconds':time.monotonic()-started,'videos':len(rows),'frames':sum(r['actual_frames'] for r in rows),
        'windows':records,'representatives':list(chosen.values()),'note':'coordinate occupancy only, no labels/features/inference used for selection'}
    save(path,result);return result


def representative_resources(config,out,condition):
    """Only deduplicated selected windows; discard two-update smoke weights."""
    from scripts.prepare_v8_assets import digest
    from .train import build
    scan=json.loads((out/'resource_windows.json').read_text());rows={r['video_id']:r for r in json.loads((out/'manifest.json').read_text())['rows']}
    scale=json.loads((out/'scale.json').read_text())['s'];records=[];model=None;started=time.monotonic()
    target=out/f'resource_{condition}_{config["execution_revision"]}.json';signature=execution_signature(config)
    if target.exists():raise FileExistsError('RESOURCE_RECORD_EXISTS:'+str(target))
    torch.set_num_threads(4)
    try:
        for window in scan['representatives']:
            r=rows[window['video_id']];model=build(config,condition).train();opt=torch.optim.AdamW(parameters(config,model),weight_decay=config['weight_decay']);steps=[]
            assert sum(p.numel() for p in model.parameters())==17821672
            left,end=window['input_range'];start=window['target_range'][0];torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
            video=Video(out,r,condition,scale)
            try:
                frames=video.frames(left,end,'cuda');gt=[video.gt(i) for i in range(start,end)];labels=[video.label(i) for i in range(start,end)]
                fine=sum(len(f['voxel_meta']['coords']) for f in frames);coarse=sum(len(f['voxel_meta']['coarse_coords']) for f in frames)
                if condition=='XYZ_PVST' and (fine,coarse)!=(window['fine_nodes'],window['coarse_nodes']):raise ValueError('CPU_GPU_OCCUPANCY_COUNT_MISMATCH')
                for step in range(2):
                    opt.zero_grad(set_to_none=True)
                    print('REPRESENTATIVE',condition,window['video_id'],window['target_range'],'update',step+1,flush=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(frames,start-left);loss=torch.stack([losses(p,g,y,config['dice_epsilon'])[0] for p,g,y in zip(pred,gt,labels)]).mean()
                    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));grad=float(model.geo[0].weight.grad.abs().sum());opt.step();model.check_bn()
                    if step==1 and grad<=0:raise ValueError('REPRESENTATIVE_GEOMETRY_GRADIENT')
                    steps.append({'step':step+1,'gradient_norm':norm,'geometry_gradient_abs_sum':grad});del pred,loss
                torch.cuda.synchronize();records.append({**window,'fine_nodes_actual':fine,'coarse_nodes_actual':coarse,'seconds':time.monotonic()-tick,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'trace':model_trace(model),'updates':steps})
                del frames
            finally:video.close()
            del model,opt;model=None
        save(target,{'status':'PASS','condition':condition,**signature,'manifest_sha256':digest(out/'manifest.json'),'resource_scale_sha256':digest(out/'scale.json'),
             'windows_sha256':digest(out/'resource_windows.json'),'seconds':time.monotonic()-started,'records':records,'weights':'discarded; formal initialization unchanged'})
    except BaseException as e:
        save(target,{'status':'RESOURCE_BLOCKED' if isinstance(e,torch.OutOfMemoryError) else 'FAILED',**signature,'condition':condition,'error':str(e),'traceback':traceback.format_exc(),
            'completed':records,'window':window,'trace':model_trace(model) if model is not None else None,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('docs/pvst/config_v02.json'));p.add_argument('--condition',required=True,choices=['UV_PVST','XYZ_PVST']);a=p.parse_args()
    c=json.loads(a.config.read_text());representative_resources(c,Path(c['output']),a.condition)

if __name__=='__main__':main()


def select_v03_scale(config,out,progress):
    """Training-only occupancy budget, then frozen-scale train/dev audit.

    Caller v03 CLI; output small per-frame/window counts and frozen config.
    Existing v0.2 scanner fixes h=1/64, so cannot select the v0.3 scale.
    No geometry/features are generated or retained here.
    """
    import numpy as np
    from .data import native_dense
    from scripts.prepare_v8_assets import digest
    source=Path(config['asset_root']);rows=json.loads((source/'manifest.json').read_text())['rows']
    scale=json.loads((source/'scale.json').read_text())['s'];out.mkdir(parents=True,exist_ok=True)
    identity={'manifest':digest(source/'manifest.json'),'scale':digest(source/'scale.json'),'candidates':config['voxel_candidates'],'budget':config['voxel_budget'],'counter':'v03_exact_native_float32_floor_v1'}
    final=out/'scale_selection.json'
    if final.exists():
        saved=json.loads(final.read_text())
        if saved['identity']!=identity:raise ValueError('SCALE_SELECTION_IDENTITY')
        return saved
    def unique(c):
        if not len(c):return c
        lo=c.min(0);span=c.max(0)-lo+1
        if int(span[0])*int(span[1])*int(span[2])>=2**62:raise ValueError('COUNT_KEY_OVERFLOW')
        a=c-lo;key=(a[:,0]*span[1]+a[:,1])*span[2]+a[:,2]
        return c[np.unique(key,return_index=True)[1]]
    def count_row(r,hs):
        p=out/'coordinate_counts'/f'{r["video_id"]}.json';sig={**identity,'video_id':r['video_id'],'video_sha256':r['video_sha256'],'hs':hs}
        if p.exists():
            data=json.loads(p.read_text())
            if data['identity']!=sig:raise ValueError('COORDINATE_COUNT_IDENTITY')
            return data['counts']
        tick=time.monotonic();counts=[]
        for i in range(r['actual_frames']):
            q,v=native_dense(r,i);q=q/scale
            coords=unique(np.floor(q[v]/np.float32(hs[0])).astype(np.int64));levels=[len(coords)]
            # Power-of-two nested floors preserve negative coordinates exactly.
            for _ in range(len(hs)):
                coords=unique(coords//2);levels.append(len(coords))
            counts.append(levels)
        save(p,{'identity':sig,'counts':counts,'seconds':time.monotonic()-tick});return counts
    def windows(r,counts,j,h):
        arr=np.asarray(counts,np.int64)[:,j:j+2];pre=np.concatenate((np.zeros((1,2),np.int64),arr.cumsum(0)));result=[]
        for start in range(0,r['actual_frames'],8):
            end=min(start+8,r['actual_frames']);left=max(0,start-16);n=pre[end]-pre[left]
            result.append({'video_id':r['video_id'],'split':r['split'],'dataset':r['dataset'],'native_hw':r['source_hw'],'target_range':[start,end],'context_range':[left,start],'input_range':[left,end],'fine_nodes':int(n[0]),'coarse_nodes':int(n[1]),'voxel_size':h})
        return result
    started=time.monotonic();train={};hs=config['voxel_candidates']
    for ri,r in enumerate([r for r in rows if r['split']=='train']):
        progress('scale_train_count',video_index=ri,video=r['video_id']);train[r['video_id']]=count_row(r,hs)
    candidates=[];chosen=None;selected=[]
    for j,h in enumerate(hs):
        ww=[w for r in rows if r['split']=='train' for w in windows(r,train[r['video_id']],j,h)]
        record={'h':h,'fine_max':max(w['fine_nodes'] for w in ww),'coarse_max':max(w['coarse_nodes'] for w in ww)}
        record['within_budget']=record['fine_max']<=config['voxel_budget']['fine'] and record['coarse_max']<=config['voxel_budget']['coarse'];candidates.append(record)
        if chosen is None and record['within_budget']:chosen=h;selected=ww
    result={'identity':identity,'candidates':candidates,'selected_h':chosen,'selection_scope':'training coordinates only; no dev labels/scores','status':'SELECTED' if chosen else 'RESOURCE_BLOCKED'}
    save(out/'training_scale_candidates.json',result)
    if chosen is not None:
        for ri,r in enumerate([r for r in rows if r['split']=='dev']):
            progress('frozen_scale_dev_count',video_index=ri,video=r['video_id']);selected+=windows(r,count_row(r,[chosen]),0,chosen)
        full=[w for w in selected if w['context_range'][1]-w['context_range'][0]==16 and w['target_range'][1]-w['target_range'][0]==8]
        worst=max(full,key=lambda w:max(w['fine_nodes']/200000,w['coarse_nodes']/50000))
        old=max([w for w in full if w['video_id']=='87664e1174a82a8f26ee463f'],key=lambda w:w['fine_nodes'])
        high=max(full,key=lambda w:(w['native_hw'][0]*w['native_hw'][1],w['fine_nodes']))
        reps={}
        for reason,w in [('highest_normalized_node_load',worst),('old_max_video',old),('high_native_pixels',high)]:
            key=(w['video_id'],*w['target_range']);reps.setdefault(key,{**w,'reasons':[]})['reasons'].append(reason)
        result.update(windows=selected,representatives=list(reps.values()))
    result['seconds']=time.monotonic()-started;save(final,result);return result
