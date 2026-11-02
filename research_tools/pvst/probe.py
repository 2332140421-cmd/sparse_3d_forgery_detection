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
    return {name:{'active_layer':getattr(module,'active_layer',None),'layers':[m.last_execution for m in module.layers]}
            for name,module in [('fine_time',model.fine_time),('coarse_time',model.coarse_time)]}


def scan_windows(config,out,progress):
    """Count exact native occupied coords only; no frontend or model inference.

    All48 cached videos, actual target8/left16 windows. Scalar injective keys
    avoid retaining per-pixel/node features; counts are a resource audit only.
    """
    import numpy as np
    from .data import native_dense
    from scripts.prepare_v8_assets import digest
    path=out/'resource_windows.json';identity={'manifest_sha256':digest(out/'manifest.json'),'scale_sha256':digest(out/'scale.json'),'target_frames':config['target_frames'],'left_halo':16,'counter':'native_floor_exact_v1'}
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('RESOURCE_WINDOW_IDENTITY_CHANGED')
        return saved
    rows=json.loads((out/'manifest.json').read_text())['rows'];scale=json.loads((out/'scale.json').read_text())['s'];records=[];started=time.monotonic()
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
        ('known_failed_video',max([w for w in records if w['video_id']=='cb41b3d004b6640f6c3b5bb1'],key=lambda w:w['coarse_nodes'])),
        ('max_fine_nodes',max(records,key=lambda w:w['fine_nodes'])),('max_coarse_nodes',max(records,key=lambda w:w['coarse_nodes']))]
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
