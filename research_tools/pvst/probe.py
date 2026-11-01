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


def resource(config,out,condition,targets,offload=False):
    rows=json.loads((out/'manifest.json').read_text())['rows'];r=max([r for r in rows if r['split']=='train'],key=lambda r:(r['native_pixels'],r['dataset']=='ANAKIN' and r['role']=='fake'))
    seed_all(config['seed']);torch.set_num_threads(4)
    model=PVST(condition,decoder_only(config),config).cuda().train();model._bn={k:v.cpu().clone() for k,v in model.decoder.named_buffers()}
    opt=torch.optim.AdamW(parameters(config,model),weight_decay=config['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=False)
    # Scale only from this real training frame for an interface resource probe;
    # discarded smoke weights; formal scale fit on all selected training later.
    from .data import native_dense
    import numpy as np
    q,v=native_dense(r,0);s=float(np.median(q[...,2][v&(q[...,2]>0)]))
    from scripts.prepare_v8_assets import digest
    scale_sha=None
    if (out/'scale.json').exists():s=json.loads((out/'scale.json').read_text())['s'];scale_sha=digest(out/'scale.json')
    read_start=time.monotonic();video=Video(out,r,condition,s);frames=video.frames(0,16+targets,'cuda');read_seconds=time.monotonic()-read_start;gt=[video.gt(i) for i in range(16,16+targets)]
    labels=[video.label(i) for i in range(16,16+targets)];video.close()
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();details=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        context=torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
        with context,torch.autocast('cuda',dtype=torch.bfloat16):
            print('RESOURCE_STEP_FORWARD',condition,step+1,flush=True);pred=model(frames,16);ls=[losses(p,g,y,config['dice_epsilon'])[0] for p,g,y in zip(pred,gt,labels)];loss=torch.stack(ls).mean()
        print('RESOURCE_STEP_BACKWARD',condition,step+1,flush=True);scaler.scale(loss).backward();scaler.unscale_(opt)
        grad=float(model.geo[0].weight.grad.abs().sum()) if model.geo[0].weight.grad is not None else 0.
        occupancy=[{'fine':p['voxels_fine'],'coarse':p['voxels_coarse'],'valid_pixels':p['valid_pixels'],
            'singleton_voxels':int((p['counts']==1).sum()),'points_per_voxel_mean':float(p['counts'].float().mean()),'points_per_voxel_max':int(p['counts'].max())} for p in pred]
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));scaler.step(opt);scaler.update();model.check_bn()
        details.append({'step':step+1,'loss':float(loss.detach()),'geometry_gradient_abs_sum':grad,'gradient_norm':norm})
        if step==1 and grad<=0:raise ValueError('GEOMETRY_PATH_NOT_TRAINED_AFTER_TWO_UPDATES')
        del pred,ls,loss
    torch.cuda.synchronize()
    return {'status':'PASS','condition':condition,'targets':targets,'left_halo':16,'offload_saved_tensors':offload,
        'video_id':r['video_id'],'native_hw':r['source_hw'],'spatial_GT':r['dataset']=='ANAKIN','static_routing_cache':True,'frames':16+targets,'data_read_and_routing_s':read_seconds,'seconds_two_updates':time.monotonic()-start,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'execution':{k:config[k] for k in ['tile_size','attention_chunk','point_chunk','attention_backend']},'parameters':sum(p.numel() for p in model.parameters()),'details':details,'probe_scale':s,
        'resource_scale_sha256':scale_sha,'scale_scope':'frozen all-training scale' if scale_sha else 'single training-frame resource probe only; formal train-only fixed scale fit separately',
        'occupancy':occupancy,'fine_neighbors':model.fine_time.last_stats,'coarse_neighbors':model.coarse_time.last_stats,
        'FLOPs':'not profiled; node/edge counts recorded, two conditions do not have matched compute',
        'weights':'discarded; no performance claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('docs/pvst/config_v02.json'));p.add_argument('--condition',required=True,choices=['UV_PVST','XYZ_PVST']);p.add_argument('--targets',type=int,default=8);p.add_argument('--offload',action='store_true');a=p.parse_args()
    c=json.loads(a.config.read_text());o=Path(c['output']);suffix='_sdpa_route_cache_spatialGT'+('_train_scale' if (o/'scale.json').exists() else '')+f"_c{c['attention_chunk']}_t{c['tile_size']}";path=o/f'resource_{a.condition}_targets{a.targets}_offload{int(a.offload)}{suffix}.json';started=time.monotonic()
    if path.exists():raise FileExistsError('RESOURCE_RECORD_ALREADY_EXISTS_DO_NOT_OVERWRITE:'+str(path))
    try:r=resource(c,o,a.condition,a.targets,a.offload)
    except BaseException as e:
        r={'status':'RESOURCE_BLOCKED' if isinstance(e,torch.OutOfMemoryError) else 'FAILED','condition':a.condition,'targets':a.targets,'offload':a.offload,
           'error':str(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-started,'pid':os.getpid(),
           'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
           'peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
        save(path,r);print(json.dumps(r),flush=True);raise
    save(path,r);print(json.dumps(r),flush=True)

if __name__=='__main__':main()
