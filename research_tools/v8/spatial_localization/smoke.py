"""Bounded real-media frontend/interface check; random detector, no metrics/training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import sha256_file, write_json_atomic
from research_tools.v8.full_coverage_observation_field.frontend import MogeV2, CoTrackerShortPrefix
from research_tools.v8.full_coverage_observation_field.pipeline import FrozenResNet18Spatial
from research_tools.v8.full_video_streaming.inputs import Frontend, decode
from research_tools.v8.component_global_context_pilot.model import GlobalContextModel
from .data import media_identity, to_source


def check_model(batch, condition, device):
    torch.manual_seed(20260909)
    # Explicit identity scaling for an interface check, not a fitted standardizer.
    stats={'absolute_mean':[0.]*3,'absolute_scale':[1.]*3,'relative_scale':[1.]*3}
    model=GlobalContextModel(condition,stats,patch_chunk=256).to(device)
    start=time.perf_counter();result=model(batch,return_evidence=True)
    assert all(torch.isfinite(v).all() for v in result.values())
    result['window_logit'].sum().backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum()>0 for g in gradients)
    # Actual padding perturbation must not affect detector outputs.
    altered={k:v.clone() for k,v in batch.items()};padding=altered['area']==0
    altered['x'][padding]=999.;altered['mu'][padding]=999.;altered['valid'][padding]=True
    with torch.no_grad():other=model(altered,return_evidence=True)
    torch.testing.assert_close(other['window_logit'],result['window_logit'],atol=1e-6,rtol=1e-6)
    if torch.device(device).type == "cuda": torch.cuda.synchronize()
    return result['cell_logits'].detach().cpu().numpy(),{'condition':condition,'initialization':'RANDOM_INTERFACE_ONLY',
        'forward_backward_padding_check_s':time.perf_counter()-start,'finite_gradients':True,
        'parameter_count':sum(p.numel() for p in model.parameters())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=4);p.add_argument('--start-s',type=float,default=4.)
    a=p.parse_args()
    if not 2<=a.frames<=16 or a.start_s<0:raise ValueError('BOUNDED_SMOKE_REQUIRED')
    if a.output.exists():raise ValueError('USE_NEW_OUTPUT_DIRECTORY')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();device='cuda:0'
    moge=MogeV2(str(a.root/'checkpoints/moge-2-vits-normal.pt'),device)
    tracker=CoTrackerShortPrefix(str(a.root/'checkpoints/scaled_online.pth'),device)
    rgb=FrozenResNet18Spatial(device)
    data=a.root/'data/sparse_3d_forgery_detection'
    fake=list((data/'media/video').rglob('1LARL+*.mp4'))
    if len(fake)!=1:raise ValueError('ONE_EXACT_FAKE_REQUIRED')
    media=[('real',data/'media/charades/Charades_v1_480/1LARL.mp4'),('fake',fake[0])]
    stats={'cell_mean':[0.]*77,'cell_scale':[1.]*77}
    grouping={'spatial_scale':1.,'motion_scale':1.,'merge_threshold':1.}
    report={'status':'INTERFACE_SMOKE_ONLY','device':torch.cuda.get_device_name(0),
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'frame_selection':{'start_s':a.start_s,'consecutive_frames':a.frames},
            'scaling':'IDENTITY_UNFITTED_SMOKE_ONLY','grouping':grouping,'videos':[],
            'weights':{str(x.relative_to(a.root)):sha256_file(x) for x in [a.root/'checkpoints/moge-2-vits-normal.pt',a.root/'checkpoints/scaled_online.pth',Path(rgb.checkpoint)]}}
    annotations=[]
    for role,path in media:
        media_path=str(path.relative_to(data/'media'));vid=media_identity('charades:1LARL',role,media_path)
        row={'source_id':'charades:1LARL','role':role,'media_path':media_path,'video_id':vid,'video_sha256':sha256_file(path)}
        front=Frontend(a.output/'no_legacy_cache',a.output,moge,tracker,rgb,stats,grouping,'new-server-interface-smoke-v1')
        arrays=[];infos=[];context=[];previous=None
        for fi,bgr,info in decode(path):
            if info['pts_s']<a.start_s:continue
            if info['timestamp_source']!='NATIVE_PTS':raise ValueError('NATIVE_PTS_REQUIRED')
            arr,meta,ctx,_=front.build(row,bgr,info,context,previous)
            arrays.append(arr);infos.append(meta);context=(context+[ctx])[-2:];previous=arr['area']>0
            name=f'{vid}_{fi:06d}.png';cv2.imwrite(str(a.output/name),bgr)
            annotations.append({**row,**info,'source_hw':list(bgr.shape[:2]),'image_path':name,
                'target':'visible_forgery_anomaly','annotation_status':'unannotated','mask_path':None,
                'mapping':meta['mapping']})
            if len(arrays)==a.frames:break
        if len(arrays)!=a.frames:raise ValueError('INSUFFICIENT_DECODED_FRAMES')
        batch={k:torch.from_numpy(np.stack([r[k] for r in arrays]))[None].to(device) for k in arrays[0]}
        models=[]
        for condition in ('GLOBAL_CELL','GLOBAL_COMPONENT'):
            logits,metrics=check_model(batch,condition,device);models.append(metrics)
            for ti,meta in enumerate(infos):
                canvas=np.repeat(np.repeat(logits[0,ti].reshape(16,16),16,0),16,1)
                mapped=to_source(canvas,meta['mapping'])
                assert mapped.shape==tuple(meta['mapping']['source_hw']) and np.isfinite(mapped).all()
        report['videos'].append({**row,'frames':infos,'frontend_cost_s':front.cost,'models':models,
            'input_shapes':{k:list(v.shape) for k,v in batch.items()}})
    report['wall_s']=time.perf_counter()-started;report['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['output_bytes_before_report']=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    write_json_atomic(a.output/'annotations.json',{'schema':'v8-visible-anomaly-annotation-v1','rows':annotations})
    write_json_atomic(a.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='videos'},indent=2))

if __name__=='__main__':main()
