"""Numerical contracts on identical prepared inputs, never rerun providers."""
import copy
import io

import numpy as np
import torch

from .model import StreamingModel, frame_at, state_to

ATOL=2e-5
RTOL=2e-5


def run_sequence(model, frames, chunk=100000, resume_at=None):
    stream=StreamingModel(model); state=None; cells=[]; logits=[]
    for start in range(0,len(frames),chunk):
        for i in range(start,min(start+chunk,len(frames))):
            o=stream.forward_step(frames[i],state); state=o['state']
            cells.append(o['cell_logits'].cpu()); logits.append(o['frame_logit'].cpu())
            if resume_at==i+1:
                buffer=io.BytesIO(); torch.save(state_to(state,'cpu'),buffer);buffer.seek(0)
                state=state_to(torch.load(buffer,weights_only=False),frames[i]['x'].device)
    return torch.stack(cells),torch.stack(logits),o['video_logit'].cpu()


def compare(a,b):
    errors=[]
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,atol=ATOL,rtol=RTOL)
        errors.append(float((x-y).abs().max()))
    return {'cell_max_abs':errors[0],'frame_max_abs':errors[1],'aggregate_max_abs':errors[2]}


def validate_continuous(model,frames):
    if len(frames)<63: raise ValueError('VALIDATION_REQUIRES_MULTIPLE_BLOCKS')
    ref=run_sequence(model,frames)
    results={str(k):compare(ref,run_sequence(model,frames,k)) for k in (7,16,31)}
    results['restore_at_31']=compare(ref,run_sequence(model,frames,16,31))
    prefix=run_sequence(model,frames[:31])
    changed=list(frames); changed[31:]=[{**f,'x':f['x']+7} for f in frames[31:]]
    altered=run_sequence(model,changed)
    torch.testing.assert_close(prefix[0],altered[0][:31],atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(prefix[1],altered[1][:31],atol=ATOL,rtol=RTOL)
    unchanged_prefix=run_sequence(model,changed[:31])
    compare(prefix,unchanged_prefix)
    results['causality_prefix']=compare(prefix,unchanged_prefix)
    # Verify two-level LSE against all actual cells, including strict zero-area exclusion.
    areas=torch.stack([f['area'].cpu() for f in frames]).double()
    direct=torch.logsumexp(ref[0].double()+torch.where(areas>0,areas.clamp_min(1).log(),-torch.inf),(0,1,2))-areas.sum().log()
    torch.testing.assert_close(direct,ref[2],atol=ATOL,rtol=RTOL)
    results['one_shot_area_lse_max_abs']=float((direct-ref[2]).abs())
    return {'status':'PASS','frames':len(frames),'atol':ATOL,'rtol':RTOL,'comparisons':results}


def validate_window(model,batch):
    with torch.inference_mode():
        old=model(batch,return_evidence=True)
        cells,frames,aggregate=run_sequence(model,[frame_at(batch,i) for i in range(batch['x'].shape[1])])
    torch.testing.assert_close(cells[:,0],old['cell_logits'][0].cpu(),atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(aggregate,old['window_logit'][0].double().cpu(),atol=ATOL,rtol=RTOL)
    return {'status':'PASS','frames':batch['x'].shape[1],'cell_max_abs':float((cells[:,0]-old['cell_logits'][0].cpu()).abs().max()),
            'aggregate_max_abs':float((aggregate-old['window_logit'][0].cpu()).abs()),'atol':ATOL,'rtol':RTOL}
