"""Small saved-model checks, with the existing streaming numerical tolerance."""
import io
import tempfile
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.full_video_streaming.model import frame_at, state_to
from research_tools.v8.full_video_streaming.validation import ATOL, RTOL, compare
from .model import GlobalStreamingModel
from .cache import read_block, write_block, FIELDS


def sequence(model,frames,chunk=100000,restore=None):
    stream=GlobalStreamingModel(model);state=None;cells=[];logits=[]
    for start in range(0,len(frames),chunk):
        for i in range(start,min(start+chunk,len(frames))):
            result=stream.forward_step(frames[i],state);state=result['state']
            cells.append(result['cell_logits'].cpu());logits.append(result['frame_logit'].cpu())
            if restore==i+1:
                buf=io.BytesIO();torch.save(state_to(state,'cpu'),buf);buf.seek(0)
                state=state_to(torch.load(buf,weights_only=False),frames[i]['x'].device)
    return torch.stack(cells),torch.stack(logits),result['video_logit'].cpu()


def validate(model,batch):
    frames=[frame_at(batch,i) for i in range(batch['x'].shape[1])]
    with torch.inference_mode():
        old=model(batch,True)
        ref=sequence(model,frames)
    torch.testing.assert_close(ref[0][:,0],old['cell_logits'][0].cpu(),atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(ref[1],old['frame_logits'][0].double().cpu(),atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(ref[2],old['window_logit'][0].double().cpu(),atol=ATOL,rtol=RTOL)
    results={'window_cell_max_abs':float((ref[0][:,0]-old['cell_logits'][0].cpu()).abs().max()),
             'window_max_abs':float((ref[2]-old['window_logit'][0].cpu()).abs()),'atol':ATOL,'rtol':RTOL}
    # Repeat exact prepared frames only for state/chunk contracts, not as frontend input.
    longer=(frames*4)[:64];base=sequence(model,longer)
    results['chunks']={str(c):compare(base,sequence(model,longer,c)) for c in (7,16,31)}
    results['restore']=compare(base,sequence(model,longer,16,31))
    changed=[{k:v.clone() for k,v in f.items()} for f in longer]
    for f in changed[31:]: f['x']+=7
    alt=sequence(model,changed)
    torch.testing.assert_close(base[0][:31],alt[0][:31],atol=ATOL,rtol=RTOL)
    areas=torch.stack([f['area'].cpu() for f in longer]).double()
    direct=torch.logsumexp(base[0].double()+torch.where(areas>0,areas.clamp_min(1).log(),-torch.inf),(0,1,2))-areas.sum().log()
    torch.testing.assert_close(base[2],direct,atol=ATOL,rtol=RTOL)
    results['direct_lse_max_abs']=float((base[2]-direct).abs())
    with tempfile.TemporaryDirectory(prefix='v8-compact-') as tmp:
        path=Path(tmp)/'block.npz'
        arrays=[{k:f[k][0].cpu().numpy() for k in FIELDS} for f in frames]
        write_block(path,arrays,[{'frame_index':i} for i in range(len(arrays))],'test','video')
        restored,_=read_block(path,'test','video')
        for a,b in zip(arrays,restored):
            for k in FIELDS:np.testing.assert_array_equal(a[k],b[k])
        restored=[{k:torch.from_numpy(v)[None].to(batch['x'].device) for k,v in f.items()} for f in restored]
        results['cache_roundtrip']=compare(ref,sequence(model,restored))
    called=[]
    hook=model.region_attention.register_forward_hook(lambda *args:called.append(True))
    sequence(model,frames[:1]);hook.remove()
    if not called: raise ValueError('GLOBAL_ATTENTION_NOT_CALLED')
    results.update(status='PASS',attention_called=True,causality_prefix_frames=31,
                   chunk_input='same supplied inputs, repeated only when shorter than 64; no new frontend or graph reconstruction')
    return results
