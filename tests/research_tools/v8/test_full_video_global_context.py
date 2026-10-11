import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research_tools.v8.component_global_context_pilot.model import GlobalContextModel
from research_tools.v8.full_video_global_context.model import GlobalStreamingModel
from research_tools.v8.full_video_global_context.validation import sequence
from research_tools.v8.full_video_global_context.cache import FIELDS, write_block, read_block
from research_tools.v8.full_video_streaming.model import frame_at

torch.set_num_threads(2)
STATS={'absolute_mean':[1.,2.,3.],'absolute_scale':[2.,3.,4.],'relative_scale':[1.,1.,1.]}


def sample():
    torch.manual_seed(4)
    content=torch.zeros(1,4,256,16,16,dtype=torch.bool);content[:,:,[0,1,200]]=True
    valid=content.clone();valid[:,:,200]=False
    ids=torch.arange(256)[None,None].expand(1,4,256).clone();ids[:,:,1]=0
    return {'x':torch.randn(1,4,256,77),'mu':torch.randn(1,4,256,3),'valid':valid,'content':content,
            'area':content.sum((-1,-2)).float(),'component_ids':ids,
            'predecessor':torch.arange(256)[None,None,:,None].expand(1,4,256,4).clone(),
            'association_weight':torch.zeros(1,4,256,4),'history_available':torch.zeros(1,4,256,dtype=torch.bool),
            'delta_t':torch.tensor([[0.,.1,.12,.13]])}


@pytest.mark.parametrize('condition',['GLOBAL_COMPONENT','GLOBAL_CELL'])
def test_stream_global_path_chunk_restore_causal(condition):
    model=GlobalContextModel(condition,STATS).eval();batch=sample()
    frames=[frame_at(batch,i) for i in range(4)]
    calls=[];hook=model.region_attention.register_forward_hook(lambda *args:calls.append(1))
    reference=sequence(model,frames);hook.remove()
    assert len(calls)==4
    with torch.no_grad():expected=model(batch,True)
    torch.testing.assert_close(reference[0][:,0],expected['cell_logits'][0],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(reference[2],expected['window_logit'][0].double(),atol=2e-5,rtol=2e-5)
    for a,b in zip(reference,sequence(model,frames,2,2)):torch.testing.assert_close(a,b,atol=0,rtol=0)
    altered=copy.deepcopy(frames);altered[2]['x']+=100
    torch.testing.assert_close(reference[0][:2],sequence(model,altered)[0][:2],atol=0,rtol=0)
    assert torch.isfinite(reference[0][:,0,[0,1,200]]).all()


def test_cache_exact_masks_identity_and_model_output(tmp_path):
    batch=sample();arrays=[{k:batch[k][0,i].numpy() for k in FIELDS} for i in range(4)]
    path=tmp_path/'block.npz';write_block(path,arrays,[{'frame_index':i} for i in range(4)],'identity','video')
    saved,_=read_block(path,'identity','video')
    for a,b in zip(arrays,saved):
        for k in FIELDS: np.testing.assert_array_equal(a[k],b[k])
    with pytest.raises(ValueError,match='IDENTITY'):read_block(path,'wrong','video')
    frames=[{k:torch.from_numpy(v)[None] for k,v in a.items()} for a in saved]
    model=GlobalContextModel('GLOBAL_COMPONENT',STATS).eval()
    for a,b in zip(sequence(model,[frame_at(batch,i) for i in range(4)]),sequence(model,frames)):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_partitions_renumbering_padding_and_lse():
    batch=sample();frames=[frame_at(batch,i) for i in range(4)]
    for condition in ('GLOBAL_COMPONENT','GLOBAL_CELL'):
        model=GlobalContextModel(condition,STATS).eval();ref=sequence(model,frames)
        changed=copy.deepcopy(frames)
        for f in changed:f['component_ids']=torch.randperm(256)[f['component_ids']]
        torch.testing.assert_close(ref[0],sequence(model,changed)[0],atol=2e-5,rtol=2e-5)
        padding=copy.deepcopy(frames)
        for f in padding:
            inactive=f['area']==0;f['x'][inactive]=1e5;f['mu'][inactive]=1e5
        torch.testing.assert_close(ref[0],sequence(model,padding)[0],atol=0,rtol=0)
        area=batch['area'][0].double();logits=ref[0][:,0].double()
        direct=torch.logsumexp(logits+torch.where(area>0,area.clamp_min(1).log(),-torch.inf),(0,1))-area.sum().log()
        torch.testing.assert_close(ref[2],direct,atol=1e-12,rtol=1e-12)
        singleton=copy.deepcopy(frames)
        for f in singleton:f['component_ids']=torch.arange(256)[None]
        if condition=='GLOBAL_CELL':torch.testing.assert_close(ref[0],sequence(model,singleton)[0],atol=0,rtol=0)
        else:assert (ref[0]-sequence(model,singleton)[0]).abs().max()>1e-7
