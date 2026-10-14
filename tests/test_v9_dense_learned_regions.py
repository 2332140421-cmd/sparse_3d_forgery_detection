"""Research-critical dense/partition/causal and supervision contracts."""
import numpy as np
import torch
from research_tools.v9.dense_learned_regions.model import DenseRegions,relations,normalized_lse,CONDITIONS
from research_tools.v9.dense_learned_regions.data import choose_clips
from research_tools.v9.dense_learned_regions.train import parent_weights


def sample():
    torch.manual_seed(17)
    b={'rgb':torch.rand(3,3,8,8),'visual':torch.rand(3,64,2,2),'content':torch.ones(3,1,8,8,dtype=torch.bool),
       'xyz':torch.rand(3,3,8,8),'valid':torch.ones(3,1,8,8,dtype=torch.bool),'dt':torch.tensor([0.,.03,.04])}
    b['content'][...,:2,:]=False;b['valid'][...,3,3]=False
    return b


def test_relations_and_labels():
    b=sample();nums,m=relations(b['xyz'],b['valid']&b['content'])
    assert not m[:,1,:,-1].any() and not m[:,2,-1,:].any()
    assert not nums[..., :2,:].any()
    y=np.zeros(100,bool);y[20:60]=True;c=choose_clips(y)
    outside=next(v for v in c if v['kind']=='outside');assert not y[outside['indices']].any()
    row={'labels':y.tolist(),'clips':c};w=parent_weights(row);selected=np.array([y[i] for a in c for i in a['indices']])
    assert np.allclose([w[k]*(selected==k).sum() for k in (0,1)],[.5,.5])


def test_matching_rgb_geometry_independence_and_partition():
    b=sample();states=[]
    for c in CONDITIONS:
        torch.manual_seed(20260909);model=DenseRegions(c);states.append(model.state_dict())
        f=model.encode(b['rgb'],b['visual'],b['content'],b['xyz'],b['valid']);_,a=model.pool(f,b['content'])
        if c=='XYZ_PARTITION':assert torch.allclose(a.sum(-1)[b['content'].flatten(1)],torch.ones_like(a.sum(-1)[b['content'].flatten(1)]),atol=1e-6)
        assert not a[~b['content'].flatten(1)].any()
        if c=='RGB_QUERY':
            before=model.sequence(b)[0];other={k:v for k,v in b.items() if k not in ('xyz','valid')}
            assert torch.equal(before,model.sequence(other)[0])
    for key in states[0]:assert torch.equal(states[0][key],states[1][key]) and torch.equal(states[1][key],states[2][key])


def test_causality_chunk_reload_and_gradient(tmp_path):
    torch.set_num_threads(2);b=sample();model=DenseRegions('XYZ_PARTITION').eval()
    full,_=model.sequence(b);first,h=model.sequence({k:v[:2] for k,v in b.items()});last,_=model.sequence({k:v[2:] for k,v in b.items()},h)
    assert torch.allclose(full,torch.cat((first,last)),atol=1e-6)
    sequential=[];hidden=torch.zeros(64,256)
    for i in range(3):
        z,hidden=model.frame(b['rgb'][i:i+1],b['visual'][i:i+1],b['content'][i:i+1],b['xyz'][i:i+1],b['valid'][i:i+1],b['dt'][i],hidden)
        sequential.append(z)
    assert torch.allclose(full,torch.stack(sequential),atol=1e-6)
    b['rgb'].requires_grad_();out,_=model.sequence(b);out[-1].backward();assert b['rgb'].grad[0].abs().sum()>0
    p=tmp_path/'model.pt';torch.save(model.state_dict(),p);clone=DenseRegions('XYZ_PARTITION');clone.load_state_dict(torch.load(p,weights_only=True))
    assert torch.equal(model.sequence(b)[0],clone.sequence(b)[0])
    z=torch.tensor([1.,1.,-2.,-2.]);assert torch.allclose(normalized_lse(z,torch.ones(4),0),normalized_lse(torch.tensor([1.,-2.]),torch.tensor([2.,2.]),0))


def test_padding_invalid_geometry_and_checkpoint_gradient():
    b=sample();model=DenseRegions('XYZ_QUERY');baseline=model.sequence(b)[0]
    altered={k:v.clone() for k,v in b.items()}
    altered['rgb'][...,:2,:]=1000;altered['xyz'][...,:2,:]=-1000
    assert torch.allclose(baseline,model.sequence(altered)[0],atol=1e-6)
    # Geometry-invalid pixels remain in the point output, not excluded as padding.
    z,_,pixels,a=model.frame(b['rgb'][:1],b['visual'][:1],b['content'][:1],b['xyz'][:1],b['valid'][:1],b['dt'][0],torch.zeros(64,256),True)
    assert torch.isfinite(pixels[3,3]) and a[3*8+3].sum()>0
    b['rgb'].requires_grad_();plain=model.sequence(b)[0].sum();plain.backward()
    grad=b['rgb'].grad.clone();model.zero_grad(set_to_none=True);b['rgb'].grad=None
    checked=model.sequence(b,checkpoint_frames=True)[0].sum();checked.backward()
    assert torch.allclose(plain,checked,atol=1e-6) and torch.allclose(grad,b['rgb'].grad,atol=1e-6)
