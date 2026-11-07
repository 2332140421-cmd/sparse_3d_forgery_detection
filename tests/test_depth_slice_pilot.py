"""Research contracts for ADR-0033, no old PVST training regression matrix."""
import torch
from research_tools.depth_slice.model import DepthSliceRelation, slab_ids, pool, terms


def test_slab_floor_holes_and_coverage():
    q=torch.tensor([[[-.2,1.,-.01],[9.,3.,.125],[.1,4.,.124],[0.,0.,1.]]])
    v=torch.tensor([[True,True,True,False]])
    ids,ix=slab_ids(q,v)
    assert ids.tolist()==[-1,0,1] and ix.tolist()==[0,2,1]
    f=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)
    pooled=pool(f,torch.tensor([0,1,0]),2)
    torch.testing.assert_close(pooled,torch.tensor([[3.,4.,5.,6.],[3.,4.,3.,4.]]))
    pooled.sum().backward();assert torch.isfinite(f.grad).all()


def example(t=3):
    torch.manual_seed(20260909);fs=[]
    for i in range(t):
        q=torch.randn(3,4,3);q[...,2]=torch.tensor([.1,.2,.3,.4])[None]
        fs.append({'visual':torch.randn(512,1,1),'xyz':q,'valid':torch.ones(3,4,dtype=torch.bool),'pts':[0.,.07,.19][i], 'gt':torch.eye(3,4,dtype=torch.bool)})
    return fs


def test_causal_relation_and_local_pixel_detail():
    torch.set_num_threads(2);torch.manual_seed(7);m=DepthSliceRelation().eval();fs=example()
    with torch.no_grad():
        a=m(fs);fs[-1]['visual']+=100;fs[-1]['xyz']*=3;b=m(fs)
    for i in [0,1]:
        torch.testing.assert_close(a[i]['pixel'],b[i]['pixel'],atol=2e-5,rtol=3e-5)
        torch.testing.assert_close(a[i]['frame'],b[i]['frame'],atol=2e-5,rtol=3e-5)
    assert a[0]['pixel'].std()>0


def test_checkpoint_all_branches_and_invalid_pixels():
    torch.set_num_threads(2);torch.manual_seed(9);m=DepthSliceRelation().train();fs=example(2)
    fs[0]['valid'][0]=False
    pred=m(fs);loss=torch.stack([terms(p,f['gt']) for p,f in zip(pred,fs)]).mean(0).sum();loss.backward()
    for prefix in ['geo','fusion','token','time','depth','context','pixel','frame_head','frame_visual']:
        gs=[p.grad for n,p in m.named_parameters() if n.startswith(prefix)]
        assert all(g is not None and torch.isfinite(g).all() for g in gs),prefix
        assert sum(float(g.abs().sum()) for g in gs)>0,prefix
    assert pred[0]['pixel'].shape==(3,4)
    fs=example(1);fs[0]['valid'][:]=False;m.eval()
    with torch.no_grad():p=m(fs)[0]
    assert p['slices']==0 and torch.isfinite(p['pixel']).all() and torch.isfinite(p['frame'])


def test_empty_gt_is_supervised_not_missing():
    p={'pixel':torch.ones(2,3,requires_grad=True),'frame':torch.tensor(1.,requires_grad=True)}
    a=terms(p,torch.zeros(2,3,dtype=torch.bool));assert torch.isfinite(a).all()
    a.sum().backward();assert (p['pixel'].grad>0).all() and p['frame'].grad>0
