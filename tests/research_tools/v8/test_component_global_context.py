import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_tools.v8.component_global_context_pilot.model import GlobalContextModel, area_lse
from research_tools.v8.component_global_context_pilot.run import save_json

torch.set_num_threads(2)
STATS = {'absolute_mean': [1., 2., 3.], 'absolute_scale': [2., 3., 4.], 'relative_scale': [1., 1., 1.]}


def batch():
    torch.manual_seed(82)
    content = torch.zeros(1, 3, 256, 16, 16, dtype=torch.bool)
    content[:, :, [0, 1, 200]] = True
    content[:, :, 1, :, 8:] = False
    valid = content.clone();valid[:, :, 200] = False
    ids = torch.arange(256)[None, None].expand(1, 3, 256).clone();ids[:, :, 1] = 0
    return {'x': torch.randn(1, 3, 256, 77), 'valid': valid, 'content': content,
            'mu': torch.randn(1, 3, 256, 3), 'area': content.sum((-1,-2)).float(), 'component_ids': ids,
            'predecessor': torch.arange(256)[None,None,:,None].expand(1,3,256,4).clone(),
            'association_weight': torch.ones(1,3,256,4)/4, 'history_available': content.any((-1,-2)),
            'delta_t': torch.tensor([[0., .13, .17]])}


def test_paired_parameters_partition_and_id_invariance():
    torch.manual_seed(3);a=GlobalContextModel('GLOBAL_COMPONENT',STATS)
    torch.manual_seed(3);b=GlobalContextModel('GLOBAL_CELL',STATS)
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    inp=batch();renumbered=copy.deepcopy(inp)
    renumbered['component_ids']=torch.randperm(256)[inp['component_ids']]
    a.eval();b.eval()
    with torch.no_grad():
        torch.testing.assert_close(a(inp),a(renumbered),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(b(inp),b(renumbered),atol=0,rtol=0)
        changed=copy.deepcopy(inp);changed['component_ids'][:, :, 1]=1
        torch.testing.assert_close(b(inp),b(changed),atol=0,rtol=0)
        assert (a(inp)-a(changed)).abs().max()>1e-7


def test_region_weighting_and_remote_information_path():
    model=GlobalContextModel('GLOBAL_COMPONENT',STATS).eval();inp=batch()
    h=torch.randn(1,256,128);counts=(inp['valid'] & inp['content']).sum((-1,-2))[:,0]
    with torch.no_grad():
        context,nodes,valid=model.region_context(h,inp['component_ids'][:,0],inp['area'][:,0],inp['mu'][:,0],counts)
        expected_h=(h[:,0]*256+h[:,1]*128)/384
        torch.testing.assert_close(nodes[:,0,:128],expected_h)
        expected_mu=(inp['mu'][:,0,0]*256+inp['mu'][:,0,1]*128)/384
        torch.testing.assert_close(nodes[:,0,128:131],(expected_mu-model.absolute_mean)/model.absolute_scale)
        assert nodes[0,0,132]==.5
        # Cell 200 is spatially distant from 0; global attention reaches it.
        altered=h.clone();altered[:,200]+=torch.linspace(-4,4,128)
        other,_,_=model.region_context(altered,inp['component_ids'][:,0],inp['area'][:,0],inp['mu'][:,0],counts)
        assert (context[:,0]-other[:,0]).abs().max()>1e-5
        assert valid.sum()==2


def test_padding_missing_empty_and_causal_aggregation():
    model=GlobalContextModel('GLOBAL_COMPONENT',STATS).eval();a=batch();b=copy.deepcopy(a)
    inactive=a['area']==0
    b['x'][inactive]=1e5;b['mu'][inactive]=1e5
    b['valid'][~b['content']]=True
    # Non-content validity must never become geometry_count or a pooling member.
    with torch.no_grad():
        za,zb=model(a,True),model(b,True)
        torch.testing.assert_close(za['window_logit'],zb['window_logit'],atol=1e-6,rtol=1e-6)
        assert torch.isfinite(za['cell_logits'][~inactive]).all()
        two=area_lse(za['frame_logits'],a['area'].sum(2),(1,))
        torch.testing.assert_close(two,za['window_logit'],atol=1e-6,rtol=1e-6)
        b=copy.deepcopy(a);b['x'][:,2]+=20
        zc=model(b,True)
        torch.testing.assert_close(zc['cell_logits'][:,:2],za['cell_logits'][:,:2],atol=1e-6,rtol=1e-6)
        missing=copy.deepcopy(a);missing['valid'][:]=False;missing['mu'][:]=0;missing['association_weight'][:]=0
        assert torch.isfinite(model(missing)).all()
        ctx,_,_=model.region_context(torch.zeros(1,256,128),a['component_ids'][:,0],torch.zeros(1,256),torch.zeros(1,256,3),torch.zeros(1,256))
        assert torch.isfinite(ctx).all() and torch.count_nonzero(ctx)==0


def test_gradient_reload_and_microbatch_equivalence(tmp_path):
    inp=batch();inp={k:torch.cat((v,v),0) for k,v in inp.items()}
    a=GlobalContextModel('GLOBAL_COMPONENT',STATS);b=copy.deepcopy(a)
    y=torch.tensor([0.,1.]);w=torch.tensor([.4,1.6])
    loss=torch.nn.functional.binary_cross_entropy_with_logits(a(inp),y,weight=w);loss.backward()
    for i in range(2):
        one={k:v[i:i+1] for k,v in inp.items()}
        (torch.nn.functional.binary_cross_entropy_with_logits(b(one),y[i:i+1],weight=w[i:i+1],reduction='sum')/2).backward()
    for x,z in zip(a.parameters(),b.parameters()):
        assert x.grad is not None and torch.isfinite(x.grad).all()
        torch.testing.assert_close(x.grad,z.grad,atol=3e-6,rtol=4e-4)
    for module in (a.region_projection,a.region_attention,a.component_residual,a.cell_head,a.gru,a.patch_encoder):
        assert any(torch.count_nonzero(p.grad)>0 for p in module.parameters())
    path=tmp_path/'model.pt';torch.save(a.state_dict(),path);b.load_state_dict(torch.load(path,weights_only=True))
    a.eval();b.eval();torch.testing.assert_close(a(inp),b(inp),atol=0,rtol=0)


def test_strict_json_cannot_hide_nonfinite(tmp_path):
    with pytest.raises(ValueError):save_json(tmp_path/'bad.json',{'effective_logit':float('nan')})
    save_json(tmp_path/'valid.json',{'undefined_metric':None,'complete':False})
    assert 'null' in (tmp_path/'valid.json').read_text()
