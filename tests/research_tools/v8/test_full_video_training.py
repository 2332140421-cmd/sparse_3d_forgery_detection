import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research_tools.v8.component_global_context_pilot.model import GlobalContextModel
from research_tools.v8.full_video_training.model import VideoTraining
from research_tools.v8.full_video_global_context.validation import sequence
from research_tools.v8.full_video_streaming.model import frame_at

torch.set_num_threads(2)
STATS={'absolute_mean':[1.,2.,3.],'absolute_scale':[2.,3.,4.],'relative_scale':[1.,1.,1.]}


def sample():
    torch.manual_seed(4)
    content=torch.zeros(6,256,16,16,dtype=torch.bool);content[:,[0,1,200]]=True
    valid=content.clone();valid[:,200]=False
    ids=torch.arange(256)[None].expand(6,256).clone();ids[:,1]=0
    return {'x':torch.randn(6,256,77),'mu':torch.randn(6,256,3),'valid':valid,'content':content,
            'area':content.sum((-1,-2)).float(),'component_ids':ids,
            'predecessor':torch.arange(256)[None,:,None].expand(6,256,4).clone(),
            'association_weight':torch.ones(6,256,4)/4,'history_available':torch.ones(6,256,dtype=torch.bool),
            'delta_t':torch.tensor([0.,.1,.12,.13,.11,.12])}


def test_eval_matches_both_streaming_conditions_and_padding():
    data=sample()
    for c in ('GLOBAL_COMPONENT','GLOBAL_CELL'):
        model=GlobalContextModel(c,STATS).eval();interface=VideoTraining(model)
        b={k:v[None] for k,v in data.items()}
        with torch.no_grad():
            old=sequence(model,[frame_at(b,i) for i in range(6)])
            for chunk in (2,4,6):
                z,cells=interface.video(data,chunk,False,True)
                torch.testing.assert_close(z,old[2],atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(cells,old[0][:,0],atol=2e-5,rtol=2e-5)
            changed=copy.deepcopy(data);padding=changed['area']==0;changed['x'][padding]=1e5;changed['mu'][padding]=1e5
            torch.testing.assert_close(interface.video(data,2,False),interface.video(changed,2,False),atol=0,rtol=0)
            assert torch.isfinite(cells[:,200]).all()


def test_checkpoint_loss_and_all_parameter_gradients():
    a=GlobalContextModel('GLOBAL_COMPONENT',STATS).train();b=copy.deepcopy(a);data=sample()
    la=torch.nn.functional.binary_cross_entropy_with_logits(VideoTraining(a).video(data,2,False),torch.tensor(1.,dtype=torch.float64))
    lb=torch.nn.functional.binary_cross_entropy_with_logits(VideoTraining(b).video(data,2,True),torch.tensor(1.,dtype=torch.float64))
    la.backward();lb.backward();torch.testing.assert_close(la,lb,atol=1e-10,rtol=1e-10)
    for pa,pb in zip(a.parameters(),b.parameters()):
        assert pa.grad is not None and torch.isfinite(pa.grad).all()
        torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-4)
    for module in (b.patch_encoder,b.region_projection,b.region_attention,b.component_residual,b.gru,b.cell_head):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())


def test_hidden_gradient_crosses_boundary_and_direct_lse():
    model=GlobalContextModel('GLOBAL_COMPONENT',STATS);run=VideoTraining(model);data=sample()
    left={k:v[:3].clone() for k,v in data.items()};right={k:v[3:].clone() for k,v in data.items()}
    left['x'].requires_grad_();h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    h,active,_,_,_=run.chunk(left,h,active,True)
    _,_,later,_,_=run.chunk(right,h,active,False)
    later.backward();assert left['x'].grad is not None and left['x'].grad.abs().sum()>0
    with torch.no_grad():
        z,cells=run.video(data,2,False,True);area=data['area'].double()
        direct=torch.logsumexp(cells.double()+torch.where(area>0,area.clamp_min(1).log(),-torch.inf),(0,1))-area.sum().log()
        torch.testing.assert_close(z,direct,atol=1e-12,rtol=1e-12)


def test_optimizer_rng_order_save_restore(tmp_path):
    from research_tools.v8.local_geometry_pilot.runner import rng_state,restore_rng,seed_all
    seed_all(20260909);m=GlobalContextModel('GLOBAL_CELL',STATS);opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    data=sample();VideoTraining(m).video(data,2,True).backward();opt.step()
    gen=torch.Generator().manual_seed(20260909);torch.randperm(122,generator=gen)
    path=tmp_path/'resume.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'order':gen.get_state()},path)
    expected=torch.randperm(122,generator=gen);expected_rng=torch.rand(4)
    cp=torch.load(path,weights_only=False);other=GlobalContextModel('GLOBAL_CELL',STATS);other.load_state_dict(cp['model'],strict=True)
    oo=torch.optim.AdamW(other.parameters(),lr=1e-4);oo.load_state_dict(cp['optimizer']);restore_rng(cp['rng']);gen.set_state(cp['order'])
    assert torch.equal(torch.rand(4),expected_rng) and torch.equal(torch.randperm(122,generator=gen),expected)
    for x,y in zip(m.parameters(),other.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert len(oo.state)==len(opt.state)


def test_actual_pair_checkpoint_resume_does_not_repeat_committed_pair(tmp_path,monkeypatch):
    import research_tools.v8.full_video_training.run as runner
    monkeypatch.setattr(runner,'CONDITIONS',('GLOBAL_COMPONENT',));monkeypatch.setattr(runner,'SEEDS',(20260909,))
    def model(*args):
        m=torch.nn.Linear(1,1);m.weight.data.fill_(.1);m.bias.data.zero_()
        return m,{'checkpoint_sha256':'frozen_test_initialization'}
    monkeypatch.setattr(runner,'load_model',model)
    calls=[]
    def step(m,row,*args):
        calls.append(row['video_id'])
        loss=(m(torch.tensor([[float(row['value'])]]))-row['video_label']).square().sum()*.5
        loss.backward();return {'loss':float(loss.detach())}
    monkeypatch.setattr(runner,'step_video',step)
    rows=[{'source_id':s,'split':'train','role':role,'video_id':s+role,'value':i+1,'video_label':int(role=='fake')}
          for i,s in enumerate(('a','b')) for role in ('real','fake')]
    resumed=tmp_path/'resumed';resumed.mkdir();fresh=tmp_path/'fresh';fresh.mkdir()
    monkeypatch.setattr(runner,'STOP',True)
    import pytest
    with pytest.raises(InterruptedError,match='SAFE_TERM_AFTER_SOURCE_PAIR'):
        runner.train(resumed,tmp_path,rows,'cache',2,'protocol',lambda *a,**k:None,False)
    assert len(calls)==2
    cp=torch.load(next((resumed/'models').glob('*.recovery.pt')),weights_only=False)
    assert cp['epoch']==0 and cp['pair_index']==1 and cp['optimizer']['state']
    monkeypatch.setattr(runner,'STOP',False)
    rr=runner.train(resumed,tmp_path,rows,'cache',2,'protocol',lambda *a,**k:None,True)
    assert len(calls)==80 and rr[0]['epoch']==20 and len(rr[0]['loss_history'])==20
    ff=runner.train(fresh,tmp_path,rows,'cache',2,'protocol',lambda *a,**k:None,False)
    a=torch.load(rr[0]['checkpoint'],weights_only=False);b=torch.load(ff[0]['checkpoint'],weights_only=False)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    assert a['loss_history']==b['loss_history']
