"""Only post-relation token norm, shared initialization and scalar observer."""
import json
from pathlib import Path
import torch
from research_tools.rgbd_slice.model import RGBDSlices,backbone
from research_tools.rgbd_slice.experiment import ContextScaleRecorder


def config():return json.loads(Path('docs/pvst/rgbd_slice_config_v1.json').read_text())


def construct(norm):
    torch.manual_seed(20260909)
    return RGBDSlices(backbone(config()),True,context_norm=norm)


def test_common_initialization_rng_and_legacy_strict_load():
    torch.set_num_threads(2);old=construct(False);old_rng=torch.get_rng_state().clone();new=construct(True)
    assert torch.equal(old_rng,torch.get_rng_state())
    assert new.context_norm.eps==1e-5 and new.context_norm.normalized_shape==(512,)
    a=old.state_dict();b=new.state_dict()
    assert set(b)-set(a)=={'context_norm.weight','context_norm.bias'}
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==1024
    for k in a:assert torch.equal(a[k],b[k]),k
    assert torch.equal(b['context_norm.weight'],torch.ones(512)) and torch.equal(b['context_norm.bias'],torch.zeros(512))
    saved=torch.load(Path(config()['output'])/'RGBD_SLICE_RELATION/best.pt',map_location='cpu',mmap=True,weights_only=False)
    old.load_state_dict(saved['model'],strict=True)


def test_per_token_before_mean_shared_by_both_heads_and_empty():
    torch.set_num_threads(2);model=construct(True).eval();fs=[]
    for i in range(3):
        xyz=torch.ones(32,48,3)*.1;xyz[:16,:,2]=.3
        valid=torch.ones(32,48,dtype=torch.bool)
        if i==1:valid[:]=False
        fs.append({'rgb':torch.randint(0,255,(32,48,3),dtype=torch.uint8),'depth':torch.randn(32,48),'xyz':xyz,'valid':valid,'gt':torch.zeros(32,48,dtype=torch.bool),'pts':i*.07})
    inputs=[];norm=[];pixels=[];frames=[]
    model.context_norm.register_forward_hook(lambda m,a,z:(inputs.append(a[0].detach().clone()),norm.append(z.detach().clone())) and None)
    model.pixel_context.register_forward_pre_hook(lambda m,a:pixels.append(a[0].detach().clone()))
    model.frame_context.register_forward_pre_hook(lambda m,a:frames.append(a[0].detach().clone()))
    with torch.no_grad():pred=model(fs)
    assert len(inputs)==len(norm)==len(pixels)==3 and len(frames)==2
    for raw,n,p in zip(inputs,norm,pixels):
        torch.testing.assert_close(n,torch.nn.functional.layer_norm(raw,(512,),eps=1e-5))
        torch.testing.assert_close(p,n)
    for actual,expected in zip(frames,[n.mean(0) for n in norm if len(n)]):torch.testing.assert_close(actual,expected)
    assert pred[1]['slices']==0 and all(torch.isfinite(p['pixel']).all() for p in pred)
    x=torch.randn(2,3,512)
    with torch.no_grad():
        y=model.context_norm(x);z=model.context_norm(x[[1,0]])
    torch.testing.assert_close(z,y[[1,0]])
    assert not torch.allclose(y.mean(1),model.context_norm(x.mean(1)))


def test_observer_uses_only_scalars_and_does_not_consume_rng():
    class Heads(torch.nn.Module):
        def __init__(self):
            super().__init__();self.context_norm=torch.nn.LayerNorm(512);self.pixel_context=torch.nn.Linear(512,32);self.frame_context=torch.nn.Linear(512,1)
    m=Heads();x=torch.randn(4,512,requires_grad=True);observer=ContextScaleRecorder(m);observer.begin(1);rng=torch.get_rng_state().clone()
    y=m.context_norm(x);p=m.pixel_context(y);f=m.frame_context(y.mean(0));stats=observer.finish();observer.close()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(v is None or isinstance(v,float) for v in stats.values()) and not observer.rows
    assert abs(stats['raw_token_rms']-float(x.detach().square().mean().sqrt()))<1e-6
    assert abs(stats['normalized_token_rms']-float(y.detach().square().mean().sqrt()))<1e-6
    (p.square().mean()+f.square().mean()).backward();assert x.grad is not None and torch.isfinite(x.grad).all()
