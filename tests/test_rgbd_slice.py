"""Critical contracts for matched RGB-D pilot; no training-effect assertions."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from research_tools.rgbd_slice.data import encoder_depth,geometry_shape
from research_tools.rgbd_slice.model import SlabPool,native_sample,RGBDSlices,backbone


def test_depth_representations_and_mapping():
    z=np.array([[.1,.125],[.25,1]],np.float32);valid=np.array([[True,True],[True,False]])
    ids=np.floor(z/.125).astype(int);display=encoder_depth(z,valid)
    assert ids.tolist()==[[0,1],[2,8]] and display[0,0]<display[1,0]
    assert np.isfinite(display).all()
    assert geometry_shape((720,1280))[0]==(432,768)
    assert geometry_shape((360,640))[0]==(360,640)


def test_all_native_pool_against_dense_reference():
    torch.set_num_threads(2);torch.manual_seed(3);m=SlabPool().eval()
    local=torch.randn(256,4,4,requires_grad=True);q=torch.randn(128,65,3,requires_grad=True)
    with torch.no_grad():q[...,2]=torch.where(q[...,2]>0,.1,.3)
    valid=torch.ones(128,65,dtype=torch.bool);valid[0,0]=False
    got,ids=m(local,q,valid);selected=torch.where(valid.reshape(-1))[0]
    sampled=native_sample(local,selected,valid.shape);scores=m.score(torch.cat((sampled,q.reshape(-1,3)[selected]),-1));values=m.value(sampled).reshape(-1,4,128)
    absolute=torch.floor(q[...,2][valid]/.125).long();expected=[]
    for sid in ids:
        mask=absolute==sid;a=torch.softmax(scores[mask],0);expected.append((a[...,None]*values[mask]).sum(0).reshape(512))
    expected=torch.stack(expected);torch.testing.assert_close(got,expected,atol=2e-5,rtol=3e-5)
    params=[local,q,*m.parameters()];a=torch.autograd.grad(got.square().mean(),params,retain_graph=True);b=torch.autograd.grad(expected.square().mean(),params)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=2e-5,rtol=3e-5)


def config():return json.loads(Path('docs/pvst/rgbd_slice_config_v1.json').read_text())


def test_official_classification_tail_does_not_change_multiscale_rgb():
    torch.set_num_threads(2);c=config();m=backbone(c).eval();ref=copy.deepcopy(m)
    spec=importlib.util.spec_from_file_location('reference_dformer',Path(c['dformer_code'])/'models/encoders/DFormer.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ref.stages[3][-1]=mod.Block(index=9,dim=512,num_head=8,mlp_ratio=4,window=7,block_index=1,last_block_index=50,drop_depth=False,dropout_layer={'type':'DropPath','drop_prob':.1})
    state=torch.load(c['dformer_weights'],map_location='cpu',weights_only=False)['state_dict']
    ref.load_state_dict({k:v for k,v in state.items() if not k.startswith('pred.')},strict=True);ref.eval()
    rgb=torch.randn(1,3,32,64);depth=torch.randn(1,1,32,64)
    with torch.no_grad():a,_=m(rgb,depth);b,_=ref(rgb,depth)
    for i,(x,y) in enumerate(zip(a,b)):
        assert x.shape[1]==[64,128,256,512][i]
        torch.testing.assert_close(x,y,atol=2e-5,rtol=3e-5)


def test_zero_residual_and_fixed_bn_affine():
    torch.set_num_threads(2);torch.manual_seed(4);a=RGBDSlices(backbone(config()),False);b=RGBDSlices(copy.deepcopy(a.local.encoder),True);b.local.load_state_dict(a.local.state_dict(),strict=True)
    a.eval();b.eval();fs=[]
    for i in range(2):
        fs.append({'rgb':torch.randint(0,255,(32,48,3),dtype=torch.uint8),'depth':torch.randn(32,48),'xyz':torch.ones(32,48,3)*.1,'valid':torch.ones(32,48,dtype=torch.bool),'gt':torch.zeros(32,48,dtype=torch.bool),'pts':float(i)*.07})
    with torch.no_grad():pa=a(fs);pb=b(fs)
    for x,y in zip(pa,pb):
        for k in ['pixel','frame']:torch.testing.assert_close(x[k],y[k],atol=2e-5,rtol=3e-5)
    b.train()
    for m in b.local.encoder.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):assert not m.training and m.weight.requires_grad


def test_trainable_encoder_and_residual_two_step_path():
    torch.set_num_threads(2);m=RGBDSlices(backbone(config()),True).train()
    fs=[]
    for i in range(2):
        q=torch.ones(32,48,3)*.1;q[:16,:,2]=.3
        gt=torch.zeros(32,48,dtype=torch.bool);gt[8:20,10:25]=True
        fs.append({'rgb':torch.randint(0,255,(32,48,3),dtype=torch.uint8),'depth':torch.randn(32,48),'xyz':q,'valid':torch.ones(32,48,dtype=torch.bool),'gt':gt,'pts':i*.07})
    from research_tools.depth_slice.model import terms
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    bn={k:v.clone() for k,v in m.local.encoder.named_buffers()}
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            p=m(fs);loss=torch.stack([terms(x,f['gt']) for x,f in zip(p,fs)]).mean().sum()
        loss.backward();assert torch.isfinite(loss)
        assert m.local.encoder.downsample_layers_e[0][0].weight.grad is not None
        if step==1:
            assert m.pool.value.weight.grad.abs().sum()>0
            assert m.time[0].q.weight.grad.abs().sum()>0
        opt.step()
    for k,v in m.local.encoder.named_buffers():assert torch.equal(v,bn[k])


def test_cuda_cpu_staged_decode_autograd():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA transfer regression needs CUDA')
    from research_tools.rgbd_slice.model import Local
    from torch.utils.checkpoint import checkpoint
    m=Local(torch.nn.Identity()).cuda().train();local=torch.randn(256,8,12,requires_grad=True)
    rgb=torch.randint(0,255,(32,48,3),dtype=torch.uint8);ctx=torch.zeros(1,32,32,48,requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):z=checkpoint(m.decode,local,rgb,ctx,use_reentrant=False)
    z.float().square().mean().backward()
    assert z.is_cuda and local.grad is not None and torch.isfinite(local.grad).all()
    assert ctx.grad is not None and ctx.grad.device.type=='cpu' and torch.isfinite(ctx.grad).all()
