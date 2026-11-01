"""Direct scientific contracts; tiny images keep FULL fixed model widths/layers."""
import copy
import torch
import numpy as np
from torch import nn
from research_tools.pvst.geometry import voxelize,scatter_mean,scatter_max,local_tiled
from research_tools.pvst.temporal import neighbors,Temporal
from research_tools.pvst.model import PVST,LocalResidual

CFG={'point_chunk':256,'tile_size':4,'attention_chunk':32}

class Up(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.op=nn.Conv2d(a,b,1)
    def forward(self,x):return self.op(x)

class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv2d(1024,512,1);self.proj_norm=nn.BatchNorm2d(512);self.proj_relu=nn.ReLU()
        self.up1=Up(512,256);self.up2=Up(256,128);self.up3=Up(128,64)
        self.final_conv_block=Up(64,32);self.dropout=nn.Dropout(.1);self.predictor=nn.Conv2d(32,1,1)

def frames(n=3,h=5,w=6):
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    uv=torch.stack((2*(xx+.5)/w-1,2*(yy+.5)/h-1,torch.zeros_like(xx)),-1)
    return [{'rgb':torch.randint(0,256,(h,w,3),dtype=torch.uint8),'tokens':torch.randn(1024,2,2),
         'q':uv.clone(),'valid':torch.ones(h,w,dtype=torch.bool),'uv':uv[...,:2],
         'pts':i*.07,'mapping448':{'pad_xy':[0,0],'content_hw':[448,448]}} for i in range(n)]


def test_whole_voxel_distribution_and_permutation():
    q=torch.tensor([[[.001,0,1],[.002,0,1],[.003,0,1],[.004,0,1]]])
    q2=q.clone();q2[0,:,0]=torch.tensor([0.,0.,.005,.005])
    a=voxelize(q,torch.ones(1,4,dtype=torch.bool),1/64);b=voxelize(q2,torch.ones(1,4,dtype=torch.bool),1/64)
    torch.testing.assert_close(a['input'][:,:3].mean(0),b['input'][:,:3].mean(0))
    assert not torch.equal(a['input'],b['input'])
    encoder=nn.Sequential(nn.Linear(9,64),nn.GELU(),nn.Linear(64,128));x=encoder(a['input']);y=encoder(b['input'])
    assert not torch.allclose(scatter_max(x,a['index'],1),scatter_max(y,b['index'],1),atol=1e-8,rtol=1e-8)
    perm=torch.tensor([3,0,2,1]);torch.testing.assert_close(scatter_mean(x[perm],a['index'][perm]),scatter_mean(x,a['index']))
    torch.testing.assert_close(scatter_max(x[perm],a['index'][perm],1),scatter_max(x,a['index'],1))
    valid=torch.tensor([[1,1,0,1]],dtype=torch.bool);c=voxelize(q,valid,1/64);assert len(c['ids'])==3


def test_exact_spatial_tile_outputs_and_gradients():
    torch.manual_seed(2);m=LocalResidual();x=torch.randn(9,11,128,requires_grad=True);v=torch.ones(9,11,dtype=torch.bool);v[3,3]=False
    a=local_tiled(x,v,m,4);b=local_tiled(x,v,m,99);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    ga=torch.autograd.grad(a.square().sum(),x,retain_graph=True)[0];gb=torch.autograd.grad(b.square().sum(),x)[0]
    torch.testing.assert_close(ga,gb,atol=1e-5,rtol=1e-4);assert not a[3,3].any()


def test_81_neighbors_causality_pts_and_all_candidates():
    c=torch.cartesian_prod(torch.arange(-1,2),torch.arange(-1,2),torch.arange(-1,2))
    idx,rel=neighbors([c,c,c],1,[0.,.03,.11]);center=((c==0).all(-1)).nonzero()[0,0]
    assert int((idx[54+center]>=0).sum())==81
    assert rel[54+center,27,3].item()==np.float32(-.08)
    m=Temporal(256,4,32).eval();a=[torch.randn(27,256) for _ in range(3)]
    with torch.no_grad():
        base=m(a,[c,c,c],[0.,.03,.11]);other=m([a[0],a[1],a[2]+100],[c,c,c],[0.,.03,.11])
        torch.testing.assert_close(base[0],other[0]);torch.testing.assert_close(base[1],other[1])
        dt=m(a,[c,c,c],[0.,.08,.11]);assert not torch.allclose(base[1],dt[1])


def test_model_matched_init_zero_residual_and_save(tmp_path):
    torch.manual_seed(1);a=PVST('UV_PVST',FakeDecoder(),CFG)
    torch.manual_seed(1);b=PVST('XYZ_PVST',FakeDecoder(),CFG)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)
    a.eval();fs=frames()
    with torch.no_grad():result=a(fs)
    assert len(result)==3 and result[0]['pixel_logits'].shape==(5,6)
    torch.testing.assert_close(result[0]['pixel_logits'],result[0]['visual_pixel_logits'])
    torch.save(a.state_dict(),tmp_path/'m.pt');b.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True));b.condition='UV_PVST';b.h=1/32
    with torch.no_grad():other=b(fs)
    torch.testing.assert_close(result[0]['frame_logit'],other[0]['frame_logit'])


def test_time_halo_and_point_gradient_after_two_updates():
    torch.manual_seed(4);m=PVST('UV_PVST',FakeDecoder(),CFG);fs=frames(18,h=2,w=2)
    m.eval()
    nn.init.normal_(m.pixel_out.weight,std=.01);nn.init.normal_(m.frame_residual[-1].weight,std=.01)
    with torch.no_grad():
        full=m(fs,target_start=17)[0];halo=m(fs[1:],target_start=16)[0]
    torch.testing.assert_close(full['frame_logit'],halo['frame_logit'],atol=2e-5,rtol=1e-5)
    torch.testing.assert_close(full['pixel_logits'],halo['pixel_logits'],atol=2e-5,rtol=1e-5)
    m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    for step in range(2):
        opt.zero_grad();r=m(fs[:3],target_start=2)[0]
        loss=r['pixel_logits'].square().mean()+r['frame_logit'].square();loss.backward();opt.step()
    assert m.geo[0].weight.grad.abs().sum()>0
    assert m.vfe[0].weight.grad.abs().sum()>0
    assert m.fine_time.layers[0].qkv.weight.grad.abs().sum()>0
    assert m.coarse_time.layers[0].qkv.weight.grad.abs().sum()>0
    m.check_bn()


def test_full_model_tile_parity_and_gradients():
    torch.manual_seed(12);cfg={**CFG,'tile_size':3};m=PVST('UV_PVST',FakeDecoder(),cfg).eval()
    nn.init.normal_(m.pixel_out.weight,std=.02);nn.init.normal_(m.frame_residual[-1].weight,std=.02)
    fs=frames(3,h=7,w=9);a=m(fs,2)[0];loss=a['pixel_logits'].square().mean()+a['frame_logit'].square()
    params=(m.geo[0].weight,m.pixel_out.weight);ga=torch.autograd.grad(loss,params)
    cfg['tile_size']=99;b=m(fs,2)[0];other=b['pixel_logits'].square().mean()+b['frame_logit'].square();gb=torch.autograd.grad(other,params)
    torch.testing.assert_close(a['pixel_logits'],b['pixel_logits'],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(a['frame_logit'],b['frame_logit'],atol=2e-5,rtol=2e-5)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-5,rtol=5e-4)


def test_uv_loader_never_opens_geometry(tmp_path,monkeypatch):
    import numpy as np
    import research_tools.pvst.data as data
    r={'video_id':'uv','source_hw':[4,5],'dataset':'ActivityForensics','role':'real','pts':[{'pts_s':0.}],'actual_frames':1,'video_sha256':'uvsha'}
    d=tmp_path/'cache/uv';d.mkdir(parents=True)
    np.save(d/'rgb.npy',np.zeros((1,4,5,3),np.uint8));np.save(d/'tokens.npy',np.zeros((1,1024,32,32),np.float16))
    import json
    (d/'complete.json').write_text(json.dumps({'video_sha256':'uvsha'}))
    def forbidden(*a,**kw):raise AssertionError('UV_READ_XYZ')
    monkeypatch.setattr(data,'native_dense',forbidden)
    v=data.Video(tmp_path,r,'UV_PVST',999);a=v.frames(0,1,'cpu')[0]
    assert a['valid'].all() and (a['q'][...,2]==0).all();v.close()


def test_loss_class_and_tail_weights():
    from research_tools.pvst.train import frame_weights
    w=frame_weights([0,0,0,1,1]);assert np.isclose(w[:3].sum(),.5) and np.isclose(w[3:].sum(),.5)
    assert np.isclose(frame_weights([0]*7).sum(),1)
    # Unequal tail chunks must add weighted frame losses, not average chunk means.
    a=np.arange(7,dtype=float);w=frame_weights([0]*5+[1]*2)
    assert np.isclose((a[:4]*w[:4]).sum()+(a[4:]*w[4:]).sum(),(a*w).sum())


def test_attention_query_chunk_is_equivalent():
    from research_tools.pvst.temporal import Temporal
    torch.manual_seed(31);m=Temporal(256,4,16).eval();coords=[torch.zeros(13,3,dtype=torch.long)]
    # Unique occupied voxels; no duplicate artificial targets.
    coords[0][:,0]=torch.arange(13);features=[torch.randn(13,256,requires_grad=True)]
    a=m(features,coords,[0.])[0];ga=torch.autograd.grad(a.square().sum(),features[0])[0]
    for layer in m.layers:layer.chunk=512
    b=m(features,coords,[0.])[0];gb=torch.autograd.grad(b.square().sum(),features[0])[0]
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5);torch.testing.assert_close(ga,gb,atol=1e-5,rtol=1e-5)


def test_actual_block_distribution_and_all_invalid_fallback():
    torch.manual_seed(71);m=PVST('XYZ_PVST',FakeDecoder(),CFG).eval();f=frames(1,h=1,w=4)[0]
    f['q']=torch.tensor([[[.001,0.,1.],[.002,0.,1.],[.003,0.,1.],[.004,0.,1.]]]);other={**f,'q':f['q'].clone()};other['q'][0,:,0]=torch.tensor([0.,0.,.005,.005])
    with torch.no_grad():
        dec=[m.decode(f['tokens'])];a=m.frame_blocks(f,voxelize(f['q'],f['valid'],m.h),dec,True)
        b=m.frame_blocks(other,voxelize(other['q'],other['valid'],m.h),dec,True)
        assert not torch.allclose(a[0],b[0],atol=1e-6,rtol=1e-6)
        invalid={**f,'q':torch.full_like(f['q'],float('nan')),'valid':torch.zeros_like(f['valid'])}
        meta=voxelize(invalid['q'],invalid['valid'],m.h);p=m.frame_blocks(invalid,meta,dec,True)
        assert not len(meta['coords']) and not p[4].any()
        result=m([invalid])[0];assert result['pixel_logits'].shape==(1,4) and torch.isfinite(result['frame_logit'])


def test_resource_gate_uses_real_json_wire_format(tmp_path):
    from research_tools.pvst.data import save
    from research_tools.pvst.train import resource_passed
    from scripts.prepare_v8_assets import digest
    cfg={'tile_size':256,'attention_chunk':512,'point_chunk':32768,'target_frames':8,'attention_backend':'torch_sdpa'};save(tmp_path/'scale.json',{'s':2.})
    for condition in ['UV_PVST','XYZ_PVST']:
        r={'condition':condition,'status':'PASS','spatial_GT':True,'static_routing_cache':True,'targets':8,'execution':{k:cfg[k] for k in ['tile_size','attention_chunk','point_chunk','attention_backend']},'resource_scale_sha256':digest(tmp_path/'scale.json')}
        save(tmp_path/f'resource_{condition}_test.json',r);assert resource_passed(tmp_path,condition,cfg)
    save(tmp_path/'scale.json',{'s':3.});assert not resource_passed(tmp_path,'XYZ_PVST',cfg)
    assert resource_passed(tmp_path,'UV_PVST',cfg)


def test_only_static_routing_reuse_has_same_outputs_and_gradients():
    torch.manual_seed(103);m=PVST('UV_PVST',FakeDecoder(),CFG).eval()
    nn.init.normal_(m.pixel_out.weight,std=.01);nn.init.normal_(m.frame_residual[-1].weight,std=.01);fs=frames()
    a=m(fs,2)[0];params=(m.geo[0].weight,m.fine_time.layers[0].qkv.weight)
    ga=torch.autograd.grad(a['pixel_logits'].square().mean()+a['frame_logit'].square(),params)
    for f in fs:f['voxel_meta']=voxelize(f['q'],f['valid'],m.h)
    b=m(fs,2)[0];gb=torch.autograd.grad(b['pixel_logits'].square().mean()+b['frame_logit'].square(),params)
    torch.testing.assert_close(a['pixel_logits'],b['pixel_logits']);torch.testing.assert_close(a['frame_logit'],b['frame_logit'])
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)


def test_checkpointed_full_path_matches_gradients_without_dropout():
    torch.manual_seed(209);m=PVST('UV_PVST',FakeDecoder(),CFG).eval();fs=frames()
    nn.init.normal_(m.pixel_out.weight,std=.02);nn.init.normal_(m.frame_residual[-1].weight,std=.02)
    params=(m.geo[0].weight,m.fine_time.layers[0].qkv.weight,m.decoder.proj.weight,m.pixel_out.weight)
    a=m(fs,2)[0];ga=torch.autograd.grad(a['pixel_logits'].square().mean()+a['frame_logit'].square(),params)
    m.train()
    for layer in m.modules():
        if isinstance(layer,(nn.Dropout,nn.MultiheadAttention)):layer.eval()
    b=m(fs,2)[0];gb=torch.autograd.grad(b['pixel_logits'].square().mean()+b['frame_logit'].square(),params)
    torch.testing.assert_close(a['pixel_logits'],b['pixel_logits']);torch.testing.assert_close(a['frame_logit'],b['frame_logit'])
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-4)


def test_sdpa_matches_standard_relative_bias_attention_and_gradients():
    from research_tools.pvst.temporal import SparseLayer
    torch.manual_seed(619);m=SparseLayer(256,4).eval();q=torch.randn(7,4,64,requires_grad=True);k=torch.randn(13,4,64,requires_grad=True);v=torch.randn(13,4,64,requires_grad=True)
    idx=torch.randint(0,13,(7,81));idx[:,70:]=-1;rel=torch.randn(7,81,4)
    actual=m.attend(q,k,v,idx,rel);score=(q[:,None]*k[idx.clamp_min(0)]).sum(-1)/8+m.bias(rel)
    score=score.masked_fill((idx<0)[...,None],-torch.inf);expected=(score.softmax(1)[...,None]*v[idx.clamp_min(0)]).sum(1).reshape(7,256)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    params=(q,k,v,m.bias[0].weight);ga=torch.autograd.grad(actual.square().sum(),params,retain_graph=True);gb=torch.autograd.grad(expected.square().sum(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
