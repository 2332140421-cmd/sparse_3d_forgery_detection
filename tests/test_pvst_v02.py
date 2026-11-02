"""Direct scientific contracts; tiny images keep FULL fixed model widths/layers."""
import copy
import torch
import numpy as np
from torch import nn
from research_tools.pvst.geometry import voxelize,scatter_mean,scatter_max,local_tiled
from research_tools.pvst.temporal import Neighborhood,Temporal
from research_tools.pvst.model import PVST,LocalResidual

# Frozen pre-compact mathematical reference, tests only; never a runtime route.
import itertools
def reference_neighbors(coords,dilation,pts,chunk=2048):
    """Exact injective integer keys and searchsorted; no truncation or empty nodes."""
    device=coords[0].device;lengths=[len(c) for c in coords];offset=[0]
    for n in lengths:offset.append(offset[-1]+n)
    if not offset[-1]:return torch.empty(0,81,dtype=torch.long,device=device),torch.empty(0,81,4,device=device)
    allc=torch.cat(coords);lo=allc.amin(0)-1;span=allc.amax(0)-lo+2
    volume=int(span.prod())
    if volume*len(coords)>=2**62:raise ValueError('EXACT_NEIGHBOR_KEY_OVERFLOW')
    def key(c,t):
        a=c-lo;return ((a[...,0]*span[1]+a[...,1])*span[2]+a[...,2])+t*volume
    times=torch.repeat_interleave(torch.arange(len(coords),device=device),torch.tensor(lengths,device=device))
    keys=key(allc,times);sorted_keys,order=keys.sort();offsets=torch.tensor(list(itertools.product([-1,0,1],repeat=3)),device=device)
    offsets=offsets.repeat(3,1);lag=torch.tensor([0]*27+[dilation]*27+[2*dilation]*27,device=device)
    indices=[];relative=[]
    for ti,c in enumerate(coords):
        for start in range(0,len(c),chunk):
            base=c[start:start+chunk];candidate=base[:,None]+offsets[None];tt=ti-lag[None].expand(len(base),-1)
            wanted=key(candidate,tt);pos=torch.searchsorted(sorted_keys,wanted);safe=pos.clamp(max=len(sorted_keys)-1)
            inside=((candidate>=lo)&(candidate<lo+span)).all(-1)
            valid=(tt>=0)&inside&(pos<len(sorted_keys))&(sorted_keys[safe]==wanted)
            idx=torch.where(valid,order[safe],-1)
            dt=torch.tensor([pts[max(0,ti-int(v))]-pts[ti] for v in lag.cpu().tolist()],device=device,dtype=torch.float32)
            rel=torch.cat((offsets.float()[None].expand(len(base),-1,-1),dt[None,:,None].expand(len(base),-1,1)),-1)
            indices.append(idx);relative.append(rel)
    return torch.cat(indices),torch.cat(relative)


def neighbors(coords,dilation,pts):
    graph=Neighborhood(coords,pts);index,frame=graph.query(0,len(graph.coords),dilation)
    return index,graph.relative[dilation][frame]

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


def test_resource_gate_rejects_old_execution_and_config_changes(tmp_path):
    from research_tools.pvst.data import save
    from research_tools.pvst.train import resource_passed
    from research_tools.pvst.probe import execution_signature
    from scripts.prepare_v8_assets import digest
    cfg={'tile_size':256,'attention_chunk':512,'point_chunk':32768,'target_frames':8,'attention_backend':'torch_sdpa','ffn_chunk':8192,'execution_revision':'ffn_chunks_v1'}
    save(tmp_path/'scale.json',{'s':2.});save(tmp_path/'manifest.json',{'rows':[]});save(tmp_path/'resource_windows.json',{'representatives':[{'video_id':'v','target_range':[16,24]}]})
    for condition in ['UV_PVST','XYZ_PVST']:
        path=tmp_path/f'resource_{condition}_ffn_chunks_v1.json'
        save(path,{'status':'PASS','condition':condition,'targets':8});assert not resource_passed(tmp_path,condition,cfg)
        r={'condition':condition,'status':'PASS',**execution_signature(cfg),'manifest_sha256':digest(tmp_path/'manifest.json'),
            'resource_scale_sha256':digest(tmp_path/'scale.json'),'windows_sha256':digest(tmp_path/'resource_windows.json'),'records':[{'video_id':'v','target_range':[16,24]}]}
        save(path,r);assert resource_passed(tmp_path,condition,cfg)
        assert not resource_passed(tmp_path,condition,{**cfg,'ffn_chunk':4096})
        r['implementation_sha256']['temporal.py']='old';save(path,r);assert not resource_passed(tmp_path,condition,cfg)


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
    actual=m.attend(q,k,v,idx,m.bias(rel));score=(q[:,None]*k[idx.clamp_min(0)]).sum(-1)/8+m.bias(rel)
    score=score.masked_fill((idx<0)[...,None],-torch.inf);expected=(score.softmax(1)[...,None]*v[idx.clamp_min(0)]).sum(1).reshape(7,256)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    params=(q,k,v,m.bias[0].weight);ga=torch.autograd.grad(actual.square().sum(),params,retain_graph=True);gb=torch.autograd.grad(expected.square().sum(),params)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)


@__import__('pytest').mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_ffn_chunks_match_whole_outputs_input_and_parameter_gradients(dtype):
    from contextlib import nullcontext
    from research_tools.pvst.temporal import SparseLayer
    torch.manual_seed(1001);a=SparseLayer(256,4,ffn_chunk=99).eval();b=copy.deepcopy(a);b.ffn_chunk=4
    xa=torch.randn(11,256,dtype=dtype,requires_grad=True);xb=xa.detach().clone().requires_grad_()
    ctx=lambda:torch.autocast('cpu',dtype=dtype) if dtype==torch.bfloat16 else nullcontext()
    graph=Neighborhood([torch.stack((torch.arange(11),torch.zeros(11,dtype=torch.long),torch.zeros(11,dtype=torch.long)),-1)],[0.])
    with ctx():ya=a(xa,graph,1);yb=b(xb,graph,1)
    ya.float().square().mean().backward();yb.float().square().mean().backward()
    pairs=[(ya,yb),(xa.grad,xb.grad),*[(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()) if p.grad is not None]]
    errors=[float((x-y).abs().max()) for x,y in pairs];print('FFN_PARITY',str(dtype),'output/input/params max_abs',errors)
    atol,rtol=(2e-6,2e-4) if dtype==torch.float32 else (2e-3,3e-2)
    for x,y in pairs:torch.testing.assert_close(x,y,atol=atol,rtol=rtol)


def test_nested_ffn_checkpoint_preserves_dropout_rng_and_gradients():
    from research_tools.pvst.temporal import Temporal
    torch.manual_seed(114);a=Temporal(256,4,chunk=8,ffn_chunk=4).train();b=copy.deepcopy(a)
    for m in b.layers:m.ffn_checkpoint=False
    ca=[torch.tensor([[i,0,0] for i in range(5)])]*2
    xa=[torch.randn(5,256,requires_grad=True) for _ in range(2)];xb=[x.detach().clone().requires_grad_() for x in xa]
    rng=torch.get_rng_state();ya=torch.cat(a(xa,ca,[0.,.04]));ya.square().mean().backward();after_a=torch.get_rng_state()
    torch.set_rng_state(rng);yb=torch.cat(b(xb,ca,[0.,.04]));yb.square().mean().backward();after_b=torch.get_rng_state()
    pairs=[(ya,yb),*[(x.grad,y.grad) for x,y in zip(xa,xb)],*[(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters())]]
    errors=[float((x-y).abs().max()) for x,y in pairs];print('NESTED_DROPOUT_CHECKPOINT max_abs',max(errors))
    for x,y in pairs:
        assert torch.isfinite(x).all() and torch.isfinite(y).all();torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-4)
    assert torch.equal(after_a,after_b)
    assert a.layers[0].ffn[0].weight.grad.abs().sum()>0
    assert set(a.state_dict())==set(b.state_dict())


def test_ffn_execution_config_keeps_parameter_and_state_contract():
    torch.manual_seed(91);a=PVST('UV_PVST',FakeDecoder(),{**CFG,'ffn_chunk':3})
    torch.manual_seed(91);b=PVST('XYZ_PVST',FakeDecoder(),{**CFG,'ffn_chunk':8192})
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert list(a.state_dict())==list(b.state_dict())
    for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)
    assert 'coarse_time.layers.0.norm2.weight' in a.state_dict()
    assert 'coarse_time.layers.0.ffn.0.weight' in a.state_dict()
    assert 'coarse_time.layers.0.ffn.3.weight' in a.state_dict()


def test_complete_cache_reuse_checks_identity_without_frontend(tmp_path,monkeypatch):
    from research_tools.pvst import data
    from research_tools.v8.spatial_localization.joint_data import mapping
    row={'video_id':'v','video_sha256':'video','actual_frames':1,'source_hw':[2,3]};folder=tmp_path/'cache/v';folder.mkdir(parents=True)
    cfg={'checkpoint':'weights','workspace':str(tmp_path),'checkpoint_sha256':'weight','official_revision':'a','dino_revision':'b'}
    data.save(folder/'complete.json',{'video_sha256':'video','checkpoint_sha256':'weight','prepared_frames':1,'frames':1,'native_hw':[2,3],
        'official_revision':'a','dino_revision':'b','tokens':'no LoRA; 448 letterbox; image patch tokens only','mapping448':mapping([2,3],448),'seconds':2.})
    np.save(folder/'rgb.npy',np.zeros((1,2,3,3),np.uint8));np.save(folder/'tokens.npy',np.zeros((1,1024,32,32),np.float16))
    monkeypatch.setattr(data,'digest',lambda p:'weight');monkeypatch.setattr(data,'official',lambda *_:(_ for _ in ()).throw(AssertionError('FRONTEND_MUST_NOT_RUN')))
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    data.prepare(cfg,tmp_path,[row],lambda *_a,**_kw:None)
    import json,pytest
    assert json.loads((tmp_path/'preparation_cost.json').read_text())['new_token_frames_this_call']==0
    value=json.loads((folder/'complete.json').read_text());value['tokens']='LoRA';data.save(folder/'complete.json',value)
    with pytest.raises(ValueError,match='COMPLETE_CACHE_FIELDS_INCOMPATIBLE'):data.prepare(cfg,tmp_path,[row],lambda *_a,**_kw:None)


def test_exact_resource_window_counts_and_representative_dedup(tmp_path,monkeypatch):
    from research_tools.pvst import data,probe
    row={'video_id':'cb41b3d004b6640f6c3b5bb1','actual_frames':24,'source_hw':[1,3],'native_pixels':3,'dataset':'ANAKIN','split':'train','role':'fake'}
    data.save(tmp_path/'manifest.json',{'rows':[row]});data.save(tmp_path/'scale.json',{'s':1.})
    q=np.array([[[-.01,0,1],[.001,0,1],[.02,0,1]]],np.float32)
    monkeypatch.setattr(data,'native_dense',lambda *_:(q,np.ones((1,3),bool)))
    result=probe.scan_windows({'target_frames':8},tmp_path,lambda *_a,**_kw:None)
    assert len(result['representatives'])==2
    w=result['representatives'][0];assert w['input_range']==[0,24] and w['target_range']==[16,24]
    assert w['fine_nodes']==72 and w['coarse_nodes']==48 and len(w['reasons'])==3


@__import__('pytest').mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_attention_projection_chunks_match_original_whole_formula(dtype):
    from contextlib import nullcontext
    from research_tools.pvst.temporal import SparseLayer
    torch.manual_seed(883);a=SparseLayer(512,8,chunk=4,ffn_chunk=4).eval();b=copy.deepcopy(a)
    xa=torch.randn(11,512,dtype=dtype,requires_grad=True);xb=xa.detach().clone().requires_grad_();coords=[torch.stack((torch.arange(11),torch.zeros(11,dtype=torch.long),torch.zeros(11,dtype=torch.long)),-1)];graph=Neighborhood(coords,[0.]);index,rel=reference_neighbors(coords,1,[0.])
    ctx=lambda:torch.autocast('cpu',dtype=dtype) if dtype==torch.bfloat16 else nullcontext()
    with ctx():
        ya=a(xa,graph,1)
        q,k,v=b.qkv(b.norm(xb)).reshape(11,3,8,64).unbind(1)
        h=xb+b.drop(b.proj(b.attend(q,k,v,index,b.bias(rel))));yb=h+b.ffn(b.norm2(h))
    ya.float().square().mean().backward();yb.float().square().mean().backward()
    pairs=[(ya,yb),(xa.grad,xb.grad),*[(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters())]]
    errors=[float((x-y).abs().max()) for x,y in pairs];print('OUTPUT_PROJECTION_PARITY',str(dtype),'output/input/params max_abs',errors)
    atol,rtol=(2e-6,2e-4) if dtype==torch.float32 else (2e-3,3e-2)
    for x,y in pairs:torch.testing.assert_close(x,y,atol=atol,rtol=rtol)


def test_all_node_checkpoints_preserve_training_dropout_rng(monkeypatch):
    from research_tools.pvst import temporal
    torch.manual_seed(452);a=temporal.Temporal(256,4,chunk=4,ffn_chunk=4).train();b=copy.deepcopy(a)
    coords=[torch.tensor([[i,0,0] for i in range(5)])]*2
    xa=[torch.randn(5,256,requires_grad=True) for _ in range(2)];xb=[x.detach().clone().requires_grad_() for x in xa]
    rng=torch.get_rng_state();ya=torch.cat(a(xa,coords,[0.,.04]));ya.square().mean().backward();end_a=torch.get_rng_state()
    monkeypatch.setattr(temporal,'checkpoint',lambda fn,*args,**kwargs:fn(*args))
    torch.set_rng_state(rng);yb=torch.cat(b(xb,coords,[0.,.04]));yb.square().mean().backward();end_b=torch.get_rng_state()
    pairs=[(ya,yb),*[(x.grad,y.grad) for x,y in zip(xa,xb)],*[(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters())]]
    print('ALL_NODE_CHECKPOINT_PARITY max_abs',max(float((x-y).abs().max()) for x,y in pairs))
    for x,y in pairs:
        assert torch.isfinite(x).all() and torch.isfinite(y).all();torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-4)
    assert torch.equal(end_a,end_b)


def test_compact_adjacency_matches_old_across_frames_and_tail():
    coords=[torch.tensor([[-2,0,0],[-1,0,0],[0,0,0]]),torch.empty(0,3,dtype=torch.long),torch.tensor([[0,0,0],[5,-3,1]]),torch.tensor([[-1,0,0],[0,0,0]])]
    pts=[0.,.017,.081,.25];graph=Neighborhood(coords,pts)
    for d in [1,2,4]:
        old_i,old_r=reference_neighbors(coords,d,pts);indices=[];relations=[]
        for start in range(0,len(graph.coords),4):
            i,f=graph.query(start,min(start+4,len(graph.coords)),d);indices.append(i);relations.append(graph.relative[d][f])
        assert torch.equal(torch.cat(indices),old_i)
        torch.testing.assert_close(torch.cat(relations),old_r,rtol=0,atol=0)
    assert graph.relative[1].shape==(4,81,4)


def test_offload_outer_layer_inputs_once_without_replicated_kv(monkeypatch):
    from contextlib import contextmanager
    captured=[]
    @contextmanager
    def track(pin_memory):
        def pack(x):captured.append(tuple(x.shape));return x
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):yield
    monkeypatch.setattr(torch.autograd.graph,'save_on_cpu',track)
    torch.manual_seed(775);m=Temporal(256,4,chunk=2,ffn_chunk=3,offload_saved_tensors=True).train()
    coords=[torch.tensor([[i,0,0] for i in range(5)])];x=[torch.randn(5,256,requires_grad=True)]
    torch.cat(m(x,coords,[0.])).square().mean().backward()
    assert [a for a in captured if a!=(0,)]==[(5,256)]*3
    assert torch.isfinite(x[0].grad).all()


@__import__('pytest').mark.parametrize('dilation',[1,2,4])
@__import__('pytest').mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_shared_relation_bias_and_on_demand_graph_match_old_gradients(dilation,dtype):
    from contextlib import nullcontext
    from research_tools.pvst.temporal import SparseLayer
    torch.manual_seed(558);a=SparseLayer(256,4,chunk=4,ffn_chunk=3).eval();b=copy.deepcopy(a)
    coords=[torch.tensor([[-2,0,0],[-1,0,0],[0,0,0]]),torch.tensor([[0,0,0],[4,-2,1]]),torch.tensor([[-1,0,0],[0,0,0]])];pts=[0.,.031,.17]
    graph=Neighborhood(coords,pts);index,relative=reference_neighbors(coords,dilation,pts)
    x=torch.randn(7,256,dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_()
    ctx=lambda:torch.autocast('cpu',dtype=dtype) if dtype==torch.bfloat16 else nullcontext()
    with ctx():
        actual=a(x,graph,dilation);q,k,v=b.qkv(b.norm(y)).reshape(7,3,4,64).unbind(1)
        h=y+b.drop(b.proj(b.attend(q,k,v,index,b.bias(relative.to(q.dtype)))));expected=h+b.ffn(b.norm2(h))
    actual.float().square().mean().backward();expected.float().square().mean().backward()
    pairs=[(actual,expected),(x.grad,y.grad),*[(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters())]]
    errors=[float((x-y).abs().max()) for x,y in pairs];print('COMPACT_OLD_PARITY',dilation,str(dtype),'output/input/parameters',[*errors[:2],max(errors[2:])])
    atol,rtol=(3e-6,3e-4) if dtype==torch.float32 else (2e-3,3e-2)
    for u,v in pairs:torch.testing.assert_close(u,v,atol=atol,rtol=rtol)
