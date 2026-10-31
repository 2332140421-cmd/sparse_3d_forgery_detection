"""Research contracts for the two-condition temporal detection comparison."""
import copy
import numpy as np
import pytest
import torch

from research_tools.v8.temporal_detection_core.model import StructureModel
from research_tools.v8.temporal_detection_core.data import matched_motion,representation,components
from research_tools.v8.temporal_detection_core.run import labels,classification,aggregate
from research_tools.v8.full_video_training.model import VideoTraining
from research_tools.v8.full_video_training.temporal import frame_logits,frame_loss

STATS={'absolute_mean':[0.]*3,'absolute_scale':[1.]*3,'relative_scale':[1.]*3}

def batch(t=5):
    n=256;a=torch.zeros(t,n);a[:,:4]=256
    p=torch.arange(n)[:,None].expand(n,4).clone();w=torch.zeros(t,n,4);w[:,:,0]=1
    content=torch.zeros(t,n,16,16,dtype=torch.bool);content[:,:4]=True
    return {'x':torch.randn(t,n,80),'mu':torch.randn(t,n,3),'valid':content.clone(),'content':content,
            'area':a,'component_ids':torch.arange(n)[None].expand(t,n),
            'predecessor':p[None].expand(t,n,4),'association_weight':w,
            'history_available':torch.ones(t,n,dtype=torch.bool),'delta_t':torch.tensor([0.,.04,.05,.03,.07][:t])}


def test_numeric_order_time_and_checkpoint_gradients():
    torch.set_num_threads(2);torch.manual_seed(1);m=StructureModel(STATS);a=batch();a['x'].requires_grad_()
    z,c=VideoTraining(m).video(a,2,use_checkpoint=True,return_cells=True)
    f=frame_logits(c,a['area']);f[-1].backward()
    assert a['x'].grad[0].abs().max()>0
    assert m.region_attention.in_proj_weight.grad.abs().sum()>0
    assert m.patch_encoder.conv1.weight.grad.abs().sum()>0
    with torch.no_grad():
        z0,c0=VideoTraining(m).video(a,5,use_checkpoint=False,return_cells=True)
        torch.testing.assert_close(z,z0,atol=1e-6,rtol=1e-6);torch.testing.assert_close(c,c0,atol=1e-6,rtol=1e-6)
        other={k:v.flip(0) for k,v in a.items()};_,c1=VideoTraining(m).video(other,2,use_checkpoint=False,return_cells=True)
        assert not torch.allclose(c0,c1.flip(0))
        perturbed={k:v.clone() for k,v in a.items()};perturbed['x'][:,4:]=1e5;perturbed['mu'][:,4:]=1e5
        zz=VideoTraining(m).video(perturbed,2,use_checkpoint=False);torch.testing.assert_close(z,zz)


def test_2d_independent_of_every_3d_field():
    class No3D(dict):
        def __getitem__(self,k):
            if k in ['mu','geometry_count','motion3','motion_valid3','valid']:raise AssertionError('3D_READ:'+k)
            return super().__getitem__(k)
    a=No3D(area=np.ones((2,256),np.float32)*256,x=np.zeros((2,256,77),np.float32),
        uv=np.ones((256,3),np.float32),motion2=np.ones((2,256,3),np.float32),
        motion_valid2=np.ones((2,256),bool),history_available=np.ones((2,256),bool))
    a['uv'][:,2]=0;a['motion2'][:,:,2]=0
    x,p,c,m,mv=representation(a,2);assert x.shape==(2,256,80)
    assert np.all(x[:,:,72:75]==a['motion2']);assert np.all(p[:,:,2]==0)
    a['motion2']*=2;assert not np.array_equal(x,representation(a,2)[0])


def test_matched_points_and_explicit_missing():
    tracks=np.zeros((16,1024,2),np.float32);tracks[15]=[32.,32.];tracks[14]=[31.,32.]
    tr={'tracks':tracks,'visibility':np.ones((16,1024),bool),'query_frame_in_padded_prefix':15}
    current=np.zeros((256,256,3),np.float32);previous=current.copy();current[32,32]=[2.,4.,6.]
    valid=np.ones((256,256),bool)
    mm=matched_motion(tr,[0,1],current,valid,previous,valid)
    np.testing.assert_allclose(mm['motion2'][:,0],1/255);np.testing.assert_allclose(mm['motion3'],np.tile([2,4,6],(256,1)))
    invalid=np.zeros_like(valid);missing=matched_motion(tr,[0,1],current,invalid,previous,valid)
    assert not missing['motion_valid3'].any();assert missing['motion_valid2'].all()
    birth=matched_motion(tr,[0],current,valid,None,None)
    assert not birth['motion_valid2'].any() and not birth['motion_valid3'].any()
    # Grouping really uses motion, not only static coordinates.
    pos=np.zeros((256,3));active=np.zeros(256,bool);active[:2]=True;v=active.copy();motion=pos.copy()
    cfg={'spatial_scale':1.,'motion_scale':1.,'merge_threshold':.1}
    merged=components(pos,v,motion,v,active,cfg);motion[1,0]=10
    separate=components(pos,v,motion,v,active,cfg)
    assert merged[0]==merged[1] and separate[0]!=separate[1]


def test_native_pts_closed_union_and_standard_metrics():
    r={'pts':[{'pts_s':p} for p in [0.,.09,.21,.24]],'intervals_s':[[.08,.22]],'role':'fake'}
    np.testing.assert_array_equal(labels(r),[0,1,1,0])
    z=torch.tensor([-1.,2.,0.,-2.],requires_grad=True);loss=frame_loss(z,labels(r));loss.backward();assert z.grad.abs().sum()>0
    m=classification([0,0,1,1],[-1,0,-.2,2]);assert [m[k] for k in ['TN','FP','FN','TP']]==[1,1,1,1]
    assert m['F1']==.5 and m['AUROC']==.75


def test_capacity_seed_pair_and_saved_weight_parity(tmp_path):
    torch.manual_seed(7);m=StructureModel(STATS)
    torch.manual_seed(7);n=StructureModel({**STATS,'absolute_mean':[1.,2.,3.]})
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in n.parameters())
    for (name,p),(other,q) in zip(m.named_parameters(),n.named_parameters()):assert name==other and torch.equal(p,q)
    torch.save(m.state_dict(),tmp_path/'m.pt');n.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True))
    for k,v in m.state_dict().items():assert torch.equal(v,n.state_dict()[k])


def test_real_mmap_prepare_and_short_final_chunk(tmp_path):
    import json
    from research_tools.v8.temporal_detection_core.data import prepare_structures,load_video,save
    r={'video_id':'test','video_sha256':'abc','actual_frames':19,'split':'train',
       'mapping256':{'pad_xy':[0,0],'content_hw':[256,256]}}
    dest=tmp_path/'observations/test';dest.mkdir(parents=True);rng=np.random.default_rng(9);t=19;n=256
    fields={'x':rng.normal(size=(t,n,77)).astype(np.float32),'mu':rng.normal(size=(t,n,3)).astype(np.float32),
       'area':np.ones((t,n),np.float32)*256,'geometry_count':np.ones((t,n),np.float32)*256,
       'valid':np.ones((t,n,16,16),bool),'motion2':rng.normal(size=(t,n,3)).astype(np.float32),
       'motion3':rng.normal(size=(t,n,3)).astype(np.float32),'motion_valid2':np.ones((t,n),bool),
       'motion_valid3':np.ones((t,n),bool),'predecessor':np.tile(np.arange(n)[None,:,None],(t,1,4)),
       'association_weight':np.tile(np.array([1,0,0,0],np.float32)[None,None],(t,n,1)),
       'history_available':np.ones((t,n),bool),'delta_t':np.full(t,.04,np.float32)}
    fields['motion2'][:,:,2]=0
    for k,v in fields.items():np.save(dest/(k+'.npy'),v)
    save(dest/'complete.json',{'video_sha256':'abc','frames':t,'fields':list(fields)})
    fitted=prepare_structures({},tmp_path,[r],lambda *args,**kwargs:None)
    for c in fitted:
        a,_=load_video(tmp_path,r,c);assert a['x'].shape==(t,n,80);assert a['mu'].shape==(t,n,3)
        assert a['valid'].shape==(t,n,16,16)
        assert a['mu'].dtype==torch.float32 and a['x'].dtype==torch.float32
        m=StructureModel(fitted[c]['statistics']);short={k:v[:2] for k,v in a.items()}
        with torch.no_grad():assert torch.isfinite(VideoTraining(m).video(short,1,use_checkpoint=False))
