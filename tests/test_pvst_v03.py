"""Only v0.3 scale/point-path/chunk/supervision regressions."""
import copy
import json
import numpy as np
import pytest
import torch
from research_tools.pvst.geometry import voxel_size,voxelize
from research_tools.pvst.temporal import SparseLayer
from research_tools.pvst.model import PVST
from research_tools.pvst.data import Video,select_clips
from research_tools.pvst.v03 import clip_terms,paired_order
from test_pvst_v02 import FakeDecoder,CFG,frames


def test_shared_scale_reader_model_and_stale_routing(tmp_path):
    c={**CFG,'voxel_sizes':{'UV_PVST':.125,'XYZ_PVST':.25},'cache_root':str(tmp_path/'shared')}
    row={'video_id':'tiny','video_sha256':'abc','actual_frames':2,'source_hw':[3,4],'dataset':'ActivityForensics','role':'real','pts':[{'pts_s':.01},{'pts_s':.13}]}
    p=tmp_path/'shared/tiny';p.mkdir(parents=True);np.save(p/'rgb.npy',np.zeros((2,3,4,3),np.uint8));np.save(p/'tokens.npy',np.zeros((2,1024,32,32),np.float16));(p/'complete.json').write_text(json.dumps({'video_sha256':'abc'}))
    v=Video(tmp_path/'run',row,'UV_PVST',2.,c);f=v.frames(0,2,'cpu');m=PVST('UV_PVST',FakeDecoder(),c)
    assert m.h==v.h==voxel_size(c,'UV_PVST')==.125
    meta=f[0]['voxel_meta'];torch.testing.assert_close(meta['centers'],(meta['coords']+.5)*m.h)
    torch.testing.assert_close(meta['coarse_centers'][meta['parent']],(torch.div(meta['coords'],2,rounding_mode='floor')+.5)*2*m.h)
    points=f[0]['q'].reshape(-1,3)[meta['ids']];torch.testing.assert_close(meta['input'][:,3:6],(points-meta['centers'][meta['index']])/m.h)
    assert len(meta['ids'])==12 and int(meta['count'].sum())==12
    f[0]['voxel_meta']['voxel_size']=.01
    with pytest.raises(ValueError,match='SCALE_MISMATCH'):m(f)
    v.close()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_chunk_norm_qkv_output_and_gradients(dtype):
    torch.manual_seed(31);a=SparseLayer(256,4);b=copy.deepcopy(a);b.qkv_chunk=7
    x=torch.randn(19,256,dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
        ref=a.norm_qkv(x);parts=[b.norm_qkv(y[i:i+7]) for i in range(0,len(y),7)];got=torch.cat(parts)
    grad=torch.randn_like(ref);ga=torch.autograd.grad(ref,[x,*a.norm.parameters(),*a.qkv.parameters()],grad);gb=torch.autograd.grad(got,[y,*b.norm.parameters(),*b.qkv.parameters()],grad)
    errors=[float((ref-got).abs().max()),float((ga[0]-gb[0]).abs().max()),max(float((u-v).abs().max()) for u,v in zip(ga[1:],gb[1:]))];print('V03_QKV_PARITY',dtype,errors)
    torch.testing.assert_close(ref,got,atol=2e-2 if dtype==torch.bfloat16 else 2e-6,rtol=2e-2 if dtype==torch.bfloat16 else 2e-5)
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=.2 if dtype==torch.bfloat16 else 2e-5,rtol=.03 if dtype==torch.bfloat16 else 3e-5)
    assert a.state_dict().keys()==b.state_dict().keys()


def test_point_bypass_and_invalid_visual_prediction_have_real_gradients():
    torch.manual_seed(9);c={**CFG,'voxel_sizes':{'XYZ_PVST':.25},'qkv_chunk':7};m=PVST('XYZ_PVST',FakeDecoder(),c).eval()
    # Activate the original zero-initialized residual for a path-usage test.
    torch.nn.init.normal_(m.pixel_out.weight,std=.1);fs=frames(1,3,4);fs[0]['valid'][0,0]=False
    out=m(fs)[0]['pixel_logits'];assert out.shape==(3,4)
    g=torch.autograd.grad(out.sum(),[m.geo[0].weight,m.visual_native[0].weight],retain_graph=True)
    assert all(torch.isfinite(x).all() and x.abs().sum()>0 for x in g)
    invalid=torch.autograd.grad(out[0,0],m.visual_native[0].weight)[0];assert invalid.abs().sum()>0
    # Hold block context fixed: pixel head still depends on each pixel's point bypass.
    h,w=3,4;q=fs[0]['q'];meta=voxelize(q,fs[0]['valid'],.25)
    point=torch.randn(h,w,128,requires_grad=True);visual=torch.randn_like(point,requires_grad=True);context=torch.randn(len(meta['coords']),256)
    gradient=torch.autograd.grad(m.pixel(point,visual,context,meta).sum(),point)[0];assert gradient.abs().sum()>0
    m.check_bn()


def test_clip_sampling_temporal_strata_labels_and_unique_targets(monkeypatch,tmp_path):
    class StubVideo:
        def __init__(self,out,row,*args):self.r=row
        def label(self,i):return self.r['role']=='fake' and (self.r.get('all_positive') or 16<=i<32)
        def close(self):pass
    monkeypatch.setattr('research_tools.pvst.data.Video',StubVideo)
    rows=[{'video_id':str(k),'source_id':str(k//2),'split':'train','dataset':d,'role':role,'actual_frames':n,'pts':[{'pts_s':i*.037+i*i*.00001} for i in range(n)]} for k,(d,role,n) in enumerate([('ActivityForensics','fake',65),('ActivityForensics','real',9),('ANAKIN','fake',80),('ANAKIN','real',3)])]
    c={'seed':20260909,'clips_per_video':4};a=select_clips(rows,tmp_path,c,1);b=select_clips(rows,tmp_path,c,1);assert a==b
    for row in rows:
        cc=[s for s in a if s['video_id']==row['video_id']];assert len(cc)==min(4,(row['actual_frames']+7)//8)
        targets=[i for s in cc for i in range(*s['target_range'])];assert len(targets)==len(set(targets))
        for s in cc:
            left,end=s['input_range'];start=s['target_range'][0];assert start-left<=16 and end-start<=8
            assert s['input_pts_s']==[p['pts_s'] for p in row['pts'][left:end]]
    af=[s for s in a if s['video_id']=='0'];assert any(any(s['frame_labels']) for s in af) and any(not any(s['frame_labels']) for s in af)
    order=paired_order(a,20260909);assert sorted(i for pair in order for i in pair)==list(range(len(a)))
    fallback=select_clips([{**rows[0],'all_positive':True}],tmp_path,c,1);assert len(fallback)==4 and all(s['shortage_filled_from_other_candidates'] for s in fallback)


def test_segment_frame_average_and_two_clip_accumulation():
    p=torch.tensor(.4,requires_grad=True)
    def predict(n):return [{'frame_logit':p,'pixel_logits':p.expand(2,3)} for _ in range(n)]
    af=clip_terms(predict(3),[None]*3,[False,True,True],1.)
    an=clip_terms(predict(2),[np.zeros((2,3),bool),np.ones((2,3),bool)],[False,True],1.)
    expected=(af[0]+an[0])/2;g=torch.autograd.grad(expected,p,retain_graph=True)[0]
    ga=torch.autograd.grad(af[0]/2,p,retain_graph=True)[0];gb=torch.autograd.grad(an[0]/2,p)[0];torch.testing.assert_close(g,ga+gb)
    torch.testing.assert_close(af[0],torch.nn.functional.binary_cross_entropy_with_logits(p.expand(3),torch.tensor([0.,1.,1.])))
    assert af[2]==af[3]==0


def test_real_chunked_qkv_checkpoint_nested_gradients():
    from research_tools.pvst.temporal import Temporal
    torch.manual_seed(7);a=Temporal(256,4,7,9).train();b=copy.deepcopy(a)
    for layer in b.layers:layer.qkv_chunk=7
    coords=[torch.tensor([[0,0,0],[1,0,0],[0,1,0]])]*3
    xs=[torch.randn(3,256,requires_grad=True) for _ in coords];ys=[x.detach().clone().requires_grad_() for x in xs]
    # Same RNG per layout; norm/QKV has no dropout, so masks remain identical.
    torch.manual_seed(2);aa=torch.cat(a(xs,coords,[0.,.04,.15]));torch.manual_seed(2);bb=torch.cat(b(ys,coords,[0.,.04,.15]))
    torch.testing.assert_close(aa,bb,atol=2e-6,rtol=2e-5)
    aa.square().mean().backward();bb.square().mean().backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=3e-4)
    for x,y in zip(xs,ys):torch.testing.assert_close(x.grad,y.grad,atol=2e-7,rtol=2e-5)


def test_scale_budget_rejects_without_reading_development(monkeypatch,tmp_path):
    from research_tools.pvst.probe import select_v03_scale
    source=tmp_path/'source';source.mkdir();out=tmp_path/'run';seen=[]
    rows=[{'video_id':s,'video_sha256':s,'split':s,'actual_frames':2,'dataset':'ANAKIN','source_hw':[1,2]} for s in ['train','dev']]
    (source/'manifest.json').write_text(json.dumps({'rows':rows}));(source/'scale.json').write_text(json.dumps({'s':2.}))
    def dense(r,i):
        seen.append((r['split'],i));assert r['split']=='train'
        return np.array([[[-.1,0,1],[.1,0,1]]],np.float32),np.ones((1,2),bool)
    monkeypatch.setattr('research_tools.pvst.data.native_dense',dense)
    c={'asset_root':str(source),'voxel_candidates':[1/32,1/16,1/8],'voxel_budget':{'fine':0,'coarse':0}}
    result=select_v03_scale(c,out,lambda *a,**k:None)
    assert result['status']=='RESOURCE_BLOCKED' and result['selected_h'] is None and len(seen)==2
    assert len(result['candidates'])==3 and all(not r['within_budget'] for r in result['candidates'])


def test_scale_counts_match_exact_voxel_occupancy(monkeypatch,tmp_path):
    from research_tools.pvst.probe import select_v03_scale
    source=tmp_path/'source';source.mkdir();out=tmp_path/'run'
    row={'video_id':'train','video_sha256':'x','split':'train','actual_frames':3,'dataset':'ANAKIN','source_hw':[2,3]}
    (source/'manifest.json').write_text(json.dumps({'rows':[row]}));(source/'scale.json').write_text(json.dumps({'s':1.}))
    q=np.array([[[-.125,-.03125,0],[0,0,0],[.03125,0,0]],[[.125,.25,0],[.125,.25,0],[1,1,1]]],np.float32);valid=np.ones((2,3),bool);valid[-1,-1]=False
    monkeypatch.setattr('research_tools.pvst.data.native_dense',lambda *args:(q,valid))
    result=select_v03_scale({'asset_root':str(source),'voxel_candidates':[1/32,1/16,1/8],'voxel_budget':{'fine':0,'coarse':0}},out,lambda *a,**k:None)
    for r in result['candidates']:
        meta=voxelize(torch.from_numpy(q),torch.from_numpy(valid),r['h'])
        assert r['fine_max']==3*len(meta['coords']) and r['coarse_max']==3*len(meta['coarse_coords'])
