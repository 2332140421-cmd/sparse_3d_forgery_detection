"""Contracts specific to the frozen pretrained spatial/context comparison."""
import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from research_tools.v8.pretrained_spatial_context.assets import normalize518,validate_state
from research_tools.v8.pretrained_spatial_context.model import SpatialModel,align256,pixel_sums,video_loss
from test_v8_joint_spatial import inputs,STATS


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv2d(1024,4,1);self.bn=nn.BatchNorm2d(4)
        self.drop=nn.Dropout(.1);self.predictor=nn.Conv2d(4,1,1)
    def forward(self,x,target_size):
        return F.interpolate(self.predictor(self.drop(self.bn(self.proj(x)))),size=target_size,mode='bilinear',align_corners=False)


MAP256={'pad_xy':[0,0],'content_hw':[256,256]}
MAP512={'pad_xy':[0,0],'content_hw':[512,512]}


def data(t=2):
    a=inputs(t);a['features']=torch.randn(t,1024,37,37)
    return a


class Video:
    def __init__(self,a):
        self.a=a;self.row={'actual_frames':len(a['gt']),'mapping256':MAP256,'mapping512':MAP512}
    def chunk(self,i,j):return {k:v[i:j] for k,v in self.a.items()}


def test_checkpoint_requires_backbone_lora_and_trained_decoder():
    state={k:torch.ones(2) for k in ['encoder.backbone.base','encoder.backbone.lora_A','encoder.localization_head.predictor','encoder.classifier.weight']}
    assert len(validate_state(state,state))==3
    for k in list(state)[:3]:
        bad=dict(state);bad.pop(k)
        with pytest.raises(ValueError,match='LOCALIZATION_KEYS'):validate_state(state,bad)
    bad=dict(state);bad['encoder.localization_head.predictor']=torch.ones(3)
    with pytest.raises(ValueError,match='LOCALIZATION_SHAPES'):validate_state(state,bad)


def test_black_padding_preserves_last_row_and_column():
    rgb=torch.zeros(1,512,512,3,dtype=torch.uint8);rgb[:,511,:,:]=255;rgb[:,:,511,:]=127
    normalized=normalize518(rgb);std=torch.tensor([.229,.224,.225])[None,:,None,None];mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    raw=(normalized*std+mean)*255
    torch.testing.assert_close(raw[:,:,:512,:512],rgb.permute(0,3,1,2).float(),atol=2e-5,rtol=0)
    torch.testing.assert_close(raw[:,:,512:,:],torch.zeros_like(raw[:,:,512:,:]),atol=2e-5,rtol=0)
    torch.testing.assert_close(raw[:,:,:,512:],torch.zeros_like(raw[:,:,:,512:]),atol=2e-5,rtol=0)


@pytest.mark.parametrize('m256,m512',[
    ({'pad_xy':[0,55],'content_hw':[145,256]},{'pad_xy':[0,109],'content_hw':[293,512]}),
    ({'pad_xy':[55,0],'content_hw':[256,145]},{'pad_xy':[109,0],'content_hw':[512,293]})])
def test_mapping_uses_actual_rounded_dimensions(m256,m512):
    y,x=torch.meshgrid(torch.arange(256)+.5,torch.arange(256)+.5,indexing='ij')
    f=torch.stack((x,y))[None];z=align256(f,m256,m512)
    px=(torch.arange(37)+.5)*14
    expected_x=(px-m512['pad_xy'][0])/m512['content_hw'][1]*m256['content_hw'][1]+m256['pad_xy'][0]
    expected_y=(px-m512['pad_xy'][1])/m512['content_hw'][0]*m256['content_hw'][0]+m256['pad_xy'][1]
    keep=(expected_x>.5)&(expected_x<255.5);keep_y=(expected_y>.5)&(expected_y<255.5)
    torch.testing.assert_close(z[0,0][keep_y][:,keep],expected_x[keep][None].expand(int(keep_y.sum()),-1),atol=4e-5,rtol=1e-5)
    torch.testing.assert_close(z[0,1][keep_y][:,keep],expected_y[keep_y][:,None].expand(-1,int(keep.sum())),atol=4e-5,rtol=1e-5)


def test_zero_projection_gt_independence_and_two_update_gradients():
    torch.manual_seed(19);decoder=Decoder();a=SpatialModel('SPATIAL_TRANSFER',decoder,STATS).eval();b=SpatialModel('SPATIAL_3D_CONTEXT',decoder,STATS).eval()
    x=data(1);h=torch.zeros(256,128);pa=torch.zeros(256,dtype=torch.bool)
    za=a.chunk(x,h,pa,True,MAP256,MAP512)[2];zb=b.chunk(x,h,pa,True,MAP256,MAP512)[2]
    torch.testing.assert_close(za,zb,atol=0,rtol=0)
    changed={**x,'gt':~x['gt'],'frame_label':1-x['frame_label']}
    torch.testing.assert_close(zb,b.chunk(changed,h,pa,True,MAP256,MAP512)[2],atol=0,rtol=0)
    assert not any(p.requires_grad for p in b.structure.cell_head.parameters())
    opt=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=1e-4)
    before=b.decoder.proj.weight.detach().clone()
    for i in range(2):
        opt.zero_grad(set_to_none=True);z=b.chunk(x,h,pa,True,MAP256,MAP512)[2]
        pixel_sums(z,x['gt'],x['content512'])[0].backward()
        for p in [b.geo1.weight,b.structure.gru.weight_ih]:
            assert p.grad is not None
            assert p.grad.abs().sum()==0 if i==0 else p.grad.abs().sum()>0
        assert b.projection[-1].weight.grad.abs().sum()>0;opt.step()
    assert not torch.equal(before,b.decoder.proj.weight)
    # Missing geometry does not remove visual prediction or supervision.
    missing={**x,'valid':torch.zeros_like(x['valid'])}
    assert torch.isfinite(b.chunk(missing,h,pa,True,MAP256,MAP512)[2]).all()


def test_checkpoint_preserves_gradients_rng_bn_and_cross_chunk_state():
    torch.manual_seed(29);m=SpatialModel('SPATIAL_3D_CONTEXT',Decoder(),STATS).train()
    nn.init.normal_(m.projection[-1].weight,std=.01)
    n=copy.deepcopy(m);x=data(2);v=Video(x)
    torch.manual_seed(37);loss,_=video_loss(m,v,1,False);loss.backward()
    torch.manual_seed(37);other,_=video_loss(n,v,1,True);other.backward()
    torch.testing.assert_close(loss,other)
    for (k,p),(kk,q) in zip(m.named_parameters(),n.named_parameters()):
        assert k==kk
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-5,rtol=3e-4)
    for p,q in zip(m.decoder.buffers(),n.decoder.buffers()):torch.testing.assert_close(p,q)
    assert int(n.decoder.bn.num_batches_tracked)==2
    m.eval();x['x'].requires_grad_();h=torch.zeros(256,128);pa=torch.zeros(256,dtype=torch.bool)
    h,pa,_=m.chunk({k:v[:1] for k,v in x.items()},h,pa,True,MAP256,MAP512)
    _,_,z=m.chunk({k:v[1:] for k,v in x.items()},h,pa,False,MAP256,MAP512);z.sum().backward()
    assert x['x'].grad[0].abs().sum()>0
    z1=m.chunk({k:v[:1] for k,v in x.items()},h,pa,True,MAP256,MAP512)[2]
    z2=m.chunk({k:v[:1] for k,v in x.items()},torch.zeros_like(h),torch.zeros_like(pa),True,MAP256,MAP512)[2]
    torch.testing.assert_close(z1,z2)


def test_spatial_loss_empty_gt_and_padding_exclusion():
    z=torch.zeros(2,4,4,requires_grad=True);gt=torch.zeros_like(z);gt[0,1,1]=1
    content=torch.zeros_like(z,dtype=torch.bool);content[:,1:3,1:3]=True
    total,bce,dice=pixel_sums(z,gt,content)
    torch.testing.assert_close(bce,torch.tensor(2.).log()*2)
    torch.testing.assert_close(dice,torch.tensor(.5)) # eps1: 1-(2*.5+1)/(2+1+1)
    changed=z.detach().clone();changed[~content]=100
    torch.testing.assert_close(total,pixel_sums(changed,gt,content)[0])
    total.backward();assert z.grad[~content].abs().sum()==0


def test_runner_resume_pair_order_and_epoch0_best(tmp_path,monkeypatch):
    import json
    from research_tools.v8.pretrained_spatial_context import run
    class Tiny(nn.Module):
        def __init__(self,c,*args,**kwargs):
            super().__init__();self.decoder=nn.Linear(1,1);self.condition=c
        def cuda(self):return self
    monkeypatch.setattr(run,'SpatialModel',Tiny);monkeypatch.setattr(run,'curves',lambda *args:None)
    for name in ['reset_peak_memory_stats','empty_cache']:monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    calls=[]
    def update(m,opt,pair,*args):
        calls.append((m.condition,pair[0]['pair_id']));opt.zero_grad()
        loss=(m.decoder(torch.ones(1,1)).squeeze()-torch.rand(())).square();loss.backward();opt.step()
        return {'loss':float(loss.detach()),'bce':float(loss.detach()),'dice':0.,'gradient_norm':1.,'seconds':.01,'frames':2}
    monkeypatch.setattr(run,'pair_update',update)
    monkeypatch.setattr(run,'evaluate',lambda *args,**kw:{'metrics':{'spatial_dice':.4},'seconds':.01})
    rows=[{'pair_id':p,'role':r,'split':'train','actual_frames':1} for p in ['a','b'] for r in ['real','fake']]
    config={'epochs':2};resumed=tmp_path/'resumed';fresh=tmp_path/'fresh'
    for out in [resumed,fresh]:
        out.mkdir();(out/'manifest.json').write_text('{}');(out/'official_load.json').write_text('{}')
        (out/'epoch0.json').write_text(json.dumps({'dev':{'metrics':{'spatial_dice':.5}}}))
    monkeypatch.setattr(run,'STOP',True)
    with pytest.raises(InterruptedError):run.train(config,resumed,rows,{'statistics':{}},None,0)
    monkeypatch.setattr(run,'STOP',False);run.train(config,resumed,rows,{'statistics':{}},None,0,True)
    order=calls.copy();calls.clear();run.train(config,fresh,rows,{'statistics':{}},None,0)
    assert order==calls and len(order)==8
    for c in run.CONDITIONS:
        a=torch.load(resumed/c/'last.pt',weights_only=False);b=torch.load(fresh/c/'last.pt',weights_only=False)
        assert a['epoch']==2 and a['updates']==4 and a['best_epoch']==0
        for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
        assert json.loads((resumed/c/'summary.json').read_text())['best_epoch']==0


def test_evaluation_loss_matches_training_and_no_detection_gate(tmp_path,monkeypatch):
    import numpy as np
    from research_tools.v8.pretrained_spatial_context import metrics
    class Cached:
        def __init__(self,*args):
            self.x=data(1);self.x['content512'][:]=True;self.x['gt'][:]=False
        def chunk(self,*args):return self.x
        def native_gt(self,i):return np.zeros((512,512),bool)
        def close(self):pass
    monkeypatch.setattr(metrics,'CachedVideo',Cached)
    m=SpatialModel('SPATIAL_TRANSFER',Decoder(),STATS).eval()
    with torch.no_grad():
        m.decoder.predictor.weight.zero_();m.decoder.predictor.bias.fill_(2.)
    r={'video_id':'real','pair_id':'p','source_id':'s','split':'dev','generator':'inpainting','role':'real',
       'actual_frames':1,'mapping256':MAP256,'mapping512':{**MAP512,'source_hw':[512,512],'canvas_hw':[512,512]},'pts':[{'pts_s':0.}]}
    result=metrics.evaluate(m,{'chunk_frames':1,'amp':False},tmp_path,[r],{})
    ref=pixel_sums(torch.full((1,512,512),2.),torch.zeros(1,512,512),torch.ones(1,512,512,dtype=torch.bool))
    assert result['metrics']['loss']==float(ref[0]);assert result['metrics']['real_spatial_fpr']==1.
    assert result['metrics']['spatial_dice'] is None and result['metrics']['pixel_ap'] is None


def test_custom_condition_report_and_native_boundary_diagnostics(tmp_path,monkeypatch):
    import json
    import numpy as np
    from research_tools.v8.pretrained_spatial_context import metrics
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    mapping={'source_hw':[10,10],'canvas_hw':[512,512],'content_hw':[512,512],'pad_xy':[0,0]}
    rows=[{'video_id':s,'pair_id':s,'source_id':s,'dataset':'ANAKIN','split':s,'role':'fake',
        'actual_frames':4,'mapping512':mapping,'task':'fixture','train_monitor':s=='train'} for s in ['train','dev']]
    metric={'spatial_dice':.5,'spatial_iou':.4,'pixel_ap':.6,'real_spatial_fpr':.1}
    rec={'video_id':'dev','pair_id':'dev','role':'fake','generator':'inpainting','spatial_dice':.5,'pixel_confusion_matrix':[[30,20],[20,30]]}
    (tmp_path/'source_references.json').write_text(json.dumps({'source_run':str(tmp_path/'unrelated')}))
    (tmp_path/'epoch0.json').write_text(json.dumps({s:{'metrics':metric} for s in ['train','dev']}))
    for c in metrics.CONDITIONS:
        d=tmp_path/c;d.mkdir();value={'status':'COMPLETE','best_epoch':0,'last_epoch':8,'updates':192,'stop_reason':'patience8_delta0.002','costs':{}}
        for ck in ['best','last']:
            value[ck]={s:{'metrics':metric,'records':[rec]} for s in ['train','dev']}
            for r in rows:
                dest=d/'samples'/ck/r['split'];dest.mkdir(parents=True)
                for fi in [1,2,3]:
                    gt=np.zeros((10,10),bool);gt[2:8,2:8]=True
                    np.savez_compressed(dest/f'{r["video_id"]}_{fi}.npz',native_gt=gt,logits=np.zeros((512,512),np.float32),
                        rgb512=np.zeros((512,512,3),np.uint8),pts_s=fi*.04)
        (d/'summary.json').write_text(json.dumps(value))
    metrics.report(tmp_path,rows)
    text=(tmp_path/'report_zh.md').read_text()
    assert 'best/0' in text and '小样本适配与过拟合' in text
    assert '独立输出目录总存储=' in text
    assert 'SPATIAL_3D_CONTEXT' in (tmp_path/'index.html').read_text()
    a=json.loads((tmp_path/'fixed_boundary_statistics.json').read_text())
    assert len(a['records'])==6
    assert a['aggregates']['SPATIAL_TRANSFER']['GT_boundary_recall']==1.
