import copy
import numpy as np
import torch

from research_tools.v8.spatial_localization.model import LocalizationModel,video_loss,loss_sums
from research_tools.v8.full_video_training.model import VideoTraining

STATS={'absolute_mean':[0.,0.,0.],'absolute_scale':[1.,1.,1.],'relative_scale':[1.,1.,1.]}


def inputs(t=2):
    g=torch.Generator().manual_seed(19)
    content=torch.zeros(t,256,16,16,dtype=torch.bool);content[:,85]=True
    area=content.sum((-1,-2)).float();valid=content.clone()
    rgb=torch.randint(0,255,(t,512,512,3),generator=g,dtype=torch.uint8)
    cm=content.reshape(t,16,16,16,16).permute(0,1,3,2,4).reshape(t,256,256)
    cm512=cm.repeat_interleave(2,1).repeat_interleave(2,2)
    gt=torch.zeros(t,512,512,dtype=torch.bool);gt[:,164:174,164:174]=True
    return {'x':torch.randn(t,256,77,generator=g),'mu':torch.zeros(t,256,3),
        'xyz':torch.randn(t,256,16,16,3,generator=g),'content':content,'valid':valid,'area':area,
        'component_ids':torch.arange(256).expand(t,256).clone(),'predecessor':torch.arange(256)[None,:,None].expand(t,256,4).clone(),
        'association_weight':torch.full((t,256,4),.25),'history_available':torch.ones(t,256,dtype=torch.bool),
        'delta_t':torch.full((t,),.04),'rgb512':rgb,'content512':cm512,'gt':gt,
        'pixel_known':torch.ones(t,dtype=torch.bool),'frame_label':torch.ones(t)}


class ToyVideo:
    def __init__(self,a):self.a=a;self.row={'actual_frames':len(a['x']),'label':1}
    def chunk(self,start,end):return {k:v[start:end] for k,v in self.a.items()}


def model(condition='CELL_LOC'):
    torch.manual_seed(20260909);return LocalizationModel(condition,STATS)


def test_paired_parameters_and_legacy_default_hidden_interface():
    a=model();b=model('COMPONENT_LOC')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):assert ka==kb and torch.equal(pa,pb)
    x=inputs(1);h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    old=VideoTraining(a.structure).chunk(x,h,active,True)
    new=VideoTraining(a.structure).chunk(x,h,active,True,True)
    assert len(old)==5 and len(new)==6
    for aa,bb in zip(old,new):torch.testing.assert_close(aa,bb)


def test_dense_geometry_within_cell_response_and_padding_invariance():
    m=model();a=inputs(1);h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    _,_,z=m.chunk(a,h,active,True)
    assert z[0,160:192,160:192].std()>0
    b={k:v.clone() for k,v in a.items()};b['xyz'][0,85,2,2]+=10
    _,_,other=m.chunk(b,h,active,True)
    assert not torch.equal(z,other)
    c={k:v.clone() for k,v in a.items()};c['xyz'][~c['content']]=999;c['valid'][~c['content']]=True
    c['rgb512'][~c['content512']]=255;c['x'][c['area']==0]=999;c['mu'][c['area']==0]=999
    _,_,changed=m.chunk(c,h,active,True)
    torch.testing.assert_close(z,changed)


def test_all_losses_reach_pixel_and_structure_and_missing_is_not_empty():
    m=model();a=inputs(1);h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    _,_,z=m.chunk(a,h,active,True);n,area,f,p,k,_=loss_sums(z,a)
    for loss in [n-area.log(),f,p]:
        m.zero_grad(set_to_none=True);loss.backward(retain_graph=True)
        for module in [m.rgb1,m.geo1,m.residual,m.structure.gru]:
            assert module.weight.grad.abs().sum()>0 if hasattr(module,'weight') else module.weight_ih.grad.abs().sum()>0
    missing={**a,'pixel_known':torch.zeros(1,dtype=torch.bool)}
    assert loss_sums(z,missing)[3]==0 and loss_sums(z,missing)[4]==0
    empty={**a,'gt':torch.zeros_like(a['gt'])}
    assert loss_sums(z,empty)[3]>0


def test_chunk_checkpoint_and_cross_boundary_gradients_match_reference():
    a=inputs(2);m=model();v=ToyVideo(a)
    loss,_=video_loss(m,v,2,False);loss.backward();reference={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True);other,_=video_loss(m,v,1,True);other.backward()
    torch.testing.assert_close(loss,other,atol=1e-5,rtol=1e-5)
    for k,p in m.named_parameters():
        if k in reference:torch.testing.assert_close(p.grad,reference[k],atol=2e-5,rtol=2e-4)
    # Future-only loss must differentiate the earlier frame's structural input.
    x=inputs(2);x['x'].requires_grad_();h=torch.zeros(256,128);pa=torch.zeros(256,dtype=torch.bool)
    h,pa,_=m.chunk({k:v[:1] for k,v in x.items()},h,pa,True)
    _,_,z=m.chunk({k:v[1:] for k,v in x.items()},h,pa,False);z.sum().backward()
    assert x['x'].grad[0].abs().sum()>0


def test_optimizer_and_rng_restore_same_next_update(tmp_path):
    from research_tools.v8.local_geometry_pilot.runner import rng_state,restore_rng
    m=model();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);v=ToyVideo(inputs(1))
    loss,_=video_loss(m,v,1,False);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    path=tmp_path/'last.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'epoch':0,'pair_index':1},path)
    expected_random=torch.rand(3);loss,_=video_loss(m,v,1,False);loss.backward();opt.step()
    n=model();other=torch.optim.AdamW(n.parameters(),lr=1e-4);cp=torch.load(path,weights_only=False)
    n.load_state_dict(cp['model']);other.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    assert torch.equal(torch.rand(3),expected_random)
    loss,_=video_loss(n,v,1,False);loss.backward();other.step()
    for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p,q)


def test_fitted_components_are_persisted_with_raw_and_grouping_identity(tmp_path):
    import json
    from research_tools.v8.spatial_localization.joint_data import Video,SCHEMA
    from scripts.prepare_v8_assets import digest
    row={'video_id':'V','video_sha256':'source-sha','spatial_status':'missing','actual_frames':1}
    path=tmp_path/'cache/V';path.mkdir(parents=True)
    a=inputs(1);a.update(motion=torch.zeros(1,256,2),motion_valid=torch.ones(1,256,dtype=torch.bool))
    for k,v in a.items():np.save(path/(k+'.npy'),v.numpy(),allow_pickle=False)
    meta={'schema':SCHEMA,'video_sha256':'source-sha','fields':list(a),'hashes':{k:digest(path/(k+'.npy')) for k in a}}
    (path/'complete.json').write_text(json.dumps(meta))
    grouping={'spatial_scale':1.,'motion_scale':1.,'merge_threshold':1.}
    v=Video(tmp_path,row,grouping=grouping);original=v.chunk(0,1)['component_ids'];v.close()
    assert len(list(path.glob('fitted_components_*.npy')))==1
    v=Video(tmp_path,row,grouping=grouping);torch.testing.assert_close(v.chunk(0,1)['component_ids'],original);v.close()
    changed={**grouping,'merge_threshold':.1};v=Video(tmp_path,row,grouping=changed);v.close()
    assert len(list(path.glob('fitted_components_*.npy')))==2


def test_metrics_exclude_empty_background_from_positive_video_dice():
    from research_tools.v8.spatial_localization.metrics import aggregate,classification
    rows=[]
    for role,label,dice in [('real',0,None),('fake',1,.2),('fake',1,.8)]:
        rows.append({'dataset':'ANAKIN','role':role,'label':label,'logit':label*2-1,
            'frame_labels':[label],'frame_scores':[label*2-1],
            'empty_spatial_fpr':.1 if not label else None,'spatial_iou':dice,'spatial_dice':dice,'pixel_ap':dice,
            'loss':1.,'loss_video':1.,'loss_frame':1.,'loss_pixel':1.})
    result=aggregate(rows)
    assert result['ANAKIN']['spatial_dice']==.5
    assert result['ANAKIN']['real_spatial_fpr']==.1
    assert classification([0,1],[0.,0.])['confusion_matrix']==[[0,1],[0,1]]


def test_joint_runner_resumes_at_committed_pair_without_repeat(tmp_path,monkeypatch):
    import json
    import pytest
    from research_tools.v8.spatial_localization import run
    class Tiny(torch.nn.Linear):
        def cuda(self):return self
    def factory(condition,stats):
        m=Tiny(1,1);m.condition=condition;return m
    monkeypatch.setattr(run,'LocalizationModel',factory)
    for name in ['synchronize','reset_peak_memory_stats','empty_cache']:
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ['max_memory_allocated','max_memory_reserved']:
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    calls=[]
    def update(m,opt,pair,*args):
        calls.append((m.condition,pair[0]['pair_id']));opt.zero_grad()
        loss=sum((m(torch.tensor([[r['value']]],dtype=torch.float32)).squeeze()-r['label']).square()*.5 for r in pair)
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
        return {'loss':float(loss.detach()),'video':float(loss.detach()),'frame':0.,'pixel':0.,'gradient_norm':norm,'elapsed_s':.01,'frames':2}
    monkeypatch.setattr(run,'update_pair',update)
    def evaluation(*args,**kwargs):
        return {'elapsed_s':.01,'records':[],'datasets':{d:{'spatial_dice':.5,'video':{'auroc':.5}} for d in ['ANAKIN','ActivityForensics']}}
    monkeypatch.setattr(run,'evaluate',evaluation);monkeypatch.setattr(run,'report',lambda out:None)
    config={'epochs':2,'patience':8,'min_delta':.002,'lr':1e-4,'weight_decay':1e-4,'chunk_frames':1,'amp':False}
    rows=[{'pair_id':d,'dataset':d,'split':'train','role':role,'label':int(role=='fake'),'value':1.,'video_id':d+role,'actual_frames':1,'train_monitor':True}
          for d in ['ActivityForensics','ANAKIN'] for role in ['real','fake']]
    fitted={'statistics':{},'grouping':{}}
    resumed=tmp_path/'resumed';fresh=tmp_path/'fresh'
    for out in [resumed,fresh]:
        out.mkdir();(out/'joint_manifest.json').write_text('{}');(out/'statistics.json').write_text('{}')
    monkeypatch.setattr(run,'STOP',True)
    with pytest.raises(InterruptedError):run.train(config,resumed,rows,fitted,False)
    assert len(calls)==1
    cp=torch.load(resumed/'CELL_LOC/last.pt',weights_only=False)
    assert cp['epoch']==0 and cp['pair_index']==1 and len(cp['online'])==1
    monkeypatch.setattr(run,'STOP',False);run.train(config,resumed,rows,fitted,True)
    resumed_order=calls.copy();calls.clear();run.train(config,fresh,rows,fitted,False)
    assert resumed_order==calls and len(calls)==8
    for condition in run.CONDITIONS:
        a=torch.load(resumed/condition/'last.pt',weights_only=False)['model']
        b=torch.load(fresh/condition/'last.pt',weights_only=False)['model']
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


def test_native_spatial_confusion_distinguishes_missing_and_empty():
    from research_tools.v8.spatial_localization.metrics import spatial_frame
    pred=np.array([[True,True],[False,False]])
    gt=np.array([[True,False],[True,False]])
    s=spatial_frame(pred,gt)
    assert s['confusion_matrix']==[[1,1],[1,1]] and s['dice']==.5 and s['iou']==1/3
    empty=spatial_frame(pred,np.zeros((2,2),bool))
    assert empty['dice'] is None and empty['empty_fpr']==.5 and empty['confusion_matrix']==[[2,2],[0,0]]
    assert spatial_frame(pred,None)['confusion_matrix'] is None


def test_final_report_renders_both_conditions_and_matching_epoch_table(tmp_path,monkeypatch):
    import json
    from research_tools.v8.spatial_localization.metrics import aggregate,report
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    records=[]
    for dataset in ['ANAKIN','ActivityForensics']:
        for role,label in [('real',0),('fake',1)]:
            records.append({'dataset':dataset,'role':role,'label':label,'logit':label*2-1,'frame_labels':[label],
                'frame_scores':[label*2-1],'generator':'unit_fixture','empty_spatial_fpr':.1 if not label else None,
                'spatial_dice':.5 if label and dataset=='ANAKIN' else None,
                'spatial_iou':.4 if label and dataset=='ANAKIN' else None,'pixel_ap':.6 if label and dataset=='ANAKIN' else None,
                'loss':1.,'loss_video':.5,'loss_frame':.5,'loss_pixel':.25})
    metrics={'datasets':aggregate(records),'records':records}
    for condition in ['CELL_LOC','COMPONENT_LOC']:
        d=tmp_path/condition;d.mkdir()
        (d/'history.json').write_text(json.dumps([{'condition':condition,'epoch':1,'online_train':{'loss':1.},'train_monitor':metrics['datasets'],'dev':metrics['datasets']}]))
        (d/'summary.json').write_text(json.dumps({'status':'COMPLETE','best_epoch':1,'last_epoch':1,'stop_reason':'unit_fixture',
            'parameters':362658,'train_videos':4,'dev_videos':4,'best_metrics':metrics,'last_metrics':metrics,'best_train_monitor':metrics}))
    report(tmp_path)
    text=(tmp_path/'report_zh.md').read_text()
    assert 'best/1' in text and 'last/1' in text and 'COMPONENT_LOC减CELL_LOC' in text
    assert all((tmp_path/c/'curves.png').is_file() for c in ['CELL_LOC','COMPONENT_LOC'])
