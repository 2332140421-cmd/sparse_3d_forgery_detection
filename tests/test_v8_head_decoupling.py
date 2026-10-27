import copy
import json

import numpy as np
import pytest
import torch

from research_tools.v8.spatial_localization.model import LocalizationModel,loss_sums,video_loss,STRUCTURE_KEYS
from research_tools.v8.full_video_training.model import VideoTraining
from research_tools.v8.spatial_localization.metrics import evaluate_video
from research_tools.v8.spatial_localization.data import to_source
from test_v8_joint_spatial import inputs,STATS,ToyVideo


def make(mode):
    torch.manual_seed(20260909)
    return LocalizationModel('COMPONENT_LOC',STATS,output_mode=mode)


def test_common_initialization_independent_copy_and_legacy_coupled_path():
    a=make('COUPLED');b=make('SEPARATE_HEADS')
    for key,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[key],atol=0,rtol=0)
    for p,q in zip(b.structure.cell_head.parameters(),b.detection_head.parameters()):
        assert p.data_ptr()!=q.data_ptr();torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())==8321
    x=inputs(1);h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    outputs=a.chunk(x,h,active,True);assert len(outputs)==3
    hh,aa,_,_,cells,history=VideoTraining(a.structure).chunk({k:x[k] for k in STRUCTURE_KEYS},h,active,True,True)
    torch.testing.assert_close(outputs[0],hh);torch.testing.assert_close(outputs[1],aa)
    torch.testing.assert_close(outputs[2],a.pixel(x,history,cells),atol=0,rtol=0)
    result=b.chunk(x,h,active,True);before=result[2].clone()
    with torch.no_grad():b.detection_head[-1].bias.add_(2)
    changed=b.chunk(x,h,active,True)
    torch.testing.assert_close(changed[2],before,atol=0,rtol=0)
    torch.testing.assert_close(changed[3],result[3]+2)


def test_separate_detection_and_pixel_gradient_routes():
    m=make('SEPARATE_HEADS');x=inputs(1);h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    out=m.chunk(x,h,active,True);n,area,frame,pixel,*_=loss_sums(out[2],x,out[3])
    (n-area.log()+frame).backward(retain_graph=True)
    assert m.detection_head[0].weight.grad.abs().sum()>0
    assert m.structure.gru.weight_ih.grad.abs().sum()>0
    for module in [m.rgb1,m.geo1,m.residual,m.structure.cell_head[0]]:assert module.weight.grad is None
    m.zero_grad(set_to_none=True);pixel.backward()
    assert all(p.grad is None for p in m.detection_head.parameters())
    for module in [m.rgb1,m.geo1,m.residual,m.structure.cell_head[0]]:assert module.weight.grad.abs().sum()>0
    assert m.structure.gru.weight_ih.grad.abs().sum()>0


def test_loss_logging_keeps_legacy_total_and_area_weighted_detection():
    x=inputs(2);z=torch.randn(2,512,512)
    sums=loss_sums(z,x,detailed=True)
    torch.testing.assert_close(sums[3],sums[6]+sums[7])
    old_frame=torch.stack([torch.logsumexp(z[i][x['content512'][i]].double(),0)-x['content512'][i].sum().double().log() for i in range(2)])
    torch.testing.assert_close(sums[5],old_frame)
    detection=torch.full((2,256),-1000.);detection[:,85]=torch.tensor([2.,-1.])
    separate=loss_sums(z,x,detection,True)
    torch.testing.assert_close(separate[5],torch.tensor([2.,-1.],dtype=torch.float64))
    torch.testing.assert_close(separate[3],sums[3])
    unknown={**x,'pixel_known':torch.zeros(2,dtype=torch.bool)}
    assert loss_sums(z,unknown,detection,True)[6]==0 and loss_sums(z,unknown,detection,True)[7]==0
    empty={**x,'gt':torch.zeros_like(x['gt'])}
    assert loss_sums(z,empty,detection,True)[7]==0 and loss_sums(z,empty,detection,True)[6]>0


def test_separate_checkpoint_gradients_and_native_evaluation_agree(tmp_path):
    x=inputs(2);m=make('SEPARATE_HEADS');v=ToyVideo(x)
    loss,parts=video_loss(m,v,2,False);loss.backward()
    reference={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True);other,_=video_loss(m,v,1,True);other.backward()
    torch.testing.assert_close(loss,other)
    for k,p in m.named_parameters():
        if k in reference:torch.testing.assert_close(p.grad,reference[k],atol=2e-5,rtol=2e-4)
    # The second block's detection must differentiate the first block's input.
    y=inputs(2);y['x'].requires_grad_();h=torch.zeros(256,128);active=torch.zeros(256,dtype=torch.bool)
    h,active,_,_=m.chunk({k:a[:1] for k,a in y.items()},h,active,True)
    _,_,_,d=m.chunk({k:a[1:] for k,a in y.items()},h,active,False);d.sum().backward()
    assert y['x'].grad[0].abs().sum()>0
    mapping={'source_hw':[512,512],'canvas_hw':[512,512],'content_hw':[512,512],'pad_xy':[0,0]}
    v.row.update(video_id='unit',pair_id='unit',source_id='unit',dataset='ANAKIN',split='train',label=1,
        role='fake',generator='unit',mapping512=mapping,pts=[{'pts_s':0.},{'pts_s':.04}],spatial_status='edit_mask')
    v.native_gt=lambda i:x['gt'][i].numpy()
    result=evaluate_video(m,v,1,sample_dir=tmp_path,save_logits=True)
    torch.testing.assert_close(torch.tensor(result['logit'],dtype=torch.float64),parts['logit'])
    assert result['loss']==pytest.approx(float(loss.detach()),rel=1e-6)
    assert result['loss_pixel']==pytest.approx(result['loss_pixel_bce']+result['loss_pixel_dice'],rel=1e-6)
    with np.load(tmp_path/'unit_0.npz',allow_pickle=False) as raw:
        assert raw['logits'].dtype==np.float32 and raw['native_gt'].dtype==np.bool_
        assert json.loads(str(raw['metadata']))['output_mode']=='SEPARATE_HEADS'
        assert to_source(raw['logits'],mapping).shape==(512,512)
    with torch.no_grad():
        for p in m.detection_head.parameters():p.zero_()
        m.detection_head[-1].bias.fill_(-3)
        for p in m.structure.cell_head.parameters():p.zero_()
        m.residual.weight.zero_();m.residual.bias.fill_(2)
    ungated=evaluate_video(m,v,1)
    assert ungated['logit']<0 and ungated['pixel_confusion_matrix'][1][1]>0


def test_read_only_cache_does_not_rebuild_missing_components(tmp_path):
    from research_tools.v8.spatial_localization.joint_data import Video,SCHEMA
    d=tmp_path/'source_cache'/'v';d.mkdir(parents=True)
    (d/'complete.json').write_text(json.dumps({'video_sha256':'sha','schema':SCHEMA,'fields':[],
        'hashes':{k:'sha' for k in ['mu','valid','area','motion','motion_valid']}}))
    with pytest.raises(FileNotFoundError,match='READ_ONLY_COMPONENT_CACHE_REQUIRED'):
        Video(tmp_path,{'video_id':'v','video_sha256':'sha'},grouping={'scale':1},cache_root=tmp_path/'source_cache',read_only=True)
    assert sorted(p.name for p in d.iterdir())==['complete.json']


def test_decoupling_runner_fixed_budget_and_pair_resume(tmp_path,monkeypatch):
    from research_tools.v8.spatial_localization import run,head_decoupling as heads
    class Tiny(torch.nn.Linear):
        def cuda(self):return self
    def factory(condition,*args):
        m=Tiny(1,1);m.condition=condition;return m
    monkeypatch.setattr(heads,'initial_model',factory);monkeypatch.setattr(heads,'curves',lambda *a:None)
    for name in ['synchronize','reset_peak_memory_stats','empty_cache']:monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ['max_memory_allocated','max_memory_reserved']:monkeypatch.setattr(torch.cuda,name,lambda:0)
    calls=[]
    def update(m,opt,pair,*args):
        calls.append((m.condition,pair[0]['pair_id']));opt.zero_grad()
        loss=sum((m(torch.tensor([[1.]])).squeeze()-r['label']).square()*.5 for r in pair)
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));opt.step()
        return {'loss':float(loss.detach()),'video':float(loss.detach()),'frame':0.,'pixel':0.,'pixel_bce':0.,'pixel_dice':0.,'gradient_norm':norm,'elapsed_s':.01,'frames':2}
    monkeypatch.setattr(run,'update_pair',update)
    monkeypatch.setattr(run,'evaluate',lambda *a,**kw:{'elapsed_s':.01,'records':[],
        'datasets':{d:{'spatial_dice':.5,'video':{'auroc':.5}} for d in ['ANAKIN','ActivityForensics']}})
    config={'source_run':str(tmp_path),'cache_root':str(tmp_path/'cache'),'epochs':2,'lr':1e-4,'weight_decay':1e-4,'chunk_frames':1,'amp':False}
    rows=[{'pair_id':d,'dataset':d,'split':'train','role':role,'label':int(role=='fake'),'video_id':d+role,'actual_frames':1,'train_monitor':True}
        for d in ['ActivityForensics','ANAKIN'] for role in ['real','fake']]
    fit={'statistics':{},'grouping':{}}
    resumed=tmp_path/'resumed';fresh=tmp_path/'fresh'
    for out in [resumed,fresh]:
        out.mkdir();(out/'joint_manifest.json').write_text('{}');(out/'statistics.json').write_text('{}')
        (out/'asset_references.json').write_text('{"small_file_hashes":{}}')
    monkeypatch.setattr(run,'STOP',True)
    with pytest.raises(InterruptedError):run.train(config,resumed,rows,fit,False)
    monkeypatch.setattr(run,'STOP',False);run.train(config,resumed,rows,fit,True)
    first=calls.copy();calls.clear();run.train(config,fresh,rows,fit,False)
    assert first==calls and len(calls)==8
    for c in heads.CONDITIONS:
        a=torch.load(resumed/c/'last.pt',weights_only=False);b=torch.load(fresh/c/'last.pt',weights_only=False)
        assert a['epoch']==2 and a['pair_index']==0 and a['updates']==4
        for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)


def test_final_matched_report_exports_binary_and_absolute_scale_index(tmp_path,monkeypatch):
    from research_tools.v8.spatial_localization import head_decoupling as heads
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    metric={'loss':1.,'loss_video':.5,'loss_frame':.4,'loss_pixel':.3,'loss_pixel_bce':.1,'loss_pixel_dice':.2,
        'spatial_dice':.3,'spatial_iou':.2,'pixel_ap':.4,'real_spatial_fpr':.1,
        'video':{'auroc':.5,'ap':.6},'frame':{'auroc':.5,'ap':.6}}
    datasets={d:copy.deepcopy(metric) for d in ['ANAKIN','ActivityForensics']};datasets['dataset_macro']={'auroc':.5}
    mapping={'source_hw':[4,4],'canvas_hw':[512,512],'content_hw':[512,512],'pad_xy':[0,0]}
    rows=[{'video_id':split,'pair_id':split,'source_id':split,'dataset':'ANAKIN','split':split,
        'role':'fake','actual_frames':4,'mapping512':mapping,'task':'unit fixture'} for split in ['train','dev']]
    for c in heads.CONDITIONS:
        directory=tmp_path/c;directory.mkdir()
        history=[{'epoch':1,'online_train':{k:1. for k in ['loss','video','frame','pixel_bce','pixel_dice']},
            'train':datasets,'dev':datasets,'gradient_norm_mean':.5}]
        (directory/'history.json').write_text(json.dumps(history))
        summary={'last_epoch':30,'best_epoch':5,'elapsed_s':1.,'parameters':362658,'additional_detection_parameters':0}
        for key in ['last_train','last_metrics','best_train','best_metrics']:summary[key]={'datasets':datasets,'records':[]}
        (directory/'summary.json').write_text(json.dumps(summary))
        for checkpoint in ['best','last']:
            for r in rows:
                d=directory/'samples'/checkpoint/r['split'];d.mkdir(parents=True)
                for fi in [1,2,3]:
                    np.savez_compressed(d/f'{r["video_id"]}_{fi}.npz',native_gt=np.eye(4,dtype=bool),
                        logits=np.zeros((512,512),np.float32),rgb512=np.zeros((512,512,3),np.uint8),pts_s=fi*.04)
    heads.report(tmp_path,rows)
    assert 'last/30' in (tmp_path/'report_zh.md').read_text() and 'best/5' in (tmp_path/'report_zh.md').read_text()
    assert '[-6,+6]' in (tmp_path/'index.html').read_text()
    assert len(json.loads((tmp_path/'fixed_sample_statistics.json').read_text()))==12
    assert len(list((tmp_path/'comparison').glob('*.png')))==12
