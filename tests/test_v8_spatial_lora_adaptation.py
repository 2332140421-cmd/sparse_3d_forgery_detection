"""Only the new trainable visual path, pair weighting and compact restore contracts."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from research_tools.v8.pretrained_spatial_context import lora
from research_tools.v8.pretrained_spatial_context.model import pixel_sums
from test_v8_pretrained_spatial_context import Decoder,STATS,MAP256,MAP512


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Conv2d(3,1024,1);self.lora_A=nn.Conv2d(3,4,1);self.lora_B=nn.Conv2d(4,1024,1);self.drop=nn.Dropout(.5)
    def forward(self,x):
        x=F.adaptive_avg_pool2d(x,(37,37));return self.base(x)+self.lora_B(self.drop(self.lora_A(x)))


class Network(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Module();self.encoder.backbone=Backbone();self.encoder.localization_head=Decoder()
        self.encoder.forward_backbone=lambda x:(None,self.encoder.backbone(x))
    @property
    def backbone(self):return self.encoder.backbone


class Video:
    def __init__(self,n):
        self.row={'actual_frames':n,'mapping256':MAP256,'mapping512':MAP512}
        self.a={'rgb512':torch.randint(256,(n,512,512,3),dtype=torch.uint8),
                'gt':torch.rand(n,512,512)>.8,'content512':torch.ones(n,512,512,dtype=torch.bool)}
    def chunk(self,i,j):return {k:v[i:j] for k,v in self.a.items()}


def test_online_lora_eval_backbone_train_decoder_gradients_and_compact_restore(tmp_path):
    torch.manual_seed(2);m=lora.LoRASpatial(Network(),STATS,False).train();v=Video(1)
    config={'lora_lr':1e-5,'decoder_lr':1e-4};opt=torch.optim.AdamW(m.parameter_groups(config))
    assert not m.network.backbone.training and not m.network.backbone.drop.training
    assert m.decoder.drop.training and not m.decoder.bn.training
    old=m.compact_state();base=m.network.backbone.base.weight.detach().clone()
    opt.zero_grad();lora.backward_video(m,v,1,amp=False)
    assert all(p.grad is not None and p.grad.abs().sum()>0 for k,p in m.network.backbone.named_parameters() if 'lora_' in k)
    assert m.network.backbone.base.weight.grad is None
    opt.step();assert torch.equal(base,m.network.backbone.base.weight)
    state=m.compact_state()
    assert set(state)=={'lora','decoder'} and all('lora_' in k for k in state['lora'])
    assert any(not torch.equal(state['lora'][k],old['lora'][k]) for k in old['lora'])
    assert any(not torch.equal(state['decoder'][k],old['decoder'][k]) for k in old['decoder'])
    assert m.bn_status()['buffers_equal']
    m.eval();cpu=v.chunk(0,1)
    with torch.no_grad():a=m.chunk(cpu,None,None,True,MAP256,MAP512)[2]
    torch.save(state,tmp_path/'compact.pt');m.restore(old);m.restore(torch.load(tmp_path/'compact.pt',weights_only=False))
    with torch.no_grad():b=m.chunk(cpu,None,None,True,MAP256,MAP512)[2]
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    m.train();assert not m.network.backbone.training and m.bn_status()['all_eval']


@pytest.mark.parametrize('chunk',[1,2,3])
def test_short_tail_frame_weighting_matches_full_video_loss_and_gradient(chunk):
    class Small(nn.Module):
        def __init__(self):super().__init__();self.w=nn.Parameter(torch.tensor(.2))
        def chunk(self,cpu,*a):return None,None,self.w*cpu['x']
    torch.manual_seed(5);m=Small();other=copy.deepcopy(m)
    arrays={'x':torch.randn(3,5,5),'gt':torch.rand(3,5,5)>.5,'content512':torch.ones(3,5,5,dtype=torch.bool)}
    v=SimpleNamespace(row={'actual_frames':3,'mapping256':{},'mapping512':{}},chunk=lambda i,j:{k:a[i:j] for k,a in arrays.items()})
    parts=lora.backward_video(m,v,chunk,False)
    full=.5*pixel_sums(other.w*arrays['x'],arrays['gt'],arrays['content512'])[0]/3;full.backward()
    assert parts['loss']==pytest.approx(float(full),rel=1e-6)
    torch.testing.assert_close(m.w.grad,other.w.grad)


def test_block_checkpoint_preserves_names_and_trainable_parameter_gradient():
    class B(nn.Module):
        def __init__(self):super().__init__();self.blocks=nn.ModuleList([nn.Linear(3,3),nn.Linear(3,3)])
        def forward(self,x):
            for b in self.blocks:x=b(x).relu()
            return x
    torch.manual_seed(8);a=B().eval();b=copy.deepcopy(a);keys=set(b.state_dict());assert lora.checkpoint_blocks(b)==2
    assert set(b.state_dict())==keys
    x=torch.randn(2,3);a(x).sum().backward();b(x).sum().backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad)


def test_rgb_video_never_opens_cached_patch_features(monkeypatch,tmp_path):
    class Base:
        def __init__(self,*a,**kw):
            assert kw['read_only'];self.arrays={'rgb512':torch.zeros(1,2,2,3).numpy(),'gt':torch.zeros(1,2,2).numpy(),'content512':torch.ones(1,2,2).numpy()}
        def close(self):pass
    monkeypatch.setattr(lora,'Video',Base)
    v=lora.RGBVideo({'source_run':str(tmp_path)},tmp_path,{}, {'statistics':{},'grouping':{}})
    assert set(v.chunk(0,1))=={'rgb512','gt','content512'}


def test_report_compares_saved_baseline_only_and_keeps_weak_six(tmp_path,monkeypatch):
    import json
    baseline=tmp_path/'baseline';original=tmp_path/'original';out=tmp_path/'new';out.mkdir()
    ids=['an:4rvro9','an:4wgx37']+[f'an:weak{i}' for i in range(6)]
    keys={'spatial_dice':.2,'spatial_iou':.1,'pixel_ap':.3,'real_spatial_fpr':.01}
    def summary(dice):
        metrics={**keys,'spatial_dice':dice}
        rr=[{'video_id':p,'pair_id':p,'generator':'inpainting','role':'fake','spatial_dice':dice,'pixel_confusion_matrix':[[80,10],[5,5]]} for p in ids]
        return {'best_epoch':1,'last_epoch':2,'updates':48,'stop_reason':'max30','costs':{},'best':{'train':{'metrics':metrics},'dev':{'metrics':metrics,'records':rr}},
                'last':{'train':{'metrics':metrics},'dev':{'metrics':metrics,'records':rr}}}
    for root,condition,dice in [(baseline,'SPATIAL_TRANSFER',.2),(original,'SPATIAL_TRANSFER',.3),(out,lora.CONDITION,.4)]:
        (root/condition).mkdir(parents=True);(root/condition/'summary.json').write_text(json.dumps(summary(dice)))
    monkeypatch.setattr(lora,'sample_index',lambda *a,**k:[])
    monkeypatch.setattr(lora,'boundary_samples',lambda *a,**k:{'FROZEN_VISUAL':{},lora.CONDITION:{}})
    lora.report({'baseline_run':str(baseline),'original_run':str(original)},out,[])
    result=json.loads((out/'result_summary.json').read_text())
    assert result['delta']['spatial_dice']==pytest.approx(.2)
    assert result['groups']['原较弱6条']['up']==6
    assert len(result['paired_dev_fake'])==8
    assert result['lora']['dev']['metrics']['spatial_dice']==.4
    assert (out/'FROZEN_VISUAL').is_symlink()
    assert 'monitor仅为固定4pair' in (out/'report_zh.md').read_text()
    repo=tmp_path/'repo';(repo/'docs/experiments').mkdir(parents=True);(repo/'docs/v8').mkdir()
    doc=repo/'docs/experiments/v8-spatial-lora-adaptation.md';doc.write_text('RUNNING；\n设计保留\n')
    status=repo/'docs/v8/current_status.md';status.write_text('## 当前执行：空间定位LoRA适配准备\n旧状态\n## 已完成：固定decoder BN\n历史保持\n')
    (out/lora.CONDITION/'curves.png').write_bytes(b'curve fixture')
    def output(args,**kw):
        if args[1]=='branch':return 'v8-spatial-lora-adaptation\n'
        if args[1]=='status':return ''
        if args[1]=='rev-parse':return 'abc123\n'
        if args[1]=='ls-remote':return 'abc123\trefs/heads/v8-spatial-lora-adaptation\n'
        raise AssertionError(args)
    calls=[];monkeypatch.setattr(lora.subprocess,'check_output',output)
    monkeypatch.setattr(lora.subprocess,'run',lambda args,**kw:calls.append(args))
    lora.publish_results({},out,repo)
    assert [a[1] for a in calls]==['add','commit','push']
    assert '设计保留' in doc.read_text() and 'COMPLETE；' in doc.read_text()
    assert '历史保持' in status.read_text() and '已完成：空间定位LoRA适配' in status.read_text()
    assert json.loads((out/'publication.json').read_text())['commit']=='abc123'


def test_publication_stops_before_writing_on_unrelated_worktree_changes(tmp_path,monkeypatch):
    def output(args,**kw):return 'v8-spatial-lora-adaptation' if args[1]=='branch' else ' M user-file'
    monkeypatch.setattr(lora.subprocess,'check_output',output)
    with pytest.raises(ValueError,match='WORKTREE_CHANGED'):lora.publish_results({},tmp_path,tmp_path)
    assert not list(tmp_path.iterdir())
