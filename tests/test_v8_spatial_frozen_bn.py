"""Frozen statistics only: affine/conv/context learning and checkpoint gradients stay live."""
import copy

import numpy as np
import pytest
import torch

from research_tools.v8.pretrained_spatial_context.model import SpatialModel,video_loss
from test_v8_pretrained_spatial_context import Decoder,STATS,Video,data


def test_fixed_bn_train_eval_cycles_and_legacy_default():
    decoder=Decoder().eval();a=SpatialModel('SPATIAL_TRANSFER',decoder,STATS)
    b=SpatialModel('SPATIAL_TRANSFER',decoder,STATS,freeze_decoder_bn=True)
    for _ in range(2):
        a.train();b.train()
        assert a.decoder.bn.training and not b.decoder.bn.training
        assert a.decoder.drop.training and b.decoder.drop.training
        assert b.bn_status()['buffers_equal']
        a.eval();b.eval()
        assert not a.decoder.drop.training and not b.decoder.drop.training
        x=torch.randn(2,1024,37,37)
        torch.testing.assert_close(a.decoder(x,(518,518)),b.decoder(x,(518,518)),atol=0,rtol=0)
    assert set(a.state_dict())==set(b.state_dict())  # Default checkpoint schemas unchanged.


@pytest.mark.parametrize('condition',['SPATIAL_TRANSFER','SPATIAL_3D_CONTEXT'])
def test_two_real_losses_keep_buffers_and_update_affine_conv_context(condition,tmp_path):
    torch.manual_seed(13);decoder=Decoder().eval();m=SpatialModel(condition,decoder,STATS,freeze_decoder_bn=True).train()
    v=Video(data(2));opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    before={k:p.detach().clone() for k,p in m.named_parameters()}
    for step in range(2):
        opt.zero_grad(set_to_none=True);loss,_=video_loss(m,v,1,True);loss.backward()
        assert m.decoder.bn.weight.grad.abs().sum()>0 and m.decoder.bn.bias.grad.abs().sum()>0
        if m.context and step==1:
            assert m.geo1.weight.grad.abs().sum()>0 and m.structure.gru.weight_ih.grad.abs().sum()>0
        opt.step();assert m.bn_status()['buffers_equal']
    assert not torch.equal(before['decoder.bn.weight'],m.decoder.bn.weight)
    assert not torch.equal(before['decoder.proj.weight'],m.decoder.proj.weight)
    path=tmp_path/'resume.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},path)
    n=SpatialModel(condition,decoder,STATS,freeze_decoder_bn=True);cp=torch.load(path,weights_only=False)
    n.load_state_dict(cp['model']);other=torch.optim.AdamW([p for p in n.parameters() if p.requires_grad],lr=1e-4);other.load_state_dict(cp['optimizer'])
    n.eval();n.train();assert n.bn_status()['buffers_equal'] and not n.decoder.bn.training and n.decoder.drop.training
    assert any(p is n.decoder.bn.weight for g in other.param_groups for p in g['params'])


def test_frozen_bn_checkpoint_matches_uncheckpointed_gradients():
    torch.manual_seed(19);a=SpatialModel('SPATIAL_3D_CONTEXT',Decoder().eval(),STATS,freeze_decoder_bn=True).train()
    torch.nn.init.normal_(a.projection[-1].weight,std=.01);b=copy.deepcopy(a);v=Video(data(2))
    torch.manual_seed(23);x,_=video_loss(a,v,1,False);x.backward()
    torch.manual_seed(23);y,_=video_loss(b,v,1,True);y.backward();torch.testing.assert_close(x,y)
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-5,rtol=3e-4)
    assert a.bn_status()['buffers_equal'] and b.bn_status()['buffers_equal']


def test_feature_root_uses_readonly_existing_map_without_copy(tmp_path,monkeypatch):
    from research_tools.v8.pretrained_spatial_context import assets
    class Base:
        def __init__(self,*a,**kw):
            assert kw['read_only'];self.arrays={'gt':np.zeros((1,2,2),bool),'content512':np.ones((1,2,2),bool)}
        def close(self):pass
    monkeypatch.setattr(assets,'Video',Base)
    features=tmp_path/'old_features'/'V';features.mkdir(parents=True);np.save(features/'patch.npy',np.ones((1,1024,37,37),np.float16))
    out=tmp_path/'new';out.mkdir();v=assets.CachedVideo({'source_run':str(tmp_path),'features_root':str(features.parent)},out,{'video_id':'V'},
        {'statistics':{},'grouping':{}},False)
    assert v.chunk(0,1)['features'].dtype==torch.float32 and not (out/'features').exists()
    assert not v.features.flags.writeable


def test_fixed_bn_report_matches_eight_identities_and_prespecified_weak_six(tmp_path):
    import json
    from research_tools.v8.pretrained_spatial_context.metrics import fixed_bn_comparison,CONDITIONS
    old=tmp_path/'original';new=tmp_path/'fixed';old.mkdir();new.mkdir()
    (new/'reference_reuse.json').write_text(json.dumps({'reference_run':str(old)}))
    ids=['an:4rvro9','an:4wgx37']+[f'an:weak{i}' for i in range(6)]
    current={}
    for ci,c in enumerate(CONDITIONS):
        for root,increment in [(old,0.),(new,.1)]:
            folder=root/c;folder.mkdir()
            dice=.3+increment-ci*.02
            metrics={'spatial_dice':dice,'spatial_iou':.2,'pixel_ap':.4,'real_spatial_fpr':.01}
            records=[{'video_id':p+'_fake','pair_id':p,'generator':'inpainting','role':'fake',
                      'spatial_dice':dice,'pixel_confusion_matrix':[[80,10],[5,5]]} for p in ids]
            summary={'best_epoch':1,'last_epoch':2,'best':{'train':{'metrics':metrics},'dev':{'metrics':metrics,'records':records}}}
            (folder/'summary.json').write_text(json.dumps(summary))
            (folder/'history.json').write_text(json.dumps([{'epoch':i,'train':metrics,'dev':metrics} for i in [1,2]]))
            if root==new:current[c]=summary
    for root in [old,new]:
        (root/'fixed_boundary_statistics.json').write_text(json.dumps({'aggregates':{c:{'GT_boundary_recall':.5} for c in CONDITIONS}}))
    fixed_bn_comparison(new,current)
    report=json.loads((new/'bn_strategy_comparison.json').read_text())
    assert len(report['paired_dev_fake'])==8
    assert report['groups']['全部8条']['delta_bn_transfer']['up']==8
    assert report['groups']['旧弱6条']['delta_bn_transfer']['up']==6
    assert report['groups']['全部8条']['delta_fixed_context']['down']==8
    assert report['coverage'][CONDITIONS[0]]['fixed']['GT_recall']==.5
    assert '停止继续修补当前融合方式' in report['recommendation']
    assert (new/'bn_strategy_curves.png').exists()
    assert (new/'report_zh.md').read_text().count('|FIXED_BN|')==2
