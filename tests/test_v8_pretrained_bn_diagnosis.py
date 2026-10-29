"""Only BN may change mode/buffers; preserve native metrics and fixed identities."""
import copy

import numpy as np
import pytest
import torch

from research_tools.v8 import pretrained_spatial_bn_diagnosis as diag
from research_tools.v8.pretrained_spatial_context.model import SpatialModel
from test_v8_pretrained_spatial_context import Decoder,STATS,data


def test_bn_only_switch_independent_weights_and_no_parameter_update():
    torch.manual_seed(43);a=SpatialModel('SPATIAL_TRANSFER',Decoder(),STATS)
    a.decoder.bn.running_mean.fill_(3.);b=copy.deepcopy(a)
    ra=diag.set_bn_mode(a,'RUNNING_STATS');rb=diag.set_bn_mode(b,'BATCH_STATS')
    assert not a.training and not b.training and not a.decoder.training and not b.decoder.training
    assert not a.decoder.drop.training and not b.decoder.drop.training
    assert not ra['bn'].training and rb['bn'].training
    sha=diag.parameter_sha(a);assert sha==diag.parameter_sha(b)
    x=torch.randn(2,1024,37,37)
    with torch.no_grad():
        za=a.decoder(x,(518,518));zb=b.decoder(x,(518,518));again=b.decoder(x,(518,518))
    assert not torch.allclose(za,zb);torch.testing.assert_close(zb,again)
    assert int(a.decoder.bn.num_batches_tracked)==0 and int(b.decoder.bn.num_batches_tracked)==2
    assert diag.parameter_sha(a)==diag.parameter_sha(b)==sha
    assert all(not p.requires_grad and p.grad is None for m in [a,b] for p in m.parameters())
    context=SpatialModel('SPATIAL_3D_CONTEXT',Decoder(),STATS);diag.set_bn_mode(context,'BATCH_STATS')
    assert not context.structure.gru.training and not context.structure.region_attention.training


def test_exact_pair_selection_and_reproduction_guard():
    rows=[{'pair_id':p,'role':role,'split':'train','source_hw':[640,360] if p==diag.PAIRS[1] else [368,640]}
        for p in diag.PAIRS for role in ['real','fake']]
    assert diag.select(rows)==rows
    with pytest.raises(ValueError,match='FIXED_TRAIN_PAIR'):diag.select(rows[:-1])
    with pytest.raises(ValueError,match='FIXED_TRAIN_PAIR'):diag.select([{**r,'split':'dev'} for r in rows])
    a={'spatial_dice':.5,'spatial_iou':.4,'empty_spatial_fpr':None};b={**a,'spatial_dice':.5000001}
    assert diag.compare_existing(a,b)['max_abs']<1e-5
    with pytest.raises(ValueError,match='REPRODUCTION_MISMATCH'):diag.compare_existing(a,{**a,'spatial_dice':.6})


def test_native_metric_mask_empty_and_two_frame_chunks(tmp_path,monkeypatch):
    mapping={'source_hw':[512,512],'canvas_hw':[512,512],'content_hw':[512,512],'pad_xy':[0,0]}
    row={'video_id':'fixture','pair_id':'fixture','source_id':'s','role':'fake','split':'train','actual_frames':3,
         'mapping256':{},'mapping512':mapping,'pts':[{'pts_s':i*.04} for i in range(3)]}
    calls=[]
    class Video:
        def __init__(self,config,source,row,fit,context):
            self.a=data(3);self.a['gt'][:]=False;self.a['gt'][[0,2],10:12,10:12]=True
            self.base=type('Base',(),{'arrays':{'rgb512':np.zeros((3,512,512,3),np.uint8)}})()
        def chunk(self,i,j):calls.append((i,j));return {k:v[i:j] for k,v in self.a.items()}
        def native_gt(self,i):return self.a['gt'][i].numpy()
        def close(self):pass
    monkeypatch.setattr(diag,'CachedVideo',Video)
    m=SpatialModel('SPATIAL_TRANSFER',Decoder(),STATS);diag.set_bn_mode(m,'BATCH_STATS')
    with torch.no_grad():m.decoder.predictor.weight.zero_();m.decoder.predictor.bias.fill_(2.)
    before=diag.parameter_sha(m);r=diag.evaluate_video(m,{'amp':False},tmp_path,tmp_path/'samples',row,{})
    assert calls==[(0,2),(2,3)] and r['positive_gt_frames']==2 and r['empty_gt_frames']==1
    assert r['gt_pixel_recall']==1 and r['predicted_area_fraction']==1 and r['empty_spatial_fpr']==1
    assert r['gt_mean_logit']==2 and r['outside_mean_logit']==2
    assert r['spatial_dice']==8/(512*512+4)
    assert diag.parameter_sha(m)==before and all(p.grad is None for p in m.parameters())
    assert len(list((tmp_path/'samples').glob('*.npz')))==3


def test_fixed_preview_distinguishes_global_shift_without_new_predictions(tmp_path,monkeypatch):
    import json
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    mapping={'source_hw':[8,8],'canvas_hw':[512,512],'content_hw':[512,512],'pad_xy':[0,0]}
    row={'video_id':'fixture','pair_id':'pair','role':'fake','actual_frames':4,'mapping512':mapping}
    for c in diag.CONDITIONS:
        for mode,value in zip(diag.MODES,[-2.,2.]):
            folder=tmp_path/c/mode;folder.mkdir(parents=True)
            for fi in [1,2,3]:
                np.savez_compressed(folder/f'fixture_{fi}.npz',logits=np.full((512,512),value,np.float32),
                    native_gt=np.eye(8,dtype=bool),rgb512=np.zeros((512,512,3),np.uint8),pts_s=fi*.04)
    diag.sample_index(tmp_path,[row])
    records=json.loads((tmp_path/'fixed_sample_differences.json').read_text())
    assert len(records)==6 and len(list((tmp_path/'comparison').glob('*.png')))==6
    assert all(r['delta_mean']==4 and r['delta_std']==0 and r['threshold_flip_fraction']==1 for r in records)
    assert 'RUNNING_STATS | BATCH_STATS' in (tmp_path/'index.html').read_text()
