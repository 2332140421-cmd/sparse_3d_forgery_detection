"""Expansion-specific source isolation, visual-only inputs and matched conditions."""
import json

import numpy as np
import pytest
import torch

from research_tools.v8.pretrained_spatial_context.expansion_data import allocate,stratified_order,source_key,VisualVideo


def candidate(vid,source,split=None,unseen=True,official='train',area='small',kind='inpainting'):
    return {'video-id':vid,'source_group':source,'retained_split':split,'test_unseen_source':unseen,
            'official_split':official,'area_group':area,'manipulation-type':kind}


def test_source_groups_retained_roles_test_unseen_and_seeded_area_strata():
    rows=[candidate('old_t','t','train',False),candidate('old_d','d','dev',False),
          candidate('old_version','t',None,False),candidate('reserved','r',official='test'),
          candidate('new_d','a',unseen=False,area='medium'),candidate('new_t','b',unseen=False,area='large')]
    result=allocate(rows,{'train':2,'dev':2,'test_reserved':1});mapping={r['video-id']:r['split'] for r in result}
    assert mapping['old_t']=='train' and mapping['old_d']=='dev' and mapping['reserved']=='test_reserved'
    assert len({r['source_group'] for r in result})==len(result)==5
    assert 'old_version' not in mapping
    assert all(r['test_unseen_source'] for r in result if r['split']=='test_reserved')
    assert stratified_order(rows)==stratified_order(list(reversed(rows)))
    assert source_key('https://www.vimeo.com/123/?x=1')==source_key('https://vimeo.com/123')


def test_visual_cache_requires_no_geometry_and_rejects_test(tmp_path):
    directory=tmp_path/'cache';directory.mkdir()
    (directory/'complete.json').write_text(json.dumps({'video_sha256':'sha'}))
    np.save(directory/'rgb512.npy',np.zeros((3,4,4,3),np.uint8));np.save(directory/'gt.npy',np.zeros((3,4,4),bool))
    np.save(directory/'content512.npy',np.ones((4,4),bool))
    row={'split':'train','cache_dir':str(directory),'video_sha256':'sha','role':'real','source_hw':[4,4]}
    v=VisualVideo({'condition':'LORA_ADAPT'},tmp_path,row)
    assert set(v.chunk(1,3))=={'rgb512','gt','content512'}
    assert v.chunk(1,3)['content512'].shape==(2,4,4) and not v.native_gt(0).any()
    assert all(not a.flags.writeable for a in v.arrays.values())
    v.close()
    with pytest.raises(ValueError,match='RESERVED_TEST'):VisualVideo({},tmp_path,{**row,'split':'test_reserved'})
    with pytest.raises(ValueError,match='IDENTITY'):VisualVideo({},tmp_path,{**row,'video_sha256':'different'})
    (directory/'gt.npy').unlink()
    with pytest.raises(FileNotFoundError):VisualVideo({},tmp_path,row)  # Missing GT is never empty GT.


@pytest.mark.parametrize('condition,updates_lora',[('FROZEN_VISUAL',False),('LORA_ADAPT',True)])
def test_matched_trainability_bn_and_compact_reload(condition,updates_lora,tmp_path):
    from research_tools.v8.pretrained_spatial_context.expansion import MatchedSpatial
    from research_tools.v8.pretrained_spatial_context.lora import backward_video
    from test_v8_spatial_lora_adaptation import Network,Video
    torch.manual_seed(17);model=MatchedSpatial(Network(),condition).train();v=Video(1)
    with torch.no_grad():features=model.network.encoder.forward_backbone(torch.zeros(1,3,518,518))[1].half().float()
    original_chunk=v.chunk
    v.chunk=lambda i,j:{**original_chunk(i,j),'features':features[i:j]}
    config={'lora_lr':1e-5,'decoder_lr':1e-4};opt=torch.optim.AdamW(model.parameter_groups(config));before=model.compact_state()
    opt.zero_grad();backward_video(model,v,1,False);opt.step();after=model.compact_state()
    changed=any(not torch.equal(before['lora'][k],t) for k,t in after['lora'].items())
    assert changed==updates_lora
    assert any(not torch.equal(before['decoder'][k],t) for k,t in after['decoder'].items())
    assert model.bn_status()['buffers_equal'] and not model.network.training and model.decoder.drop.training
    path=tmp_path/'state.pt';torch.save(after,path);model.eval();cpu=v.chunk(0,1)
    with torch.no_grad():a=model.chunk(cpu,None,None,True,{}, {})[2]
    model.restore(before);model.restore(torch.load(path,weights_only=False))
    with torch.no_grad():b=model.chunk(cpu,None,None,True,{}, {})[2]
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_partial_checkpoint_preserves_keys_and_gradient():
    import copy
    from torch import nn
    from research_tools.v8.pretrained_spatial_context.lora import checkpoint_blocks
    class B(nn.Module):
        def __init__(self):super().__init__();self.blocks=nn.ModuleList([nn.Linear(3,3) for _ in range(4)])
        def forward(self,x):
            for b in self.blocks:x=b(x).tanh()
            return x
    torch.manual_seed(9);a=B();b=copy.deepcopy(a);assert checkpoint_blocks(b,every=2)==2
    assert set(a.state_dict())==set(b.state_dict())
    x=torch.ones(2,3);a(x).sum().backward();b(x).sum().backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad)


def test_remote_publication_failure_keeps_completed_experiment_and_true_exit_history(tmp_path,monkeypatch):
    import subprocess
    from research_tools.v8.pretrained_spatial_context import expansion
    progress={'status':'COMPLETE','stage':'report_complete','pid':99,'publication_status':'pending'}
    raw=json.dumps(progress);(tmp_path/'progress.json').write_text(raw);(tmp_path/'exit_code.txt').write_text('1\n')
    def fail(config,out):
        (out/'publication.json').write_text(json.dumps({'status':'PUSHED','commit':'abc'}))
        raise subprocess.CalledProcessError(128,['git','ls-remote'])
    monkeypatch.setattr(expansion,'publish',fail)
    with pytest.raises(subprocess.CalledProcessError):expansion.publish_after_completion({},tmp_path)
    assert (tmp_path/'progress.json').read_text()==raw
    assert (tmp_path/'exit_code.txt').read_text()=='1\n'
    publication=json.loads((tmp_path/'publication.json').read_text())
    assert publication['status']=='FAILED' and publication['last_completed_publication_stage']=='PUSHED'
    assert publication['commit']=='abc' and publication['traceback']


def test_new_visual_cache_checks_pts_and_mapping(tmp_path):
    import hashlib
    directory=tmp_path/'visual';directory.mkdir();pts=[{'pts_s':0.}];mapping={'canvas_hw':[4,4]}
    meta={'schema':'v8-spatial-visual-expansion-v1','video_sha256':'s','mapping512':mapping,
          'pts_sha256':hashlib.sha256(json.dumps(pts,sort_keys=True).encode()).hexdigest()}
    (directory/'complete.json').write_text(json.dumps(meta))
    for k,a in {'rgb512':np.zeros((1,4,4,3),np.uint8),'gt':np.zeros((1,4,4),bool),'content512':np.ones((4,4),bool)}.items():np.save(directory/(k+'.npy'),a)
    row={'split':'dev','cache_dir':str(directory),'video_sha256':'s','role':'real','source_hw':[4,4],'actual_frames':1,'pts':pts,'mapping512':mapping}
    VisualVideo({},tmp_path,row).close()
    with pytest.raises(ValueError,match='MAPPING_OR_PTS'):VisualVideo({},tmp_path,{**row,'pts':[{'pts_s':1.}]})


def test_expansion_report_uses_full_dev_and_fixed_area_groups(tmp_path,monkeypatch):
    from research_tools.v8.pretrained_spatial_context import expansion
    old=tmp_path/'old';old.mkdir();(old/'manifest.json').write_text(json.dumps({'rows':[{'video_id':'a','split':'dev'},{'video_id':'b','split':'dev'}]}))
    rows=[{'video_id':v,'area_group':area,'split':'dev','role':'fake'} for v,area in [('a','small'),('b','large')]]
    def result(value):
        metrics={'spatial_dice':value/2,'spatial_iou':.1,'pixel_ap':.3,'gt_recall':.4,'real_spatial_fpr':.02}
        rr=[{'video_id':v,'pair_id':v,'generator':'inpainting','role':'fake','spatial_dice':dice,'spatial_iou':.1,'pixel_ap':.3,'pixel_confusion_matrix':[[80,10],[5,5]]} for v,dice in [('a',0.),('b',value)]]
        return {'metrics':metrics,'records':rr}
    for c,value in [('FROZEN_VISUAL',.4),('LORA_ADAPT',.6)]:
        (tmp_path/c).mkdir();r=result(value)
        (tmp_path/c/'summary.json').write_text(json.dumps({'best_epoch':2,'last_epoch':10,'updates':290,'costs':{},'best':{'train':r,'dev':r},'last':{'train':r,'dev':r}}))
    (tmp_path/'LORA_24_REFERENCE').mkdir();(tmp_path/'LORA_24_REFERENCE/summary.json').write_text(json.dumps(result(.5)))
    (tmp_path/'asset_report.json').write_text(json.dumps({'actual_pairs':{'train':29,'dev':20,'test_reserved':20}}));(tmp_path/'resource_smoke.json').write_text('{}')
    monkeypatch.setattr(expansion,'sample_index',lambda *a,**kw:[]);monkeypatch.setattr(expansion,'boundary_samples',lambda *a,**kw:{})
    expansion.report({'original_run':str(old)},tmp_path,rows)
    saved=json.loads((tmp_path/'result_summary.json').read_text())
    assert saved['delta']['spatial_dice']==pytest.approx(.1)
    assert saved['distributions']['LORA_ADAPT']['dice_below_0_05']==1
    assert saved['distributions']['LORA_ADAPT']['median']==pytest.approx(.3)
    assert saved['distributions']['LORA_ADAPT']['groups']['area']['small']['mean']==0.
    assert len(saved['distributions']['LORA_ADAPT']['videos'])==2


def test_frozen_manifest_preserves_roles_and_has_no_source_leakage():
    from pathlib import Path
    path=Path(__file__).parents[1]/'docs/v8/evidence/spatial_supervision_expansion_manifest.json'
    m=json.loads(path.read_text());rows=m['rows'];groups={};pairs={}
    assert m['actual_pairs']['train']>24 and m['actual_pairs']['dev']>8
    for r in rows:
        assert groups.setdefault(r['source_group'],r['split'])==r['split']
        pairs.setdefault(r['pair_id'],[]).append(r['role'])
        assert len(r['pts'])==r['actual_frames'] and r['official_revision']==m['revision']
    assert all(sorted(roles)==['fake','real'] for roles in pairs.values())
    monitors={r['pair_id'] for r in rows if r.get('train_monitor')}
    assert monitors=={'an:95qlg9','an:76rjm7','an:7p8r14','an:40y1x9'}


def test_probe_reuse_checks_final_execution_config_without_repeating_gpu_trials(tmp_path):
    import hashlib
    from research_tools.v8.pretrained_spatial_context.expansion import benchmark
    config={'chunk_frames':4,'checkpoint_strategy':'none'};(tmp_path/'manifest.json').write_text('{}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record={'identity':{'config_sha256':'original_pre_probe_sha','manifest_sha256':sha(tmp_path/'manifest.json')},
            'frozen_execution_config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            'execution':{'chunk_frames':4,'checkpoint_strategy':'none'}}
    (tmp_path/'resource_smoke.json').write_text(json.dumps(record))
    assert benchmark(config,tmp_path,[],None)==record
    with pytest.raises(ValueError,match='IDENTITY_CHANGED'):benchmark({**config,'chunk_frames':8},tmp_path,[],None)
