"""Only fixed identity, output isolation and full-B initialization route."""
import json
from pathlib import Path
import pytest
import torch
from research_tools.rgbd_slice import b_fit_check as diagnostic


def fixture_source(tmp_path):
    src=tmp_path/'formal';(src/'fit').mkdir(parents=True)
    c={'output':str(src),'seed':20260909,'fit_clear_dice':.6,'fit_clear_recall':.6,'encoder_lr':1e-5,'new_lr':1e-4,'weight_decay':.01,'fit_updates':150,'fit_seconds':2700,'fit_goal_dice':.8,'fit_goal_real_fpr':.05}
    rows=[{'video_id':r,'role':r,'split':'train','pair_id':'fixed','pts':[{'pts_s':i*.04} for i in range(24)]} for r in ['fake','real']]
    clips=[{'video_id':r['video_id'],'start':0,'end':24} for r in rows]
    for name,value in [('config.json',c),('manifest.json',{'rows':rows,'clips':clips}),('fit/result.json',{'pair':'fixed','clips':clips,'updates':10}),('fit/history.json',[{'update':i,**({'monitor':{'dice':.1}} if i==10 else {})} for i in range(1,11)])]:
        (src/name).write_text(json.dumps(value))
    return src


def test_same_saved_clips_and_separate_outputs(tmp_path):
    src=fixture_source(tmp_path);*_,rows,clips,protocol=diagnostic.inputs(src)
    assert [r['role'] for r in rows]==['fake','real'] and protocol['configuration_difference']==[]
    diagnostic.separate_output(src,tmp_path/'diagnostic')
    with pytest.raises(ValueError):diagnostic.separate_output(src,src/'fit/new')
    a=json.loads((src/'fit/result.json').read_text());a['clips'][0]['end']=23;(src/'fit/result.json').write_text(json.dumps(a))
    with pytest.raises(ValueError,match='CLIP_MISMATCH'):diagnostic.inputs(src)


def test_complete_b_calls_original_common_initializer(tmp_path,monkeypatch):
    class B(torch.nn.Module):
        def __init__(self):
            super().__init__();self.local=torch.nn.Linear(2,2);self.relation=True
            self.pool=torch.nn.Identity();self.time=torch.nn.ModuleList();self.depth=torch.nn.ModuleList()
            self.pixel_context=torch.nn.Linear(2,2);self.frame_context=torch.nn.Linear(2,1)
            for m in [self.pixel_context,self.frame_context]:torch.nn.init.zeros_(m.weight);torch.nn.init.zeros_(m.bias)
    b=B();(tmp_path/'initialization.json').write_text(json.dumps({'common_sha256':diagnostic.state_hash(b.local.state_dict())}));calls=[]
    def make(config,source,condition):calls.append((source,condition));return b
    monkeypatch.setattr(diagnostic,'make',make)
    assert diagnostic.complete_b({},tmp_path) is b
    assert calls==[(tmp_path,'RGBD_SLICE_RELATION')]
    with torch.no_grad():b.pixel_context.weight.fill_(1)
    with pytest.raises(ValueError,match='ZERO_INITIALIZATION'):diagnostic.complete_b({},tmp_path)
