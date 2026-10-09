from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research_tools.v8.local_geometry_pilot.model import LocalGeometryModel
from research_tools.v8.full_video_streaming.model import frame_at,StreamingModel
from research_tools.v8.full_video_streaming.validation import validate_window,validate_continuous
from research_tools.v8.full_video_streaming.inputs import tracker_summary,patch,decode,Frontend
from research_tools.v8.full_video_streaming.runner import checkpoint_block,process_video
from research_tools.v8.full_video_streaming.completion import finalize

torch.set_num_threads(2)
STATS={'absolute_mean':[0.,0.,0.],'absolute_scale':[1.,1.,1.],'relative_scale':[1.,1.,1.]}


def inputs(t=64):
    torch.manual_seed(44)
    content=torch.zeros(1,t,256,16,16,dtype=torch.bool);content[:,:,:3]=True;content[:,:,2,:,8:]=False
    valid=content.clone();valid[:,:,1]=False
    xyz=torch.randn(1,t,256,16,16,3);mu=(xyz*valid[...,None]).sum((3,4))/valid.sum((3,4)).clamp_min(1)[...,None]
    b={'x':torch.randn(1,t,256,77),'content':content,'valid':valid,'xyz':xyz,'mu':mu,
       'area':content.sum((3,4)).float(),'component_ids':torch.arange(256)[None,None].expand(1,t,256),
       'predecessor':torch.zeros(1,t,256,4,dtype=torch.long),'association_weight':torch.zeros(1,t,256,4),
       'history_available':content.any((3,4)),'delta_t':torch.full((1,t),.033)}
    b['predecessor'][:,:,0]=0;b['association_weight'][:,:,0,0]=1;b['delta_t'][:,0]=0
    return b


def test_window_chunk_restore_causality_and_area():
    b=inputs();m=LocalGeometryModel('MEAN_REPLICATED',STATS).eval()
    validate_window(m,{k:v[:,:16] for k,v in b.items()})
    r=validate_continuous(m,[frame_at(b,i) for i in range(64)])
    assert r['status']=='PASS' and max(r['comparisons']['16'].values())==0


def test_padding_missing_and_new_video_state():
    b=inputs(2);m=LocalGeometryModel('MEAN_REPLICATED',STATS).eval();s=StreamingModel(m)
    f=frame_at(b,0);first=s.forward_step(f);changed=copy.deepcopy(f);changed['x'][:,3:]=1e5;changed['xyz'][:,3:]=1e5
    other=s.forward_step(changed)
    torch.testing.assert_close(first['frame_logit'],other['frame_logit'],atol=1e-6,rtol=1e-6)
    assert torch.isfinite(first['cell_logits'][:,:3]).all()
    following=s.forward_step(frame_at(b,1),first['state']);reset=s.forward_step(frame_at(b,1))
    assert following['state']['frame_count']==2 and reset['state']['frame_count']==1
    assert not torch.equal(following['cell_logits'],reset['cell_logits'])


def test_tracker_axis_and_padding_predecessor():
    q=np.arange(1024);xy=np.stack((q%32*255/31,q//32*255/31),-1).astype(np.float32)
    tr={'tracks':np.tile(xy,(16,1,1)),'visibility':np.ones((16,1024),bool),'endpoint_padding_count':13,'query_frame_in_padded_prefix':15}
    active=np.ones(256,bool);prev=active.copy();prev[:16]=False
    cov,motion,mv,pi,aw,meta=tracker_summary(tr,[8,9,10],prev,active)
    assert meta['previous_output_index']==14 and meta['output_axis_frame_indices'][14:]==[9,10]
    assert aw[:16].sum()==0 and np.allclose(aw[16:].sum(-1),1) and cov.min()==1
    tr['tracks']=tr['tracks'][:15]
    with pytest.raises(ValueError,match='TIME_CONTRACT'):tracker_summary(tr,[8,9,10],prev,active)


def test_patch_coordinates_and_atomic_progress(tmp_path):
    image=np.arange(256*256).reshape(256,256);p=patch(image)
    assert p[17,0,0]==image[16,16] and p[255,15,15]==image[-1,-1]
    b=inputs(2);m=LocalGeometryModel('MEAN_REPLICATED',STATS).eval();o=StreamingModel(m).forward_step(frame_at(b,0))
    r={'frame_index':0,'pts_s':0.,'video_id':'a','video_sha256':'sha','cell_logits':o['cell_logits'][0].numpy(),'frame_logit':float(o['frame_logit'])}
    records=checkpoint_block(tmp_path,'id',o['state'],[],[r],{},[(0,None,0.)])
    c=torch.load(tmp_path/'state.pt',weights_only=False)
    assert c['next_frame_index']==1 and c['previous_pts_s']==0 and c['context_frame_indices']==[0] and len(records)==1
    assert c['video_id']=='a' and c['identity']=='id'


def test_provider_numeric_flags_restore_detector(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    f=Frontend.__new__(Frontend);f.cost={'geometry_s':0.}
    old=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32=False
        assert f.clock('geometry_s',lambda:torch.backends.cudnn.allow_tf32) is True
        assert torch.backends.cudnn.allow_tf32 is False
    finally: torch.backends.cudnn.allow_tf32=old


def test_resume_rejects_wrong_statistics_or_video(tmp_path):
    row={'video_id':'expected','video_sha256':'sha'}
    path=tmp_path/'responses'/'expected';path.mkdir(parents=True)
    for fields in ({'identity':'wrong-stat-config','video_id':'expected','video_sha256':'sha'},
                   {'identity':'id','video_id':'other-video','video_sha256':'sha'},
                   {'identity':'id','video_id':'expected','video_sha256':'wrong-sha'}):
        torch.save(fields,path/'state.pt')
        with pytest.raises(ValueError,match='STATE_IDENTITY_MISMATCH'):
            process_video(row,tmp_path,None,None,'id')


def test_completion_closes_progress_without_false_success(tmp_path):
    p=tmp_path/'final_status.json'
    s={'status':'RUNNING'};p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='INCOMPLETE'):finalize(tmp_path)
    s={'status':'COMPLETE','completion_scope':'two_video_engineering_pilot','completed_video_count':2,
       'decoded_frame_count':1828,'frontend_frame_count':1828,'response_frame_count':1828}
    p.write_text(json.dumps(s));finalize(tmp_path)
    assert json.loads(p.read_text())['training_started'] is False
    assert json.loads((tmp_path/'progress.json').read_text())['status']=='COMPLETE'
