import numpy as np
import pytest
import torch

from research_tools.v8.full_video_training.temporal import ROOT, exact_annotation, interval_labels, frame_logits, frame_loss


def test_exact_video_not_pair_fallback_and_native_pts_union():
    raw=ROOT/'datasets/v7_core_candidates/activityforensics_charades_v1/source/activityforensics/raw'
    row={'source_id':'a','video_id':'id','role':'fake','video_path':str(raw/'video/a.mp4')}
    entry={'charades_source_id':'a','activityforensics_file':'video/a.mp4','lineage_status':'EXACT',
           'manipulation_intervals':[{'start_s':.2,'end_s':.4},{'start_s':.3,'end_s':.8}]}
    assert exact_annotation(row,[{**entry,'charades_source_id':'b'},entry])==entry
    with pytest.raises(ValueError,match='ANNOTATION_IDENTITY'):
        exact_annotation({**row,'source_id':'missing'},[entry])
    pts=np.array([0.,.199,.2,.35,.8,.801])
    assert interval_labels(pts,entry['manipulation_intervals']).tolist()==[False,False,True,True,True,False]


def test_frame_area_lse_padding_gradient_and_balanced_video_loss():
    c=torch.tensor([[1.,3.,999.],[2.,-1.,-999.]],requires_grad=True)
    area=torch.tensor([[1.,3.,0.],[2.,2.,0.]])
    z=frame_logits(c,area)
    direct=torch.stack([torch.logsumexp(c[i,:2].double()+area[i,:2].double().log(),0)-area[i].sum().double().log() for i in range(2)])
    torch.testing.assert_close(z,direct)
    loss=frame_loss(z,np.array([False,True]));loss.backward()
    assert torch.isfinite(c.grad).all() and c.grad[:,:2].abs().sum()>0 and c.grad[:,2].abs().sum()==0
    assert frame_loss(z.detach(),np.array([False,False]))==torch.nn.functional.binary_cross_entropy_with_logits(z.detach(),torch.zeros_like(z))


def test_one_seed_reuses_source_pair_loop_and_last_epoch(tmp_path,monkeypatch):
    from research_tools.v8.full_video_training import run
    monkeypatch.setattr(run,'load_model',lambda *a:(torch.nn.Linear(1,1),{'checkpoint_sha256':'frozen'}))
    monkeypatch.setattr(run,'STOP',False)
    called=[]
    def step(m,row,*a):
        called.append(row['video_id']);loss=m(torch.ones(1,1)).square().mean()*.5;loss.backward()
        return {'loss':float(loss.detach())}
    rows=[{'source_id':'a','split':'train','role':role,'video_id':role} for role in ('real','fake')]
    r=run.train(tmp_path,tmp_path,rows,'cache',64,'id',lambda *a,**k:None,False,
                conditions=('GLOBAL_COMPONENT',),epochs=5,seeds=(20260909,),train_step=step)
    assert len(r)==1 and r[0]['epoch']==5 and len(called)==10 and len(r[0]['loss_history'])==5


def test_cached_evaluation_creates_missing_directory_without_forward(tmp_path,monkeypatch):
    from research_tools.v8.full_video_training import temporal as t
    model=torch.nn.Linear(1,1)
    cp={'epoch':5,'identity':{'seed':20260909},'model':model.state_dict()}
    checkpoint=tmp_path/'model.pt';torch.save(cp,checkpoint)
    record={'checkpoint':str(checkpoint),'checkpoint_sha256':t.sha256_file(checkpoint),'identity':cp['identity']}
    rows=[{'video_id':role,'source_id':'sample','split':'validation','role':role,'video_label':int(role=='fake'),
           'video_sha256':role*8} for role in ('real','fake')]
    labels={'real':np.array([False,False]),'fake':np.array([False,True])}
    timing={r['video_id']:np.array([0.,1.]) for r in rows}
    for name in ('VIDEO_ONLY','TEMPORAL'):
        directory=tmp_path/'responses'/name;directory.mkdir(parents=True)
        for row in rows:
            p=directory/f'{row["video_id"]}.npz'
            np.savez(p,video_logit=np.array(1. if row['role']=='fake' else -1.),frame_logits=np.array([-1.,1.] if row['role']=='fake' else [-2.,-1.]),
                     pts_s=timing[row['video_id']],labels=labels[row['video_id']])
            t.save_json(p.with_suffix('.json'),{'checkpoint_sha256':record['checkpoint_sha256'],'video_sha256':row['video_sha256'],'sha256':t.sha256_file(p)})
    t.save_json(tmp_path/'annotation_details.json',{'fake':{'intervals':[{'start_s':1.,'end_s':1.}]}})
    read=t.read_json
    old_scores={r['video_id']:{'video_logits':{'GLOBAL_COMPONENT__seed20260909':1. if r['role']=='fake' else -1.}} for r in rows}
    monkeypatch.setattr(t,'read_json',lambda p:old_scores if p==t.OLD/'scores.json' else read(p))
    monkeypatch.setattr(t.training,'load_model',lambda *a:(torch.nn.Linear(1,1),None))
    monkeypatch.setattr(t,'load_video',lambda *a:pytest.fail('legal cached response must not run forward'))
    assert not (tmp_path/'evaluation').exists()
    result=t.evaluate(tmp_path,rows,labels,timing,{'cache_identity':'unused'},record,record,lambda *a,**k:None)
    assert (tmp_path/'evaluation/summary.json').is_file()
    assert result['TEMPORAL']['frame']['eligible_fake_videos']==1
