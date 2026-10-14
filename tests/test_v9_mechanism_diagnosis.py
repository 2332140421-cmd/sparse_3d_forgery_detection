import torch
from research_tools.v9.dense_learned_regions.model import DenseRegions,normalized_lse
from research_tools.v9.dense_learned_regions.diagnose import intervene_block,loss_summary
from research_tools.v9.dense_learned_regions.data import save,read_json,digest


def test_reset_only_state_and_zero_context_at_head():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=DenseRegions('XYZ_PARTITION').eval()
    b={'rgb':torch.rand(3,3,8,8),'visual':torch.rand(3,64,2,2),'xyz':torch.rand(3,3,8,8),
       'content':torch.ones(3,1,8,8,dtype=torch.bool),'valid':torch.ones(3,1,8,8,dtype=torch.bool),'dt':torch.tensor([0.,.033,.034])}
    captured=[];hook=model.head.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
    with torch.no_grad():reset,zero,f=intervene_block(model,b);reference,_=model.sequence(b)
    hook.remove()
    assert torch.allclose(reset,reference,atol=1e-6)
    zero_input=captured[1]
    assert torch.equal(zero_input[...,:256],f) and not zero_input[...,256:].any()
    with torch.no_grad():manual=normalized_lse(model.head(torch.cat((f,torch.zeros_like(f)),-1))[:,:,0],b['content'].flatten(1).float(),1)
    assert torch.equal(zero,manual)
    hidden_dtypes=[]
    h=model.gru.register_forward_pre_hook(lambda m,a:hidden_dtypes.append(a[1].dtype))
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        _,_,f_amp=intervene_block(model,b)
    h.remove()
    assert f_amp.dtype==torch.bfloat16 and hidden_dtypes[0]==b['rgb'].dtype==torch.float32


def test_loss_record_parent_normalization():
    rows=[{'epoch':i,'loss':1.,'clip_bce_parent_sum':1.2,'frame_bce_parent_sum':1.6} for i in range(1,21)]
    s=loss_summary({'history':rows})
    assert s['first']['clip_bce_parent_mean']==.6 and s['first']['frame_bce_parent_mean']==.8
    assert s['epoch16_to20_total_change']==0


def test_original_boolean_metadata_hash_roundtrip(tmp_path):
    stats={'mean':[.1],'scale':[.2],'unique_frames_only':True}
    path=tmp_path/'stats.json';save(path,stats);loaded=read_json(path)
    assert loaded['unique_frames_only']==1
    assert digest({**loaded,'unique_frames_only':True})==digest(stats)
    assert digest({**loaded,'mean':[.11],'unique_frames_only':True})!=digest(stats)
