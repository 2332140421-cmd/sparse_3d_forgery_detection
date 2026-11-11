"""A single shared cross-attention readout; only candidate visibility differs.

Immediate caller: slice_relation_experiment. Returns one logit per current slab.
No old relation/GRU/pixel predictions or label-derived numeric inputs.
"""
import torch
from torch import nn

CONDITIONS=('CURRENT_ONLY','CURRENT_HISTORY')


def candidate_bank(frames,t,history):
    visible=frames[:t+1] if history else frames[t:t+1]
    c=torch.cat([f['features'] for f in visible]);ids=torch.cat([f['ids'] for f in visible])
    dt=torch.cat([c.new_full((len(f['ids']),1),f['pts']-frames[t]['pts']) for f in visible])
    n=len(frames[t]['ids']);offset=len(c)-n
    # Identity is (frame, slab slot), not absolute slab number across time.
    blocked=torch.zeros(n,len(c),device=c.device,dtype=torch.bool)
    blocked[torch.arange(n,device=c.device),offset+torch.arange(n,device=c.device)]=True
    return c,ids,dt,blocked


class SliceRelationReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_position=nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,512))
        self.time_position=nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,512))
        self.attention=nn.MultiheadAttention(512,8,dropout=0.,batch_first=True)
        self.empty_reference=nn.Parameter(torch.randn(1,512)*.02)
        self.current_norm=nn.LayerNorm(512);self.reference_norm=nn.LayerNorm(512)
        self.compare=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,frames,t,history):
        current=frames[t]['features'];n=len(current)
        if not n:return current.new_empty(0),{'queries':0,'candidate_edges':0,'empty_queries':0}
        bank,ids,dt,blocked=candidate_bank(frames,t,history)
        empty=~(~blocked).any(1)
        q=current+self.depth_position(frames[t]['ids'].float()[:,None]*.125)
        # Every nonempty query has all original visible, non-self candidates.
        if empty.any():
            assert n==1 and len(bank)==1
            reference=self.empty_reference.expand(n,-1)
        else:
            k=bank+self.depth_position(ids.float()[:,None]*.125)+self.time_position(dt)
            reference,_=self.attention(q[None],k[None],bank[None],attn_mask=blocked,need_weights=False)
            reference=reference[0]
        a=self.current_norm(current);b=self.reference_norm(reference)
        z=self.compare(torch.cat((a-b,a*b),-1))[:,0]
        return z,{'queries':n,'candidate_edges':int((~blocked).sum()),'empty_queries':int(empty.sum())}


def project_gt(xyz,valid,gt):
    from research_tools.depth_slice.model import slab_ids
    ids,index=slab_ids(xyz,valid);counts=torch.bincount(index,minlength=len(ids))
    edits=torch.bincount(index[gt[valid]],minlength=len(ids))
    assert int(counts.sum())==int(valid.sum()) and int(edits.sum())==int((gt&valid).sum())
    return ids,index,counts,edits


def windows(row,clips=None):
    n=row['actual_frames']
    return [(x['start'],x['end'],x['start']) for x in clips if x['video_id']==row['video_id']] if clips is not None else [(s,min(s+24,n),max(0,n-24) if s+24>=n else s) for s in range(0,n,24)]
