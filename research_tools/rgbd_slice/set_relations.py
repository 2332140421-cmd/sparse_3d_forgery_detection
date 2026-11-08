"""Overlapping induced role sets for the two-pair fitting question (ADR-0036).

RGBDSetSlices is the sole model caller. Outputs preserve each current slab and
provide a relation-only frame logit. PairRelation cannot express unrestricted
overlapping sets, so this small independent module replaces only that path.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SetRelations(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.depth_position = nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,512))
        self.time_position = nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,512))
        self.units = nn.Parameter(torch.randn(8,512)*.02)
        self.participation = nn.ModuleList([nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Linear(128,1)) for _ in range(2)])
        self.queries = nn.Parameter(torch.randn(8,2,4,512)*.02)
        self.empty = nn.Parameter(torch.randn(1,512)*.02)
        self.read_members = nn.MultiheadAttention(512,8,dropout=.1,batch_first=True)
        self.roles = nn.Parameter(torch.randn(2,512)*.02)
        self.interaction = nn.ModuleList([nn.TransformerEncoderLayer(512,8,2048,.1,batch_first=True,norm_first=True) for _ in range(2)])
        self.unit_readout = nn.MultiheadAttention(512,8,dropout=.1,batch_first=True)
        self.current_readout = nn.MultiheadAttention(512,8,dropout=.1,batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1,1,512)*.02)
        self.global_readout = nn.MultiheadAttention(512,8,dropout=.1,batch_first=True)
        self.delta_time = nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,512))
        self.gru = nn.GRUCell(512,512)
        self.frame_head = nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
        self.zero_messages = False
        self.capture = False
        self.observations = []

    def at_time(self,c,depth,dt,current):
        x = self.norm(c)+self.depth_position(depth[:,None])+self.time_position(dt[:,None])
        # Independent sigmoid over (element, unit, role); no unit competition.
        membership = torch.stack([p(x[:,None]+self.units[None]).squeeze(-1).sigmoid() for p in self.participation],-1)
        bank = torch.cat((x,self.empty),0)[None].expand(16,-1,-1)
        bias = membership.permute(1,2,0).reshape(16,-1).clamp_min(1e-6).log()
        bias = torch.cat((bias,bias.new_zeros(16,1)),1)
        bias = bias[:,None,None].expand(-1,8,4,-1).reshape(128,4,-1)
        q = self.queries.reshape(16,4,512)
        states,weights = self.read_members(q,bank,bank,attn_mask=bias,need_weights=self.capture)
        states = (states+q).reshape(8,2,4,512)+self.roles[None,:,None]
        states = states.reshape(8,8,512)
        for block in self.interaction:
            states = block(states)
        z,_ = self.unit_readout(self.units[:,None],states,states,need_weights=False)
        z = z[:,0]
        current_x = x[current][None]
        message,_ = self.current_readout(current_x,states.reshape(1,64,512),states.reshape(1,64,512),need_weights=False)
        g,_ = self.global_readout(self.global_query,z[None],z[None],need_weights=False)
        if self.capture:
            # Only small summaries/first elements, never retain GPU graphs.
            w=weights.detach().float().cpu();a=membership.detach().float().cpu();zz=z.detach().float().cpu()
            cosine=torch.nn.functional.cosine_similarity(zz[:,None],zz[None],dim=-1)
            self.observations.append({'tokens':len(c),'membership_mean':a.mean((0,2)).tolist(),
                'overlap_mean_sum_over_units':a.sum(1).mean().item(),'A_B_overlap_mean':(a[:,:,0]*a[:,:,1]).mean().item(),
                'membership_first16':a[:16].tolist(),'read_max_weight_mean':w.max(-1).values.mean().item(),
                'empty_weight_mean':w[:,:,-1].mean().item(),'unit_cosine_offdiag_mean':cosine[~torch.eye(8,dtype=torch.bool)].mean().item(),
                'states_first4_channels':states.detach().float().cpu()[:,:,:4].tolist()})
        if self.zero_messages:
            message=torch.zeros_like(message);g=torch.zeros_like(g)
        return c[current]+message[0],g[0,0]

    def forward(self,encoded,pts):
        counts=[len(c) for c,_ in encoded]
        tokens=torch.cat([c for c,_ in encoded])
        depth=torch.cat([ids for _,ids in encoded]).to(tokens.dtype)*.125
        times=torch.cat([tokens.new_full((n,),p) for n,p in zip(counts,pts)])
        contexts=[];logits=[];lo=0;hidden=tokens.new_zeros(1,512)
        for t,(n,p) in enumerate(zip(counts,pts)):
            hi=lo+n;current=torch.arange(lo,hi,device=tokens.device)
            args=(tokens[:hi],depth[:hi],times[:hi]-p,current)
            context,g=checkpoint(self.at_time,*args,use_reentrant=False,preserve_rng_state=True) if self.training else self.at_time(*args)
            delta=tokens.new_tensor([[0. if t==0 else p-pts[t-1]]])
            hidden=self.gru((g[None]+self.delta_time(delta)).to(hidden.dtype),hidden)
            contexts.append(context);logits.append(self.frame_head(hidden)[0,0]);lo=hi
        return contexts,logits
