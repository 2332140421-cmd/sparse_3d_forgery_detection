"""Experiment-only frozen pooling and current-frame readout helpers.

Caller: slab_grid_experiment; outputs C512 or unit logits/metrics. The existing
slab-only pool and full-query MHA cannot express this matched grouping probe.
No old default/model parameters are changed; readout state keys stay identical.
"""
from contextlib import nullcontext
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
from .model import native_sample
from .slice_relation_readout import SliceRelationReadout
from research_tools.depth_slice.model import slab_ids

CONDITIONS=('SLAB_CURRENT','SLAB_GRID_CURRENT')


def grouping(xyz,valid):
    """Exactly the oracle native top-left grid, without reading GT."""
    ids,slab=slab_ids(xyz,valid);h,w=valid.shape;L=math.ceil(min(h,w)/16)
    points=torch.where(valid.reshape(-1))[0];ny=(h+L-1)//L;nx=(w+L-1)//L
    key=slab*(ny*nx)+(points//w//L)*nx+(points%w//L)
    unique,grid=torch.unique(key,sorted=True,return_inverse=True)
    assert torch.equal(unique[grid]//(ny*nx),slab)
    return points,[(ids,slab),(ids[unique//(ny*nx)],grid)],L


@torch.no_grad()
def frozen_pool(pool,local,points,groups,hw,chunk=8192):
    """Global per-unit/head softmax; stream value accumulation, no N*512 list.

    score/value are the very same frozen weights for both groups. Only routing
    changes. No per-chunk softmax; score maxima/denominators cover whole units.
    """
    assert pool.rgb_only
    scores=[]
    for p in points.split(chunk):
        visual=native_sample(local,p,hw)
        scores.append(pool.score(torch.cat((visual,visual.new_zeros(len(p),3)),-1)).float())
    if not len(points):return [local.new_empty(0,512,dtype=torch.float32) for _ in groups]
    score=torch.cat(scores);weights=[];outputs=[]
    for ids,index in groups:
        maximum=score.new_full((len(ids),4),-torch.inf).scatter_reduce(0,index[:,None].expand(-1,4),score,reduce='amax',include_self=True)
        a=(score-maximum[index]).exp();denom=a.new_zeros(len(ids),4).index_add(0,index,a)
        weights.append(a/denom[index]);outputs.append(score.new_zeros(len(ids),512))
    offset=0
    for p in points.split(chunk):
        value=pool.value(native_sample(local,p,hw)).float().reshape(-1,4,128)
        for j,(_,index) in enumerate(groups):
            weighted=(value*weights[j][offset:offset+len(p),:,None]).reshape(-1,512)
            partial=outputs[j].new_zeros(outputs[j].shape).index_add(0,index[offset:offset+len(p)],weighted)
            outputs[j].add_(partial)
        offset+=len(p)
    return outputs


class CurrentReadout(SliceRelationReadout):
    """Same learned parameters as CURRENT_ONLY; query blocks read ALL keys."""
    def __init__(self,chunk=512,use_checkpoint=True):
        super().__init__();self.chunk=chunk;self.use_checkpoint=use_checkpoint
    @staticmethod
    def attend(q,k,v,start):
        # SDPA True=allowed, whereas MHA True=blocked. Only current self excluded.
        allow=torch.arange(k.shape[-2],device=q.device)[None,:]!=torch.arange(start,start+q.shape[-2],device=q.device)[:,None]
        context=sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if q.is_cuda else nullcontext()
        with context:return F.scaled_dot_product_attention(q,k,v,attn_mask=allow,dropout_p=0.)
    def forward(self,features,ids):
        n=len(features)
        if not n:return features.new_empty(0)
        if n==1:reference=self.empty_reference
        else:
            p=self.depth_position(ids.float()[:,None]*.125)
            query=features+p;key=features+p+self.time_position(features.new_zeros(n,1))
            wq,wk,wv=self.attention.in_proj_weight.chunk(3);bq,bk,bv=self.attention.in_proj_bias.chunk(3)
            def heads(x):return x.reshape(n,8,64).transpose(0,1)[None]
            q=heads(F.linear(query,wq,bq));k=heads(F.linear(key,wk,bk));v=heads(F.linear(features,wv,bv))
            pieces=[];start=0
            for block in q.split(self.chunk,dim=-2):
                # Bind start to this block. Non-reentrant checkpoint recomputes
                # only this block/mask; no full N*N weights or masks are retained.
                def run(a,b,c,offset=start):return self.attend(a,b,c,offset)
                out=checkpoint(run,block,k,v,use_reentrant=False,preserve_rng_state=True) if self.training and self.use_checkpoint else run(block,k,v)
                pieces.append(out);start+=block.shape[-2]
            attended=torch.cat(pieces,dim=-2)[0].transpose(0,1).reshape(n,512)
            reference=F.linear(attended,self.attention.out_proj.weight,self.attention.out_proj.bias)
        a=self.current_norm(features);b=self.reference_norm(reference)
        return self.compare(torch.cat((a-b,a*b),-1))[:,0]


def count_loss(z,n,g,positive_weight,denominator):
    return (positive_weight*g*F.softplus(-z)+(n-g)*F.softplus(z)).sum()/denominator


def count_metrics(z,n,g):
    """Exact tie-grouped native valid-pixel AP and logit>=0 overlap counts."""
    z=np.asarray(z);n=np.asarray(n,dtype=np.int64);g=np.asarray(g,dtype=np.int64)
    assert z.shape==n.shape==g.shape and np.isfinite(z).all() and np.all((0<=g)&(g<=n))
    G=int(g.sum());N=int(n.sum());positive=z>=0;TP=int(g[positive].sum());P=int(n[positive].sum())
    ap=None
    if G:
        order=np.argsort(-z,kind='stable');sorted_z=z[order];ends=np.flatnonzero(np.r_[sorted_z[1:]!=sorted_z[:-1],True]);tp=np.cumsum(g[order])[ends];area=np.cumsum(n[order])[ends]
        ap=float((np.diff(np.r_[0,tp])/G*(tp/area)).sum())
    return {'pixel_AP':ap,'Dice':2*TP/(G+P) if G else None,'IoU':TP/(G+P-TP) if G else None,'GT_recall':TP/G if G else None,'FPR':P/N if not G and N else None,'prediction_area':P/N if N else None,'valid_pixels':N,'valid_GT_pixels':G}
