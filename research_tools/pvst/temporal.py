"""All occupied27-neighborhoods at t,t-d,t-2d, true PTS bias, d=1/2/4."""
import itertools
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .geometry import LayerNorm


def neighbors(coords,dilation,pts,chunk=2048):
    """Exact injective integer keys and searchsorted; no truncation or empty nodes."""
    device=coords[0].device;lengths=[len(c) for c in coords];offset=[0]
    for n in lengths:offset.append(offset[-1]+n)
    if not offset[-1]:return torch.empty(0,81,dtype=torch.long,device=device),torch.empty(0,81,4,device=device)
    allc=torch.cat(coords);lo=allc.amin(0)-1;span=allc.amax(0)-lo+2
    volume=int(span.prod())
    if volume*len(coords)>=2**62:raise ValueError('EXACT_NEIGHBOR_KEY_OVERFLOW')
    def key(c,t):
        a=c-lo;return ((a[...,0]*span[1]+a[...,1])*span[2]+a[...,2])+t*volume
    times=torch.repeat_interleave(torch.arange(len(coords),device=device),torch.tensor(lengths,device=device))
    keys=key(allc,times);sorted_keys,order=keys.sort();offsets=torch.tensor(list(itertools.product([-1,0,1],repeat=3)),device=device)
    offsets=offsets.repeat(3,1);lag=torch.tensor([0]*27+[dilation]*27+[2*dilation]*27,device=device)
    indices=[];relative=[]
    for ti,c in enumerate(coords):
        for start in range(0,len(c),chunk):
            base=c[start:start+chunk];candidate=base[:,None]+offsets[None];tt=ti-lag[None].expand(len(base),-1)
            wanted=key(candidate,tt);pos=torch.searchsorted(sorted_keys,wanted);safe=pos.clamp(max=len(sorted_keys)-1)
            inside=((candidate>=lo)&(candidate<lo+span)).all(-1)
            valid=(tt>=0)&inside&(pos<len(sorted_keys))&(sorted_keys[safe]==wanted)
            idx=torch.where(valid,order[safe],-1)
            dt=torch.tensor([pts[max(0,ti-int(v))]-pts[ti] for v in lag.cpu().tolist()],device=device,dtype=torch.float32)
            rel=torch.cat((offsets.float()[None].expand(len(base),-1,-1),dt[None,:,None].expand(len(base),-1,1)),-1)
            indices.append(idx);relative.append(rel)
    return torch.cat(indices),torch.cat(relative)


class SparseLayer(nn.Module):
    def __init__(self,dim,heads,chunk=128,ffn_chunk=8192):
        super().__init__();self.dim=dim;self.heads=heads;self.chunk=chunk;self.ffn_chunk=ffn_chunk;self.ffn_checkpoint=True;self.last_execution={}
        if ffn_chunk<=0:raise ValueError("FFN_CHUNK_MUST_BE_POSITIVE")
        self.norm=LayerNorm(dim);self.qkv=nn.Linear(dim,dim*3);self.bias=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,heads))
        self.proj=nn.Linear(dim,dim);self.drop=nn.Dropout(.1);self.norm2=LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Dropout(.1),nn.Linear(4*dim,dim),nn.Dropout(.1))
    def attend(self,q,k,v,index,rel):
        self.last_execution={'operation':'sparse_attention','query_shape':list(q.shape),'key_shape':list(k.shape),'dtype':str(q.dtype)}
        valid=index>=0;idx=index.clamp_min(0)
        kk=k[idx].permute(0,2,1,3);vv=v[idx].permute(0,2,1,3)
        bias=self.bias(rel.to(q.dtype)).permute(0,2,1)[:,:,None,:]
        bias=bias.masked_fill(~valid[:,None,None,:],-torch.inf)
        result=F.scaled_dot_product_attention(q[:,:,None,:],kk,vv,attn_mask=bias,
            dropout_p=self.drop.p if self.drop.training else 0.)
        return result[:,:,0].reshape(len(q),self.dim)
    def ffn_block(self,x):
        self.last_execution={'operation':'norm2+FFN block','input_shape':list(x.shape),'hidden_shape':[len(x),4*self.dim],'dtype':str(x.dtype)}
        return self.ffn(self.norm2(x))
    def forward(self,x,index,relative):
        if not len(x):return x
        self.last_execution={'operation':'norm+QKV','input_shape':list(x.shape),'dtype':str(x.dtype)}
        q,k,v=self.qkv(self.norm(x)).reshape(len(x),3,self.heads,64).unbind(1);values=[]
        for start in range(0,len(x),self.chunk):
            args=(q[start:start+self.chunk],k,v,index[start:start+self.chunk],relative[start:start+self.chunk])
            y=checkpoint(self.attend,*args,use_reentrant=False) if self.training else self.attend(*args)
            values.append(y)
        x=x+self.drop(self.proj(torch.cat(values)))
        values=[]
        self.last_execution={'operation':'norm2+FFN','input_shape':list(x.shape),'dtype':str(x.dtype),
            'ffn_chunk':self.ffn_chunk,'hidden_shape_upper_bound':[min(len(x),self.ffn_chunk),4*self.dim]}
        for start in range(0,len(x),self.ffn_chunk):
            # Bound method, not a closure over a loop slice; norm+hidden stay local.
            part=x[start:start+self.ffn_chunk]
            y=checkpoint(self.ffn_block,part,use_reentrant=False,preserve_rng_state=True) if self.training and self.ffn_checkpoint else self.ffn_block(part)
            values.append(y)
        return x+torch.cat(values)


class Temporal(nn.Module):
    def __init__(self,dim,heads,chunk=128,ffn_chunk=8192):
        super().__init__();self.layers=nn.ModuleList([SparseLayer(dim,heads,chunk,ffn_chunk) for _ in range(3)])
    def forward(self,features,coords,pts):
        lengths=[len(a) for a in features];x=torch.cat(features);self.last_stats=[];self.active_layer=None
        for dilation,layer in zip([1,2,4],self.layers):
            index,rel=neighbors(coords,dilation,pts)
            degree=(index>=0).sum(-1)
            self.last_stats.append({'dilation':dilation,'nodes':len(x),'edges':int(degree.sum()),
                'neighbors_min':int(degree.min()) if len(degree) else 0,'neighbors_mean':float(degree.float().mean()) if len(degree) else 0.,
                'neighbors_max':int(degree.max()) if len(degree) else 0})
            self.active_layer={"dilation":dilation,"nodes":len(x),"width":layer.dim,"dtype":str(x.dtype)}
            # Bind layer and graph now; backward must not close over next iteration.
            def apply(value,idx,relationship,mod=layer,d=dilation):
                self.active_layer={'dilation':d,'nodes':len(value),'width':mod.dim,'dtype':str(value.dtype)}
                return mod(value,idx,relationship)
            x=checkpoint(apply,x,index,rel,use_reentrant=False) if self.training else apply(x,index,rel)
        return list(x.split(lengths))
