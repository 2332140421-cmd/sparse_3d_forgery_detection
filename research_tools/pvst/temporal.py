"""All occupied27-neighborhoods at t,t-d,t-2d, true PTS bias, d=1/2/4."""
import itertools
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .geometry import LayerNorm


class Neighborhood:
    """Compact deterministic topology shared by three layers; no expanded graph.

    Immediate caller Temporal.forward. A query chunk regenerates its exact81
    indices; only [T,81,4] relations are retained rather than [N,81,4].
    """
    def __init__(self,coords,pts):
        self.lengths=[len(c) for c in coords];self.coords=torch.cat(coords);device=self.coords.device
        boundaries=[0]
        for n in self.lengths:boundaries.append(boundaries[-1]+n)
        self.boundaries=torch.tensor(boundaries[1:],device=device)
        self.offsets=torch.tensor(list(itertools.product([-1,0,1],repeat=3)),device=device).repeat(3,1)
        self.lags={d:torch.tensor([0]*27+[d]*27+[2*d]*27,device=device) for d in [1,2,4]}
        self.relative={}
        for d in [1,2,4]:
            lag=[0]*27+[d]*27+[2*d]*27
            dt=torch.tensor([[pts[max(0,t-v)]-pts[t] for v in lag] for t in range(len(pts))],device=device,dtype=torch.float32)
            self.relative[d]=torch.cat((self.offsets.float()[None].expand(len(pts),-1,-1),dt[:,:,None]),-1)
        if not len(self.coords):return
        self.lo=self.coords.amin(0)-1;self.span=self.coords.amax(0)-self.lo+2;self.volume=int(self.span.prod())
        if self.volume*len(coords)>=2**62:raise ValueError('EXACT_NEIGHBOR_KEY_OVERFLOW')
        keys=torch.cat([self.key(c,t) for t,c in enumerate(coords)])
        self.sorted_keys,self.order=keys.sort()
    def key(self,c,t):
        a=c-self.lo;return ((a[...,0]*self.span[1]+a[...,1])*self.span[2]+a[...,2])+t*self.volume
    def query(self,start,end,dilation):
        frame=torch.bucketize(torch.arange(start,end,device=self.coords.device),self.boundaries,right=True)
        candidate=self.coords[start:end,None]+self.offsets[None]
        time=frame[:,None]-self.lags[dilation][None]
        wanted=self.key(candidate,time);pos=torch.searchsorted(self.sorted_keys,wanted);safe=pos.clamp(max=len(self.sorted_keys)-1)
        inside=((candidate>=self.lo)&(candidate<self.lo+self.span)).all(-1)
        valid=(time>=0)&inside&(pos<len(self.sorted_keys))&(self.sorted_keys[safe]==wanted)
        return torch.where(valid,self.order[safe],-1),frame


class SparseLayer(nn.Module):
    def __init__(self,dim,heads,chunk=128,ffn_chunk=8192):
        super().__init__();self.dim=dim;self.heads=heads;self.chunk=chunk;self.ffn_chunk=ffn_chunk;self.qkv_chunk=0;self.sparse_edge_attention=False;self.ffn_checkpoint=True;self.last_execution={}
        if ffn_chunk<=0:raise ValueError("FFN_CHUNK_MUST_BE_POSITIVE")
        self.norm=LayerNorm(dim);self.qkv=nn.Linear(dim,dim*3);self.bias=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,heads))
        self.proj=nn.Linear(dim,dim);self.drop=nn.Dropout(.1);self.norm2=LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Dropout(.1),nn.Linear(4*dim,dim),nn.Dropout(.1))
    def attend(self,q,k,v,index,bias):
        self.last_execution={'operation':'sparse_attention','query_shape':list(q.shape),'key_shape':list(k.shape),'dtype':str(q.dtype)}
        valid=index>=0;idx=index.clamp_min(0)
        kk=k[idx].permute(0,2,1,3);vv=v[idx].permute(0,2,1,3)
        bias=bias.permute(0,2,1)[:,:,None,:]
        bias=bias.masked_fill(~valid[:,None,None,:],-torch.inf)
        result=F.scaled_dot_product_attention(q[:,:,None,:],kk,vv,attn_mask=bias,
            dropout_p=self.drop.p if self.drop.training else 0.)
        return result[:,:,0].reshape(len(q),self.dim)
    def attention_block(self,residual,q,k,v,bias,graph,start,end,dilation):
        # These large indices only exist within the checkpoint body/recomputation.
        with torch.profiler.record_function('pvst.neighbor_query'):
            index,frame=graph.query(start,end,dilation)
        attended=self.attend(q,k,v,index,bias[frame])
        self.last_execution={'operation':'attention projection+dropout+residual block','input_shape':list(attended.shape),'dtype':str(attended.dtype)}
        return residual+self.drop(self.proj(attended))
    def project_block(self,x,attended):
        return x+self.drop(self.proj(attended))
    def ffn_block(self,x):
        self.last_execution={'operation':'norm2+FFN block','input_shape':list(x.shape),'hidden_shape':[len(x),4*self.dim],'dtype':str(x.dtype)}
        return x+self.ffn(self.norm2(x))
    def norm_qkv(self,x):
        return self.qkv(self.norm(x))
    def forward(self,x,graph,dilation):
        if not len(x):return x
        self.last_execution={'operation':'norm+QKV','input_shape':list(x.shape),'dtype':str(x.dtype)}
        if self.qkv_chunk:
            parts=[]
            for start in range(0,len(x),self.qkv_chunk):
                part=x[start:start+self.qkv_chunk]
                value=checkpoint(self.norm_qkv,part,use_reentrant=False,preserve_rng_state=True) if self.training else self.norm_qkv(part)
                parts.append(value)
            projected=torch.cat(parts);del parts,value
        else:projected=self.norm_qkv(x)  # historical v0.2 execution
        q,k,v=projected.reshape(len(x),3,self.heads,64).unbind(1)
        del projected
        bias=self.bias(graph.relative[dilation].to(q.dtype));values=[]
        if self.sparse_edge_attention:
            from .sparse_attention import edge_topology,sparse_head
            crow,col,relation=edge_topology(graph,dilation,self.chunk)
            self.edge_stats={'nodes':len(x),'edges':len(col),'average_degree':len(col)/len(x),
                'topology_bytes':sum(a.numel()*a.element_size() for a in [crow,col,relation]),'dilation':dilation}
            heads=[]
            for head in range(self.heads):
                args=(q[:,head],k[:,head],v[:,head],bias,crow,col,relation,head,self.drop.p,self.drop.training)
                h=checkpoint(sparse_head,*args,use_reentrant=False,preserve_rng_state=True) if self.training else sparse_head(*args)
                heads.append(h)
            attended=torch.stack(heads,1).reshape(len(x),self.dim)
            del heads,h,crow,col,relation
            for start in range(0,len(x),self.chunk):
                args=(x[start:start+self.chunk],attended[start:start+self.chunk])
                y=checkpoint(self.project_block,*args,use_reentrant=False,preserve_rng_state=True) if self.training else self.project_block(*args)
                values.append(y)
            del attended
        else:
            for start in range(0,len(x),self.chunk):
                end=min(start+self.chunk,len(x));args=(x[start:end],q[start:end],k,v,bias,graph,start,end,dilation)
                y=checkpoint(self.attention_block,*args,use_reentrant=False,preserve_rng_state=True) if self.training else self.attention_block(*args)
                values.append(y)
        x=torch.cat(values)
        del q,k,v,bias,y,args
        values=[]
        self.last_execution={'operation':'norm2+FFN','input_shape':list(x.shape),'dtype':str(x.dtype),
            'ffn_chunk':self.ffn_chunk,'hidden_shape_upper_bound':[min(len(x),self.ffn_chunk),4*self.dim]}
        for start in range(0,len(x),self.ffn_chunk):
            # Bound method, not a closure over a loop slice; norm+hidden stay local.
            part=x[start:start+self.ffn_chunk]
            y=checkpoint(self.ffn_block,part,use_reentrant=False,preserve_rng_state=True) if self.training and self.ffn_checkpoint else self.ffn_block(part)
            values.append(y)
        return torch.cat(values)


class Temporal(nn.Module):
    def __init__(self,dim,heads,chunk=128,ffn_chunk=8192,offload_saved_tensors=False):
        super().__init__();self.layers=nn.ModuleList([SparseLayer(dim,heads,chunk,ffn_chunk) for _ in range(3)])
        self.offload_saved_tensors=offload_saved_tensors
    def forward(self,features,coords,pts):
        from contextlib import nullcontext
        lengths=[len(a) for a in features];x=torch.cat(features);self.last_stats=[];self.active_layer=None
        with torch.profiler.record_function('pvst.compact_topology'):
            graph=Neighborhood(coords,pts)
        for dilation,layer in zip([1,2,4],self.layers):
            self.last_stats.append({'dilation':dilation,'nodes':len(x),'neighbors_max_bound':81,'graph':'on demand; edge count not collected'})
            def apply(value,topology,d,mod=layer):
                self.active_layer={'dilation':d,'nodes':len(value),'width':mod.dim,'dtype':str(value.dtype)}
                return mod(value,topology,d)
            # Offload ONLY each outer checkpoint's layer input, once; inner query
            # checkpoints remain under the outer recomputation hook, no K/V copies.
            context=torch.autograd.graph.save_on_cpu(pin_memory=True) if self.training and self.offload_saved_tensors else nullcontext()
            with context:
                x=checkpoint(apply,x,graph,dilation,use_reentrant=False,preserve_rng_state=True) if self.training else apply(x,graph,dilation)
        return list(x.split(lengths))
