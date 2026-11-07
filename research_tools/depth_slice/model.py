"""Native learned point features -> slabs -> explicit temporal/depth relations.

Called by the pilot runner; returns distinct frame and native-pixel logits. It
shares observations only with PVST; no cube, detector or pretrained forensic head.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def slab_ids(xyz,valid,h=.125):
    ids=torch.floor(xyz[...,2][valid]/h).long()
    unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
    return unique,inverse


def pool(features,index,n):
    counts=torch.bincount(index,minlength=n).to(features.dtype)
    sums=features.new_zeros(n,features.shape[-1]).index_add(0,index,features)
    maximum=features.new_full((n,features.shape[-1]),-torch.inf).scatter_reduce(0,index[:,None].expand_as(features),features,reduce='amax',include_self=True)
    return torch.cat((sums/counts[:,None],maximum),-1)


class PairRelation(nn.Module):
    """Eight independent head relations, conditioned on actual signed distance.

    Each edge explicitly forms r=GELU(q*k/sqrt(d)+distance_embedding).
    Its learned score normalizes over all valid neighbors; its vector augments V.
    Query chunking preserves the same softmax and all edges, using native torch.
    """
    def __init__(self,dim=512,heads=8):
        super().__init__();self.heads=heads;self.width=dim//heads
        self.norm=nn.LayerNorm(dim);self.q=nn.Linear(dim,dim);self.k=nn.Linear(dim,dim);self.v=nn.Linear(dim,dim)
        self.distance=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,dim))
        self.score=nn.Parameter(torch.ones(heads,self.width)/math.sqrt(self.width))
        self.out=nn.Linear(dim,dim);self.drop=nn.Dropout(.1)
        self.ffn=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,4*dim),nn.GELU(),nn.Dropout(.1),nn.Linear(4*dim,dim))
    def forward(self,x,position,mask):
        single=x.ndim==2
        if single:x=x[None];position=position[None];mask=mask[None]
        batch,n,dim=x.shape
        if not n or not batch:return x[0] if single else x
        z=self.norm(x);q=self.q(z).reshape(batch,n,self.heads,self.width);k=self.k(z).reshape_as(q);v=self.v(z).reshape_as(q)
        groups=[]
        for bi in range(0,batch,16):
            pieces=[]
            for lo in range(0,n,16):
                def attend(qc,kc,vc,delta,allowed):
                    b,nq,nk=delta.shape
                    rel=F.gelu(qc[:,:,None]*kc[:,None]/math.sqrt(self.width)+self.distance(delta[...,None]).reshape(b,nq,nk,self.heads,self.width))
                    logits=(rel*self.score).sum(-1).permute(0,1,3,2)
                    available=allowed.any(-1)
                    logits=logits.masked_fill(~allowed[:,:,None],-torch.inf)
                    logits=torch.where(available[:,:,None,None],logits,torch.zeros_like(logits))
                    weights=self.drop(torch.softmax(logits.float(),-1).to(vc.dtype))
                    message=torch.einsum('bqhk,bqkhd->bqhd',weights,vc[:,None]+rel).reshape(b,nq,-1)
                    return torch.where(available[:,:,None],message,0)
                pos=position[bi:bi+16]
                args=(q[bi:bi+16,lo:lo+16],k[bi:bi+16],v[bi:bi+16],pos[:,None]-pos[:,lo:lo+16,None],mask[bi:bi+16,lo:lo+16])
                pieces.append(checkpoint(attend,*args,use_reentrant=False,preserve_rng_state=True) if self.training else attend(*args))
            groups.append(torch.cat(pieces,1))
        update=self.drop(self.out(torch.cat(groups)))
        x=x+torch.where(mask.any(-1)[:,:,None],update,0)
        x=x+self.ffn(x)
        return x[0] if single else x


class DepthSliceRelation(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo=nn.Sequential(nn.Linear(5,128),nn.GELU(),nn.Linear(128,256),nn.GELU())
        self.visual_projection=nn.Conv2d(512,64,1)
        self.fusion=nn.Sequential(nn.Linear(64+256+2,128),nn.GELU(),nn.LayerNorm(128))
        self.token=nn.Linear(256,512)
        self.time=nn.ModuleList([PairRelation(),PairRelation()]);self.depth=nn.ModuleList([PairRelation(),PairRelation()])
        self.context=nn.Linear(512,128)
        self.pixel=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
        self.frame_visual=nn.Linear(512,1024)
        self.frame_head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
    def local(self,visual,xyz,valid):
        h,w=valid.shape;device=next(self.parameters()).device
        xyz=xyz.to(device);valid=valid.to(device);visual=visual.to(device)
        # Native pixel-center interpolation of frozen normal-stride backbone map.
        visual=F.interpolate(self.visual_projection(visual[None].float()),size=(h,w),mode='bilinear',align_corners=False)[0].permute(1,2,0).reshape(-1,64)
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        uv=torch.stack((2*(xx+.5)/w-1,2*(yy+.5)/h-1),-1).reshape(-1,2)
        q=xyz.reshape(-1,3);ok=valid.reshape(-1);parts=[]
        for lo in range(0,len(q),16384):
            u=uv[lo:lo+16384];good=ok[lo:lo+16384]
            g=self.geo(torch.cat((q[lo:lo+16384],u),-1))
            g=torch.where(good[:,None],g,0)
            parts.append(self.fusion(torch.cat((visual[lo:lo+16384],g,u),-1)))
        return torch.cat(parts),valid
    def encode(self,visual,xyz,valid):
        f,v=self.local(visual,xyz,valid);ids,index=slab_ids(xyz.to(f.device),v)
        c=self.token(pool(f[v.reshape(-1)],index,len(ids))) if len(ids) else f.new_empty(0,512)
        return c,ids
    def forward(self,frames):
        encoded=[];device=next(self.parameters()).device
        for f in frames:
            args=(f['visual'],f['xyz'],f['valid'])
            encoded.append(checkpoint(self.encode,*args,use_reentrant=False) if self.training else self.encode(*args))
        counts=[len(c) for c,_ in encoded];tokens=torch.cat([c for c,_ in encoded]);ids=torch.cat([i for _,i in encoded])
        times=torch.cat([torch.full((n,),f['pts'],device=device) for n,f in zip(counts,frames)])
        frame_index=torch.cat([torch.full((n,),j,device=device,dtype=torch.long) for j,n in enumerate(counts)])
        for tm,dm in zip(self.time,self.depth):
            unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
            size=len(frames);flat=inverse*size+frame_index
            grid=tokens.new_zeros(len(unique)*size,512).index_copy(0,flat,tokens).reshape(len(unique),size,512)
            present=torch.zeros(len(unique)*size,device=device,dtype=torch.bool);present[flat]=True;present=present.reshape(len(unique),size)
            pts=torch.tensor([f['pts'] for f in frames],device=device).expand(len(unique),-1)
            mask=present[:,:,None]&present[:,None,:]&(pts[:,None]<pts[:,:,None])
            updated=tm(grid,pts,mask)
            h=updated.reshape(-1,512)[flat]
            chunks=[];lo=0
            for n,(_,sid) in zip(counts,encoded):
                x=h[lo:lo+n];pos=sid.float()*.125;mask=~torch.eye(n,device=device,dtype=torch.bool)
                chunks.append(dm(x,pos,mask));lo+=n
            tokens=torch.cat(chunks)
        result=[];lo=0
        for j,(f,n) in enumerate(zip(frames,counts)):
            ctx=tokens[lo:lo+n];sid=ids[lo:lo+n];lo+=n
            def pixels(visual,xyz,valid,c,unique):
                local,v=self.local(visual,xyz,valid);context=local.new_zeros(len(local),128)
                if len(unique):
                    _,ix=slab_ids(xyz.to(local.device),v)
                    context=context.index_copy(0,torch.where(v.reshape(-1))[0],self.context(c)[ix])
                outputs=[self.pixel(torch.cat((local[k:k+16384],context[k:k+16384]),-1)) for k in range(0,len(local),16384)]
                return torch.cat(outputs).reshape(v.shape)
            args=(f['visual'],f['xyz'],f['valid'],ctx,sid)
            z=checkpoint(pixels,*args,use_reentrant=False) if self.training else pixels(*args)
            # Invalid-depth pixels retain local visual localization. Frame fallback
            # is explicit visual evidence rather than a fabricated empty slab.
            if n:desc=torch.cat((ctx.mean(0),ctx.amax(0)))
            else:desc=tokens.new_zeros(1024)
            desc=desc+self.frame_visual(f['visual'].to(device).float().mean((-2,-1)))
            result.append({'pixel':z,'frame':self.frame_head(desc).squeeze(),'slices':n})
        return result


def terms(pred,gt):
    z=pred['pixel'].float();g=gt.to(z.device).float();p=z.sigmoid()
    # Empty GT participates: epsilon=1 penalizes predicted foreground as in the
    # same per-frame soft Dice used for edited frames, no missing-label fill.
    return torch.stack((F.binary_cross_entropy_with_logits(pred['frame'].float(),g.any().float()),
                        F.binary_cross_entropy_with_logits(z,g),1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)))
