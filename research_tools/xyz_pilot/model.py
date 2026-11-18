"""Single-scale learnable XYZ-block pilot; called only by this experiment.

All valid working-grid points route by XYZ floor; invalid geometry retains the
local visual path. No labels/provenance enter encode or relation computations.
"""
import torch
from torch import nn
from torch.nn import functional as F


def route(xyz,valid,h):
    points=torch.where(valid.reshape(-1))[0]
    coords=torch.floor(xyz.reshape(-1,3)[points]/h).long()
    keys,index=torch.unique(coords,dim=0,sorted=True,return_inverse=True)
    return points,keys,index


def history_indices(pts,target,slots=8,step=.1):
    import bisect
    # Each requested slot uses the most recent observation at or BEFORE its
    # requested PTS; a negative/missing past slot stays explicitly None.
    t=pts[target];result=[]
    for lag in range(slots-1,-1,-1):
        j=bisect.bisect_right(pts,t-lag*step+1e-9,0,target+1)-1
        result.append(j if j>=0 else None)
    assert result[-1]==target and all(j is None or j<=target for j in result)
    return result


class Interaction(nn.Module):
    def __init__(self,d=128,heads=4):
        super().__init__();self.qnorm=nn.LayerNorm(d);self.knorm=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=0,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,q,k,v):
        if not len(q):return q
        a,_=self.attn(self.qnorm(q)[None],self.knorm(k)[None],v[None],need_weights=False)
        x=q+a[0];return x+self.ff(x)


class XYZPilot(nn.Module):
    def __init__(self,h,d=128):
        super().__init__();self.h=h;self.d=d
        self.lateral=nn.ModuleList([nn.Conv2d(c,d,1) for c in [64,128,256,512]])
        self.visual=nn.Sequential(nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(8,d),nn.GELU())
        self.geometry=nn.Sequential(nn.Linear(6,64),nn.GELU(),nn.Linear(64,2*d))
        self.point=nn.Sequential(nn.Linear(d,d),nn.GELU())
        self.score=nn.Linear(d,1);self.value=nn.Linear(d,d)
        self.position=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,d))
        self.time=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,d))
        self.space=Interaction(d);self.context=Interaction(d)
        self.empty=nn.Parameter(torch.zeros(1,d));self.pixel_context=nn.Linear(d,32)
        self.local_out=nn.Conv2d(d,32,1);self.detail=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU())
        self.pixel=nn.Sequential(nn.Conv2d(80,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,1,1))
        self.frame=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def encode(self,f):
        maps=f['visual'];y=None
        for i in [3,2,1,0]:
            a=self.lateral[i](maps[i][None]);y=a if y is None else a+F.interpolate(y,size=a.shape[-2:],mode='bilinear',align_corners=False)
        y=self.visual(y);H,W=f['rgb'].shape[:2];ph,pw=f['padded_hw']
        local=F.interpolate(y,size=(ph,pw),mode='bilinear',align_corners=False)[...,:H,:W]
        xyz,valid=f['xyz'],f['valid'];points,keys,index=route(xyz,valid,self.h)
        if len(points):
            absolute=xyz.reshape(-1,3)[points];relative=absolute/self.h-torch.floor(absolute/self.h)-.5
            gamma,beta=self.geometry(torch.cat((absolute,relative),-1)).chunk(2,-1)
            v=local[0].permute(1,2,0).reshape(-1,self.d)[points]
            x=self.point(v*(1+.1*gamma.tanh())+.1*beta)
            z=self.score(x).float()[:,0]
            mx=z.new_full((len(keys),),-torch.inf).scatter_reduce(0,index,z,reduce='amax',include_self=True)
            a=(z-mx[index]).exp();den=a.new_zeros(len(keys)).index_add(0,index,a)
            C=x.new_zeros((len(keys),self.d),dtype=torch.float32).index_add(0,index,self.value(x).float()*(a/den[index])[:,None])
            pos=self.position((keys.float()+.5)*self.h)
            C=self.space(C+pos,C+pos,C)
        else:C=local.new_empty((0,self.d));pos=C
        return {'C':C,'position':pos,'keys':keys,'points':points,'index':index,'local':local}
    def read(self,current,contexts,relative_pts):
        C=current['C']
        if len(C):
            ks=[];vs=[]
            for e,dt in zip(contexts,relative_pts):
                if len(e['C']):
                    ks.append(e['C']+e['position']+self.time(e['C'].new_full((len(e['C']),1),dt)));vs.append(e['C'])
            if ks:context=self.context(C+current['position'],torch.cat(ks),torch.cat(vs))
            else:context=C+self.empty
            frame=self.frame(context.mean(0)).squeeze()
        else:context=C;frame=self.frame(self.empty[0]).squeeze()
        return context,frame
    def decode(self,f,current,context):
        H,W=f['rgb'].shape[:2];ctx=current['local'].new_zeros(H*W,32)
        if len(context):ctx=ctx.index_copy(0,current['points'],self.pixel_context(context)[current['index']].to(ctx.dtype))
        ctx=ctx.T.reshape(1,32,H,W)
        rgb=f['rgb'].permute(2,0,1)[None].float()/255
        p=self.pixel(torch.cat((self.local_out(current['local']),self.detail(rgb),ctx),1))[0,0]
        return p
    def forward(self,frames,temporal):
        # None slots are missing observations, never zero/future images.
        current_f=frames[-1];encoded=[];times=[];current=None
        for f in (frames if temporal else [current_f]):
            if f is None:continue
            e=self.encode(f);encoded.append(e);times.append(f['pts']-current_f['pts'])
            if f is current_f:current=e
        context,frame=self.read(current,encoded,times)
        return {'pixel':self.decode(current_f,current,context),'frame':frame,'blocks':len(current['C'])}
