"""Whole-frame occupied voxels; no point limits, image-grid pooling or queries."""
import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """FP32 normalization accumulation, preserve feature storage dtype under AMP."""
    def forward(self,x):
        return F.layer_norm(x.float(),self.normalized_shape,self.weight.float() if self.weight is not None else None,
                            self.bias.float() if self.bias is not None else None,self.eps).to(x.dtype)


def scatter_mean(x,index,count=None):
    n=int(index.max())+1 if len(index) else 0
    if count is None:count=torch.bincount(index,minlength=n)
    out=torch.zeros((n,x.shape[-1]),device=x.device,dtype=torch.float32).index_add(0,index,x.float())
    return (out/count.float().clamp_min(1)[:,None]).to(x.dtype)


def scatter_max(x,index,n):
    if not n:return x.new_zeros((0,x.shape[-1]))
    out=x.new_full((n,x.shape[-1]),-torch.inf)
    return out.scatter_reduce(0,index[:,None].expand_as(x),x,reduce='amax',include_self=True)


def voxelize(q,valid,h):
    flat=q.reshape(-1,3);ids=valid.reshape(-1).nonzero()[:,0];points=flat[ids]
    if not torch.isfinite(points).all():raise ValueError('VALID_POINT_NONFINITE')
    v=torch.floor(points/h).long();coords,index,count=torch.unique(v,dim=0,sorted=True,return_inverse=True,return_counts=True)
    centers=(coords.to(q.dtype)+.5)*h;mean=scatter_mean(points,index,count)
    x=torch.cat((points,(points-centers[index])/h,(points-mean[index])/h),-1)
    coarse,parent=torch.unique(torch.div(coords,2,rounding_mode='floor'),dim=0,sorted=True,return_inverse=True)
    return {'ids':ids,'index':index,'count':count,'coords':coords,'centers':centers,'input':x,
            'coarse_coords':coarse,'parent':parent,'coarse_centers':(coarse.to(q.dtype)+.5)*(2*h)}


def tiles(h,w,size,halo=1):
    """Core and clipped halo; image boundary zero padding is applied by the conv."""
    for y in range(0,h,size):
        for x in range(0,w,size):
            yy,xx=max(0,y-halo),max(0,x-halo);ye,xe=min(h,y+size+halo),min(w,x+size+halo)
            yield (y,min(h,y+size),x,min(w,x+size)),(yy,ye,xx,xe)


def local_tiled(x,valid,module,size):
    """x[H,W,C], one spatial3x3 layer; shared whole-image axes, exact halo."""
    h,w,_=x.shape;rows=[];parts=[];last_y=None
    for core,box in tiles(h,w,size,1):
        y,ye,z,ze=core;by,bye,bx,bxe=box
        if last_y is not None and y!=last_y:rows.append(torch.cat(parts,1));parts=[]
        crop=x[by:bye,bx:bxe];mask=valid[by:bye,bx:bxe]
        a=module(crop.permute(2,0,1)[None],mask[None,None])[0].permute(1,2,0)
        parts.append(a[y-by:ye-by,z-bx:ze-bx]);last_y=y
    if parts:rows.append(torch.cat(parts,1))
    return torch.cat(rows,0)
