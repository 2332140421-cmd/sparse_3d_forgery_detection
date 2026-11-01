"""Fixed128/256/512 PVST; checkpoint frame construction, exact halo tiles.

No sampled points, limited voxels or learned partition queries. Eight coarse
context tokens only read/broadcast to occupied nodes; all voxels remain.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import LayerNorm,voxelize,scatter_mean,scatter_max,local_tiled,tiles
from .temporal import Temporal


def mlp(a,b,c):
    return nn.Sequential(nn.Linear(a,b),nn.GELU(),LayerNorm(b),nn.Linear(b,c),nn.GELU(),LayerNorm(c))


def run_chunks(module,x,size):
    if not len(x):return module(x)
    return torch.cat([module(x[i:i+size]) for i in range(0,len(x),size)])


class LocalResidual(nn.Module):
    def __init__(self,c=128):
        super().__init__();self.depth=nn.Conv2d(c,c,3,padding=1,groups=c);self.mix=nn.Conv2d(c,c,1);self.norm=LayerNorm(c)
    def forward(self,x,mask):
        h=self.mix(F.gelu(self.depth(torch.where(mask,x,0))))
        h=self.norm(h.permute(0,2,3,1)).permute(0,3,1,2)
        return torch.where(mask,x+h,0)


class PVST(nn.Module):
    def __init__(self,condition,decoder,config):
        super().__init__()
        if condition not in ['UV_PVST','XYZ_PVST']:raise ValueError(condition)
        self.condition=condition;self.h=1/32 if condition=='UV_PVST' else 1/64;self.config=config;self.decoder=decoder
        self.geo=mlp(9,64,128);self.geo_local=LocalResidual()
        self.vfe=mlp(256,128,128);self.block=mlp(257,256,256)
        self.visual_projection=nn.Conv2d(32,128,1)
        self.visual_time=nn.Conv3d(128,128,(3,3,3),groups=128,padding=(0,1,1))
        self.visual_mix=nn.Conv2d(128,128,1);self.visual_norm=LayerNorm(128)
        self.visual_native=mlp(133,128,128);self.visual_local=LocalResidual()
        self.geo_norm=LayerNorm(256);self.a_projection=nn.Linear(128,256);self.a_norm=LayerNorm(256)
        self.child=mlp(259,256,256);self.coarse=mlp(512,512,512)
        self.context_tokens=nn.Parameter(torch.randn(8,512)*.02)
        self.context_read=nn.MultiheadAttention(512,8,.1,batch_first=True)
        self.context_write=nn.MultiheadAttention(512,8,.1,batch_first=True);self.context_norm=LayerNorm(512)
        self.fine_time=Temporal(256,4,config['attention_chunk']);self.coarse_time=Temporal(512,8,config['attention_chunk'])
        self.coarse_projection=nn.Linear(512,256);self.fuse=mlp(256,512,256)
        self.pixel_projection=nn.Linear(512,128);self.pixel_local=LocalResidual();self.pixel_out=nn.Linear(128,1)
        self.frame_visual=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
        self.frame_residual=nn.Sequential(nn.Linear(1024,256),nn.GELU(),LayerNorm(256),nn.Linear(256,128),nn.GELU(),LayerNorm(128),nn.Linear(128,1))
        nn.init.zeros_(self.pixel_out.weight);nn.init.zeros_(self.pixel_out.bias)
        nn.init.zeros_(self.frame_residual[-1].weight);nn.init.zeros_(self.frame_residual[-1].bias)
        self._bn={k:v.detach().cpu().clone() for k,v in self.decoder.named_buffers()}
        self.train(False)
    def train(self,mode=True):
        super().train(mode)
        for mod in self.decoder.modules():
            if isinstance(mod,nn.modules.batchnorm._BatchNorm):mod.eval()
        return self
    def check_bn(self):
        for k,v in self.decoder.named_buffers():
            if not torch.equal(v.detach().cpu(),self._bn[k]):raise ValueError('OFFICIAL_BN_CHANGED')
    def decode(self,tokens):
        d=self.decoder;x=self.decoder.proj_relu(d.proj_norm(d.proj(tokens[None].float())))
        x=d.up3(d.up2(d.up1(x)));x=d.dropout(d.final_conv_block(x))
        return x[0],d.predictor(x)[0,0]
    def sample_native(self,x,frame):
        h,w=frame['rgb'].shape[:2];m=frame['mapping448'];left,top=m['pad_xy'];ch,cw=m['content_hw']
        # Native pixel centers -> visual448 canvas -> decoder grid, align_corners=False.
        yy=(torch.arange(h,device=x.device,dtype=torch.float32)+.5)/h*ch+top
        xx=(torch.arange(w,device=x.device,dtype=torch.float32)+.5)/w*cw+left
        y,xg=torch.meshgrid(yy/448*2-1,xx/448*2-1,indexing='ij');grid=torch.stack((xg,y),-1)[None]
        return F.grid_sample(x[None],grid,mode='bilinear',align_corners=False)[0].permute(1,2,0)
    def visual(self,decoded,frame):
        features=[self.visual_projection(x[0][None])[0] for x in decoded]
        while len(features)<3:features.insert(0,torch.zeros_like(features[0]))
        stack=torch.stack(features[-3:],1)[None]
        timed=self.visual_time(stack)[:,:,0];timed=self.visual_mix(F.gelu(timed))
        timed=self.visual_norm((timed+features[-1][None]).permute(0,2,3,1)).permute(0,3,1,2)[0]
        native=self.sample_native(timed,frame);rgb=frame['rgb'].float()/255
        inp=torch.cat((native,rgb,frame['uv']),-1);h,w=rgb.shape[:2]
        base=run_chunks(self.visual_native,inp.reshape(-1,133),self.config['point_chunk']).reshape(h,w,128)
        a=local_tiled(base,torch.ones(h,w,device=rgb.device,dtype=torch.bool),self.visual_local,self.config['tile_size'])
        vis=self.sample_native(decoded[-1][1][None],frame)[...,0]
        return a,vis
    def frame_blocks(self,frame,meta,decoded,keep_pixels):
        h,w=frame['rgb'].shape[:2];valid=frame['valid'];chunk=self.config['point_chunk'];ids=meta['ids'];idx=meta['index'];n=len(meta['coords'])
        point=run_chunks(self.geo,meta['input'],chunk)
        g=point.new_zeros(h*w,128).index_copy(0,ids,point).reshape(h,w,128)
        g=local_tiled(g,valid,self.geo_local,self.config['tile_size']);gp=g.reshape(-1,128)[ids]
        maxg=scatter_max(gp,idx,n)
        # VFE reads global voxel maxima, even when points straddle spatial tiles.
        uf=[]
        for start in range(0,len(gp),chunk):
            uf.append(self.vfe(torch.cat((gp[start:start+chunk],maxg[idx[start:start+chunk]]),-1)))
        u=torch.cat(uf) if uf else gp
        b=torch.cat((scatter_mean(u,idx,meta['count']),scatter_max(u,idx,n),meta['count'].float().log1p().to(g.dtype)[:,None]),-1)
        G=run_chunks(self.block,b,chunk)
        a,vis=self.visual(decoded,frame);ap=a.reshape(-1,128)[ids];A=scatter_mean(ap,idx,meta['count'])
        z=self.geo_norm(G)+self.a_norm(self.a_projection(A))
        relative=(meta['centers']-meta['coarse_centers'][meta['parent']])/(2*self.h)
        e=self.child(torch.cat((z,relative.to(z.dtype)),-1));parent=meta['parent'];cn=len(meta['coarse_coords'])
        weight=meta['count'].float()
        sums=torch.zeros(cn,256,device=e.device,dtype=torch.float32).index_add(0,parent,e.float()*weight[:,None]);mass=weight.new_zeros(cn).index_add(0,parent,weight)
        coarse=self.coarse(torch.cat(((sums/mass.clamp_min(1)[:,None]).to(e.dtype),scatter_max(e,parent,cn)),-1))
        if cn:
            token=self.context_tokens.to(coarse.dtype)[None]
            context,_=self.context_read(token,coarse[None],coarse[None],need_weights=False)
            context=token+context
            # Query chunking avoids materializing M2xM2; no nodes dropped.
            out=[]
            for start in range(0,cn,self.config['attention_chunk']):
                c=coarse[start:start+self.config['attention_chunk']][None]
                y,_=self.context_write(c,context,context,need_weights=False);out.append(self.context_norm(c+y)[0])
            coarse=torch.cat(out)
        gd=torch.cat((gp.mean(0),gp.amax(0))) if len(gp) else g.new_zeros(256)
        av=a.reshape(-1,128);ad=torch.cat((av.mean(0),av.amax(0)))
        if not keep_pixels:g=g.new_empty(0,128);a=a.new_empty(0,128);vis=vis.new_empty(0)
        return z,coarse,gd,ad,g,a,vis
    def pixel(self,g,a,z,meta):
        h,w=g.shape[:2];size=self.config['tile_size'];rows=[];parts=[];last=None
        # Native point inputs and global voxel context are gathered only for tile+halo.
        lookup=torch.full((h*w,),-1,device=g.device,dtype=torch.long);lookup[meta['ids']]=meta['index'];lookup=lookup.reshape(h,w)
        for core,box in tiles(h,w,size,1):
            y,ye,x,xe=core;by,bye,bx,bxe=box
            if last is not None and y!=last:rows.append(torch.cat(parts,1));parts=[]
            ix=lookup[by:bye,bx:bxe]
            def decode_tile(gc,ac,zc,indices):
                ctx=zc[indices.clamp_min(0)] if len(zc) else gc.new_zeros(*indices.shape,256)
                ctx=torch.where((indices>=0)[...,None],ctx,0)
                hidden=self.pixel_projection(torch.cat((gc,ac,ctx),-1))
                mask=torch.ones(indices.shape,device=gc.device,dtype=torch.bool)
                hidden=self.pixel_local(hidden.permute(2,0,1)[None],mask[None,None])[0].permute(1,2,0)
                return self.pixel_out(hidden)[...,0]
            args=(g[by:bye,bx:bxe],a[by:bye,bx:bxe],z,ix)
            out=checkpoint(decode_tile,*args,use_reentrant=False) if self.training else decode_tile(*args)
            parts.append(out[y-by:ye-by,x-bx:xe-bx]);last=y
        if parts:rows.append(torch.cat(parts,1))
        return torch.cat(rows,0)
    def forward(self,frames,target_start=0):
        # q/valid supplied by condition-specific data loader. UV never opens XYZ.
        meta=[f['voxel_meta'] if 'voxel_meta' in f else voxelize(f['q'],f['valid'],self.h) for f in frames]
        decoded=[]
        for f in frames:
            fn=lambda token:self.decode(token)
            decoded.append(checkpoint(fn,f['tokens'],use_reentrant=False) if self.training else fn(f['tokens']))
        prepared=[]
        for ti,f in enumerate(frames):
            def build(frame,m,dec,keep):return self.frame_blocks(frame,m,dec,keep)
            args=(f,meta[ti],decoded[max(0,ti-2):ti+1],ti>=target_start)
            prepared.append(checkpoint(build,*args,use_reentrant=False) if self.training else build(*args))
        pts=[f['pts'] for f in frames]
        # PARALLEL scale paths: coarse initial depends on Z0, never fine-ST output.
        fine=self.fine_time([p[0] for p in prepared],[m['coords'] for m in meta],pts)
        coarse=self.coarse_time([p[1] for p in prepared],[m['coarse_coords'] for m in meta],pts)
        predictions=[]
        for ti in range(target_start,len(frames)):
            m=meta[ti];p=prepared[ti];z=fine[ti]+self.coarse_projection(coarse[ti])[m['parent']]
            z=z+self.fuse(z)
            g,a,vis=p[4:];delta=self.pixel(g,a,z,m)
            weight=m['count'].float()
            zd=torch.cat((((z.float()*weight[:,None]).sum(0)/weight.sum().clamp_min(1)).to(z.dtype),z.amax(0))) if len(z) else p[2].new_zeros(512)
            desc=torch.cat((p[3],p[2],zd))
            d=self.frame_visual(p[3])+self.frame_residual(desc)
            predictions.append({'pixel_logits':vis+delta,'visual_pixel_logits':vis,'frame_logit':d.squeeze(),
                                'voxels_fine':len(m['coords']),'voxels_coarse':len(m['coarse_coords']),
                                'counts':m['count'],'valid_pixels':len(m['ids'])})
        return predictions
