"""Official DFormer-S + shared multiscale decoder, optional residual slab branch.

Immediate caller: matched pilot runner. Full native outputs, no forensic weights.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from research_tools.depth_slice.model import PairRelation,slab_ids
from research_tools.depth_slice.data import save
from scripts.prepare_v8_assets import digest


def backbone(config,record=None):
    source=Path(config['dformer_code'])/'models/encoders/DFormer.py'
    if digest(source)!=config['dformer_source_sha256']:raise ValueError('OFFICIAL_SOURCE_CHANGED')
    spec=importlib.util.spec_from_file_location('dformer_original_small',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.DFormer_Small(pretrained=False)
    if digest(Path(config['dformer_weights']))!=config['dformer_weights_sha256']:raise ValueError('WEIGHT_IDENTITY')
    state=torch.load(config['dformer_weights'],map_location='cpu',weights_only=False)
    container='state_dict_ema' if 'state_dict_ema' in state else 'state_dict'
    raw=state[container];weights={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in raw.items()}
    target=model.state_dict();ignored={k:list(v.shape) for k,v in weights.items() if k not in target}
    if any(not k.startswith(('pred.','stages.3.1.layer_scale_1_e','stages.3.1.layer_scale_2_e','stages.3.1.attn.proj_e.','stages.3.1.mlp_e2.')) for k in ignored):raise ValueError('UNEXPECTED_OFFICIAL_KEYS:'+str(ignored))
    filtered={k:v for k,v in weights.items() if k in target}
    model.load_state_dict(filtered,strict=True)
    if record:save(record,{'revision':config['dformer_revision'],'source_sha256':digest(source),'weights_sha256':digest(Path(config['dformer_weights'])),
                          'container':container,'missing_keys':[],'unexpected_backbone_keys':[],'ignored_classifier':ignored,
                          'loaded_keys':len(filtered),'rgb_stem':len([k for k in filtered if k.startswith('downsample_layers.')]),
                          'depth_stem':len([k for k in filtered if k.startswith('downsample_layers_e.')]),
                          'interaction':len([k for k in filtered if '.attn.' in k]),'classification_tail_note':'pred and final depth-only update feed classification concatenation only; official segmentation drops this final depth update, RGB multiscale outputs unchanged; strict loading of all retained backbone keys','channels':[64,128,256,512],'strides':[4,8,16,32]})
    return model


def conv(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.GELU())


class Local(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.lateral=nn.ModuleList([nn.Conv2d(c,256,1) for c in [64,128,256,512]])
        self.refine=nn.ModuleList([conv(256,256) for _ in range(4)])
        self.up=conv(256,64);self.detail=nn.Sequential(conv(3,16),conv(16,16));self.native=conv(80,32);self.final=nn.Sequential(conv(32,32),nn.Conv2d(32,1,1))
        self.frame=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1))
    def encode(self,rgb,depth):
        device=next(self.parameters()).device
        # Pretraining uses RGB, normalized ImageNet. See pinned RGBD-Pretrain.
        x=rgb.to(device).permute(2,0,1)[None].float()/255
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        d=depth.to(device)[None,None]
        features,_=self.encoder(x,d)
        y=None
        for i in [3,2,1,0]:
            z=self.lateral[i](features[i]);y=z if y is None else z+F.interpolate(y,size=z.shape[-2:],mode='bilinear',align_corners=False);y=self.refine[i](y)
        return y[0],features[-1].float().mean((-2,-1))[0]
    def decode(self,local,rgb,context):
        local=local.to(next(self.parameters()).device);rgb=rgb.to(local.device).permute(2,0,1)[None].float()/255;h,w=rgb.shape[-2:]
        x=F.interpolate(local[None],size=((h+1)//2,(w+1)//2),mode='bilinear',align_corners=False);x=self.up(x)
        x=F.interpolate(x,size=(h,w),mode='bilinear',align_corners=False);detail=self.detail(rgb)
        x=self.native(torch.cat((x,detail),1));x=x+context.to(device=x.device,dtype=x.dtype)
        return self.final(x)[0,0]


def native_sample(local,linear_ids,hw):
    h,w=hw;y=linear_ids//w;x=linear_ids%w
    grid=torch.stack((2*(x+.5)/w-1,2*(y+.5)/h-1),-1).float()[None,None]
    return F.grid_sample(local[None],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T


class SlabPool(nn.Module):
    def __init__(self):
        super().__init__();self.score=nn.Sequential(nn.Linear(259,128),nn.GELU(),nn.Linear(128,4));self.value=nn.Linear(256,512)
    def forward(self,local,xyz,valid):
        device=next(self.parameters()).device;local=local.to(device);xyz=xyz.to(device);valid=valid.to(device)
        ids,index=slab_ids(xyz,valid);points=torch.where(valid.reshape(-1))[0];n=len(ids)
        if not n:return local.new_empty(0,512),ids
        q=xyz.reshape(-1,3);hw=valid.shape;scores=[]
        for p in points.split(8192):
            def score_chunk(l,coordinates,selected):return self.score(torch.cat((native_sample(l,selected,hw),coordinates[selected]),-1)).float()
            args=(local,q,p);scores.append(checkpoint(score_chunk,*args,use_reentrant=False) if self.training else score_chunk(*args))
        score=torch.cat(scores);maximum=score.new_full((n,4),-torch.inf).scatter_reduce(0,index[:,None].expand(-1,4),score,reduce='amax',include_self=True)
        weights=(score-maximum[index]).exp();denom=weights.new_zeros(n,4).index_add(0,index,weights);weights=weights/denom[index]
        parts=[];lo=0
        for p in points.split(8192):
            ix=index[lo:lo+len(p)];a=weights[lo:lo+len(p)];lo+=len(p)
            def value_chunk(l,selected,routing,weight):
                value=self.value(native_sample(l,selected,hw)).float().reshape(-1,4,128)*weight[...,None]
                return value.new_zeros(n,512).index_add(0,routing,value.reshape(-1,512))
            args=(local,p,ix,a);parts.append(checkpoint(value_chunk,*args,use_reentrant=False) if self.training else value_chunk(*args))
        return torch.stack(parts).sum(0),ids


class RGBDSlices(nn.Module):
    def __init__(self,encoder,relation=False,context_norm=False):
        super().__init__();self.local=Local(encoder);self.relation=relation
        if relation:
            self.pool=SlabPool();self.time=nn.ModuleList([PairRelation(),PairRelation()]);self.depth=nn.ModuleList([PairRelation(),PairRelation()])
            self.pixel_context=nn.Linear(512,32);self.frame_context=nn.Linear(512,1)
            for m in [self.pixel_context,self.frame_context]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
            # Appended after all old modules; LayerNorm initialization uses no RNG.
            self.context_norm=nn.LayerNorm(512,eps=1e-5) if context_norm else None
    def train(self,mode=True):
        super().train(mode)
        for m in self.local.encoder.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self
    def relations(self,encoded,frames):
        tokens=torch.cat([e[0] for e in encoded]);ids=torch.cat([e[1] for e in encoded]);counts=[len(e[0]) for e in encoded];device=tokens.device
        fi=torch.cat([torch.full((n,),i,device=device,dtype=torch.long) for i,n in enumerate(counts)])
        for tm,dm in zip(self.time,self.depth):
            unique,inverse=torch.unique(ids,sorted=True,return_inverse=True);t=len(frames);flat=inverse*t+fi
            grid=tokens.new_zeros(len(unique)*t,512).index_copy(0,flat,tokens).reshape(len(unique),t,512)
            present=torch.zeros(len(unique)*t,device=device,dtype=torch.bool);present[flat]=True;present=present.reshape(len(unique),t)
            pts=torch.tensor([f['pts'] for f in frames],device=device).expand(len(unique),-1)
            mask=present[:,:,None]&present[:,None,:]&(pts[:,None]<pts[:,:,None]);h=tm(grid,pts,mask).reshape(-1,512)[flat]
            parts=[];lo=0
            for n,(_,sid) in zip(counts,encoded):
                parts.append(dm(h[lo:lo+n],sid.float()*.125,~torch.eye(n,device=device,dtype=torch.bool)));lo+=n
            tokens=torch.cat(parts)
        return list(tokens.split(counts))
    def forward(self,frames):
        device=next(self.parameters()).device;locals=[];desc=[];encoded=[]
        for f in frames:
            args=(f['rgb'],f['depth']);local,d=checkpoint(self.local.encode,*args,use_reentrant=False) if self.training else self.local.encode(*args)
            local=local.to('cpu');locals.append(local);desc.append(d)
            if self.relation:
                args=(local,f['xyz'],f['valid']);encoded.append(checkpoint(self.pool,*args,use_reentrant=False) if self.training else self.pool(*args))
        contexts=self.relations(encoded,frames) if self.relation else [None]*len(frames)
        if self.relation and self.context_norm is not None:contexts=[self.context_norm(c) for c in contexts]
        result=[]
        for f,local,d,c in zip(frames,locals,desc,contexts):
            if self.relation:
                _,index=slab_ids(f['xyz'].to(device),f['valid'].to(device));valid=f['valid'].to(device);h,w=valid.shape
                projected=self.pixel_context(c);ctx=projected.new_zeros(h*w,32)
                if len(c):ctx=ctx.index_copy(0,torch.where(valid.reshape(-1))[0],projected[index])
                ctx=ctx.reshape(h,w,32).permute(2,0,1)[None].to('cpu');extra=self.frame_context(c.mean(0)).squeeze() if len(c) else d.new_zeros(())
            else:ctx=torch.zeros(1,32,*f['gt'].shape);extra=d.new_zeros(())
            args=(local,f['rgb'],ctx);pixel=checkpoint(self.local.decode,*args,use_reentrant=False) if self.training else self.local.decode(*args)
            result.append({'pixel':pixel,'frame':self.local.frame(d).squeeze()+extra,'slices':len(c) if c is not None else 0})
        return result
