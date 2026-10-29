"""Official spatial decoder, optionally augmented before decoding by full V8."""
import copy
from contextlib import contextmanager,nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from research_tools.v8.component_global_context_pilot.model import GlobalContextModel
from research_tools.v8.full_video_training.model import VideoTraining
from research_tools.v8.spatial_localization.model import dense_image,STRUCTURE_KEYS

CONDITIONS=('SPATIAL_TRANSFER','SPATIAL_3D_CONTEXT')


def align256(field,m256,m512):
    """Patch centers in518 -> native-normalized512 content -> actual256 mapping."""
    x=(torch.arange(37,device=field.device,dtype=torch.float32)+.5)*14
    y=x.clone();x0,y0=m512['pad_xy'];h,w=m512['content_hw']
    gx0,gy0=m256['pad_xy'];gh,gw=m256['content_hw']
    x=((x-x0)/w*gw+gx0)/256*2-1;y=((y-y0)/h*gh+gy0)/256*2-1
    yy,xx=torch.meshgrid(y,x,indexing='ij');grid=torch.stack((xx,yy),-1)[None].expand(len(field),-1,-1,-1)
    return F.grid_sample(field.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)


class SpatialModel(nn.Module):
    def __init__(self,condition,decoder,statistics,*,freeze_decoder_bn=False):
        super().__init__()
        if condition not in CONDITIONS:raise ValueError(condition)
        self.condition=condition;self.context=condition==CONDITIONS[1]
        self.decoder=copy.deepcopy(decoder).requires_grad_(True)
        self.freeze_decoder_bn=freeze_decoder_bn
        if freeze_decoder_bn:
            self._bn_initial={k:v.detach().cpu().clone() for k,v in self.decoder.named_buffers()
                if k.endswith(('running_mean','running_var','num_batches_tracked'))}
        if self.context:
            self.structure=GlobalContextModel('GLOBAL_COMPONENT',statistics,patch_chunk=512)
            # Kept for compatibility with the shared hidden-state path, never optimized.
            self.structure.cell_head.requires_grad_(False)
            self.geo1=nn.Conv2d(7,32,3,padding=1);self.geo2=nn.Conv2d(32,32,3,padding=1)
            self.projection=nn.Sequential(nn.Conv2d(160,128,1),nn.ReLU(),nn.Conv2d(128,1024,1))
            nn.init.zeros_(self.projection[-1].weight);nn.init.zeros_(self.projection[-1].bias)

    def train(self,mode=True):
        super().train(mode)
        if mode and self.freeze_decoder_bn:
            for layer in self.decoder.modules():
                if isinstance(layer,nn.BatchNorm2d):layer.eval()
        return self

    def bn_status(self):
        """Small per-epoch assertion against official initialization, also after resume."""
        layers=[m for m in self.decoder.modules() if isinstance(m,nn.BatchNorm2d)]
        current=dict(self.decoder.named_buffers())
        equal=all(torch.equal(current[k].detach().cpu(),v) for k,v in self._bn_initial.items())
        affine=all(m.weight.requires_grad and m.bias.requires_grad for m in layers)
        if not layers or not equal or not affine or any(m.training for m in layers):raise ValueError('FIXED_BN_CONTRACT')
        return {'strategy':'FIXED_BN','layers':len(layers),'buffers_equal':equal,'all_eval':True,'affine_trainable':affine}

    def chunk(self,cpu,previous,active,first,m256,m512):
        device=next(self.parameters()).device;features=cpu['features'].to(device)
        if first:
            previous=torch.zeros_like(previous);active=torch.zeros_like(active)
        if self.context:
            with torch.autocast(device_type=device.type,enabled=False):
                previous,active,_,_,_,hidden=VideoTraining(self.structure).chunk(
                    {k:cpu[k] for k in STRUCTURE_KEYS},previous.float(),active,first,True)
            content=dense_image(cpu['content'].to(device)).bool();valid=dense_image(cpu['valid'].to(device)).bool()&content
            xyz=dense_image(cpu['xyz'].to(device));mu=F.interpolate(cpu['mu'].to(device).transpose(1,2).reshape(-1,3,16,16),size=(256,256),mode='nearest')
            m=self.structure;safe=torch.where(valid,xyz,0.)
            absolute=torch.where(valid,(safe-m.absolute_mean[None,:,None,None])/m.absolute_scale[None,:,None,None],0.)
            relative=torch.where(valid,(safe-mu)/m.relative_scale[None,:,None,None],0.)
            g=torch.where(content,torch.cat((absolute,relative,valid.float()),1),0.)
            g=torch.where(content,F.relu(self.geo1(g)),0.);g=torch.where(content,F.relu(self.geo2(g)),0.)
            c=F.interpolate(hidden.transpose(1,2).reshape(-1,128,16,16),size=(256,256),mode='nearest')
            projected=self.projection(align256(torch.cat((g,c),1),m256,m512))
            cm=cpu['content512'].to(device)[:,None].float();pm=F.avg_pool2d(F.pad(cm,(0,6,0,6)),14,14)>0
            features=features+torch.where(pm,projected,0.)
        logits=self.decoder(features,target_size=(518,518))[:,0,:512,:512]
        return previous,active,logits


def pixel_sums(logits,gt,mask):
    gt=gt.float();mask=mask.bool();area=mask.sum((-1,-2)).float()
    b=F.binary_cross_entropy_with_logits(logits.float(),gt,reduction='none')
    b=torch.where(mask,b,0.).sum((-1,-2))/area
    p=torch.sigmoid(logits.float())*mask;y=gt*mask;foreground=y.sum((-1,-2))
    d=torch.where(foreground>0,1-(2*(p*y).sum((-1,-2))+1)/(p.sum((-1,-2))+foreground+1),0.)
    return (b+d).sum(),b.sum(),d.sum()


@contextmanager
def preserve_bn_buffers(model):
    """Recompute the same training batch stats without updating BN a second time."""
    saved=[(m,m.running_mean,m.running_var,m.num_batches_tracked)
        for m in model.modules() if isinstance(m,nn.BatchNorm2d) and m.track_running_stats]
    for m,mean,var,count in saved:
        m.running_mean=mean.clone();m.running_var=var.clone();m.num_batches_tracked=count.clone()
    try:yield
    finally:
        for m,mean,var,count in saved:
            m.running_mean=mean;m.running_var=var;m.num_batches_tracked=count


def video_loss(model,video,chunk_size,use_checkpoint=True):
    device=next(model.parameters()).device;h=torch.zeros(256,128,device=device);active=torch.zeros(256,dtype=torch.bool,device=device)
    total=torch.zeros(3,device=device);r=video.row
    def step(cpu,hidden,pa,first):
        hidden,pa,z=model.chunk(cpu,hidden,pa,first,r['mapping256'],r['mapping512'])
        loss,b,d=pixel_sums(z,cpu['gt'].to(device),cpu['content512'].to(device))
        return hidden,pa,torch.stack((loss,b,d))
    for start in range(0,r['actual_frames'],chunk_size):
        cpu=video.chunk(start,min(start+chunk_size,r['actual_frames']))
        if use_checkpoint:
            h,active,part=checkpoint(step,cpu,h,active,start==0,use_reentrant=False,
                context_fn=lambda:(nullcontext(),preserve_bn_buffers(model.decoder)))
        else:h,active,part=step(cpu,h,active,start==0)
        total=total+part
    total=total/r['actual_frames']
    return total[0],{'loss':total[0],'bce':total[1],'dice':total[2]}
