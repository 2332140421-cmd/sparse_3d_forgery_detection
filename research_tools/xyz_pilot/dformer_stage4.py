"""Thin split of the pinned official DFormer forward, not an output adapter.

Stage4 experiment caller: preserve raw RGB/depth stage3 state; rerun actual
stage4 for every unique current/history observation on every optimizer update.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .insource_model import XYZLocalization


def stage4_parameter(name):
    return name.startswith(('downsample_layers.3.','downsample_layers_e.3.','stages.3.'))


def backbone_input(frame):
    x=frame['rgb'].permute(2,0,1)[None].float()/255
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    h,w=frame['rgb'].shape[:2];ph,pw=frame['padded_hw'];x=F.pad(x,(0,pw-w,0,ph-h),value=0)
    return x,x.new_zeros(1,1,ph,pw)


class OfficialStage4(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        for name,p in encoder.named_parameters():p.requires_grad_(stage4_parameter(name))
        self.calls=0;self.train(False)

    def train(self,mode=True):
        # eval fixes all BN buffers/DropPath. It does not disable autograd.
        super().train(False);return self

    @torch.no_grad()
    def prefix(self,x,depth):
        depth=depth[:,0,:,:].unsqueeze(1);outs=[]
        for i in range(3):
            x=self.encoder.downsample_layers[i](x);depth=self.encoder.downsample_layers_e[i](depth)
            x=x.permute(0,2,3,1);depth=depth.permute(0,2,3,1)
            for block in self.encoder.stages[i]:x,depth=block(x,depth)
            x=x.permute(0,3,1,2);depth=depth.permute(0,3,1,2);outs.append(x)
        return outs,x,depth

    def forward(self,x,depth):
        self.calls+=len(x)
        x=self.encoder.downsample_layers[3](x);depth=self.encoder.downsample_layers_e[3](depth)
        x=x.permute(0,2,3,1);depth=depth.permute(0,2,3,1)
        for block in self.encoder.stages[3]:x,depth=block(x,depth)
        return x.permute(0,3,1,2)


class Stage4Localization(nn.Module):
    def __init__(self,h,encoder):
        super().__init__();self.pvst=XYZLocalization(h);self.stage4=OfficialStage4(encoder)

    def train(self,mode=True):
        super().train(mode);self.stage4.eval();return self

    def forward(self,frames,temporal=False):
        assert temporal,'This experiment always uses complete XYZ_TEMPORAL history'
        memo={};adapted=[]
        for f in frames:
            if f is None:adapted.append(None);continue
            i=f['frame_index']
            if i not in memo:
                top=self.stage4(f['prefix_rgb'],f['prefix_depth'])
                # Match original cache half-storage -> FP32 input, retaining its
                # differentiable cast. Raw stage4 entry is never quantized.
                q=dict(f);q['visual']=[*f['visual'][:3],top[0].half().float()];memo[i]=q
            adapted.append(memo[i])
        return self.pvst(adapted,temporal=True)
