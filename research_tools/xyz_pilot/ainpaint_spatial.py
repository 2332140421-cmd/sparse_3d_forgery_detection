"""One official spatial baseline; caller ainpaint_spatial_experiment.

Only full-frame padding/token checks differ from the author's crop-based
wrapper. Official DINOv2, PEFT LoRA and HighResContextHead are used directly.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

EXTERNAL=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/external')
DINO=EXTERNAL/'dinov2-ainpaint'
AINPAINT=EXTERNAL/'AINPAINT-CVIU26'
WEIGHT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth')
TARGETS=['qkv','proj','fc1','fc2']


def normalize_pad(rgb):
    x=rgb.permute(2,0,1)[None].float()/255
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    h,w=rgb.shape[:2]
    return F.pad(x,(0,(-w)%14,0,(-h)%14),value=0.)


def token_grid(out,hw):
    # Official forward_features already removes CLS and four register tokens.
    tokens=out['x_norm_patchtokens'];h,w=hw;gh,gw=h//14,w//14
    assert h%14==w%14==0 and tokens.shape[1]==gh*gw,(tokens.shape,hw)
    return tokens.transpose(1,2).reshape(tokens.shape[0],tokens.shape[-1],gh,gw)


class SpatialBaseline(nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.backbone=backbone;self.head=head;self.calls=0

    def forward(self,frames,temporal=False):
        assert len(frames)==1 and not temporal,'T=1 spatial baseline only'
        f=frames[0];rgb=f['rgb'];h,w=rgb.shape[:2];x=normalize_pad(rgb)
        self.calls+=1
        out=self.backbone.forward_features(x);feat=token_grid(out,x.shape[-2:])
        padded=self.head(feat,target_size=x.shape[-2:])
        assert padded.shape==(1,1,*x.shape[-2:])
        # Crop before native resizing/loss, never stretch padded content.
        return dict(pixel=padded[0,0,:h,:w],blocks=0,window_blocks=0)


def build():
    from peft import LoraConfig,get_peft_model
    sys.path.insert(0,str(AINPAINT))
    from ainpaint.modules.context_decoder import HighResContextHead
    base=torch.hub.load(str(DINO),'dinov2_vitl14_reg',source='local',pretrained=False)
    base.load_state_dict(torch.load(WEIGHT,weights_only=True,map_location='cpu'),strict=True)
    assert base.num_register_tokens==4 and base.embed_dim==1024 and len(base.blocks)==24
    lora=LoraConfig(r=4,lora_alpha=8,lora_dropout=.25,target_modules=TARGETS,bias='none',modules_to_save=[])
    base=get_peft_model(base,lora)
    assert all(('lora_' in n)==p.requires_grad for n,p in base.named_parameters())
    return SpatialBaseline(base,HighResContextHead(1024,1,norm='BN'))


def adaptation_state(model):
    return {k:v for k,v in model.state_dict().items() if k.startswith('head.') or 'lora_' in k}


def load_adaptation(model,state):
    assert set(state)==set(adaptation_state(model)),'Incomplete LoRA/head/buffer checkpoint'
    for k,v in adaptation_state(model).items():assert v.shape==state[k].shape
    loaded=model.load_state_dict(state,strict=False)
    assert not loaded.unexpected_keys and set(loaded.missing_keys)==set(model.state_dict())-set(state)
