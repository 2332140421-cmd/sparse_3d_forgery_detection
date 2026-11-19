"""Localization-only paths for the fixed same-source time split (ADR-0045).

Caller insource_probe: returns pixel logits. Reuse the existing decoder and
relations; remove unused classification parameters instead of padding I.
"""
import copy
import torch
from torch import nn
from .model import XYZPilot
from .joint_model import XYZJointPilot

COMMON = ['lateral', 'visual', 'local_out', 'detail', 'pixel']


class XYZLocalization(XYZJointPilot):
    def __init__(self, h):
        super().__init__(h)
        del self.evidence, self.empty

    def forward(self, frames, temporal=False):
        target=frames[-1];encoded=[];times=[];identities=[];memo={}
        for f in (frames if temporal else [target]):
            if f is None:continue
            assert f['pts']<=target['pts']
            identity=f['frame_index']
            if identity not in memo:memo[identity]=self.encode(f)
            encoded.append(memo[identity]);times.append(f['pts']-target['pts']);identities.append(identity)
        r=self.relate_window(encoded,times,identities,temporal)
        return dict(pixel=self.decode(target,encoded[-1],r),blocks=len(r),window_blocks=sum(len(x['C']) for x in encoded))


class ImageOnly(nn.Module):
    encode_visual=XYZPilot.encode_visual
    decode=XYZPilot.decode

    def __init__(self, reference):
        super().__init__();self.d=128
        for name in COMMON:setattr(self,name,copy.deepcopy(getattr(reference,name)))

    def forward(self, frames, temporal=False):
        f=frames[-1];local=self.encode_visual(f)
        # Empty context makes the shared decoder's entire context channel zero,
        # without reading XYZ, validity, indices, or any block module.
        p=self.decode(f,dict(local=local),local.new_empty((0,self.d)))
        return dict(pixel=p,blocks=0,window_blocks=0)


def losses(output, frame):
    from torch.nn import functional as F
    z=output['pixel'].float();gt=frame['gt_work'].float();p=z.sigmoid()
    return torch.stack((F.binary_cross_entropy_with_logits(z,gt),1-(2*(p*gt).sum()+1)/(p.sum()+gt.sum()+1)))
