"""Thin official Encoder/Decoder adapter, with explicit initialization coverage.

Called only by experiment.py. Exposes continuous middle-frame logits and the
actual author loss/SGD groups; a spatial substitute cannot answer this baseline.
External source stays immutable. All compatibility edits are local and listed.
"""
from pathlib import Path
import importlib.util
import sys
import types
from unittest.mock import patch
import torch
from torch import nn

OFFICIAL=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/third_party/Mumpy')
REVISION='bc9280a0839f97375b8dda2dcd6d5b5052083f13'
WEIGHT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/weights/mumpy/weight.pth')


def official_imports():
    sys.path.insert(0,str(OFFICIAL)) if str(OFFICIAL) not in sys.path else None
    from models.factory import modelFactory
    from models.encoder.encoder import Encoder
    from models.decoder.decoder import Decoder
    from utils.loss import WeightedFocalLoss,softIoULoss
    from utils.optimizer.scheduler import PolynomialLR
    return modelFactory,Encoder,Decoder,WeightedFocalLoss,softIoULoss,PolynomialLR


def allowed_new_key(name):
    # Only method-specific modules absent from the author's ImageNet initializer,
    # plus deterministic shifted-window masks rebuilt by the official class.
    return ('.cva.' in name or '.pre.' in name or
            name in ('globalembedding.weight','globalembedding.bias') or
            name.endswith('.attn_mask'))


class MumPy(nn.Module):
    def __init__(self, initialize=True):
        super().__init__();factory,Encoder,Decoder,*_=official_imports();self.loading={}
        def checked(model,path,strict=False):
            if not initialize:return model
            raw=torch.load(WEIGHT,map_location='cpu',weights_only=False)
            state=raw['state_dict'] if isinstance(raw,dict) and 'state_dict' in raw else raw
            assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values()),'INITIALIZER_SCHEMA'
            expected=model.state_dict()
            shape_errors=[k for k,v in state.items() if k in expected and v.shape!=expected[k].shape]
            assert not shape_errors,('INITIALIZER_SHAPES',shape_errors)
            result=model.load_state_dict(state,strict=False)
            loaded=[k for k in state if k in expected]
            parameter_names=set(dict(model.named_parameters()))
            self.loading=dict(source=str(WEIGHT),loaded_keys=len(loaded),loaded_numel=sum(expected[k].numel() for k in loaded),
                loaded_parameter_numel=sum(expected[k].numel() for k in loaded if k in parameter_names),
                new_parameter_numel=sum(expected[k].numel() for k in result.missing_keys if k in parameter_names),
                missing=list(result.missing_keys),unexpected=list(result.unexpected_keys),
                loaded_by_root={name:sum(expected[k].numel() for k in loaded if k.startswith(name+'.')) for name in ['tokenize','layers','globalembedding','globalblocks']})
            # No arbitrary strict=False success: audit backbone coverage before use.
            critical=[k for k in result.missing_keys if not allowed_new_key(k)]
            assert not critical,('UNINITIALIZED_CRITICAL_KEYS',critical)
            assert not result.unexpected_keys,('UNEXPECTED_INITIALIZER_KEYS',result.unexpected_keys)
            return model
        # The author's factory alone decides ALL three views and global modules.
        with patch.object(factory,'load_model_weights',checked):self.encoder=Encoder()
        self.decoder=Decoder()
        assert type(self.encoder).__name__=='Encoder' and type(self.encoder.base).__name__=='ThreeViewSwinTransformer'

    def forward(self,x):
        assert x.ndim==5 and x.shape[1:]==(3,3,224,224)
        final,views,frequency=self.encoder(x)
        logits,features=self.decoder(final,views,frequency)
        assert logits.shape==(len(x),1,224,224)
        return logits


def losses(logits,targets):
    """Released focal + official softIoU helper, explicit default epsilon1e-6.

    The released softIoULoss wrapper accidentally binds recall=False to epsilon
    as a positional argument. Use the author's helper with named arguments so
    empty ANAKIN masks remain numerically valid, including saturated logits.
    """
    *_,Focal,_IoU,_scheduler=official_imports()
    from utils.loss import softIoU
    focal=Focal()
    shape=(len(logits),-1);a=targets.float().reshape(shape);b=logits.float().reshape(shape)
    fi=focal(a,b);si=softIoU(a,b,e=1e-6,recall=False).mean();total=fi+si
    assert torch.isfinite(total),'NONFINITE_OFFICIAL_LOSS'
    return total,dict(weighted_focal=float(fi.detach()),soft_IoU=float(si.detach()))


def optimizers(model,total_updates):
    *_,PolynomialLR=official_imports()
    cva=[];encoder=[]
    for name,p in model.encoder.named_parameters():(cva if 'cva' in name else encoder).append(p)
    params=[encoder,list(model.decoder.parameters()),cva];lrs=[1e-3,1e-2,1e-2]
    assert sum(len(x) for x in params)==len(list(model.parameters()))
    opts=[torch.optim.SGD(ps,lr=lr,momentum=.9,weight_decay=1e-4) for ps,lr in zip(params,lrs)]
    scheds=[PolynomialLR(o,step_size=1,iter_warmup=0,iter_max=total_updates,power=.9,min_lr=1e-5) for o in opts]
    return opts,scheds


def augmentation(policy="single"):
    if policy == "double":
        from .double_augment import DoubleAugment
        return DoubleAugment()
    assert policy == "single"
    mod=official_augmentation_module()
    return mod.RandAugment(1,7,mod.OneAugmentStrategy())


def official_augmentation_module():
    """Released optional SingleAugment policy: shared Identity/H/V/dihedral.

    Importing author config would open DAVIS paths at module import. Supply only
    its two literal augmentation constants, without loading another dataset.
    """
    cfgmodule=types.ModuleType('configs.davis.config')
    cfgmodule.cfg=types.SimpleNamespace(DATASET=types.SimpleNamespace(IGNORE_LABEL=0,FILL_COLOR=(0,0,0)))
    spec=importlib.util.spec_from_file_location('mumpy_augmentation',OFFICIAL/'utils/randaugment.py')
    mod=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'configs.davis.config':cfgmodule}):spec.loader.exec_module(mod)
    return mod
