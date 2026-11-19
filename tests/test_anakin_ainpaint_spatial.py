import sys
import unittest
import torch
from torch import nn
from research_tools.xyz_pilot.ainpaint_spatial import normalize_pad,token_grid,SpatialBaseline,adaptation_state,load_adaptation,AINPAINT,TARGETS

class FakeBackbone(nn.Module):
    def forward_features(self,x):
        self.input=x.detach().clone();h,w=x.shape[-2:]
        return {'x_norm_patchtokens':x.new_zeros((1,h//14*(w//14),2)),
                'x_norm_clstoken':x.new_zeros((1,2))}
class CoordinateHead(nn.Module):
    def forward(self,x,target_size):
        self.target=target_size;h,w=target_size
        return torch.arange(h*w,device=x.device).reshape(1,1,h,w).float()

class SpatialContracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(19)
    def test_full_picture_crop_before_resize(self):
        rgb=torch.randint(0,256,(17,31,3),dtype=torch.uint8);base=FakeBackbone();head=CoordinateHead();m=SpatialBaseline(base,head)
        out=m([{'rgb':rgb}])['pixel'];x=normalize_pad(rgb)
        self.assertEqual(tuple(x.shape),(1,3,28,42));self.assertEqual(tuple(out.shape),(17,31))
        self.assertEqual(out[-1,-1],16*42+30)
        expected=(rgb.permute(2,0,1).float()/255-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        torch.testing.assert_close(x[0,:,:17,:31],expected,atol=0,rtol=0)
        self.assertEqual(int(x[0,:,17:,:].count_nonzero()),0)
        self.assertEqual(int(x[0,:,:17,31:].count_nonzero()),0)
        with self.assertRaises(AssertionError):m([{'rgb':rgb},{'rgb':rgb}])
    def test_patch_count_and_order(self):
        t=torch.arange(12.).reshape(1,6,2)
        x=token_grid({'x_norm_patchtokens':t},(28,42));self.assertEqual(tuple(x.shape),(1,2,2,3))
        torch.testing.assert_close(x[0,:,1,2],t[0,5])
        with self.assertRaises(AssertionError):token_grid({'x_norm_patchtokens':torch.zeros(1,10,2)},(28,42))
    def test_official_head_gradient_BN(self):
        sys.path.insert(0,str(AINPAINT));from ainpaint.modules.context_decoder import HighResContextHead
        h=HighResContextHead(4,1,norm='BN').train();x=torch.randn(1,4,2,3,requires_grad=True)
        y=h(x,target_size=(28,42));self.assertEqual(tuple(y.shape),(1,1,28,42));y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertEqual(int(h.proj_norm.num_batches_tracked),1)
    def test_actual_peft_targets_and_restore(self):
        from peft import LoraConfig,get_peft_model
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.qkv=nn.Linear(4,4);self.proj=nn.Linear(4,4);self.fc1=nn.Linear(4,4);self.fc2=nn.Linear(4,4)
            def forward(self,x):return self.fc2(self.fc1(self.proj(self.qkv(x))))
        b=get_peft_model(Tiny(),LoraConfig(r=4,lora_alpha=8,lora_dropout=.25,target_modules=TARGETS,bias='none'))
        m=SpatialBaseline(b,nn.Linear(4,1));original={n:p.clone() for n,p in b.named_parameters() if not p.requires_grad}
        saved={n:v.clone() for n,v in adaptation_state(m).items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
        for _ in range(3):
            opt.zero_grad(set_to_none=True);m.head(b(torch.ones(3,4))).square().mean().backward();opt.step()
        self.assertTrue(any(not torch.equal(v,adaptation_state(m)[n]) for n,v in saved.items()))
        self.assertTrue(all(torch.equal(original[n],p) and p.grad is None for n,p in b.named_parameters() if not p.requires_grad))
        load_adaptation(m,saved)
        self.assertTrue(all(torch.equal(v,adaptation_state(m)[n]) for n,v in saved.items()))
        with self.assertRaises(AssertionError):load_adaptation(m,{})

if __name__=='__main__':unittest.main()
