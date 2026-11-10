import unittest
import torch
from torch import nn
from research_tools.rgbd_slice.pixel_message_dev_check import pixel_top1,pixel_message_mode
from research_tools.rgbd_slice.model import RGBDSetSlices
from test_rgbd_set_relations import TinyEncoder

class Tests(unittest.TestCase):
    def test_top1_ceil_logits_all_pixels(self):
        z=torch.arange(201,dtype=torch.float32).reshape(3,67)
        self.assertEqual(float(pixel_top1(z)),199.) # ceil(2.01)=3, logits not sigmoid
        self.assertEqual(float(pixel_top1(torch.tensor([[-5.,-2.]]))),-2.)
    def test_zero_after_affine_bias_and_cleanup(self):
        model=nn.Module();model.pixel_context=nn.Linear(2,3);model.pixel_context.bias.data.fill_(5.)
        x=torch.ones(4,2);original=model.pixel_context(x).detach().clone();keys=list(model.state_dict())
        with pixel_message_mode(model,False) as normal:torch.testing.assert_close(model.pixel_context(x),original)
        with pixel_message_mode(model,True) as zero:
            z=model.pixel_context(x);self.assertEqual(int(z.count_nonzero()),0);self.assertEqual(z.shape,original.shape);self.assertEqual(z.dtype,original.dtype)
        self.assertEqual(zero['delivered_nonzero_elements'],0);self.assertEqual(zero['calls'],1)
        self.assertFalse(model.pixel_context._forward_hooks);self.assertEqual(keys,list(model.state_dict()));torch.testing.assert_close(model.pixel_context(x),original)
    def test_actual_model_frame_head_unchanged(self):
        torch.set_num_threads(2);torch.manual_seed(20260909);model=RGBDSetSlices(TinyEncoder(),rgb_only=True).eval()
        with torch.no_grad():model.pixel_context.bias.fill_(.25)
        xyz=torch.randn(16,16,3);xyz[...,2]=torch.arange(16)[:,None]/8
        f=dict(rgb=torch.randint(0,256,(16,16,3),dtype=torch.uint8),xyz=xyz,valid=torch.ones(16,16,dtype=torch.bool),depth=torch.randn(16,16),pts=0.)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            with pixel_message_mode(model,False):a=model([f])[0]
            with pixel_message_mode(model,True):b=model([f])[0]
        torch.testing.assert_close(a['frame'],b['frame'],rtol=0,atol=0);self.assertFalse(torch.equal(a['pixel'],b['pixel']))
        self.assertFalse(model.sets.zero_messages);self.assertFalse(model.sets.capture)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
if __name__=='__main__':unittest.main()
