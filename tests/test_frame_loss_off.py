import unittest
import torch
from research_tools.rgbd_slice.set_fit_check import clip_objective
from research_tools.rgbd_slice.model import RGBDSetSlices
from research_tools.depth_slice.model import terms
from test_rgbd_set_relations import TinyEncoder

class FrameWeightTests(unittest.TestCase):
    def test_original_exact_and_spatial_unscaled(self):
        p=torch.tensor([.7,.3,.8],requires_grad=True)
        self.assertTrue(torch.equal(clip_objective(p),p.sum()))
        self.assertTrue(torch.equal(clip_objective(p,1.),p.sum()))
        self.assertTrue(torch.equal(clip_objective(p,0.),p[1:].sum()))
        torch.testing.assert_close(torch.autograd.grad(clip_objective(p,0.),p)[0],torch.tensor([0.,1.,1.]),rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(clip_objective(p,1.),p)[0],torch.ones(3),rtol=0,atol=0)
        with self.assertRaises(ValueError):clip_objective(p,.5)

    def test_full_tiny_spatial_path_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(20260909);model=RGBDSetSlices(TinyEncoder(),rgb_only=True).train()
        xyz=torch.randn(16,16,3);xyz[...,2]=torch.arange(16)[:,None]/8
        gt=torch.zeros(16,16,dtype=torch.bool);gt[3:8,4:10]=True
        f=dict(rgb=torch.randint(0,256,(16,16,3),dtype=torch.uint8),xyz=xyz,valid=torch.ones(16,16,dtype=torch.bool),depth=torch.randn(16,16),gt=gt,pts=0.)
        p=model([f])[0];loss=clip_objective(terms(p,gt),0.);loss.backward()
        for prefix in ['local.encoder','local.final','pool','pixel_context']:
            g=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(g,prefix);self.assertTrue(all(torch.isfinite(v).all() for v in g),prefix)
        self.assertGreater(float(model.local.final[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.pixel_context.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in model.sets.frame_head.parameters()))

if __name__=='__main__':unittest.main()
