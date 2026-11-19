import random
import unittest
import cv2
import numpy as np
import torch
from torch import nn
from research_tools.xyz_pilot.native_patch_data import crop,choose_crop,positions,tiles,fuse,degrade,masked_losses,patch_logits,SIZE

class PatchContracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_same_seed_crop_and_tiny_foreground_no_rejection(self):
        g=np.zeros((57,63),bool);g[56,62]=True
        a=random.Random(20260909);b=random.Random(20260909)
        p=[choose_crop(g,a,32) for _ in range(100)];q=[choose_crop(g,b,32) for _ in range(100)]
        self.assertEqual(p,q);self.assertTrue(any(x['foreground_guided'] for x in p));self.assertTrue(any(not x['foreground_guided'] for x in p))
        for x in p:
            gt,v=crop(g,x['y'],x['x'],32)
            self.assertEqual(x['GT_pixels'],int(gt.sum()))
            if x['foreground_guided']:self.assertTrue(gt.any());self.assertEqual(x['GT_pixels'],1)
        real=choose_crop(np.zeros((8,9),bool),a,32);self.assertEqual((real['y'],real['x']),(0,0));self.assertFalse(real['foreground_guided'])
    def test_whole_frame_shrink_before_same_crop(self):
        rng=np.random.default_rng(8);rgb=rng.integers(0,256,(360,640,3),dtype=np.uint8)
        expected=cv2.resize(cv2.resize(rgb,(384,216),interpolation=cv2.INTER_AREA),(640,360),interpolation=cv2.INTER_LINEAR)
        low=degrade(rgb,(216,384));np.testing.assert_array_equal(low,expected)
        native,v=crop(rgb,60,100);degraded,w=crop(low,60,100)
        np.testing.assert_array_equal(v,w);np.testing.assert_array_equal(degraded,expected[60:348,100:388]);self.assertGreater(int((native!=degraded).sum()),0)
    def test_native_padding_and_masked_loss(self):
        rgb=np.ones((5,7,3),np.uint8)*17;a,v=crop(rgb,0,0,8)
        self.assertEqual(int(v.sum()),35);self.assertEqual(int(a[5:].sum()),0)
        z=torch.zeros(8,8,requires_grad=True);g=torch.zeros(8,8);g[1,2]=1;valid=torch.from_numpy(v)
        loss=masked_losses(z,g,valid);other=z.detach().clone();other[~valid]=100
        torch.testing.assert_close(masked_losses(other,g,valid),loss)
        loss.sum().backward();self.assertEqual(float(z.grad[~valid].abs().sum()),0);self.assertTrue(torch.isfinite(z.grad).all())
    def test_blind_sliding_tail_and_uniform_logit_fusion(self):
        self.assertEqual(positions(11,4,3),[0,3,6,7]);coords=tiles((7,11),4,3)
        target=np.arange(77,dtype=np.float32).reshape(7,11);p=[crop(target,y,x,4)[0] for y,x in coords]
        actual,count=fuse((7,11),coords,p,4);np.testing.assert_array_equal(actual,target);self.assertGreaterEqual(count.min(),1)
        little=crop(target[:2,:3],0,0,4)[0];actual,c=fuse((2,3),[(0,0)],[little],4);np.testing.assert_array_equal(actual,target[:2,:3])
        varied=[np.full((4,4),i,np.float32) for i in range(len(coords))];total=np.zeros((7,11),np.float32);n=np.zeros((7,11),int)
        for i,(y,x) in enumerate(coords):total[y:y+4,x:x+4]+=i;n[y:y+4,x:x+4]+=1
        result,_=fuse((7,11),coords,varied,4);np.testing.assert_allclose(result,total/n)
        with self.assertRaises(AssertionError):fuse((7,11),coords[:-1],p[:-1],4)
    def test_model_padding_crop_and_batch(self):
        class Backbone(nn.Module):
            def forward_features(self,x):
                self.shape=tuple(x.shape);return dict(x_norm_patchtokens=x.new_zeros(len(x),21*21,2))
        class Head(nn.Module):
            def __init__(self):super().__init__();self.bias=nn.Parameter(torch.tensor(0.))
            def forward(self,x,target_size):
                h,w=target_size;return torch.arange(h*w).reshape(1,1,h,w).expand(len(x),1,h,w).float()+self.bias
        m=nn.Module();m.backbone=Backbone();m.head=Head();m.calls=0
        y=patch_logits(m,[np.zeros((SIZE,SIZE,3),np.uint8)]*2)
        self.assertEqual(tuple(y.shape),(2,288,288));self.assertEqual(m.backbone.shape,(2,3,294,294));self.assertEqual(float(y[0,-1,-1]),287*294+287);self.assertEqual(m.calls,2)

if __name__=='__main__':unittest.main()
