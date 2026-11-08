import unittest
import torch
from torch import nn
from research_tools.rgbd_slice.set_relations import SetRelations
from research_tools.rgbd_slice.model import RGBDSetSlices


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.rgb=nn.Conv2d(3,64,3,padding=1,stride=4)
        self.layers=nn.ModuleList([nn.Conv2d(64,c,1) for c in [64,128,256,512]])
    def forward(self,x,d):
        x=self.rgb(x)+d.mean()*0.01
        return [p(x) for p in self.layers],None


class SetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(20260909)
    def test_actual_path_causality_permutation_and_overlap(self):
        model=SetRelations().eval()
        e=[(torch.randn(3,512),torch.tensor([-4,2,10])) for _ in range(3)]
        pts=[0.,.041,.113]
        with torch.no_grad():
            c,y=model(e,pts);future=e[:2]+[(e[2][0]*90,e[2][1])]
            cc,yy=model(future,pts)
            for a,b in zip(c[:2],cc[:2]):torch.testing.assert_close(a,b,atol=2e-5,rtol=3e-5)
            torch.testing.assert_close(torch.stack(y[:2]),torch.stack(yy[:2]),atol=2e-5,rtol=3e-5)
            order=torch.tensor([2,0,1]);permuted=[(x[order],i[order]) for x,i in e]
            cp,yp=model(permuted,pts)
            for a,b in zip(c,cp):torch.testing.assert_close(a[order],b,atol=2e-5,rtol=3e-5)
            torch.testing.assert_close(torch.stack(y),torch.stack(yp),atol=2e-5,rtol=3e-5)
            model.capture=True;model(e,pts)
        self.assertGreater(model.observations[-1]['overlap_mean_sum_over_units'],1)
        self.assertGreater(model.observations[-1]['A_B_overlap_mean'],0)

    def test_full_native_model_gradients_and_message_intervention(self):
        model=RGBDSetSlices(TinyEncoder()).train()
        frames=[]
        for t in range(3):
            xyz=torch.randn(16,16,3);xyz[...,2]=torch.arange(16)[:,None]/8
            frames.append(dict(rgb=torch.randint(0,256,(16,16,3),dtype=torch.uint8),depth=torch.randn(16,16),xyz=xyz,valid=torch.ones(16,16,dtype=torch.bool),gt=torch.ones(16,16,dtype=torch.bool),pts=[0.,.039,.101][t]))
        output=model(frames)
        loss=sum(p['pixel'].float().square().mean()+p['frame'].float().square() for p in output);loss.backward()
        for prefix in ['local.encoder','pool','sets.participation.0','sets.participation.1','sets.read_members','sets.interaction','sets.gru','sets.frame_head','pixel_context']:
            g=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(any(x is not None and x.abs().sum()>0 for x in g),prefix)
            self.assertTrue(all(torch.isfinite(x).all() for x in g if x is not None),prefix)
        model.eval()
        with torch.no_grad():
            a=model(frames);model.sets.zero_messages=True;b=model(frames)
        self.assertGreater(float((a[0]['frame']-b[0]['frame']).abs()),0)
        self.assertEqual(a[0]['pixel'].shape,(16,16))

    def test_checkpoint_rng_and_empty_frame(self):
        model=SetRelations().train()
        e=[(torch.randn(2,512,requires_grad=True),torch.tensor([-2,8])),(torch.empty(0,512),torch.empty(0,dtype=torch.long))]
        state=torch.get_rng_state();a,y=model(e,[0.,.047]);loss=sum(x.square().mean() for x in a if x.numel())+torch.stack(y).square().mean();loss.backward();grad=e[0][0].grad.clone()
        model.zero_grad();e[0][0].grad=None;torch.set_rng_state(state)
        b,z=model(e,[0.,.047]);(sum(x.square().mean() for x in b if x.numel())+torch.stack(z).square().mean()).backward()
        torch.testing.assert_close(grad,e[0][0].grad,atol=2e-5,rtol=3e-5)
        self.assertTrue(torch.isfinite(grad).all())


if __name__=='__main__':unittest.main()
