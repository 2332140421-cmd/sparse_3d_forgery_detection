import unittest
import numpy as np
import torch
from research_tools.xyz_pilot.model import XYZPilot,route,history_indices
from research_tools.xyz_pilot.data import occupancy_counts

torch.set_num_threads(1)

class XYZContracts(unittest.TestCase):
    def frame(self,t):
        y,x=torch.meshgrid(torch.linspace(-.9,.9,32),torch.linspace(-.9,.9,32),indexing='ij')
        return {'visual':[torch.randn(c,s,s) for c,s in [(64,8),(128,4),(256,2),(512,1)]],'rgb':torch.randint(0,255,(32,32,3),dtype=torch.uint8),'xyz':torch.stack((x,y,torch.ones_like(x)),dim=-1),'valid':torch.ones(32,32,dtype=torch.bool),'pts':t,'padded_hw':[32,32]}
    def test_floor_route_and_counts(self):
        q=np.array([[-1.01,0,.1],[-.1,0,.1],[.1,0,.1],[.9,0,1.1],[1,0,1]],np.float32)
        hs=[.125,.25,.5,1.,2.,4.];a=occupancy_counts(q,hs)
        for h in hs:self.assertEqual(a[str(h)],len(np.unique(np.floor(q/h).astype(np.int64),axis=0)))
        p,k,i=route(torch.tensor(q).reshape(1,-1,3),torch.ones(1,5,dtype=torch.bool),1.)
        self.assertEqual(len(p),5);self.assertTrue(torch.equal(k[i],torch.floor(torch.tensor(q)).long()))
    def test_history_causal_and_model_gradient(self):
        self.assertEqual(history_indices([0.,.1,.2,.3,.4],3,4,.1),[0,1,2,3])
        self.assertEqual(history_indices([0.,.1],0,4,.1),[None,None,None,0])
        torch.manual_seed(20260909);m=XYZPilot(.5);a=self.frame(0.);b=self.frame(.1)
        z=m([a,b],False);a['visual']=[v*3 for v in a['visual']]
        zz=m([a,b],False);torch.testing.assert_close(z['pixel'],zz['pixel'],atol=0,rtol=0)
        p=m([a,b],True);(p['pixel'].square().mean()+p['frame'].square()).backward()
        for module in [m.geometry,m.score,m.value,m.space,m.context,m.pixel,m.frame]:
            gs=[x.grad for x in module.parameters() if x.grad is not None]
            self.assertTrue(gs);self.assertTrue(all(torch.isfinite(g).all() for g in gs));self.assertGreater(sum(float(g.abs().sum()) for g in gs),0)
        a2=self.frame(0.);p2=m([a2,b],True)
        self.assertGreater(float((p2['pixel']-p['pixel']).abs().max()),0)
    def test_invalid_geometry_keeps_pixel_output(self):
        f=self.frame(0.);f['valid'].zero_();m=XYZPilot(1.)
        o=m([f],True);self.assertEqual(o['blocks'],0);self.assertEqual(o['pixel'].shape,(32,32));self.assertTrue(torch.isfinite(o['pixel']).all())
    def test_frame_target_preserves_native_positive(self):
        from research_tools.xyz_pilot.experiment import terms
        pred={'pixel':torch.zeros(2,2),'frame':torch.tensor(1.)}
        t=terms(pred,{'gt_work':torch.zeros(2,2,dtype=torch.bool),'frame_positive':True})
        self.assertAlmostEqual(float(t[2]),float(torch.nn.functional.softplus(torch.tensor(-1.))),places=6)
