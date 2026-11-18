import copy
import unittest
import torch
from research_tools.xyz_pilot.model import XYZPilot, history_indices
from research_tools.xyz_pilot.joint_model import XYZJointPilot, JointLayer


def frame(index=0, valid=True):
    torch.manual_seed(21 + index)
    return dict(visual=[torch.randn(c, 4//(2**min(i,2)), 4//(2**min(i,2))) for i,c in enumerate([64,128,256,512])],
                rgb=torch.randint(0,256,(8,8,3),dtype=torch.uint8), padded_hw=(8,8),
                xyz=torch.randn(8,8,3), valid=torch.full((8,8),valid), pts=index*.1, frame_index=index)


class JointRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mask_matches_explicit_mha_and_gradients(self):
        torch.manual_seed(4)
        layer=JointLayer(8,2)
        x=torch.randn(7,8,requires_grad=True); p=torch.randn(7,8)
        ids=torch.tensor([0,0,1,1,1,0,2])
        actual=layer(x,p,ids,False)
        n=layer.norm(x); qk=n+p
        a,_=layer.attn(qk[None],qk[None],n[None],attn_mask=ids[:,None]!=ids[None,:],need_weights=False)
        y=x+a[0]; expected=y+layer.ff(y)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        ga=torch.autograd.grad(actual.square().sum(),(x,*layer.parameters()),retain_graph=True)
        gb=torch.autograd.grad(expected.square().sum(),(x,*layer.parameters()))
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
        self.assertGreater(float((layer(x,p,ids,True)-actual).abs().max()),1e-4)

    def test_a_history_independence_b_history_gradients(self):
        torch.manual_seed(5); model=XYZJointPilot(.5)
        a=torch.randn(3,128,requires_grad=True); b=torch.randn(2,128,requires_grad=True)
        enc=[dict(C=a,position=torch.zeros_like(a)),dict(C=b,position=torch.zeros_like(b))]
        r,s=model.read_window(enc,[-.1,0],[0,1],False)
        changed=[dict(C=a*100,position=torch.randn_like(a)),enc[1]]
        r2,s2=model.read_window(changed,[-.1,0],[0,1],False)
        torch.testing.assert_close(r,r2,atol=0,rtol=0);torch.testing.assert_close(s,s2,atol=0,rtol=0)
        r,s=model.read_window(enc,[-.1,0],[0,1],True)
        r.square().mean().backward()
        self.assertGreater(float(a.grad.norm()),0)
        self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(AssertionError):model.read_window(enc,[.1,0],[0,1],True)

    def test_c0_bypass_old_default_and_mapping(self):
        torch.manual_seed(8);old=XYZPilot(.5).eval();f=frame()
        got=[];handle=old.space.register_forward_pre_hook(lambda m,args:got.append(args[2].detach()))
        with torch.no_grad():old.encode(f)
        handle.remove()
        torch.testing.assert_close(old.encode(f,spatial=False)['C'],got[0],atol=0,rtol=0)
        joint=XYZJointPilot(.5).eval()
        for n in ['space','context','frame']:self.assertFalse(hasattr(joint,n))
        e=joint.encode(f);r=torch.randn_like(e['C'])
        seen=[];h=joint.pixel.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach()))
        joint.decode(f,e,r);h.remove()
        expected=torch.zeros(64,32);expected[e['points']]=joint.pixel_context(r)[e['index']]
        torch.testing.assert_close(seen[0][0,-32:].permute(1,2,0).reshape(64,32),expected)

    def test_empty_missing_duplicate_and_causality(self):
        m=XYZJointPilot(.5).eval();f=frame(0,False)
        out=m([None,None,f],True)
        self.assertTrue(torch.isfinite(out['pixel']).all());self.assertTrue(torch.isfinite(out['frame']))
        f=frame(1)
        for temporal in [False,True]:
            out=m([None,f,f],temporal)
            self.assertTrue(torch.isfinite(out['frame']))
        with self.assertRaises(AssertionError):m([frame(2),f],True)
        self.assertEqual(history_indices([0,.1,.2],1),[None,None,None,None,None,None,0,1])


class JointProtocolTests(unittest.TestCase):
    def test_fixed_orders_and_monitor_exclusions(self):
        from research_tools.xyz_pilot.joint_experiment import observation_targets, coverage
        from research_tools.xyz_pilot.experiment import orders
        rows=[]
        for pair in ['an:40y1x9','an:other']:
            for role in ['real','fake']:
                rows.append(dict(pair_id=pair,video_id=pair[3:]+'_'+role,split='train',role=role,actual_frames=15,pts=[i*.1 for i in range(15)]))
        config=dict(train=['an:40y1x9','an:other'],updates=20,seed=20260909,history_slots=8,history_step_seconds=.1)
        short=orders(rows,config)
        config['updates']=40
        long=orders(rows,config)
        self.assertEqual(short,long[:20])
        monitor=observation_targets(rows)
        self.assertTrue(all(i not in [6,7,8,9] for i in monitor['40y1x9_fake']))
        self.assertEqual(coverage(long,rows)['updates'],len(long))
