import copy
import inspect
import unittest
import numpy as np
import torch
from research_tools.xyz_pilot.readout_adaptation import (
    HEADS, MODE, freeze, frozen_hash, frozen_optimizer_hash, tree_hash, add_area)
from research_tools.xyz_pilot.insource_probe import epoch_orders, evaluate
from research_tools.xyz_pilot.insource_model import XYZLocalization, losses
from test_xyz_joint_relation import frame


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_model_freezing_restored_moments_and_readout_update(self):
        torch.manual_seed(12);m=XYZLocalization(.5)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
        f=frame();f['gt_work']=torch.zeros(8,8,dtype=torch.bool);f['gt_work'][2:4,2:4]=True
        losses(m([f],True),f).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
        saved=copy.deepcopy(opt.state_dict());m2=XYZLocalization(.5);m2.load_state_dict(m.state_dict(),strict=True)
        opt2=torch.optim.AdamW(m2.parameters(),lr=1e-4,weight_decay=.01);opt2.load_state_dict(copy.deepcopy(saved))
        self.assertEqual(tree_hash(saved),tree_hash(opt2.state_dict()))
        freeze(m2,True);self.assertEqual(sum(p.numel() for p in m2.parameters() if p.requires_grad),31873)
        self.assertEqual({n.split('.')[0] for n,p in m2.named_parameters() if p.requires_grad},HEADS)
        before=frozen_hash(m2);osig=frozen_optimizer_hash(m2,opt2)
        w={h:tree_hash(getattr(m2,h).state_dict()) for h in HEADS}
        losses(m2([f],True),f).sum().backward()
        self.assertTrue(all(p.grad is None for p in m2.parameters() if not p.requires_grad))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m2.parameters() if p.requires_grad))
        opt2.step()
        self.assertEqual(before,frozen_hash(m2));self.assertEqual(osig,frozen_optimizer_hash(m2,opt2))
        self.assertTrue(all(w[h]!=tree_hash(getattr(m2,h).state_dict()) for h in HEADS))
        for p in m2.parameters():self.assertEqual(int(opt2.state[p]['step']),2 if p.requires_grad else 1)
        freeze(m2,False);self.assertEqual(sum(p.numel() for p in m2.parameters() if p.requires_grad),631170)

    def test_continuation_epochs_not_reused_and_exact_coverage(self):
        rows=[dict(video_id=r,train_targets=list(range(7))) for r in ['real','fake']]
        orders=epoch_orders(rows,13);self.assertEqual(orders[:10],epoch_orders(rows,10))
        for ep,items in enumerate(orders[10:],11):
            self.assertEqual({i['epoch'] for i in items},{ep});self.assertEqual(len(items),14)
            self.assertEqual(len({(i['video_id'],i['target']) for i in items}),14)
        self.assertNotEqual(orders[0][0]['epoch'],orders[10][0]['epoch'])
        self.assertEqual(MODE,'XYZ_TEMPORAL')

    def test_area_identity_and_video_macro(self):
        ev=dict(frames=[dict(video_id='f',role='fake',native_pixels=100,GT_pixels=20,GT_recall=.5,fake_background_FPR=.25),dict(video_id='r',role='real',real_FPR=.1)],videos=[dict(video_id='f',role='fake'),dict(video_id='r',role='real')],macro={})
        add_area(ev);self.assertAlmostEqual(ev['macro']['prediction_area'],.3)
        self.assertAlmostEqual(ev['frames'][1]['prediction_area'],.1)

    def test_fp32_capture_optin_preserves_original_default(self):
        self.assertIs(inspect.signature(evaluate).parameters['capture_dtype'].default,np.float16)
        x=np.array([-1e-9,1e-9],np.float32)
        np.testing.assert_array_equal(x.astype(np.float32)>=0,[False,True])
