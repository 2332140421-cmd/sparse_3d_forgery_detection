import copy
import unittest
from pathlib import Path
import torch
from research_tools.xyz_pilot.dformer_stage4 import OfficialStage4,Stage4Localization,backbone_input,stage4_parameter
from research_tools.xyz_pilot.insource_model import losses
from research_tools.xyz_pilot.data import read,ROOT
from research_tools.rgbd_slice.model import backbone
from research_tools.xyz_pilot.dformer_stage4_experiment import frozen_hash
from test_xyz_joint_relation import frame


class RealOfficialStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.official=backbone(read(ROOT/'pvst_xyz_source_expansion_v1/provider_config.json')).eval()

    def test_raw_split_exact_official_forward_and_real_depth_state(self):
        stage=OfficialStage4(copy.deepcopy(self.official));torch.manual_seed(4)
        x=torch.randn(1,3,32,32);depth=torch.zeros(1,1,32,32)
        seen={};h=stage.encoder.stages[2][-1].register_forward_hook(lambda m,a,o:seen.update(rgb=o[0].permute(0,3,1,2),depth=o[1].permute(0,3,1,2)))
        with torch.no_grad():full,_=stage.encoder(x,depth)
        h.remove();low,rgb,e=stage.prefix(x,depth)
        torch.testing.assert_close(rgb,seen['rgb'],atol=0,rtol=0);torch.testing.assert_close(e,seen['depth'],atol=0,rtol=0)
        for a,b in zip(low,full[:3]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(stage(rgb,e),full[3],atol=0,rtol=0)
        self.assertGreater(int(e.count_nonzero()),0)
        self.assertEqual(sum(p.numel() for p in stage.parameters() if p.requires_grad),10670080)
        self.assertEqual(list(stage.encoder._modules),['downsample_layers','downsample_layers_e','stages'])
        self.assertFalse(stage.encoder.stages[3][0].drop_depth);self.assertTrue(stage.encoder.stages[3][1].drop_depth)

    def test_eval_gradient_boundary_history_and_no_cross_update_cache(self):
        model=Stage4Localization(.5,copy.deepcopy(self.official));model.train()
        self.assertTrue(model.pvst.training);self.assertFalse(any(m.training for m in model.stage4.encoder.modules()))
        self.assertTrue(all(p.requires_grad==stage4_parameter(n) for n,p in model.stage4.encoder.named_parameters()))
        frozen=frozen_hash(model.stage4);frames=[]
        for i in [0,1]:
            f=frame(i);low,x,e=model.stage4.prefix(torch.randn(1,3,32,32),torch.zeros(1,1,32,32))
            f.update(prefix_rgb=x,prefix_depth=e,gt_work=torch.zeros(8,8,dtype=torch.bool));f['gt_work'][2:4,2:4]=True;frames.append(f)
        retained=[]
        def capture(m,a,r):r.retain_grad();retained.append(r)
        hook=model.stage4.register_forward_hook(capture)
        out=model([frames[0],frames[0],frames[1]],True);parts=losses(out,frames[-1]);parts.sum().backward();hook.remove()
        self.assertEqual(model.stage4.calls,2)
        self.assertTrue(all(r.grad is not None and torch.isfinite(r.grad).all() and r.grad.norm()>0 for r in retained))
        self.assertTrue(all(p.grad is None for n,p in model.stage4.encoder.named_parameters() if not p.requires_grad))
        grads=[p.grad for p in model.stage4.encoder.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        before=model.stage4.encoder.downsample_layers[3][1].weight.detach().clone();opt.step()
        self.assertFalse(torch.equal(before,model.stage4.encoder.downsample_layers[3][1].weight));self.assertEqual(frozen,frozen_hash(model.stage4))
        model.zero_grad(set_to_none=True);model(frames,True);self.assertEqual(model.stage4.calls,4)
        with self.assertRaises(AssertionError):model(frames,False)

    def test_original_rgb_padding_normalization_and_external_zero_depth(self):
        f=dict(rgb=torch.full((8,10,3),128,dtype=torch.uint8),padded_hw=[32,32]);x,d=backbone_input(f)
        self.assertEqual(tuple(x.shape),(1,3,32,32));self.assertEqual(tuple(d.shape),(1,1,32,32));self.assertEqual(int(d.count_nonzero()),0)
        self.assertEqual(int(x[:,:,8:].count_nonzero()),0);self.assertEqual(int(x[:,:,:,10:].count_nonzero()),0)
        expected=(torch.tensor(128/255)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
        torch.testing.assert_close(x[0,:,0,0],expected)
