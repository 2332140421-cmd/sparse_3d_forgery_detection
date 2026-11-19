import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from research_tools.xyz_pilot.indataset_gap_screen import (
    DEV,full_video_rows,label_reference,load_model)
from research_tools.xyz_pilot.insource_probe import partition_history
from research_tools.xyz_pilot.insource_model import XYZLocalization,ImageOnly


class IndependentDevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def manifest(self):
        return {'rows':[dict(pair_id=p,split='dev',role=role,video_id=p+'_'+role,source_id=p,video_sha256=p+role,actual_frames=4,pts=[0,.1,.2,.3]) for p in DEV for role in ['real','fake']]}

    def test_full_video_no_second_partition_no_cross_source(self):
        m=self.manifest();before=copy.deepcopy(m)
        rows=full_video_rows(m,{'rows':[dict(source_id='train',role='real',video_sha256='train')]})
        self.assertEqual(m,before)
        for r in rows:
            self.assertEqual(r['evaluation_range'],[0,4]);self.assertEqual(r['evaluation_targets'],[0,1,2,3])
            self.assertEqual(r['preview_targets'],[0,2,3]);self.assertEqual(partition_history(r,0,'evaluation'),[None]*7+[0])
            self.assertTrue(all(i is None or i<=3 for i in partition_history(r,3,'evaluation')))
        with self.assertRaises(AssertionError):full_video_rows(m,{'rows':[dict(source_id=DEV[0],role='real',video_sha256='other')]})

    def test_label_reference_exact_cached_target_and_geometry_denominators(self):
        gt=np.array([[1,0],[0,1]],bool);valid=np.array([[1,1],[0,0]],bool)
        r=label_reference(gt,gt,valid)
        self.assertEqual(r['label_roundtrip_Dice'],1);self.assertEqual(r['label_roundtrip_IoU'],1)
        self.assertEqual(r['native_GT_geometry_coverage'],.5);self.assertEqual(r['work_GT_geometry_coverage'],.5)
        with self.assertRaises(AssertionError):label_reference(gt,~gt,valid)
        empty=label_reference(np.zeros((2,2),bool),np.zeros((2,2),bool),valid)
        self.assertIsNone(empty['native_GT_geometry_coverage'])

    def test_strict_correct_localization_class_and_wrong_checkpoint_rejected(self):
        c=dict(h=.5);ref=XYZLocalization(.5)
        for mode in ['IMAGE_ONLY','XYZ_CURRENT','XYZ_TEMPORAL']:
            model=ImageOnly(ref) if mode=='IMAGE_ONLY' else ref
            cp=dict(condition=mode,updates=10260,identity={},state_dict=model.state_dict(),initial_sha256='initial')
            with patch('torch.load',return_value=cp),patch('research_tools.xyz_pilot.indataset_gap_screen.read',return_value={}),patch('research_tools.xyz_pilot.indataset_gap_screen.digest',return_value='digest'),patch.object(torch.nn.Module,'cuda',lambda self:self):
                out,_=load_model(Path('test.pt'),mode,10260,c)
                self.assertIsInstance(out,ImageOnly if mode=='IMAGE_ONLY' else XYZLocalization)
                bad=copy.deepcopy(cp);bad['state_dict']['frame.weight']=torch.ones(1)
                with patch('torch.load',return_value=bad),self.assertRaises(RuntimeError):load_model(Path('test.pt'),mode,10260,c)
                bad['condition']='old_frame_head_model'
                with patch('torch.load',return_value=bad),self.assertRaises(AssertionError):load_model(Path('test.pt'),mode,10260,c)
