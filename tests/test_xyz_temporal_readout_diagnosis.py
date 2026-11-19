import unittest
import numpy as np
import torch
from research_tools.xyz_pilot.temporal_readout_diagnosis import (LOGIT_THRESHOLDS,counts_and_regions,region_statistics,curve_values,aggregate,history_info)
from research_tools.xyz_pilot.insource_model import XYZLocalization
from test_xyz_joint_relation import frame


class TemporalReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_threshold_counts_and_regions_match_direct_native_values(self):
        z=np.array([[-4.,0.,1.],[-1.,0.,5.]],dtype=np.float32);original=z.copy();gt=np.array([[1,1,0],[0,1,0]],bool)
        counts,pos,neg=counts_and_regions(z,gt)
        for j,t in enumerate(LOGIT_THRESHOLDS):
            pred=z.astype(np.float64)>=t
            self.assertEqual(counts[j].tolist(),[int((pred&gt).sum()),int((pred&~gt).sum()),int((~pred&gt).sum()),int((~pred&~gt).sum())])
        np.testing.assert_array_equal(z,original)
        for region,stats in [(gt,pos),(~gt,neg)]:
            for q in [.05,.25,.5,.75,.95]:self.assertAlmostEqual(stats[f'logit_p{int(q*100):02d}'],float(np.quantile(z[region],q)),places=6)
        vals=curve_values(counts);self.assertAlmostEqual(vals['Dice'][49],.5714285714285714)
        self.assertIsNone(region_statistics(np.array([]))['logit_p50'])

    def test_hierarchy_does_not_pool_unequal_frame_counts(self):
        records=[]
        for pair,n,ap in [('one',1,.2),('two',3,.8)]:
            for role in ['real','fake']:
                for i in range(n):records.append(dict(video_id=pair+role,pair_id=pair,role=role,pixel_AP=ap if role=='fake' else None,real_FPR=.1 if role=='real' else None,unique_history=0))
        result=aggregate(records)
        self.assertAlmostEqual(result['macro']['pixel_AP'],.5)
        self.assertAlmostEqual(result['macro']['real_FPR'],.1)
        self.assertIsNone(result['macro']['Dice'])

    def test_history_slots_and_distinct_observations_are_separate(self):
        row=dict(pts=[0.,.5,1.],evaluation_range=[0,3])
        info=history_info(row,2)
        self.assertEqual(info['H'],7);self.assertEqual(info['unique_history'],2)
        row['evaluation_range']=[2,3];self.assertEqual(history_info(row,2)['H'],0)

    def test_same_b_weights_h0_paths_match(self):
        model=XYZLocalization(.5).eval().requires_grad_(False);f=frame()
        with torch.no_grad():
            e=model.encode(f)
            a=model.relate_window([e],[0],[0],True);b=model.relate_window([e],[0],[0],False)
            torch.testing.assert_close(model.decode(f,e,a),model.decode(f,e,b),atol=0,rtol=0)
