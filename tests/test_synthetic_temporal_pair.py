import unittest
import numpy as np
from scripts.synthetic_temporal_pair import assets, composite, check_arrays, MAPPING

class PairChecks(unittest.TestCase):
    def test_pair_and_corruption(self):
        bg,layers=assets();a=[composite(bg,l) for l in layers];b=[a[s].copy() for s in MAPPING]
        ma=[l[...,3]>0 for l in layers];mb=[ma[s] for s in MAPPING]
        self.assertEqual(check_arrays(a,b,ma,mb,[a[s] for s in MAPPING])['status'],'PASS')
        b[4][0,0]=0
        r=check_arrays(a,b,ma,mb,[a[s] for s in MAPPING])
        self.assertEqual(r['status'],'FAILED')
        self.assertIn('outside_alpha_union',[x['check'] for x in r['failures']])

    def test_alpha_rounding(self):
        bg=np.full((1,3,3),100,np.uint8);layer=np.zeros((1,3,4),np.uint8)
        layer[...,:3]=200;layer[...,3]=[0,128,255]
        self.assertEqual(composite(bg,layer)[0,:,0].tolist(),[100,150,200])
