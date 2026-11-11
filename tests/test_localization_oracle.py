import itertools
import unittest
import numpy as np
import torch
from research_tools.rgbd_slice.localization_oracle import optimum,partitions

class OracleTest(unittest.TestCase):
    def test_exact_prefix_against_exhaustive(self):
        rng=np.random.default_rng(20260909)
        for _ in range(80):
            n=rng.integers(1,15,size=int(rng.integers(1,9)));g=np.array([rng.integers(0,x+1) for x in n]);G=int(g.sum())
            result,chosen=optimum(n,g)
            if not G:self.assertIsNone(result);continue
            bestD=bestI=0
            for flags in itertools.product([False,True],repeat=len(n)):
                flags=np.array(flags);TP=int(g[flags].sum());P=int(n[flags].sum());bestD=max(bestD,2*TP/(G+P));bestI=max(bestI,TP/(G+P-TP))
            self.assertAlmostEqual(result['Dice'],bestD,places=14);self.assertAlmostEqual(result['IoU'],bestI,places=14)
            self.assertEqual(result['TP'],int(g[chosen].sum()));self.assertEqual(result['prediction_pixels'],int(n[chosen].sum()))
    def test_empty_GT_and_empty_valid(self):
        result,chosen=optimum([2,3],[0,0]);self.assertIsNone(result);self.assertFalse(chosen.any())
        xyz=torch.zeros(2,3,3);valid=torch.zeros(2,3,dtype=torch.bool);gt=~valid
        out,L=partitions(xyz,valid,gt)
        for q in out.values():self.assertEqual(q['units'],0);self.assertIsNone(q['result'])
    def test_negative_floor_boundary_and_refinement(self):
        xyz=torch.zeros(17,19,3);xyz[...,2]=.125;xyz[0,0,2]=-.01;xyz[0,1,2]=0
        valid=torch.ones(17,19,dtype=torch.bool);valid[-1,-1]=False;gt=torch.zeros_like(valid);gt[0,0]=True;gt[5:7,5:7]=True;gt[-1,-1]=True
        out,L=partitions(xyz,valid,gt);self.assertEqual(L,2);self.assertEqual(out['SLAB']['units'],3)
        for name,q in out.items():
            self.assertEqual(len(q['inverse']),int(valid.sum()));self.assertGreaterEqual(q['result']['TP'],1)
        self.assertGreaterEqual(out['SLAB_GRID']['result']['Dice'],out['SLAB']['result']['Dice'])
    def test_optimum_not_all_GT_touched(self):
        result,chosen=optimum([100,1],[1,1]);self.assertEqual(chosen.tolist(),[False,True]);self.assertAlmostEqual(result['Dice'],2/3)

if __name__=='__main__':unittest.main()
