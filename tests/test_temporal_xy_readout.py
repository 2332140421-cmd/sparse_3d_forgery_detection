import unittest
import torch
from research_tools.rgbd_slice.temporal_xy_readout import collapse_xy
from research_tools.rgbd_slice.synthetic_temporal_readout import aligned_history

class XYReadout(unittest.TestCase):
    def test_weighted_collapse_and_keys(self):
        x=torch.zeros(3,512);x[:,0]=torch.tensor([1.,3.,5.])
        e={'keys':torch.tensor([[8,2,4],[9,2,4],[8,3,4]]),'features':x,'counts':torch.tensor([1,3,2]),'valid_pixels':6,'L':23,'source_frame':0}
        a=collapse_xy(e)
        self.assertEqual(a['keys'].tolist(),[[2,4],[3,4]])
        self.assertEqual(a['counts'].tolist(),[4.,2.]);self.assertEqual(a['features'][:,0].tolist(),[2.5,5.])
        entries={i:a for i in range(4)}
        r=aligned_history(entries,[0,1,2,3]);self.assertEqual(r['common_keys'],2);self.assertEqual(r['total'],0)
        self.assertTrue(all(c['pixel_coverage']==1 for c in r['coverage']))
