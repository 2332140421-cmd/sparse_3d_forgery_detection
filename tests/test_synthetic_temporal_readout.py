import unittest
import torch
from research_tools.rgbd_slice.synthetic_temporal_readout import aligned_history

class TemporalDistance(unittest.TestCase):
    def test_alignment_order_and_zero(self):
        e={i:{'keys':torch.tensor([[1,0,0],[3,1,1]]),'features':torch.tensor([[1.,float(i)],[0.,0.]]),'counts':torch.tensor([2,1])} for i in range(4)}
        a=aligned_history(e,[0,1,2,3]);b=aligned_history(e,[0,2,1,3])
        self.assertEqual(a['common_keys'],1);self.assertEqual(a['excluded_zero_norm_keys'],1)
        self.assertGreater(b['total'],a['total'])
        self.assertEqual(aligned_history(e,[0,1,2,3]),a)
        e[2]['keys'][0,0]=9
        self.assertIsNone(aligned_history(e,[0,1,2,3])['total'])
