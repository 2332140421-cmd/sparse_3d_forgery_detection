import unittest
import torch
from research_tools.rgbd_slice.pooling_parity_check import difference,trace_pool
from research_tools.rgbd_slice.model import SlabPool
from research_tools.depth_slice.model import slab_ids

class DifferenceTest(unittest.TestCase):
    def test_unchanged_tolerance_and_coordinate(self):
        b=torch.tensor([[1.,0.],[2.,3.]]);a=b.clone();a[1,0]+=.001;x=difference(a,b,torch.tensor([-3,7]))
        self.assertFalse(x['passed']);self.assertEqual(x['mismatches'],1);self.assertEqual(x['max_absolute_element']['slab_id'],7);self.assertEqual(x['max_absolute_element']['index'],[1,0]);self.assertAlmostEqual(x['bad_elements_first'][0]['allowed'],2e-5+3e-5*2)
    def test_all_elements_recorded_before_pass_and_zero(self):
        a=torch.tensor([[1e-6,1.],[0.,-1.]]);b=torch.tensor([[0.,1.],[0.,-1.]])
        x=difference(a,b);self.assertTrue(x['passed']);self.assertFalse(x['bitwise_equal']);self.assertEqual(x['mismatches'],0)
    def test_passive_pool_stage_capture(self):
        torch.manual_seed(20260909);pool=SlabPool(rgb_only=True).eval();local=torch.randn(256,3,4);xyz=torch.randn(5,7,3);valid=torch.ones(5,7,dtype=torch.bool);ids,index=slab_ids(xyz,valid);points=torch.where(valid.reshape(-1))[0];affected=torch.arange(len(ids))
        a=trace_pool(pool,local,xyz,valid,points,ids,index,affected,False);b=trace_pool(pool,local,xyz,valid,points,ids,index,affected,True)
        self.assertTrue(a['points_equal']);self.assertTrue(a['routing_equal']);self.assertTrue(b['routing_equal'])
        for name in ['score','value','maximum','denominator','weights','output']:torch.testing.assert_close(a[name],b[name],atol=2e-5,rtol=3e-5)
        self.assertEqual(len(a['partials']),1)
    def test_partial_axis_mapping(self):
        a=torch.zeros(3,2,4);b=a.clone();a[2,1,3]=1.;x=difference(a,b,torch.tensor([4,9]),slab_axis=1)
        self.assertEqual(x['max_absolute_element']['slab_id'],9);self.assertEqual(x['max_absolute_element']['index'],[2,1,3])

if __name__=='__main__':unittest.main()
