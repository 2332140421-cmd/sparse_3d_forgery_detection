import unittest
import numpy as np
from research_tools.rgbd_slice.cat_boundary_evidence import regions,crop_box,paint
from matplotlib.colors import Normalize

class CatBoundaryTest(unittest.TestCase):
    def test_bands_counts_and_empty(self):
        gt=np.zeros((15,15),bool);gt[4:11,4:11]=1;valid=np.ones_like(gt);valid[5,5]=0
        r=regions(gt,valid,1)
        np.testing.assert_array_equal(r['GT_inner_band']|r['GT_interior'],gt&valid)
        self.assertFalse((r['outside_adjacent']&gt).any());self.assertFalse((r['GT_inner_band']&r['GT_interior']).any())
        self.assertEqual(int(r['GT_inner_band'].sum()),24)
        for mask in regions(np.zeros_like(gt),valid,1).values():self.assertFalse(mask.any())
    def test_crop_and_invalid_display(self):
        gt=np.zeros((10,20),bool);gt[0:2,18:20]=1
        self.assertEqual(crop_box(gt,3),[15,0,20,5]);self.assertIsNone(crop_box(np.zeros_like(gt),3))
        v=np.array([[False,True]]);image=paint(np.array([[np.nan,.5]]),v,Normalize(0,1),'viridis')
        np.testing.assert_array_equal(image[0,0],[255,0,255])

if __name__=='__main__':unittest.main()
