import unittest
import numpy as np
from scripts.grip_temporal_controls import compare_known_background,timelines_equal,eligibility

class TemporalAssetTest(unittest.TestCase):
    def test_background_excludes_only_supplied_GT(self):
        a=np.zeros((4,5,3),np.uint8);b=a.copy();mask=np.zeros((4,5),bool);mask[1,1]=True;b[1,1]=255
        self.assertEqual(compare_known_background(a,b,mask)['nonzero_pixels'],0)
        b[0,0]=3;r=compare_known_background(a,b,mask);self.assertEqual(r['nonzero_pixels'],1);self.assertEqual(r['pixels'],19)
    def test_pts_not_equal_from_count_alone(self):
        a={'frame_count':2,'sizes_hw':[(2,3)],'pts':[{'seconds':0.},{'seconds':.1}]};b={**a,'pts':[{'seconds':0.},{'seconds':.2}]}
        self.assertTrue(timelines_equal([a,a]));self.assertFalse(timelines_equal([a,b]))
    def test_undocumented_operation_not_strict(self):
        r=eligibility(True,True,True,True,False);self.assertFalse(r['strict_temporal_control']);self.assertFalse(r['target_only_pixel_GT'])
        self.assertTrue(eligibility(True,True,True,True,True)['strict_temporal_control'])

if __name__=='__main__':unittest.main()
