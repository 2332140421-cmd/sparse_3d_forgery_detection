import unittest
import numpy as np
from PIL import Image
from scripts.prepare_geometry_screen import archive_key,motion_stats,ranges,camera_check,ain_mask

class GeometryScreenTest(unittest.TestCase):
    def test_archive_exact_allowed_source_variant(self):
        self.assertEqual(archive_key('DATASET_AInpaint/CoCoCo/432x240/boat/0000.png','CoCoCo','432x240',{'boat'}),('boat','0000.png'))
        for path in ['CoCoCo/432x240_postprocessed/boat/0000.png','CoCoCo/432x240/test_other/0000.png','input_frames/432x240/boat/0000.png']:
            self.assertIsNone(archive_key(path,'CoCoCo','432x240',{'boat'}))
    def test_motion_and_empty_ranges(self):
        a=np.zeros((20,30),bool);b=a.copy();a[2:4,3:5]=1;b[2:4,6:8]=1
        s=motion_stats([a,b]);self.assertAlmostEqual(s['centroid_span_pixels'],3);self.assertEqual(s['positive_frames'],[0,1]);self.assertEqual(ranges([0,1,4]),[[0,2],[4,5]])
        self.assertEqual(ranges([]),[])
    def test_camera_unreliable_not_moving(self):
        black=Image.fromarray(np.zeros((240,432,3),np.uint8));r=camera_check([(0,black),(10,black)])
        self.assertFalse(r['stable_candidate']);self.assertIn('不确定',r['state'])
    def test_camera_identical_textured(self):
        a=Image.fromarray(np.random.default_rng(7).integers(0,256,(240,432,3),dtype=np.uint8));r=camera_check([(0,a),(20,a)])
        self.assertTrue(r['stable_candidate']);self.assertAlmostEqual(r['intervals'][0]['rotation_degrees'],0,places=5)

    def test_soft_mask_matches_final_author_target(self):
        a=Image.fromarray(np.array([[0,127,128,255]],dtype=np.uint8))
        np.testing.assert_array_equal(ain_mask(a),[[False,False,True,True]])

if __name__=='__main__':unittest.main()
