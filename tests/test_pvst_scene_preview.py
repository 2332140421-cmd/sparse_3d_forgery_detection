import unittest
import numpy as np
from scripts.review_pvst_scenes import clip_timing,overlay,candidates
from scripts.review_anakin import numeric_members,alignment


class ScenePreviewTests(unittest.TestCase):
    def test_duration_includes_final_frame_and_actual_pts(self):
        a=clip_timing([i*.04 for i in range(75)],0,24)
        self.assertAlmostEqual(a['PTS_span_s'],.92)
        self.assertAlmostEqual(a['presentation_duration_s'],.96)
        b=clip_timing([0.,.03,.071,.12],1,4)
        self.assertAlmostEqual(b['presentation_duration_s'],.139)

    def test_official_mask_order_not_zip_enumeration(self):
        m=numeric_members(['frame_10.png','frame_2.png','frame_1.png'])
        self.assertEqual([x[0] for x in m],[1,2,10])
        self.assertEqual(alignment([1,2,3],3)['index_base'],1)
        self.assertEqual(alignment([0,1,3],3)['status'],'alignment_unresolved')

    def test_overlay_does_not_expand_gt_or_change_background(self):
        rgb=np.full((3,4,3),100,np.uint8);gt=np.zeros((3,4),bool);gt[1,2]=True
        out=overlay(rgb,gt)
        np.testing.assert_equal(out[~gt],rgb[~gt]);self.assertTrue((out[gt]!=rgb[gt]).any())

    def test_candidates_stable_and_training_identity_only(self):
        rows=[{'pair_id':x} for x in ['an:40y1x9','an:95qvg9','an:4wdo37','an:7ogg04']]
        meta={x['pair_id'][3:]:{'task':'remove a person walking'} for x in rows}
        a=candidates(rows,meta);b=candidates(list(reversed(rows)),meta)
        self.assertEqual(a,b);self.assertEqual(len(a),2);self.assertNotIn('an:7ogg04',a)


if __name__=='__main__':unittest.main()
