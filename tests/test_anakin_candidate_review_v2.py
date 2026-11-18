import unittest
from scripts.anakin_candidate_review_v2 import group,object_candidate,source_role

class ReviewV2(unittest.TestCase):
    def test_human_and_source_exclusions(self):
        self.assertEqual(group('an:95qlg9'),'pending_verification')
        self.assertEqual(group('an:40y1x9'),'main_candidate')
        self.assertEqual(source_role({'train','test_reserved'}),'excluded_test')
        self.assertEqual(source_role({'train','dev'}),'excluded_role_conflict')
        self.assertEqual(source_role(set()),'candidate_unassigned')
        self.assertTrue(object_candidate('remove people walking behind'))
        self.assertFalse(object_candidate('add tattoo on chest'))

    def test_published_mask_count(self):
        import tempfile,zipfile
        from pathlib import Path
        from scripts.anakin_candidate_review_v2 import published_mask_count
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('0000.png',b'');z.writestr('0001.png',b'')
            self.assertEqual(published_mask_count(p),2)

    def test_dimension_failure_stays_unknown(self):
        import tempfile,zipfile,io
        import numpy as np
        from pathlib import Path
        from PIL import Image
        from scripts.anakin_candidate_review_v2 import checked_mask
        from scripts.prepare_v8_assets import digest
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.zip';b=io.BytesIO();Image.fromarray(np.zeros((2,2),np.uint8)).save(b,format='PNG')
            with zipfile.ZipFile(p,'w') as z:z.writestr('0000.png',b.getvalue())
            row={'pair_id':'an:x','mask_zip':str(p),'mask_sha256':digest(p),'source_hw':[3,3]}
            r=checked_mask(row,{'frame_count':1})
            self.assertEqual(r['status'],'UNRESOLVED');self.assertIn('MASK_SIZE',r['technical_error'])
