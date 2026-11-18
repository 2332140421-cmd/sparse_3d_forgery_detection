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
