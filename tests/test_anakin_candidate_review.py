import unittest
from scripts.anakin_candidate_review import merge_rows
from scripts.review_anakin import alignment
from scripts.prepare_geometry_screen import ranges

class CandidateIdentity(unittest.TestCase):
    def test_source_split_and_annotation_gaps(self):
        def r(role,split='train'):return {'pair_id':'an:x','role':role,'source_id':'url:x','split':split,'video_id':role,'video_sha256':role,'actual_frames':3,'source_hw':[2,2],'pts':[0,1,2]}
        a=[r('real'),r('fake')];self.assertEqual(len(merge_rows(a,a)),1)
        with self.assertRaises(AssertionError):merge_rows(a,[r('fake','dev')])
        self.assertEqual(alignment([1,2,3],3)['index_base'],1)
        self.assertEqual(alignment([0,2],3)['status'],'alignment_unresolved')
        self.assertEqual(ranges([1,2,4]),[[1,3],[4,5]])
