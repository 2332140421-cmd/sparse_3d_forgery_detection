"""Research-critical CPU aggregation/identity regressions, no model imports."""
import unittest
from research_tools.mumpy_anakin.analysis import sources, signature, KEYS
class AnalysisTest(unittest.TestCase):
    def test_real_not_pooled_into_fake_AP(self):
        ev={'videos':[dict(pair_id='p',role='fake',**{k:.2 for k in KEYS}),dict(pair_id='p',role='real',**{k:.8 for k in KEYS})]}
        s=sources(ev)['p'];self.assertEqual(s['pixel_AP'],.2);self.assertEqual(s['real_FPR'],.8)
    def test_duplicate_frames_rejected(self):
        f=dict(video_id='v',frame_index=0,pts=0.,GT_pixels=1,native_pixels=2)
        with self.assertRaises(AssertionError):signature({'frames':[f]*5152})
if __name__=='__main__':unittest.main()
