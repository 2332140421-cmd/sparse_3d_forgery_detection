"""Research-critical arithmetic checks for saved-result CPU analysis only."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('matched_analysis', Path(__file__).parents[1] / 'scripts/analyze_pvst_fullvideo_matched_localization_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SavedResultAnalysisTests(unittest.TestCase):
    def test_missing_metrics_remain_missing(self):
        self.assertEqual(m.difference({'x': None, 'y': 3}, {'x': 1, 'y': 2}, ['x', 'y']), {'x': None, 'y': 1})
        self.assertIsNone(m.average([None, None]))

    def test_source_equal_weight_and_influence(self):
        rows = [{'pair_id': 'a', 'pixel_AP': .09}, {'pair_id': 'b', 'pixel_AP': -.03}, {'pair_id': 'c', 'pixel_AP': 0}]
        self.assertAlmostEqual(m.average(r['pixel_AP'] for r in rows), .02)
        self.assertEqual({k: m.directions(rows, 'pixel_AP')[k] for k in ['up', 'down', 'same']}, {'up': 1, 'down': 1, 'same': 1})
        c = m.contribution(rows)
        self.assertEqual(c['largest_positive'], 'a')
        self.assertAlmostEqual(c['contribution_to_full_mean'], .03)
        self.assertAlmostEqual(c['remaining_mean'], -.015)

    def test_duplicate_or_changed_frame_identity_rejected(self):
        row = dict(video_id='v', source_group_id='s', evaluation_targets=[0], pts=[0.0], native_hw=[2, 2])
        frame = dict(video_id='v', source_group_id='s', frame_index=0, pts=0.1, native_pixels=4)
        with self.assertRaises(AssertionError):
            m.validate_evaluation(dict(frames=[frame], scored_frames=1), [row])


if __name__ == '__main__':
    unittest.main()
