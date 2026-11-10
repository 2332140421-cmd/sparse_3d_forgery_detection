import unittest

import numpy as np

from research_tools.rgbd_slice.gru_readout_probe import align, fit_probe, score, validate_features


class TestGRUReadoutProbe(unittest.TestCase):
    def test_frozen_feature_shape_and_finiteness(self):
        x = np.zeros((1152 + 924, 2, 512), dtype=np.float32)
        validate_features(x, 512)
        with self.assertRaisesRegex(ValueError, "Expected GRU feature shape"):
            validate_features(x[:-1], 512)
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_features(x, 512)

    def test_fit_uses_only_supplied_training_rows(self):
        train = np.array([[-2.0, 5.0], [-1.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
        model = fit_probe(train, [0, 0, 1, 1])
        np.testing.assert_array_equal(model[0].mean_, [0.0, 5.0])
        before = model.decision_function(train)
        model.decision_function(np.array([[1000.0, -800.0]]))
        np.testing.assert_array_equal(model.decision_function(train), before)
        np.testing.assert_array_equal(before >= 0, [False, False, True, True])

    def test_frame_label_and_video_role_are_scored_separately(self):
        rows = [
            dict(video_id="r", pair_id="p", role="real", label=False),
            dict(video_id="f", pair_id="p", role="fake", label=False),
            dict(video_id="f", pair_id="p", role="fake", label=True),
        ]
        result = score(rows, [0.0, 0.0, float(np.log(3))])
        self.assertAlmostEqual(result["pairs"]["p"]["fake_minus_real"], np.log(2))
        self.assertEqual(result["frame"], {"auroc": 1.0, "ap": 1.0})
        self.assertEqual(result["video"], {"auroc": 1.0, "ap": 1.0})

    def test_alignment_rejects_repeated_ids_or_changed_logits(self):
        rows = {4: {"label": True, "head_logit": 2.0},
                3: {"label": False, "head_logit": -1.0}}
        record = {"labels": [False, True], "scores": [-1.0, 2.0]}
        self.assertEqual(align(rows, record, [3, 4]), [rows[3], rows[4]])
        with self.assertRaises(ValueError):
            align(rows, record, [3, 4, 4])
        with self.assertRaises(ValueError):
            align(rows, {**record, "scores": [-1.0, 3.0]}, [3, 4])


if __name__ == "__main__":
    unittest.main()
