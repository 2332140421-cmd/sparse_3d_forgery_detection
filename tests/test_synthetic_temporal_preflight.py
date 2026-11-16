import unittest
from scripts.synthetic_temporal_preflight import eligible

class AssetGate(unittest.TestCase):
    def test_requires_all_three_assets(self):
        self.assertTrue(eligible(True, True, True))
        for flags in [(False, True, True), (True, False, True), (True, True, False)]:
            self.assertFalse(eligible(*flags))
