import ast
import unittest
from pathlib import Path
from research_tools.xyz_pilot import expanded_spatial_analysis as analysis

class SavedMetricsAnalysis(unittest.TestCase):
    def test_equal_video_weight_and_undefined_AP(self):
        ev={'videos':[dict(pair_id='an:a',role='fake',frames=100,pixel_AP=.2),dict(pair_id='an:b',role='fake',frames=1,pixel_AP=.8),dict(pair_id='an:c',role='fake',frames=10,pixel_AP=None),dict(pair_id='an:a',role='real',frames=100,real_FPR=.1),dict(pair_id='an:b',role='real',frames=1,real_FPR=.3)]}
        got=analysis.aggregate(ev,{'an:a','an:b','an:c'})
        self.assertAlmostEqual(got['pixel_AP'],.5);self.assertAlmostEqual(got['real_FPR'],.2)
        self.assertIsNone(got['Dice']);self.assertEqual(got['sources'],3)
    def test_official_action_not_clothing_keyword(self):
        self.assertEqual(analysis.official_action('remove the rider in the white hat along with the horse'),'removal')
        self.assertEqual(analysis.official_action('Remove the walking man in the blue shirt'),'removal')
        self.assertEqual(analysis.official_action('add a picture'),'addition_copy')
    def test_no_model_checkpoint_or_media_dependencies(self):
        tree=ast.parse(Path(analysis.__file__).read_text());names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(n.name for n in node.names)
            elif isinstance(node,ast.ImportFrom):names.append(node.module or '')
        self.assertFalse(any(n.startswith(('torch','av','cv2','research_tools.xyz_pilot.ainpaint')) for n in names))
        self.assertNotIn('torch.load',Path(analysis.__file__).read_text())
