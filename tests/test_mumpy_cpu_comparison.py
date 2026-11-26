"""Saved-statistics arithmetic only, intentionally no torch/model imports."""
import unittest
from research_tools.mumpy_anakin.double_comparison import augmentation_totals,directions
class CPUComparisonTests(unittest.TestCase):
    def test_augmentation_weights_and_operation_conservation(self):
        def rec(n,b,a,empty):return dict(targets=n,augmentation_seconds=.2,operation_counts={'HFlip':n,'RandomCrop':n},fake_nonempty=dict(targets=n,before_fraction_sum=b,after_fraction_sum=a,became_empty=empty))
        x=augmentation_totals([rec(1,.1,.2,0),rec(3,.6,.9,1)])
        self.assertEqual(x['operation_total'],8)
        self.assertAlmostEqual(x['groups']['fake_nonempty']['before_fraction_mean'],.175)
        self.assertAlmostEqual(x['groups']['fake_nonempty']['became_empty_ratio'],.25)
        bad=rec(1,.1,.2,0);bad['operation_counts']['RandomCrop']=0
        with self.assertRaises(AssertionError):augmentation_totals([bad])
    def test_direction_counts_do_not_hide_losses_or_small_changes(self):
        a={p:{'pixel_AP':.2} for p in 'abc'};b={p:{'pixel_AP':v} for p,v in zip('abc',[.4,.1,.2+1e-10])}
        x=directions(a,b,set(a))
        self.assertEqual((x['up'],x['down'],x['near']),(1,1,1))
        self.assertLess(x['without_largest_gain_mean'],0)
if __name__=='__main__':unittest.main()
