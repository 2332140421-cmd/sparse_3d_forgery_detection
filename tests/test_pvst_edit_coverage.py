import unittest
import csv
import tempfile
from pathlib import Path
import torch
from scripts.audit_pvst_edit_coverage import counts,table


def frame():
    q=torch.zeros(2,3,3);q[...,2]=torch.tensor([[-.01,-.01,.13],[.13,.13,.26]])
    v=torch.tensor([[True,True,False],[True,True,True]])
    g=torch.tensor([[True,False,True],[False,True,False]])
    return {'xyz':q,'valid':v,'gt':g,'rgb':torch.zeros(2,3,3,dtype=torch.uint8)}


class CoverageTests(unittest.TestCase):
    def test_native_selection_counts_and_signed_absolute_slices(self):
        a,s=counts(frame())
        self.assertEqual((a['GT_pixels'],a['GT_selected_pixels'],a['GT_excluded_pixels']),(3,2,1))
        self.assertAlmostEqual(a['GT_pool_coverage'],2/3)
        self.assertEqual([r['absolute_slice_id'] for r in s],[-1,1])
        self.assertEqual(sum(r['edited_valid_pixels'] for r in s),2)
        self.assertEqual(sum(r['nonedited_valid_pixels'] for r in s),2)
        for r in s:self.assertEqual(r['all_valid_pixels'],r['edited_valid_pixels']+r['nonedited_valid_pixels'])

    def test_empty_gt_is_null_not_perfect_or_zero(self):
        f=frame();f['gt'].zero_();a,s=counts(f)
        self.assertIsNone(a['GT_pool_coverage']);self.assertEqual(s,[])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'empty.csv';table(p,[a])
            with p.open() as stream:self.assertEqual(next(csv.DictReader(stream))['GT_pool_coverage'],'null')

    def test_shape_mismatch_is_not_silently_resized(self):
        f=frame();f['gt']=torch.zeros(1,3,dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,'NATIVE_GT_VALID_RGB_SHAPE'):counts(f)

    def test_selected_xyz_finite_and_invalid_never_repaired(self):
        f=frame();f['xyz'][0,2]=torch.nan
        a,_=counts(f);self.assertEqual(a['selected_pixel_count'],5)
        f['xyz'][0,0]=torch.nan
        with self.assertRaisesRegex(ValueError,'NONFINITE_SELECTED_XYZ'):counts(f)


if __name__=='__main__':unittest.main()
