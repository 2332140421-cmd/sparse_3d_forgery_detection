import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from research_tools.rgbd_slice.slab_grid_experiment import _reuse_video,_record_mismatch
from research_tools.rgbd_slice.grid_localization import grouping,CONDITIONS

class ResumeTest(unittest.TestCase):
    def fixture(self,p):
        f={'xyz':torch.zeros(3,5,3),'valid':torch.ones(3,5,dtype=torch.bool),'gt':torch.zeros(3,5,dtype=torch.bool),'pts':.25};f['gt'][0,0]=True;_,groups,_=grouping(f['xyz'],f['valid'])
        entry={'pts':.25,'GT_pixels':1,'valid_GT_pixels':1,'valid_pixels':15}
        for name,(ids,index) in zip(CONDITIONS,groups):entry[name]={'ids':ids,'counts':torch.bincount(index),'edit_counts':torch.bincount(index[f['gt'][f['valid']]],minlength=len(ids)),'features':torch.zeros(len(ids),512)}
        torch.save({0:entry},p);base={**entry[CONDITIONS[0]],'pts':.25};row={'video_id':'x','pair_id':'an:x','role':'fake','split':'train','source_hw':[3,5],'pts':[{'pts_s':.25}]}
        return f,entry,base,row
    def test_reuse_identity_and_no_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.pt';f,e,b,r=self.fixture(p);video=SimpleNamespace(frame=lambda i:f)
            rows,error=_reuse_video(p,r,[0],{0:b},video,{});self.assertEqual(len(rows),1);self.assertEqual(error,0)
    def test_reuse_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.pt';f,e,b,r=self.fixture(p);e[CONDITIONS[1]]['edit_counts'][0]+=1;torch.save({0:e},p)
            with self.assertRaises(AssertionError):_reuse_video(p,r,[0],{0:b},SimpleNamespace(frame=lambda i:f),{})
    def record(self,d,broken=False):
        ids=torch.tensor([3,9]);B=torch.ones(2,512);C=torch.zeros(2,512)
        def pool(*args):
            if broken:raise ValueError('recording failed')
            return C,ids
        _record_mismatch(SimpleNamespace(pool=pool),torch.zeros(1),torch.zeros(1),torch.ones(1,dtype=torch.bool),ids,B,C,{'video_id':'x','pair_id':'an:x','role':'fake','source_hw':[1,1]},{'frame_index':7,'pts':.2},Path(d),AssertionError('PRIMARY'))
    def test_live_ABC_and_primary_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.record(d);e=json.loads((Path(d)/'same_call_ABC.json').read_text());self.assertEqual(e['additional_encoder_calls'],0);self.assertEqual(e['identity']['frame_index'],7);self.assertEqual(e['comparisons']['A_minus_C']['mismatches'],0)
            self.assertTrue((Path(d)/'mismatch_B_C.pt').exists())
    def test_record_failure_preserves_primary_and_BC(self):
        with tempfile.TemporaryDirectory() as d:
            self.record(d,True);self.assertEqual((Path(d)/'original_mismatch_error.txt').read_text(),'PRIMARY');self.assertTrue((Path(d)/'mismatch_B_C.pt').exists());self.assertIn('PRIMARY',(Path(d)/'evidence_recording_error.json').read_text())

if __name__=='__main__':unittest.main()
