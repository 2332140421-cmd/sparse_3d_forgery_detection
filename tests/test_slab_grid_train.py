import copy
from pathlib import Path
import tempfile
import unittest
from research_tools.rgbd_slice.slab_grid_train import assert_gate,archive_first,assert_same_frames

class FormalTrainingTest(unittest.TestCase):
    def gate(self):return {'status':'COMPLETE','cache_status':'PASS','resource_status':'PASS','resource_check':{'passed':True,'discarded_update':True,'parameter_changed':True,'optimizer_states':20,'dev_max_frame':{'SLAB_GRID_CURRENT_units':10},'train_max_frame':{'SLAB_GRID_CURRENT_units':20}},'extraction':{'frames':2076,'train_frames':1152,'dev_frames':924}}
    def test_PASS_proof_and_larger_dev_requirement(self):
        s=self.gate();assert_gate(s);s['resource_check']['dev_max_frame']['SLAB_GRID_CURRENT_units']=30
        with self.assertRaises(AssertionError):assert_gate(s)
        s['resource_check']['larger_dev_forward_seconds']=.1;assert_gate(s);s['resource_check']['parameter_changed']=False
        with self.assertRaises(AssertionError):assert_gate(s)
    def test_original_failed_exit_and_report_archive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'exit_code.txt').write_text('1\n');(p/'report.md').write_text('original failure');a=archive_first(p);(p/'report.md').write_text('formal result')
            self.assertEqual((Path(a['path'])/'report.md').read_text(),'original failure');self.assertEqual((p/'exit_code.txt').read_text(),'1\n')
    def test_tail_unique_and_pair_frame_identity(self):
        a={'frames':2,'frame_records':[{'video_id':'x','frame_index':i,'pts':i*.1,'valid_pixels':10,'valid_GT_pixels':2} for i in range(2)],'records':[{'video_id':'x','GT_pixels':4,'valid_GT_pixels':4,'GT_coverage':1.}]};b=copy.deepcopy(a);assert_same_frames(a,b,2);b['frame_records'][1]['frame_index']=0
        with self.assertRaises(AssertionError):assert_same_frames(a,b,2)

if __name__=='__main__':unittest.main()
