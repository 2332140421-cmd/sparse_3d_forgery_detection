import copy
import unittest
from research_tools.rgbd_slice.rgb_partition_anakin_dev import validate_split, epoch_order, source_key, augment_evaluation


class ProtocolTests(unittest.TestCase):
    def fixture(self):
        rows=[];clips=[]
        for split,count in [('train',8),('dev',4)]:
            for pair in range(count):
                source=f'{split}{pair}'
                for role in ['fake','real']:
                    vid=source+role
                    rows.append(dict(split=split,pair_id=source,role=role,video_id=vid,source_id=source,
                                     video_sha256=vid,original_url='https://vimeo.com/'+source,actual_frames=30))
                    if split=='train':
                        clips.extend(dict(video_id=vid,start=s,end=s+24) for s in [0,3,6])
        return rows,clips

    def test_coverage_and_deterministic_order(self):
        rows,clips=self.fixture();v=validate_split(rows,clips)
        self.assertEqual(v['unique_training_clip_frames'],16*30)
        self.assertEqual(v['frame_presentations_per_epoch'],1152)
        a=epoch_order(clips,20260909,1);b=epoch_order(clips,20260909,1)
        self.assertEqual(a,b);self.assertNotEqual(a,epoch_order(clips,20260909,2))
        self.assertEqual(len(clips),48)

    def test_source_overlap_and_role_conflict_rejected(self):
        rows,clips=self.fixture();changed=copy.deepcopy(rows)
        for row in changed:
            if row['pair_id']=='dev0':row['source_id']='train0'
        with self.assertRaisesRegex(ValueError,'SOURCE_LEAKAGE'):validate_split(changed,clips)
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['pair_id']=='dev0':row['original_url']='http://www.vimeo.com/train0/'
        with self.assertRaisesRegex(ValueError,'SOURCE_URL_LEAKAGE'):validate_split(changed,clips)
        self.assertEqual(source_key('https://vimeo.com/1'),source_key('http://www.vimeo.com/1/'))

    def test_fp32_bce_frame_weighting(self):
        x={'records':[dict(scores=[0.],labels=[0],frames=1),dict(scores=[0.,0.,0.],labels=[1,1,1],frames=3)],'metrics':{}}
        v=augment_evaluation(x)
        self.assertAlmostEqual(v['metrics']['frame_bce_fp32'],0.69314718,places=6)


if __name__=='__main__':unittest.main()
