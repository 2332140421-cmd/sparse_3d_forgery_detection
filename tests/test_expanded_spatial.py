import unittest
from research_tools.xyz_pilot.expanded_spatial import split_for,aggregate
from research_tools.xyz_pilot.insource_probe import epoch_orders
class FrozenProtocol(unittest.TestCase):
    def row(self,**kw):return dict(source_group_id='vimeo:1',role_conflict=False,role='candidate_unassigned',human_mask_status='unconfirmed',technical_status='PASS',historical_roles=[],**kw)
    def test_unknown_human_is_authorized_not_approved(self):
        r=self.row();self.assertEqual(split_for(r,'train_draft'),'train');self.assertEqual(r['human_mask_status'],'unconfirmed')
    def test_test_dev_negative_and_mapping_boundaries(self):
        for change,proposal in [({'role':'test_reserved'},'train_draft'),({'historical_roles':['dev']},'train_draft'),({'human_mask_status':'not_corresponding'},'dev_draft'),({'technical_status':'UNRESOLVED'},'train_draft')]:
            r=self.row();r.update(change)
            with self.assertRaises(AssertionError):split_for(r,proposal)
    def test_all_frames_once_every_epoch(self):
        rows=[dict(video_id='a',train_targets=[0,1,4]),dict(video_id='b',train_targets=[0,1])];orders=epoch_orders(rows,10)
        self.assertEqual(orders,epoch_orders(rows,10))
        for sequence in orders:self.assertEqual({(x['video_id'],x['target']) for x in sequence},{('a',0),('a',1),('a',4),('b',0),('b',1)});self.assertEqual(len(sequence),5)

class MacroMetrics(unittest.TestCase):
    def test_source_equal_weight_and_undefined_AP(self):
        from research_tools.xyz_pilot.expanded_spatial import aggregate
        ev={'videos':[{'pair_id':'an:a','role':'fake','frames':100,'pixel_AP':.2,'constant_AP':.1},{'pair_id':'an:b','role':'fake','frames':1,'pixel_AP':.8,'constant_AP':.7},{'pair_id':'an:c','role':'fake','frames':5,'pixel_AP':None,'constant_AP':None}]}
        got=aggregate(ev,{'a','b','c'});self.assertAlmostEqual(got['pixel_AP'],.5);self.assertEqual(got['sources'],3)
