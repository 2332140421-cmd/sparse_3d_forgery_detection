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

class OfficialAction(unittest.TestCase):
    def test_clothing_is_not_the_operation(self):
        from research_tools.xyz_pilot.expanded_spatial import official_action
        self.assertEqual(official_action('remove the man in the white shirt'),'removal')
        self.assertEqual(official_action('remove the rider in the white hat along with the horse'),'removal')
        self.assertEqual(official_action('add a picture to the wall'),'addition_copy')

class CompletionIntegrity(unittest.TestCase):
    def test_missing_exposure_cannot_publish_complete(self):
        import tempfile,json,torch
        from pathlib import Path
        from research_tools.xyz_pilot import expanded_spatial as module
        prior=module.OUT
        with tempfile.TemporaryDirectory() as temp:
            module.OUT=Path(temp);row=dict(video_id='v',pair_id='an:x',role='fake',split='train',train_targets=[0,1]);summary=dict(protocol=dict(rows=[row],dev_frames=1),updates=20,identity={'test':1},reference={'frame_identity_sha256':'same'})
            initial={'head.weight':torch.zeros(1),**{f'bn{k}.num_batches_tracked':torch.tensor(0) for k in range(5)}};torch.save(initial,module.OUT/'initial_adaptation.pt')
            lines=[json.dumps(dict(video_id='v',target=i)) for _ in range(10) for i in [0,1]];(module.OUT/'updates.jsonl').write_text('\n'.join(lines)+'\n')
            for epoch in module.EVAL_EPOCHS:
                cp=dict(condition=module.NAME,epoch=epoch,updates=epoch*2,identity=summary['identity'],adaptation_state={'head.weight':torch.ones(1),**{f'bn{k}.num_batches_tracked':torch.tensor(epoch*102) for k in range(5)}},optimizer={'state':{0:{'step':torch.tensor(epoch*2)}}},random_state={x:None for x in ['python','numpy','torch','cuda']});torch.save(cp,module.OUT/f'epoch{epoch:02d}.pt')
            (module.OUT/'evaluations.json').write_text(json.dumps({str(e):{'evaluation':{'frame_identity_sha256':'same','scored_frames':1}} for e in module.EVAL_EPOCHS}))
            try:
                module.completed_checks(summary);self.assertEqual(json.loads((module.OUT/'verification.json').read_text())['status'],'PASS')
                (module.OUT/'updates.jsonl').write_text('\n'.join(lines[:-1])+'\n')
                with self.assertRaises(AssertionError):module.completed_checks(summary)
            finally:module.OUT=prior
