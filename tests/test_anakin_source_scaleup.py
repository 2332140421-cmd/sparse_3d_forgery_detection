import unittest
from scripts.anakin_source_scaleup import source,protect,tier
class SourceContracts(unittest.TestCase):
    def test_source_alias(self):
        self.assertEqual(source('https://vimeo.com/123'),source('https://player.vimeo.com/video/123'))
        self.assertIsNone(source('unknown'))
    def test_test_quarantine(self):
        self.assertEqual(protect({'train','dev'}, {'train','test'}),'test_reserved')
        self.assertEqual(protect({'train','dev'}, {'train'}),'dev')
    def test_metadata_only_tiers(self):
        self.assertEqual(tier('add a cat'),'addition_copy')
        self.assertEqual(tier('remove running man'),'removal')
        self.assertEqual(tier('add tattoo'),'appearance_reference')

class StrictMapping(unittest.TestCase):
    def test_no_silent_crop_or_shift(self):
        from scripts.review_anakin import alignment,numeric_members
        self.assertEqual(alignment([0,1,2],3)['status'],'aligned')
        self.assertEqual(alignment([1,2,3],3)['index_base'],1)
        self.assertNotEqual(alignment([0,1,2],4)['status'],'aligned')
        self.assertNotEqual(alignment([9,10,11],3)['status'],'aligned')
        with self.assertRaises(ValueError):numeric_members(['a_0.png','b_0.png'])

class SelectionIsolation(unittest.TestCase):
    def test_group_not_record_is_quarantined(self):
        from scripts.anakin_source_scaleup import validate_selection
        rows=[{'source_group_id':'vimeo:1','role':'test_reserved'}]
        with self.assertRaises(AssertionError):
            validate_selection(rows,[{'source_group_id':'vimeo:1','role_conflict':False}])
        with self.assertRaises(AssertionError):
            validate_selection([], [{'source_group_id':'vimeo:2','role_conflict':False}]*2)

class ReviewPage(unittest.TestCase):
    def test_exact_ids_blank_fields_and_batch_scope(self):
        import json,tempfile
        from pathlib import Path
        from scripts import anakin_source_scaleup as module
        original=module.OUT
        with tempfile.TemporaryDirectory() as temp:
            module.OUT=Path(temp)
            (module.OUT/'human_evidence.json').write_text(json.dumps({'verbatim_sources':[]}))
            media={'frame_count':2,'pts':[{'seconds':0.0},{'seconds':0.1}]}
            record={'pair_id':'an:example','source_group_id':'vimeo:123','split':'candidate_unassigned','generator':'splicing','official_task':'add object','camera':'待确认','camera_auto':{'state':'待确认'},'edit_object':'待确认','relation_note':'待确认','human':None,'mask':{'status':'PASS','profile':'all_edited','positive_frames':2,'frames':2,'positive_runs_half_open':[[0,2]]},'sync_note':'未证明配准','media':{'real':media,'fake':media},'fixed_frames':[]}
            try:
                module.render([record]);page=(module.OUT/'index.html').read_text()
                self.assertIn('"pair": "an:example"',page)
                self.assertIn('batch_events:events',page)
                self.assertIn("el.value=data[name]||''",page)
                self.assertIn('jump-frame',page)
                self.assertNotIn('<textarea',page)
                self.assertNotIn('id="export"',page)
            finally:module.OUT=original

class EditorialCoverage(unittest.TestCase):
    def test_addition_does_not_imply_motion_object(self):
        from scripts.anakin_source_scaleup import content_priority
        self.assertEqual(content_priority('add a graph to monitor'),'appearance_background_reference')
        self.assertEqual(content_priority('add flying bird'),'object_edit_pending_human')
