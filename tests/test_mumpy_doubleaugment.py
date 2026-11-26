"""Only contracts specific to the official augmentation comparison."""
import copy,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from research_tools.mumpy_anakin import data,experiment as run
from research_tools.mumpy_anakin.model import augmentation
from research_tools.mumpy_anakin.double_augment import observe
from research_tools.mumpy_anakin.double_comparison import compare,KEYS

class DoubleTests(unittest.TestCase):
    def test_official_distribution_and_every_shape_path_is_shared_categorical(self):
        a=augmentation('double');single=augmentation();self.assertEqual(type(single.strategy).__name__,'OneAugmentStrategy')
        self.assertEqual(type(a.policy.strategy).__name__,'DoubleAugmentStrategy')
        self.assertEqual([(getattr(op,'__name__'),lo,hi) for op,lo,hi in a.policy.none_shape_change_augment_list],[('HFlip',1,1),('VFlip',1,1),('PsccAug',0,1)])
        shape=a.policy.shape_change_augment_list;self.assertEqual([(lo,hi) for _,lo,hi in shape],[(20,220),(0,180),(20,220),(20,220)])
        mask=np.zeros((224,224),np.uint8);mask[50:150,70:170]=255
        for empty in [False,True]:
            y=mask*int(not empty);im=Image.fromarray(np.repeat(y[...,None],3,axis=2))
            for op,lo,hi in shape:
                random.seed(23);np.random.seed(23)
                images,gt=op(([im.copy() for _ in range(3)],Image.fromarray(y)),(lo+hi)/2)
                self.assertEqual(len(images),3)
                for img in images[1:]:self.assertTrue(np.array_equal(images[0],img))
                self.assertTrue(set(np.unique(gt))<={0,255})
                x=np.asarray(images[0])[...,0]>127;m=np.asarray(gt)>0
                if not empty:self.assertGreater((x&m).sum()/max(1,(x|m).sum()),.94)
                else:self.assertFalse(m.any())
        images,gt=a([im.copy() for _ in range(3)],Image.fromarray(y));self.assertEqual(len(a.ops),2)

    def test_stats_empty_crops_are_not_resampled(self):
        s={};observe(s,[('OriginalRandomCrop',20,220)],'fake',.1,0.,.02);observe(s,[],'real',0.,0.,.01)
        self.assertEqual(s['fake_nonempty']['became_empty'],1);self.assertEqual(s['real_empty']['targets'],1)
        self.assertEqual(s['operation_counts']['OriginalRandomCrop'],1)

    def test_only_augmentation_config_changes_and_identical_orders(self):
        config=data.read(run.REPO/'docs/pvst/anakin_mumpy_doubleaugment_config_v1.json')
        old=Path(config['single_reference_root'])/'formal'
        run.verify_reference(old,config)
        changed=copy.deepcopy(config);changed['encoder_lr']=.3
        with self.assertRaises(AssertionError):run.verify_reference(old,changed)
        run.check_initialization(data.read(old/'initialization.json')['model_sha256'],config)
        with self.assertRaises(AssertionError):run.check_initialization('different',config)

    def test_readonly_preparation_missing_never_calls_prepare(self):
        with tempfile.TemporaryDirectory() as t,patch.object(data,'prepare_video',side_effect=AssertionError('no write')):
            with self.assertRaisesRegex(AssertionError,'READ_ONLY_CACHE_MISSING'):run.prepare(Path(t),[dict(video_id='missing')],read_only=True)

    def test_readonly_existing_cache_eval_never_augments(self):
        config=data.read(run.REPO/'docs/pvst/anakin_mumpy_doubleaugment_config_v1.json');root=Path(config['single_reference_root'])
        p=data.read(root/'formal/protocol.json');row=next(r for r in p['rows'] if r['video_id']=='40y1x9_real')
        marker=root/'rgb224'/row['video_id']/'complete.json';before=marker.read_bytes();mtime=marker.stat().st_mtime_ns
        with tempfile.TemporaryDirectory() as t:
            result=run.prepare(root,[row],read_only=True,record_root=Path(t));self.assertEqual(result['new_RGB224_GT_frames'],0)
        self.assertEqual(before,marker.read_bytes());self.assertEqual(mtime,marker.stat().st_mtime_ns)
        with patch.object(data,'augmentation',side_effect=AssertionError('eval must not augment')):
            cache=data.Clips(root,[row]);x,y=cache.get((row['video_id'],0));self.assertEqual(x.shape,(3,3,224,224));self.assertFalse(y.any())

    def test_cpu_comparison_uses_saved_values_and_all_sources(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old/formal';new=root/'new';old.mkdir(parents=True);new.mkdir()
            (new/'config.json').write_text(json.dumps(dict(single_reference_root=str(old.parent))))
            for folder,n in [(old,.2),(new,.3)]:
                ev=dict(groups={'新增15开发来源':{k:n for k in KEYS}},sources=[dict(pair_id='p',**{k:n for k in KEYS})])
                (folder/'epoch01_dev.json').write_text(json.dumps(ev))
            compare(new)
            self.assertIn('0.099999',(new/'comparison/double_minus_single_sources.csv').read_text())
            self.assertTrue((new/'comparison/report.md').is_file())
if __name__=='__main__':unittest.main()
