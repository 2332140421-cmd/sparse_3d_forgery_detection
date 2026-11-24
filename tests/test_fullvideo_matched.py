"""Research-critical data, macro metrics, continuation and fallback contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from research_tools.xyz_pilot import fullvideo_matched_data as data
from research_tools.xyz_pilot import fullvideo_matched as run
from research_tools.xyz_pilot.fullvideo_direct_model import FullVideoDirect,video_loss,prepare_frame,SCALE
from research_tools.xyz_pilot.fullvideo_direct_probe import seed,state_hash


def tiny_frames():
    fs=[]
    for i in range(2):
        q=torch.zeros(4,5,3);q[...,0]=torch.arange(5)[None]*SCALE*.3;q[...,2]=SCALE
        fs.append(prepare_frame(dict(xyz_provider=q,valid=torch.ones(4,5,dtype=torch.bool),work_hw=[4,5],
             dino_padded_hw=[14,14],patch=torch.randn(1024,1,1),detail=torch.randn(32,2,2),
             z0=torch.randn(4,5),gt_work=torch.ones(4,5,dtype=torch.bool),supervised=True,pts=i*.1)))
    return fs


class MatchedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(20260909)

    def test_frozen_protocol_budget_orders_and_observer_selection(self):
        p=data.read(data.SOURCE/'protocol.json');data.validate_protocol(p)
        orders,obs=data.orders_and_monitor(p['rows']);again=data.orders_and_monitor(p['rows'])
        self.assertEqual((orders,obs),again);self.assertEqual(len(obs),4)
        expected={r['video_id'] for r in p['rows'] if r['split']=='train'}
        self.assertTrue(all(len(o)==178 and set(o)==expected for o in orders))
        self.assertEqual(sum(len(r['train_targets']) for r in p['rows'])*3,68634)
        self.assertEqual(sum(len(r['preview_targets']) for r in p['rows'] if r['split']=='dev'),54)
        self.assertTrue(set(obs)<=expected)
        broken=copy.deepcopy(p);broken['rows'][0]['source_group_id']=next(r['source_group_id'] for r in broken['rows'] if r['split']=='train')
        with self.assertRaises(AssertionError):data.validate_protocol(broken)

    def test_compatibility_ignores_enclosing_manifest_but_not_actual_inputs(self):
        row=dict(video_sha256='v',mask_sha256='g',pts=[0,.1],work_hw=[4,5],native_hw=[40,50])
        old=dict(row,adaptation_sha256='a',provider_config_sha256='p',scale=SCALE,h=.5,origin=[0,0,0],source_manifest_sha256='old')
        self.assertTrue(data.observation_compatible(row,old,'a','p'))
        for key,value in [('video_sha256','other'),('pts',[0,.2]),('work_hw',[5,4])]:
            wrong=dict(old);wrong[key]=value;self.assertFalse(data.observation_compatible(row,wrong,'a','p'))

    def test_work_rgb_reader_is_cpu_bounded_and_rejects_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=dict(video_id='x_real',cache_root=tmp,cache_kind='work_rgb',role='real',actual_frames=2,
                   video_sha256='v',mask_sha256='m',pts=[0.,.1],work_hw=[4,5],native_hw=[40,50])
            folder=Path(tmp)/'x_real';folder.mkdir()
            np.save(folder/'rgb.npy',np.zeros((2,4,5,3),np.uint8));np.save(folder/'gt.npy',np.zeros((2,4,5),bool))
            meta={k:r[k] for k in ['video_sha256','mask_sha256','pts','work_hw','native_hw']};meta['native_GT_pixels']=[0,0]
            data.save(folder/'identity.json',meta)
            reader=data.WorkVideo(r);self.assertEqual(reader.get(1)['rgb'].device.type,'cpu')
            bad=dict(r,video_sha256='wrong')
            with self.assertRaises(AssertionError):data.WorkVideo(bad)

    def test_restore_optimizer_rng_and_next_update_exactly(self):
        from research_tools.xyz_pilot.readout_adaptation import tree_hash
        seed();fs=tiny_frames();m=FullVideoDirect(activation_checkpoint=False)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
        loss=video_loss(m(fs,'FULL_VIDEO'),fs)[0];loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt';identity={'trial':'test'}
            cp=run.save_checkpoint(path,m,opt,identity,'FULL_VIDEO',1,1,[dict(video_id='one',supervised_frames=2)])
            expected_opt=tree_hash(opt.state_dict());expected_rng=tree_hash(cp['random_state'])
            loss=video_loss(m(fs,'FULL_VIDEO'),fs)[0];loss.backward();opt.step();expected=state_hash(m.state_dict())
            restored=FullVideoDirect(activation_checkpoint=False);opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=.01)
            got=run.restore_checkpoint(path,restored,opt2,identity,'FULL_VIDEO')
            self.assertEqual(got['updates'],1);self.assertEqual(tree_hash(opt2.state_dict()),expected_opt)
            self.assertEqual(tree_hash(run.rng_state()),expected_rng)
            opt2.zero_grad(set_to_none=True);video_loss(restored(fs,'FULL_VIDEO'),fs)[0].backward();opt2.step()
            self.assertEqual(state_hash(restored.state_dict()),expected)
            with self.assertRaises(AssertionError):run.restore_checkpoint(path,restored,opt2,{'trial':'wrong'},'FULL_VIDEO')

    def test_completed_video_cursor_does_not_repeat_exposure(self):
        orders=[['a','b'],['b','a'],['a','b']]
        run.validate_cursor(dict(epoch=1,position=2,updates=2,records=[{'video_id':'a'},{'video_id':'b'}]),orders)
        run.validate_cursor(dict(epoch=2,position=0,updates=2,records=[{'video_id':'a'},{'video_id':'b'}]),orders)
        with self.assertRaises(AssertionError):run.validate_cursor(dict(epoch=2,position=1,updates=2,records=[{'video_id':'a'},{'video_id':'b'}]),orders)

    def test_macro_equal_source_weight_not_frame_or_pixel_pool(self):
        def rec(source,i,ap):
            return dict(video_id=source+'_fake',pair_id=source,source_group_id=source,role='fake',
                        frame_index=i,GT_pixels=1,pixel_AP=ap,Dice=ap,IoU=ap,GT_recall=ap,
                        real_FPR=None,fake_background_FPR=0.,prediction_area=.1,constant_AP=.01)
        ev=run.aggregate_records([rec('a',0,1.),rec('a',1,1.),rec('b',0,0.)])
        self.assertEqual(ev['macro']['pixel_AP'],.5)
        self.assertEqual(ev['scored_frames'],3)

    def test_exact_OOM_only_fallback_is_uniform_and_not_semantic_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);data.save(out/'resource_current.json',dict(video='max',M=100,T=3))
            with patch.object(run,'extreme_once',side_effect=[torch.cuda.OutOfMemoryError('test'),[]]) as call,patch.object(run,'query_equivalence') as parity,patch.object(torch.cuda,'empty_cache'),patch.object(run,'memory',return_value={}):
                q=run.resource_gate(out,{},dict(identity=1),lambda *a,**k:None)
                self.assertEqual(q,1024);self.assertEqual(call.call_count,2);parity.assert_called_once()
            self.assertEqual(data.read(out/'compute.json')['query_chunk'],1024)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run,'extreme_once',side_effect=ValueError('bad input')) as call,patch.object(run,'query_equivalence') as parity,patch.object(torch.cuda,'empty_cache'):
                with self.assertRaises(ValueError):run.resource_gate(Path(tmp),{},dict(identity=1),lambda *a,**k:None)
                self.assertEqual(call.call_count,1);parity.assert_not_called()

    def test_cached_geometry_stage_writes_cost_without_inference(self):
        import types
        from unittest.mock import Mock
        provider=dict(geometry_long_edge=768,geometry_num_tokens='round(input_h*input_w/14**2)',
                      moge_weights='/unused/local.pt',moge_code='/unused/code',moge_revision='revision',
                      utils3d_code='/unused/utils')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);provider_path=out/'provider.json';data.save(provider_path,provider)
            rows=[]
            for name,n in [('first_real',2),('second_fake',3)]:
                row=dict(video_id=name,actual_frames=n,video_sha256=name,pts=[i*.1 for i in range(n)],
                         native_hw=[4,6],work_hw=[2,3])
                rows.append(row);folder=out/'geometry'/name;folder.mkdir(parents=True)
                ident={k:row[k] for k in ['video_sha256','pts','native_hw','work_hw']}
                ident.update(provider_sha256='provider',moge_weights_sha256='weights',scale=SCALE,h=.5)
                data.save(folder/'identity.json',ident)
                for i in range(n):(folder/f'{i:05d}.pt').write_bytes(b'existing-cache-not-loaded')
            fake_moge=types.ModuleType('moge.model.v2');fake_moge.MoGeModel=Mock()
            fake_moge.MoGeModel.from_pretrained.side_effect=AssertionError('Must not instantiate MoGe')
            with patch.object(data,'PROVIDER',provider_path),patch.object(data,'digest',return_value='weights'), \
                 patch.object(data.subprocess,'check_output',return_value='revision\n'), \
                 patch.object(data,'cached_geometry_index',return_value={}), \
                 patch.object(data,'WorkVideo',side_effect=AssertionError('Must not read media')) as work, \
                 patch.object(data.av,'open',side_effect=AssertionError('Must not decode')) as decode, \
                 patch.object(data,'prepare_frame',side_effect=AssertionError('Must not route')) as routing, \
                 patch.object(data.torch,'load',side_effect=AssertionError('Must not load tensors')) as tensors, \
                 patch.object(data.torch.cuda,'empty_cache') as gpu_cleanup, \
                 patch.dict('sys.modules',{'moge.model.v2':fake_moge}):
                data.prepare_geometry(out,rows,dict(provider_sha256='provider'),lambda *a,**k:None)
            files=list(out.glob('geometry_cost_*.json'));self.assertEqual(len(files),1)
            self.assertRegex(files[0].name,r'^geometry_cost_[0-9]+\.json$')
            cost=data.read(files[0]);self.assertEqual(cost['cached_route_frames'],5)
            self.assertEqual(cost['new_MoGe'],0);self.assertEqual(cost['reused_geometry'],0)
            self.assertEqual(cost['new_DFormer'],0);self.assertGreaterEqual(cost['seconds'],0)
            work.assert_not_called();decode.assert_not_called();routing.assert_not_called();tensors.assert_not_called()
            fake_moge.MoGeModel.from_pretrained.assert_not_called();gpu_cleanup.assert_called_once()

    def test_same_output_refuses_concurrent_invocation_without_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=run.exclusive_lock(Path(tmp))
            try:
                with self.assertRaisesRegex(RuntimeError,'active invocation'):run.exclusive_lock(Path(tmp))
            finally:first.close()
            second=run.exclusive_lock(Path(tmp));second.close()

    def test_disk_guard_precedes_any_frontend_call(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);data.save(out/'space_estimate.json',dict(required_free_bytes_with_margin=100))
            with patch.object(data.shutil,'disk_usage',return_value=SimpleNamespace(free=0)),patch.object(data,'prepare_geometry') as geometry:
                with self.assertRaisesRegex(AssertionError,'DISK_BUDGET'):data.prepare_all(out,{'rows':[]},{},lambda *a,**k:None)
                geometry.assert_not_called()

    def test_automatic_final_report_and_fixed_preview_identities(self):
        p=data.read(data.SOURCE/'protocol.json');byid={r['video_id']:r for r in p['rows']}
        original=data.read(data.SOURCE/'epoch01_dev.json')
        records=[dict(f,source_group_id=byid[f['video_id']]['source_group_id']) for f in original['frames']]
        ev=run.aggregate_records(records);ev['groups']=run.groups(ev,data.categories())
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);data.save(out/'protocol.json',p);data.save(out/'frozen_reference.json',ev)
            for mode in ['FRAME_ONLY','FULL_VIDEO']:
                for epoch in [1,2,3]:data.save(out/mode/f'epoch{epoch:02d}_dev.json',ev)
                data.save(out/mode/'complete.json',dict(best_epoch=1,status='COMPLETE'))
            run.report(out)
            self.assertEqual(data.read(out/'summary.json')['status'],'COMPLETE')
            self.assertTrue((out/'primary_epoch3.csv').exists())
            self.assertTrue((out/'best_development_supplement.csv').exists())
            self.assertEqual((out/'preview/index.html').read_text().count('<h3>'),54)
            import csv
            rows=list(csv.DictReader((out/'FULL_VIDEO_minus_FRAME_ONLY_sources.csv').read_text().splitlines()))
            self.assertEqual(len(rows),19)
            self.assertTrue(all(float(r['pixel_AP'])==0 for r in rows))

    def test_fixed_query1024_output_and_gradient_equivalence(self):
        run.query_equivalence()

    def test_finalized_categories_and_reference_group_sizes(self):
        cats=data.categories();self.assertEqual(len(cats),19)
        self.assertEqual(sum(c['content_priority']=='object_edit_pending_human' for c in cats.values()),16)
        ev=data.read(data.SOURCE/'epoch01_dev.json');p=data.read(data.SOURCE/'protocol.json')
        byid={r['video_id']:r for r in p['rows']}
        records=[dict(f,source_group_id=byid[f['video_id']]['source_group_id']) for f in ev['frames']]
        actual=run.aggregate_records(records)
        for k in run.METRICS:self.assertAlmostEqual(actual['macro'][k],ev['macro'][k],places=12)
        grouped=run.groups(actual,cats);self.assertEqual(grouped['原4开发来源']['frames'],824)
        self.assertEqual(grouped['新增15开发来源']['frames'],4328)


if __name__=='__main__':unittest.main()
