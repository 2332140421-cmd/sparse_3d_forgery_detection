"""Only CPU contracts essential to the fixed-weight context diagnosis."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from research_tools.xyz_pilot import fullvideo_context_diagnosis as d
from research_tools.xyz_pilot.fullvideo_direct_model import FullVideoDirect,prepare_frame,SCALE


def frame():
    xyz=torch.zeros(4,5,3);xyz[...,0]=torch.arange(5)[None]*SCALE*.3;xyz[...,2]=SCALE
    valid=torch.ones(4,5,dtype=torch.bool);valid[0,0]=False
    return prepare_frame(dict(xyz_provider=xyz,valid=valid,work_hw=[4,5],dino_padded_hw=[14,14],patch=torch.randn(1024,1,1),
        detail=torch.randn(32,2,2),z0=torch.randn(4,5),gt_work=torch.ones(4,5,dtype=torch.bool),pts=0.,supervised=True))


class ContextTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(19)

    @torch.no_grad()
    def test_exact_adapter_covers_projection_bias_and_keeps_input(self):
        f=frame();m=FullVideoDirect().eval().requires_grad_(False)
        m.pixel_context.weight.zero_();m.pixel_context.bias.fill_(2)
        torch.nn.init.normal_(m.correction[-1].weight,std=.1)
        r=torch.zeros(len(f['keys']),128);before=r.clone();calls=Counter()
        normal,zero,detail,ctx,check=d.DecodeAdapter(m,calls).pair(f,r,parity=True,repeat=True)
        self.assertEqual(check['adapter_normal_max_abs'],0)
        self.assertEqual(check['repeat_normal_max_abs'],0)
        flat=ctx[0].permute(1,2,0).reshape(-1,32)
        self.assertTrue(torch.all(flat[f['points']]==2));self.assertTrue(torch.all(flat[0]==0))
        expected=f['z0']+m.correction(torch.cat((detail,torch.zeros_like(ctx)),dim=1))[0,0]
        torch.testing.assert_close(zero,expected,atol=0,rtol=0)
        self.assertGreater(float((normal-zero).abs().max()),0)
        self.assertTrue(torch.equal(r,before));self.assertEqual(len(m.correction._forward_pre_hooks),0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_empty_region_null_and_signed_flip_counts(self):
        row=dict(video_id='x_fake',pair_id='x',source_group_id='s',role='fake',pts=[0])
        base=torch.tensor([[-1.,1.],[-1.,1.]])
        normal=-base;zero=base.clone();gt=np.zeros((2,2),bool)
        stats=d.region_statistics(row,0,gt,normal,zero,base)
        empty=next(r for r in stats if r['region']=='fake_GT' and r['difference']=='total')
        self.assertEqual(empty['pixels'],0);self.assertIsNone(empty['mean_abs'])
        self.assertIsNone(empty['positive_to_negative'])
        active=next(r for r in stats if r['region']=='fake_empty' and r['difference']=='total')
        self.assertEqual(active['positive_to_negative'],2);self.assertEqual(active['negative_to_positive'],2)
        self.assertEqual(active['frame_abs_p95'],2)

    def test_same_relation_encode_path_matches_original_forward(self):
        fs=[frame(),frame()];fs[1]['pts']=.1
        model=FullVideoDirect().eval().requires_grad_(False)
        with torch.no_grad():
            original,info=model(fs,'FULL_VIDEO',diagnostics=True)
            c,r,counts=d.encode_relations(model,fs,'FULL_VIDEO',Counter(),lambda *a,**k:None)
            torch.testing.assert_close(c,info['tokens'],rtol=0,atol=0)
            torch.testing.assert_close(r,info['relations'],rtol=0,atol=0)
            for f,x,z in zip(fs,r.split(counts),original):torch.testing.assert_close(model.decode(f,x),z,rtol=0,atol=0)

    def test_native_metrics_match_original_score_formula(self):
        f=frame();row=dict(video_id='x_real',pair_id='x',source_group_id='s',role='real',native_hw=[8,10],actual_frames=1,pts=[0.],preview_targets=[])
        f['native_GT_pixels']=0;z=f['z0']
        expected=d.original.score_video(row,[f],[z])[0]
        actual=d.native_record(row,0,np.zeros((8,10),bool),z)
        self.assertEqual(actual,expected)

    def test_positive_native_AP_matches_original_even_with_empty_work_mask(self):
        import io
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mask.zip';gt=np.zeros((8,10),bool);gt[2:5,3:6]=True
            image=io.BytesIO();Image.fromarray((gt*255).astype(np.uint8)).save(image,format='PNG')
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('00000.png',image.getvalue())
            row=dict(video_id='x_fake',pair_id='x',source_group_id='s',role='fake',native_hw=[8,10],actual_frames=1,pts=[0.],preview_targets=[],mask_zip=str(path),index_base=0)
            f=frame();f['gt_work'].zero_();f['native_GT_pixels']=int(gt.sum())
            expected=d.original.score_video(row,[f],[f['z0']])[0]
            actual=d.native_record(row,0,gt,f['z0'])
            self.assertEqual(actual,expected);self.assertIsNotNone(actual['pixel_AP'])

    def test_observer_cpu_reads_z0_without_model_or_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);names=['4dpnv4_fake','4dpnv4_real','4qw154_fake','4qw154_real'];rows=[]
            for name in names:
                # Real-only numerical fixture; fake semantics exercised elsewhere.
                row=dict(video_id=name,pair_id=name,source_group_id=name,role='real',actual_frames=1,pts=[0.],work_hw=[2,2],native_hw=[4,4])
                rows.append(row);path=root/(name+'.pt');torch.save(dict(frame_index=0,pts=0.,work_hw=[2,2],z0=torch.ones(2,2)),path)
                d.save(root/'visual'/name/'frames.json',[str(path)])
            d.save(root/'monitor.json',names)
            with patch.object(d.data,'WorkVideo') as work,patch.object(d,'load_model',side_effect=AssertionError('No checkpoint')):
                work.return_value.get.return_value={'native_GT_pixels':0}
                initial=d.original.aggregate_records([d.native_record(r,0,np.zeros((4,4),bool),torch.ones(2,2)) for r in rows])
                for mode in d.MODES:
                    for e in [1,2,3]:d.save(root/mode/f'epoch{e:02d}_monitor.json',initial)
                ev,comparison=d.monitor_cpu(root,{'rows':rows},root/'new')
                self.assertEqual(ev['scored_frames'],4);self.assertEqual(len(comparison),7)

    def test_real_finalization_empty_regions_and_report_json_paths(self):
        row=dict(video_id='x_real',pair_id='x',source_group_id='s',role='real',pts=[0.],native_hw=[4,4])
        z=torch.zeros(4,4);rec=d.native_record(row,0,np.zeros((4,4),bool),z)
        result=dict(video_id='x_real',metrics={k:d.original.aggregate_records([rec]) for k in ['NORMAL','CTX_ZERO']},
            region_frames=d.region_statistics(row,0,np.zeros((4,4),bool),z,z,z),input_frames=[dict(video_id='x_real',pair_id='x',source_group_id='s',valid_pixels=0,frame_pixels=16,coverage=0,context_mean_abs=None,context_RMS=None,detail_mean_abs=None,detail_RMS=None)],checks=[dict(frame_index=0,adapter_normal_max_abs=0,repeat_normal_max_abs=0)],calls={})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'nested';out.mkdir()
            summary=d.finalize(out,[dict(mode=m,**result) for m in d.MODES],Path(tmp),'smoke',Counter(),.1)
            parsed=d.data.read(out/'summary.json')
            self.assertEqual(parsed['status'],'SMOKE_PASS_FULL_NOT_STARTED')
            self.assertTrue((out/'report.md').is_file())
            missing=[r for r in parsed['region_macro'] if r['region']=='fake_GT']
            self.assertTrue(all(r['mean_abs'] is None for r in missing))

    def test_full_development_finalize_replay_and_all_groups_cpu_only(self):
        # Exercise the OTHER finalization branch (full5152/groups/replay), no model.
        cats={}
        pairs=sorted(d.original.HIST)+['an:fixture'+str(i) for i in range(15)]
        frames=[]
        for j,pair in enumerate(pairs):
            cats[pair]=dict(official_action='removal',camera_group='unknown',content_priority='object_edit_pending_human' if j<16 else 'appearance_background_reference')
            for role in ['fake','real']:
                vid=pair+'_'+role;n=136 if len(frames)<22*136 else 135
                for i in range(n):
                    frames.append(dict(video_id=vid,pair_id=pair,source_group_id=pair,role=role,frame_index=i,pts=i*.1,
                        native_pixels=100,GT_pixels=10 if role=='fake' else 0,pixel_AP=.5 if role=='fake' else None,
                        constant_AP=.1 if role=='fake' else None,Dice=.4 if role=='fake' else None,IoU=.25 if role=='fake' else None,
                        GT_recall=.5 if role=='fake' else None,real_FPR=.01 if role=='real' else None,fake_background_FPR=.02 if role=='fake' else None,prediction_area=.1 if role=='fake' else None))
        self.assertEqual(len(frames),5152)
        ev=d.original.aggregate_records(frames);ev['groups']=d.original.groups(ev,cats)
        results=[dict(mode=m,video_id='fixture_all',metrics={label:ev for label in ['NORMAL','CTX_ZERO']},region_frames=[],input_frames=[],checks=[dict(frame_index=0,adapter_normal_max_abs=0,repeat_normal_max_abs=0)],calls={}) for m in d.MODES]
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';out=Path(tmp)/'new';out.mkdir()
            d.save(source/'categories.json',cats)
            for m in d.MODES:d.save(source/m/'epoch03_dev.json',ev)
            # No per-frame diagnostics in this fixture; generate a tiny null row to
            # cover the summary formatter without fabricating large pixel arrays.
            row=dict(video_id='v',pair_id=pairs[0],source_group_id=pairs[0],role='real',pts=[0.])
            stats=d.region_statistics(row,0,np.zeros((1,1),bool),torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1))
            for result in results:result['region_frames']=stats
            summary=d.finalize(out,results,source,'development',Counter(),.1)
            self.assertEqual(summary['status'],'COMPLETE');self.assertTrue((out/'report.md').exists())
            primary=d.data.read(out/'summary.json')['primary']
            self.assertEqual(len([r for r in primary if r['group']=='全部19开发来源']),4)
            self.assertTrue((out/'replay_sources.csv').is_file())


if __name__=='__main__':unittest.main()
