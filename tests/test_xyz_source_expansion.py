import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io
import zipfile
import av
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from research_tools.xyz_pilot.source_expansion import explicit_exclusions,full_train_row,training_order,coverage
from research_tools.xyz_pilot.data import Cache,extract_video,remap_geometry
from research_tools.xyz_pilot.insource_probe import PartitionCache,partition_history,epoch_orders


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_explicit_mask_negative_wins_batch_permission(self):
        r={'batch':'镜头基本固定，允许试跑','records':[{'pair':'an:a','mask':'不对应'},{'pair':'an:b','mask':'对应'},{'pair_id':'an:c','human_semantic_alignment':'does_not_correspond'}]}
        self.assertEqual(explicit_exclusions(r),{'an:a','an:c'})
        self.assertEqual(explicit_exclusions({'raw':[{'pair':'an:d','mask':''}]}),set())

    def test_full_source_legal_targets_and_causal_histories(self):
        source=dict(pair_id='an:40y1x9',actual_frames=20,video_id='dog_fake',role='fake',pts=[i*.1 for i in range(20)])
        r=full_train_row(source)
        self.assertEqual(r['train_range'],[0,20]);self.assertEqual(r['evaluation_targets'],[])
        self.assertTrue({6,7,8,9}.isdisjoint(r['train_targets']));self.assertIn(19,r['train_targets'])
        self.assertEqual(partition_history(r,0,'train'),[None]*7+[0])
        self.assertTrue(all(x is None or x<=19 for x in partition_history(r,19,'train')))
        self.assertNotIn('train_range',source)

    def test_matched_budget_distinct_sequences_and_exposures(self):
        old=[dict(video_id='r',pair_id='p',role='real',train_targets=list(range(5)))]
        extra=[dict(video_id='f',pair_id='q',role='fake',train_targets=list(range(3)))]
        a=training_order(old,80,20260909);b=training_order(old+extra,80,20260909)
        self.assertEqual(len(a),80);self.assertEqual(len(b),80);self.assertNotEqual(a,b)
        self.assertEqual(a[:5],epoch_orders(old,1)[0]);self.assertEqual(a,training_order(old,80,20260909))
        ca=coverage(a,old);cb=coverage(b,old+extra)
        self.assertEqual(ca['per_video'][0]['min_exposure'],16)
        self.assertTrue(all(v['min_exposure']==10==v['max_exposure'] for v in cb['per_video']))
        self.assertEqual(cb['real'],50);self.assertEqual(cb['fake'],30)

    def test_shared_extractor_missing_only_original_rgb_gt_geometry_and_scale(self):
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.inputs=[]
            def forward(self,x,d):
                self.inputs.append((x.clone(),d.clone()))
                return [x[:,:1]+self.anchor for _ in range(4)],None
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'v.mp4'
            with av.open(str(video),'w') as out:
                stream=out.add_stream('libx264',rate=10);stream.width=24;stream.height=16;stream.pix_fmt='yuv420p'
                for i in range(2):
                    image=np.full((16,24,3),60+i*60,np.uint8);frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            with av.open(str(video)) as inp:frames=list(inp.decode(video=0))
            pts=[float(f.pts*f.time_base) for f in frames]
            gt=np.zeros((16,24),np.uint8);gt[2:8,4:10]=255;buf=io.BytesIO();Image.fromarray(gt).save(buf,format='PNG')
            mask=root/'m.zip'
            with zipfile.ZipFile(mask,'w') as z:
                for i in range(2):z.writestr(f'{i:04d}.png',buf.getvalue())
            geometry=root/'geometry';geometry.mkdir();xyz=np.full((2,16,24,3),4,np.float32);valid=np.ones((2,16,24),bool);valid[:,0]=False
            np.save(geometry/'xyz.npy',xyz);np.save(geometry/'valid.npy',valid)
            row=dict(pair_id='an:new',role='fake',video_id='new_fake',video_path=str(video),mask_zip=str(mask),actual_frames=2,index_base=0,pts=pts,native_hw=[16,24],work_hw=[8,12],padded_hw=[32,32])
            folder=root/'cache'/'new_fake';folder.mkdir(parents=True);(folder/'00000.pt').write_bytes(b'preserve-existing')
            encoder=Encoder().eval().requires_grad_(False)
            with patch('research_tools.xyz_pilot.data.compatible_geometry',return_value=geometry):
                rec,_=extract_video(row,{},encoder,folder,missing=[1])
            self.assertEqual((folder/'00000.pt').read_bytes(),b'preserve-existing')
            self.assertEqual(rec['new_visual_calls'],1);self.assertEqual(rec['new_geometry_calls'],0);self.assertEqual(rec['reused_geometry_frames'],1)
            e=torch.load(folder/'00001.pt',weights_only=True)
            rgb=frames[1].to_ndarray(format='rgb24');image=cv2.resize(rgb,(12,8),interpolation=cv2.INTER_AREA)
            x=torch.from_numpy(image).permute(2,0,1)[None].float()/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None];x=F.pad(x,(0,20,0,24))
            torch.testing.assert_close(encoder.inputs[0][0],x,atol=0,rtol=0);self.assertEqual(int(encoder.inputs[0][1].count_nonzero()),0)
            self.assertEqual(e['frame_index'],1);self.assertEqual(e['pts'],pts[1]);self.assertEqual(e['native_GT_pixels'],int((gt>0).sum()))
            np.testing.assert_array_equal(e['gt_work'],cv2.resize((gt>0).astype(np.uint8),(12,8),interpolation=cv2.INTER_NEAREST).astype(bool))
            q,v=remap_geometry(xyz[1],valid[1],(8,12));np.testing.assert_array_equal(e['xyz_provider'],q)
            cache=Cache({'scale':2},root/'cache');a=cache.get(row,1,device='cpu');b=cache.get(row,1,device='cpu')
            torch.testing.assert_close(a['xyz'],e['xyz_provider']/2,atol=0,rtol=0);torch.testing.assert_close(a['xyz'],b['xyz'],atol=0,rtol=0)
            pc=PartitionCache({'scale':2},root/'cache');row['train_range']=[0,2]
            image_only=pc.get(row,1,'train',image_only=True,device='cpu');self.assertNotIn('xyz',image_only);self.assertNotIn('valid',image_only)
