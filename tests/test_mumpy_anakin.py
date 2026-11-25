"""CPU-only research contracts for the one official MumPy transfer baseline."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from research_tools.mumpy_anakin import data
from research_tools.mumpy_anakin import model as adapter
from research_tools.mumpy_anakin import experiment as run


class MumpyTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_original_source_separation_targets_orders_and_monitor(self):
        p=data.read(data.SOURCE/'protocol.json');ts=data.targets(p['rows'],'train')
        self.assertEqual(len(ts),22878);self.assertEqual(len(data.targets(p['rows'],'dev')),5152)
        orders=data.epoch_orders(p['rows']);self.assertEqual(orders,data.epoch_orders(p['rows']))
        self.assertTrue(all(len(o)==len(set(o))==22878 and set(o)==set(ts) for o in orders))
        self.assertEqual(len(data.monitor_targets(p['rows'])),32)
        self.assertTrue(all(not(v=='40y1x9_fake' and 6<=i<10) for v,i in ts))
        self.assertEqual(10*((22878+11)//12),19070)
        self.assertEqual(22878%12,6)

    def test_initializer_whitelist_never_allows_missing_backbone_weights(self):
        self.assertTrue(adapter.allowed_new_key('globalembedding.weight'))
        self.assertTrue(adapter.allowed_new_key('layers.layers.0.blocks.1.block1.attn_mask'))
        self.assertTrue(adapter.allowed_new_key('layers.layers.0.blocks.0.block1.cva.crossattn.proj_out.weight'))
        for key in ['tokenize.project1.weight','globalblocks.blocks.0.attn.qkv.weight','layers.layers.2.blocks.0.block3.attn.qkv.weight']:
            self.assertFalse(adapter.allowed_new_key(key))

    def test_boundary_mapping_center_and_unlabeled_context(self):
        self.assertEqual(data.clip_indices(0,4),[0,0,1]);self.assertEqual(data.clip_indices(3,4),[2,3,3])
        self.assertEqual(data.clip_indices(10,150),[9,10,11])
        with self.assertRaises(AssertionError):data.clip_indices(4,4)

    def test_shared_official_augmentation_and_categorical_mask(self):
        row=dict(video_id='fixture_fake',actual_frames=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'rgb224'/row['video_id'];folder.mkdir(parents=True)
            a=np.zeros((224,224),np.uint8);a[30:80,45:100]=255
            rgb=np.stack([np.repeat(a[...,None],3,axis=-1)]*3);gt=np.stack([a>0]*3)
            np.save(folder/'rgb.npy',rgb);np.save(folder/'gt.npy',gt);run.save(folder/'complete.json',dict(status='COMPLETE'))
            clips=data.Clips(root,[row],training=True)
            for _ in range(12):
                x,y=clips.get(('fixture_fake',1));self.assertTrue(torch.equal(x[0],x[1]) and torch.equal(x[1],x[2]))
                restored=(x[1]*data.STD+data.MEAN)[0]>.5
                self.assertTrue(torch.equal(restored,y[0].bool()));self.assertTrue(set(y.unique().tolist())<={0.,1.})
            clips.rows['40y1x9_fake']=dict(row,video_id='40y1x9_fake')
            with self.assertRaises(AssertionError):clips.get(('40y1x9_fake',6))

    def test_released_loss_formula_and_real_empty_mask(self):
        z=torch.tensor([[-2.,0.,2.,1.],[0.,1.,-1.,2.]])
        y=torch.tensor([[0.,1.,1.,0.],[0.,0.,0.,0.]])
        with patch.object(torch.Tensor,'cuda',lambda x,*a,**kw:x):loss,parts=adapter.losses(z,y)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')
        focal=((1-(-bce).exp())**2*bce).mean()
        prob=z.sigmoid();iou=1-((prob*y).sum(1)/((prob+y-prob*y).sum(1)+1e-6)).mean()
        torch.testing.assert_close(loss,focal+iou)
        self.assertAlmostEqual(parts['weighted_focal'],float(focal));self.assertTrue(torch.isfinite(loss))
        with patch.object(torch.Tensor,'cuda',lambda x,*a,**kw:x):
            extreme=torch.full((1,4),-1000.,requires_grad=True)
            empty,_=adapter.losses(extreme,torch.zeros_like(extreme));empty.backward()
            self.assertTrue(torch.isfinite(empty));self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_save_restore_all_optimizers_schedulers_rng_and_json(self):
        m=torch.nn.Module();m.encoder=torch.nn.ModuleDict(dict(body=torch.nn.Linear(2,2),cva=torch.nn.Linear(2,2)));m.decoder=torch.nn.Linear(2,1)
        opts,sche=adapter.optimizers(m,19070)
        for o in opts:o.zero_grad()
        sum(p.square().sum() for p in m.parameters()).backward()
        for o in opts:o.step()
        for s in sche:s.step()
        before={k:v.clone() for k,v in m.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch.cuda,'is_available',return_value=False):
            before_rng=run.rng_state();p=Path(tmp)/'nested/test.pt';run.checkpoint(p,m,opts,sche,{'test':1},1,1,6)
            with torch.no_grad():next(m.parameters()).add_(1)
            rec=run.restore(p,m,opts,sche,{'test':1});self.assertEqual(rec['updates'],1)
            after_rng=run.rng_state();self.assertEqual(before_rng['python'],after_rng['python'])
            self.assertTrue(np.array_equal(before_rng['numpy'][1],after_rng['numpy'][1]));self.assertTrue(torch.equal(before_rng['torch'],after_rng['torch']))
            for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            self.assertEqual([s.last_epoch for s in sche],[1,1,1]);self.assertTrue(all(o.state for o in opts))
            with self.assertRaises(AssertionError):run.restore(p,m,opts,sche,{'test':2})

    def test_micro_batch_weighting_matches_full_batch_and_keeps_tail(self):
        torch.manual_seed(8)
        for n in [12,6]:
            a=torch.nn.Linear(3,1);b=copy.deepcopy(a);x=torch.randn(n,3);y=torch.randn(n,1)
            torch.nn.functional.mse_loss(a(x),y).backward()
            for i in range(0,n,2):
                (torch.nn.functional.mse_loss(b(x[i:i+2]),y[i:i+2])*len(x[i:i+2])/n).backward()
            for u,v in zip(a.parameters(),b.parameters()):torch.testing.assert_close(u.grad,v.grad)

    def test_report_preview_json_csv_finalization_and_no_premature_complete(self):
        macro={k:.2 for k in ['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR']}
        ev=dict(macro=macro,groups={'新增15开发来源':dict(macro)})
        row=dict(video_id='v_fake',split='dev',native_hw=[8,10],preview_targets=[0],pts=[0.])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for e in range(1,11):run.save(out/f'epoch{e:02d}_dev.json',ev)
            run.preview_frame(out/'preview',row,0,np.zeros((8,10),bool),np.zeros((8,10)),np.zeros((224,224,3),np.uint8),'last')
            run.report(out,[row]);self.assertEqual(data.read(out/'summary.json')['status'],'PARTIAL')
            self.assertTrue((out/'preview/index.html').is_file());self.assertTrue((out/'group_metrics.csv').is_file())
            self.assertTrue((out/'report.md').is_file());self.assertEqual(data.read(out/'selection.json')['epoch'],1)
            for e in range(1,11):(out/f'epoch{e:02d}.pt').write_bytes(b'fixture-not-a-model')
            run.preview_frame(out/'preview',row,0,np.zeros((8,10),bool),np.zeros((8,10)),np.zeros((224,224,3),np.uint8),'selected')
            run.report(out,[row]);self.assertEqual(data.read(out/'summary.json')['status'],'COMPLETE')
            self.assertEqual(data.read(out/'summary.json')['updates'],19070)

    def test_report_only_entry_never_initializes_or_loads_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'formal';out.mkdir();run.save(out/'protocol.json',dict(rows=[]))
            with patch('sys.argv',['mumpy','--output',str(root),'--report-only']), \
                 patch.object(run,'MumPy',side_effect=AssertionError('No network')), \
                 patch.object(run,'identity',side_effect=AssertionError('No checkpoint/weight access')):
                run.main()
            self.assertTrue((out/'report.md').exists())


if __name__=='__main__':unittest.main()
