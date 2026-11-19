import unittest
import torch
from research_tools.xyz_pilot.insource_probe import boundary_rows,partition_history,epoch_orders
from research_tools.xyz_pilot.insource_model import XYZLocalization,ImageOnly,COMMON,losses
from test_xyz_joint_relation import frame


class InsourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def rows(self):
        return boundary_rows([dict(pair_id='an:40y1x9',video_id='40y1x9_'+role,role=role,split='train',actual_frames=20,pts=[i*.1 for i in range(20)]) for role in ['real','fake']])

    def test_shared_time_boundary_and_history_partition(self):
        rows=self.rows();self.assertEqual(rows[0]['boundary_pts'],rows[1]['boundary_pts'])
        for row in rows:
            cut=row['train_range'][1];self.assertEqual(cut,row['evaluation_range'][0])
            self.assertEqual(partition_history(row,cut,'evaluation'),[None]*7+[cut])
            for i in row['evaluation_targets']:
                self.assertTrue(all(j is None or cut<=j<=i for j in partition_history(row,i,'evaluation')))
            for i in row['train_targets']:
                self.assertTrue(all(j is None or 0<=j<=i<cut for j in partition_history(row,i,'train')))
        self.assertFalse(set([6,7,8,9])&set(rows[1]['train_targets']))
        self.assertTrue(all(i not in [6,7,8,9] for i in rows[1]['monitor_targets']))

    def test_full_epochs_without_replacement_and_reproducible(self):
        rows=self.rows();a=epoch_orders(rows,10);b=epoch_orders(rows,10)
        self.assertEqual(a,b)
        expected={(r['video_id'],i) for r in rows for i in r['train_targets']}
        for epoch in a:
            actual=[(x['video_id'],x['target']) for x in epoch]
            self.assertEqual(set(actual),expected);self.assertEqual(len(actual),len(expected))

    def test_image_isolation_common_initialization_and_gradients(self):
        torch.manual_seed(20260909);ref=XYZLocalization(.5);image=ImageOnly(ref)
        for name in COMMON:
            for x,y in zip(getattr(ref,name).parameters(),getattr(image,name).parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
        f=frame();f['gt_work']=torch.zeros(8,8,dtype=torch.bool);f['gt_work'][2:4,2:4]=True
        minimal={k:v for k,v in f.items() if k not in ['xyz','valid']}
        out=image([minimal]);changed=dict(minimal,xyz=torch.randn(8,8,3)*999,valid=torch.zeros(8,8,dtype=torch.bool))
        torch.testing.assert_close(out['pixel'],image([changed])['pixel'],atol=0,rtol=0)
        self.assertFalse(any(k.startswith(('geometry','point','layers','pixel_context','evidence','frame')) for k in image.state_dict()))
        parts=losses(out,f);self.assertEqual(parts.shape,(2,));parts.sum().backward()
        self.assertTrue(torch.isfinite(image.pixel[-1].weight.grad).all())
        self.assertFalse(hasattr(ref,'evidence'));self.assertFalse(hasattr(ref,'empty'))
        self.assertGreater(sum(p.numel() for p in ref.parameters()),sum(p.numel() for p in image.parameters()))
        for temporal in [False,True]:
            output=ref([f],temporal);losses(output,f).sum().backward()
            self.assertTrue(torch.isfinite(ref.geometry[0].weight.grad).all())
