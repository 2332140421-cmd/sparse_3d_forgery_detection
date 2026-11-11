import copy
import unittest
import torch
from research_tools.rgbd_slice.slice_relation_readout import SliceRelationReadout,candidate_bank,project_gt,windows


def frame(ids,pts):return {'features':torch.randn(len(ids),512),'ids':torch.tensor(ids),'pts':pts}


class SliceRelationReadoutTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(20260909)
    def test_self_exclusion_history_same_id_and_no_future(self):
        fs=[frame([-1,2,9],.1),frame([2,5],.17),frame([2,8],.31)]
        bank,ids,dt,blocked=candidate_bank(fs,1,True)
        self.assertEqual(ids.tolist(),[-1,2,9,2,5]);self.assertEqual(blocked.tolist(),[[False,False,False,True,False],[False,False,False,False,True]])
        self.assertTrue((dt[:3]<0).all());self.assertTrue((dt[3:]==0).all())
        a=SliceRelationReadout().eval();z,_=a(fs,1,True);fs[2]['features'].fill_(999);zz,_=a(fs,1,True);torch.testing.assert_close(z,zz,atol=0,rtol=0)
        current=candidate_bank(fs,1,False);self.assertEqual(current[1].tolist(),[2,5]);self.assertEqual(current[3].tolist(),[[True,False],[False,True]])
    def test_first_frame_same_and_empty_reference(self):
        a=SliceRelationReadout();fs=[frame([2,9],0),frame([2],.2)]
        x,_=a(fs,0,False);y,_=a(fs,0,True);torch.testing.assert_close(x,y,atol=0,rtol=0)
        z,info=a([frame([2],0)],0,False);self.assertEqual(info['empty_queries'],1);self.assertTrue(torch.isfinite(z).all());z.sum().backward();self.assertIsNotNone(a.empty_reference.grad)
    def test_native_projection_and_backfill_alignment(self):
        xyz=torch.zeros(2,3,3);xyz[...,2]=torch.tensor([[-.01,.01,.13],[.14,.26,.26]])
        valid=torch.tensor([[True,True,True],[True,True,False]]);gt=torch.tensor([[True,False,False],[True,False,True]])
        ids,index,counts,edits=project_gt(xyz,valid,gt)
        self.assertEqual(ids.tolist(),[-1,0,1,2]);self.assertEqual(counts.tolist(),[1,1,2,1]);self.assertEqual(edits.tolist(),[1,0,1,0]);self.assertEqual(int(edits.sum()),int((gt&valid).sum()))
        score=torch.tensor([3.,-1.,2.,-2.]);native=torch.zeros(2,3);native[valid]=score[index]
        torch.testing.assert_close(native[valid],torch.tensor([3.,-1.,2.,2.,-2.]))
    def test_tail_unique_and_framewise_loss_normalization(self):
        seen=[]
        for start,end,lo in windows({'actual_frames':102}):self.assertLessEqual(lo,start);seen.extend(range(start,end))
        self.assertEqual(seen,list(range(102)))
        a=SliceRelationReadout();b=copy.deepcopy(a);fs=[frame([-1,1],0),frame([1,2,7],.09)];ys=[torch.tensor([0.,1.]),torch.tensor([1.,0.,0.])]
        loss=sum(torch.nn.functional.binary_cross_entropy_with_logits(a(fs,t,True)[0],y,reduction='sum',pos_weight=torch.tensor(2.)) for t,y in enumerate(ys))/5;loss.backward()
        for t,y in enumerate(ys):(torch.nn.functional.binary_cross_entropy_with_logits(b(fs,t,True)[0],y,reduction='sum',pos_weight=torch.tensor(2.))/5).backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is None:self.assertIsNone(q.grad)
            else:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-5,msg=name)
        # GT statistics are not numeric model inputs.
        before=b(fs,1,True)[0].detach();fs[0]['edit_counts']=torch.tensor([999,999]);torch.testing.assert_close(before,b(fs,1,True)[0].detach(),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
