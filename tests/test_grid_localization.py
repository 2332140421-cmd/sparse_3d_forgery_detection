import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from research_tools.rgbd_slice.grid_localization import CurrentReadout,grouping,frozen_pool,count_loss,count_metrics
from research_tools.rgbd_slice.slice_relation_readout import SliceRelationReadout
from research_tools.rgbd_slice.model import SlabPool,native_sample

class GridTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(20260909)
    def equivalence(self,device):
        a=SliceRelationReadout().to(device);b=CurrentReadout(chunk=7).to(device);b.load_state_dict(a.state_dict(),strict=True)
        x=torch.randn(19,512,device=device,requires_grad=True);y=x.detach().clone().requires_grad_();ids=torch.arange(-4,15,device=device)
        za=a([{'features':x,'ids':ids,'pts':.17}],0,False)[0];zb=b(y,ids)
        torch.testing.assert_close(za,zb,atol=2e-5,rtol=3e-5)
        (za.square().sum()).backward();(zb.square().sum()).backward()
        errors={'output':float((za-zb).abs().max()),'input_gradient':float((x.grad-y.grad).abs().max()),'parameter_gradient':0.}
        torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=3e-5)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is None:self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-5,msg=name);errors['parameter_gradient']=max(errors['parameter_gradient'],float((p.grad-q.grad).abs().max()))
        self.assertEqual(list(a.state_dict()),list(b.state_dict()));return errors
    def test_current_only_equivalence(self):self.equivalence('cpu')
    def test_mask_and_empty_reference(self):
        q=torch.randn(1,8,3,64);k=torch.randn(1,8,5,64);v=torch.randn_like(k);out=CurrentReadout.attend(q,k,v,1);changed=v.clone();changed[:,:,2]+=100
        z=CurrentReadout.attend(q,k,changed,1);torch.testing.assert_close(out[:,:,1],z[:,:,1],atol=0,rtol=0)
        b=CurrentReadout();x=torch.randn(1,512);z=b(x,torch.tensor([5]));z.sum().backward();self.assertIsNotNone(b.empty_reference.grad)
    def test_pooling_shared_weights_and_routing(self):
        h,w=17,19;xyz=torch.randn(h,w,3);valid=torch.rand(h,w)>.2;local=torch.randn(256,5,6);pool=SlabPool(rgb_only=True).eval();points,groups,L=grouping(xyz,valid)
        self.assertEqual(L,2);out=frozen_pool(pool,local,points,groups,(h,w),chunk=31);old=pool(local,xyz,valid)[0]
        torch.testing.assert_close(out[0],old,atol=2e-5,rtol=3e-5)
        visual=native_sample(local,points,(h,w));scores=pool.score(torch.cat((visual,torch.zeros(len(points),3)),-1));values=pool.value(visual).reshape(-1,4,128)
        for result,(ids,index) in zip(out,groups):
            expected=torch.stack([(values[index==i]*scores[index==i].softmax(0)[...,None]).sum(0).reshape(512) for i in range(len(ids))]);torch.testing.assert_close(result,expected,atol=2e-5,rtol=3e-5)
            self.assertEqual(int(torch.bincount(index).sum()),int(valid.sum()))
    def test_count_BCE_and_tied_pixel_AP(self):
        n=torch.tensor([2.,3.,4.,1.]);g=torch.tensor([1.,0.,3.,1.]);z=torch.tensor([.2,-.3,.2,1.],requires_grad=True);q=z.detach().clone().requires_grad_();ix=torch.repeat_interleave(torch.arange(4),n.long());truth=torch.cat([torch.cat((torch.ones(int(a)),torch.zeros(int(b-a)))) for a,b in zip(g,n)])
        a=count_loss(z,n,g,2.3,n.sum());b=F.binary_cross_entropy_with_logits(q[ix],truth,pos_weight=torch.tensor(2.3));torch.testing.assert_close(a,b);a.backward();b.backward();torch.testing.assert_close(z.grad,q.grad)
        metric=count_metrics(z.detach().numpy(),n.numpy(),g.numpy());self.assertAlmostEqual(metric['pixel_AP'],average_precision_score(truth.numpy(),z.detach().numpy()[ix.numpy()]),places=14)
        self.assertIsNone(count_metrics([1.],[2],[0])['pixel_AP'])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA preflight in original environment')
    def test_CUDA_equivalence(self):print('CUDA_READOUT_ERRORS',self.equivalence('cuda'),flush=True)

if __name__=='__main__':unittest.main()
