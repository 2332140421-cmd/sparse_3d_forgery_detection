import unittest
import torch
from torch import nn
from torch.nn import functional as F
from research_tools.xyz_pilot.ainpaint_bn_diagnosis import image_bn,BN_NAMES,mode_check


def model():
    h=nn.Module();h.proj_norm=nn.BatchNorm2d(3)
    for n in ['up1','up2','up3']:
        branch=nn.Sequential();branch.add_module('norm',nn.BatchNorm2d(3));setattr(h,n,branch)
    h.final_conv_block=nn.Sequential(nn.Identity(),nn.BatchNorm2d(3));h.dropout=nn.Dropout(.25)
    m=nn.Module();m.head=h;m.other_bn=nn.BatchNorm2d(3);return m.eval()

class BNIntervention(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)
    def test_functional_statistics_preserve_state(self):
        m=model();b=m.head.proj_norm
        with torch.no_grad():b.running_mean.fill_(8);b.running_var.fill_(.2);b.weight.copy_(torch.tensor([1.,2.,3.]));b.bias.fill_(.3)
        before={k:v.clone() for k,v in m.state_dict().items()};x=torch.randn(1,3,7,9)*2+3
        with torch.no_grad(),image_bn(m):
            actual=b(x);expected=F.batch_norm(x,None,None,b.weight,b.bias,True,0.,b.eps)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertTrue(mode_check(m)['dropout_all_disabled'])
            self.assertFalse(torch.allclose(m.other_bn(x),expected))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        with torch.no_grad():self.assertFalse(torch.allclose(b(x),expected))
    def test_no_cross_image_accumulation_and_dropout_off(self):
        m=model();b=m.head.proj_norm;x=torch.randn(1,3,6,6);y=torch.randn(1,3,6,6)*9+19
        with torch.no_grad(),image_bn(m):
            first=b(x);b(y);again=b(x);torch.testing.assert_close(first,again,rtol=0,atol=0)
            torch.testing.assert_close(m.head.dropout(x),x,rtol=0,atol=0)
        self.assertEqual(int(b.num_batches_tracked),0)
    def test_restore_on_exception_and_scope(self):
        m=model();old=m.head.proj_norm.forward
        with self.assertRaisesRegex(ValueError,'test'):
            with torch.no_grad(),image_bn(m):raise ValueError('test')
        self.assertEqual(m.head.proj_norm.forward,old)
        m.head.train()
        with self.assertRaises(AssertionError):
            with image_bn(m):pass
    def test_requires_no_grad(self):
        m=model()
        with image_bn(m):
            with self.assertRaises(AssertionError):m.head.proj_norm(torch.randn(1,3,3,3))

if __name__=='__main__':unittest.main()
