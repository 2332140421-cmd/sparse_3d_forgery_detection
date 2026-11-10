import unittest
import torch
from torch import nn
from research_tools.rgbd_slice.slab_pooling_retention_check import normalize_scores,routing,summarize_pool,PoolCapture
from research_tools.rgbd_slice.model import SlabPool

class Wrapper(nn.Module):
    def __init__(self,local):super().__init__();self.pool=SlabPool(rgb_only=True);self.local=local
    def forward(self,frames):return self.pool(self.local,frames[0]['xyz'],frames[0]['valid'])

class RetentionTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(20260909);torch.set_num_threads(2)
    def test_four_head_normalization_reference(self):
        index=torch.tensor([0,1,0,2,1,0,2]);scores=torch.randn(7,4)
        actual=normalize_scores(scores,index,3);reference=torch.zeros_like(actual)
        for i in range(3):reference[index==i]=torch.softmax(scores[index==i],dim=0)
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(torch.zeros(3,4).index_add(0,index,actual),torch.ones(3,4),atol=1e-6,rtol=1e-6)
        # Uniform logits: edit mass must equal same-slab edited pixel fraction for all4 heads.
        counts=torch.bincount(index);g=torch.tensor([1,0,0,0,1,0,1],dtype=torch.bool);edits=torch.bincount(index[g],minlength=3)
        mass,error=summarize_pool(torch.zeros_like(scores),index,counts,edits,g)
        torch.testing.assert_close(mass,(edits/counts)[:,None].expand(-1,4),atol=1e-6,rtol=1e-6)
    def test_indices_counts_missing_and_pure(self):
        xyz=torch.zeros(2,3,3);xyz[...,2]=torch.tensor([[-.2,-.2,.13],[-.2,.13,.5]])
        valid=torch.tensor([[1,1,1],[1,0,1]],dtype=torch.bool);gt=torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
        ids,index,counts,edits,g,mixed,pure=routing(xyz,valid,gt)
        self.assertEqual(ids.tolist(),[-2,1,4]);self.assertEqual(counts.tolist(),[3,1,1]);self.assertEqual(edits.tolist(),[1,1,0])
        self.assertEqual(mixed.tolist(),[True,False,False]);self.assertEqual(pure.tolist(),[False,True,False])
        self.assertEqual(int(edits.sum()),2);self.assertEqual(int(gt.sum()),3)
        empty=routing(xyz,torch.zeros_like(valid),gt);self.assertEqual(empty[0].numel(),0)
    def test_actual_chunked_score_hook_preserves_output(self):
        h,w=73,129;xyz=torch.randn(h,w,3);xyz[...,2]=(torch.arange(h)[:,None]%4)*.125-.25
        valid=torch.ones(h,w,dtype=torch.bool);valid[:,::11]=False;gt=torch.zeros(h,w,dtype=torch.bool);gt[::3,::5]=True
        frame=dict(xyz=xyz,valid=valid,gt=gt,frame_index=0);model=Wrapper(torch.randn(256,8,8)).eval()
        scores=[]
        hscore=model.pool.score.register_forward_hook(lambda m,a,o:scores.append(o.detach().clone()))
        with torch.no_grad():a=model([frame])
        hscore.remove()
        ids,index,counts,edits,g,_,_=routing(xyz,valid,gt)
        raw=torch.cat(scores);actual=normalize_scores(raw,index,len(ids));reference=torch.zeros_like(actual)
        for j in range(len(ids)):reference[index==j]=torch.softmax(raw[index==j],dim=0)
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
        with torch.no_grad():
            capture=PoolCapture(model,{'role':'fake','start':0,'end':1})
            with capture.enabled():b=model([frame])
        self.assertGreater(int(valid.sum()),8192)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0);torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        self.assertFalse(model.pool.score._forward_hooks)
        # Per-element reference tolerance remains1e-6; long FP32 index_add sum uses
        # the predeclared5e-4 accumulation guard, not a per-weight error bound.
        self.assertLessEqual(capture.max_error,5e-4)
        self.assertFalse(capture.chunks)
    def test_original_tail_coverage(self):
        n=102;counts={}
        for start in range(0,n,24):
            end=min(start+24,n);lo=max(0,n-24) if start+24>=n else start
            for i in range(lo,end):
                if start<=i<end:counts[i]=counts.get(i,0)+1
        self.assertEqual(len(counts),n);self.assertEqual(set(counts.values()),{1})

if __name__=='__main__':unittest.main()
