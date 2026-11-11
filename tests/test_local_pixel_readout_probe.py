import unittest
from types import SimpleNamespace
import numpy as np
import torch
from research_tools.rgbd_slice.local_pixel_readout_probe import sample_indices,feature_pixels,folded,scored_range
from research_tools.rgbd_slice.model import Local


class LocalPixelReadoutTest(unittest.TestCase):
    def test_train_only_reproducible_native_sampling(self):
        gt=np.zeros((23,31),bool);gt[3:11,4:23]=True
        ids,y=sample_indices(gt,'v',7,'train');again,yy=sample_indices(gt,'v',7,'train')
        np.testing.assert_array_equal(ids,again);np.testing.assert_array_equal(y,yy)
        self.assertEqual(len(set(ids)),256);self.assertEqual(int(y.sum()),128)
        np.testing.assert_array_equal(y,gt.reshape(-1)[ids]);self.assertRaises(ValueError,sample_indices,gt,'v',7,'dev')
        ids,y=sample_indices(np.zeros((2,3),bool),'r',0,'train');self.assertEqual(len(ids),6);self.assertEqual(y.sum(),0)
        x=torch.arange(32*23*31).reshape(1,32,23,31)
        selected=feature_pixels(x,gt.shape,ids=np.array([0,32,712]))
        torch.testing.assert_close(selected,torch.stack([x[0,:,i//31,i%31] for i in [0,32,712]]))
        self.assertRaises(ValueError,feature_pixels,x,(31,23))

    def test_folded_fp32_matches_standardized_linear(self):
        rng=np.random.default_rng(20260909);x=rng.normal(size=(257,32)).astype(np.float32)
        scaler=SimpleNamespace(mean_=rng.normal(size=32),scale_=rng.uniform(.5,2,size=32))
        clf=SimpleNamespace(coef_=rng.normal(size=(1,32)),intercept_=np.array([.37]))
        w,b=folded(scaler,clf);ref=((x-scaler.mean_)/scaler.scale_)@clf.coef_[0]+clf.intercept_[0]
        np.testing.assert_allclose(x@w+b,ref,atol=1e-5,rtol=1e-5)

    def test_tail_context_is_not_scored_twice(self):
        n=102;seen=[]
        for s in range(0,n,24):
            end=min(s+24,n);lo=max(0,n-24) if s+24>=n else s
            fs=[{'frame_index':i} for i in range(lo,end)];a,b=scored_range(fs,n,None,'v');seen+=list(range(a,b))
        self.assertEqual(seen,list(range(n)))
        clips=[{'video_id':'v','start':25,'end':49}]
        self.assertEqual(scored_range([{'frame_index':i} for i in range(25,49)],100,clips,'v'),(25,49))

    def test_actual_native_hook_preserves_decode(self):
        torch.set_num_threads(1);torch.manual_seed(20260909);model=Local(torch.nn.Identity()).eval()
        local=torch.randn(256,2,3);rgb=torch.randint(0,256,(5,7,3),dtype=torch.uint8);context=torch.zeros(1,32,5,7);found=[]
        with torch.no_grad():before=model.decode(local,rgb,context)
        handle=model.native.register_forward_hook(lambda m,a,o: found.append(feature_pixels(o,(5,7),[0,34]).clone()))
        try:
            with torch.no_grad():after=model.decode(local,rgb,context)
        finally:handle.remove()
        torch.testing.assert_close(before,after,atol=0,rtol=0);self.assertEqual(found[0].shape,(2,32));self.assertFalse(found[0].requires_grad)


if __name__=='__main__':unittest.main()
