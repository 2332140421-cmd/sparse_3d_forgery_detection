import unittest
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from research_tools.rgbd_slice.grid_feature_readout_probe import sample_pixels,pixel_units,grid_mean
from research_tools.rgbd_slice.grid_localization import grouping,count_metrics
from research_tools.rgbd_slice.model import native_sample
from research_tools.rgbd_slice.local_pixel_readout_probe import folded


class GridFeatureProbeTest(unittest.TestCase):
    def test_shared_train_samples(self):
        gt=np.zeros((20,30),bool);gt[:,15:]=1;valid=np.ones_like(gt);valid[0]=0
        a,y=sample_pixels(gt,valid,'video',12,'train');b,z=sample_pixels(gt,valid,'video',12,'train')
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(y,z)
        self.assertEqual(len(a),256);self.assertEqual(len(set(a)),256);self.assertTrue(valid.ravel()[a].all())
        with self.assertRaises(ValueError):sample_pixels(gt,valid,'video',12,'dev')
    def test_native_index_and_grid_counts_AP(self):
        h,w=5,7;local=torch.arange(256*h*w).float().reshape(256,h,w)
        pts=torch.tensor([0,6,7,17,34]);actual=native_sample(local,pts,(h,w))
        torch.testing.assert_close(actual,local.reshape(256,-1)[:,pts].T)
        xyz=torch.zeros(h,w,3);xyz[...,2]=.3;valid=torch.ones(h,w,dtype=torch.bool);valid[0,0]=False
        points,groups,L=grouping(xyz,valid);ids,index=groups[1]
        chosen=points.numpy()[[0,7,-1]];routing=pixel_units(points.numpy(),index.numpy(),chosen)
        np.testing.assert_array_equal(routing,index.numpy()[[0,7,-1]])
        idx=np.array([0,0,1,1,1,2]);gt=np.array([1,0,0,1,0,1]);z=np.array([.2,.4,-.3,.5,.1,.6]);n=np.bincount(idx);g=np.bincount(idx,weights=gt).astype(int)
        means=grid_mean(z,idx,n);np.testing.assert_allclose(means,[.3,.1,.6])
        for unit_scores in [means,np.array([.1,.1,.5])]:
            self.assertAlmostEqual(count_metrics(unit_scores,n,g)['pixel_AP'],average_precision_score(gt,unit_scores[idx]))
        self.assertEqual(int(n.sum()),len(idx));self.assertEqual(int(g.sum()),int(gt.sum()))
    def test_folded_linear(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(100,256)).astype('float32');y=np.arange(100)%2
        scale=StandardScaler().fit(x);clf=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,class_weight='balanced').fit(scale.transform(x),y)
        w,b=folded(scale,clf);np.testing.assert_allclose(x@w+b,clf.decision_function(scale.transform(x)),atol=1e-5,rtol=1e-5)

if __name__=='__main__':unittest.main()
