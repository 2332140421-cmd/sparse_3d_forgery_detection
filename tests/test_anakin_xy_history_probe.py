import unittest
import numpy as np
from research_tools.rgbd_slice.anakin_xy_history_probe import collapse,window,weighted_rows
from research_tools.rgbd_slice.grid_localization import count_metrics
from sklearn.metrics import average_precision_score

class XYHistory(unittest.TestCase):
    def test_count_collapse_history_and_ap(self):
        C=np.array([[1.,0],[3.,0],[0.,2]])
        F,n,g=collapse(C,np.array([0,0,1]),np.array([1,3,2]),np.array([1,0,1]))
        np.testing.assert_array_equal(F,[[2.5,0],[0,2]]);self.assertEqual(n.sum(),6);self.assertEqual(g.sum(),2)
        frames={i:{'F':F,'keys':[4,9]} for i in range(4)}
        pos,h,_=window(frames,[0,1,2,3]);np.testing.assert_array_equal(h,[0,0]);np.testing.assert_array_equal(pos,[0,1])
        x,y,w=weighted_rows(F,n,g)
        self.assertEqual(w[y==1].sum(),2);self.assertEqual(w[y==0].sum(),4)
        z=np.array([.1,.2]);actual=count_metrics(z,n,g)['pixel_AP']
        labels=np.r_[np.ones(g[0]),np.zeros(n[0]-g[0]),np.ones(g[1]),np.zeros(n[1]-g[1])]
        self.assertAlmostEqual(actual,average_precision_score(labels,np.repeat(z,n)))
