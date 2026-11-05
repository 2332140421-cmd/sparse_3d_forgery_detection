"""Only exact native counting/point weights/prefix-window contracts."""
import numpy as np
import torch
from research_tools.pvst.geometry import voxelize
from research_tools.pvst.xyz_scale_budget_screen import frame_levels,windows,member_summary


def test_nested_negative_floor_preserves_raw_members_and_matches_model():
    points=np.array([[-.001,0,0],[-.25,0,0],[-.251,0,0],[.001,0,0],[.251,0,0],[.251,0,0],[1.,1.,1.]],np.float32)
    levels=frame_levels(points,.25)
    for h,(coords,members) in levels.items():
        direct,count=np.unique(np.floor(points/np.float32(h)).astype(np.int64),axis=0,return_counts=True)
        np.testing.assert_array_equal(coords,direct);np.testing.assert_array_equal(members,count);assert members.sum()==len(points)
        meta=voxelize(torch.from_numpy(points[None]),torch.ones(1,len(points),dtype=torch.bool),h)
        np.testing.assert_array_equal(coords,meta['coords'].numpy());np.testing.assert_array_equal(members,meta['count'].numpy())
        if h<2:assert len(levels[h*2][0])==len(meta['coarse_coords'])


def test_original_stride8_halo16_tail_and_temporal_sum():
    row={'video_id':'v','source_id':'s','actual_frames':25};counts=np.tile([3,2],(25,1));ww=windows(row,counts)
    assert [x['fine_nodes'] for x in ww]==[24,48,72,51]
    assert [x['coarse_nodes'] for x in ww]==[16,32,48,34]
    assert ww[-1]['target_range']==[24,25] and ww[-1]['input_range']==[8,25]
    assert len(ww)==4  # Same coordinates in different frames still different nodes.


def test_exact_block_weighted_quantiles_are_not_fine_block_counts():
    from collections import Counter
    a=[1,1,3,10];r=member_summary(Counter(a))
    assert r['median']==float(np.median(a));assert np.isclose(r['P90'],np.quantile(a,.9))
    assert r['valid_point_members']==15 and r['singleton_fraction']==.5
