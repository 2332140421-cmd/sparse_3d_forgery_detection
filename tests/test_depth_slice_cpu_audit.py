import numpy as np
from research_tools.pvst.depth_slice_cpu_audit import slice_stats,selection


def test_floor_boundary_empty_and_causal_edges():
    hist={};records=[]
    for z in [[-.01,.0],[0.,.125],[0.],[]]:
        d=np.array([z],np.float32);v=np.ones_like(d,bool)
        records.append(slice_stats(d,v,.125,0,hist))
    assert records[0]['id_min']==-1 and records[0]['id_max']==0
    assert records[1]['id_max']==1
    assert sum(x['temporal_edges'] for x in records)==3
    assert sum(x['depth_directed_edges'] for x in records)==4
    assert sum(x['units_with_history'] for x in records)==2
    assert records[-1]['occupied']==0 and records[-1]['members_median'] is None
    shifted=slice_stats(np.array([[0.,.125]],np.float32),np.ones((1,2),bool),.125,.0625,{})
    assert shifted['id_min']==-1 and shifted['id_max']==0


def test_selection_has_no_depth_or_readability_replacement():
    def row(source,role='real',split='train'):
        return {'dataset':'ANAKIN','source_id':source,'video_id':source,'role':role,'split':split,'actual_frames':30}
    r=selection([row('b'),row('a'),row('c'),row('a'),row('d'),row('0','fake'),row('00',split='dev')])
    assert [x['source_id'] for x in r]==['a','b','c']
    assert all(x['frame_range']==[3,27] for x in r)
