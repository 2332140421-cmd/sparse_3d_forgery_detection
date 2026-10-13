import numpy as np
import pytest

from research_tools.v8.full_video_global_context.deepaction import REVISION, ordered_actions, prefix_within_budget, validate_manifest
from research_tools.v8.full_video_global_context.independent_test import GROUPS, bootstrap


def test_selection_fixed_and_not_length_sorted():
    assert ordered_actions()[:5]==[76,47,18,32,29]
    pairs=[[{'metadata_frame_count':7}],[{'metadata_frame_count':8}],[{'metadata_frame_count':1}]]
    selected,total=prefix_within_budget(pairs,10)
    assert selected==pairs[:1] and total==7  # Do not skip the second to favor a shorter third.


def test_action_cluster_not_media_source():
    rows=[{'source_id':f'unique_media_{i}','action_id':str(i//2),'video_label':i%2} for i in range(4)]
    z={g:np.array([0.,1.,0.,1.]) for g in GROUPS}
    own,paired=bootstrap(rows,z,cluster_key='action_id')
    assert all(r['auroc']==1 and r['ci95']==[1,1] for r in own.values())
    assert all(r['difference']==0 and r['ci95']==[0,0] for r in paired.values())


def test_manifest_semantic_cluster_not_shared_source_and_labels():
    rows=[{'source_id':str(i),'video_id':str(i),'action_id':'7','generator':g,'role':role,'video_label':i,
           'official_path':f'{g}/7/a.mp4','official_revision':REVISION} for i,(g,role) in enumerate([('Pexels','real'),('RunwayML','fake')])]
    validate_manifest(rows)
    rows[1]['source_id']=rows[0]['source_id']
    with pytest.raises(ValueError,match='IDENTITY_NOT_UNIQUE'):validate_manifest(rows)
    rows[1]['source_id']='1';rows[1]['video_label']=0
    with pytest.raises(ValueError,match='LABEL_IDENTITY'):validate_manifest(rows)
