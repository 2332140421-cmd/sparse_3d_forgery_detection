import io
import json

import numpy as np
from PIL import Image
import pytest

from scripts.review_anakin import alignment, mask_array, mask_timing, numeric_members
from research_tools.v8.spatial_localization.data import mask_to_canvas, to_source


def test_zip_members_are_numeric_not_archive_or_lexical_order():
    assert numeric_members(['m/10.png','m/2.png','m/1.png'])==[(1,'m/1.png'),(2,'m/2.png'),(10,'m/10.png')]
    with pytest.raises(ValueError,match='DUPLICATE'):
        numeric_members(['a/1.png','b/1.png'])


def test_complete_frame_correspondence_and_missing_are_distinct():
    assert alignment([0,1,2],3)['index_base']==0
    assert alignment([1,2,3],3)['index_base']==1
    assert alignment([0,2],3)['status']=='alignment_unresolved'
    assert alignment([0,1,2],4)['missing_annotation_not_normal']


def test_soft_boundary_threshold_preserves_original_values_and_empty():
    a=np.array([[0,127,128,255]],dtype=np.uint8);b=io.BytesIO();Image.fromarray(a).save(b,format='PNG')
    raw,binary,mode=mask_array(b.getvalue())
    assert np.array_equal(raw,a) and binary.tolist()==[[False,False,True,True]]
    b=io.BytesIO();Image.fromarray(np.zeros((2,3),np.uint8)).save(b,format='PNG')
    assert not mask_array(b.getvalue())[1].any()
    assert alignment([],1)['status']=='alignment_unresolved'


def test_rgb_uses_explicit_luminance_not_any_channel():
    b=io.BytesIO();Image.fromarray(np.array([[[127,128,129],[255,255,255],[0,0,0]]],np.uint8)).save(b,format='PNG')
    raw,binary,mode=mask_array(b.getvalue())
    assert raw.tolist()==[[128,255,0]] and binary.tolist()==[[True,True,False]] and mode=='RGB'


def test_existing_mask_mapping_round_trip_and_padding():
    m=np.array([[False,True],[True,False]])
    mapping={'source_hw':[2,2],'canvas_hw':[6,8],'content_hw':[4,4],'pad_xy':[2,1]}
    canvas=mask_to_canvas(m,mapping)
    assert not canvas[:1].any() and not canvas[:,:2].any()
    assert np.array_equal(to_source(canvas,mapping,semantic_mask=True),m)


def test_temporal_ranges_require_complete_masks_and_use_actual_pts():
    pts=[{'seconds':t} for t in [0,.04,.10,.14]]
    result=mask_timing(alignment([1,2,3,4],4),[1,3,4],pts)
    assert result['profile']=='mixed_empty_nonempty'
    assert result['positive_runs'][1]['first_pts_s']==.10
    assert mask_timing(alignment([0,2],4),[0,2],pts)['profile']=='unresolved'
    assert mask_timing(alignment([0,1,2,3],4),[0,1,2,3],pts)['profile']=='all_nonempty'


def test_selection_rejects_modified_pinned_metadata(tmp_path):
    import hashlib
    from scripts.review_anakin import select
    refs=tmp_path/'references';refs.mkdir()
    entries=[]
    for name in ['metadata.csv','ANAKIN.py']:
        data=b'original'
        (refs/name).write_bytes(data)
        blob=hashlib.sha1(b'blob 8\0'+data).hexdigest()
        entries.append({'rfilename':name,'size':8,'blobId':blob})
    (refs/'dataset_info.json').write_text(json.dumps({'sha':'fixed','siblings':entries}))
    (refs/'metadata.csv').write_bytes(b'modified')
    with pytest.raises(ValueError,match='PINNED_REFERENCE_INTEGRITY_FAILED'):
        select(tmp_path,tmp_path/'selection.json')
