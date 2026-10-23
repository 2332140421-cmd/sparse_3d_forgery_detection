import cv2
import numpy as np
import pytest
from research_tools.v8.spatial_localization.data import (
    media_identity, match_media, content_mask, mask_to_canvas, to_source, validate_annotation,
)
from research_tools.v8.full_coverage_observation_field.frontend import letterbox


def test_exact_identity_ignores_colliding_pair_id():
    rows=[dict(pair_id='collision',source_id=s,role=r,media_path=p) for s,r,p in
          [('a','real','a.mp4'),('a','fake','fake/a.mp4'),('b','real','b.mp4')]]
    assert match_media(rows,'a','fake','fake/a.mp4') is rows[1]
    assert len({media_identity(r['source_id'],r['role'],r['media_path']) for r in rows})==3
    with pytest.raises(ValueError):match_media(rows,'a','real','b.mp4')
    with pytest.raises(ValueError):match_media(rows+rows,'a','real','a.mp4')


def test_rounded_letterbox_and_padding():
    image=np.zeros((101,203,3),np.uint8);_,mapping=letterbox(image)
    mask=np.zeros((101,203),bool);mask[20:70,40:140]=True
    canvas=mask_to_canvas(mask,mapping);content=content_mask(mapping)
    assert not canvas[~content].any()
    restored=to_source(canvas,mapping,semantic_mask=True)
    assert (restored&mask).sum()/(restored|mask).sum()>.94
    # Padding response must not leak into inverse-mapped content.
    response=np.where(content,3.,999.).astype(np.float32)
    np.testing.assert_allclose(to_source(response,mapping),3.)


def test_annotation_states_are_not_empty_truth(tmp_path):
    row={'target':'visible_forgery_anomaly','annotation_status':'unannotated','mask_path':None,'source_hw':[5,7]}
    assert validate_annotation(row,tmp_path) is None
    row.update(annotation_status='normal_empty')
    with pytest.raises(ValueError):validate_annotation(row,tmp_path)
    cv2.imwrite(str(tmp_path/'empty.png'),np.zeros((5,7),np.uint8))
    row['mask_path']='empty.png';assert not validate_annotation(row,tmp_path).any()
    row['annotation_status']='annotated'
    with pytest.raises(ValueError):validate_annotation(row,tmp_path)


def test_official_metadata_variants_and_intervals(tmp_path):
    import json
    import zipfile
    from research_tools.v8.spatial_localization.prepare import inventory
    names=['video/02_wan/A+1.00=2.00=charades@train_delete@A@214@wan.mp4',
           'video/05_ltx/B+1.00=2.00=hcv1@test_delete@B@ltx.mp4']
    (tmp_path/'dataset-info.json').write_text(json.dumps({'sha':'fixed-revision','siblings':[{'rfilename':n} for n in names]}))
    for split,name in zip(('train','test'),names):
        (tmp_path/f'{split}.csv').write_text('file_name\n'+name+'\n')
    with zipfile.ZipFile(tmp_path/'annot.zip','w') as z:
        for split,generator,name in zip(('train','test'),('wan','ltx'),names):
            z.writestr(f'annot/{split}@{generator}.txt',name.rsplit('/',1)[1]+' 3.00 1.00=2.00\n')
    result=inventory(tmp_path)
    assert result['summary']['unique_sources']==2
    assert result['rows'][1]['source_id']=='hcv1:B'
    assert result['rows'][0]['intervals_s']==[[1.,2.]]


def test_random_model_interface_and_gradients():
    import torch
    from research_tools.v8.spatial_localization.smoke import check_model
    torch.set_num_threads(2)
    content=torch.zeros(1,2,256,16,16,dtype=torch.bool);content[:,:,:2]=True
    batch={'x':torch.zeros(1,2,256,77),'content':content,'valid':content.clone(),
           'mu':torch.zeros(1,2,256,3),'area':content.sum((-1,-2)).float(),
           'component_ids':torch.arange(256)[None,None].expand(1,2,256),
           'predecessor':torch.zeros(1,2,256,4,dtype=torch.long),
           'association_weight':torch.zeros(1,2,256,4),'history_available':content.any((-1,-2)),
           'delta_t':torch.tensor([[0.,.033]])}
    for condition in ('GLOBAL_CELL','GLOBAL_COMPONENT'):
        response,result=check_model(batch,condition,'cpu')
        assert response.shape==(1,2,256) and result['finite_gradients']
