import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts.prepare_v8_assets import digest, verify, real_path, extract_charades


def test_official_hash_and_lfs_pointer_are_distinct(tmp_path):
    p=tmp_path/'video.mp4';p.write_bytes(b'real media bytes')
    entry={'size':p.stat().st_size,'lfs':{'sha256':digest(p)}}
    assert verify(p,entry)['status']=='VERIFIED'
    p.write_bytes(b'wrong same bytes')
    assert p.stat().st_size==entry['size']
    assert verify(p,entry)['status']=='HASH_MISMATCH'
    p.write_text('version https://git-lfs.github.com/spec/v1\n')
    assert verify(p,entry)['status']=='SIZE_MISMATCH'
    assert verify(tmp_path/'missing',entry)['status']=='MISSING'


def test_git_blob_checksum_is_not_plain_sha(tmp_path):
    p=tmp_path/'metadata.csv';p.write_bytes(b'file_name\n')
    blob=hashlib.sha1(b'blob 10\0file_name\n').hexdigest()
    assert verify(p,{'size':10,'lfs':None,'blob_id':blob})['status']=='VERIFIED'
    assert blob!=digest(p,'sha1')


def test_source_namespace_and_existing_real_reuse(tmp_path):
    old=tmp_path/'media/charades/Charades_v1_480/S.mp4';old.parent.mkdir(parents=True);old.write_bytes(b'existing')
    assert real_path(tmp_path,'charades','S')==old
    assert real_path(tmp_path,'acnet','S')!=old
    assert real_path(tmp_path,'hcv1','S')!=real_path(tmp_path,'acnet','S')


def test_charades_exact_members_crc_and_no_unrelated_extraction(tmp_path,monkeypatch):
    import scripts.prepare_v8_assets as assets
    monkeypatch.setattr(assets,'identities',lambda root:[{'source_id':'charades:S','scene_source':'charades'},
                                                       {'source_id':'charades:S','scene_source':'charades'}])
    archive=tmp_path/'real_archives/Charades_v1_480.zip';archive.parent.mkdir()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('Charades_v1_480/S.mp4',b'exact source');z.writestr('Charades_v1_480/OTHER.mp4',b'unrelated')
    extract_charades(tmp_path)
    assert real_path(tmp_path,'charades','S').read_bytes()==b'exact source'
    assert not real_path(tmp_path,'charades','OTHER').exists()
    real_path(tmp_path,'charades','S').write_bytes(b'wrong source')
    with pytest.raises(ValueError,match='EXISTING_REAL_MISMATCH'):extract_charades(tmp_path)


def test_resume_cannot_silently_change_official_revision(tmp_path):
    import json
    from scripts.prepare_v8_assets import official
    audit=tmp_path/'asset_preparation';audit.mkdir()
    (audit/'official_files.json').write_text(json.dumps({'revision':'frozen','files':[]}))
    with pytest.raises(ValueError,match='PINNED_REVISION_CONFLICT'):official(tmp_path,'different')


def test_long_published_name_uses_official_blob_cache(tmp_path,monkeypatch):
    import json
    import scripts.prepare_v8_assets as assets
    name='video/'+('x'*243)+'.mp4'
    cache=tmp_path/'blob';cache.write_bytes(b'official content')
    audit=tmp_path/'asset_preparation';audit.mkdir()
    entry={'path':name,'size':cache.stat().st_size,'lfs':{'sha256':digest(cache)}}
    (audit/'official_files.json').write_text(json.dumps({'revision':'fixed','files':[entry]}))
    calls=[]
    monkeypatch.setattr(assets,'snapshot_download',lambda *a,**k:calls.append(k))
    monkeypatch.setattr(assets,'hf_hub_download',lambda *a,**k:str(cache))
    assets.official(tmp_path,'fixed')
    assert calls[0]['ignore_patterns']==[name] and calls[0]['max_workers']==4
    assert verify(tmp_path/'official/fixed'/name,entry)['status']=='VERIFIED'


def test_sample_decode_checks_actual_middle_and_end(tmp_path):
    import av
    import numpy as np
    from scripts.prepare_v8_assets import sample_decode
    path=tmp_path/'short.mp4'
    with av.open(str(path),'w') as c:
        s=c.add_stream('mpeg4',rate=10);s.width=32;s.height=32;s.pix_fmt='yuv420p'
        for i in range(20):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i,np.uint8),format='rgb24')
            for packet in s.encode(frame):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    result=sample_decode(path)
    assert len(result)==3 and result[0]['pts']<result[1]['pts']<result[2]['pts']
    assert all(r['hw']==[32,32] for r in result)
