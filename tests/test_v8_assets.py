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


@pytest.mark.parametrize('suffix',['.mp4','.mkv'])
def test_sample_decode_checks_actual_middle_and_end(tmp_path,suffix):
    import av
    import numpy as np
    from scripts.prepare_v8_assets import sample_decode
    path=tmp_path/('short'+suffix)
    with av.open(str(path),'w') as c:
        s=c.add_stream('mpeg4',rate=10);s.width=32;s.height=32;s.pix_fmt='yuv420p'
        for i in range(20):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),i,np.uint8),format='rgb24')
            for packet in s.encode(frame):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    result=sample_decode(path)
    assert len(result)==3 and result[0]['pts']<result[1]['pts']<result[2]['pts']
    assert all(r['hw']==[32,32] for r in result)


def test_segmented_remote_zip_range_and_crc(monkeypatch):
    import io
    from types import SimpleNamespace
    from scripts.download_v8_archive_sources import SegmentedZip
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('video_parts/0/S.mkv',b'source frames'*100)
    payload=b.getvalue();cut=len(payload)//2;parts=[payload[:cut],payload[cut:]]
    meta={'repo':'mirror','revision':'fixed','files':[{'path':'video_parts_part_'+suffix,'size':len(part)} for suffix,part in zip(('aa','ab'),parts)]}
    reader=SegmentedZip(meta)
    def get(url,headers,timeout):
        i=0 if url.endswith('_aa') else 1;a,b=map(int,headers['Range'].removeprefix('bytes=').split('-'))
        return SimpleNamespace(status_code=206,content=parts[i][a:b+1],url=url,raise_for_status=lambda:None)
    reader.session=SimpleNamespace(get=get)
    with zipfile.ZipFile(reader) as z:
        reader.prefetch(z.getinfo('video_parts/0/S.mkv'))
        assert z.read('video_parts/0/S.mkv')==b'source frames'*100


def test_native_real_container_is_explicit(tmp_path):
    p=tmp_path/'real/hcv1/S.mkv';p.parent.mkdir(parents=True);p.write_bytes(b'native container')
    assert real_path(tmp_path,'hcv1','S')==p
    p.with_suffix('.mp4').write_bytes(b'ambiguous second container')
    with pytest.raises(ValueError,match='AMBIGUOUS_REAL_CONTAINERS'):real_path(tmp_path,'hcv1','S')


def test_activitynet_direct_identity_and_provenance(tmp_path,monkeypatch):
    import json
    import scripts.prepare_v8_assets as assets
    audit=tmp_path/'asset_preparation';refs=audit/'references';refs.mkdir(parents=True)
    record={'video_id':'ABC','youtube_url':'https://www.youtube.com/watch?v=ABC',
            'video_url':'https://public-mirror/ABC.mp4','size_bytes':5}
    target={'repo':'mirror','revision':'fixed','matches':{'v_ABC':record},'unmatched_sources':[]}
    (refs/'activitynet-mirror-targets.json').write_text(json.dumps(target))
    (audit/'activitynet_sources.json').write_text(json.dumps({'v_ABC':{}}))
    class Response:
        headers={'ETag':'mirror-etag'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,n):yield b'media'
    calls=[]
    monkeypatch.setattr(assets.requests,'get',lambda *a,**k:(calls.append(k) or Response()))
    assets.download_activitynet(tmp_path)
    sidecar=json.loads((tmp_path/'real/acnet/v_ABC.mp4.json').read_text())
    assert sidecar['official_original_sha_verified'] is False
    assert sidecar['member_crc'] is None and sidecar['sha256']==digest(tmp_path/'real/acnet/v_ABC.mp4')
    assert calls[0]['headers']=={'Range':'bytes=0-4'}
    record['youtube_url']='https://www.youtube.com/watch?v=WRONG'
    (refs/'activitynet-mirror-targets.json').write_text(json.dumps(target))
    assets.download_activitynet(tmp_path)
    result=json.loads((audit/'activitynet_download_status.json').read_text())
    assert result['status']=='PARTIAL' and result['downloaded']==0


def test_activitynet_tar_extracts_only_exact_targets_and_resumes(tmp_path,monkeypatch):
    import io
    import json
    import tarfile
    import scripts.prepare_v8_assets as assets
    refs=tmp_path/'asset_preparation/references';refs.mkdir(parents=True)
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for source,data in [('v_OLD',b'existing'),('v_TARGET',b'wanted'),('v_UNRELATED',b'excluded')]:
            member=tarfile.TarInfo('v1-3/train_val/'+source+'.mp4');member.size=len(data)
            archive.addfile(member,io.BytesIO(data))
    payload=buffer.getvalue();entries=[];dest=tmp_path/'real_archives/activitynet_train_val/fixed';dest.mkdir(parents=True)
    split=len(payload)//2
    for suffix,data in [('00',payload[:split]),('01',payload[split:])]:
        p=dest/('v1-3_train_val.tar.gz.'+suffix);p.write_bytes(data)
        entries.append({'path':p.name,'size':len(data),'lfs':{'sha256':digest(p)}})
    (refs/'YimuWang_ActivityNet-files.json').write_text(json.dumps({'repo':'mirror','revision':'fixed','files':entries}))
    (refs/'activitynet-archive-targets.json').write_text(json.dumps({'unmatched_sources':['v_OLD','v_TARGET']}))
    old=tmp_path/'real/acnet/v_OLD.mp4';old.parent.mkdir(parents=True);old.write_bytes(b'existing')
    old.with_suffix('.mp4.json').write_text(json.dumps({'source_id':'acnet:v_OLD','revision':'fixed',
        'member':'v1-3/train_val/v_OLD.mp4','sha256':digest(old)}))
    monkeypatch.setattr(assets,'hf_hub_download',lambda *a,**k:str(k['local_dir']/a[1]))
    assets.recover_activitynet_tar(tmp_path)
    assert (tmp_path/'real/acnet/v_TARGET.mp4').read_bytes()==b'wanted'
    assert not (tmp_path/'real/acnet/v_UNRELATED.mp4').exists()
    status=json.loads((tmp_path/'asset_preparation/activitynet_tar_status.json').read_text())
    assert status['status']=='COMPLETE_SCAN' and status['recovered']==2 and status['archive_video_members']==3
