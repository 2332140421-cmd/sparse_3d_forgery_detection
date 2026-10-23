"""Download/verify fixed official assets; no frontend, detector or training calls.

Supports V8 probe-media recovery. Direct caller is the documented CLI; outputs
are a pinned download manifest and media JSONL. Existing experiment runners
require absent old caches and cannot serve this raw-asset preparation task.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import av
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import requests


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def digest(path, algorithm='sha256', git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob: h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify(path, entry):
    if not path.is_file(): return {'status':'MISSING'}
    size = path.stat().st_size
    if size != entry['size']: return {'status':'SIZE_MISMATCH','bytes':size}
    sha = digest(path)
    expected = entry.get('lfs')
    matched = sha == expected['sha256'] if expected else digest(path,'sha1',True) == entry['blob_id']
    return {'status':'VERIFIED' if matched else 'HASH_MISMATCH','bytes':size,'sha256':sha,
            'official_checksum_type':'LFS_SHA256' if expected else 'GIT_BLOB_SHA1'}


def official(root, revision=None):
    audit = root/'asset_preparation'; manifest = audit/'official_files.json'
    if manifest.exists():
        data=json.loads(manifest.read_text())
        if revision and data['revision']!=revision:raise ValueError('PINNED_REVISION_CONFLICT')
    else:
        info=HfApi().dataset_info('ActivityForensics/ActivityForensics',revision=revision,files_metadata=True)
        data={'revision':info.sha,'files':[{'path':x.rfilename,'size':x.size,'blob_id':x.blob_id,
                'lfs':dict(x.lfs) if x.lfs else None} for x in info.siblings]};save(manifest,data)
    start=time.time();dest=root/'official'/data['revision']
    save(audit/'download_status.json',{'status':'RUNNING','revision':data['revision'],'started_unix':start})
    try:
        # local_dir metadata adds '.metadata'; one published basename is 247 bytes.
        # Only these exceptional names use HF's hash-named blob cache.
        long_names=[e['path'] for e in data['files'] if len(Path(e['path']).name.encode())+len('.metadata')>255]
        snapshot_download('ActivityForensics/ActivityForensics',repo_type='dataset',revision=data['revision'],
                          local_dir=dest,max_workers=4,ignore_patterns=long_names or None)
        for name in long_names:
            cached=Path(hf_hub_download('ActivityForensics/ActivityForensics',name,repo_type='dataset',
                                       revision=data['revision']))
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            entry=next(e for e in data['files'] if e['path']==name)
            if target.exists():
                if verify(target,entry)['status']!='VERIFIED':raise ValueError('EXISTING_LONG_FILE_INVALID:'+name)
            else:os.link(cached.resolve(),target)
    except Exception as e:
        save(audit/'download_status.json',{'status':'PARTIAL','revision':data['revision'],'elapsed_s':time.time()-start,'error':str(e)})
        raise
    save(audit/'download_status.json',{'status':'DOWNLOADED_PENDING_VERIFICATION','revision':data['revision'],'download_s':time.time()-start})


def download_charades(root):
    url='https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip'
    dest=root/'real_archives/Charades_v1_480.zip';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): return
    partial=dest.with_suffix('.zip.part');offset=partial.stat().st_size if partial.exists() else 0
    start=time.time()
    with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,90)) as r:
        r.raise_for_status()
        if offset and (r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
            raise ValueError('RANGE_RESUME_NOT_HONORED')
        with partial.open('ab' if offset else 'wb') as f:
            for chunk in r.iter_content(4*1024*1024):f.write(chunk)
    partial.replace(dest)
    save(root/'asset_preparation/charades_archive.json',{'url':url,'bytes':dest.stat().st_size,
        'download_s_this_invocation':time.time()-start,'resumed_bytes':offset,'official_hash_available':False})


def real_path(root, dataset, source):
    old=root/'media/charades/Charades_v1_480'/f'{source}.mp4'
    return old if dataset=='charades' and old.exists() else root/'real'/dataset/f'{source}.mp4'


def identities(root):
    # Reuse the already-audited official CSV/annotation parser; no old numeric assets.
    from research_tools.v8.spatial_localization.prepare import inventory
    manifest=json.loads((root/'asset_preparation/official_files.json').read_text())
    entries={r['path']:r for r in manifest['files']}
    for local,remote in [('train.csv','metadata/train.csv'),('test.csv','metadata/test.csv'),('annot.zip','annot.zip')]:
        if verify(root/'metadata'/local,entries[remote])['status']!='VERIFIED':
            raise ValueError('LOCAL_METADATA_CHECKSUM_MISMATCH:'+local)
    parsed=inventory(root/'metadata')
    rows=parsed['rows']
    pinned=manifest['revision']
    if parsed['revision']!=pinned: raise ValueError('METADATA_REVISION_MISMATCH')
    return rows


def extract_charades(root):
    rows=identities(root);sources=sorted({r['source_id'].split(':',1)[1] for r in rows if r['scene_source']=='charades'})
    start=time.time();results=[]
    with zipfile.ZipFile(root/'real_archives/Charades_v1_480.zip') as z:
        members=set(z.namelist())
        for source in sources:
            member=f'Charades_v1_480/{source}.mp4';out=real_path(root,'charades',source)
            if member not in members: raise ValueError(f'OFFICIAL_MEMBER_MISSING:{source}')
            if out.exists():
                # Existing smoke media must match official member content, not just its name.
                import zlib
                crc=0
                with out.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):crc=zlib.crc32(block,crc)
                if out.stat().st_size!=z.getinfo(member).file_size or crc!=z.getinfo(member).CRC:
                    raise ValueError(f'EXISTING_REAL_MISMATCH:{source}')
            else:
                out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.mp4.part')
                with z.open(member) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                tmp.replace(out)  # ZipFile checks CRC on the complete member read.
            results.append({'source_id':'charades:'+source,'path':str(out),'bytes':out.stat().st_size,'zip_crc_checked':True})
        save(root/'asset_preparation/charades_extraction.json',{'status':'COMPLETE','target':len(sources),
            'archive_video_count':sum(n.endswith('.mp4') for n in members),'extraction_s':time.time()-start,'rows':results})


def container_info(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        if s.width<=0 or s.height<=0:raise ValueError('INVALID_VIDEO_DIMENSIONS')
        return {'readable':True,'hw':[s.height,s.width],'frames_metadata':s.frames,
                'time_base':str(s.time_base),'duration_s':float(s.duration*s.time_base) if s.duration else None}


def sample_decode(path):
    results=[]
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        for fraction in (0.,.5,.98):
            target=int((s.duration or 0)*fraction)
            c.seek(target,stream=s)
            frame=None
            for candidate in c.decode(s):
                frame=candidate
                if candidate.pts is not None and candidate.pts>=target:break
            if frame is None:raise ValueError('SAMPLE_DECODE_EMPTY')
            results.append({'fraction':fraction,'pts':frame.pts,'hw':[frame.height,frame.width]})
    return results


def index(root):
    started=time.time();audit=root/'asset_preparation';manifest=json.loads((audit/'official_files.json').read_text())
    base=root/'official'/manifest['revision'];checks={};failures=[]
    # All published files, including feat.zip; local SHA is explicitly separate from official checks.
    for i,e in enumerate(manifest['files']):
        result=verify(base/e['path'],e);checks[e['path']]=result
        if result['status']!='VERIFIED':failures.append({'path':e['path'],**result})
        if i%100==0:print('verified',i,'/',len(manifest['files']),flush=True)
    rows=identities(root);media=[];real={};samples={}
    for r in rows:
        dataset,source=r['source_id'].split(':',1);path=base/r['media_path'];rp=real_path(root,dataset,source)
        row={**r,'local_path':str(path),'real_path':str(rp),'real_available':rp.is_file(),**checks[r['media_path']]}
        if row['status']=='VERIFIED':
            try:row['container']=container_info(path)
            except Exception as e:row['container']={'readable':False,'error':str(e)};failures.append({'path':r['media_path'],'status':'CONTAINER_FAILURE','error':str(e)})
        stratum=(dataset,r['generator'])
        if stratum not in samples and row.get('container',{}).get('readable'):
            try:samples[stratum]={'path':str(path),'frames':sample_decode(path)}
            except Exception as e:samples[stratum]={'path':str(path),'error':str(e)};failures.append({'path':r['media_path'],'status':'SAMPLE_DECODE_FAILURE','error':str(e)})
        media.append(row)
        if r['source_id'] not in real:
            realrow={'source_id':r['source_id'],'role':'real','local_path':str(rp),'available':rp.is_file()}
            if rp.is_file():
                realrow.update(bytes=rp.stat().st_size,sha256=digest(rp),official_checksum_type=None)
                try:realrow['container']=container_info(rp)
                except Exception as e:realrow['container']={'readable':False,'error':str(e)}
            real[r['source_id']]=realrow
    with (audit/'media_index.jsonl').open('w') as f:
        for row in media+list(real.values()):f.write(json.dumps(row,ensure_ascii=False)+'\n')
    real_sample_failures=[]
    for row in real.values():
        key=(row['source_id'].split(':',1)[0],'REAL')
        if key not in samples and row.get('container',{}).get('readable'):
            try:samples[key]={'path':row['local_path'],'frames':sample_decode(Path(row['local_path']))}
            except Exception as e:
                samples[key]={'path':row['local_path'],'error':str(e)}
                real_sample_failures.append({'path':row['local_path'],'error':str(e)})
    missing=[r for r in real.values() if not r.get('container',{}).get('readable')]
    save(audit/'missing_real.json',missing);save(audit/'integrity_failures.json',failures)
    train_sources={r['source_id'] for r in rows if r['official_split']=='train'}
    test_sources={r['source_id'] for r in rows if r['official_split']=='test'}
    summary={'official_source_overlap_count':len(train_sources & test_sources),
        'official_source_overlap':sorted(train_sources & test_sources),
        'official_status':'COMPLETE' if not failures else 'PARTIAL','revision':manifest['revision'],
        'official_files_target':len(checks),'official_files_verified':sum(r['status']=='VERIFIED' for r in checks.values()),
        'fake_target':len(rows),'fake_verified':sum(r['status']=='VERIFIED' and r.get('container',{}).get('readable',False) for r in media),
        'split_target':dict(Counter(r['official_split'] for r in rows)),
        'generator_target':dict(Counter(r['generator'] for r in rows)),
        'split_verified':dict(Counter(r['official_split'] for r in media if r['status']=='VERIFIED' and r.get('container',{}).get('readable'))),
        'generator_verified':dict(Counter(r['generator'] for r in media if r['status']=='VERIFIED' and r.get('container',{}).get('readable'))),
        'real_by_source':dict(Counter(r['source_id'].split(':',1)[0] for r in real.values() if r.get('container',{}).get('readable'))),
        'real_target':len(real),'real_readable':len(real)-len(missing),'real_status':'COMPLETE' if not missing and not real_sample_failures else 'PARTIAL',
        'real_sample_failures':real_sample_failures,
        'source_targets':dict(Counter(r.split(':',1)[0] for r in real)),
        'official_verified_bytes':sum(r.get('bytes',0) for r in checks.values() if r['status']=='VERIFIED'),
        'real_bytes':sum(r.get('bytes',0) for r in real.values()),'verification_s':time.time()-started,
        'sample_decodes':{':'.join(k):v for k,v in samples.items()}}
    save(audit/'asset_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['official','charades-download','charades-extract','index']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--revision',help='Official revision on first download; existing manifest always stays pinned')
    a=p.parse_args()
    if a.action=='official':official(a.root,a.revision)
    else:{'charades-download':download_charades,'charades-extract':extract_charades,'index':index}[a.action](a.root)

if __name__=='__main__':main()
