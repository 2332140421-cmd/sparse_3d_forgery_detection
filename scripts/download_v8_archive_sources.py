"""Fetch exact AF-related original videos from pinned public mirror ZIPs.

HC-STVG v1 OneDrive is inaccessible; ActivityNet has missing YouTube videos.
This CLI outputs native containers
and mirror-member CRC/SHA provenance, never claims official-original SHA match.
Only a segmented remote ZIP reader is needed; downloading all 115GB is unnecessary.
"""
from __future__ import annotations
import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import zlib

import requests

from scripts.prepare_v8_assets import digest, save


class SegmentedZip(io.RawIOBase):
    def __init__(self, metadata, paths=None):
        self.parts=([next(x for x in metadata['files'] if x['path']==name) for name in paths] if paths else
                    sorted((x for x in metadata['files'] if x['path'].startswith('video_parts_part_')),key=lambda x:x['path']))
        self.offsets=[0]
        for part in self.parts:self.offsets.append(self.offsets[-1]+part['size'])
        endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
        self.urls=[f"{endpoint}/datasets/{metadata['repo']}/resolve/{metadata['revision']}/{x['path']}" for x in self.parts]
        self.orig_urls=self.urls.copy()
        self.pos=0;self.cache_start=0;self.cache=b'';self.session=requests.Session()

    def seek(self, offset, whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.offsets[-1]+offset
        return self.pos

    def tell(self):return self.pos

    def fetch(self, offset, count):
        result=[]
        while count>0 and offset<self.offsets[-1]:
            i=bisect.bisect_right(self.offsets,offset)-1;local=offset-self.offsets[i];size=min(count,self.offsets[i+1]-offset)
            for attempt in range(5):
                try:
                    r=self.session.get(self.urls[i],headers={'Range':f'bytes={local}-{local+size-1}'},timeout=(20,90))
                    r.raise_for_status()
                    if r.status_code!=206 or len(r.content)!=size:raise ValueError('EXACT_HTTP_RANGE_REQUIRED')
                    # Keep signed redirects in memory only; never write them to provenance/logs.
                    self.urls[i]=r.url;result.append(r.content);break
                except requests.RequestException:
                    self.urls[i]=self.orig_urls[i]  # Refresh an expired public signed redirect on retry.
                    if attempt==4:raise
                    time.sleep(2)
            offset+=size;count-=size
        return b''.join(result)

    def read(self, size=-1):
        size=self.offsets[-1]-self.pos if size<0 else min(size,self.offsets[-1]-self.pos)
        if self.cache_start<=self.pos and self.pos+size<=self.cache_start+len(self.cache):
            start=self.pos-self.cache_start;data=self.cache[start:start+size]
        else:data=self.fetch(self.pos,size)
        self.pos+=len(data);return data

    def prefetch(self, entry):
        # One bounded member transfer per video; zipfile handles decompression and CRC.
        self.cache_start=entry.header_offset
        self.cache=self.fetch(entry.header_offset,entry.compress_size+len(entry.filename.encode())+len(entry.extra)+1054)


def crc_file(path):
    crc=0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):crc=zlib.crc32(block,crc)
    return crc


def download(root, workers=2, dataset="hcv1"):
    audit=root/'asset_preparation';refs=audit/'references'
    meta_name,members_name,status_name=(('CircleRadon_hc-stvg-files.json','hc-mirror-zip-members.json','hc_download_status.json')
        if dataset=='hcv1' else ('YimuWang_ActivityNet-files.json','activitynet-archive-targets.json','activitynet_archive_status.json'))
    meta=json.loads((refs/meta_name).read_text());membership=json.loads((refs/members_name).read_text())
    archives=sorted({r['archive'] for r in membership['matches'].values()}) if dataset=='acnet' else None
    if archives is not None and len(archives)!=1:raise ValueError('EXPECTED_ONE_ACTIVITYNET_RECOVERY_ZIP')
    if membership['revision']!=meta['revision'] or (dataset=='hcv1' and membership['unmatched_sources']):raise ValueError('INCOMPLETE_OR_UNPINNED_MEMBERSHIP')
    items=sorted(membership['matches'].items());started=time.time()
    save(audit/status_name,{'status':'RUNNING','started_unix':started,'target':len(items),'mirror':meta['repo'],'revision':meta['revision']})

    def worker(chunk, number):
        result=[];reader=SegmentedZip(meta,archives)
        with zipfile.ZipFile(reader) as archive,(audit/f'{dataset}-archive-worker-{number}.jsonl').open('a') as log:
            for source,record in chunk:
                entry=archive.getinfo(record['member']);out=root/'real'/dataset/(source+Path(entry.filename).suffix)
                if Path(entry.filename).stem!=source or entry.file_size!=record['bytes'] or entry.CRC!=record['crc']:raise ValueError('MIRROR_MEMBER_IDENTITY_CHANGED')
                row={'source_id':dataset+':'+source,'mirror':meta['repo'],'revision':meta['revision'],'member':entry.filename,
                     'local_path':str(out),'official_original_sha_verified':False,'member_crc':entry.CRC}
                try:
                    out.parent.mkdir(parents=True,exist_ok=True)
                    if out.exists():
                        if out.stat().st_size!=entry.file_size or crc_file(out)!=entry.CRC:raise ValueError('EXISTING_HC_SOURCE_MISMATCH')
                    else:
                        reader.prefetch(entry);tmp=out.with_suffix(out.suffix+'.part')
                        with archive.open(entry) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
                        tmp.replace(out)
                    row.update(status='DOWNLOADED_CRC_VERIFIED',bytes=out.stat().st_size,sha256=digest(out))
                    save(out.with_suffix(out.suffix+'.json'),row)
                except Exception as e:row.update(status='FAILED',error_type=type(e).__name__)
                log.write(json.dumps(row)+'\n');log.flush();result.append(row)
        reader.session.close();return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,items[i::workers],i) for i in range(workers)]
        results=[r for f in futures for r in f.result()]
    failed=[r for r in results if r['status']=='FAILED']
    save(audit/status_name,{'status':'PARTIAL' if failed else 'COMPLETE_MIRROR_MEMBERS',
        'target':len(items),'downloaded':len(results)-len(failed),'started_unix':started,'download_s':time.time()-started,
        'mirror':meta['repo'],'revision':meta['revision'],'official_original_sha_verified':False,'not_in_mirror':membership['unmatched_sources'],'failures':failed})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--dataset',choices=['hcv1','acnet'],default='hcv1')
    a=p.parse_args()
    try:download(a.root,a.workers,a.dataset)
    except Exception as e:
        status=a.root/'asset_preparation'/('hc_download_status.json' if a.dataset=='hcv1' else 'activitynet_archive_status.json')
        data=json.loads(status.read_text()) if status.exists() else {}
        data.update(status='PARTIAL',error_type=type(e).__name__)
        save(status,data)
        raise SystemExit(type(e).__name__)

if __name__=='__main__':main()
