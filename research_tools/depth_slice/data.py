"""Pilot caller: verified frozen ANAKIN observations -> native frames and GT.

No voxel routing or old visual/detector features are read. A separate loader is
needed because PVST.Video unconditionally builds cubic voxels and reads DINO.
"""
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch

from research_tools.pvst.data import native_dense
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import alignment, numeric_members, mask_array


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)


class Video:
    def __init__(self,row,config):
        self.row=row;self.config=config
        folder=Path(config['cache_root'])/row['video_id']
        meta=json.loads((folder/'complete.json').read_text())
        self.rgb=np.load(folder/'rgb.npy',mmap_mode='r')
        n=row['actual_frames'];h,w=row['source_hw']
        if meta['video_sha256']!=row['video_sha256'] or self.rgb.shape!=(n,h,w,3):raise ValueError('RGB_IDENTITY')
        dense=json.loads((Path(row['dense_source'])/'complete.json').read_text())
        if dense['video_sha256']!=row['video_sha256'] or len(dense['frames'])!=n:raise ValueError('DENSE_IDENTITY')
        for i,p in enumerate(row['pts']):
            f=dense['frames'][i]
            if f['frame_index']!=i or abs(f['pts_s']-p['pts_s'])>1e-8:raise ValueError('DENSE_PTS')
        self.zip=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        self.members=numeric_members(self.zip.namelist()) if self.zip else []
        if self.zip and alignment([i for i,_ in self.members],n)!={'status':'aligned','index_base':row['index_base']}:raise ValueError('MASK_ALIGNMENT')
    def close(self):
        if self.zip:self.zip.close()
    def gt(self,i):
        a=mask_array(self.zip.read(self.members[i][1]))[1] if self.zip else np.zeros(self.row['source_hw'],bool)
        if list(a.shape)!=self.row['source_hw']:raise ValueError('MASK_NATIVE_DIMENSION')
        return a
    def frame(self,i):
        xyz,valid=native_dense(self.row,i)
        q=xyz/self.config['scale']
        if not np.isfinite(q[valid]).all():raise ValueError('NONFINITE_VALID_XYZ')
        return {'rgb':torch.from_numpy(np.array(self.rgb[i],copy=True)), 'xyz':torch.from_numpy(q),
                'valid':torch.from_numpy(valid),'gt':torch.from_numpy(self.gt(i)),
                'pts':self.row['pts'][i]['pts_s'],'frame_index':i}


def prepare(config,out,progress):
    source=Path(config['source_manifest']);frozen=json.loads(source.read_text())['rows']
    # Reuse the registered small queue, including its unchanged source roles.
    rows=[r for r in frozen if r['dataset']=='ANAKIN']
    tr={r['source_id'] for r in rows if r['split']=='train'};de={r['source_id'] for r in rows if r['split']=='dev'}
    if tr&de or any(r['split'] not in ['train','dev'] for r in rows):raise ValueError('SOURCE_SPLIT_LEAKAGE')
    stats={};clips=[]
    for r in rows:
        progress('verify_assets',video=r['video_id'])
        if digest(Path(r['video_path']))!=r['video_sha256']:raise ValueError('VIDEO_SHA')
        if r['role']=='fake' and digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SHA')
        v=Video(r,config)
        try:positive=[bool(v.gt(i).any()) for i in range(r['actual_frames'])]
        finally:v.close()
        stats[r['video_id']]={'positive_frames':sum(positive),'frames':len(positive),'all_nonempty':all(positive),'mixed':any(positive) and not all(positive)}
        if r['split']=='train':
            # Three fixed spread clips. No GT or model response used for selection.
            starts=sorted(set(int(x) for x in np.linspace(0,max(0,r['actual_frames']-24),3)))
            clips += [{'video_id':r['video_id'],'start':s,'end':min(s+24,r['actual_frames'])} for s in starts]
    result={'source_manifest_sha256':digest(source),'selection':'all registered ANAKIN train8/dev4 pairs; no new scene filter',
            'rows':rows,'label_statistics':stats,'clips':clips}
    save(out/'manifest.json',result)
    return result
