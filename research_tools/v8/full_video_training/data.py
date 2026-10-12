"""Read one existing compact video at a time, with native frame identity checks."""
import time
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import read_json,sha256_file
from research_tools.v8.full_video_global_context.cache import read_block,FIELDS


def audit(cache):
    status=read_json(cache/'final_status.json');protocol=read_json(cache/'protocol.json')
    rows=read_json(cache/'video_manifest.json')['rows']
    if status['status']!='COMPLETE' or len(rows)!=272 or len({r['video_id'] for r in rows})!=272:
        raise ValueError('CONTINUOUS_CACHE_NOT_COMPLETE')
    manifest=[];total=0
    for row in rows:
        s=read_json(cache/'responses'/row['video_id']/'summary.json')
        if s['identity']!=protocol['identity'] or s['video_sha256']!=row['video_sha256']:
            raise ValueError('VIDEO_CACHE_IDENTITY')
        end=0;blocks=[]
        for b in s['blocks']:
            p=Path(b['input_path']);m=read_json(p.with_suffix('.json'))
            if b['start']!=end or m['identity']!=protocol['identity'] or m['video_sha256']!=row['video_sha256'] or sha256_file(p)!=m['sha256']:
                raise ValueError(f'INPUT_BLOCK_IDENTITY:{row["video_id"]}')
            if [f['frame_index'] for f in m['frames']]!=list(range(b['start'],b['end']+1)):
                raise ValueError('SOURCE_FRAME_SEQUENCE')
            if any(f['video_sha256']!=row['video_sha256'] for f in m['frames']):raise ValueError('FRAME_VIDEO_IDENTITY')
            blocks.append({'path':str(p),'sha256':m['sha256'],'start':b['start'],'end':b['end']});end=b['end']+1
        if end!=s['decoded_frames'] or end!=s['frontend_frames'] or end!=s['response_frames_per_model']:
            raise ValueError('VIDEO_FRAME_COVERAGE')
        manifest.append({**row,'actual_frames':end,'blocks':blocks});total+=end
    train=[r for r in manifest if r['split']=='train'];val=[r for r in manifest if r['split']=='validation']
    if (len(train),len(val),len({r['source_id'] for r in train}),len({r['source_id'] for r in val}),total)!=(244,28,122,14,234284):
        raise ValueError('FROZEN_VIDEO_COUNTS')
    if {r['source_id'] for r in train}&{r['source_id'] for r in val}:raise ValueError('SOURCE_LEAKAGE')
    for source in {r['source_id'] for r in manifest}:
        pair=[r for r in manifest if r['source_id']==source]
        if len(pair)!=2 or {r['role'] for r in pair}!={'real','fake'} or any(r['video_label']!=int(r['role']=='fake') for r in pair):
            raise ValueError('SOURCE_PAIR_OR_LABEL')
    return manifest,protocol


def load_video(row,identity):
    started=time.monotonic();parts={k:[] for k in FIELDS};last=None;count=0
    for b in row['blocks']:
        arrays,metadata=read_block(b['path'],identity,row['video_sha256'])
        for a,m in zip(arrays,metadata):
            if m['frame_index']!=count or (last is not None and m['pts_s']<last): raise ValueError('VIDEO_FRAME_PTS')
            if last is not None and abs(float(a['delta_t'])-(m['pts_s']-last))>1e-6: raise ValueError('ACTUAL_DELTA_T')
            last=m['pts_s'];count+=1
        for k in FIELDS:parts[k].append(np.stack([a[k] for a in arrays]))
    result={k:torch.from_numpy(np.concatenate(v)) for k,v in parts.items()}
    if count!=row['actual_frames']:raise ValueError('VIDEO_LOAD_COUNT')
    return result,time.monotonic()-started
