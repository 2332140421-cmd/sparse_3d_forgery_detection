"""Native PTS decoding and one-frame features; bounded, past-only context."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import av
import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file, write_json_atomic, write_npz_atomic
from research_tools.v8.full_coverage_observation_field.frontend import letterbox, _cell_reduce
from research_tools.v8.local_geometry_pilot.data import cell_mean, components, digest


def patch(a):
    if a.ndim == 3:
        return a.reshape(16,16,16,16,3).transpose(0,2,1,3,4).reshape(256,16,16,3)
    return a.reshape(16,16,16,16).transpose(0,2,1,3).reshape(256,16,16)


def metadata(path):
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        return {'metadata_frame_count': int(s.frames), 'fps': float(s.average_rate) if s.average_rate else None,
                'time_base': str(s.time_base), 'stream_start_time': s.start_time,
                'duration_s': float(s.duration*s.time_base) if s.duration is not None else container.duration/1e6 if container.duration else None,
                'source_hw': [s.height, s.width]}


def decode(path):
    # Decode from start without seek/FPS frame selection; frame indices are decode ordinal.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.thread_type = 'AUTO'; stream.codec_context.thread_count = 2
        origin = float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.
        last = None
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                if not stream.average_rate: raise ValueError('NO_PTS_OR_FPS')
                raw = origin+i/float(stream.average_rate); source='FPS_ESTIMATED'
            else:
                raw=float(frame.pts*frame.time_base); source='NATIVE_PTS'
            seconds=raw-origin
            if last is not None and seconds < last: raise ValueError(f'NONMONOTONIC_PTS:{i}')
            last=seconds
            yield i, frame.to_ndarray(format='bgr24'), {'frame_index': i, 'pts_ticks': frame.pts,
                    'time_base': str(frame.time_base), 'absolute_pts_s': raw, 'pts_s': seconds,
                    'time_origin_s': origin, 'timestamp_source': source}


def video_manifest(base):
    index={}
    for r in read_json(base/'data_manifest.json')['rows']:
        key=(r['pair_id'], r['source_id'], r['role'])
        fields={k:r[k] for k in ('pair_id','source_id','role','video_path','video_sha256','split')}
        fields['video_label']=0 if r['role']=='real' else 1
        fields['label_source']='role/video-level dataset identity; no frame truth'
        if key in index and any(index[key][k]!=fields[k] for k in fields):
            raise ValueError(f'VIDEO_IDENTITY_CONFLICT:{key}')
        index[key]=fields
    rows=[]
    for key, row in sorted(index.items()):
        row['video_id']=digest({k:row[k] for k in ('pair_id','source_id','role','video_sha256')})[:24]
        row.update(metadata(row['video_path']))
        row.update(policy='ALL_NORMAL_DECODE_FRAMES_NO_DOWNSAMPLING',processed=False,
                   actual_decoded_frame_count=None,timestamp_source='pending native decode')
        rows.append(row)
    candidates=[]
    for pair, source in sorted({(r['pair_id'],r['source_id']) for r in rows if r['split']=='train'}, key=lambda x:(x[1],x[0])):
        both=[r for r in rows if (r['pair_id'],r['source_id'])==(pair,source) and r['split']=='train']
        if {r['role'] for r in both}=={'real','fake'} and all(r['metadata_frame_count']>=63 for r in both):
            candidates=sorted(both,key=lambda r:r['role']); break
    if len(candidates)!=2: raise ValueError('NO_LONG_TRAIN_PAIR')
    return rows,candidates


def tracker_summary(result, context_indices, previous_active, active):
    tracks,vis=result['tracks'], result['visibility']
    real=len(context_indices); pad=result['endpoint_padding_count']
    timeline=[context_indices[0]]*pad+context_indices
    if tracks.shape!=(16,1024,2) or vis.shape!=(16,1024) or len(timeline)!=16:
        raise ValueError('TRACKER_OUTPUT_TIME_CONTRACT')
    now=pad+real-1; prior=now-1 if real>1 else None
    if now!=result['query_frame_in_padded_prefix']: raise ValueError('QUERY_OUTPUT_TIME_MISMATCH')
    # Explicit time axis: official predictor init performs no inference, next call writes [0,T).
    q=np.arange(1024); cell=(q//32//2)*16+(q%32//2)
    uv=tracks[now]; nv=vis[now]&np.isfinite(uv).all(-1)
    cov=np.bincount(cell,weights=nv.astype(float),minlength=256).astype(np.float32)/4
    motion=np.zeros((256,2),np.float32); mv=np.zeros(256,bool)
    pred=np.zeros((256,4),np.int64); aw=np.zeros((256,4),np.float32)
    if prior is not None:
        pv=vis[prior]&np.isfinite(tracks[prior]).all(-1); both=nv&pv
        np.add.at(motion,cell[both],(uv[both]-tracks[prior,both])/4)
        mv[np.unique(cell[both])]=True
        xy=np.floor(np.where(pv[:,None],tracks[prior],0)/16).astype(np.int64)
        inside=both&(xy[:,0]>=0)&(xy[:,0]<16)&(xy[:,1]>=0)&(xy[:,1]<16)
        counts=np.zeros((256,256),np.int16)
        np.add.at(counts,(cell[inside],xy[inside,1]*16+xy[inside,0]),1)
        if previous_active is None: raise ValueError('PREVIOUS_CONTENT_MISSING')
        counts[:,~previous_active]=0; counts[~active]=0
        for ci in range(256):
            candidates=np.flatnonzero(counts[ci]); order=sorted(candidates,key=lambda j:(-int(counts[ci,j]),int(j)))[:4]
            if order:
                pred[ci,:len(order)]=order; aw[ci,:len(order)]=counts[ci,order]/counts[ci,order].sum()
    return cov,motion,mv,pred,aw,{'input_frame_indices':context_indices,'output_axis_frame_indices':timeline,
                               'current_output_index':now,'previous_output_index':prior,
                               'endpoint_padding_count':pad,'query_output_index':now}


class Frontend:
    def __init__(self, base, out, moge, tracker, rgb, statistics, grouping, config_id):
        self.base,self.out,self.moge,self.tracker,self.rgb=base,out,moge,tracker,rgb
        self.stats,self.grouping,self.config_id=statistics,grouping,config_id
        self.cost={'geometry_s':0.,'rgb_s':0.,'tracking_s':0.,'features_s':0.,'write_s':0.,'cache_hit':0,'new_geometry':0,'cache_rejected':0}

    def clock(self, name, fn):
        torch.cuda.synchronize(); start=time.monotonic()
        # Preserve the frozen frontend's original cuDNN runtime; only the streaming detector
        # uses TF32-disabled equivalence settings. Context restores detector flags afterwards.
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            result=fn()
        torch.cuda.synchronize()
        self.cost[name]+=time.monotonic()-start
        return result

    def build(self, row, bgr, frame_info, context, previous_active):
        fi=frame_info['frame_index']; canvas,mapping=letterbox(bgr)
        content=np.zeros((256,256),bool); left,top=mapping['pad_xy']; h,w=mapping['content_hw']; content[top:top+h,left:left+w]=True
        key=hashlib.sha256(f"{row['video_sha256']}:{fi}:v8-canvas256".encode()).hexdigest()[:24]
        old=self.base/'raw_frontend'/'frames'/f'{key}.npz'
        new=self.out/'frames'/row['video_id']/f'{fi:06d}.npz'; new.parent.mkdir(parents=True,exist_ok=True)
        points=valid=None; origin=None
        if new.exists():
            meta=read_json(new.with_suffix('.json'))
            if meta['config_id']!=self.config_id or meta['video_sha256']!=row['video_sha256'] or meta['frame_index']!=fi: raise ValueError('NEW_FRAME_IDENTITY')
            with np.load(new,allow_pickle=False) as z:
                if not np.array_equal(z['rgb'],canvas): raise ValueError('NEW_CACHE_DECODE_MISMATCH')
                points,valid,rgb_map=z['points_raw'],z['validity'],z['rgb_map']
            self.cost['cache_hit']+=1; origin=str(new)
        else:
            if old.exists() and old.with_suffix('.json').exists():
                om=read_json(old.with_suffix('.json'))
                if om['video_sha256']!=row['video_sha256'] or om['frame_index']!=fi or om.get('provider')!='MoGe-2-v2': raise ValueError('LEGACY_FRAME_IDENTITY')
                with np.load(old,allow_pickle=False) as z:
                    if np.array_equal(z['rgb'],canvas) and om['mapping']==mapping:
                        points,valid=z['points_raw'].copy(),z['validity'].copy(); origin=str(old)
                        depth=z['depth_raw'].copy(); self.cost['cache_hit']+=1
                    else: self.cost['cache_rejected']+=1
            if points is None:
                r=self.clock('geometry_s',lambda:self.moge.infer(canvas))
                points,valid,depth=r['points_raw'],r['validity'],r['depth_raw'];self.cost['new_geometry']+=1
            rgb_map=self.clock('rgb_s',lambda:self.rgb.encode(canvas[None,:,:,::-1].copy()))[0]
            ts=time.monotonic()
            write_npz_atomic(new,rgb=canvas,points_raw=points,validity=valid,depth_raw=depth,rgb_map=rgb_map)
            write_json_atomic(new.with_suffix('.json'),{**frame_info,'video_sha256':row['video_sha256'],'mapping':mapping,
                                'config_id':self.config_id,'old_geometry_source':origin,'old_geometry_sha256':sha256_file(origin) if origin else None,
                                'canvas_sha256':hashlib.sha256(canvas.tobytes()).hexdigest(),'npz_sha256':sha256_file(new)})
            self.cost['write_s']+=time.monotonic()-ts
        prefix=(context+[(fi,canvas[:,:,::-1].copy())])[-3:]
        tr=self.clock('tracking_s',lambda:self.tracker.query([x[1] for x in prefix],len(prefix)-1))
        ts=time.monotonic(); cp=patch(content); area=cp.sum((1,2)).astype(np.float32); active=area>0
        xyz=patch(points)[None]; mu,gv,vp=cell_mean(xyz,patch(valid)[None],cp[None]); mu,gv,vp=mu[0],gv[0],vp[0]
        cov,motion,mv,pred,aw,time_meta=tracker_summary(tr,[x[0] for x in prefix],previous_active,active)
        hist=np.full(256,fi>0,bool)
        rgb_cells,_=_cell_reduce(canvas[:,:,::-1].astype(np.float32)/255)
        x=np.concatenate((rgb_map,rgb_cells,mu,gv[:,None],cov[:,None],motion,mv[:,None],hist[:,None],(~gv)[:,None]),-1).astype(np.float32)
        x=((x-np.asarray(self.stats['cell_mean'],np.float32))/np.asarray(self.stats['cell_scale'],np.float32)).astype(np.float32); x[~active]=0
        comp=components(mu,gv,motion,mv,active,self.grouping)
        dt=0. if not context else frame_info['pts_s']-context[-1][2]
        # Context images carry true native PTS, never synthetic endpoint timestamps.
        arr={'x':x,'xyz':np.where(vp[...,None],xyz[0],0.).astype(np.float32),'valid':vp,'content':cp,'mu':mu,
             'area':area,'component_ids':comp,'predecessor':pred,'association_weight':aw,'history_available':hist,'delta_t':np.array(dt,np.float32)}
        if not all(np.isfinite(a).all() for a in arr.values()): raise ValueError('NONFINITE_FRAME_INPUT')
        self.cost['features_s']+=time.monotonic()-ts
        info={**frame_info,'mapping':mapping,'tracker_time':time_meta,'geometry_source':origin or str(new),
              'component_count':len(np.unique(comp[active])),'active_cells':int(active.sum()),'content_area':int(area.sum()),
              'geometry_valid_content_pixels':int(vp.sum()),'no_history_cells':int((active&(aw.sum(-1)==0)).sum())}
        return arr,info,(fi,canvas[:,:,::-1].copy(),frame_info['pts_s']),tr
