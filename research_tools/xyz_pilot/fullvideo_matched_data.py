"""ADR-0056 observation preparation for the frozen 89/19 full-video trial.

Caller fullvideo_matched: bounded per-video CPU RGB/GT access, compatible
geometry and adapted DINO observations, immutable routes and disk estimate.
No C/R cache, DFormer extraction, re-selection or generic cache framework.
"""
import csv
import gc
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import av
import cv2
import numpy as np
import torch
from .ainpaint_spatial import build, load_adaptation, DINO, AINPAINT, WEIGHT
from .data import remap_geometry
from .fullvideo_direct_model import prepare_frame, SCALE
from .fullvideo_direct_probe import digest, save, state_hash, seed
from research_tools.rgbd_slice.data import geometry_shape

ROOT = Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
SOURCE = ROOT/'pvst_anakin_expanded_spatial_training_v1'
PROVIDER = ROOT/'pvst_xyz_temporal_pilot_v1/provider_config.json'
PROBE = ROOT/'pvst_fullvideo_direct_prototype_v1/original_environment_run/observations'
OUT = ROOT/'pvst_fullvideo_matched_localization_v1'
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO/'docs/pvst/pvst_fullvideo_matched_config_v1.json'


def read(path):return json.loads(Path(path).read_text())


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def record_call(out,kind,video,frame):
    with (out/'frontend_calls.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,video_id=video,frame_index=frame,time_ns=time.time_ns()))+'\n')


def supervised(row,i):
    return not(row['video_id']=='40y1x9_fake' and 6<=i<10)


def validate_protocol(p):
    rows=p['rows'];groups={};seen=set()
    for r in rows:
        assert r['video_id'] not in seen;seen.add(r['video_id'])
        assert r['split'] in ('train','dev') and r['role'] in ('real','fake')
        assert r['cache_kind'] in ('existing','work_rgb') and r['source_group_id']
        groups.setdefault(r['source_group_id'],set()).add(r['split'])
        assert len(groups[r['source_group_id']])==1,'SOURCE_LEAKAGE'
        n=r['actual_frames'];assert n==len(r['pts']) and n>0
        assert all(b>a for a,b in zip(r['pts'],r['pts'][1:]))
        expected=[i for i in range(n) if supervised(r,i)]
        assert r['train_targets' if r['split']=='train' else 'evaluation_targets']==expected
    assert (sum(r['split']=='train' for r in rows),sum(r['split']=='dev' for r in rows))==(178,38)
    for g in groups:
        members=[r for r in rows if r['source_group_id']==g]
        assert len(members)==2 and {r['role'] for r in members}=={'real','fake'}
    assert len([g for g,s in groups.items() if s=={'train'}])==89
    assert len([g for g,s in groups.items() if s=={'dev'}])==19
    assert sum(len(r['train_targets']) for r in rows)==22878
    assert sum(len(r['evaluation_targets']) for r in rows)==5152
    assert sum(r['actual_frames'] for r in rows)==28034 and p['protected_test_sources']==94


def observation_compatible(row, old, adaptation, provider_sha):
    """Compare input identities, not obsolete enclosing manifest hashes/paths."""
    return all(old.get(k)==v for k,v in dict(video_sha256=row['video_sha256'],
        mask_sha256=row['mask_sha256'],pts=row['pts'],work_hw=row['work_hw'],
        native_hw=row['native_hw'],adaptation_sha256=adaptation,
        provider_config_sha256=provider_sha,scale=SCALE,h=.5,origin=[0,0,0]).items())


def categories():
    # Reuse the finalized CPU closeout table, not the protocol's earlier rough
    # keywords (white shirt/hat are object descriptions, not appearance edits).
    path=SOURCE/'analysis/source_comparison.csv'
    result={}
    for r in csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()):
        value={k:r[k] for k in ['official_action','content_priority','camera_group']}
        assert r['pair_id'] not in result or result[r['pair_id']]==value
        result[r['pair_id']]=value
    assert len(result)==19
    return result


def orders_and_monitor(rows):
    from .expanded_spatial import official_action
    ids=sorted(r['video_id'] for r in rows if r['split']=='train')
    orders=[]
    for epoch in range(1,4):
        order=ids.copy();random.Random(20260909+epoch).shuffle(order);orders.append(order)
    rng=random.Random(20260909);selected=[]
    for action in ['addition_copy','removal']:
        candidates=sorted(r['pair_id'] for r in rows if r['split']=='train' and r['role']=='fake'
                          and official_action(r['official_task'])==action)
        chosen=rng.choice(candidates)
        pair=[r['video_id'] for r in rows if r['pair_id']==chosen and r['split']=='train']
        assert len(pair)==2
        selected.extend(sorted(pair))
    return orders,selected


def plan(out, emit):
    p=read(SOURCE/'protocol.json');validate_protocol(p);c=read(CONFIG)
    for k,v in dict(seed=20260909,epochs=3,updates_per_condition=534,supervised_presentations_per_condition=68634,
                    scale=SCALE,h=.5,D=128,heads=4,layers=2,FFN=256,lr=1e-4,weight_decay=.01,clip_grad=1).items():
        assert c[k]==v,('FIXED_CONFIG',k)
    orders,monitor=orders_and_monitor(p['rows']);cats=categories()
    source_identity=read(SOURCE/'identity.json')
    adapted=digest(SOURCE/'epoch01.pt')
    assert adapted==read(PROBE/'4ym5n4_fake/identity.json')['adaptation_sha256']
    ident=dict(protocol_sha256=digest(SOURCE/'protocol.json'),config=c,
               categories_sha256=digest(SOURCE/'analysis/source_comparison.csv'),
               provider_sha256=digest(PROVIDER),adaptation_sha256=adapted,spatial=source_identity,
               code_sha256={n:digest(REPO/'research_tools/xyz_pilot'/n) for n in
                            ['fullvideo_direct_model.py','fullvideo_matched_data.py','fullvideo_matched.py',
                             'joint_model.py','model.py','ainpaint_spatial.py','data.py']})
    ident['code_sha256']['rgbd_geometry_shape']=digest(REPO/'research_tools/rgbd_slice/data.py')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():assert read(out/'identity.json')==ident,'PROTOCOL_OR_CODE_CHANGED'
    else:
        assert not (out/'FRAME_ONLY').exists() and not (out/'FULL_VIDEO').exists()
        save(out/'identity.json',ident);save(out/'protocol.json',p);save(out/'config.json',c)
        save(out/'orders.json',orders);save(out/'monitor.json',monitor);save(out/'categories.json',cats)
    assert read(out/'orders.json')==orders and read(out/'monitor.json')==monitor
    # Metadata-only conservative upper bound, no queue feature extraction/load.
    visual=geometry=0;reuse_visual=known_geo=0
    for r in p['rows']:
        h,w=r['work_hw'];gh,gw=math.ceil(h/14),math.ceil(w/14)
        visual+=r['actual_frames']*(8192*gh*gw+4*h*w+4096)
        # XYZ12 + valid1 + points/index16 + worst possible XYZ keys24 bytes/pixel.
        geometry+=r['actual_frames']*(53*h*w+4096)
        known_geo+=r['actual_frames'] if r['cache_kind']=='existing' else 0
        ip=PROBE/r['video_id']/'identity.json'
        if ip.exists() and observation_compatible(r,read(ip),adapted,ident['provider_sha256']):
            reuse_visual+=r['actual_frames']
    density={}
    for name in ['4ym5n4_fake','40y1x9_fake','9mygy9_fake']:
        result_path=PROBE.parent/'cases'/name/'result.json'
        if result_path.exists():
            result=read(result_path)
            density[name]=result['frozen_cache_bytes']/result['modes'][0]['T']
    need=int((visual+geometry)*1.15)+2*1024**3
    estimate=dict(kind='metadata conservative estimate, not measured cache size',
        observations=28034,known_existing_geometry_frames=known_geo,
        geometry_not_in_existing_packets=28034-known_geo,compatible_probe_visual_frames=reuse_visual,
        remaining_visual_frames=28034-reuse_visual,upper_visual_bytes=visual,upper_geometry_route_bytes=geometry,
        required_free_bytes_with_margin=need,free_bytes=shutil.disk_usage(out).free,
        excludes_optional_additional_geometry_reuse=True,probe_packet_bytes_per_frame=density,
        note='Geometry upper bound assumes every pixel is its own block; new caches keep actual dtypes. No DFormer calls.')
    save(out/'space_estimate.json',estimate)
    emit('PLAN',train_videos=178,dev_videos=38,updates_per_mode=534,supervised_exposures_per_mode=68634,
         monitor=monitor,**estimate)
    return p,ident


class WorkVideo:
    """One video only; release mmap handles when this instance goes out of scope."""
    def __init__(self,row):
        self.row=row;self.folder=Path(row['cache_root'])/row['video_id']
        self.arrays=None
        if row['cache_kind']=='work_rgb':
            m=read(self.folder/'identity.json')
            for k in ['video_sha256','mask_sha256','pts','work_hw','native_hw']:assert m[k]==row[k]
            self.arrays=(np.load(self.folder/'rgb.npy',mmap_mode='r'),np.load(self.folder/'gt.npy',mmap_mode='r'),m['native_GT_pixels'])
            assert len(self.arrays[0])==len(self.arrays[1])==row['actual_frames']
    def get(self,i):
        r=self.row;assert 0<=i<r['actual_frames']
        if self.arrays:
            rgb,gt,counts=self.arrays
            e=dict(rgb=torch.from_numpy(rgb[i].copy()),gt_work=torch.from_numpy(gt[i].copy()),native_GT_pixels=counts[i])
        else:
            e=torch.load(self.folder/f'{i:05d}.pt',map_location='cpu',weights_only=True)
            assert e['frame_index']==i and e['pts']==r['pts'][i] and e['native_hw']==r['native_hw']
        assert list(e['rgb'].shape[:2])==r['work_hw'] and e['gt_work'].shape==e['rgb'].shape[:2]
        if r['role']=='real':assert not e['gt_work'].any() and e['native_GT_pixels']==0
        return e


def cached_geometry_index(provider, weight_sha):
    result={}
    for root in [ROOT/'pvst_rgbd_slice_relations_v1/geometry',ROOT/'pvst_rgb_partition_anakin_scale_v1/geometry']:
        for path in sorted(root.glob('*/complete.json')):
            m=read(path)
            if m.get('moge_revision')==provider['moge_revision'] and m.get('moge_weights_sha256')==weight_sha:
                result.setdefault(m['video_sha256'],[]).append((path.parent,m))
    return result


def prepare_geometry(out, rows, ident, emit):
    provider=read(PROVIDER)
    assert provider['geometry_long_edge']==768 and provider['geometry_num_tokens']=='round(input_h*input_w/14**2)'
    weight_sha=digest(Path(provider['moge_weights']))
    assert subprocess.check_output(['git','-C',provider['moge_code'],'rev-parse','HEAD'],text=True).strip()==provider['moge_revision']
    index=cached_geometry_index(provider,weight_sha);moge=None;counts=dict(new_MoGe=0,reused_geometry=0,new_DFormer=0)
    start=time.monotonic()
    for vi,r in enumerate(rows):
        folder=out/'geometry'/r['video_id'];folder.mkdir(parents=True,exist_ok=True)
        geo_id={k:r[k] for k in ['video_sha256','pts','native_hw','work_hw']}
        geo_id.update(provider_sha256=ident['provider_sha256'],moge_weights_sha256=weight_sha,scale=SCALE,h=.5)
        if (folder/'identity.json').exists():assert read(folder/'identity.json')==geo_id
        else:save(folder/'identity.json',geo_id)
        missing=[i for i in range(r['actual_frames']) if not (folder/f'{i:05d}.pt').exists()]
        if not missing:continue
        work=WorkVideo(r);gp=None
        for path,m in index.get(r['video_sha256'],[]):
            pts=[x['pts_s'] if 'pts_s' in x else x['seconds'] for x in m['pts']]
            if pts==r['pts'] and m['output_hw']==list(geometry_shape(r['native_hw'])[0]):gp=path;break
        xyz=np.load(gp/'xyz.npy',mmap_mode='r') if gp else None
        valid=np.load(gp/'valid.npy',mmap_mode='r') if gp else None
        def store(i,q,v):
            q,v=remap_geometry(q,v,r['work_hw'])
            f=prepare_frame(dict(xyz_provider=torch.from_numpy(q),valid=torch.from_numpy(v),work_hw=r['work_hw']))
            value={k:f[k] for k in ['xyz_provider','valid','points','keys','index']}
            value.update(frame_index=i,pts=r['pts'][i],scale=SCALE,h=.5)
            atomic_torch(folder/f'{i:05d}.pt',value)
        if r['cache_kind']=='existing' or gp:
            for i in missing:
                if gp:q,v=np.asarray(xyz[i]),np.asarray(valid[i])
                else:
                    e=work.get(i);q=e['xyz_provider'].numpy();v=e['valid'].numpy()
                store(i,q,v);counts['reused_geometry']+=1
        else:
            if moge is None:
                sys.path[:0]=[provider['moge_code'],provider['utils3d_code']]
                from moge.model.v2 import MoGeModel
                moge=MoGeModel.from_pretrained(provider['moge_weights']).cuda().eval().requires_grad_(False)
            wanted=set(missing);n=0
            with av.open(r['video_path']) as container:
                for i,f in enumerate(container.decode(video=0)):
                    assert i<r['actual_frames'] and [f.height,f.width]==r['native_hw']
                    assert abs(float(f.pts*f.time_base)-r['pts'][i])<1e-8;n+=1
                    if i not in wanted:continue
                    rgb=f.to_ndarray(format='rgb24');hw,tokens,_=geometry_shape(r['native_hw'])
                    im=cv2.resize(rgb,(hw[1],hw[0]),interpolation=cv2.INTER_AREA) if rgb.shape[:2]!=hw else rgb
                    with torch.no_grad():
                        res=moge.infer(torch.from_numpy(im).permute(2,0,1).cuda().float()/255,
                                       num_tokens=tokens,force_projection=True,apply_mask=False,use_fp16=True)
                    record_call(out,'MoGe',r['video_id'],i)
                    q=res['points'].float().cpu().numpy();v=res['mask'].cpu().numpy().astype(bool)&np.isfinite(q).all(-1)
                    assert q.shape==(*hw,3);store(i,q,v);counts['new_MoGe']+=1
                    del res
                    if i%20==0:emit('geometry_frame',video=r['video_id'],frame=i+1,T=r['actual_frames'],**counts)
            assert n==r['actual_frames']
        emit('geometry_video',video=r['video_id'],video_position=vi+1,total=len(rows),**counts)
        save(out/'geometry_progress.json',dict(**counts,seconds=time.monotonic()-start))
        del work,xyz,valid
    del moge;gc.collect();torch.cuda.empty_cache()
    save(out/'geometry_cost_'+str(time.time_ns())+'.json',dict(**counts,seconds=time.monotonic()-start))


def prepare_visual(out,rows,ident,emit):
    model=None;counts=dict(new_frozen_visual_forwards=0,reused_visual_frames=0);start=time.monotonic()
    from . import ainpaint_spatial as spatial
    original=spatial.token_grid;captured={};handle=None;before=None
    try:
        for vi,r in enumerate(rows):
            folder=out/'visual'/r['video_id'];folder.mkdir(parents=True,exist_ok=True)
            identity={k:r[k] for k in ['video_sha256','mask_sha256','pts','work_hw','native_hw']}
            identity.update(adaptation_sha256=ident['adaptation_sha256'],preprocessing='SpatialBaseline epoch01 / BF16 eval / pad14')
            if (folder/'identity.json').exists():assert read(folder/'identity.json')==identity
            else:save(folder/'identity.json',identity)
            index_file=folder/'frames.json'
            paths=read(index_file) if index_file.exists() else [None]*r['actual_frames']
            old_id=PROBE/r['video_id']/'identity.json'
            compatible=old_id.exists() and observation_compatible(r,read(old_id),ident['adaptation_sha256'],ident['provider_sha256'])
            work=WorkVideo(r)
            for i in range(r['actual_frames']):
                if paths[i] is not None:
                    assert Path(paths[i]).is_file();counts['reused_visual_frames']+=1;continue
                if compatible and (PROBE/r['video_id']/f'{i:05d}.pt').exists():
                    paths[i]=str(PROBE/r['video_id']/f'{i:05d}.pt');counts['reused_visual_frames']+=1
                else:
                    target=folder/f'{i:05d}.pt'
                    if not target.exists():
                        if model is None:
                            assert digest(SOURCE/'epoch01.pt')==ident['adaptation_sha256']
                            cp=torch.load(SOURCE/'epoch01.pt',map_location='cpu',weights_only=False)
                            assert cp['epoch']==1 and cp['condition']=='EXPANDED_SPATIAL'
                            assert digest(WEIGHT)==cp['identity']['generic_weight_sha256']
                            for repo,key in [(DINO,'official_DINO_revision'),(AINPAINT,'official_AINPAINT_revision')]:
                                assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==cp['identity'][key]
                            model=build();load_adaptation(model,cp['adaptation_state']);del cp
                            model.requires_grad_(False).eval().cuda();before=state_hash(model.state_dict())
                            def grid(result,hw):
                                x=original(result,hw);captured['patch']=x;return x
                            def detail(_m,args):captured['detail']=args[0]
                            spatial.token_grid=grid;handle=model.head.predictor.register_forward_pre_hook(detail)
                        e=work.get(i);h,w=r['work_hw'];pad=[math.ceil(h/14)*14,math.ceil(w/14)*14]
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            z=model([dict(rgb=e['rgb'].cuda())])['pixel']
                        record_call(out,'frozen_DINO',r['video_id'],i)
                        assert captured['patch'].shape==(1,1024,pad[0]//14,pad[1]//14)
                        value=dict(patch=captured['patch'][0].detach().cpu(),detail=captured['detail'][0].detach().cpu(),
                                   z0=z.float().cpu(),dino_padded_hw=pad,work_hw=r['work_hw'],native_hw=r['native_hw'],
                                   frame_index=i,pts=r['pts'][i])
                        assert all(torch.isfinite(value[k]).all() for k in ['patch','detail','z0'])
                        atomic_torch(target,value);counts['new_frozen_visual_forwards']+=1
                        del value,e,z
                    else:counts['reused_visual_frames']+=1
                    paths[i]=str(target)
                save(index_file,paths)
                if i%20==0:emit('visual_frame',video=r['video_id'],frame=i+1,T=r['actual_frames'],**counts)
            emit('visual_video',video=r['video_id'],video_position=vi+1,total=len(rows),**counts)
            del work
        if model is not None:assert before==state_hash(model.state_dict()),'FROZEN_VISUAL_CHANGED'
    finally:
        spatial.token_grid=original
        if handle:handle.remove()
        del model,captured;gc.collect();torch.cuda.empty_cache()
    save(out/('visual_cost_'+str(time.time_ns())+'.json'),dict(**counts,seconds=time.monotonic()-start,frozen_unchanged=True))


class VideoCache:
    def __init__(self,out):self.out=out
    def load(self,r):
        paths=read(self.out/'visual'/r['video_id']/'frames.json');assert len(paths)==r['actual_frames']
        work=WorkVideo(r);frames=[]
        for i,path in enumerate(paths):
            v=torch.load(path,map_location='cpu',weights_only=True)
            assert v['frame_index']==i and v['pts']==r['pts'][i] and v['work_hw']==r['work_hw']
            h,w=r['work_hw'];assert v['dino_padded_hw']==[math.ceil(h/14)*14,math.ceil(w/14)*14]
            g=torch.load(self.out/'geometry'/r['video_id']/f'{i:05d}.pt',map_location='cpu',weights_only=True)
            assert g['frame_index']==i and g['pts']==r['pts'][i] and g['scale']==SCALE and g['h']==.5
            e=work.get(i)
            f={k:v[k] for k in ['patch','detail','z0','dino_padded_hw']}
            f.update({k:g[k] for k in ['points','keys','index']})
            f.update(xyz_normalized=g['xyz_provider'].float()/SCALE,gt_work=e['gt_work'],
                     supervised=supervised(r,i),work_hw=r['work_hw'],native_hw=r['native_hw'],pts=r['pts'][i],frame_index=i,
                     native_GT_pixels=e['native_GT_pixels'])
            assert len(f['points'])==len(f['index'])==int(g['valid'].sum())
            assert tuple(f['z0'].shape)==tuple(r['work_hw'])
            frames.append(f)
        return frames


def prepare_all(out,p,ident,emit):
    if (out/'prepared.json').exists():
        assert read(out/'prepared.json')['identity']==ident
        assert (out/'queue_stats.json').exists()
        emit('reuse_complete_preparation');return
    need=read(out/'space_estimate.json')['required_free_bytes_with_margin']
    if not (out/'prepared.json').exists():
        assert shutil.disk_usage(out).free>need,'DISK_BUDGET: no deletion or precision fallback'
    start=time.monotonic();rows=p['rows']
    prepare_geometry(out,rows,ident,emit);prepare_visual(out,rows,ident,emit)
    stats=[]
    for r in rows:
        counts=[];size=0
        for i in range(r['actual_frames']):
            path=out/'geometry'/r['video_id']/f'{i:05d}.pt'
            g=torch.load(path,map_location='cpu',weights_only=True);counts.append(len(g['keys']));size+=path.stat().st_size
        vp=read(out/'visual'/r['video_id']/'frames.json')
        stats.append(dict(video_id=r['video_id'],pair_id=r['pair_id'],split=r['split'],role=r['role'],
                          T=r['actual_frames'],supervised_frames=sum(supervised(r,i) for i in range(r['actual_frames'])),
                          N_min=min(counts),N_max=max(counts),M=sum(counts),geometry_route_bytes=size,
                          visual_referenced_bytes=sum(Path(v).stat().st_size for v in vp)))
    save(out/'queue_stats.json',stats)
    with (out/'queue_stats.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    save(out/'prepared.json',dict(status='COMPLETE',identity=ident,seconds=time.monotonic()-start,
                                  videos=len(stats),observations=sum(x['T'] for x in stats),new_DFormer=0))
