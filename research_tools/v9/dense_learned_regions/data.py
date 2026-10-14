"""Frozen native-PTS clips and video-block dense observations (no tracking)."""
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from research_tools.v8.full_coverage_observation_field.common import sha256_file, read_json, write_json_atomic
from research_tools.v8.full_coverage_observation_field.frontend import letterbox, MogeV2
from research_tools.v8.full_video_streaming.inputs import decode
from research_tools.v8.full_video_training.temporal import exact_annotation, interval_labels, MAPPING

ROOT = Path('/root/autodl-tmp/data/sparse_3d_forgery_detection')
OLD = ROOT / 'derived/v8_full_video_training_budget5_v1/input_manifest.json'
OUT = ROOT / 'derived/v9_dense_learned_regions_pilot_v1'
MOGE = ROOT / 'external/v8_full_coverage/checkpoints/moge-2-vits-normal-model.pt'
RESNET = Path('/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')


def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()
def save(path, x): write_json_atomic(Path(path), x)


def runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices): return []
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def choose_clips(y):
    pos = sorted(runs(y), key=lambda a: (-len(a), int(a[0])))
    neg = sorted(runs(~y), key=lambda a: (-len(a), int(a[0])))
    result = []
    for fraction in (.25, .75):
        if not pos: raise ValueError('NO_POSITIVE_INTERVAL')
        a = pos[0]; length = min(32, len(a)); start = int(a[0] + round((len(a) - length) * fraction))
        result.append({'kind':f'inside_{fraction}', 'indices':list(range(start, start+length))})
    if neg:
        a = neg[0]; length = min(32, len(a)); start = int(a[0] + (len(a)-length)//2)
        result.append({'kind':'outside', 'indices':list(range(start, start+length))})
    boundaries = np.flatnonzero(y[1:] != y[:-1]) + 1
    if len(boundaries):
        start = max(0, min(int(boundaries[0])-16, len(y)-32)); end = min(start+32, len(y))
        result.append({'kind':'boundary', 'indices':list(range(start, end))})
    return result


def prepare(out):
    old = read_json(OLD); rows = old['rows']; mapping = read_json(MAPPING)
    if len(rows) != 272: raise ValueError('VIDEO_COUNT')
    prepared = []
    for r in rows:
        frames = []
        for b in r['blocks']:
            meta = read_json(Path(b['path']).with_suffix('.json'))
            if meta['video_sha256'] != r['video_sha256']: raise ValueError('CACHE_VIDEO_IDENTITY')
            frames.extend(meta['frames'])
        pts = np.array([f['pts_s'] for f in frames])
        if [f['frame_index'] for f in frames] != list(range(r['actual_frames'])) or np.any(np.diff(pts)<0): raise ValueError('FRAME_AXIS')
        if any(f['timestamp_source']!='NATIVE_PTS' or abs(f['absolute_pts_s']-f['time_origin_s']-f['pts_s'])>1e-8 or f['time_origin_s']!=0 for f in frames): raise ValueError('PTS_ORIGIN')
        annotation = exact_annotation(r, mapping)
        intervals = annotation['manipulation_intervals'] if annotation else []
        labels = interval_labels(pts, intervals)
        p = {k:r[k] for k in ('video_id','source_id','role','split','video_path','video_sha256','actual_frames')}
        p.update(pts_s=pts.tolist(), labels=labels.astype(int).tolist(), intervals=intervals,
                 annotation_evidence=annotation['lineage_evidence'] if annotation else [], clips=[],
                 geometry_sources=[f['geometry_source'] for f in frames], mappings=[f['mapping'] for f in frames])
        prepared.append(p)
    train = [r for r in prepared if r['split']=='train']; val = [r for r in prepared if r['split']=='validation']
    if (len(train),len(val),sum(r['actual_frames'] for r in train),sum(r['actual_frames'] for r in val)) != (244,28,206779,27505): raise ValueError('FROZEN_SPLIT')
    if {r['source_id'] for r in train}&{r['source_id'] for r in val}: raise ValueError('SOURCE_LEAK')
    for f in [r for r in train if r['role']=='fake']:
        real = next(r for r in train if r['source_id']==f['source_id'] and r['role']=='real')
        f['clips'] = choose_clips(np.array(f['labels'],bool))
        for c in f['clips']:
            start = int(np.abs(np.array(real['pts_s'])-f['pts_s'][c['indices'][0]]).argmin())
            length = min(len(c['indices']),real['actual_frames']); start=min(start,real['actual_frames']-length)
            real['clips'].append({'kind':c['kind'],'indices':list(range(start,start+length)), 'matched_fake_start_pts':f['pts_s'][c['indices'][0]]})
    for r in prepared:
        r['selected_indices'] = sorted({i for c in r['clips'] for i in c['indices']}) if r['split']=='train' else list(range(r['actual_frames']))
        for c in r['clips']: c['clip_label']=int(any(r['labels'][i] for i in c['indices']))
        if r['split']=='validation':
            r['diagnostic_indices'] = [int(np.abs(np.array(r['pts_s'])-r['pts_s'][-1]*fraction).argmin()) for fraction in (.25,.5,.75)] if r['source_id'] in sorted({v['source_id'] for v in val})[:3] else []
    protocol = {'version':'v9-dense-learned-regions-v1','seed':20260909,'epochs':20,'conditions':['RGB_QUERY','XYZ_QUERY','XYZ_PARTITION'],
        'source_manifest_sha256':sha256_file(OLD),'annotation_mapping_sha256':sha256_file(MAPPING),
        'selection':'longest positive run quarter/three-quarter valid-start positions; longest negative run middle; first chronological boundary +/-16; short runs retained; absent outside/boundary omitted with count',
        'real_selection':'nearest actual PTS to fake clip start; same length, clamp to valid real start',
        'endpoint':'closed union [start,end], native video-relative PTS','canvas':[256,256], 'rgb_cache':'RGB (old V8 rgb is BGR, converted once)',
        'moge_sha256':sha256_file(MOGE),'resnet_sha256':sha256_file(RESNET),'geometry':'MoGe2 vits normal,1200tokens,apply_mask=False,use_fp16=True; no tracking',
        'model':'128 visual +128 local +RGB3+UV2 ->256;64 queries; QKV256;4head region attention;GRU257->256;pixel attention decoder and512->128->1 head',
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001,'gradient_clip':1},
        'loss':'clip BCE mean per parent +0.5 globally class-balanced frame BCE per parent; parent real/fake half; step per source pair',
        'precision':'CUDA bfloat16 autocast, FP32 parameters and losses','normalization':'train selected unique frames, valid content/edge pooled moments; raw differences before standardization',
        'aggregation':'beta1 normalized content-area LSE pixel->frame->clip/video', 'threshold':0,'bootstrap':{'seed':20260909,'replicates':10000,'cluster':'source real/fake pair'},
        'coverage':{'clips':sum(len(r['clips']) for r in train),'training_frame_uses':sum(len(c['indices']) for r in train for c in r['clips']),
            'training_unique_frames':sum(len(r['selected_indices']) for r in train),'validation_frames':27505,
            'short_clips':sum(len(c['indices'])<32 for r in train for c in r['clips']),
            'missing_clip_types':976-sum(len(r['clips']) for r in train)},
        'manifest_hash':digest(prepared)}
    for name,obj in [('protocol.json',protocol),('video_manifest.json',{'rows':prepared})]:
        if (out/name).exists() and read_json(out/name)!=obj: raise ValueError('FROZEN_PLAN_CHANGED')
        save(out/name,obj)
    return prepared,protocol


@lru_cache(maxsize=1)
def backbone():
    from torchvision.models import resnet18
    net=resnet18(weights=None);net.load_state_dict(torch.load(RESNET,map_location='cpu',weights_only=True))
    net=torch.nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1).cuda().eval()
    net.requires_grad_(False)
    return net


def gpu_batch(cpu):
    batch={k:v.cuda() for k,v in cpu.items()}
    with torch.no_grad(), torch.autocast('cuda',enabled=False):
        x=torch.nn.functional.interpolate(batch['rgb'],size=(224,224),mode='bilinear',align_corners=False)
        x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        batch['visual']=backbone()(x)
    return batch


def frontend(out, rows, progress):
    costs=read_json(out/'frontend_costs.json') if (out/'frontend_costs.json').exists() else {'new_moge_frames':0,'reused_moge_frames':0,'elapsed_s':0.,'blocks':0}
    moge=None
    moge_hash=sha256_file(MOGE);resnet_hash=sha256_file(RESNET)
    old_protocol=read_json(ROOT/'derived/v8_full_video_global_context_v1/protocol.json')
    if old_protocol['providers_sha']['moge']!=moge_hash:raise ValueError('OLD_GEOMETRY_PROVIDER_CHANGED')
    for n,r in enumerate(rows):
        target=out/'dense'/r['video_id'];target.mkdir(parents=True,exist_ok=True)
        selected=r['selected_indices']; blocks=[selected[i:i+32] for i in range(0,len(selected),32)]
        missing=[]
        for bi,indices in enumerate(blocks):
            p=target/f'{bi:06d}.npz';meta=p.with_suffix('.json')
            expected={'video_sha256':r['video_sha256'],'indices':indices,'moge_sha256':moge_hash,'resnet_sha256':resnet_hash}
            if p.exists() and meta.exists():
                m=read_json(meta)
                if m['identity']!=expected or m['sha256']!=sha256_file(p):raise ValueError('DENSE_BLOCK_IDENTITY')
            else: missing.append((bi,indices,expected))
        if not missing: progress('frontend',n+1,len(rows),video_id=r['video_id']);continue
        if sha256_file(r['video_path'])!=r['video_sha256']:raise ValueError('SOURCE_SHA')
        needed={i for _,indices,_ in missing for i in indices};buffer={};started=time.monotonic()
        for fi,bgr,info in decode(r['video_path']):
            if fi not in needed:continue
            if abs(info['pts_s']-r['pts_s'][fi])>1e-8:raise ValueError('DECODE_NATIVE_PTS')
            canvas,mapping=letterbox(bgr)
            if mapping!=r['mappings'][fi]:raise ValueError('LETTERBOX_MAPPING')
            content=np.zeros((256,256),bool);left,top=mapping['pad_xy'];h,w=mapping['content_hw'];content[top:top+h,left:left+w]=True
            old=Path(r['geometry_sources'][fi]);points=valid=None
            if old.exists() and old.with_suffix('.json').exists():
                m=read_json(old.with_suffix('.json'))
                if m['video_sha256']!=r['video_sha256'] or m['frame_index']!=fi or m['mapping']!=mapping:raise ValueError('OLD_DENSE_IDENTITY')
                # Only content-hashed streaming records with matching provider config.
                if m.get('config_id')==old_protocol['legacy_identity'] and m.get('npz_sha256')==sha256_file(old):
                    with np.load(old,allow_pickle=False) as z:
                        if not np.array_equal(z['rgb'],canvas):raise ValueError('OLD_DENSE_CANVAS')
                        points=z['points_raw'].copy();valid=z['validity'].copy()
                    costs['reused_moge_frames']+=1
            if points is None:
                if moge is None:moge=MogeV2(str(MOGE))
                pred=moge.infer(canvas);points=pred['points_raw'];valid=pred['validity'];costs['new_moge_frames']+=1
            if points.shape!=(256,256,3):raise ValueError('NOT_DENSE_XYZ')
            rgb=canvas[:,:,::-1].copy();valid=valid&content&np.isfinite(points).all(-1)
            buffer[fi]=(rgb,np.where(valid[...,None],points,0).astype(np.float32),valid,content)
            for bi,indices,identity in list(missing):
                if all(i in buffer for i in indices):
                    p=target/f'{bi:06d}.npz';items=[buffer.pop(i) for i in indices]
                    with p.with_suffix('.tmp').open('wb') as f:
                        np.savez(f,rgb=np.stack([a[0] for a in items]),xyz=np.stack([a[1] for a in items]),
                            valid=np.packbits(np.stack([a[2] for a in items]),axis=-1),content=np.packbits(np.stack([a[3] for a in items]),axis=-1),indices=np.array(indices,np.int64))
                    p.with_suffix('.tmp').replace(p);save(p.with_suffix('.json'),{'identity':identity,'sha256':sha256_file(p)})
                    costs['blocks']+=1;costs['elapsed_s']+=time.monotonic()-started;started=time.monotonic();save(out/'frontend_costs.json',costs)
                    missing.remove((bi,indices,identity));progress('frontend',n,len(rows),video_id=r['video_id'],last_frame=fi,remaining_blocks=len(missing))
            if not missing:break
        if missing:raise ValueError('MISSING_DECODE_FRAMES')
        progress('frontend',n+1,len(rows),video_id=r['video_id'])
    del moge;torch.cuda.empty_cache()


def read_frames(out, row, indices, geometry=True):
    lookup={v:i for i,v in enumerate(row['selected_indices'])};parts={}
    for i in indices: parts.setdefault(lookup[i]//32,[]).append((i,lookup[i]%32))
    result={}
    for bi,pairs in parts.items():
        with np.load(out/'dense'/row['video_id']/f'{bi:06d}.npz',allow_pickle=False) as z:
            keys=['rgb','content']+(['xyz','valid'] if geometry else [])
            for key in keys:
                a=z[key]
                if key in ('valid','content'):a=np.unpackbits(a,axis=-1).astype(bool)
                for i,j in pairs:result.setdefault(i,{})[key]=a[j].copy()
    batch={k:torch.from_numpy(np.stack([result[i][k] for i in indices])) for k in next(iter(result.values()))}
    for k in ('rgb','xyz'):
        if k in batch:batch[k]=batch[k].permute(0,3,1,2).float()
    batch['rgb']/=255
    for k in ('valid','content'):
        if k in batch:batch[k]=batch[k][:,None]
    pts=np.array([row['pts_s'][i] for i in indices]);batch['dt']=torch.tensor(np.r_[0,np.diff(pts)],dtype=torch.float32)
    return batch


def standardize(out,rows):
    from .model import relations
    sums=np.zeros(9);squares=np.zeros(9);counts=np.zeros(9)
    for r in rows:
        if r['split']!='train':continue
        for offset in range(0,len(r['selected_indices']),32):
            b=read_frames(out,r,r['selected_indices'][offset:offset+32]);nums,masks=relations(b['xyz'],b['valid']&b['content'])
            masks=masks.repeat_interleave(3,1)
            a=nums.numpy().astype(np.float64);m=masks.numpy()
            sums+=(a*m).sum((0,2,3));squares+=(a*a*m).sum((0,2,3));counts+=m.sum((0,2,3))
    if (counts==0).any():raise ValueError('NO_TRAIN_GEOMETRY')
    mean=sums/counts;scale=np.sqrt(np.maximum(squares/counts-mean**2,1e-12))
    obj={'mean':mean.tolist(),'scale':scale.tolist(),'counts':counts.tolist(),'split':'train','unique_frames_only':True}
    save(out/'standardizer.json',obj);return obj
