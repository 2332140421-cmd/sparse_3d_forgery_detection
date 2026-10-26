"""This pilot's exact queue and per-video mmap inputs; no provider factories."""
import csv
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import requests

from scripts.prepare_v8_assets import digest, identities, real_path, verify
from scripts.review_anakin import alignment, mask_array, numeric_members
from research_tools.v8.full_coverage_observation_field.common import cell_neighbors, write_json_atomic
from research_tools.v8.full_video_streaming.inputs import Frontend, decode, patch
from research_tools.v8.local_geometry_pilot.data import components
from .data import content_mask, mask_to_canvas

SEED=20260909
REV='978fe26e78ecaaf6562aa249be43c2c690484d0a'
SCHEMA='v8-joint-spatial-v1'


def save(path,value):write_json_atomic(Path(path),value)


def mapping(hw,size):
    h,w=hw;s=min(size/h,size/w);nh,nw=max(1,round(h*s)),max(1,round(w*s))
    return {'source_hw':[h,w],'canvas_hw':[size,size],'content_hw':[nh,nw],
            'pad_xy':[(size-nw)//2,(size-nh)//2],'scale':s}


def canvas(bgr,m):
    result=np.zeros((*m['canvas_hw'],3),np.uint8);h,w=m['content_hw'];x,y=m['pad_xy']
    result[y:y+h,x:x+w]=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_AREA)
    return result


def scan(path):
    pts=[];hw=None
    with av.open(str(path)) as c:
        s=c.streams.video[0];s.thread_count=2;origin=float((s.start_time or 0)*s.time_base)
        for f in c.decode(s):
            if f.pts is None:raise ValueError('NATIVE_PTS_REQUIRED')
            size=[f.height,f.width]
            if hw is not None and size!=hw:raise ValueError('VARIABLE_FRAME_SIZE')
            hw=size;t=float(f.pts*f.time_base)-origin
            if pts and t<=pts[-1]['pts_s']:raise ValueError('STRICT_PTS_REQUIRED')
            pts.append({'pts_ticks':f.pts,'time_base':str(f.time_base),'pts_s':t})
    if not pts:raise ValueError('NO_FRAMES')
    return {'actual_frames':len(pts),'source_hw':hw,'pts':pts,'mapping256':mapping(hw,256),'mapping512':mapping(hw,512)}


def row(path,dataset,split,source,role,**extra):
    sha=digest(path);vid=hashlib.sha256(f'{dataset}:{source}:{role}:{sha}'.encode()).hexdigest()[:24]
    return {'dataset':dataset,'split':split,'source_id':source,'role':role,'label':int(role=='fake'),
            'video_path':str(path),'video_sha256':sha,'video_id':vid,'spatial_target':'edit_region',
            'spatial_status':'known_background' if role=='real' else 'missing',**extra,**scan(path)}


def af_candidates(config):
    root=Path(config['data_root']);rng=random.Random(SEED)
    af=[r for r in identities(root) if r['scene_source']=='charades' and r['generator'] in ['wan','fcvg'] and r['official_split']=='train']
    official=json.loads((root/'asset_preparation/official_files.json').read_text());rows=[];used=set()
    for split,count in [('train',24),('dev',8)]:
        buckets={}
        for gen in ['wan','fcvg']:
            candidates=sorted([r for r in af if r['generator']==gen],key=lambda r:r['media_path']);rng.shuffle(candidates);buckets[gen]=iter(candidates)
        for i in range(count):
            gen=['wan','fcvg'][i%2]
            r=next(x for x in buckets[gen] if x['source_id'] not in used);used.add(r['source_id'])
            source=r['source_id'];sid=source.split(':')[1];pair='af:'+sid
            for role,path in [('real',real_path(root,'charades',sid)),('fake',root/'official'/official['revision']/r['media_path'])]:
                rows.append(row(path,'ActivityForensics',split,source,role,pair_id=pair,generator=gen,
                    official_revision=official['revision'],official_split=r['official_split'],
                    annotation_source='official annot.zip',intervals_s=r['intervals_s'] if role=='fake' else []))
    return rows,rng


def fetch_anakin(path,entry,root,out):
    name=path.relative_to(root).as_posix()
    check={'size':entry['size'],'lfs':entry.get('lfs'),'blob_id':entry.get('blobId')}
    if verify(path,check)['status']=='VERIFIED':return
    # Official Hub client was tried first in preparation, but its HEAD/CDN path
    # stalled or rate-limited. This measured full-object Range transport still
    # requires the pinned official size and SHA before publishing a file.
    url=f'https://hf-mirror.com/datasets/AlexBlck/ANAKIN/resolve/{REV}/{name}'
    session=requests.Session();session.trust_env=False;started=time.monotonic()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.joint.part')
    for attempt in range(4):
        try:
            with session.get(url,headers={'Range':f"bytes=0-{entry['size']-1}"},stream=True,timeout=(15,30)) as response:
                response.raise_for_status()
                with tmp.open('wb') as f:
                    for block in response.iter_content(1024*1024):f.write(block)
            if verify(tmp,check)['status']!='VERIFIED':raise ValueError('PINNED_ANAKIN_DOWNLOAD_CHECKSUM:'+name)
            tmp.replace(path)
            with (out/'download_transport.jsonl').open('a') as f:f.write(json.dumps({'path':name,'revision':REV,'transport':'direct_hf_mirror_full_range','bytes':path.stat().st_size,'elapsed_s':time.monotonic()-started,'official_check':'VERIFIED'})+'\n')
            return
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2*(attempt+1))


def select(config,out):
    dest=out/'candidate_manifest.json'
    if dest.exists():return json.loads(dest.read_text())['rows']
    root=Path(config['data_root']);an=root/'secondary_datasets/anakin';rows,rng=af_candidates(config)
    refs=an/'references';meta=list(csv.DictReader((refs/'metadata.csv').open()))
    audit=json.loads((an/'metadata_audit.json').read_text());info=json.loads((refs/'dataset_info.json').read_text())
    if info['sha']!=REV:raise ValueError('ANAKIN_REVISION')
    entries={e['rfilename']:e for e in info['siblings']};known=json.loads(Path(config['anakin_scene_review']).read_text())['rows']
    bad={r['video_id'] for r in known if r['scene_status']=='not_matching'}
    bad|={r['video_id'] for r in json.loads((an/'media_audit.json').read_text())['rows'] if r['alignment']['status']!='aligned'}
    policy=json.loads(Path(config['anakin_selection']).read_text());import re
    if digest(refs/'metadata.csv')!=policy['metadata_sha256'] or digest(refs/'ANAKIN.py')!=policy['loader_sha256']:
        raise ValueError('ANAKIN_REFERENCE_CHANGED')
    mask_ids=[r['video-id'] for r in meta if r['has-masks']=='TRUE'];random.Random(47).shuffle(mask_ids)
    n=len(mask_ids);expected={'train':mask_ids[:int(n*.7)],'validation':mask_ids[int(n*.7):int(n*.8)],'test':mask_ids[int(n*.8):]}
    if audit['config_splits']['has-masks']!=expected:raise ValueError('OFFICIAL_HAS_MASKS_SPLIT_MISMATCH')
    rejected=[];urls=set()
    for split,count,official_split in [('train',24,'train'),('dev',8,'validation')]:
        ids=set(audit['config_splits']['has-masks'][official_split])
        candidates=sorted([r for r in meta if r['video-id'] in ids and r['video-id'] not in bad
            and r['manipulation-type'] in ['splicing','inpainting','swap']
            and re.search(policy['human_regex'],r['task'],re.I) and not re.search(policy['exclude_regex'],r['task'],re.I)],key=lambda r:r['video-id'])
        rng.shuffle(candidates);selected=0
        for r in candidates:
            if r['url'] in urls:continue
            vid=r['video-id'];paths={k:an/k/(vid+('.zip' if k=='masks' else '.mp4')) for k in ['trimmed','edited','masks']}
            for k,path in paths.items():
                name=path.relative_to(an).as_posix();e=entries[name];check={'size':e['size'],'lfs':e.get('lfs'),'blob_id':e.get('blobId')}
                fetch_anakin(path,e,an,out)
                if verify(path,check)['status']!='VERIFIED':raise ValueError('ANAKIN_DOWNLOAD_INTEGRITY:'+name)
            fake_meta=scan(paths['edited'])
            with zipfile.ZipFile(paths['masks']) as z:
                members=numeric_members(z.namelist());match=alignment([i for i,n in members],fake_meta['actual_frames'])
                sizes=set()
                for i,name in members:
                    raw,binary,mode=mask_array(z.read(name));sizes.add(raw.shape)
            if match['status']!='aligned' or sizes!={tuple(fake_meta['source_hw'])}:
                rejected.append({'video_id':vid,'reason':'mask_dimension_mismatch' if sizes!={tuple(fake_meta['source_hw'])} else 'alignment_unresolved',
                    'alignment':match,'mask_count':len(members),'edited_frames':fake_meta['actual_frames'],
                    'mask_sizes_hw':sorted([list(x) for x in sizes]),'edited_hw':fake_meta['source_hw']})
                save(out/'selection_rejections.json',rejected);continue
            pair='an:'+vid;urls.add(r['url']);source='anakin:'+r['url']
            for role,k in [('real','trimmed'),('fake','edited')]:
                rows.append(row(paths[k],'ANAKIN',split,source,role,pair_id=pair,generator=r['manipulation-type'],
                    original_url=r['url'],official_revision=REV,official_config='has-masks',official_split=official_split,
                    annotation_source='official masks/'+vid+'.zip',task=r['task'],mask_zip=str(paths['masks']),
                    mask_sha256=digest(paths['masks']),mask_members=members,index_base=match['index_base'],
                    spatial_status='edit_mask' if role=='fake' else 'known_background'))
            selected+=1
            save(out/'selection_progress.json',{'dataset':'ANAKIN','split':split,'selected':selected,'target':count})
            if selected==count:break
        if selected!=count:raise ValueError(f'INSUFFICIENT_ELIGIBLE_ANAKIN:{split}:{selected}/{count}')
    train={r['source_id'] for r in rows if r['split']=='train'};dev={r['source_id'] for r in rows if r['split']=='dev'}
    if train&dev:raise ValueError('SOURCE_LEAKAGE')
    save(dest,{'schema':SCHEMA,'seed':SEED,'rows':rows,'source_overlap':sorted(train&dev)})
    save(out/'selection_rejections.json',rejected)
    return rows


class Video:
    def __init__(self,out,r,stats=None,grouping=None):
        self.row=r;self.directory=out/'cache'/r['video_id'];self.stats=stats;self.grouping=grouping
        meta=json.loads((self.directory/'complete.json').read_text())
        if meta['video_sha256']!=r['video_sha256'] or meta['schema']!=SCHEMA:raise ValueError('VIDEO_CACHE_IDENTITY')
        self.arrays={k:np.load(self.directory/(k+'.npy'),mmap_mode='r') for k in meta['fields']}
        self.fitted_components=None
        if grouping:
            identity={'grouping':grouping,'video_sha256':r['video_sha256'],
                'raw_hashes':{k:meta['hashes'][k] for k in ['mu','valid','area','motion','motion_valid']}}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
            path=self.directory/('fitted_components_'+key+'.npy');sidecar=path.with_suffix('.json')
            if not sidecar.exists():
                tmp=path.with_name(path.name+'.tmp');a=self.arrays
                comp=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.int64,shape=(r['actual_frames'],256))
                for i in range(r['actual_frames']):
                    comp[i]=components(a['mu'][i],a['valid'][i].any((-1,-2)),a['motion'][i],a['motion_valid'][i],a['area'][i]>0,grouping)
                comp.flush();del comp;tmp.replace(path)
                save(sidecar,{'identity':identity,'sha256':digest(path)})
            saved=json.loads(sidecar.read_text())
            if saved['identity']!=identity or digest(path)!=saved['sha256']:raise ValueError('FITTED_COMPONENT_CACHE_IDENTITY')
            self.fitted_components=np.load(path,mmap_mode='r')
        if r['spatial_status']=='edit_mask' and digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SOURCE_CHANGED')
        self.zip=zipfile.ZipFile(r['mask_zip']) if r['spatial_status']=='edit_mask' else None
    def close(self):
        if self.zip:self.zip.close()
    def native_gt(self,i):
        if self.row['spatial_status']=='missing':return None
        if self.zip:return mask_array(self.zip.read(self.row['mask_members'][i][1]))[1]
        return np.zeros(self.row['source_hw'],bool)
    def chunk(self,start,end):
        a={k:np.array(v[start:end],copy=True) for k,v in self.arrays.items()}
        if self.stats:
            a['x']=((a['x']-np.asarray(self.stats['cell_mean'],np.float32))/np.asarray(self.stats['cell_scale'],np.float32)).astype(np.float32)
            a['x'][a['area']==0]=0
        if self.grouping:
            a['component_ids']=np.array(self.fitted_components[start:end],copy=True)
        return {k:torch.from_numpy(v) for k,v in a.items() if k not in ['motion','motion_valid']}


def extract(config,out,rows,progress):
    from research_tools.v8.full_coverage_observation_field.frontend import MogeV2,CoTrackerShortPrefix
    from research_tools.v8.full_coverage_observation_field.pipeline import FrozenResNet18Spatial
    work=Path(config['workspace']);started=time.monotonic()
    moge=MogeV2(str(work/'checkpoints/moge-2-vits-normal.pt'),'cuda');tracker=CoTrackerShortPrefix(str(work/'checkpoints/scaled_online.pth'),'cuda');rgb=FrozenResNet18Spatial('cuda')
    providers={'moge':digest(work/'checkpoints/moge-2-vits-normal.pt'),'cotracker':digest(work/'checkpoints/scaled_online.pth'),'resnet18':digest(Path(rgb.checkpoint))}
    front=None
    for ri,r in enumerate(rows):
        directory=out/'cache'/r['video_id'];complete=directory/'complete.json'
        if complete.exists():
            meta=json.loads(complete.read_text())
            if meta['video_sha256']!=r['video_sha256'] or meta['schema']!=SCHEMA:raise ValueError('CACHE_IDENTITY')
            if meta['providers']!=providers:raise ValueError('FRONTEND_WEIGHTS_CHANGED')
            for k,v in meta['hashes'].items():
                if digest(directory/(k+'.npy'))!=v:raise ValueError('CACHE_CHECKSUM:'+k)
            continue
        if digest(Path(r['video_path']))!=r['video_sha256']:raise ValueError('SOURCE_MEDIA_CHANGED')
        directory.mkdir(parents=True,exist_ok=True);arrays={};context=[];previous=None;frames=[]
        front=Frontend(out/'unused_legacy',directory,moge,tracker,rgb,{'cell_mean':[0.]*77,'cell_scale':[1.]*77},
            {'spatial_scale':1.,'motion_scale':1.,'merge_threshold':0.},SCHEMA,persist_raw=False)
        masks=zipfile.ZipFile(r['mask_zip']) if r['spatial_status']=='edit_mask' else None
        for i,bgr,info in decode(r['video_path']):
            if info['timestamp_source']!='NATIVE_PTS' or abs(info['pts_s']-r['pts'][i]['pts_s'])>1e-8:raise ValueError('DECODE_IDENTITY')
            a,fm,ctx,tr=front.build(r,bgr,info,context,previous)
            context=(context+[ctx])[-2:];previous=a['area']>0;frames.append(info)
            # Frontend components above are deliberately unused until train-only fitting.
            a.pop('component_ids')
            a['motion']=a['x'][:,72:74].copy();a['motion_valid']=a['x'][:,74].astype(bool)
            a['rgb512']=canvas(bgr,r['mapping512'])[:,:,::-1].copy()
            gt=np.zeros((512,512),bool);known=r['spatial_status']!='missing'
            native_positive=False
            if masks:
                native_mask=mask_array(masks.read(r['mask_members'][i][1]))[1]
                native_positive=bool(native_mask.any());gt=mask_to_canvas(native_mask,r['mapping512'])
            a['content512']=content_mask(r['mapping512']);a['gt']=gt;a['pixel_known']=np.array(known,bool)
            a['frame_label']=np.array(0. if r['role']=='real' else float(native_positive) if masks else float(any(lo<=info['pts_s']<=hi for lo,hi in r['intervals_s'])),np.float32)
            if not arrays:
                arrays={k:np.lib.format.open_memmap(directory/(k+'.npy'),mode='w+',dtype=v.dtype,shape=(r['actual_frames'],*v.shape)) for k,v in a.items()}
            for k,v in a.items():arrays[k][i]=v
            if i%100==0:progress('frontend',video=r['video_id'],video_index=ri,frames=i+1,total_frames=r['actual_frames'])
        if len(frames)!=r['actual_frames']:raise ValueError('FRAME_COVERAGE')
        if masks:masks.close()
        for a in arrays.values():a.flush()
        save(complete,{'schema':SCHEMA,'video_sha256':r['video_sha256'],'fields':list(arrays),
            'hashes':{k:digest(directory/(k+'.npy')) for k in arrays},'frames':frames,'cost':front.cost,'elapsed_s':time.monotonic()-started,
            'providers':providers})
        progress('frontend_video_complete',video=r['video_id'],frames=len(frames))
    del front,moge,tracker,rgb;torch.cuda.empty_cache()


def fit(out,rows,path):
    if path.exists():
        result=json.loads(path.read_text())
        if result['train_video_ids']!=[r['video_id'] for r in rows if r['split']=='train']:raise ValueError('STATISTICS_TRAIN_IDENTITY')
        return result
    sums=np.zeros(77);squares=sums.copy();xn=0;gs=np.zeros(3);gss=gs.copy();rs=gs.copy();gn=0;sd=[];md=[];edges=[]
    ii,jj=cell_neighbors().T
    for r in rows:
        if r['split']!='train':continue
        v=Video(out,r)
        for start in range(0,r['actual_frames'],8):
            a={k:np.asarray(x[start:start+8]) for k,x in v.arrays.items() if k in ['x','area','xyz','valid','mu','motion','motion_valid']}
            active=a['area']>0;x=a['x'][active].astype(np.float64);sums+=x.sum(0);squares+=(x*x).sum(0);xn+=len(x)
            xyz=a['xyz'];valid=a['valid'];values=xyz[valid].astype(np.float64);gs+=values.sum(0);gss+=(values*values).sum(0);gn+=len(values)
            relative=(xyz-a['mu'][:,:,None,None,:])[valid].astype(np.float64);rs+=(relative*relative).sum(0)
            gv=valid.any((-1,-2));mv=a['motion_valid'];both=active[:,ii]&active[:,jj];sg=both&gv[:,ii]&gv[:,jj];mg=both&mv[:,ii]&mv[:,jj]
            ds=np.linalg.norm(a['mu'][:,ii]-a['mu'][:,jj],axis=-1);dm=np.linalg.norm(a['motion'][:,ii]-a['motion'][:,jj],axis=-1)
            sd.append(ds[sg]);md.append(dm[mg]);edges.append((ds,dm,sg,mg))
        v.close()
    if not xn or not gn:raise ValueError('NO_TRAIN_VALID_GEOMETRY')
    scale=lambda a:np.where(a<1e-6,1.,a)
    mean=sums/xn;am=gs/gn
    stats={'cell_mean':mean.tolist(),'cell_scale':scale(np.sqrt(np.maximum(squares/xn-mean**2,0))).tolist(),
           'absolute_mean':am.tolist(),'absolute_scale':scale(np.sqrt(np.maximum(gss/gn-am**2,0))).tolist(),'relative_scale':scale(np.sqrt(rs/gn)).tolist()}
    ss=np.concatenate(sd);ms=np.concatenate(md)
    if not np.any(ss>1e-8) or not np.any(ms>1e-8):raise ValueError('TRAIN_GROUPING_SCALE_DEGENERATE')
    ss=float(np.median(ss[ss>1e-8]));ms=float(np.median(ms[ms>1e-8]));cost=[]
    for ds,dm,sg,mg in edges:
        n=sg.astype(int)+mg.astype(int);cost.append(((ds/ss*sg+dm/ms*mg)/np.maximum(n,1))[n>0])
    result={'statistics':stats,'grouping':{'spatial_scale':ss,'motion_scale':ms,'merge_threshold':float(np.quantile(np.concatenate(cost),.25))},
        'train_video_ids':[r['video_id'] for r in rows if r['split']=='train'],'fit_valid_pixels':gn,'fit_active_cell_times':xn}
    save(path,result);return result
