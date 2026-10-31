"""One frozen AF queue, shared observations and condition-specific structure arrays.

Caller: this experiment's prepare/run CLIs. Output: mmap chunk inputs for the
existing VideoTraining. Old caches lack matched 3D displacement and cannot be
sliced into a valid 2D comparator; no old cache is modified.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.prepare_v8_assets import digest,identities,real_path
from research_tools.v8.spatial_localization.joint_data import row,save
from research_tools.v8.full_video_streaming.inputs import Frontend,decode,patch,tracker_summary
from research_tools.v8.full_coverage_observation_field.frontend import letterbox
from research_tools.v8.full_coverage_observation_field.common import cell_neighbors
from research_tools.v8.local_geometry_pilot.data import components
from .model import CONDITIONS

SEED=20260909

def candidates(config,out):
    path=out/'candidate_order.json'
    if path.exists():return json.loads(path.read_text())
    root=Path(config['data_root']);official=identities(root)
    old=json.loads((Path(config['old_run'])/'joint_manifest.json').read_text())['rows']
    old=[r for r in old if r['dataset']=='ActivityForensics']
    known={r['source_id'] for r in old}
    history=json.loads((root/'asset_preparation/historical_probe_recovery.json').read_text())
    known|={'charades:'+s for s in history['validation_source_ids']};known.add('charades:1LARL')
    # Previous V8 training protocol used official train only; holdout must also
    # exclude any source with another edit in official train, not just same file.
    official_train={r['source_id'] for r in official if r['official_split']=='train'}
    pool=[r for r in official if r['scene_source']=='charades' and r['generator'] in ['wan','fcvg'] and r['source_id'] not in known]
    order=[];rng=random.Random(SEED)
    for split in ['test','train']:
        buckets={}
        for gen in ['wan','fcvg']:
            b=sorted([r for r in pool if r['generator']==gen and r['official_split']==split
                      and (split!='test' or r['source_id'] not in official_train)],key=lambda r:r['media_path'])
            rng.shuffle(b);buckets[gen]=b
        used=set();queues={g:iter(v) for g,v in buckets.items()}
        while queues:
            for g in list(queues):
                r=next((r for r in queues[g] if r['source_id'] not in used),None)
                if r is None:del queues[g];continue
                used.add(r['source_id']);order.append({**r,'review_group':split})
    result={'seed':SEED,'old_rows':old,'order':order,'known_used_sources':sorted(known),
            'test_rule':'official test source absent from every official train row and all known past-used sources',
            'old_train_manifest_unavailable':True,'official_revision':old[0]['official_revision']}
    save(path,result);return result


def previews(config,out,limit=100):
    order=candidates(config,out)['order'];root=Path(config['data_root']);dest=out/'scene_preview';dest.mkdir(exist_ok=True)
    # Fixed prefix before any media inspection, enough reserve for scene rejects.
    chosen=[r for group,n in [('test',40),('train',60)] for r in [x for x in order if x['review_group']==group][:n]]
    for batch in range(0,len(chosen),8):
        p=dest/f'sheet_{batch//8:02d}.jpg'
        if p.exists():continue
        sheet=np.full((8*130,3*240,3),245,np.uint8)
        for j,r in enumerate(chosen[batch:batch+8]):
            cap=cv2.VideoCapture(str(real_path(root,'charades',r['source_id'].split(':')[1])))
            count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            for k,f in enumerate([.25,.5,.75]):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int((count-1)*f));ok,img=cap.read()
                if not ok:raise ValueError('REAL_PREVIEW_DECODE:'+r['source_id'])
                thumb=cv2.resize(img,(240,105));sheet[j*130+25:j*130+130,k*240:(k+1)*240]=thumb
            cv2.putText(sheet,f"{batch+j}: {r['source_id']} {r['review_group']}",(4,j*130+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
            cap.release()
        cv2.imwrite(str(p),sheet)
    save(out/'scene_preview_candidates.json',{'rows':chosen,'rule':'original only; fixed 25/50/75%; no fake/model inspected'})
    return chosen


def freeze(config,out):
    target=out/'manifest.json'
    if target.exists():return json.loads(target.read_text())['rows']
    c=candidates(config,out);review=json.loads((out/'scene_review.json').read_text())
    selected=list(c['old_rows']);used={r['source_id'] for r in selected};root=Path(config['data_root'])
    quota={'train':48,'dev':20,'test':20};counts=Counter(r['split'] for r in selected if r['role']=='fake')
    gen_counts={s:Counter(r['generator'] for r in selected if r['split']==s and r['role']=='fake') for s in quota}
    for split,group in [('test','test'),('dev','train'),('train','train')]:
        while counts[split]<quota[split]:
            gen=min(['wan','fcvg'],key=lambda g:gen_counts[split][g])
            available=[r for r in c['order'] if r['review_group']==group and r['source_id'] not in used and r['generator']==gen and review.get(r['source_id'],{}).get('status')=='matching']
            if not available:raise ValueError(f'INSUFFICIENT_QUALIFIED:{split}:{counts[split]}/{quota[split]}:{gen}')
            r=available[0];source=r['source_id'];used.add(source);counts[split]+=1;gen_counts[split][gen]+=1
            for role,p in [('real',real_path(root,'charades',source.split(':')[1])),('fake',root/'official'/c['official_revision']/r['media_path'])]:
                selected.append(row(p,'ActivityForensics',split,source,role,pair_id='af:'+source.split(':')[1],generator=gen,
                    official_revision=c['official_revision'],official_split=r['official_split'],
                    intervals_s=r['intervals_s'] if role=='fake' else [],annotation_source='official annot.zip exact composite identity',
                    scene_review=review[source]))
    validate_manifest(selected)
    save(target,{'schema':'v8-3d-temporal-detection-core-v1','seed':SEED,'rows':selected,'pairs':dict(counts),
                 'generator_counts':{k:dict(v) for k,v in gen_counts.items()},'test_rule':c['test_rule'],
                 'historical_identity_limit':'Full historical train list absent; test eligibility supported by official test-only sources and old official-train protocol, not a recovered exhaustive intersection.',
                 'candidate_order_sha256':digest(out/'candidate_order.json'),'scene_review_sha256':digest(out/'scene_review.json')})
    return selected


def validate_manifest(rows):
    sets={s:{r['source_id'] for r in rows if r['split']==s} for s in ['train','dev','test']}
    if any(sets[a]&sets[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):raise ValueError('SOURCE_LEAKAGE')
    for s,n in [('train',48),('dev',20),('test',20)]:
        if len(sets[s])!=n:raise ValueError('QUEUE_SIZE:'+s)
    for source in set.union(*sets.values()):
        rr=[r for r in rows if r['source_id']==source]
        if len(rr)!=2 or {r['role'] for r in rr}!={'real','fake'}:raise ValueError('PAIR_IDENTITY')


def matched_motion(tr,context_indices,current_xyz,current_valid,previous_xyz,previous_valid):
    """Current query cell assignment; same visible correspondences in 2D and 3D.

    3D additionally requires sampled geometry at BOTH endpoints. Missing remains
    explicit; displacements are observations in camera coordinates, not velocity.
    """
    tracks,vis=tr['tracks'],tr['visibility'];now=tr['query_frame_in_padded_prefix']
    q=np.arange(1024);cell=(q//32//2)*16+(q%32//2)
    result={}
    for dim in [2,3]:
        delta=np.zeros((256,3),np.float32);count=np.zeros(256,np.int32)
        if len(context_indices)>1:
            a,b=tracks[now],tracks[now-1]
            good=vis[now]&vis[now-1]&np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
            good&=(a>=0).all(-1)&(a<=255).all(-1)&(b>=0).all(-1)&(b<=255).all(-1)
            ia=np.rint(np.where(good[:,None],a,0)).astype(int);ib=np.rint(np.where(good[:,None],b,0)).astype(int)
            if dim==3:
                good&=current_valid[ia[:,1],ia[:,0]]&previous_valid[ib[:,1],ib[:,0]]
                diff=current_xyz[ia[:,1],ia[:,0]]-previous_xyz[ib[:,1],ib[:,0]]
            else:
                diff=np.pad((a-b)/255,((0,0),(0,1)))
            np.add.at(delta,cell[good],diff[good]);np.add.at(count,cell[good],1)
            delta/=np.maximum(count[:,None],1)
        result[f'motion{dim}']=delta;result[f'motion_valid{dim}']=count>0
    return result


def unpatch(a):
    if a.ndim==4:return a.reshape(16,16,16,16,3).transpose(0,2,1,3,4).reshape(256,256,3)
    return a.reshape(16,16,16,16).transpose(0,2,1,3).reshape(256,256)


def extract(config,out,rows,progress):
    from research_tools.v8.full_coverage_observation_field.frontend import MogeV2,CoTrackerShortPrefix
    from research_tools.v8.full_coverage_observation_field.pipeline import FrozenResNet18Spatial
    work=Path(config['workspace']);start=time.monotonic();old=Path(config['old_run'])/'cache'
    providers={'moge':digest(work/'checkpoints/moge-2-vits-normal.pt'),'cotracker':digest(work/'checkpoints/scaled_online.pth'),
               'resnet18':digest(work/'cache/torch/hub/checkpoints/resnet18-f37072fd.pth')}
    # Providers are lazy; old RGB/XYZ are read-only. Only absent raw tracking is recomputed.
    tracker=CoTrackerShortPrefix(str(work/'checkpoints/scaled_online.pth'),'cuda');moge=rgb=None
    costs=[]
    for ri,r in enumerate(rows):
        dest=out/'observations'/r['video_id'];meta=dest/'complete.json'
        if meta.exists():
            m=json.loads(meta.read_text())
            if m['video_sha256']!=r['video_sha256'] or m['frames']!=r['actual_frames']:raise ValueError('CACHE_IDENTITY')
            if m.get('providers')!=providers:raise ValueError('PROVIDER_IDENTITY')
            continue
        dest.mkdir(parents=True,exist_ok=True);oldmeta=old/r['video_id']/'complete.json';reuse=None
        if oldmeta.exists():
            om=json.loads(oldmeta.read_text())
            if om['video_sha256']!=r['video_sha256'] or om['providers']!=providers:raise ValueError('OLD_CACHE_IDENTITY')
            for k in ['x','mu','xyz','valid','content','area','predecessor','association_weight','history_available','delta_t']:
                if digest(old/r['video_id']/(k+'.npy'))!=om['hashes'][k]:raise ValueError('OLD_CACHE_CHECKSUM:'+k)
            reuse={k:np.load(old/r['video_id']/(k+'.npy'),mmap_mode='r') for k in ['x','xyz','valid','content','mu','area','predecessor','association_weight','history_available','delta_t']}
        elif moge is None:
            moge=MogeV2(str(work/'checkpoints/moge-2-vits-normal.pt'),'cuda');rgb=FrozenResNet18Spatial('cuda')
        front=None if reuse else Frontend(out/'no_legacy',dest,moge,tracker,rgb,{'cell_mean':[0.]*77,'cell_scale':[1.]*77},
                {'spatial_scale':1.,'motion_scale':1.,'merge_threshold':0.},'v8-core-observations-v1',persist_raw=False)
        context=[];previous=None;prev_xyz=prev_valid=None;arrays={};tr_s=0.;t0=time.monotonic()
        for i,bgr,info in decode(r['video_path']):
            if i>=r['actual_frames'] or info['timestamp_source']!='NATIVE_PTS' or abs(info['pts_s']-r['pts'][i]['pts_s'])>1e-8:raise ValueError('NATIVE_PTS_IDENTITY')
            if reuse:
                a={k:np.array(v[i],copy=True) for k,v in reuse.items()};can,_=letterbox(bgr)
                prefix=(context+[(i,can[:,:,::-1].copy(),info['pts_s'])])[-3:]
                ts=time.monotonic();tr=tracker.query([p[1] for p in prefix],len(prefix)-1);torch.cuda.synchronize();tr_s+=time.monotonic()-ts
                ctx=(i,can[:,:,::-1].copy(),info['pts_s'])
            else:a,_,ctx,tr=front.build(r,bgr,info,context,previous)
            xyz=unpatch(a['xyz']);valid=unpatch(a['valid']&a['content'])
            mm=matched_motion(tr,[p[0] for p in context]+[i],xyz,valid,prev_xyz,prev_valid)
            a={**{k:a[k] for k in ['x','mu','area','predecessor','association_weight','history_available','delta_t']},
               'valid':a['valid']&a['content'],
               'geometry_count':(a['valid']&a['content']).sum((-1,-2)).astype(np.float32),**mm}
            if not arrays:
                arrays={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=v.dtype,shape=(r['actual_frames'],*v.shape)) for k,v in a.items()}
            for k,v in a.items():arrays[k][i]=v
            context=(context+[ctx])[-2:];previous=a['area']>0;prev_xyz,prev_valid=xyz,valid
            if i%200==0:progress('observations',video_index=ri,video=r['video_id'],frames=i+1,total=r['actual_frames'])
        if i+1!=r['actual_frames']:raise ValueError('ALL_FRAME_COVERAGE')
        for v in arrays.values():v.flush()
        cost={'video_id':r['video_id'],'frames':i+1,'elapsed_s':time.monotonic()-t0,'old_RGB_XYZ_reused':bool(reuse),
              'missing_raw_tracking_recomputed':bool(reuse),'tracking_s_reused_geometry':tr_s,'frontend_cost':front.cost if front else None}
        save(meta,{'schema':'v8-core-observations-v1','video_sha256':r['video_sha256'],'frames':i+1,'fields':list(arrays),
                   'hashes':{k:digest(dest/(k+'.npy')) for k in arrays},'cost':cost,'providers':providers})
        costs.append(cost);print('OBSERVATION_COMPLETE',ri+1,len(rows),r['video_id'],cost['elapsed_s'],flush=True)
    save(out/'preparation_cost.json',{'current_call_s':time.monotonic()-start,'videos':[json.loads((out/'observations'/r['video_id']/'complete.json').read_text())['cost'] for r in rows]})
    del tracker,moge,rgb;torch.cuda.empty_cache()


def base_arrays(out,r,dim=None):
    dest=out/'observations'/r['video_id'];m=json.loads((dest/'complete.json').read_text())
    if m['video_sha256']!=r['video_sha256'] or m['frames']!=r['actual_frames']:raise ValueError('OBSERVATION_IDENTITY')
    fields=m['fields']
    if dim==2:fields=[k for k in fields if k not in ['mu','geometry_count','motion3','motion_valid3','valid']]
    return {k:np.load(dest/(k+'.npy'),mmap_mode='r') for k in fields}


def representation(a,dim):
    # 2D NEVER accesses mu, geometry_count or motion3. Current UV is content-aware.
    area=np.asarray(a['area']);t=len(area)
    # Exact content-aware UV means include partial boundary cells.
    uv=np.broadcast_to(a['uv'],(t,256,3))
    pos=uv.copy() if dim==2 else np.asarray(a['mu']).copy()
    count=area.copy() if dim==2 else np.asarray(a['geometry_count']).copy()
    valid=count>0;motion=np.asarray(a[f'motion{dim}']);mv=np.asarray(a[f'motion_valid{dim}'])
    x=np.concatenate((np.asarray(a['x'])[...,:67],pos,valid[...,None],np.asarray(a['x'])[...,71:72],motion,
        mv[...,None],np.asarray(a['history_available'])[...,None],(~valid)[...,None],uv[...,:2]),-1).astype(np.float32)
    x[area==0]=0
    return x,pos,count,motion,mv


def prepare_structures(config,out,rows,progress):
    fitted_path=out/'statistics.json'
    if fitted_path.exists():return json.loads(fitted_path.read_text())
    # Fit each representation on exactly the same training cell-times only.
    stats={};edges=cell_neighbors();ii,jj=edges.T
    for dim,condition in zip([2,3],CONDITIONS):
        sx=np.zeros(80);sq=sx.copy();n=0;ps=np.zeros(3);pq=ps.copy();pn=0;spatial=[];motions=[];edge_values=[]
        for r in rows:
            a=base_arrays(out,r,dim);a['uv']=uv_means(r)
            if r['split']!='train':continue
            for start in range(0,r['actual_frames'],16):
                b={k:v[start:start+16] if k!='uv' else v for k,v in a.items()}
                x,p,count,m,mv=representation(b,dim);active=b['area']>0;z=x[active].astype(float);sx+=z.sum(0);sq+=(z*z).sum(0);n+=len(z)
                valid=count>0;z=p[valid].astype(float);ps+=z.sum(0);pq+=(z*z).sum(0);pn+=len(z)
                both=active[:,ii]&active[:,jj];sg=both&valid[:,ii]&valid[:,jj];mg=both&mv[:,ii]&mv[:,jj]
                ds=np.linalg.norm(p[:,ii]-p[:,jj],axis=-1);dm=np.linalg.norm(m[:,ii]-m[:,jj],axis=-1)
                spatial.append(ds[sg]);motions.append(dm[mg]);edge_values.append((ds,dm,sg,mg))
        if not n or not pn:raise ValueError('NO_VALID_TRAIN_OBSERVATIONS')
        safe=lambda s:np.where(s<1e-6,1.,s)
        mean=sx/n;pm=ps/pn;ss=np.concatenate(spatial);ms=np.concatenate(motions)
        if not np.any(ss>1e-8) or not np.any(ms>1e-8):raise ValueError('GROUPING_SCALE_DEGENERATE')
        ss=float(np.median(ss[ss>1e-8]));ms=float(np.median(ms[ms>1e-8]));cost=[]
        for ds,dm,sg,mg in edge_values:
            nn=sg.astype(int)+mg.astype(int);cost.append(((ds/ss*sg+dm/ms*mg)/np.maximum(nn,1))[nn>0])
        stats[condition]={'statistics':{'cell_mean':mean.tolist(),'cell_scale':safe(np.sqrt(np.maximum(sq/n-mean**2,0))).tolist(),
            'absolute_mean':pm.tolist(),'absolute_scale':safe(np.sqrt(np.maximum(pq/pn-pm**2,0))).tolist(),'relative_scale':[1.]*3},
            'grouping':{'spatial_scale':ss,'motion_scale':ms,'merge_threshold':float(np.quantile(np.concatenate(cost),.25))},
            'train_video_ids':[r['video_id'] for r in rows if r['split']=='train']}
        progress('fit_structure',condition=condition)
    for dim,c in zip([2,3],CONDITIONS):
        for ri,r in enumerate(rows):
            dest=out/'structures'/c/r['video_id'];dest.mkdir(parents=True,exist_ok=True);a=base_arrays(out,r,dim);a['uv']=uv_means(r)
            result={};fitted=stats[c];st=fitted['statistics']
            for start in range(0,r['actual_frames'],16):
                b={k:v[start:start+16] if k!='uv' else v for k,v in a.items()};x,p,count,m,mv=representation(b,dim);area=b['area']
                x=((x-np.array(st['cell_mean'],np.float32))/np.array(st['cell_scale'],np.float32)).astype(np.float32);x[area==0]=0
                comp=np.stack([components(p[j],count[j]>0,m[j],mv[j],area[j]>0,fitted['grouping']) for j in range(len(x))])
                value={'x':x,'mu':p,'geometry_count':count,'area':np.asarray(area),'component_ids':comp,
                    **{k:np.asarray(b[k]) for k in ['predecessor','association_weight','history_available','delta_t']}}
                if not result:result={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=v.dtype,shape=(r['actual_frames'],*v.shape[1:])) for k,v in value.items()}
                for k,v in value.items():result[k][start:start+len(x)]=v
            for v in result.values():v.flush()
            save(dest/'complete.json',{'fields':list(result),'video_sha256':r['video_sha256'],'condition':c,
                'hashes':{k:digest(dest/(k+'.npy')) for k in result}})
            progress('structure_cache',condition=c,video_index=ri,total=len(rows))
    save(fitted_path,stats);return stats


def uv_means(r):
    m=r['mapping256'];x,y=m['pad_xy'];h,w=m['content_hw'];mask=np.zeros((256,256),bool);mask[y:y+h,x:x+w]=True
    yy,xx=np.mgrid[:256,:256];p=patch(np.stack([xx/255,yy/255,np.zeros_like(xx)],-1).astype(np.float32));cm=patch(mask)
    return ((p*cm[...,None]).sum((1,2))/np.maximum(cm.sum((1,2))[:,None],1)).astype(np.float32)


def load_video(out,r,condition):
    started=time.monotonic();dest=out/'structures'/condition/r['video_id'];m=json.loads((dest/'complete.json').read_text())
    if m['video_sha256']!=r['video_sha256'] or m['condition']!=condition:raise ValueError('STRUCTURE_IDENTITY')
    a={k:torch.from_numpy(np.load(dest/(k+'.npy'),allow_pickle=False)) for k in m['fields']}
    # Mean replicated geometry only needs validity/content masks, not saved dense XYZ.
    count=a.pop('geometry_count');cm=content_patches(r)
    a['content']=torch.from_numpy(np.broadcast_to(cm,(r['actual_frames'],256,16,16)).copy())
    # Validity pattern is preserved separately for 3D patch encoding; see obs mask below.
    if condition=='STRUCTURE_3D':
        p=out/'observations'/r['video_id']/'valid.npy';a['valid']=torch.from_numpy(np.load(p,allow_pickle=False))
    else:a['valid']=a['content'].clone()
    return a,time.monotonic()-started


def content_patches(r):
    m=r['mapping256'];x,y=m['pad_xy'];h,w=m['content_hw'];cm=np.zeros((256,256),bool);cm[y:y+h,x:x+w]=True;return patch(cm)
