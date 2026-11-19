"""Frozen 11-source manifest and shared generic visual/MoGe observation cache.

Immediate caller experiment.prepare; no forensic weights/C or held-out test
media are read. New provider calls are restricted to the frozen 22 videos.
"""
import json
import math
from pathlib import Path
import sys
import time
import zipfile
import av
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from research_tools.rgbd_slice.model import backbone
from research_tools.rgbd_slice.data import geometry_shape
from research_tools.rgbd_slice.experiment import state_hash
from research_tools.depth_slice.data import save
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import numeric_members,mask_array
from .model import route,history_indices

REPO=Path(__file__).resolve().parents[2]
DATA=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection')
ROOT=DATA/'derived';OUT=ROOT/'pvst_xyz_temporal_pilot_v1'
CAND=DATA/'secondary_datasets/pvst_geometry_screen_v1'
CONFIG=REPO/'docs/pvst/xyz_temporal_config_v1.json'


def read(p):return json.loads(Path(p).read_text())
def work_shape(hw,long_edge):
    factor=min(1.,long_edge/max(hw));return tuple(max(1,round(x*factor)) for x in hw)
def progress(stage,**kw):
    save(OUT/'status.json',{'status':'RUNNING','stage':stage,**kw});print(stage,kw,flush=True)


def freeze(c):
    records=read(CAND/'anakin_pvst_candidate_review_v1/inventory.json')+read(CAND/'anakin_pvst_candidate_review_v2/inventory.json')
    allrows=[]
    for split in ['train','dev']:
        for pair in c[split]:
            r=next(x for x in records if x['pair_id']==pair)
            assert r['split']==split and r['mask']['status']=='PASS'
            for role in ['real','fake']:
                m=r['media'][role];path=Path(m['path']);assert digest(path)==m['sha256']
                mask=Path(r['mask']['path']);assert digest(mask)==r['mask']['mask_sha256']
                hw=list(m['sizes_hw'][0]);wh=work_shape(hw,c['work_long_edge']);ph=[math.ceil(x/32)*32 for x in wh]
                allrows.append({'pair_id':pair,'source_id':r['source_id'],'split':split,'role':role,'video_id':pair[3:]+'_'+role,'video_path':str(path),'video_sha256':m['sha256'],'mask_zip':str(mask),'mask_sha256':r['mask']['mask_sha256'],'index_base':r['mask']['index_base'],'actual_frames':m['frame_count'],'pts':[p['seconds'] for p in m['pts']],'native_hw':hw,'work_hw':list(wh),'padded_hw':ph,'official_task':r['official_task'],'human_approval':'explicit current user authorization; fixed/near-fixed and semantic mask correspondence; not anomaly truth'})
    tr={r['source_id'] for r in allrows if r['split']=='train'};de={r['source_id'] for r in allrows if r['split']=='dev'}
    assert len(tr)==7 and len(de)==4 and tr.isdisjoint(de) and len(allrows)==22
    save(OUT/'manifest.json',{'rows':allrows,'excluded_fake_supervised_frames':{'an:40y1x9':[6,7,8,9]},'user_fixed_sources':True})
    return allrows


def occupancy_counts(points,hs):
    assert hs and all(abs(b/a-2)<1e-9 for a,b in zip(hs,hs[1:]))
    coords=np.unique(np.floor(points/hs[0]).astype(np.int64),axis=0);counts=[len(coords)]
    for _ in hs[1:]:coords=np.unique(coords//2,axis=0);counts.append(len(coords))
    return dict(zip(map(str,hs),counts))


def compatible_geometry(row,provider):
    for root in [ROOT/'pvst_rgbd_slice_relations_v1/geometry',ROOT/'pvst_rgb_partition_anakin_scale_v1/geometry']:
        for p in sorted(root.glob('*/complete.json')):
            m=read(p)
            if m.get('video_sha256')!=row['video_sha256']:continue
            expected,_,_=geometry_shape(row['native_hw'])
            pts=[x['pts_s'] for x in m['pts']]
            assert m['moge_weights_sha256']==digest(Path(provider['moge_weights'])) and m['moge_revision']==provider['moge_revision']
            assert m['output_hw']==list(expected) and pts==row['pts']
            return p.parent
    return None


def remap_geometry(q,v,hw):
    h,w=hw
    if q.shape[:2]!=(h,w):
        weights=cv2.resize(v.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        q=cv2.resize(np.where(v[...,None],q,0),(w,h),interpolation=cv2.INTER_LINEAR)
        v=(weights>=1-1e-6)&np.isfinite(q).all(-1)
    return np.where(v[...,None],q,0).astype(np.float32),v


@torch.no_grad()
def extract_video(row,provider,encoder,folder,moge=None,missing=None):
    """Same generic observations for one registered video; only missing packets.

    Called by original prepare and source-expansion preparation. No scale/C/R or
    split selection occurs here. Local official weights only, no downloads.
    """
    start=time.monotonic();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    wanted=set(range(row['actual_frames'])) if missing is None else set(missing)
    assert wanted<=set(range(row['actual_frames']))
    gp=compatible_geometry(row,provider);xyz=valid=None
    if gp:
        xyz=np.load(gp/'xyz.npy',mmap_mode='r');valid=np.load(gp/'valid.npy',mmap_mode='r')
    elif wanted and moge is None:
        sys.path[:0]=[provider['moge_code'],provider['utils3d_code']]
        from moge.model.v2 import MoGeModel
        moge=MoGeModel.from_pretrained(provider['moge_weights']).cuda().eval().requires_grad_(False)
    device=next(encoder.parameters()).device
    z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
    members=numeric_members(z.namelist()) if z else None
    if z:assert len(members)==row['actual_frames'] and [i for i,_ in members]==list(range(row['index_base'],row['index_base']+row['actual_frames']))
    new_geo=reused_geo=visual_calls=count=0
    with av.open(row['video_path']) as container:
        for i,f in enumerate(container.decode(video=0)):
            assert abs(float(f.pts*f.time_base)-row['pts'][i])<1e-8 and [f.height,f.width]==row['native_hw']
            count+=1
            if i not in wanted:continue
            assert not (folder/f'{i:05d}.pt').exists(),'Never overwrite existing observations'
            rgb=f.to_ndarray(format='rgb24');h,w=row['work_hw'];ph,pw=row['padded_hw']
            if gp:q=np.array(xyz[i]);v=np.array(valid[i]);reused_geo+=1
            else:
                ghw,tokens,_=geometry_shape(row['native_hw']);im=cv2.resize(rgb,(ghw[1],ghw[0]),interpolation=cv2.INTER_AREA) if rgb.shape[:2]!=ghw else rgb
                res=moge.infer(torch.from_numpy(im).permute(2,0,1).cuda().float()/255,num_tokens=tokens,force_projection=True,apply_mask=False,use_fp16=True)
                q=res['points'].float().cpu().numpy();v=res['mask'].cpu().numpy().astype(bool)&np.isfinite(q).all(-1);new_geo+=1;del res
            q,v=remap_geometry(q,v,(h,w));image=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA) if rgb.shape[:2]!=(h,w) else rgb
            gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool)
            assert list(gt.shape)==row['native_hw']
            target=cv2.resize(gt.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
            x=torch.from_numpy(image).permute(2,0,1)[None].to(device).float()/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            x=F.pad(x,(0,pw-w,0,ph-h),value=0)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):features,_=encoder(x,x.new_zeros(1,1,ph,pw))
            assert len(features)==4
            packet={'visual':[a[0].half().cpu() for a in features],'rgb':torch.from_numpy(image.copy()),'xyz_provider':torch.from_numpy(q),'valid':torch.from_numpy(v.copy()),'gt_work':torch.from_numpy(target),'frame_positive':bool(gt.any()),'native_GT_pixels':int(gt.sum()),'pts':row['pts'][i],'frame_index':i,'padded_hw':row['padded_hw'],'native_hw':row['native_hw'],'supervised':not(row['pair_id']=='an:40y1x9' and row['role']=='fake' and 6<=i<10)}
            assert all(torch.isfinite(a).all() for a in packet['visual'])
            torch.save(packet,folder/f'{i:05d}.pt');visual_calls+=1
    if z:z.close()
    assert count==row['actual_frames'] and visual_calls==len(wanted)
    record=dict(video=row['video_id'],frames=count,work_hw=row['work_hw'],geometry_reused_from=str(gp) if gp else None,new_geometry_calls=new_geo,reused_geometry_frames=reused_geo,new_visual_calls=visual_calls,seconds=time.monotonic()-start)
    save(folder/'complete.json',record)
    return record,moge


@torch.no_grad()
def prepare():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(2)
    if OUT.exists():raise RuntimeError('Existing output: preserve and inspect, no duplicate preparation')
    OUT.mkdir();c=read(CONFIG);rows=freeze(c);provider=read(ROOT/'pvst_rgbd_slice_relations_v1/config.json')
    save(OUT/'provider_config.json',provider)
    identity={'source_head':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'config_sha256':digest(CONFIG),'manifest_sha256':digest(OUT/'manifest.json'),'DFormer_generic_weight_sha256':provider['dformer_weights_sha256'],'DFormer_revision':provider['dformer_revision'],'MoGe_weight_sha256':digest(Path(provider['moge_weights'])),'MoGe_revision':provider['moge_revision'],'forensic_weights_used':False}
    save(OUT/'identity.json',identity)
    encoder=backbone(provider,OUT/'official_encoder_load.json').cuda().eval().requires_grad_(False);before=state_hash(encoder.state_dict())
    sys.path[:0]=[provider['moge_code'],provider['utils3d_code']]
    from moge.model.v2 import MoGeModel
    moge=None;new_geo=0;reused_geo=0;visual_calls=0;cost=[]
    for row in rows:
        progress('prepare_video',video=row['video_id'])
        record,moge=extract_video(row,provider,encoder,OUT/'cache'/row['video_id'],moge)
        new_geo+=record['new_geometry_calls'];reused_geo+=record['reused_geometry_frames'];visual_calls+=record['new_visual_calls']
        cost.append(record);save(OUT/'preparation_progress.json',cost)
    assert state_hash(encoder.state_dict())==before
    del encoder,moge;torch.cuda.empty_cache()
    # One global train-only scale, sampled solely to estimate a median; no point
    # sampling or node deletion is applied to the actual model inputs.
    samples=[]
    for r in rows:
        if r['split']!='train':continue
        for i in range(r['actual_frames']):
            e=torch.load(OUT/'cache'/r['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
            z=e['xyz_provider'][...,2][e['valid']].numpy();z=z[np.isfinite(z)&(z>0)]
            if len(z):samples.extend(z[np.linspace(0,len(z)-1,min(256,len(z)),dtype=int)].tolist())
    c['scale']=float(np.median(samples));assert c['scale']>0
    maxima={str(h):0 for h in c['h_candidates']};counts={}
    for r in rows:
        per={str(h):[] for h in c['h_candidates']}
        for i in range(r['actual_frames']):
            e=torch.load(OUT/'cache'/r['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
            points=(e['xyz_provider'][e['valid']].numpy()/c['scale'])
            occupied=occupancy_counts(points,c['h_candidates'])
            for h in c['h_candidates']:
                n=occupied[str(h)];per[str(h)].append(n)
                if r['split']=='train':maxima[str(h)]=max(maxima[str(h)],n)
        counts[r['video_id']]=per
    allowed=[h for h in c['h_candidates'] if maxima[str(h)]<=c['train_max_blocks_budget']]
    save(OUT/'scale_selection.json',{'train_sample_median_scale':c['scale'],'median_sample_points':len(samples),'train_maxima':maxima,'node_counts':counts,'no_test_or_dev_scale_selection':True})
    if not allowed:raise RuntimeError('RESOURCE_BLOCKED: no authorized train-count scale candidate')
    c['h']=min(allowed);save(OUT/'config.json',c)
    save(OUT/'preparation.json',{'status':'COMPLETE','exit_code':0,'seconds':time.monotonic()-start,'new_geometry_frames':new_geo,'reused_geometry_frames':reused_geo,'new_frozen_visual_frames':visual_calls,'generic_encoder_unchanged':True,'per_video':cost})
    save(OUT/'status.json',{'status':'PREPARED','exit_code':0});print('PREPARED',c['scale'],c['h'],flush=True)


class Cache:
    def __init__(self,c,root=None):self.c=c;self.root=OUT/'cache' if root is None else Path(root);self.memory={}
    def get(self,row,i,device='cuda'):
        e=torch.load(self.root/row['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
        assert e['frame_index']==i and e['pts']==row['pts'][i] and e['native_hw']==row['native_hw'] and e['padded_hw']==row['padded_hw']
        assert tuple(e['rgb'].shape[:2])==tuple(row['work_hw'])
        e['xyz']=e.pop('xyz_provider')/self.c['scale']
        for k in ['xyz','rgb','valid','gt_work']:e[k]=e[k].to(device)
        e['visual']=[x.to(device=device,dtype=torch.float32) for x in e['visual']]
        return e
    def window(self,row,target,temporal=True):
        ids=history_indices(row['pts'],target,self.c['history_slots'],self.c['history_step_seconds']) if temporal else [target]
        memo={}
        for i in ids:
            if i is not None and i not in memo:memo[i]=self.get(row,i)
        return [memo[i] if i is not None else None for i in ids]
