"""Fixed original-role pairs, native pixels, PTS, dense-map/token compatibility.

Called only by PVST train/evaluate. No old model/statistics or component input.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scripts.prepare_v8_assets import digest
from research_tools.v8.spatial_localization.joint_data import save,mapping
from research_tools.v8.full_video_streaming.inputs import decode
from research_tools.v8.temporal_detection_core.data import unpatch
from scripts.review_anakin import mask_array


def freeze(config,out):
    path=out/'manifest.json'
    if path.exists():return json.loads(path.read_text())['rows']
    original=Path(config['source_run'])/'joint_manifest.json';rows=json.loads(original.read_text())['rows'];selected=[]
    rng=random.Random(config['seed'])
    for dataset,kinds in [('ActivityForensics',['wan','fcvg']),('ANAKIN',['inpainting','splicing'])]:
        for split,n in [('train',8),('dev',4)]:
            chosen=[]
            for kind in kinds:
                rr=sorted([r for r in rows if r['dataset']==dataset and r['split']==split and r['role']=='fake' and r['generator']==kind],key=lambda r:(r['source_id'],r['pair_id'],r['video_sha256']))
                unique={}
                for r in rr:unique.setdefault(r['source_id'],r)
                rr=list(unique.values());rng.shuffle(rr)
                rr=[r for r in rr if r['source_id'] not in config.get('excluded_original_scene_sources',[])]
                if len(rr)<n//2:raise ValueError('INSUFFICIENT_FIXED_ROLE_PAIRS')
                chosen+=rr[:n//2]
            for fake in sorted(chosen,key=lambda r:r['source_id']):
                pair=[r for r in rows if r['dataset']==dataset and r['split']==split and r['source_id']==fake['source_id'] and r['pair_id']==fake['pair_id']]
                if len(pair)!=2 or {r['role'] for r in pair}!={'real','fake'}:raise ValueError('COMPOSITE_PAIR_IDENTITY')
                for r in sorted(pair,key=lambda r:r['role']):
                    selected.append({**r,'native_pixels':int(np.prod(r['source_hw'])),'dense_source':str(Path(config['source_run'])/'cache'/r['video_id']),
                                     'selection':'original train/dev role; stable identity then seeded draw; no prediction used'})
    tr={r['source_id'] for r in selected if r['split']=='train'};de={r['source_id'] for r in selected if r['split']=='dev'}
    if tr&de:raise ValueError('SOURCE_LEAKAGE')
    save(path,{'schema':'pvst-native-v02','seed':config['seed'],'source_manifest_sha256':digest(original),'rows':selected})
    return selected


def native_dense(row,index):
    """Old cache contains REAL per-pixel MoGe points, not x/mu/component fields.

    Crop original256 letterbox then bilinear XYZ; valid only if all interpolated
    contributors valid. This is upsampling a measured dense field, not new geometry.
    """
    p=Path(row['dense_source']);meta=json.loads((p/'complete.json').read_text())
    if meta['video_sha256']!=row['video_sha256']:raise ValueError('DENSE_SOURCE_IDENTITY')
    xyz=np.load(p/'xyz.npy',mmap_mode='r')[index];valid=np.load(p/'valid.npy',mmap_mode='r')[index]
    xyz=unpatch(xyz);valid=unpatch(valid);m=row['mapping256'];x,y=m['pad_xy'];h,w=m['content_hw'];hw=row['source_hw']
    a=xyz[y:y+h,x:x+w];v=valid[y:y+h,x:x+w]
    mapped=cv2.resize(np.where(v[...,None],a,0),(hw[1],hw[0]),interpolation=cv2.INTER_LINEAR)
    weights=cv2.resize(v.astype(np.float32),(hw[1],hw[0]),interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(a[v]).all():raise ValueError('VALID_DENSE_POINT_NONFINITE')
    good=(weights>=1-1e-6)&np.isfinite(mapped).all(-1)
    return np.where(good[...,None],mapped,0).astype(np.float32),good


def official(config):
    """Load official trained decoder; unwrap BASE DINO weights, discard all LoRA.

    No old adapted detector. Frozen backbone is used once for shared448 tokens.
    """
    from research_tools.v8.pretrained_spatial_context.assets import official as load
    import tempfile
    with tempfile.TemporaryDirectory(prefix='pvst-official-') as directory:m=load(config,Path(directory))
    # PEFT's get_base_model still has LoraLayer children. Build plain official DINO
    # and map only base_layer weights; never merge pretrained LoRA deltas.
    original=torch.hub.load
    backbone=original(config['dino_code'],'dinov2_vitl14_reg',source='local',pretrained=False)
    state={}
    for key,value in m.backbone.state_dict().items():
        if 'lora_' in key:continue
        key=key.removeprefix('base_model.model.').replace('.base_layer.','.')
        state[key]=value
    backbone.load_state_dict(state,strict=True);backbone.eval().requires_grad_(False)
    if any('lora_' in key for key in backbone.state_dict()):raise ValueError('LORA_MUST_NOT_ENTER_PVST')
    print('PVST encoder: frozen plain DINOv2; all LoRA tensors discarded, not merged',flush=True)
    decoder=m.encoder.localization_head;decoder.requires_grad_(True)
    del m
    return backbone,decoder


def decoder_only(config):
    sys.path.insert(0,config['official_code'])
    from ainpaint.modules.context_decoder import HighResContextHead
    decoder=HighResContextHead(1024,1,norm='BN')
    state=torch.load(config['checkpoint'],map_location='cpu',mmap=True,weights_only=False)['model_state_dict'];prefix='encoder.localization_head.'
    decoder.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    return decoder


def prepare(config,out,rows,progress,limit=None):
    """Native RGB stays mmap/disk; tokens need new pass: old ones contain LoRA/518."""
    from research_tools.v8.spatial_localization.joint_data import canvas,mapping
    call_started=time.monotonic();encoded_frames=0;backbone=None;cost=[];weights=digest(Path(config['checkpoint']));moge_sha=digest(Path(config['workspace'])/'checkpoints/moge-2-vits-normal.pt')
    if weights!=config['checkpoint_sha256']:raise ValueError('PINNED_INITIALIZATION_CHANGED')
    for ri,r in enumerate(rows):
        folder=out/'cache'/r['video_id'];complete=folder/'complete.json';partial=folder/'partial.json';prepared=0;prior_seconds=0.
        if complete.exists():
            m=json.loads(complete.read_text())
            if m['video_sha256']!=r['video_sha256'] or m['checkpoint_sha256']!=weights:raise ValueError('CACHE_IDENTITY')
            expected={'prepared_frames':r['actual_frames'],'frames':r['actual_frames'],'native_hw':r['source_hw'],
                'official_revision':config['official_revision'],'dino_revision':config['dino_revision'],
                'tokens':'no LoRA; 448 letterbox; image patch tokens only','mapping448':mapping(r['source_hw'],448)}
            if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('COMPLETE_CACHE_FIELDS_INCOMPATIBLE:'+r['video_id'])
            rgb=np.load(folder/'rgb.npy',mmap_mode='r');tokens=np.load(folder/'tokens.npy',mmap_mode='r')
            if rgb.shape!=(r['actual_frames'],*r['source_hw'],3) or rgb.dtype!=np.uint8 or tokens.shape!=(r['actual_frames'],1024,32,32) or tokens.dtype!=np.float16:raise ValueError('COMPLETE_CACHE_HEADER_INCOMPATIBLE:'+r['video_id'])
            del rgb,tokens
            continue
        folder.mkdir(parents=True,exist_ok=True);h,w=r['source_hw'];n=r['actual_frames'];started=time.monotonic()
        if partial.exists():
            oldpartial=json.loads(partial.read_text())
            if oldpartial['video_sha256']!=r['video_sha256'] or oldpartial['checkpoint_sha256']!=weights:raise ValueError('PARTIAL_CACHE_IDENTITY')
            prepared=oldpartial['prepared_frames'];prior_seconds=oldpartial['seconds']
        needed=n if limit is None else min(n,limit)
        if prepared>=needed:continue
        if digest(Path(r['video_path']))!=r['video_sha256']:raise ValueError('SOURCE_MEDIA_SHA_CHANGED')
        old=json.loads((Path(r['dense_source'])/'complete.json').read_text())
        if old['video_sha256']!=r['video_sha256'] or old['providers']['moge']!=moge_sha:raise ValueError('DENSE_PROVIDER_IDENTITY')
        xyzheader=np.load(Path(r['dense_source'])/'xyz.npy',mmap_mode='r')
        if xyzheader.shape!=(n,256,16,16,3):raise ValueError('TRUE_DENSE_XYZ_SHAPE')
        if r['dataset']=='ANAKIN' and digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SOURCE_SHA_CHANGED')
        if len(old['frames'])!=n:raise ValueError('DENSE_FRAME_COVERAGE')
        for i,p in enumerate(r['pts']):
            if old['frames'][i]['frame_index']!=i or abs(old['frames'][i]['pts_s']-p['pts_s'])>1e-8:raise ValueError('DENSE_PTS')
        if backbone is None:backbone,_=official(config);backbone.cuda()
        rgb=np.lib.format.open_memmap(folder/'rgb.npy',mode='r+' if prepared else 'w+',dtype=np.uint8,shape=(n,h,w,3))
        tokens=np.lib.format.open_memmap(folder/'tokens.npy',mode='r+' if prepared else 'w+',dtype=np.float16,shape=(n,1024,32,32))
        mm=mapping([h,w],448)
        for i,bgr,info in decode(r['video_path']):
            if i<prepared:continue
            if i>=needed:break
            if info['timestamp_source']!='NATIVE_PTS' or abs(info['pts_s']-r['pts'][i]['pts_s'])>1e-8:raise ValueError('NATIVE_DECODE_PTS')
            rgb[i]=bgr[:,:,::-1]
            t=torch.from_numpy(canvas(bgr,mm)[:,:,::-1].copy()).permute(2,0,1)[None].cuda().float()/255
            t=(t-t.new_tensor([.485,.456,.406])[None,:,None,None])/t.new_tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                image=backbone.forward_features(t)['x_norm_patchtokens']
                if image.shape!=(1,1024,1024):raise ValueError('NO_CLS_REGISTER_IN_GRID')
                image=image.transpose(1,2).reshape(1,1024,32,32)
            tokens[i]=image.float().cpu().numpy()[0].astype(np.float16)
            if i%100==0:progress('frozen_tokens',video_index=ri,frames=i+1,total=n)
        if needed==n and i+1!=n:raise ValueError('ALL_NATIVE_FRAMES')
        rgb.flush();tokens.flush();torch.cuda.synchronize()
        save(complete if needed==n else partial,{'prepared_frames':needed,'video_sha256':r['video_sha256'],'checkpoint_sha256':weights,'tokens':'no LoRA; 448 letterbox; image patch tokens only','official_revision':'753ae1b700b054c97e15569718690211e36cdf53','dino_revision':'7764ea0f912e53c92e82eb78a2a1631e92725fc8',
             'native_hw':[h,w],'frames':n,'mapping448':mm,'seconds':prior_seconds+time.monotonic()-started,
             'geometry':'reuse real dense256 XYZ, crop letterbox/bilinear to native; strict valid interpolation; no cell mean'})
        cost.append(time.monotonic()-started);encoded_frames+=needed-prepared
        if needed==n:partial.unlink(missing_ok=True)
    del backbone;torch.cuda.empty_cache()
    save(out/'preparation_cost.json',{'current_call_s':time.monotonic()-call_started,'new_token_frames_this_call':encoded_frames,
             'requested_frames':sum(r['actual_frames'] for r in rows),'total_completed_video_prepare_s':sum(json.loads((out/'cache'/r['video_id']/'complete.json').read_text())['seconds'] for r in rows if (out/'cache'/r['video_id']/'complete.json').exists()),
             'old_518_LoRA_tokens_compatible':False,'moge_recomputed':False})


def fit_scale(config,out,rows):
    path=out/'scale.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['train_video_ids']!=[r['video_id'] for r in rows if r['split']=='train']:raise ValueError('SCALE_TRAIN_IDENTITY_CHANGED')
        return saved['s']
    values=[]
    for r in rows:
        if r['split']!='train':continue
        # Sample from reused dense source BEFORE native interpolation to avoid storing
        # billions of values; fixed every16th valid positive source Z, each frame.
        for i in range(r['actual_frames']):
            p=Path(r['dense_source']);xyz=np.load(p/'xyz.npy',mmap_mode='r')[i];v=np.load(p/'valid.npy',mmap_mode='r')[i]
            z=xyz[...,2];v=v&np.isfinite(xyz).all(-1)&(z>0);values.append(np.asarray(z[v][::16]))
    if not values or not any(len(a) for a in values):raise ValueError('NO_POSITIVE_TRAIN_DEPTH')
    a=np.concatenate(values);s=float(np.median(a))
    save(path,{'s':s,'algorithm':'per-frame every16th valid finite positive dense-source Z, stable row-major flatten then exact median',
        'count':len(a),'train_video_ids':[r['video_id'] for r in rows if r['split']=='train'],
        'coordinate':'MoGe camera-relative arbitrary scale; no per-frame centering/scaling/rotation'})
    return s


class Video:
    def __init__(self,out,row,condition,scale,config=None):
        import zipfile
        from scripts.review_anakin import numeric_members
        from .geometry import voxel_size
        config=config or {};self.h=voxel_size(config,condition)
        self.row=row;self.condition=condition;self.scale=scale;self._frames={};p=Path(config.get('cache_root',out/'cache'))/row['video_id']
        self.rgb=np.load(p/'rgb.npy',mmap_mode='r');self.tokens=np.load(p/'tokens.npy',mmap_mode='r')
        meta=p/'complete.json' if (p/'complete.json').exists() else p/'partial.json'
        state=json.loads(meta.read_text())
        if state['video_sha256']!=row['video_sha256']:raise ValueError('VIDEO_CACHE_IDENTITY')
        self.available=row['actual_frames'] if meta.name=='complete.json' else state['prepared_frames']
        if self.rgb.shape!=(row['actual_frames'],*row['source_hw'],3) or self.tokens.shape!=(row['actual_frames'],1024,32,32):raise ValueError('NATIVE_RGB_TOKEN_SHAPE')
        self.zip=zipfile.ZipFile(row['mask_zip']) if row['dataset']=='ANAKIN' and row['role']=='fake' else None
        self.members=numeric_members(self.zip.namelist()) if self.zip else None
        if row['dataset']=='ActivityForensics' and row['role']=='fake' and not row.get('intervals_s'):raise ValueError('MISSING_OFFICIAL_FAKE_INTERVAL')
        if self.zip is not None:
            from scripts.review_anakin import alignment
            match=alignment([i for i,_ in self.members],row['actual_frames'])
            if match['status']!='aligned' or match['index_base']!=row['index_base']:raise ValueError('MASK_FRAME_ALIGNMENT')
    def close(self):
        if self.zip:self.zip.close()
        self._frames.clear()
    def gt(self,i):
        if self.row['dataset']!='ANAKIN':return None
        if self.zip:
            mask=mask_array(self.zip.read(self.members[i][1]))[1]
            if list(mask.shape)!=self.row['source_hw']:raise ValueError('NATIVE_MASK_DIMENSION')
            return mask
        return np.zeros(self.row['source_hw'],bool)
    def label(self,i):
        if self.row['role']=='real':return False
        if self.row['dataset']=='ANAKIN':return bool(self.gt(i).any())
        t=self.row['pts'][i]['pts_s'];return any(lo<=t<=hi for lo,hi in self.row['intervals_s'])
    def frames(self,start,end,device):
        if start<0 or end>self.available:raise ValueError('MISSING_PREPARED_NATIVE_FRAMES')
        # Only immutable observations/routing, never decoder or learned features.
        self._frames={i:f for i,f in self._frames.items() if start<=i<end}
        result=[];h,w=self.row['source_hw']
        yy,xx=np.mgrid[:h,:w];uv=np.stack((2*(xx+.5)/w-1,2*(yy+.5)/h-1,np.zeros_like(xx)),-1).astype(np.float32)
        for i in range(start,end):
            if i in self._frames:result.append(self._frames[i]);continue
            if self.condition=='UV_PVST':q=uv;valid=np.ones((h,w),bool)
            else:
                with torch.profiler.record_function('pvst.native_geometry_read'):
                    q,valid=native_dense(self.row,i);q=q/self.scale
            with torch.profiler.record_function('pvst.input_transfer'):
                frame={'rgb':torch.from_numpy(np.array(self.rgb[i],copy=True)).to(device),'tokens':torch.from_numpy(np.array(self.tokens[i],copy=True)).to(device),
                    'q':torch.from_numpy(q).to(device),'valid':torch.from_numpy(valid).to(device),
                    'uv':torch.from_numpy(uv[...,:2]).to(device),'pts':self.row['pts'][i]['pts_s'],
                    'mapping448':mapping([h,w],448)}
            from .geometry import voxelize
            with torch.profiler.record_function('pvst.voxelization'):
                frame['voxel_meta']=voxelize(frame['q'],frame['valid'],self.h)
            self._frames[i]=frame;result.append(frame)
        return result


def preview_originals(out,rows):
    """One fixed original-only contact sheet per six selected sources; no scores."""
    original=sorted([r for r in rows if r['role']=='real'],key=lambda r:(r['dataset'],r['split'],r['source_id']))
    dest=out/'original_preview';dest.mkdir(exist_ok=True);records=[]
    for start in range(0,len(original),6):
        target=dest/f'sheet_{start//6:02d}.jpg';sheet=np.full((6*135,3*240,3),245,np.uint8)
        for j,r in enumerate(original[start:start+6]):
            pts=np.array([p['pts_s'] for p in r['pts']]);indices=[int(np.argmin(abs(pts-(pts[0]+f*(pts[-1]-pts[0]))))) for f in [.25,.5,.75]]
            cap=cv2.VideoCapture(r['video_path'])
            for k,frame in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read()
                if not ok:raise ValueError('ORIGINAL_PREVIEW_DECODE')
                h,w=bgr.shape[:2];scale=min(240/w,110/h);nh,nw=round(h*scale),round(w*scale)
                sheet[j*135+25:j*135+25+nh,k*240:k*240+nw]=cv2.resize(bgr,(nw,nh))
            cap.release();label=r['dataset']+'/'+r['split']+'/'+r['pair_id']
            cv2.putText(sheet,label,(4,j*135+17),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
            records.append({'source_id':r['source_id'],'video_id':r['video_id'],'sheet':target.name,'row':j,'frame_indices':indices})
        cv2.imwrite(str(target),sheet)
    save(dest/'selection.json',{'rule':'all selected original real videos, native PTS 25/50/75%; no fake/model selection','rows':records})


def select_clips(rows,out,config,scale):
    """Freeze <=4 distinct stride8 targets/video using training GT/time only."""
    rng=random.Random(config['seed']);clips=[]
    for r in sorted([r for r in rows if r['split']=='train'],key=lambda r:r['video_id']):
        v=Video(out,r,'UV_PVST',scale,config)
        try:labels=[v.label(i) for i in range(r['actual_frames'])]
        finally:v.close()
        starts=list(range(0,len(labels),8));n=min(config['clips_per_video'],len(starts));selected=[];fallback=False
        def spread(pool,k):
            # Sorted temporal strata, one seeded actual candidate per stratum.
            return [rng.choice(list(a)) for a in np.array_split(sorted(pool),min(k,len(pool))) if len(a)] if k and len(pool) else []
        if r['dataset']=='ActivityForensics' and r['role']=='fake':
            positive=[s for s in starts if any(labels[s:s+8])];negative=[s for s in starts if not any(labels[s:s+8])]
            selected=spread(positive,(n+1)//2)+spread(negative,n//2)
            if len(selected)<n:selected+=spread([s for s in starts if s not in selected],n-len(selected));fallback=True
        else:selected=spread(starts,n)
        for start in sorted(selected):
            end=min(start+8,len(labels));left=max(0,start-16)
            clips.append({'video_id':r['video_id'],'source_id':r['source_id'],'dataset':r['dataset'],'role':r['role'],
                'target_range':[int(start),int(end)],'context_range':[int(left),int(start)],'input_range':[int(left),int(end)],
                'input_pts_s':[p['pts_s'] for p in r['pts'][left:end]],'frame_labels':labels[start:end],
                'label_source':'official AF PTS intervals' if r['dataset']=='ActivityForensics' else 'actual spatial mask foreground (real=zero)',
                'shortage_filled_from_other_candidates':fallback})
    return clips
