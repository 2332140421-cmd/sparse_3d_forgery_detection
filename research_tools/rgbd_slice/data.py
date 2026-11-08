"""Runner input: registered native RGB/GT and independent high-resolution MoGe.

No old low-resolution XYZ or frozen visual features enter this experiment.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps

from research_tools.depth_slice.data import Video as RegisteredVideo, save
from scripts.prepare_v8_assets import digest


def geometry_shape(hw):
    h,w=hw;scale=min(1.,768/max(h,w));ih=max(1,round(h*scale));iw=max(1,round(w*scale))
    tokens=max(1,round(ih*iw/14**2));aspect=iw/ih
    return (ih,iw),tokens,(round(math.sqrt(tokens/aspect))*14,round(math.sqrt(tokens*aspect))*14)


def encoder_depth(z,valid):
    """Official PNG Greys_r per-image mapping, then single-channel normalization.

    Separate from metric/slice Z. Invalid pixels get explicit black display-fill,
    not observed geometry and not a negative label; native RGB/GT stay present.
    """
    v=valid&np.isfinite(z);out=np.zeros(z.shape,np.uint8)
    if v.any():
        lo,hi=float(z[v].min()),float(z[v].max());a=np.zeros_like(z,np.float32)
        if hi>lo:a[v]=(z[v]-lo)/(hi-lo)
        out[v]=(colormaps['Greys_r'](a[v],bytes=True)[...,0])
    return (out.astype(np.float32)/255-.48)/.28


def geometry(config,out,rows,progress):
    import sys,time
    sys.path[:0]=[config['moge_code'],config['utils3d_code']]
    from moge.model.v2 import MoGeModel
    expected_sha=digest(Path(config['moge_weights']));model=None;started=time.monotonic();calls=0
    for r in rows:
        folder=out/'geometry'/r['video_id'];folder.mkdir(parents=True,exist_ok=True)
        hw,tokens,internal=geometry_shape(r['source_hw']);n=r['actual_frames']
        identity={'video_sha256':r['video_sha256'],'moge_weights_sha256':expected_sha,'moge_revision':config['moge_revision'],
                  'native_hw':r['source_hw'],'input_hw':list(hw),'output_hw':list(hw),'num_tokens':tokens,'internal_vit_hw':list(internal),
                  'force_projection':True,'apply_mask':False,'use_fp16':True,'frames':n,'pts':r['pts'],'units':'provider camera-forward Z; normalized slice units are Z/fixed scale, not certified metres'}
        complete=folder/'complete.json'
        if complete.exists():
            if json.loads(complete.read_text())!=identity:raise ValueError('HIGH_RES_GEOMETRY_IDENTITY')
            continue
        partial=folder/'partial.json';count=0
        if partial.exists():
            record=json.loads(partial.read_text());count=record.pop('prepared_frames')
            if record!=identity:raise ValueError('PARTIAL_HIGH_RES_IDENTITY')
        if model is None:model=MoGeModel.from_pretrained(config['moge_weights']).cuda().eval()
        v=RegisteredVideo(r,config);mode='r+' if count else 'w+'
        xyz=np.lib.format.open_memmap(folder/'xyz.npy',mode=mode,dtype=np.float32,shape=(n,*hw,3))
        valid=np.lib.format.open_memmap(folder/'valid.npy',mode=mode,dtype=bool,shape=(n,*hw))
        for i in range(count,n):
            image=np.array(v.rgb[i]);image=cv2.resize(image,(hw[1],hw[0]),interpolation=cv2.INTER_AREA) if tuple(r['source_hw'])!=hw else image
            t=torch.from_numpy(image).permute(2,0,1).cuda().float()/255
            with torch.inference_mode():res=model.infer(t,num_tokens=tokens,force_projection=True,apply_mask=False,use_fp16=True)
            q=res['points'].float().cpu().numpy();mask=res['mask'].cpu().numpy().astype(bool)&np.isfinite(q).all(-1)
            if q.shape!=(*hw,3):raise ValueError('MOGE_OUTPUT_SHAPE')
            xyz[i]=np.where(mask[...,None],q,0);valid[i]=mask;calls+=1
            xyz.flush();valid.flush();save(partial,{**identity,'prepared_frames':i+1})
            if i%25==0:progress('geometry',video=r['video_id'],frames=i+1,total=n,input_hw=hw,num_tokens=tokens)
        v.close();save(complete,identity);partial.unlink(missing_ok=True)
    del model;torch.cuda.empty_cache();save(out/'geometry_cost.json',{'seconds':time.monotonic()-started,'new_frontend_frames':calls})


class Video(RegisteredVideo):
    def __init__(self,row,config):
        super().__init__(row,config)
        folder=Path(config.get('geometry_root',Path(config['output'])/'geometry'))/row['video_id'];meta=json.loads((folder/'complete.json').read_text())
        if meta['video_sha256']!=row['video_sha256'] or meta['pts']!=row['pts']:raise ValueError('HIGH_RES_FRAME_IDENTITY')
        self.xyz=np.load(folder/'xyz.npy',mmap_mode='r');self.valid=np.load(folder/'valid.npy',mmap_mode='r')
    def frame(self,i):
        h,w=self.row['source_hw'];q=np.array(self.xyz[i]);v=np.array(self.valid[i])
        if q.shape[:2]!=(h,w):
            weights=cv2.resize(v.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
            q=cv2.resize(np.where(v[...,None],q,0),(w,h),interpolation=cv2.INTER_LINEAR);v=(weights>=1-1e-6)&np.isfinite(q).all(-1)
        q=np.where(v[...,None],q,0).astype(np.float32)
        return {'rgb':torch.from_numpy(np.array(self.rgb[i],copy=True)),'xyz':torch.from_numpy(q/self.config['scale']),
                'valid':torch.from_numpy(v),'depth':torch.from_numpy(encoder_depth(q[...,2],v)),
                'gt':torch.from_numpy(self.gt(i)),'pts':self.row['pts'][i]['pts_s'],'frame_index':i}


def frames(video,start,end):return [video.frame(i) for i in range(start,end)]
