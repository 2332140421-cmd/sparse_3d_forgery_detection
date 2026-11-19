"""Fixed native RGB/crop plan for ADR-0053; no learned features or geometry.

Caller native_patch_experiment. One lossless uint8 mmap/video avoids decoding
random training frames repeatedly. GT stays native and shared by both arms.
"""
import hashlib
import random
import time
import zipfile
import av
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .data import ROOT,read
from .joint_experiment import save,digest
from .ainpaint_spatial import normalize_pad,token_grid
from scripts.review_anakin import numeric_members,mask_array

SOURCE=ROOT/'pvst_xyz_source_expansion_v1'
OUT=ROOT/'pvst_anakin_native_patch_detail_probe_v1'
SIZE=288;STRIDE=144;STEPS=12845
CONDITIONS=['NATIVE_PATCH','LOWDETAIL_PATCH']


def degrade(rgb,work_hw):
    h,w=rgb.shape[:2];wh,ww=work_hw
    small=cv2.resize(rgb,(ww,wh),interpolation=cv2.INTER_AREA) if (h,w)!=(wh,ww) else rgb
    return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR) if small.shape[:2]!=(h,w) else small.copy()


def crop(a,y,x,size=SIZE):
    h,w=a.shape[:2];assert 0<=y<=max(0,h-size) and 0<=x<=max(0,w-size)
    vh,vw=min(size,h-y),min(size,w-x);out=np.zeros((size,size,*a.shape[2:]),dtype=a.dtype)
    out[:vh,:vw]=a[y:y+vh,x:x+vw];valid=np.zeros((size,size),bool);valid[:vh,:vw]=True
    return out,valid


def choose_crop(gt,rng,size=SIZE):
    h,w=gt.shape;positive=np.flatnonzero(gt)
    guided=bool(len(positive) and rng.random()<.5)
    pixel=None
    if guided:
        index=int(positive[rng.randrange(len(positive))]);py,px=divmod(index,w);pixel=[py,px]
        y=rng.randint(max(0,py-size+1),min(py,max(0,h-size)))
        x=rng.randint(max(0,px-size+1),min(px,max(0,w-size)))
    else:y=rng.randint(0,max(0,h-size));x=rng.randint(0,max(0,w-size))
    mask,valid=crop(gt,y,x,size)
    assert not guided or mask.any()
    return dict(y=y,x=x,foreground_guided=guided,selected_positive_pixel=pixel,GT_pixels=int(mask.sum()),valid_pixels=int(valid.sum()))


def positions(n,size=SIZE,stride=STRIDE):
    if n<=size:return [0]
    result=list(range(0,n-size+1,stride))
    if result[-1]!=n-size:result.append(n-size)
    return result


def tiles(hw,size=SIZE,stride=STRIDE):return [(y,x) for y in positions(hw[0],size,stride) for x in positions(hw[1],size,stride)]


def fuse(hw,coords,predictions,size=SIZE):
    """Test/reference uniform-logit backfill; no GT is accepted."""
    total=np.zeros(hw,np.float32);count=np.zeros(hw,np.uint16)
    for (y,x),p in zip(coords,predictions):
        vh,vw=min(size,hw[0]-y),min(size,hw[1]-x)
        total[y:y+vh,x:x+vw]+=p[:vh,:vw];count[y:y+vh,x:x+vw]+=1
    assert np.all(count>0),'Uncovered native pixels'
    return total/count,count


def patch_logits(model,rgbs):
    assert rgbs and all(tuple(x.shape)==(SIZE,SIZE,3) for x in rgbs)
    device=next(model.parameters()).device
    x=torch.cat([normalize_pad(torch.as_tensor(a.copy(),device=device)) for a in rgbs],0)
    model.calls+=len(rgbs)
    out=model.backbone.forward_features(x);features=token_grid(out,x.shape[-2:])
    p=model.head(features,target_size=x.shape[-2:]);assert p.shape==(len(rgbs),1,*x.shape[-2:])
    return p[:,0,:SIZE,:SIZE]


def masked_losses(logits,gt,valid):
    z=logits.float()[valid];g=gt.float()[valid];assert z.numel()>0
    p=z.sigmoid()
    return torch.stack([F.binary_cross_entropy_with_logits(z,g),1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)])


class NativeData:
    def __init__(self):self.images={};self.masks={}
    def rgb(self,row,i):
        vid=row['video_id']
        if vid not in self.images:self.images[vid]=np.load(OUT/'native'/f'{vid}.npy',mmap_mode='r')
        return np.asarray(self.images[vid][i])
    def gt(self,row,i):
        if row['role']=='real':return np.zeros(row['native_hw'],bool)
        vid=row['video_id']
        if vid not in self.masks:self.masks[vid]=np.load(OUT/'native'/f'{vid}_gt.npy',mmap_mode='r')
        return np.asarray(self.masks[vid][i])
    def training_crop(self,row,item,condition):
        image=self.rgb(row,item['target']);gt=self.gt(row,item['target'])
        if condition=='LOWDETAIL_PATCH':image=degrade(image,row['work_hw'])
        else:assert condition=='NATIVE_PATCH'
        rgb,valid=crop(image,item['y'],item['x']);mask,v=crop(gt,item['y'],item['x']);assert np.array_equal(valid,v)
        assert int(mask.sum())==item['GT_pixels'] and int(valid.sum())==item['valid_pixels']
        return rgb,mask,valid
    def window(self,row,i,partition,condition):
        assert partition=='evaluation' and condition in CONDITIONS
        return [dict(rgb=self.rgb(row,i),native_GT_pixels=int(self.gt(row,i).sum()))]


@torch.no_grad()
def tiled_logits(model,image,work_hw,condition,batch):
    if condition=='LOWDETAIL_PATCH':image=degrade(image,work_hw)
    else:assert condition=='NATIVE_PATCH'
    h,w=image.shape[:2];coords=tiles((h,w));total=np.zeros((h,w),np.float32);count=np.zeros((h,w),np.uint16)
    for start in range(0,len(coords),batch):
        chunk=coords[start:start+batch];rgbs=[crop(image,y,x)[0] for y,x in chunk]
        with torch.autocast('cuda',dtype=torch.bfloat16):z=patch_logits(model,rgbs).float().cpu().numpy()
        for (y,x),p in zip(chunk,z):
            vh,vw=min(SIZE,h-y),min(SIZE,w-x);total[y:y+vh,x:x+vw]+=p[:vh,:vw];count[y:y+vh,x:x+vw]+=1
    assert np.all(count>0) and np.isfinite(total).all()
    return total/count,dict(tiles=len(coords),coverage_min=int(count.min()),coverage_max=int(count.max()))


class TiledModel(torch.nn.Module):
    def __init__(self,model,condition,batch):
        super().__init__();self.model=model;self.condition=condition;self.batch=batch;self.work_hw=None;self.records=[]
    def forward(self,frames,temporal=False):
        assert len(frames)==1 and not temporal
        z,record=tiled_logits(self.model,frames[0]['rgb'],self.work_hw,self.condition,self.batch);self.records.append(record)
        return dict(pixel=torch.from_numpy(z),blocks=0,window_blocks=0)


class EvalData:
    def __init__(self,data,wrapper):self.data=data;self.wrapper=wrapper
    def window(self,row,i,partition,condition):
        self.wrapper.work_hw=row['work_hw']
        return self.data.window(row,i,partition,condition)


def prepare(protocol,orders):
    start=time.monotonic();folder=OUT/'native';folder.mkdir(parents=True,exist_ok=False);records=[];decoded=0
    for row in protocol['expanded']+protocol['dev']:
        n=row['actual_frames'];h,w=row['native_hw'];vid=row['video_id']
        assert digest(row['video_path'])==row['video_sha256']
        array=np.lib.format.open_memmap(folder/f'{vid}.npy',mode='w+',dtype=np.uint8,shape=(n,h,w,3))
        z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        members=numeric_members(z.namelist()) if z else None
        if z:
            assert len(members)==n and [j for j,_ in members]==list(range(row['index_base'],row['index_base']+n))
            masks=np.lib.format.open_memmap(folder/f'{vid}_gt.npy',mode='w+',dtype=bool,shape=(n,h,w))
        hashes=[];parity=[];count=0
        with av.open(row['video_path']) as container:
            stream=container.streams.video[0];stream.thread_count=2
            for i,f in enumerate(container.decode(stream)):
                assert i<n and [f.height,f.width]==[h,w] and abs(float(f.pts*f.time_base)-row['pts'][i])<1e-8
                rgb=f.to_ndarray(format='rgb24');array[i]=rgb;hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest());count+=1
                gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros((h,w),bool)
                assert gt.shape==(h,w)
                if z:masks[i]=gt
                if i in {0,n//2,n-1}:
                    old=torch.load(SOURCE/'cache'/vid/f'{i:05d}.pt',weights_only=True,map_location='cpu')
                    small=cv2.resize(rgb,(row['work_hw'][1],row['work_hw'][0]),interpolation=cv2.INTER_AREA)
                    assert np.array_equal(small,old['rgb'].numpy()) and int(gt.sum())==old['native_GT_pixels']
                    parity.append(i)
        assert count==n;array.flush();del array
        if z:masks.flush();del masks;z.close()
        decoded+=count;records.append(dict(video_id=vid,video_sha256=row['video_sha256'],native_hw=[h,w],frames=n,frame_RGB_sha256=hashes,PTS=row['pts'],legacy_RGB_shrink_exact_at=parity))
        print('NATIVE_DECODE',vid,decoded,flush=True)
    save(OUT/'native_manifest.json',dict(rows=records,decoded_unique_frames=decoded,bytes=sum(p.stat().st_size for p in folder.glob('*.npy')),seconds=time.monotonic()-start,new_MoGe=0,new_DFormer=0))
    data=NativeData();byid={r['video_id']:r for r in protocol['expanded']};rng=random.Random(20260909);plan=[]
    for step,item in enumerate(orders,1):
        row=byid[item['video_id']];selected=choose_crop(data.gt(row,item['target']),rng)
        plan.append(dict(step=step,**item,**selected))
    groups={}
    for x in plan:
        key=byid[x['video_id']]['role']+('_guided' if x['foreground_guided'] else '_uniform')
        r=groups.setdefault(key,dict(crops=0,GT_pixels=0,valid_pixels=0,positive_crops=0))
        r['crops']+=1;r['GT_pixels']+=x['GT_pixels'];r['valid_pixels']+=x['valid_pixels'];r['positive_crops']+=int(x['GT_pixels']>0)
    for r in groups.values():r['GT_fraction']=r['GT_pixels']/r['valid_pixels']
    save(OUT/'sampling_summary.json',groups)
    save(OUT/'crop_plan.json',plan);save(OUT/'BN_crop_plan.json',[plan[e*2569:e*2569+100] for e in range(5)])
    save(OUT/'tile_plan.json',[dict(video_id=r['video_id'],native_hw=r['native_hw'],positions=tiles(r['native_hw']),targets=r['evaluation_targets']) for r in protocol['dev']])
    return plan
