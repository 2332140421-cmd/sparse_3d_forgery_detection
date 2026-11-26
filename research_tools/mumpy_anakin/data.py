"""ANAKIN RGB-only clips for the single MumPy transfer run.

Caller experiment.py. Reuses frozen89/19 rows and prior decoded work RGB, then
caches RGB224/nearest official masks. No geometry/neural feature is ever loaded
as numeric model input. Old .pt packets are read only to obtain existing RGB.
"""
import json,random,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from research_tools.xyz_pilot.fullvideo_matched_data import WorkVideo,supervised
from research_tools.xyz_pilot.fullvideo_matched import native_mask,numeric_members
from research_tools.xyz_pilot.fullvideo_direct_probe import save,digest
from .model import augmentation

SOURCE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1')
MEAN=torch.tensor([.4776,.479,.4465])[:,None,None]
STD=torch.tensor([.230,.2085,.2324])[:,None,None]


def read(p):return json.loads(Path(p).read_text())

def clip_indices(index,length):
    assert 0<=index<length
    return [max(0,min(length-1,i)) for i in [index-1,index,index+1]]


def targets(rows,split):
    key='train_targets' if split=='train' else 'evaluation_targets'
    return [(r['video_id'],i) for r in rows if r['split']==split for i in r[key]]


def epoch_orders(rows):
    ts=targets(rows,'train');orders=[]
    for e in range(1,11):
        order=ts.copy();random.Random(20260909+e).shuffle(order);orders.append(order)
    return orders


def monitor_targets(rows):
    sources=sorted({r['source_group_id'] for r in rows if r['split']=='train'})
    chosen=set(random.Random(20260909).sample(sources,8))
    result=[]
    for r in rows:
        if r['split']=='train' and r['source_group_id'] in chosen:
            ts=r['train_targets'];result.extend((r['video_id'],ts[int((len(ts)-1)*f)]) for f in [1/3,2/3])
    assert len(result)==32
    return result


def prepare_video(root,row,emit):
    folder=root/'rgb224'/row['video_id'];marker=folder/'complete.json'
    identity={k:row[k] for k in ['video_sha256','mask_sha256','pts','native_hw','work_hw','actual_frames']}
    identity.update(size=[224,224],RGB='existing work RGB -> PIL bicubic full-frame224',GT='official native -> nearest224; categorical')
    if marker.exists():
        old=read(marker);assert old['identity']==identity
        assert (folder/'rgb.npy').stat().st_size>0 and (folder/'gt.npy').stat().st_size>0
        return dict(reused_frames=row['actual_frames'],new_frames=0,seconds=0)
    start=time.monotonic();folder.mkdir(parents=True,exist_ok=True)
    n=row['actual_frames'];rgb=np.lib.format.open_memmap(folder/'rgb.npy',mode='w+',dtype=np.uint8,shape=(n,224,224,3))
    masks=np.lib.format.open_memmap(folder/'gt.npy',mode='w+',dtype=np.bool_,shape=(n,224,224));work=WorkVideo(row);counts=[]
    z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None;members=numeric_members(z.namelist()) if z else None
    if z:assert len(members)==n and [m[0] for m in members]==list(range(row['index_base'],row['index_base']+n))
    try:
        for i in range(n):
            old=work.get(i);gt=native_mask(z,members,row,i);assert int(gt.sum())==old['native_GT_pixels']
            rgb[i]=np.asarray(Image.fromarray(old['rgb'].numpy()).resize((224,224),Image.Resampling.BICUBIC))
            masks[i]=np.asarray(Image.fromarray((gt*255).astype(np.uint8)).resize((224,224),Image.Resampling.NEAREST))>=128
            counts.append(int(gt.sum()))
            if i%50==0 or i+1==n:emit('prepare_RGB224',video=row['video_id'],frame=i+1,T=n)
    finally:
        if z:z.close()
    rgb.flush();masks.flush();del rgb,masks
    save(marker,dict(identity=identity,native_GT_pixels=counts,supervision_excluded_indices=[i for i in range(n) if not supervised(row,i)],seconds=time.monotonic()-start,status='COMPLETE'))
    return dict(new_frames=n,reused_frames=0,seconds=time.monotonic()-start)


class Clips:
    def __init__(self,root,rows,training=False,augmentation_policy="single"):
        self.root=root;self.rows={r['video_id']:r for r in rows};self.arrays={};self.training=training
        self.augment=augmentation(augmentation_policy) if training else None
        self.augmentation_policy=augmentation_policy;self.augmentation_stats={}
    def arrays_for(self,vid):
        if vid not in self.arrays:
            path=self.root/'rgb224'/vid
            assert read(path/'complete.json')['status']=='COMPLETE'
            self.arrays[vid]=(np.load(path/'rgb.npy',mmap_mode='r'),np.load(path/'gt.npy',mmap_mode='r'))
        return self.arrays[vid]
    def get(self,target,augment=None):
        vid,i=target;row=self.rows[vid]
        assert supervised(row,i),'EXCLUDED_TARGET_IS_CONTEXT_ONLY'
        rgb,mask=self.arrays_for(vid);indices=clip_indices(i,len(rgb))
        assert indices[1]==i
        images=[Image.fromarray(np.asarray(rgb[j]).copy()) for j in indices]
        gt=Image.fromarray((np.asarray(mask[i])*255).astype(np.uint8))
        if (self.training if augment is None else augment):
            assert self.augment is not None,'Evaluation clips never enable training augmentation'
            before=float((np.asarray(gt)>0).mean());started=time.monotonic()
            images,gt=self.augment(images,gt)
            if self.augmentation_policy=='double':
                from .double_augment import observe
                observe(self.augmentation_stats,self.augment.ops,row['role'],before,float((np.asarray(gt.resize((224,224),Image.Resampling.NEAREST))>0).mean()),time.monotonic()-started)
        assert len(images)==3
        x=torch.stack([(torch.from_numpy(np.asarray(im.resize((224,224),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255-MEAN)/STD for im in images])
        y=torch.from_numpy((np.asarray(gt.resize((224,224),Image.Resampling.NEAREST))>=128).copy()).float()[None]
        return x,y
    def batch(self,ts):
        pairs=[self.get(t) for t in ts];return torch.stack([p[0] for p in pairs]),torch.stack([p[1] for p in pairs])
