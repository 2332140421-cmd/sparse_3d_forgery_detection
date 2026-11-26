"""Official DoubleAugment with categorical-mask interpolation only.

Called by Clips for the one augmentation comparison. Keeps the official strategy,
operation lists, random calls and RGB math; copying only two short shape paths
is necessary because their released functions interpolate label masks as RGB.
"""
import collections
import random
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageOps
from .model import official_augmentation_module


def categorical_corner(mod,images,mask):
    w,h=mask.size;corners=mod.FindCorners(np.asarray(mask))
    if corners is None:return images,mask
    left,top,right,bottom=corners
    l=np.random.randint(0,left-1) if left>1 else 0
    r=np.random.randint(right+1,w-1) if right+1<w-1 else w-1
    t=np.random.randint(0,top-1) if top>1 else 0
    b=np.random.randint(bottom+1,h-1) if bottom+1<h-1 else h-1
    box=(l,t,r,b)
    return [im.crop(box).resize((w,h)) for im in images],mask.crop(box).resize((w,h),Image.Resampling.NEAREST)


class DoubleAugment:
    def __init__(self):
        self.module=mod=official_augmentation_module()
        self.policy=mod.RandAugment(1,7,mod.DoubleAugmentStrategy())
        # Keep original named operations and ranges; replace only mask resampling.
        self.policy.shape_change_augment_list=[(self.rotate if op.__name__=='RandomRotate' else self.scale if op.__name__=='RandomScaleCrop' else op,lo,hi) for op,lo,hi in self.policy.shape_change_augment_list]
        self.ops=[]

    def rotate(self,pair,v):
        mod=self.module;images,mask=pair;angle=int(v)
        images=[mod.rotate_image(im,angle,True) for im in images]
        original=mod.cv2.warpAffine
        def nearest(*args,**kwargs):
            kwargs['flags']=mod.cv2.INTER_NEAREST
            return original(*args,**kwargs)
        # Exactly the same OpenCV affine/crop, only categorical mask resampling.
        with patch.object(mod.cv2,'warpAffine',nearest):mask=mod.rotate_image(mask.convert('RGB'),angle,True).convert('L')
        return images,mask

    def scale(self,pair,v):
        mod=self.module;images,mask=pair;crop_size=int(v);short=random.randint(256,1024)
        w,h=images[0].size
        if h>w:ow=short;oh=int(h*ow/w)
        else:oh=short;ow=int(w*oh/h)
        resized=[im.resize((ow,oh),Image.Resampling.BICUBIC) for im in images]
        mask=mask.resize((ow,oh),Image.Resampling.NEAREST)
        if short<crop_size:
            pad=(0,0,max(0,crop_size-ow),max(0,crop_size-oh))
            resized=[ImageOps.expand(im,border=pad,fill=0) for im in resized];mask=ImageOps.expand(mask,border=pad,fill=0)
        return categorical_corner(mod,resized,mask)

    def __call__(self,images,mask):
        result=self.policy(images,mask)
        names={'rotate':'RandomRotate','scale':'RandomScaleCrop'}
        self.ops=[(names.get(op.__name__,op.__name__),lo,hi) for op,lo,hi in self.policy.ops]
        assert set(np.unique(np.asarray(result[1])))<={0,255},'NONCATEGORICAL_AUGMENTED_MASK'
        return result


def observe(stats,ops,role,before,after,seconds):
    stats['targets']=stats.get('targets',0)+1;stats['augmentation_seconds']=stats.get('augmentation_seconds',0.)+seconds
    counts=stats.setdefault('operation_counts',{})
    for op,_,_ in ops:counts[op]=counts.get(op,0)+1
    group='real_empty' if role=='real' and before==0 else 'fake_nonempty' if role=='fake' and before>0 else 'fake_empty' if role=='fake' else 'real_nonempty'
    row=stats.setdefault(group,dict(targets=0,before_fraction_sum=0.,after_fraction_sum=0.,became_empty=0))
    row['targets']+=1;row['before_fraction_sum']+=before;row['after_fraction_sum']+=after;row['became_empty']+=int(before>0 and after==0)
    row['before_fraction_mean']=row['before_fraction_sum']/row['targets'];row['after_fraction_mean']=row['after_fraction_sum']/row['targets']
