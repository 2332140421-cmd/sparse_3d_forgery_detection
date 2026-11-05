"""Six fixed real-train clips, RGB-confirmed background, CPU-only slice audit.

Direct caller this CLI; outputs selection/RGB contacts/JSON statistics. Reuses
native_dense; does not instantiate Video, a model, frontend or token extractor.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
import json
import hashlib
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
OUT=BASE/'depth_slice_cpu_audit_v1'
HS=[1/32,1/16,1/8,1/4]


def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selection(rows):
    result=[]
    for ds in ['ActivityForensics','ANAKIN']:
        rr=sorted([r for r in rows if r['dataset']==ds and r['split']=='train' and r['role']=='real'],key=lambda r:(r['source_id'],r['video_id']));seen=set()
        for r in rr:
            if r['source_id'] in seen:continue
            seen.add(r['source_id']);n=min(24,r['actual_frames']);lo=max(0,(r['actual_frames']-n)//2)
            result.append({'dataset':ds,'source_id':r['source_id'],'video_id':r['video_id'],'frame_range':[lo,lo+n],'row':r})
            if len(seen)==3:break
    return result


def slice_stats(depth,valid,h,origin,history):
    ids,count=np.unique(np.floor((depth[valid]-origin)/h).astype(np.int64),return_counts=True)
    temporal=sum(history.get(int(i),0) for i in ids);has_history=sum(int(i) in history for i in ids)
    for i in ids:history[int(i)]=history.get(int(i),0)+1
    n=len(ids);total=int(count.sum())
    return {'valid_points':total,'occupied':n,'id_min':int(ids[0]) if n else None,'id_max':int(ids[-1]) if n else None,
            'members_median':float(np.median(count)) if n else None,'members_P90':float(np.quantile(count,.9)) if n else None,'members_max':int(count.max()) if n else None,
            'max_share':int(count.max())/total if total else None,'top4_share':int(np.sort(count)[-4:].sum())/total if total else None,
            'temporal_edges':temporal,'depth_directed_edges':n*(n-1),'units_with_history':has_history}


def rgb_matches(rgb,a,b,roi):
    """LK with backward and photometric checks only; no depth used to accept."""
    H,W=rgb.shape[1:3];x0,y0,x1,y1=roi;mask=np.zeros((H,W),np.uint8);mask[int(y0*H):int(y1*H),int(x0*W):int(x1*W)]=255
    ga=cv2.cvtColor(np.asarray(rgb[a]),cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(np.asarray(rgb[b]),cv2.COLOR_RGB2GRAY)
    p=cv2.goodFeaturesToTrack(ga,200,.02,7,mask=mask,blockSize=7)
    if p is None:return {'a':a,'b':b,'detected':0,'matched':0,'points_a':[],'points_b':[]}
    q,s,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(21,21),maxLevel=3)
    back,t,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(21,21),maxLevel=3)
    good=s[:,0].astype(bool)&t[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<=.75)
    aa=np.rint(p[:,0]).astype(int);bb=np.rint(q[:,0]).astype(int)
    good&=(bb[:,0]>=max(3,int(x0*W)))&(bb[:,0]<min(W-3,int(x1*W)))&(bb[:,1]>=max(3,int(y0*H)))&(bb[:,1]<min(H-3,int(y1*H)))
    good&=(aa[:,0]>=3)&(aa[:,0]<W-3)&(aa[:,1]>=3)&(aa[:,1]<H-3)
    for k in np.flatnonzero(good):
        x,y=aa[k];u,v=bb[k];z=ga[y-3:y+4,x-3:x+4].astype(float).ravel();w=gb[v-3:v+4,u-3:u+4].astype(float).ravel();z-=z.mean();w-=w.mean();den=np.linalg.norm(z)*np.linalg.norm(w)
        if den==0 or float(z@w/den)<.9:good[k]=False
    return {'a':a,'b':b,'detected':len(p),'matched':int(good.sum()),'points_a':aa[good].tolist(),'points_b':bb[good].tolist()}


def prepare():
    if (OUT/'selection.json').exists():raise RuntimeError('Existing selection: inspect/reuse, do not replace')
    c=read(ROOT/'docs/pvst/config_v03.json');source=Path(c['asset_root']);ss=selection(read(source/'manifest.json')['rows'])
    manifest={'base_HEAD':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'rule':'each dataset first3 unique source real/train sorted(source_id,video_id); middle min24 continuous frames; no depth/scene replacement','manifest_sha256':sha(source/'manifest.json'),'scale_sha256':sha(source/'scale.json'),'scale':read(source/'scale.json')['s'],'config':{k:c[k] for k in ['asset_root','cache_root']},'samples':ss,'h_candidates':HS,'origins':['0','h/2'],'grid':'native remapped grid; interpolated points not independent depth measurements'}
    save(OUT/'selection.json',manifest)
    # Selection is durably written BEFORE RGB viewing and any depth arrays opened.
    for item in ss:
        try:
            r=item['row'];rgb=np.load(Path(c['cache_root'])/r['video_id']/'rgb.npy',mmap_mode='r');lo,hi=item['frame_range'];tiles=[]
            for i in range(lo,hi):
                f=cv2.cvtColor(np.asarray(rgb[i]),cv2.COLOR_RGB2BGR);f=cv2.resize(f,(240,160));cv2.putText(f,str(i),(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);tiles.append(f)
            while len(tiles)%6:tiles.append(np.zeros_like(tiles[0]))
            contact=np.vstack([np.hstack(tiles[j:j+6]) for j in range(0,len(tiles),6)]);cv2.imwrite(str(OUT/(r['video_id']+'_rgb.jpg')),contact)
        except Exception as e:item['prepare_error']=str(e)
    save(OUT/'selection.json',manifest)
    compact={k:v for k,v in manifest.items() if k!='samples'};compact['samples']=[{k:v for k,v in item.items() if k!='row'} for item in ss]
    if OUT==BASE/'depth_slice_cpu_audit_v1':save(ROOT/'docs/pvst/depth_slice_cpu_audit_v1_samples.json',compact)


def audit():
    from .data import native_dense
    started=time.monotonic();cv2.setNumThreads(1);manifest=read(OUT/'selection.json');review=read(OUT/'rgb_review.json');results=[]
    if (OUT/'results.json').exists():raise RuntimeError('Existing result: inspect/reuse, no automatic rerun')
    for item in manifest['samples']:
        r=item['row'];lo,hi=item['frame_range'];vid=r['video_id'];result={k:v for k,v in item.items() if k!='row'};result['review']=review[vid];results.append(result)
        try:
            cache=Path(manifest['config']['cache_root'])/vid;dense=Path(r['dense_source']);cm=read(cache/'complete.json');dm=read(dense/'complete.json')
            if cm['video_sha256']!=r['video_sha256'] or dm['video_sha256']!=r['video_sha256']:raise ValueError('CACHE_SHA_IDENTITY')
            pts=[p['pts_s'] for p in r['pts'][lo:hi]]
            if len(pts)!=hi-lo or not np.all(np.diff(pts)>0):raise ValueError('PTS_NOT_VERIFIED')
            for i in range(lo,hi):
                p=dm['frames'][i]
                if p['frame_index']!=i or abs(p['pts_s']-r['pts'][i]['pts_s'])>1e-8 or p.get('timestamp_source')!='NATIVE_PTS' or p['pts_ticks']!=r['pts'][i]['pts_ticks'] or p['time_base']!=r['pts'][i]['time_base']:raise ValueError('DENSE_PTS_IDENTITY')
            rgb=np.load(cache/'rgb.npy',mmap_mode='r')
            if list(rgb.shape[1:3])!=r['source_hw'] or len(rgb)!=r['actual_frames']:raise ValueError('NATIVE_RGB_SHAPE')
            result['identity']={'native_hw':r['source_hw'],'video_sha256':r['video_sha256'],'dense_complete_sha256':sha(dense/'complete.json'),'RGB_complete_sha256':sha(cache/'complete.json'),'PTS':pts,'timestamp_source':[p.get('timestamp_source') for p in dm['frames'][lo:hi]],'depth_axis':'+Z','normalization':'native_dense XYZ / fixed s once','scale':manifest['scale'],'unit':'normalized depth, not metres','grid':'native remapped pixels'}
            pairs=[]
            if review[vid]['comparable']:
                roi=review[vid]['roi'];pairs=[rgb_matches(rgb,i,i+1,roi) for i in range(lo,hi-1)]+[rgb_matches(rgb,lo,lo+(hi-lo)//2,roi),rgb_matches(rgb,lo,hi-1,roi)]
            save(OUT/(vid+'_RGB_matches.json'),pairs)  # RGB correspondence fixed BEFORE depths.
            depths={};configs={f'{h}:{shift}':{'h':h,'d0':h*shift,'frames':[],'pairs':[]} for h in HS for shift in [0.,.5]};hist={key:{} for key in configs}
            # Retain only the small ROI correspondences' per-frame depth values, not full fields.
            requests={i:set() for i in range(lo,hi)}
            for pair in pairs:
                for axis in ['a','b']:requests[pair[axis]].update(tuple(p) for p in pair['points_'+axis])
            for i in range(lo,hi):
                q,v=native_dense(r,i);d=q[...,2]/manifest['scale'];del q
                for key,conf in configs.items():conf['frames'].append({'frame':i,'pts_s':r['pts'][i]['pts_s'],**slice_stats(d,v,conf['h'],conf['d0'],hist[key])})
                depths[i]={p:(float(d[p[1],p[0]]),bool(v[p[1],p[0]])) for p in requests[i]}
            drift=[]
            for pair in pairs:
                a,b=pair['a'],pair['b'];values=[(depths[a][tuple(p)],depths[b][tuple(q)]) for p,q in zip(pair['points_a'],pair['points_b'])];good=[(x[0],y[0]) for x,y in values if x[1] and y[1]];g=np.asarray(good,float).reshape(-1,2)
                reliable=len(g)>=12 and pair['matched']/max(pair['detected'],1)>=.5
                record={'a':a,'b':b,'dt_s':r['pts'][b]['pts_s']-r['pts'][a]['pts_s'],'adjacent':b==a+1,'RGB_detected':pair['detected'],'RGB_matched':pair['matched'],'tracking_coverage':pair['matched']/pair['detected'] if pair['detected'] else None,'valid_matches':len(g),'depth_valid_coverage':len(g)/pair['matched'] if pair['matched'] else None,'reliable':reliable,'median_abs_depth_delta':float(np.median(np.abs(g[:,1]-g[:,0]))) if len(g) else None,'P95_abs_depth_delta':float(np.quantile(np.abs(g[:,1]-g[:,0]),.95)) if len(g) else None}
                if not (record['valid_matches']<=record['RGB_matched']<=record['RGB_detected']):raise ValueError('MATCH_DENOMINATOR')
                drift.append(record)
                for conf in configs.values():
                    cross=int((np.floor((g[:,0]-conf['d0'])/conf['h'])!=np.floor((g[:,1]-conf['d0'])/conf['h'])).sum()) if len(g) else 0
                    conf['pairs'].append({'a':a,'b':b,'adjacent':b==a+1,'reliable':reliable,'crossed':cross,'valid_match_denominator':len(g),'rate':cross/len(g) if len(g) else None})
            result.update(status='COMPLETE',drift_pairs=drift,configs=list(configs.values()),comparability='RGB-reviewed background; see reliable pair count' if pairs else 'unable: no confirmed static ROI')
        except Exception as e:result.update(status='FAILED',error=str(e))
        save(OUT/'partial_results.json',results)
    save(OUT/'results.json',{'samples':results,'elapsed_s':time.monotonic()-started,'CUDA_VISIBLE_DEVICES':''})


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('stage',choices=['prepare','audit']);a=p.parse_args();OUT=a.output;OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare()
    else:audit()
if __name__=='__main__':main()
