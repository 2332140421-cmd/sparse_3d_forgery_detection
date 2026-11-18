"""CPU cached-C probe: current XY feature vs same feature plus H=3 change.

CLI caller. Only restores native routing from frozen geometry and fits two
count-weighted linear classifiers. No encoder, frontend or model forward runs.
"""
import csv
import json
import os
from pathlib import Path
import time
import traceback
import warnings
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .data import Video,save
from .grid_localization import grouping,count_metrics
from .experiment import git
from scripts.prepare_v8_assets import digest

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
BASE=ROOT/'pvst_rgb_partition_anakin_dev_v1'
GRID=ROOT/'pvst_slab_grid_localization_v1'
OUT=ROOT/'pvst_anakin_xy_history_probe_v1'
NAME='SLAB_GRID_CURRENT'


def read(p):return json.loads(p.read_text())
def csvsave(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collapse(C,index,n,g):
    """Count conservation and weighted F within a native XY tile."""
    size=int(index.max())+1;counts=np.bincount(index,weights=n,minlength=size)
    edits=np.bincount(index,weights=g,minlength=size)
    sums=np.zeros((size,C.shape[1]),np.float64);np.add.at(sums,index,C*n[:,None])
    assert counts.sum()==n.sum() and edits.sum()==g.sum() and (counts>0).all()
    return (sums/counts[:,None]).astype(np.float32),counts.astype(np.int64),edits.astype(np.int64)


def window(frames,indices):
    common=sorted(set.intersection(*(set(frames[i]['keys']) for i in indices)))
    maps=[{k:j for j,k in enumerate(frames[i]['keys'])} for i in indices]
    pos=[np.array([m[k] for k in common],dtype=np.int64) for m in maps]
    x=[frames[i]['F'][p].astype(np.float64) for i,p in zip(indices,pos)]
    norms=[np.linalg.norm(a,axis=1) for a in x];keep=np.logical_and.reduce([n>0 for n in norms])
    unit=[a[keep]/n[keep,None] for a,n in zip(x,norms)]
    history=sum(1-np.sum(a*b,axis=1) for a,b in zip(unit[:-1],unit[1:]))
    return pos[-1][keep],history,{'shared_before_zero':len(common),'zero_removed':int((~keep).sum())}


def weighted_rows(x,n,g):
    """Equivalent to all native pixels, without expanding pixel features."""
    positive=g>0;negative=(n-g)>0
    return np.concatenate([x[negative],x[positive]]),np.r_[np.zeros(negative.sum()),np.ones(positive.sum())].astype(np.int64),np.r_[(n-g)[negative],g[positive]]


def prepare(manifest,config):
    train=[];dev=[];coverage=[];hashes={};started=time.monotonic()
    for row in manifest['rows']:
        assert row['split'] in ['train','dev']
        path=GRID/'cache'/f"{row['video_id']}.pt";hashes[str(path)]=digest(path)
        cache=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        ranges=[(x['start'],x['end']) for x in manifest['clips'] if x['video_id']==row['video_id']] if row['split']=='train' else [(0,row['actual_frames'])]
        wanted=sorted({i for a,b in ranges for i in range(a,b)})
        assert sorted(cache)==wanted
        video=Video(row,config);frames={}
        try:
            for i in wanted:
                f=video.frame(i);e=cache[i];q=e[NAME];assert e['pts']==f['pts']==row['pts'][i]['pts_s']
                points,groups,L=grouping(f['xyz'],f['valid']);ids,index=groups[1]
                n=torch.bincount(index,minlength=len(ids));g=torch.bincount(index[f['gt'][f['valid']]],minlength=len(ids))
                assert torch.equal(ids,q['ids']) and torch.equal(n,q['counts']) and torch.equal(g,q['edit_counts'])
                assert int(n.sum())==e['valid_pixels'] and int(g.sum())==e['valid_GT_pixels'] and int(f['gt'].sum())==e['GT_pixels']
                nx=(row['source_hw'][1]+L-1)//L
                p=points.numpy();xy=(p//row['source_hw'][1]//L)*nx+(p%row['source_hw'][1]//L)
                # Recover each cached unit's XY from a member pixel; then check
                # ALL member pixels agree, not just the representative.
                routing=index.numpy();unit_xy=np.full(len(ids),-1,np.int64);unit_xy[routing]=xy
                assert np.array_equal(unit_xy[routing],xy)
                unique,inv=np.unique(unit_xy,return_inverse=True)
                C=q['features'].numpy();assert C.shape==(len(ids),512) and np.isfinite(C).all()
                F,nn,gg=collapse(C,inv,n.numpy(),g.numpy())
                frames[i]={'keys':unique.tolist(),'F':F,'n':nn,'g':gg,'GT':e['GT_pixels'],'valid_GT':e['valid_GT_pixels'],'valid':e['valid_pixels'],'pts':e['pts'],'L':L}
        finally:video.close()
        seen=set()
        for lo,hi in ranges:
            for i in range(lo,hi):
                assert (row['video_id'],i) not in seen;seen.add((row['video_id'],i))
                if i-lo<3:
                    coverage.append({'split':row['split'],'pair':row['pair_id'],'role':row['role'],'video_id':row['video_id'],'frame':i,'pts':frames[i]['pts'],'status':'WARMUP','units':0,'zero_removed':0,'full_GT':frames[i]['GT'],'geometry_GT':frames[i]['valid_GT'],'scored_GT':0,'geometry_pixels':frames[i]['valid'],'scored_pixels':0});continue
                indices=list(range(i-3,i+1));p,h,detail=window(frames,indices)
                f=frames[i];n=f['n'][p];g=f['g'][p]
                record={'split':row['split'],'pair':row['pair_id'],'role':row['role'],'video_id':row['video_id'],'frame':i,'pts':f['pts'],'status':'SCORED' if len(p) else 'NO_COMMON','units':len(p),'zero_removed':detail['zero_removed'],'full_GT':f['GT'],'geometry_GT':f['valid_GT'],'scored_GT':int(g.sum()),'geometry_pixels':f['valid'],'scored_pixels':int(n.sum())}
                coverage.append(record)
                if len(p):
                    item={'x':np.c_[f['F'][p],h].astype(np.float32),'n':n,'g':g,'record':record}
                    (train if row['split']=='train' else dev).append(item)
        print('prepared',row['split'],row['pair_id'],row['role'],len(wanted),flush=True)
        del frames,cache
    csvsave(OUT/'coverage.csv',coverage);save(OUT/'cache_hashes.json',hashes)
    return train,dev,coverage,hashes,time.monotonic()-started


def fit(train,name,width):
    start=time.monotonic();x=np.concatenate([e['x'][:,:width] for e in train]);n=np.concatenate([e['n'] for e in train]);g=np.concatenate([e['g'] for e in train])
    scaler=StandardScaler().fit(x,sample_weight=n);z=scaler.transform(x)
    xx,y,pixels=weighted_rows(z,n,g);P=int(g.sum());N=int((n-g).sum());assert P>0 and N>0
    weights=pixels*np.where(y==1,(P+N)/(2*P),(P+N)/(2*N))
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,class_weight=None,random_state=20260909)
    with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(xx,y,sample_weight=weights)
    errors=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    record={'name':name,'width':width,'C':1,'solver':'lbfgs','max_iter':1000,'class_weight':'pixel-balanced via explicit sample_weight; no double balancing','positive_pixels':P,'negative_pixels':N,'rows':len(x),'n_iter':model.n_iter_.tolist(),'warnings':errors,'seconds':time.monotonic()-start,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}
    save(OUT/f'{name}_coefficients.json',record)
    if errors or model.n_iter_.max()>=1000:raise RuntimeError(name+'_NOT_CONVERGED_NO_SEARCH')
    return scaler,model,record


def evaluate(data,scaler,model,width,name):
    frames=[]
    for e in data:
        z=model.decision_function(scaler.transform(e['x'][:,:width]));m=count_metrics(z,e['n'],e['g'])
        frames.append({**e['record'],'pixel_AP':m['pixel_AP'],'real_FPR':m['FPR'] if e['record']['role']=='real' else None})
    videos=[]
    for vid in dict.fromkeys(f['video_id'] for f in frames):
        fs=[f for f in frames if f['video_id']==vid];ap=[f['pixel_AP'] for f in fs if f['pixel_AP'] is not None];fp=[f['real_FPR'] for f in fs if f['real_FPR'] is not None]
        videos.append({'condition':name,'video_id':vid,'pair':fs[0]['pair'],'role':fs[0]['role'],'frames':len(fs),'positive_valid_GT_frames':len(ap),'pixel_AP':float(np.mean(ap)) if ap else None,'real_FPR':float(np.mean(fp)) if fp else None})
    csvsave(OUT/f'{name}_dev_frames.csv',frames);csvsave(OUT/f'{name}_dev_videos.csv',videos)
    return videos


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(4);start=time.monotonic()
    if OUT.exists():raise RuntimeError('Existing result preserved; no duplicate run')
    OUT.mkdir();save(OUT/'status.json',{'status':'RUNNING','pid':os.getpid()})
    manifest=read(BASE/'manifest.json');config=read(BASE/'config.json');old=read(GRID/'identity.json');passed=read(ROOT/'pvst_slab_grid_cache_resume_check_v1/summary.json')
    assert digest(BASE/'manifest.json')==old['manifest_sha256'] and digest(BASE/'config.json')==old['config_sha256']
    assert len(manifest['rows'])==24 and len(manifest['clips'])==48
    train_sources={r['source_id'] for r in manifest['rows'] if r['split']=='train'};dev_sources={r['source_id'] for r in manifest['rows'] if r['split']=='dev'};assert train_sources.isdisjoint(dev_sources)
    identity={'execution_head':git('rev-parse','HEAD'),'manifest_sha256':digest(BASE/'manifest.json'),'config_sha256':digest(BASE/'config.json'),'frozen_C_encoder_sha256':old['weight_sha256'],'H':3,'readout_A':'F_t512','readout_B':'same F_t512 plus sum of three per-tile adjacent cosine distances','fit':'C1 lbfgs max_iter1000 weighted StandardScaler and pixel-balanced L2 logistic regression','scoring':'same shared nonzero XY cells, current geometry-valid pixels; logit>=0 fixed'};save(OUT/'identity.json',identity)
    for name,sha in passed['extraction']['files'].items():assert digest(GRID/'cache'/name)==sha
    train,dev,coverage,hashes,prep_s=prepare(manifest,config)
    fits={};results={}
    for name,width in [('A',512),('B',513)]:
        print('fit',name,flush=True);scaler,model,record=fit(train,name,width);fits[name]={k:v for k,v in record.items() if k not in ['scaler_mean','scaler_scale','coef','intercept']};results[name]=evaluate(dev,scaler,model,width,name)
    comparison=[]
    for a,b in zip(results['A'],results['B']):
        assert (a['video_id'],a['frames'])==(b['video_id'],b['frames'])
        comparison.append({'pair':a['pair'],'role':a['role'],'frames':a['frames'],'AP_A':a['pixel_AP'],'AP_B':b['pixel_AP'],'AP_delta':b['pixel_AP']-a['pixel_AP'] if a['pixel_AP'] is not None else None,'FPR_A':a['real_FPR'],'FPR_B':b['real_FPR']})
    macros={name:float(np.mean([v['pixel_AP'] for v in vs if v['role']=='fake' and v['pixel_AP'] is not None])) for name,vs in results.items()}
    coverage_summary={}
    for split in ['train','dev']:
        rows=[r for r in coverage if r['split']==split];scored=[r for r in rows if r['status']=='SCORED'];den=sum(r['full_GT'] for r in scored)
        coverage_summary[split]={'source_frames':len(rows),'scored_frames':len(scored),'warmup_frames':sum(r['status']=='WARMUP' for r in rows),'no_common_frames':sum(r['status']=='NO_COMMON' for r in rows),'GT_coverage':sum(r['scored_GT'] for r in scored)/den if den else None,'pixel_coverage_of_geometry':sum(r['scored_pixels'] for r in scored)/sum(r['geometry_pixels'] for r in scored)}
    assert hashes=={p:digest(Path(p)) for p in hashes}
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'macro_AP':macros,'macro_delta':macros['B']-macros['A'],'comparison':comparison,'coverage':coverage_summary,'fits':fits,'prepare_seconds':prep_s,'seconds':time.monotonic()-start,'new_depth':0,'new_features':0,'model_forwards':0,'cache_unchanged':True}
    csvsave(OUT/'comparison.csv',comparison);save(OUT/'summary.json',summary);save(OUT/'status.json',{'status':'COMPLETE','exit_code':0});print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',{'status':'FAILED','exit_code':1})
        raise
