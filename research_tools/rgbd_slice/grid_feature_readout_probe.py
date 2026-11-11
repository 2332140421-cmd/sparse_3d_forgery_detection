"""Frozen prepool256 vs cached grid-C512 linear evidence diagnostic.

CLI caller only; outputs train-only linear coefficients and native valid-pixel AP.
Existing Video/native_sample/grouping/count_metrics own the research semantics.
No encoder, pooling, relation or formal entry behavior is modified.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .data import Video,save
from .experiment import REPO,git,state_hash
from .set_fit_check import build
from .model import native_sample
from .grid_localization import grouping,count_metrics
from .local_pixel_readout_probe import csvsave,read
from .slab_grid_experiment import table
from scripts.prepare_v8_assets import digest

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
BASE=ROOT/'pvst_rgb_partition_anakin_dev_v1'
GRID=ROOT/'pvst_slab_grid_localization_v1'
PASS=ROOT/'pvst_slab_grid_cache_resume_check_v1'
OUT=ROOT/'pvst_grid_feature_readout_probe_v1'
DOC=REPO/'docs/pvst/grid_feature_readout_probe_v1.md'
SEED=20260909
NAME='SLAB_GRID_CURRENT'


def sample_pixels(gt,valid,video_id,frame_index,split):
    if split!='train':raise ValueError('TRAIN_ONLY_SAMPLING')
    gt=np.asarray(gt,bool).reshape(-1);valid=np.asarray(valid,bool).reshape(-1)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{SEED}:{video_id}:{frame_index}'.encode()).digest()[:8],'little'))
    ids=[]
    for label in [0,1]:
        candidates=np.flatnonzero(valid&(gt==label))
        ids.append(rng.choice(candidates,128,replace=False) if len(candidates)>128 else candidates)
    ids=np.concatenate(ids)
    assert len(np.unique(ids))==len(ids) and valid[ids].all()
    return ids,gt[ids].astype(np.int64)


def pixel_units(points,index,selected):
    pos=np.searchsorted(points,selected)
    assert (pos<len(points)).all() and np.array_equal(points[pos],selected)
    return index[pos]


def grid_mean(z,index,n):
    assert len(z)==len(index) and np.array_equal(np.bincount(index,minlength=len(n)),n)
    return np.bincount(index,weights=z,minlength=len(n))/n


def progress(stage,**extra):
    save(OUT/'status.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),**extra})
    print(stage,json.dumps(extra),flush=True)


def fit_one(x,y,name):
    start=time.monotonic();scaler=StandardScaler();clf=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,class_weight='balanced')
    with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        xx=scaler.fit_transform(x);clf.fit(xx,y)
    problems=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    info={'n_iter':clf.n_iter_.tolist(),'warnings':problems,'seconds':time.monotonic()-start}
    save(OUT/f'{name}_convergence.json',info)
    if problems or int(clf.n_iter_.max())>=1000:raise ValueError(name+'_NOT_CONVERGED_NO_SEARCH')
    parameters={'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'scaler_var':scaler.var_.tolist(),'coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),**info}
    save(OUT/f'{name}_coefficients.json',parameters)
    ref=clf.decision_function(scaler.transform(x[:1024]));actual=linear_scores(x[:1024],parameters)
    np.testing.assert_allclose(actual,ref,atol=1e-5,rtol=1e-5)
    info['explicit_standardization_max_abs']=float(np.max(abs(ref-actual)))
    parameters.update(info);save(OUT/f'{name}_coefficients.json',parameters)
    return parameters,info


def linear_scores(x,parameters):
    # Match StandardScaler.transform on float32 input: subtract/divide in-place
    # with its trained float64 statistics, retaining float32 standardized data.
    # Avoid raw-feature folded weights: cancellation failed the original tolerance.
    xx=np.asarray(x,dtype=np.float32).copy()
    xx-=np.asarray(parameters['scaler_mean'],dtype=np.float64)
    xx/=np.asarray(parameters['scaler_scale'],dtype=np.float64)
    return xx@np.asarray(parameters['coef'],dtype=np.float32)+np.float32(parameters['intercept'])



@torch.no_grad()
def visual_scores(local,points,hw,parameters):
    pieces=[]
    for chunk in points.split(65536):
        # Exact native_sample used by frozen_pool; only linear arithmetic is FP32.
        with torch.autocast('cuda',dtype=torch.bfloat16):x=native_sample(local,chunk,hw)
        assert x.ndim==2 and x.shape[1]==256
        pieces.append(linear_scores(x.float().cpu().numpy(),parameters))
    return np.concatenate(pieces)


@torch.no_grad()
def scan(model,c,rows,clips,phase,coeff=None):
    features=[[],[]];labels=[];records=[];seen=set();start=time.monotonic()
    for row in rows:
        path=GRID/'cache'/f"{row['video_id']}.pt"
        cache=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        wanted=sorted({i for x in clips if x['video_id']==row['video_id'] for i in range(x['start'],x['end'])}) if phase=='train' else list(range(row['actual_frames']))
        assert sorted(cache)==wanted
        v=Video(row,c)
        try:
            for i in wanted:
                f=v.frame(i);entry=cache[i];q=entry[NAME]
                assert f['pts']==entry['pts']==row['pts'][i]['pts_s'] and list(f['gt'].shape)==row['source_hw']
                points,groups,L=grouping(f['xyz'],f['valid']);ids,index=groups[1]
                n=torch.bincount(index,minlength=len(ids));g=torch.bincount(index[f['gt'][f['valid']]],minlength=len(ids))
                assert torch.equal(ids,q['ids']) and torch.equal(n,q['counts']) and torch.equal(g,q['edit_counts'])
                assert q['features'].shape==(len(n),512) and q['features'].dtype==torch.float32
                assert torch.isfinite(q['features']).all() and int(n.sum())==entry['valid_pixels'] and int(g.sum())==entry['valid_GT_pixels'] and int(f['gt'].sum())==entry['GT_pixels']
                key=(row['video_id'],i);assert key not in seen;seen.add(key)
                if row['role']=='real':assert not f['gt'].any()
                with torch.autocast('cuda',dtype=torch.bfloat16):local,_=model.local.encode(f['rgb'],torch.zeros_like(f['depth']))
                assert local.ndim==3 and local.shape[0]==256
                rec={k:row[k] for k in ['video_id','pair_id','role','split']}
                rec.update(frame_index=i,pts=f['pts'],native_H=row['source_hw'][0],native_W=row['source_hw'][1],L=L,units=len(n),valid_pixels=int(n.sum()),valid_GT_pixels=int(g.sum()),GT_pixels=entry['GT_pixels'])
                if phase=='train':
                    selected,y=sample_pixels(f['gt'].numpy(),f['valid'].numpy(),row['video_id'],i,row['split'])
                    routing=pixel_units(points.numpy(),index.numpy(),selected)
                    with torch.autocast('cuda',dtype=torch.bfloat16):x=native_sample(local,torch.tensor(selected,device='cuda'),f['valid'].shape)
                    assert x.shape==(len(y),256)
                    a=x.float().cpu().numpy();b=q['features'][torch.from_numpy(routing)].numpy().copy()
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    features[0].append(a);features[1].append(b);labels.append(y)
                    rec.update(negative_samples=int((y==0).sum()),positive_samples=int(y.sum()),indices_sha256=hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest())
                else:
                    prepool,grid=coeff
                    z=visual_scores(local,points.cuda(),f['valid'].shape,prepool)
                    unit_z=grid_mean(z,index.numpy(),n.numpy())
                    cz=linear_scores(q['features'].numpy(),grid)
                    assert np.isfinite(z).all() and np.isfinite(cz).all()
                    gt=f['gt'][f['valid']].numpy()
                    rec.update(PREPOOL_PIXEL=float(average_precision_score(gt,z)) if gt.any() else None,PREPOOL_TO_GRID=count_metrics(unit_z,n.numpy(),g.numpy())['pixel_AP'],GRID_C_LINEAR=count_metrics(cz,n.numpy(),g.numpy())['pixel_AP'])
                records.append(rec)
                del local,f,entry,q,points,groups,ids,index,n,g
                if len(records)%24==0:progress(phase,completed_frames=len(records),video_id=row['video_id'],frame_index=i)
        finally:v.close()
        del cache
        csvsave(OUT/f'{phase}_frames.csv',records)
        progress(phase+'_video_complete',video_id=row['video_id'],frames=len(wanted),completed_frames=len(records))
    return features,labels,records,seen,time.monotonic()-start


def main():
    start=time.monotonic();torch.set_num_threads(4)
    assert git('branch','--show-current')=='pvst-grid-feature-readout-probe'
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt'
    if (OUT/'summary.json').exists():
        s=read(OUT/'summary.json')
        if s.get('status')=='COMPLETE' and s['identity']['weight_sha256']==digest(weight) and s['identity']['manifest_sha256']==digest(BASE/'manifest.json'):
            print('REUSE COMPLETE');return
        raise ValueError('EXISTING_OUTPUT_NO_RETRY')
    assert not OUT.exists() or set(p.name for p in OUT.iterdir())<={'execution.log','command.txt'}
    OUT.mkdir(exist_ok=True);progress('identity')
    c=read(BASE/'config.json');m=read(BASE/'manifest.json');baseline=read(BASE/'identity.json');old=read(GRID/'identity.json');passed=read(PASS/'summary.json');formal=read(GRID/'summary.json')
    assert passed['cache_status']=='PASS' and formal['status']=='COMPLETE' and formal['formal_training']
    for name,key in [('manifest.json','manifest_sha256'),('config.json','config_sha256')]:assert digest(BASE/name)==old[key]==passed['identity'][key]
    assert digest(weight)==old['weight_sha256']==passed['identity']['weight_sha256']
    for path,sha in baseline['code'].items():assert digest(REPO/path)==sha
    for path in ['research_tools/rgbd_slice/grid_localization.py','research_tools/rgbd_slice/slab_grid_experiment.py']:
        expected=read(GRID/'formal_identity.json')['model_code_sha256' if path.endswith('grid_localization.py') else 'update_eval_source_sha256'];assert digest(REPO/path)==expected
    files=passed['extraction']['files']
    for name,sha in files.items():assert digest(GRID/'cache'/name)==sha
    train=[r for r in m['rows'] if r['split']=='train'];dev=[r for r in m['rows'] if r['split']=='dev'];clips=m['clips']
    assert len(train)==16 and len(dev)==8 and len(clips)==48
    expected={(x['video_id'],i) for x in clips for i in range(x['start'],x['end'])};assert len(expected)==1152
    assert {r['source_id'] for r in train}.isdisjoint({r['source_id'] for r in dev})
    for row in train+dev:
        assert digest(Path(row['video_path']))==row['video_sha256']
        if row['role']=='fake':assert digest(Path(row['mask_zip']))==row['mask_sha256']
        for path,sha in next(a for a in baseline['assets'] if a['video_id']==row['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha
    d=read(GRID/NAME/'final_dev.json');cp=torch.load(GRID/NAME/'final.pt',map_location='cpu',weights_only=True)
    assert cp['condition']==NAME and cp['updates']==576 and cp['epoch']==12 and cp['identity']==read(GRID/'formal_identity.json')
    assert cp['initial_sha256']==read(GRID/'formal_initialization.json')['state_dict_sha256'];del cp
    assert d['frames']==924 and len(d['frame_records'])==924
    dframes={(r['video_id'],r['frame_index']):r for r in d['frame_records']};assert len(dframes)==924
    identity={'execution_head':git('rev-parse','HEAD'),'base_head':'e4b3b02e337527bfe4b74216199f0c16dd08494f','weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'config_sha256':digest(BASE/'config.json'),'grid_cache_sha256':files,'reference_final_dev_sha256':digest(GRID/NAME/'final_dev.json'),'reference_final_weight_sha256':digest(GRID/NAME/'final.pt'),'sampling':'SHA256(seed,video,frame) RNG; up to128/class, native geometric valid; same indices both models','seed':SEED,'feature':'Local.encode [256,h,w] -> native_sample; C512 read only from verified cache','fit':{'C':1,'solver':'lbfgs','max_iter':1000,'class_weight':'balanced','scaler':'train only'},'GPU':torch.cuda.get_device_name(0),'initial_free_total_bytes':list(torch.cuda.mem_get_info())}
    save(OUT/'identity.json',identity)
    model=build(c,OUT,rgb_only=True).eval();cp=torch.load(weight,map_location='cpu',weights_only=True)
    assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==baseline['identity']
    assert cp['initial_model_sha256']==read(OUT/'initialization.json')['model_sha256'];model.load_state_dict(cp['state_dict'],strict=True);del cp
    model.requires_grad_(False);before=state_hash(model.state_dict());torch.cuda.reset_peak_memory_stats()
    features,labels,tr,seen,train_s=scan(model,c,train,clips,'train');assert seen==expected
    y=np.concatenate(labels);x=np.concatenate(features[0]);del features[0][:]
    prepool,fit_a=fit_one(x,y,'PREPOOL_LINEAR');del x
    x=np.concatenate(features[1]);del features,labels
    grid,fit_c=fit_one(x,y,'GRID_C_LINEAR');del x
    sampling={'frames':len(tr),'negative_samples':int((y==0).sum()),'positive_samples':int(y.sum()),'same_pixel_indices_and_labels':True,'videos':[]}
    for row in train:
        rr=[r for r in tr if r['video_id']==row['video_id']];sampling['videos'].append({'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frames':len(rr),'negative_samples':sum(r['negative_samples'] for r in rr),'positive_samples':sum(r['positive_samples'] for r in rr)})
    del y
    coeff=(prepool,grid)
    _,_,dr,seen,dev_s=scan(model,c,dev,clips,'dev',coeff)
    assert len(seen)==924 and seen==set(dframes)
    for r in dr:
        ref=dframes[(r['video_id'],r['frame_index'])]
        for k in ['pts','valid_pixels','valid_GT_pixels']:assert r[k]==ref[k]
    assert sum(r['GT_pixels']>0 for r in dr)==462
    assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    for name,sha in files.items():assert digest(GRID/'cache'/name)==sha
    assert digest(weight)==identity['weight_sha256'] and digest(GRID/NAME/'final_dev.json')==identity['reference_final_dev_sha256'] and digest(GRID/NAME/'final.pt')==identity['reference_final_weight_sha256']
    result=[]
    for row in dev:
        if row['role']!='fake':continue
        rr=[r for r in dr if r['video_id']==row['video_id'] and r['PREPOOL_PIXEL'] is not None];ref=next(r for r in d['records'] if r['video_id']==row['video_id'])
        entry={'pair_id':row['pair_id'],'video_id':row['video_id'],'positive_valid_GT_frames':len(rr),'GT_coverage':ref['GT_coverage']}
        for k in ['PREPOOL_PIXEL','PREPOOL_TO_GRID','GRID_C_LINEAR']:entry[k]=float(np.mean([r[k] for r in rr]))
        entry['GRID_RELATION']=ref['pixel_AP'];entry.update(B_minus_A=entry['PREPOOL_TO_GRID']-entry['PREPOOL_PIXEL'],C_minus_B=entry['GRID_C_LINEAR']-entry['PREPOOL_TO_GRID'],D_minus_C=entry['GRID_RELATION']-entry['GRID_C_LINEAR']);result.append(entry)
    csvsave(OUT/'video_AP.csv',result);keys=['PREPOOL_PIXEL','PREPOOL_TO_GRID','GRID_C_LINEAR','GRID_RELATION','B_minus_A','C_minus_B','D_minus_C']
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'sampling':sampling,'train_frames':1152,'dev_frames':924,'dev_fake_frames':462,'fits':{'PREPOOL_LINEAR':fit_a,'GRID_C_LINEAR':fit_c},'video_AP':result,'macro':{k:float(np.mean([r[k] for r in result])) for k in keys},'train_extract_seconds':train_s,'dev_seconds':dev_s,'wall_seconds':time.monotonic()-start,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'model_and_original_assets_unchanged':True,'unique_scoring':True}
    save(OUT/'summary.json',summary)
    lines=['# 方格池化前后冻结特征线性读出诊断',table(['pair','A池化前像素','B像素均值到格','C缓存格线性','D正式格关系','B−A','C−B','D−C'],[[r['pair_id']]+[r[k] for k in keys] for r in result]),'宏平均：'+json.dumps(summary['macro'],ensure_ascii=False),'A为Local.encode真正送入池化的256维特征，经原native_sample在原生像素采样；不是32维Local.native。B仅对A在完整原单元有效像素取均值，不新增拟合。C是已PASS缓存C512，不重算池化，不读取位置、时间或其他格。D引用同一正式实验576步final_dev，不重新运行关系模型。','两模型共用训练1152唯一帧同一采样索引和GT；每类最多128不放回，负样本含real及fake背景。StandardScaler和L2逻辑回归只读训练；固定C1/lbfgs/1000/balanced，无调参。采样摘要：'+json.dumps(sampling,ensure_ascii=False),'开发8视频924唯一帧，462fake正GT帧；与正式逐帧身份、PTS、n/g及有效像素范围一致。每帧只计算一次，无尾部上下文重复计分；冻结编码为逐帧路径，不读取旧关系/GRU/像素头。AP在全部原生几何有效像素上计算，先帧→视频→四fake宏平均。无有效编辑GT不填0。不报告不同读出零阈值Dice/FPR。','B/C同粒度；C/D读出与训练不同，其差值不是纯关系消融。该诊断只测本次线性可读性，不证明特征完全没有信息，也不能外推泛化或三维独立贡献。','拟合与显式标准化FP32读出：'+json.dumps(summary['fits'],ensure_ascii=False),'资源：'+json.dumps({k:summary[k] for k in ['train_extract_seconds','dev_seconds','wall_seconds','peak_allocated','peak_reserved','peak_RSS']},ensure_ascii=False),'模型参数/缓冲和原缓存SHA前后核对不变；退出码0。执行HEAD '+identity['execution_head']+'；产物 '+str(OUT)]
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);DOC.write_text(report);save(OUT/'status.json',{'status':'COMPLETE','stage':'report_complete','exit_code':0});(OUT/'exit_code.txt').write_text('0\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():
            save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()});(OUT/'exit_code.txt').write_text('1\n')
        raise
