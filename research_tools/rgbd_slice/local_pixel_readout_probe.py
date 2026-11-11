"""One frozen native-feature linear readout, train-only sampling and dev AP.

CLI caller: this diagnostic. Outputs small coefficients, identities and AP tables;
existing evaluate owns video windows/GT/AP. No model or training entry changes.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback
import warnings

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import Video,save
from .experiment import REPO,evaluate,git,state_hash,average_precision_score
from .set_fit_check import build
from .pixel_message_dev_check import pixel_message_mode,table
from scripts.prepare_v8_assets import digest

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1')
PRIOR=BASE.parent/'pvst_slab_pooling_retention_check_v1'
OUT=BASE.parent/'pvst_local_pixel_readout_probe_v1'
DOC='docs/pvst/local_pixel_readout_probe_v1.md'
SEED=20260909


def read(p):return json.loads(Path(p).read_text())
def csvsave(p,rows):
    if rows:
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sample_indices(gt,video_id,frame_index,split):
    if split!='train':raise ValueError('SAMPLING_REQUIRES_TRAIN')
    key=f'{SEED}:{video_id}:{frame_index}'.encode();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
    flat=np.asarray(gt,dtype=bool).reshape(-1);parts=[]
    for label in [0,1]:
        ids=np.flatnonzero(flat==label);parts.append(rng.choice(ids,size=128,replace=False) if len(ids)>128 else ids)
    ids=np.concatenate(parts);return ids,flat[ids].astype(np.int64)


def feature_pixels(x,hw,ids=None):
    if tuple(x.shape)!=(1,32,*hw):raise ValueError('NATIVE_FEATURE_GT_SHAPE:'+str((tuple(x.shape),hw)))
    flat=x[0].permute(1,2,0).reshape(-1,32)
    return flat if ids is None else flat[torch.as_tensor(ids,device=x.device)]


def folded(scaler,clf):
    w=clf.coef_[0]/scaler.scale_;b=float(clf.intercept_[0]-np.dot(w,scaler.mean_))
    return w.astype(np.float32),np.float32(b)


def scored_range(fs,n,clips,video_id):
    lo=fs[0]['frame_index'];end=fs[-1]['frame_index']+1
    ranges=[(x['start'],x['end'],x['start']) for x in clips if x['video_id']==video_id] if clips is not None else [(s,min(s+24,n),max(0,n-24) if s+24>=n else s) for s in range(0,n,24)]
    selected=[(s,e) for s,e,l in ranges if l==lo and e==end]
    if len(selected)!=1:raise ValueError('WINDOW_IDENTITY')
    return selected[0]


class NativeReadout:
    def __init__(self,model,rows,clips=None,coeff=None):
        self.model=model;self.rows={r['video_id']:r for r in rows};self.clips=clips;self.coeff=coeff;self.current=None;self.seen=set();self.features=[];self.labels=[];self.records=[];self.started=time.monotonic()
    def progress(self,stage,**extra):
        if 'video' in extra:self.current=self.rows[extra['video']]
        save(OUT/'status.json',{'status':'RUNNING','stage':'sample_train' if self.coeff is None else 'evaluate_dev','pid':os.getpid(),**extra});print(stage,extra,flush=True)
    def before(self,module,args):
        self.fs=args[0];self.ordinal=0;self.start,self.end=scored_range(self.fs,self.current['actual_frames'],self.clips,self.current['video_id'])
    def native(self,module,args,output):
        f=self.fs[self.ordinal];self.ordinal+=1
        assert not torch.is_grad_enabled() and not self.model.training
        hw=tuple(f['gt'].shape);features=feature_pixels(output,hw);i=f['frame_index'];r=self.current
        if not self.start<=i<self.end:return
        assert f['pts']==r['pts'][i]['pts_s'];key=(r['video_id'],i)
        if key in self.seen:raise ValueError('DUPLICATE_SCORED_FRAME')
        self.seen.add(key);gt=f['gt'].numpy()
        if r['role']=='real':assert not gt.any()
        record={'video_id':r['video_id'],'pair_id':r['pair_id'],'role':r['role'],'split':r['split'],'frame_index':i,'pts':f['pts'],'GT_positive_pixels':int(gt.sum()),'native_pixels':gt.size}
        if self.coeff is None:
            ids,y=sample_indices(gt,r['video_id'],i,r['split']);x=feature_pixels(output,hw,ids).float().cpu().numpy()
            assert np.isfinite(x).all() and np.array_equal(gt.reshape(-1)[ids],y.astype(bool))
            self.features.append(x);self.labels.append(y);record.update(negative_samples=int((y==0).sum()),positive_samples=int(y.sum()),indices_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest())
        else:
            # BF16 main forward remains unchanged. Only linear dot/bias use FP32.
            w,b=self.coeff
            with torch.autocast('cuda',enabled=False):
                z=torch.cat([chunk.float().matmul(w)+b for chunk in features.split(65536)]).cpu().numpy()
            assert np.isfinite(z).all()
            record['linear_pixel_ap']=float(average_precision_score(gt.reshape(-1),z)) if gt.any() else None
        self.records.append(record)
    def run(self,c):
        handles=[self.model.register_forward_pre_hook(self.before),self.model.local.native.register_forward_hook(self.native)]
        try:
            with pixel_message_mode(self.model,True) as projection:
                result=evaluate(self.model,list(self.rows.values()),c,self.progress,clip_list=self.clips,pixel_ap=self.coeff is not None)
                assert projection['delivered_nonzero_elements']==0
        finally:
            for h in handles:h.remove()
            self.fs=[]
        return result,projection


def main():
    assert git('branch','--show-current')=='pvst-local-pixel-readout-probe'
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt';identity=read(BASE/'identity.json');prior=read(PRIOR/'summary.json')
    if OUT.exists():
        s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if s.get('status')=='COMPLETE' and s.get('weight_sha256')==digest(weight) and s.get('manifest_sha256')==digest(BASE/'manifest.json'):print('REUSE COMPLETE',flush=True);return
        raise ValueError('EXISTING_OUTPUT_INSPECT_NO_RETRY')
    assert not git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice')
    OUT.mkdir();started=time.monotonic();torch.set_num_threads(4)
    c=read(BASE/'config.json');manifest=read(BASE/'manifest.json');train=[r for r in manifest['rows'] if r['split']=='train'];dev=[r for r in manifest['rows'] if r['split']=='dev'];clips=manifest['clips']
    assert digest(weight)==prior['weight_sha256'] and digest(BASE/'manifest.json')==identity['frozen_manifest_sha256'] and digest(BASE/'config.json')==identity['config_sha256']
    for path,sha in identity['code'].items():assert digest(REPO/path)==sha
    assert len(train)==16 and len(dev)==8 and len(clips)==48
    expected={(x['video_id'],i) for x in clips for i in range(x['start'],x['end'])};assert len(expected)==sum(x['end']-x['start'] for x in clips)==1152
    assert {x['video_id'] for x in clips}=={r['video_id'] for r in train}
    assert {r['source_id'] for r in train}.isdisjoint({r['source_id'] for r in dev})
    for r in train+dev:
        assert digest(Path(r['video_path']))==r['video_sha256']
        if r['role']=='fake':assert digest(Path(r['mask_zip']))==r['mask_sha256']
        for path,sha in next(a for a in identity['assets'] if a['video_id']==r['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha
        v=Video(r,c);v.close()
    save(OUT/'identity.json',{'execution_head':git('rev-parse','HEAD'),'weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'baseline_identity':identity['identity'],'config_sha256':digest(BASE/'config.json'),'checkpoint':{'condition':'RGB_DEPTH_PARTITION','epoch':12,'updates':576},'sampling':{'seed':SEED,'per_class_per_unique_frame_max':128,'replacement':False,'selection':'SHA256(seed,video,frame) RNG; all native pixels, no valid/score filter'},'linear':{'C':1,'solver':'lbfgs','max_iter':1000,'class_weight':'balanced','standard_scaler':'train samples only'},'feature':'Local.native output [1,32,H,W], before +context','sklearn_version':sklearn.__version__})
    model=build(c,OUT,rgb_only=True).eval();cp=torch.load(weight,map_location='cpu',weights_only=True)
    assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==identity['identity']
    assert read(OUT/'initialization.json')['model_sha256']==cp['initial_model_sha256'];model.load_state_dict(cp['state_dict'],strict=True);del cp
    before=state_hash(model.state_dict());torch.cuda.reset_peak_memory_stats();sample_start=time.monotonic()
    capture=NativeReadout(model,train,clips);train_result,train_projection=capture.run(c);assert capture.seen==expected
    csvsave(OUT/'training_sampling.csv',capture.records);x=np.concatenate(capture.features);y=np.concatenate(capture.labels);sample_records=capture.records
    del capture.features,capture.labels;sample_seconds=time.monotonic()-sample_start
    assert len(x)==len(y) and set(y)=={0,1};sample_hash=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest();fit_start=time.monotonic()
    scaler=StandardScaler();clf=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,class_weight='balanced')
    with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);xx=scaler.fit_transform(x);clf.fit(xx,y)
    convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    if convergence or int(clf.n_iter_.max())>=1000:
        save(OUT/'convergence.json',{'status':'NOT_CONVERGED','n_iter':clf.n_iter_.tolist(),'warnings':convergence});raise ValueError('LINEAR_NOT_CONVERGED_NO_SEARCH')
    w,b=folded(scaler,clf);ref=clf.decision_function(scaler.transform(x[:1024]));actual=x[:1024]@w+b;error=float(np.max(abs(ref-actual)));np.testing.assert_allclose(actual,ref,atol=1e-5,rtol=1e-5)
    fit_seconds=time.monotonic()-fit_start
    save(OUT/'linear_coefficients.json',{'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'scaler_var':scaler.var_.tolist(),'logreg_coefficient':clf.coef_[0].tolist(),'logreg_intercept':clf.intercept_.tolist(),'folded_weight_fp32':w.tolist(),'folded_bias_fp32':float(b),'classes':clf.classes_.tolist(),'n_iter':clf.n_iter_.tolist(),'fold_check_max_abs':error})
    sampling={'frames':len(sample_records),'negative_samples':int((y==0).sum()),'positive_samples':int(y.sum()),'sample_feature_label_sha256':sample_hash,'feature_bytes':x.nbytes,'seconds':sample_seconds,'fit_seconds':fit_seconds,'n_iter':clf.n_iter_.tolist(),'converged':True,'videos':[]}
    for r in train:
        rr=[a for a in sample_records if a['video_id']==r['video_id']];sampling['videos'].append({k:r[k] for k in ['video_id','pair_id','role']}|{'frames':len(rr),'negative_samples':sum(a['negative_samples'] for a in rr),'positive_samples':sum(a['positive_samples'] for a in rr)})
    save(OUT/'sampling_summary.json',sampling);del x,y,xx,capture,train_result,scaler,clf
    assert state_hash(model.state_dict())==before
    coeff=(torch.tensor(w,device='cuda'),torch.tensor(b,device='cuda'));dev_start=time.monotonic();capture=NativeReadout(model,dev,coeff=coeff);existing,dev_projection=capture.run(c)
    assert len(capture.seen)==len(capture.records)==924 and sum(bool(r['GT_positive_pixels']) for r in capture.records)==462
    assert capture.seen=={(r['video_id'],i) for r in dev for i in range(r['actual_frames'])}
    assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters());assert not model.sets.capture and not model.sets.zero_messages
    csvsave(OUT/'dev_frame_AP.csv',capture.records);records=[];old={r['video_id']:r for r in prior['spatial_by_video'] if r['mode']=='PIXEL_CONTEXT_ZERO'}
    for r in existing['records']:
        if r['role']!='fake':continue
        rr=[a for a in capture.records if a['video_id']==r['video_id'] and a['linear_pixel_ap'] is not None];ap=float(np.mean([a['linear_pixel_ap'] for a in rr]));records.append({'pair_id':r['pair_id'],'video_id':r['video_id'],'positive_GT_frames':len(rr),'EXISTING_ZERO_AP':r['pixel_ap'],'LINEAR_LOCAL_AP':ap,'linear_minus_existing':ap-r['pixel_ap'],'prior_ZERO_AP':old[r['video_id']]['pixel_ap'],'baseline_delta':r['pixel_ap']-old[r['video_id']]['pixel_ap']})
    csvsave(OUT/'dev_video_AP.csv',records);torch.cuda.synchronize()
    summary={'status':'COMPLETE','exit_code':0,'execution_head':git('rev-parse','HEAD'),'weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'sampling':sampling,'dev_frames':924,'dev_positive_frames':462,'dev_negative_frames':462,'scored_once':True,'video_AP':records,'macro':{k:float(np.mean([r[k] for r in records])) for k in ['EXISTING_ZERO_AP','LINEAR_LOCAL_AP','linear_minus_existing','baseline_delta']},'fold_check_max_abs':error,'projection':{'train':train_projection,'dev':dev_projection},'model_unchanged':True,'dev_seconds':time.monotonic()-dev_start,'wall_seconds':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    save(OUT/'summary.json',summary)
    lines=['# 冻结局部像素特征线性读出诊断','仅原RGB_DEPTH_PARTITION last epoch12/576；主模型eval/no_grad，参数和缓冲hash不变。Local.native为原生32维、加上下文前特征。原头完整pixel_context投影含bias置零；线性头只读取同一份局部特征。',table(list(records[0]),[list(r.values()) for r in records]),'宏平均：'+json.dumps(summary['macro'],ensure_ascii=False),'训练采样：'+json.dumps(sampling,ensure_ascii=False),'训练48片段1152唯一帧；开发8视频924唯一帧/462正462负；尾段上下文不重复采样或计分。像素AP用完整原生像素，正GT帧先视频内平均再fake视频宏平均。空GT不填AP0。不报告阈值Dice/召回/FPR，也不将AP称为real误报验证。','固定StandardScaler+L2 LogisticRegression C1/lbfgs/max_iter1000/balanced；只拟合一个读出，不读开发作采样、拟合或调参。GT不作为主模型输入，几何valid不筛选采样/评分。',f'线性折入FP32最大绝对误差{error}，预设atol/rtol均1e-5；n_iter {sampling["n_iter"]}，无ConvergenceWarning。','资源：'+json.dumps({k:summary[k] for k in ['dev_seconds','wall_seconds','peak_allocated','peak_reserved','peak_RSS']},ensure_ascii=False),'执行HEAD '+summary['execution_head']+'，退出码0；计算和发布状态分开。']
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);(REPO/DOC).write_text(report);save(OUT/'status.json',{'status':'COMPLETE','exit_code':0,'stage':'report_complete'})


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()})
        raise
