"""Frozen slab evidence-retention check, actual score hooks and native metrics.

CLI caller reuses SlabPool, evaluate and post-projection zero. No new model or
training switch. Per-frame temporary scores/weights are discarded after pooling.
"""
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np
import torch
from .data import Video,save
from .experiment import REPO,evaluate,git,state_hash
from .set_fit_check import build
from .pixel_message_dev_check import pixel_message_mode,pixel_top1,score,metrics_delta,table
from research_tools.depth_slice.model import slab_ids
from scripts.prepare_v8_assets import digest

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1')
PRIOR=BASE.parent/'pvst_pixel_message_dev_check_v1'
OUT=BASE.parent/'pvst_slab_pooling_retention_check_v1'
BRANCH='pvst-slab-pooling-retention-check'
DOC='docs/pvst/slab_pooling_retention_check_v1.md'
MODES=('NORMAL','PIXEL_CONTEXT_ZERO')


def read(p):return json.loads(Path(p).read_text())
def csvsave(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def mean(values):
    a=[v for v in values if v is not None]
    return float(np.mean(a)) if a else None

def normalize_scores(scores,index,n):
    """Exact FP32 scatter/exp/index_add formula used by current SlabPool."""
    scores=scores.float();maximum=scores.new_full((n,4),-torch.inf).scatter_reduce(0,index[:,None].expand(-1,4),scores,reduce='amax',include_self=True)
    w=(scores-maximum[index]).exp();denom=w.new_zeros(n,4).index_add(0,index,w)
    return w/denom[index]

def routing(xyz,valid,gt):
    ids,index=slab_ids(xyz,valid);counts=torch.bincount(index,minlength=len(ids));g=gt[valid]
    edits=torch.bincount(index[g],minlength=len(ids));mixed=(edits>0)&(edits<counts);pure=(edits==counts)&(counts>0)
    assert int(counts.sum())==int(valid.sum()) and int(edits.sum())==int((gt&valid).sum())
    return ids,index,counts,edits,g,mixed,pure

def summarize_pool(scores,index,counts,edits,g):
    w=normalize_scores(scores,index,len(counts))
    # Audit only: long original FP32 index_add reductions can drift. Do not
    # silently repair or re-normalize the actual model weights. FP64 below is
    # only for measuring their sum, never used by the model or GT mass.
    assert torch.isfinite(w).all() and (w>=0).all()
    audit=w.new_zeros(len(counts),4,dtype=torch.float64).index_add(0,index,w.double())
    error=float((audit-1).abs().max()) if audit.numel() else 0.
    mass=w.new_zeros(len(counts),4).index_add(0,index[g],w[g])
    return mass,error


class PoolCapture:
    """Only tiny per-slab GT masses survive until this window's pixel decode."""
    def __init__(self,model,context):self.model=model;self.context=context;self.pending={};self.chunks=[];self.active=False;self.ordinal=0;self.max_error=0.
    def root_pre(self,module,args):
        self.fs=args[0];self.ordinal=0;self.pending.clear()
    def pool_pre(self,module,args):
        f=self.fs[self.ordinal];self.ordinal+=1;self.frame=f;self.chunks.clear()
        self.active=self.context['role']=='fake' and self.context['start']<=f['frame_index']<self.context['end']
    def score_hook(self,module,args,out):
        if self.active:self.chunks.append(out.detach().float())
    def pool_post(self,module,args,out):
        if not self.active:return
        f=self.frame;device=out[0].device;v=f['valid'].to(device);g=f['gt'].to(device)
        ids,index,counts,edits,gg,mixed,pure=routing(f['xyz'].to(device),v,g)
        assert torch.equal(ids,out[1])
        scores=torch.cat(self.chunks) if self.chunks else out[0].new_empty((0,4)).float();assert scores.shape==(int(v.sum()),4)
        mass,error=summarize_pool(scores,index,counts,edits,gg);self.max_error=max(self.max_error,error)
        self.pending[f['frame_index']]={'ids':ids,'counts':counts,'edits':edits,'mass':mass}
        self.chunks.clear() # No full pixel weights/scores retained beyond this frame.
    @contextmanager
    def enabled(self):
        handles=[self.model.register_forward_pre_hook(self.root_pre),self.model.pool.register_forward_pre_hook(self.pool_pre),self.model.pool.score.register_forward_hook(self.score_hook),self.model.pool.register_forward_hook(self.pool_post)]
        try:yield self
        finally:
            for h in handles:h.remove()
            self.chunks.clear();self.pending.clear();self.fs=[];self.frame=None


def collect(model,rows,c,mode):
    context={};current=[None];byid={r['video_id']:r for r in rows};frame_rows=[];group_rows=[];seen=[]
    pool=PoolCapture(model,context)
    def progress(stage,**extra):
        if 'video' in extra:current[0]=extra['video']
        save(OUT/'status.json',{'status':'RUNNING','mode':mode,'stage':stage,'pid':os.getpid(),**extra});print(mode,stage,extra,flush=True)
    def before(module,args):
        fs=args[0];row=byid[current[0]];end=fs[-1]['frame_index']+1;lo=fs[0]['frame_index'];n=row['actual_frames']
        match=[s for s in range(0,n,24) if min(s+24,n)==end and (max(0,n-24) if s+24>=n else s)==lo]
        assert len(match)==1;context.update(start=match[0],end=end,role=row['role'])
    def after(module,args,pred):
        fs=args[0];row=byid[current[0]];assert not torch.is_grad_enabled() and not module.training
        for f,p in zip(fs,pred,strict=True):
            i=f['frame_index']
            if not context['start']<=i<context['end']:continue
            assert f['pts']==row['pts'][i]['pts_s'];assert bool(f['gt'].any())==(row['role']=='fake')
            if not torch.isfinite(p['pixel']).all() or not torch.isfinite(p['frame']):raise ValueError('NONFINITE_OUTPUT')
            seen.append((row['video_id'],i));device=p['pixel'].device;gt=f['gt'].to(device);valid=f['valid'].to(device)
            ids,index,counts,edits,g,mixed,pure=routing(f['xyz'].to(device),valid,gt)
            gt_count=int(gt.sum());vg_count=int(edits.sum());z=p['pixel'].float()[valid]
            frame_rows.append({'mode':mode,'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frame_index':i,'pts':f['pts'],'label':bool(gt_count),
                'pixel_top1_logit':float(pixel_top1(p['pixel'])),'head_logit':float(p['frame']),
                'GT_pixels':gt_count,'valid_GT_pixels':vg_count,'GT_valid_coverage':vg_count/gt_count if gt_count else None,
                'mixed_slabs':int(mixed.sum()),'pure_edit_slabs':int(pure.sum()),'no_valid_GT':vg_count==0})
            if row['role']!='fake':continue
            area=edits.float()/counts;mass=None
            if mode=='NORMAL':
                pooled=pool.pending.pop(i);assert torch.equal(ids,pooled['ids']) and torch.equal(counts,pooled['counts']) and torch.equal(edits,pooled['edits']);mass=pooled['mass']
            for name,select in [('small_lt1pct',mixed&(area<.01)),('other_ge1pct',mixed&(area>=.01))]:
                n=int(select.sum());e_mask=select[index]&g;b_mask=select[index]&~g
                rowg={'mode':mode,'video_id':row['video_id'],'pair_id':row['pair_id'],'frame_index':i,'pts':f['pts'],'group':name,'slabs':n,
                      'edit_pixels':int(e_mask.sum()),'background_pixels':int(b_mask.sum()),'mean_edit_fraction':float(area[select].mean()) if n else None,
                      'edit_logit_mean':float(z[e_mask].mean()) if n else None,'background_logit_mean':float(z[b_mask].mean()) if n else None}
                rowg['edit_minus_background']=rowg['edit_logit_mean']-rowg['background_logit_mean'] if n else None
                for head in range(4):
                    value=float(mass[select,head].mean()) if n and mass is not None else None
                    rowg['head'+str(head+1)+'_edit_mass']=value
                    rowg['head'+str(head+1)+'_mass_minus_uniform']=value-rowg['mean_edit_fraction'] if value is not None else None
                    rowg['head'+str(head+1)+'_fraction_above_uniform']=float((mass[select,head]>area[select]).float().mean()) if n and mass is not None else None
                group_rows.append(rowg)
        assert not pool.pending
        if mode=='NORMAL':pool.fs=[];pool.frame=None
    first=model.register_forward_pre_hook(before);last=model.register_forward_hook(after);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        with pixel_message_mode(model,mode!='NORMAL') as projection:
            if mode=='NORMAL':
                with pool.enabled():value=evaluate(model,rows,c,progress,pixel_ap=True)
            else:value=evaluate(model,rows,c,progress,pixel_ap=True)
    finally:first.remove();last.remove()
    torch.cuda.synchronize();assert len(seen)==len(set(seen))==924
    assert sum(r['label'] for r in frame_rows)==462
    if mode!='NORMAL':assert projection['delivered_nonzero_elements']==0
    result={'spatial':value['metrics'],'video_records':value['records'],'detection':score(frame_rows,[r['pixel_top1_logit'] for r in frame_rows]),
            'projection':projection,'normalization_max_error':pool.max_error,'seconds':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    save(OUT/(mode+'_result.json'),result);csvsave(OUT/(mode+'_frame_statistics.csv'),frame_rows);csvsave(OUT/(mode+'_frame_group_statistics.csv'),group_rows)
    return result,frame_rows,group_rows


def aggregate_groups(rows):
    result=[]
    for mode in MODES:
        for vid in sorted({r['video_id'] for r in rows}):
            for group in ['small_lt1pct','other_ge1pct']:
                rr=[r for r in rows if r['mode']==mode and r['video_id']==vid and r['group']==group];present=[r for r in rr if r['slabs']>0]
                if not rr:continue
                a={'mode':mode,'video_id':vid,'pair_id':rr[0]['pair_id'],'group':group,'comparable_frames':len(present),'mixed_slabs_total':sum(r['slabs'] for r in present),
                   'edit_pixels_total':sum(r['edit_pixels'] for r in present),'background_pixels_total':sum(r['background_pixels'] for r in present)}
                for k in ['mean_edit_fraction','edit_logit_mean','background_logit_mean','edit_minus_background']+[f'head{h}_{suffix}' for h in range(1,5) for suffix in ['edit_mass','mass_minus_uniform','fraction_above_uniform']]:a[k]=mean([r[k] for r in present])
                result.append(a)
    return result


def main():
    if git('branch','--show-current')!=BRANCH:raise ValueError('WRONG_BRANCH')
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt';identity=read(BASE/'identity.json');prior=read(PRIOR/'summary.json')
    if OUT.exists():
        s=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if s.get('status')=='COMPLETE' and s.get('weight_sha256')==digest(weight) and s.get('manifest_sha256')==digest(BASE/'manifest.json'):return
        raise ValueError('EXISTING_OUTPUT_INSPECT_NO_RETRY')
    assert not git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice')
    OUT.mkdir();started=time.monotonic();torch.set_num_threads(4);manifest=read(BASE/'manifest.json');c=read(BASE/'config.json')
    assert digest(weight)==prior['weight_sha256'] and digest(BASE/'manifest.json')==identity['frozen_manifest_sha256'] and digest(BASE/'config.json')==identity['config_sha256']
    for rel,sha in identity['code'].items():assert digest(REPO/rel)==sha
    rows=[r for r in manifest['rows'] if r['split']=='dev'];assert len(rows)==8 and sum(r['actual_frames'] for r in rows)==924
    for r in rows:
        assert digest(Path(r['video_path']))==r['video_sha256']
        if r['role']=='fake':assert digest(Path(r['mask_zip']))==r['mask_sha256']
        for path,sha in next(a for a in identity['assets'] if a['video_id']==r['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha
        v=Video(r,c);v.close()
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    assert checkpoint['condition']=='RGB_DEPTH_PARTITION' and checkpoint['epoch']==12 and checkpoint['updates']==576 and checkpoint['identity']==identity['identity']
    model=build(c,OUT,rgb_only=True).eval();assert read(OUT/'initialization.json')['model_sha256']==checkpoint['initial_model_sha256'];model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    before=state_hash(model.state_dict());results={};frames={};groups=[]
    save(OUT/'identity.json',{'head':git('rev-parse','HEAD'),'weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'baseline_identity':identity['identity'],'normalization':'actual SlabPool.score output -> original FP32 scatter softmax; no changed values','small_group':'mixed slab valid edited fraction <.01, fixed before evaluation'})
    for mode in MODES:
        results[mode],frames[mode],gg=collect(model,rows,c,mode);groups+=gg
        assert state_hash(model.state_dict())==before and not model.sets.capture and not model.sets.zero_messages
        assert all(p.grad is None for p in model.parameters())
    assert [(r['video_id'],r['frame_index'],r['pts'],r['label'],r['valid_GT_pixels'],r['mixed_slabs'],r['pure_edit_slabs']) for r in frames['NORMAL']]==[(r['video_id'],r['frame_index'],r['pts'],r['label'],r['valid_GT_pixels'],r['mixed_slabs'],r['pure_edit_slabs']) for r in frames['PIXEL_CONTEXT_ZERO']]
    for a,b in zip([r for r in groups if r['mode']=='NORMAL'],[r for r in groups if r['mode']=='PIXEL_CONTEXT_ZERO'],strict=True):
        assert [a[k] for k in ['video_id','frame_index','group','slabs','edit_pixels','background_pixels','mean_edit_fraction']]==[b[k] for k in ['video_id','frame_index','group','slabs','edit_pixels','background_pixels','mean_edit_fraction']]
    grouped=aggregate_groups(groups);csvsave(OUT/'video_group_statistics.csv',grouped)
    coverage=[]
    for row in rows:
        if row['role']!='fake':continue
        rr=[r for r in frames['NORMAL'] if r['video_id']==row['video_id']]
        coverage.append({'video_id':row['video_id'],'pair_id':row['pair_id'],'frames':len(rr),'GT_valid_coverage_mean':mean([r['GT_valid_coverage'] for r in rr]),'GT_valid_coverage_min':min(r['GT_valid_coverage'] for r in rr),'GT_pixels':sum(r['GT_pixels'] for r in rr),'valid_GT_pixels':sum(r['valid_GT_pixels'] for r in rr),'no_valid_GT_frames':sum(r['no_valid_GT'] for r in rr),'pure_edit_slabs_total':sum(r['pure_edit_slabs'] for r in rr),'mixed_slabs_total':sum(r['mixed_slabs'] for r in rr)})
    csvsave(OUT/'video_GT_coverage.csv',coverage)
    old={(r['video_id'],int(r['frame_index'])):r for r in csv.DictReader((PRIOR/'NORMAL_frames.csv').open())}
    reproduction={'max_abs_top1_score_delta':max(abs(r['pixel_top1_logit']-float(old[r['video_id'],r['frame_index']]['pixel_top1_logit'])) for r in frames['NORMAL']),
                  'metric_delta':metrics_delta(results['NORMAL']['detection'],prior['results']['NORMAL']['pixel'])}
    head_difference=max(abs(a['head_logit']-b['head_logit']) for a,b in zip(frames['NORMAL'],frames['PIXEL_CONTEXT_ZERO'],strict=True))
    spatial_rows=[]
    for mode,result in results.items():
        for r in result['video_records']:
            if r['role']!='fake':continue
            real=next(x for x in result['video_records'] if x['pair_id']==r['pair_id'] and x['role']=='real')
            spatial_rows.append({'mode':mode,'pair_id':r['pair_id'],'video_id':r['video_id'],'dice':r['dice'],'iou':r['iou'],'pixel_ap':r['pixel_ap'],'GT_recall':r['gt_recall'],'pred_area':r['pred_area'],'paired_real_FPR':real['empty_fpr']})
    csvsave(OUT/'spatial_by_video.csv',spatial_rows)
    summary={'status':'COMPLETE','exit_code':0,'execution_head':git('rev-parse','HEAD'),'weight_sha256':digest(weight),'manifest_sha256':digest(BASE/'manifest.json'),'results':results,'grouped_statistics':grouped,'GT_coverage':coverage,'spatial_by_video':spatial_rows,'baseline_reproduction':reproduction,'head_max_abs_difference':head_difference,'wall_s':time.monotonic()-started}
    save(OUT/'summary.json',summary)
    lines=['# 冻结切片汇聚与像素上下文保留诊断',
        '两条件均原4对/8视频/924唯一计分帧，462正/462负；固定RGB_DEPTH_PARTITION last epoch12/576，原BF16 eval/no_grad及24帧窗口，尾段历史不重复计分。未排除视频，不训练、不实现方格。',
        'ZERO是pixel_context(C+M)完整输出含bias清零；不是独立训练RGB模型，也不是单独去掉M。正式空间评价使用完整原生画面，混合片分析才使用模型valid。',
        '## 逐视频定位',table(list(spatial_rows[0]),[list(r.values()) for r in spatial_rows]),
        '## 宏平均',table(['模式','Dice','IoU','pixel AP','GT召回','real FPR'],[[mode]+[result['spatial'][k] for k in ['dice','iou','pixel_ap','GT_recall','real_fpr']] for mode,result in results.items()]),
        '## GT覆盖及纯编辑片',table(list(coverage[0]),[list(r.values()) for r in coverage]),
        '## NORMAL四头权重与面积参照（先片内、再帧、再视频）',
        table(['pair','group','可比帧','混合片数','平均编辑占比','head1质量','head2质量','head3质量','head4质量'],[[r['pair_id'],r['group'],r['comparable_frames'],r['mixed_slabs_total'],r['mean_edit_fraction']]+[r[f'head{h}_edit_mass'] for h in range(1,5)] for r in grouped if r['mode']=='NORMAL']),
        '均匀参照为同片有效编辑像素占比。每帧对混合片的面积及各头GT权重质量取等片平均，再对有该组的帧取均值；四头分别保留。mass_minus_uniform及fraction_above_uniform在CSV/summary中。相对增加不等于绝对质量高。',
        '## 混合片内局部响应',table(['模式','pair','group','可比帧','编辑像素总数','同片背景总数','编辑均值logit','同片背景均值logit','差'],[[r[k] for k in ['mode','pair_id','group','comparable_frames','edit_pixels_total','background_pixels_total','edit_logit_mean','background_logit_mean','edit_minus_background']] for r in grouped]),
        '局部logit按每帧该组全部有效编辑像素/同片背景像素分别均值，再等帧平均；没有混合片的帧不参与该组均值。缺失项保持None/不可比较，不填0。片数只描述规模，不当独立视频证据。组固定<1%与>=1%，不按结果改分组。',
        '## 核验与资源', 'NORMAL检测基线复现：'+json.dumps(reproduction,ensure_ascii=False),
        f'帧头两条件最大logit差{head_difference}；正常路径仅被动捕获score，不改need_weights/capture或任何模型值。原FP32权重的FP64求和审计最大偏差'+str(results['NORMAL']['normalization_max_error'])+'。',
        '资源：'+json.dumps({mode:{k:result[k] for k in ['seconds','peak_allocated','peak_reserved','peak_RSS','projection']} for mode,result in results.items()},ensure_ascii=False),
        f"总墙钟{summary['wall_s']:.3f}秒，退出码0；执行HEAD {summary['execution_head']}。所有像素score/权重帧内释放，无全量权重图落盘。产物{OUT}。计算与Git发布状态分开。"]
    text='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(text);(REPO/DOC).write_text(text)
    save(OUT/'status.json',{'status':'COMPLETE','exit_code':0,'stage':'report_complete'})


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()})
        raise
