"""Frozen development-only post-projection context intervention.

Caller CLI; existing evaluate supplies native frames and causal windows. Output
is two score sets and a report; no new model, training or permanent model switch.
"""
from contextlib import contextmanager
import csv
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
from .gru_readout_probe import score
from scripts.prepare_v8_assets import digest

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1')
EVIDENCE=BASE.parent/'pvst_frame_detection_evidence_check_v1'
PRIOR=BASE.parent/'pvst_rgb_partition_pixel_message_ablation_v1'
OUT=BASE.parent/'pvst_pixel_message_dev_check_v1'
BRANCH='pvst-pixel-message-dev-check'
DOC='docs/pvst/pixel_message_dev_check_v1.md'
MODES=('NORMAL','PIXEL_MESSAGE_ZERO')
def read(p):return json.loads(Path(p).read_text())
def csvsave(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pixel_top1(z):
    z=z.float().reshape(-1);k=max(1,math.ceil(.01*z.numel()))
    return z.topk(k,sorted=False).values.mean()


@contextmanager
def pixel_message_mode(model,zero):
    stats={'calls':0,'elements':0,'delivered_nonzero_elements':0,'first_shape':None,'dtype':None,'device':None}
    def hook(module,args,output):
        value=torch.zeros_like(output) if zero else output
        assert value.shape==output.shape and value.dtype==output.dtype and value.device==output.device
        stats['calls']+=1;stats['elements']+=value.numel()
        stats['delivered_nonzero_elements']+=int(value.count_nonzero())
        if stats['first_shape'] is None:stats.update(first_shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
        return value if zero else None
    handle=model.pixel_context.register_forward_hook(hook)
    try:yield stats
    finally:handle.remove()


def collect(model,rows,c,mode):
    records=[];windows=[];current=[None]
    expected={r['video_id']:r for r in rows}
    def progress(stage,**extra):
        if 'video' in extra:current[0]=extra['video']
        save(OUT/'status.json',{'status':'RUNNING','mode':mode,'stage':stage,'pid':os.getpid(),**extra})
        print(mode,stage,extra,flush=True)
    def capture(module,args,output):
        assert not torch.is_grad_enabled() and not module.training
        fs=args[0];row=expected[current[0]];lo,end=fs[0]['frame_index'],fs[-1]['frame_index']+1;n=row['actual_frames']
        matches=[s for s in range(0,n,24) if min(s+24,n)==end and (max(0,n-24) if s+24>=n else s)==lo]
        if len(matches)!=1:raise ValueError('UNEXPECTED_CAUSAL_WINDOW')
        start=matches[0];windows.append((row['video_id'],start,end,lo))
        for f,p in zip(fs,output,strict=True):
            i=f['frame_index']
            if not start<=i<end:continue
            assert f['pts']==row['pts'][i]['pts_s']
            label=bool(f['gt'].any());assert label==(row['role']=='fake')
            if not torch.isfinite(p['pixel']).all() or not torch.isfinite(p['frame']):raise ValueError('NONFINITE_SCORES')
            z=float(pixel_top1(p['pixel']))
            records.append({'mode':mode,'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frame_index':i,'pts':f['pts'],'label':label,'pixel_top1_logit':z,'head_logit':float(p['frame'])})
    handle=model.register_forward_hook(capture);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        with pixel_message_mode(model,mode=='PIXEL_MESSAGE_ZERO') as delivered:
            evaluate(model,rows,c,progress,pixel_ap=False)
    finally:handle.remove()
    torch.cuda.synchronize()
    assert len(records)==924 and len({(r['video_id'],r['frame_index']) for r in records})==924
    assert sum(r['label'] for r in records)==462
    assert len(windows)==sum(math.ceil(r['actual_frames']/24) for r in rows)
    if mode=='PIXEL_MESSAGE_ZERO':assert delivered['delivered_nonzero_elements']==0
    result={'pixel':score(records,[r['pixel_top1_logit'] for r in records]),'head':score(records,[r['head_logit'] for r in records]),
            'frames':924,'positive_frames':462,'negative_frames':462,'windows':windows,'delivered_projection':delivered,'seconds':time.monotonic()-start,
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    csvsave(OUT/(mode+'_frames.csv'),records);save(OUT/(mode+'_result.json'),result)
    return result,records


def metrics_delta(a,b):return {level:{k:a[level][k]-b[level][k] for k in ['auroc','ap']} for level in ['frame','video']}
def table(headers,rows):
    def fmt(x):return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|']+['|'+'|'.join(fmt(x) for x in r)+'|' for r in rows])


def main():
    if git('branch','--show-current')!=BRANCH:raise ValueError('WRONG_BRANCH')
    identity=read(BASE/'identity.json');config=read(BASE/'config.json');manifest=read(BASE/'manifest.json');weight=BASE/'RGB_DEPTH_PARTITION/last.pt'
    if OUT.exists():
        previous=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if previous.get('status')=='COMPLETE' and previous.get('weight_sha256')==digest(weight) and previous.get('manifest_sha256')==digest(BASE/'manifest.json') and previous.get('intervention')=='post_pixel_context_zeros_like':return
        raise ValueError('EXISTING_INCOMPLETE_OR_DIFFERENT_IDENTITY_NO_RETRY')
    if git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice'):raise ValueError('UNCOMMITTED_EXECUTION_SOURCE')
    OUT.mkdir();started=time.monotonic();torch.set_num_threads(4)
    save(OUT/'status.json',{'status':'RUNNING','stage':'identity','pid':os.getpid()})
    assert identity['head']=='5ffe208bc9bcac16ccb0429ebc8659b6bea36267'
    assert digest(BASE/'manifest.json')==identity['frozen_manifest_sha256'] and digest(BASE/'config.json')==identity['config_sha256']
    for rel,sha in identity['code'].items():assert digest(REPO/rel)==sha,rel
    ref=read(EVIDENCE/'summary.json');assert ref['status']=='COMPLETE'
    expected_weight=next(r['checkpoint_sha256'] for r in ref['checkpoints'] if r['condition']=='RGB_DEPTH_PARTITION');assert digest(weight)==expected_weight
    source=(EVIDENCE/'diagnostic.py').read_text()
    if 'math.ceil(.01*z.numel())' not in source or 'z.topk(k,sorted=False).values.mean()' not in source:raise ValueError('PREVIOUS_TOP1_DEFINITION_UNCONFIRMED')
    assert read(PRIOR/'summary.json')['status']=='COMPLETE'
    rows=[r for r in manifest['rows'] if r['split']=='dev'];assert len(rows)==8 and sum(r['actual_frames'] for r in rows)==924
    for r in rows:
        assert digest(Path(r['video_path']))==r['video_sha256']
        if r['role']=='fake':assert digest(Path(r['mask_zip']))==r['mask_sha256']
        for path,sha in next(a for a in identity['assets'] if a['video_id']==r['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha,path
        v=Video(r,config)
        try:assert all(bool(v.gt(i).any())==(r['role']=='fake') for i in range(r['actual_frames']))
        finally:v.close()
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    assert checkpoint['condition']=='RGB_DEPTH_PARTITION' and checkpoint['epoch']==12 and checkpoint['updates']==576 and checkpoint['identity']==identity['identity']
    model=build(config,OUT,rgb_only=True).eval();assert read(OUT/'initialization.json')['model_sha256']==checkpoint['initial_model_sha256']
    model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    assert model.rgb_only and not model.sets.zero_messages and not model.sets.capture
    before=state_hash(model.state_dict());results={};frames={}
    save(OUT/'identity.json',{'head':git('rev-parse','HEAD'),'weight_sha256':expected_weight,'original_identity':identity['identity'],'manifest_sha256':digest(BASE/'manifest.json'),
        'baseline_top1_source_sha256':digest(EVIDENCE/'diagnostic.py'),'prior_intervention_source_sha256':digest(PRIOR/'diagnostic.py'),
        'intervention':'zero entire pixel_context output after affine including bias; preserves shape/dtype/device; previous C-only pre-projection intervention is different'})
    for mode in MODES:
        results[mode],frames[mode]=collect(model,rows,config,mode)
        assert state_hash(model.state_dict())==before and not model.sets.zero_messages and not model.sets.capture
        assert all(p.grad is None for p in model.parameters())
    a,b=results['NORMAL'],results['PIXEL_MESSAGE_ZERO'];fa,fb=frames['NORMAL'],frames['PIXEL_MESSAGE_ZERO']
    assert [(r['video_id'],r['frame_index'],r['pts'],r['label']) for r in fa]==[(r['video_id'],r['frame_index'],r['pts'],r['label']) for r in fb]
    oldrows=[r for r in csv.DictReader((EVIDENCE/'frame_scores.csv').open()) if r['condition']=='RGB_DEPTH_PARTITION' and r['split']=='dev']
    oldmap={(r['video_id'],int(r['frame_index'])):r for r in oldrows};assert len(oldmap)==924
    baseline=ref['results']['RGB_DEPTH_PARTITION']['dev']['metrics']['pixel_top1pct'];baseline_delta=metrics_delta(a['pixel'],baseline)
    normal_delta=max(abs(r['pixel_top1_logit']-float(oldmap[r['video_id'],r['frame_index']]['pixel_top1pct_logit'])) for r in fa)
    head_delta=max(abs(x['head_logit']-y['head_logit']) for x,y in zip(fa,fb,strict=True))
    head_metric_delta=metrics_delta(a['head'],b['head']);pairrows=[];videos=[]
    oldpairs={r['pair_id']:r for r in csv.DictReader((EVIDENCE/'pair_video_scores.csv').open()) if r['condition']=='RGB_DEPTH_PARTITION' and r['split']=='dev' and r['readout']=='pixel_top1pct'}
    baseline_pair_flips=0
    for pair,p in a['pixel']['pairs'].items():
        q=b['pixel']['pairs'][pair];oldgap=float(oldpairs[pair]['fake_minus_real'])
        baseline_pair_flips+=int(np.sign(p['fake_minus_real'])!=np.sign(oldgap))
        pairrows.append({'pair':pair,'normal_real':p['real'],'normal_fake':p['fake'],'normal_gap':p['fake_minus_real'],'zero_real':q['real'],'zero_fake':q['fake'],'zero_gap':q['fake_minus_real'],
                         'normal_minus_zero_gap':p['fake_minus_real']-q['fake_minus_real'],'direction_flipped':bool(np.sign(p['fake_minus_real'])!=np.sign(q['fake_minus_real']))})
    for mode in MODES:
        for row in rows:
            rr=[r for r in frames[mode] if r['video_id']==row['video_id']];z=[r['pixel_top1_logit'] for r in rr]
            videos.append({'mode':mode,'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frames':len(z),'pixel_video_score':float(np.logaddexp.reduce(z)-np.log(len(z)))})
    delta=metrics_delta(a['pixel'],b['pixel'])
    summary={'status':'COMPLETE','exit_code':0,'execution_head':git('rev-parse','HEAD'),'weight_sha256':expected_weight,'manifest_sha256':digest(BASE/'manifest.json'),
             'intervention':'post_pixel_context_zeros_like','results':results,'normal_minus_zero':delta,'pairs':pairrows,
             'baseline_reproduction':{'max_abs_pixel_score_difference':normal_delta,'metric_delta':baseline_delta,'pair_direction_flips':baseline_pair_flips,'saved_metrics':baseline},
             'head_consistency':{'max_abs_logit_difference':head_delta,'changed_frame_scores':sum(x['head_logit']!=y['head_logit'] for x,y in zip(fa,fb)),
                                 'normal_minus_zero_metrics':head_metric_delta,'head_pair_flips':int(sum(np.sign(p['fake_minus_real'])!=np.sign(b['head']['pairs'][k]['fake_minus_real']) for k,p in a['head']['pairs'].items())),
                                 'scope_check':'Only pixel_context forward output replaced; frame_logits computed beforehand; weights/buffers unchanged and hook removed'},
             'wall_s':time.monotonic()-started}
    csvsave(OUT/'frame_scores.csv',fa+fb);csvsave(OUT/'pair_scores.csv',pairrows);csvsave(OUT/'video_scores.csv',videos);save(OUT/'summary.json',summary)
    lines=['# 冻结模型：像素上下文投影后清零的开发检测对照',
           'COMPLETE；只用原RGB_DEPTH_PARTITION epoch12/576的last.pt，原4对8视频924唯一计分帧（462正/462负）。eval/no_grad、原BF16，保留原生画面、PTS、24帧窗口及尾段左上下文；上下文不重复计分。',
           '## 主表：像素top-1%检测',table(['条件','帧AUROC','帧AP','视频AUROC','视频AP'],[[mode]+[results[mode]['pixel'][level][k] for level in ['frame','video'] for k in ['auroc','ap']] for mode in MODES]+[['NORMAL−ZERO']+[delta[level][k] for level in ['frame','video'] for k in ['auroc','ap']]]),
           '## 每对视频分数及方向',table(list(pairrows[0]),[list(r.values()) for r in pairrows]),
           '## 核验', '基线复现：'+json.dumps(summary['baseline_reproduction'],ensure_ascii=False), '原帧头一致性：'+json.dumps(summary['head_consistency'],ensure_ascii=False),
           '干预是pixel_context(C+M)完整投影输出（含bias）→zeros_like，再沿原代码回填像素；不是投影前置零，也不是上次保留C的pixel_context(C)。本轮不拆分C/M/bias各自贡献。原集合关系与GRU完整执行，zero_messages/capture始终关闭。',
           'top1%口径已按旧脚本核对：所有原生像素FP32 logit、k=max(1,ceil(.01×H×W))、topk均值；视频log-mean-exp。GT仅作标签，不选像素或按valid裁画面。没有调阈值/比例。开发fake全程mask非空，此为真假排序，不检验编辑起止定位。',
           '本表尚需结合基线重算误差与逐pair方向解释：微小BF16路径重算差异本身不判失败，但若NORMAL偏差足以改变关键排序结论，则不作消息归因。仅此权重的推理依赖，不能分离深度/时序/集合构造贡献，也不能替代移除模块后的匹配重训。',
           '资源：'+json.dumps({mode:{k:results[mode][k] for k in ['seconds','peak_allocated','peak_reserved','peak_RSS','delivered_projection']} for mode in MODES},ensure_ascii=False),
           f"总墙钟{summary['wall_s']:.3f}秒；真实退出码0；执行HEAD {summary['execution_head']}；完整产物{OUT}。计算与发布状态独立。"]
    text='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(text);(REPO/DOC).write_text(text)
    save(OUT/'status.json',{'status':'COMPLETE','exit_code':0,'stage':'report_complete'})


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()})
        raise
