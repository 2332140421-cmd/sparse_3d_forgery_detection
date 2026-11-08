"""Single context-norm condition; reuse the existing matched train/eval loop.

This entry owns only pinned checks, old-result comparisons and reporting. It
never prepares assets, fits a pair, trains old conditions or scores old models.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from .experiment import (REPO,CONDITIONS,NORM_CONDITION,git,make,optimizer,forward,update,
                         train,state_hash,load_clip,ContextScaleRecorder)
from .data import save
from .b_fit_check import source_snapshot
from scripts.prepare_v8_assets import digest

DOC='docs/pvst/rgbd_slice_context_norm_v1.md'


def groups(model,opt):
    names={id(p):n for n,p in model.named_parameters()}
    return [{'index':i,'lr':g['lr'],'weight_decay':g['weight_decay'],'parameters':sum(p.numel() for p in g['params']),'names':[names[id(p)] for p in g['params']]} for i,g in enumerate(opt.param_groups)]


def checks(c,out,m,progress):
    path=out/'checks.json'
    if path.exists():
        if json.loads(path.read_text())['status']!='PASS':raise ValueError('EXISTING_FAILED_CHECK')
        return
    source=Path(c['reference_run']);progress('initialization_check')
    old=make(c,out,CONDITIONS[1]);cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all();new=make(c,out,NORM_CONDITION)
    if not torch.equal(cpu_rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):raise ValueError('INITIAL_RNG_ORDER_CHANGED')
    a=old.state_dict();b=new.state_dict();extra=set(b)-set(a)
    if extra!={'context_norm.weight','context_norm.bias'} or set(a)-set(b):raise ValueError('UNEXPECTED_STATE_DICT_DIFFERENCE')
    for name in a:
        if not torch.equal(a[name],b[name]):raise ValueError('COMMON_INITIAL_PARAMETER_DIFFERENT:'+name)
    if not torch.equal(new.context_norm.weight,torch.ones_like(new.context_norm.weight)) or not torch.equal(new.context_norm.bias,torch.zeros_like(new.context_norm.bias)):raise ValueError('NORM_INITIALIZATION')
    old_n=sum(p.numel() for p in old.parameters());new_n=sum(p.numel() for p in new.parameters())
    if new_n-old_n!=1024:raise ValueError('UNEXPECTED_PARAMETER_CHANGE')
    oo=optimizer(old,c);nn=optimizer(new,c);og=groups(old,oo);ng=groups(new,nn)
    old_assignment={name:(g['index'],g['lr'],g['weight_decay']) for g in og for name in g['names']};new_assignment={name:(g['index'],g['lr'],g['weight_decay']) for g in ng for name in g['names']}
    if any(new_assignment[k]!=v for k,v in old_assignment.items()) or any(new_assignment[k]!=(1,1e-4,.01) for k in extra):raise ValueError('OPTIMIZER_GROUP_CHANGED')
    save(out/'optimizer_groups.json',ng)
    fake=next(r for r in m['rows'] if r['split']=='train' and r['role']=='fake');clip=next(x for x in m['clips'] if x['video_id']==fake['video_id']);fs=load_clip(fake,clip,c)
    if not any(f['gt'].any() for f in fs):raise ValueError('SMOKE_GT_EMPTY')
    old.eval();new.eval()
    with torch.no_grad():pa=forward(old,fs);pb=forward(new,fs)
    errors={key:max(float((x[key].float()-y[key].float()).abs().max()) for x,y in zip(pa,pb)) for key in ['pixel','frame']}
    for x,y in zip(pa,pb):
        for key in errors:torch.testing.assert_close(x[key].float(),y[key].float(),atol=.002,rtol=.002)
    del pa,pb
    # Compatibility only: do not run the trained old model or inherit its weights.
    legacy=torch.load(source/CONDITIONS[1]/'best.pt',map_location='cpu',mmap=True,weights_only=False)
    old.load_state_dict(legacy['model'],strict=True);del old,oo,legacy,a,b
    records=[];bn=state_hash(dict(new.local.encoder.named_buffers()));scales=ContextScaleRecorder(new);torch.cuda.reset_peak_memory_stats()
    for step in [1,2]:
        progress('two_update_check',step=step);scales.begin(len(fs));r=update(new,nn,fs,diagnose=True,context_diagnostics=True);r.update(scales.finish());records.append(r)
        if not all(v['present'] and v['finite'] for v in r['gradients'].values()):raise ValueError('NONFINITE_OR_MISSING_GRADIENT')
    for name in ['context_norm','pool','time','depth','pixel_context','frame_context']:
        if records[1]['gradients'][name]['norm']<=0:raise ValueError('NO_SECOND_STEP_GRADIENT:'+name)
    if state_hash(dict(new.local.encoder.named_buffers()))!=bn:raise ValueError('PRETRAINED_BN_CHANGED')
    scales.close();save(path,{'status':'PASS','old_parameters':old_n,'new_parameters':new_n,'new_parameter_count':new_n-old_n,'new_keys':sorted(extra),
                            'all_shared_parameters_and_buffers_exact':True,'initial_rng_identical':True,'legacy_checkpoint_strict_load':True,'eps':new.context_norm.eps,'zero_projection_output_max_error':errors,
                            'BF16_tolerance':{'atol':.002,'rtol':.002},'video_id':fake['video_id'],'clip':clip,'two_updates':records,'smoke_weights_discarded':True})
    del new,nn,fs;torch.cuda.empty_cache()


def collages(c,out):
    folder=out/'comparison_samples';folder.mkdir(exist_ok=True);old=Path(c['reference_run'])/CONDITIONS[1]/'samples';new=out/NORM_CONDITION/'samples';missing=[]
    for path in sorted(new.glob('*.jpg')):
        prior=old/path.name
        if not prior.exists():missing.append(path.name);continue
        a=cv2.imread(str(prior));b=cv2.imread(str(path))
        if a is None or b is None or a.shape!=b.shape or b.shape[1]%4:missing.append(path.name+':shape');continue
        w=b.shape[1]//4
        image=np.concatenate([b[:,:w],b[:,w:2*w],a[:,2*w:3*w],a[:,3*w:],b[:,2*w:3*w],b[:,3*w:]],1);cv2.imwrite(str(folder/path.name),image)
    (folder/'index.html').write_text('<meta charset="utf-8"><p>固定开发PTS 25/50/75%；RGB | GT | old B P(0–1) | old B logit≥0 | NORM P(0–1) | NORM logit≥0。旧B直接读取保存JPEG，不重新推理，色阶均0–1。JPEG仅作图示，指标来自JSON。</p>'+''.join(f'<p>{p.name}</p><img width="100%" src="{p.name}">' for p in sorted(folder.glob('*.jpg'))))
    return {'paired_samples':len(list(folder.glob('*.jpg'))),'missing':missing}


def plots(c,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(c['reference_run']);new=json.loads((out/NORM_CONDITION/'history.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(11,3))
    for label,folder in [('A',source/CONDITIONS[0]),('B',source/CONDITIONS[1]),('B_NORM',out/NORM_CONDITION)]:
        h=json.loads((folder/'history.json').read_text());x=[r['epoch'] for r in h]
        ax[0].plot(x,[r['train_monitor_clips']['dice'] for r in h],label=label);ax[1].plot(x,[r['dev']['dice'] for r in h],label=label)
    ax[0].set_title('Same fixed training pair / actual trained clips');ax[1].set_title('Full development fake macro Dice')
    for a in ax:a.legend();a.set_xlabel('epoch')
    fig.tight_layout();fig.savefig(out/'monitor_curves.png');plt.close(fig)
    records=[json.loads(line) for line in (out/NORM_CONDITION/'update_history.jsonl').read_text().splitlines()];x=[r['update'] for r in records]
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    axes[0,0].plot(x,[r['gradient_norm'] for r in records],label='global preclip')
    for name in ['frame_context','pixel_context','context_norm']:axes[0,0].plot(x,[r['preclip_context_gradients'][name] for r in records],label=name)
    axes[0,1].plot(x,[r['clip_coefficient'] for r in records]);axes[0,1].set_title('Actual global clip coefficient')
    for key in ['raw_token_rms','normalized_token_rms']:axes[1,0].plot(x,[r[key] for r in records],label=key)
    for key in ['pixel_projection_rms','frame_residual_mean_abs']:axes[1,1].plot(x,[r[key] for r in records],label=key)
    for a in [axes[0,0],axes[1,0],axes[1,1]]:a.legend()
    for a in axes.flat:a.set_xlabel('update')
    fig.tight_layout();fig.savefig(out/'context_diagnostics.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3))
    for key in ['frame_BCE','pixel_BCE','dice_loss']:ax.plot([r['epoch'] for r in new],[r[key] for r in new],label=key)
    ax.legend();fig.tight_layout();fig.savefig(out/'loss_curves.png');plt.close(fig)
    return records


def report(c,out,m,summary,status,error=None,reference_unchanged=None):
    source=Path(c['reference_run']);old=[json.loads((source/x/'summary.json').read_text()) for x in CONDITIONS];result={'status':status,'summary':summary,'error':error,'reference_results':old,'reference_unchanged':reference_unchanged,'execution_head':json.loads((out/'identity.json').read_text())['head']}
    lines=['# 最终关系上下文LayerNorm匹配实验',f'状态：{status}。唯一方法差异：最终关系C逐token共享LayerNorm(512,eps=1e-5,affine=True)，先归一化再求均值；两残差零初始化。',
           '原生观测、MoGe缓存、切片身份、池化/两组时空关系、解码器、数据与训练协议保持不变。仅训练新条件，旧A/B不重新评分。',
           '|条件/checkpoint|epoch|Dice|IoU|pixel AP|GT召回|real FPR|frame AUROC/AP|video AUROC/AP/F1/ACC|',
           '|---|---:|---:|---:|---:|---:|---:|---|---|']
    rows=[(x['condition']+'/best',x['best_epoch'],x['best_dev']) for x in old]
    if summary:rows.extend([(NORM_CONDITION+'/best',summary['best_epoch'],summary['best_dev']),(NORM_CONDITION+'/last',summary['last_epoch'],summary['last_dev'])])
    for label,ep,v in rows:
        f=v['frame'];vv=v['video'];lines.append('|'+ '|'.join(map(str,[label,ep,v['dice'],v['iou'],v['pixel_ap'],v['GT_recall'],v['real_fpr'],f"{f['auroc']:.6f}/{f['ap']:.6f}",f"{vv['auroc']:.6f}/{vv['ap']:.6f}/{vv['f1']:.6f}/{vv['acc']:.6f}"]))+'|')
    if summary:
        records=plots(c,out);paired=collages(c,out);br=json.loads((out/NORM_CONDITION/'best_dev.json').read_text())['records'];reference={name:json.loads((source/name/'best_dev.json').read_text())['records'] for name in CONDITIONS};videos=[]
        lines+=['','## 四条开发fake（同一best）','|pair/类别|A Dice/IoU/AP|B Dice/IoU/AP|NORM Dice/IoU/AP|NORM−B Dice/IoU/AP|','|---|---|---|---|---|']
        for r in [r for r in br if r['role']=='fake']:
            a=next(x for x in reference[CONDITIONS[0]] if x['video_id']==r['video_id']);b=next(x for x in reference[CONDITIONS[1]] if x['video_id']==r['video_id']);keys=['dice','iou','pixel_ap'];diff={k:r[k]-b[k] for k in keys}
            videos.append({'video_id':r['video_id'],'pair_id':r['pair_id'],'generator':r['generator'],'A':{k:a[k] for k in keys},'B':{k:b[k] for k in keys},'NORM':{k:r[k] for k in keys},'NORM_minus_B':diff})
            fmt=lambda v:'/'.join(f'{v[k]:.6f}' for k in keys)
            lines.append('|'+ '|'.join([r['pair_id']+'/'+r['generator'],fmt(a),fmt(b),fmt(r),fmt(diff)])+'|')
        v=summary['best_dev'];b=old[1]['best_dev'];change={k:v[k]-b[k] for k in ['dice','iou','pixel_ap','GT_recall','real_fpr']};h=json.loads((out/NORM_CONDITION/'history.json').read_text());monitor=[x['train_monitor_clips']['dice'] for x in h];recall=[x['train_monitor_clips']['GT_recall'] for x in h]
        drops=[{'from_epoch':a['epoch'],'to_epoch':b['epoch'],'dice_before':a['train_monitor_clips']['dice'],'dice_after':b['train_monitor_clips']['dice'],'recall_before':a['train_monitor_clips']['GT_recall'],'recall_after':b['train_monitor_clips']['GT_recall']} for a,b in zip(h,h[1:]) if a['train_monitor_clips']['dice']>b['train_monitor_clips']['dice']]
        drops=sorted(drops,key=lambda x:x['dice_before']-x['dice_after'],reverse=True)[:3]
        largest=max(videos,key=lambda x:x['NORM_minus_B']['dice']);others=[x['NORM_minus_B']['dice'] for x in videos if x is not largest];mean_others=float(np.mean(others));positive=sum(x['NORM_minus_B']['dice']>0 for x in videos)
        counts={g:{'improved':sum(x['NORM_minus_B']['dice']>0 for x in videos if x['generator']==g),'decreased':sum(x['NORM_minus_B']['dice']<0 for x in videos if x['generator']==g)} for g in ['inpainting','splicing']}
        diag={k:{'first':records[0][k],'last':records[-1][k],'median':float(np.median([x[k] for x in records if x[k] is not None]))} for k in ['gradient_norm','clip_coefficient','raw_token_rms','normalized_token_rms','pixel_projection_rms','frame_residual_mean_abs']}
        result.update(comparison_videos=videos,best_minus_old_B=change,monitor_drops_at_least_point3=drops,type_directions=counts,largest_dice_change_pair=largest['pair_id'],mean_other_dice_change=mean_others,improved_fake_videos=positive,diagnostic_summary=diag,samples=paired)
        lines+=['','## 结论', '新best减旧B：'+json.dumps(change,ensure_ascii=False),
                '开发空间定位：'+('Dice/IoU/pixel AP均改善。' if all(change[k]>0 for k in ['dice','iou','pixel_ap']) else '未同时改善Dice、IoU和pixel AP。'),
                'GT召回与real误报需合看：'+str(v['GT_recall'])+' / '+str(v['real_fpr'])+'；方向见上表，不把更低误报单独当定位收益。',
                '固定监测（仅原第一个pair实际训练首段）Dice全曲线：'+str(monitor)+'；GT召回：'+str(recall),
                '最大的三次相邻轮监测下降（含同轮GT召回）：'+json.dumps(drops,ensure_ascii=False)+'。直接列实测变化，不改变训练/评价规则。',
                '相对旧B，Dice改善'+str(positive)+'/4；类别方向：'+json.dumps(counts,ensure_ascii=False)+'。最大Dice差来自'+largest['pair_id']+'；除它以外三条平均差='+str(mean_others)+'。'+('其余总体也改善，收益不只来自单条。' if mean_others>0 else '其余总体无增益，平均收益需警惕单条主导。'),
                '检测与定位：新best开发frame AUROC='+str(v['frame']['auroc'])+'，旧B='+str(old[1]['best_dev']['frame']['auroc'])+'；video AUROC='+str(v['video']['auroc'])+'，旧B='+str(old[1]['best_dev']['video']['auroc'])+'。全部同best，不拼不同轮次。',
                '保留判断：'+('这四条开发fake支持保留该接入作为当前候选，但不足以推断整体稳定性。' if change['dice']>0 and change['iou']>0 and change['pixel_ap']>0 else '本轮不支持该归一化接入有效，保留原方法与结果。'),
                '训练尺度摘要：'+json.dumps(diag,ensure_ascii=False)+'。范数稳定不能替代定位指标，旧B无同口径训练尺度，不补造。',
                'RMS统计：raw/normalized为所有实际占据token通道元素；pixel投影为占据token×32输出；frame残差为实际帧残差绝对值均值，空观测帧为原路径0。每更新只转移标量到CPU，不额外前向，不是原生像素面积统计。',
                '成本与真实轮数：'+json.dumps(summary,ensure_ascii=False),
                '固定样例：comparison_samples/index.html，直接用旧B保存的JPEG，不重推理，概率TURBO色阶均0–1；不根据画面选择。完整监测/开发曲线monitor_curves.png，梯度和尺度context_diagnostics.png。']
    if error:lines+=['','## 实际执行错误',error]
    result['storage_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());save(out/'result_summary.json',result)
    lines+=['','正式A/B与拟合诊断源文件未变：'+str(result['reference_unchanged']), '执行代码：'+result['execution_head']+'；独立资产目录：`'+str(out)+'`。计算与发布状态分开。',
            '空间指标：正GT帧均值再fake视频宏平均；pixel AP正GT帧AP再fake视频宏平均，不是AP@IoU。完整开发每帧一次，固定阈值logit≥0、视频log-mean-exp温度1，与旧协议一致。','本轮到此停止，不追加其他训练或调参。']
    text=re.sub(r'(\|[^\n]*\|)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))+'\n';(out/'report.md').write_text(text);(REPO/DOC).write_text(text)


def publish(out):
    try:
        git('add',DOC);git('commit','-m','Report final-context LayerNorm matched experiment');head=git('rev-parse','HEAD');git('push','origin','pvst-rgbd-slice-relations')
        remote=git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/pvst-rgbd-slice-relations').split()[0];save(out/'publication.json',{'status':'COMPLETE' if head==remote else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(out/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'head':git('rev-parse','HEAD')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='docs/pvst/rgbd_slice_context_norm_config_v1.json');args=parser.parse_args();c=json.loads(Path(args.config).read_text());out=Path(c['output']);source=Path(c['reference_run']);m=json.loads((source/'manifest.json').read_text())
    if git('branch','--show-current')!='pvst-rgbd-slice-relations':raise ValueError('WRONG_BRANCH')
    if c['conditions']!=[NORM_CONDITION] or source.resolve() in out.resolve().parents or out.resolve()==source.resolve():raise ValueError('CONDITION_OR_OUTPUT_ISOLATION')
    if git('status','--porcelain','--','research_tools/rgbd_slice','docs/pvst/rgbd_slice_context_norm_config_v1.json'):raise ValueError('EXECUTION_SOURCE_UNCOMMITTED')
    original=json.loads((source/'config.json').read_text())
    for key in ['seed','encoder_lr','new_lr','weight_decay','frames','epochs','min_epochs','patience','scale','h','d0','D','heads','blocks','dice_epsilon']:
        if c[key]!=original[key]:raise ValueError('FROZEN_PROTOCOL_DIFFERENCE:'+key)
    code={str(p.relative_to(REPO)):digest(p) for p in (REPO/'research_tools/rgbd_slice').glob('*.py')};identity=hashlib.sha256(json.dumps({'code':code,'config':c,'manifest':digest(source/'manifest.json'),'common':digest(source/'common_initial.pt')},sort_keys=True).encode()).hexdigest()
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())['identity']!=identity:raise ValueError('PROTOCOL_CHANGED')
        if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text())['status']=='COMPLETE':print('REUSE_COMPLETE',out);return
        raise ValueError('EXISTING_INCOMPLETE_RUN_NO_AUTO_RETRY')
    out.mkdir(parents=True,exist_ok=True);head=git('rev-parse','HEAD');save(out/'identity.json',{'identity':identity,'head':head,'code':code});save(out/'config.json',c);save(out/'manifest.json',m)
    snapshots={str(p):source_snapshot(p) for p in [source,source.parent/'pvst_rgbd_slice_b_fit_check_v1']};started=time.monotonic();summary=None
    def progress(stage,**kw):
        r={'status':'RUNNING','stage':stage,'pid':os.getpid(),'execution_head':head,'wall_s':time.monotonic()-started,**kw};save(out/'progress.json',r);print(json.dumps(r),flush=True)
    try:
        torch.set_num_threads(4);save(out/'launch.json',{'head':head,'pid':os.getpid(),'started_unix':time.time(),'gpu':torch.cuda.get_device_name(),'free_total':torch.cuda.mem_get_info()});progress('prepare_readonly')
        checks(c,out,m,progress)
        summary=train(NORM_CONDITION,c,out,m,identity,progress);summary['total_wall_s']=time.monotonic()-started
        if any(source_snapshot(Path(p))!=snapshot for p,snapshot in snapshots.items()):raise ValueError('REFERENCE_ASSETS_CHANGED')
        report(c,out,m,summary,'COMPLETE',reference_unchanged=True);save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'exit_code':0,'execution_head':head});(out/'exit_code.txt').write_text('0\n');publish(out)
    except BaseException as error:
        detail=traceback.format_exc();print(detail,flush=True);(out/'error.txt').write_text(detail);status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'FAILED';save(out/'progress.json',{'status':status,'stage':'execution_failed','pid':os.getpid(),'error':str(error),'exit_code':1});(out/'exit_code.txt').write_text('1\n');report(c,out,m,summary,status,detail,reference_unchanged=all(source_snapshot(Path(p))==snapshot for p,snapshot in snapshots.items()));publish(out);raise


if __name__=='__main__':main()
