"""ADR-0040: one fixed ANAKIN8 FRAME_BCE_OFF run; original model/queue.

Caller CLI; wraps existing build/update_once/evaluate and read-only baseline.
Output: bounded one-condition training, final common-clip fit and comparison.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from .data import Video,save
from .experiment import REPO,evaluate,forward,git,load_clip,optimizer,state_hash
from .set_fit_check import build,update_once,clip_objective
from .rgb_partition_anakin_dev import validate_split,read,fingerprint
from research_tools.depth_slice.model import terms
from scripts.prepare_v8_assets import digest

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1')
FIT=BASE.parent/'pvst_rgb_partition_train_fit_eval_v1'
OUT=BASE.parent/'pvst_rgb_partition_frame_loss_off_v1'
BRANCH='pvst-rgb-partition-frame-loss-off'
DOC='docs/pvst/rgb_partition_frame_loss_off_v1.md'
LIMIT=14400


def prepare():
    if OUT.exists():raise ValueError('EXISTING_OUTPUT_NO_AUTOMATIC_RESTART')
    OUT.mkdir();save(OUT/'status.json',{'status':'RUNNING','stage':'verify_baseline','pid':os.getpid()});identity=read(BASE/'identity.json');summary=read(BASE/'summary.json');m=read(BASE/'manifest.json');c=read(BASE/'config.json')
    assert identity['head']=='5ffe208bc9bcac16ccb0429ebc8659b6bea36267' and summary['status']=='COMPLETE'
    assert digest(BASE/'manifest.json')==identity['frozen_manifest_sha256'] and digest(BASE/'orders.json')==identity['orders_sha256'] and digest(BASE/'config.json')==identity['config_sha256']
    counts=validate_split(m['rows'],m['clips']);assert counts['dev_frames']==924 and counts['clips']==48
    fit=next(r for r in read(FIT/'summary.json')['results'] if r['condition']=='RGB_DEPTH_PARTITION' and r['checkpoint']=='last')
    checkpoint=torch.load(BASE/'RGB_DEPTH_PARTITION/last.pt',map_location='cpu',weights_only=True)
    assert checkpoint['condition']=='RGB_DEPTH_PARTITION' and checkpoint['epoch']==12 and checkpoint['updates']==576 and checkpoint['identity']==identity['identity']
    assert digest(BASE/'RGB_DEPTH_PARTITION/last.pt')==fit['sha256'] and checkpoint['initial_model_sha256']==fit['initial_model_sha256'];del checkpoint
    for rel,sha in identity['code'].items():
        if rel!='research_tools/rgbd_slice/set_fit_check.py' and (REPO/rel).exists():
            assert digest(REPO/rel)==sha,rel
    for r in m['rows']:
        print('VERIFY',r['video_id'],flush=True)
        assert digest(Path(r['video_path']))==r['video_sha256']
        if r['role']=='fake':assert digest(Path(r['mask_zip']))==r['mask_sha256']
        for path,sha in next(a for a in identity['assets'] if a['video_id']==r['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha,path
        v=Video(r,c);v.close()
    for file in ['manifest.json','orders.json']:shutil.copyfile(BASE/file,OUT/file)
    c.update(output=str(OUT),protocol='rgb-partition-frame-loss-off-v1',conditions=['FRAME_BCE_OFF'],frame_bce_weight=0.)
    save(OUT/'config.json',c)
    value={'head':git('rev-parse','HEAD'),'baseline_execution_head':identity['head'],'baseline_result_head':'950e8c41f60856f57179cce4bad9690dfc258075',
           'baseline_identity':identity['identity'],'baseline_weight_sha256':fit['sha256'],'initial_model_sha256':fit['initial_model_sha256'],
           'manifest_sha256':digest(OUT/'manifest.json'),'orders_sha256':digest(OUT/'orders.json'),'config_sha256':digest(OUT/'config.json'),
           'counts':counts,'code':{str(p.relative_to(REPO)):digest(p) for p in (REPO/'research_tools/rgbd_slice').glob('*.py')}}
    value['identity']=fingerprint(value);save(OUT/'identity.json',value);save(OUT/'baseline.json',{'last':summary['runs']['RGB_DEPTH_PARTITION']['result'],'train_fit':fit})


def frame_metrics(value):
    z=torch.tensor([s for r in value['records'] for s in r['scores']],dtype=torch.float32)
    y=torch.tensor([v for r in value['records'] for v in r['labels']],dtype=torch.float32)
    t=F.binary_cross_entropy_with_logits(z,y,reduction='none')
    value['metrics'].update(frame_bce_fp32=float(t.mean()),positive_frame_bce=float(t[y.bool()].mean()) if y.any() else None,
                            negative_frame_bce=float(t[~y.bool()].mean()) if (~y.bool()).any() else None)
    return value


def train_eval(model,rows,clips,c,progress):
    seen=[];current=[None];background={r['video_id']:[] for r in rows}
    def track(stage,**extra):
        if 'video' in extra:current[0]=extra['video']
        progress(stage,**extra)
    def hook(module,args,pred):
        assert not torch.is_grad_enabled() and not module.training
        fs=args[0];seen.append((current[0],fs[0]['frame_index'],fs[-1]['frame_index']+1))
        for f,p in zip(fs,pred):
            g=f['gt'].numpy();z=p['pixel'].float().cpu().numpy();bg=~g
            background[current[0]].append(float(((z>=0)&bg).sum()/bg.sum()) if bg.any() else None)
    h=model.register_forward_hook(hook)
    try:v=frame_metrics(evaluate(model,rows,c,track,clip_list=clips,pixel_ap=False))
    finally:h.remove()
    assert len(seen)==48 and set(seen)=={(x['video_id'],x['start'],x['end']) for x in clips}
    assert sum(r['frames'] for r in v['records'])==1152
    for r in v['records']:
        r['fake_nonedit_FPR']=float(np.mean([x for x in background[r['video_id']] if x is not None])) if r['role']=='fake' else None
    v['metrics']['fake_nonedit_FPR']=float(np.mean([r['fake_nonedit_FPR'] for r in v['records'] if r['role']=='fake']))
    return v


def worker():
    started=time.monotonic();result={'status':'FAILED','condition':'FRAME_BCE_OFF','updates':0,'last_epoch':0,'exit_code':1}
    def progress(stage,**extra):
        if time.monotonic()-started>=LIMIT:raise TimeoutError('FOUR_HOUR_LIMIT')
        value={'status':'RUNNING','stage':stage,'pid':os.getpid(),'wall_s':time.monotonic()-started,**extra};save(OUT/'progress.json',value);save(OUT/'status.json',value);print(stage,extra,flush=True)
    try:
        torch.set_num_threads(4);c=read(OUT/'config.json');m=read(OUT/'manifest.json');identity=read(OUT/'identity.json')
        assert git('rev-parse','HEAD')==identity['head'] and c['frame_bce_weight']==0.
        for rel,sha in identity['code'].items():assert digest(REPO/rel)==sha
        for n in ['manifest','orders','config']:assert digest(OUT/(n+'.json'))==identity[n+'_sha256']
        save(OUT/'launch.json',{'pid':os.getpid(),'head':identity['head'],'gpu':torch.cuda.get_device_name(),'free_total':torch.cuda.mem_get_info(),'started_unix':time.time()})
        trainrows={r['video_id']:r for r in m['rows'] if r['split']=='train'};dev=[r for r in m['rows'] if r['split']=='dev']
        model=build(c,OUT,rgb_only=True);assert read(OUT/'initialization.json')['model_sha256']==identity['initial_model_sha256']
        # Same real clip, training backward, no optimizer/update; discarded afterwards.
        clip=m['clips'][0];fs=load_clip(trainrows[clip['video_id']],clip,c);model.train();progress('preflight')
        output=forward(model,fs);parts=torch.stack([terms(p,f['gt']) for p,f in zip(output,fs)]).mean(0)
        torch.testing.assert_close(clip_objective(parts,1.),parts.sum(),rtol=0,atol=0)
        torch.testing.assert_close(clip_objective(parts,0.),parts[1:].sum(),rtol=0,atol=0)
        clip_objective(parts,0.).backward()
        gradients={}
        for prefix in ['local.encoder','local.refine','local.final','pool','sets.current_readout','pixel_context','sets.frame_head']:
            gg=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            gradients[prefix]={'present':bool(gg),'finite':all(bool(torch.isfinite(g).all()) for g in gg),'norm':float(torch.sqrt(sum((g.float().square().sum() for g in gg),torch.zeros((),device='cuda'))))}
        assert all(v['finite'] for v in gradients.values()) and gradients['local.final']['norm']>0 and gradients['pixel_context']['norm']>0
        save(OUT/'preflight.json',{'status':'PASS','clip':clip,'frames':24,'initial_model_sha256':identity['initial_model_sha256'],'terms':parts.detach().cpu().tolist(),'spatial_gradients':gradients,'optimizer_updates':0,'zero_upstream_first_step_allowed':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        del model,output,parts,fs;torch.cuda.empty_cache();model=build(c,OUT,rgb_only=True);opt=optimizer(model,c)
        bn=state_hash(dict(model.local.encoder.named_buffers()));torch.cuda.reset_peak_memory_stats();history=[];best=-1.;best_epoch=0
        def assess(epoch,ap=False):
            value=frame_metrics(evaluate(model,dev,c,progress,pixel_ap=ap));assert sum(r['frames'] for r in value['records'])==924
            save(OUT/(f'epoch{epoch:02d}_dev_ap.json' if ap else f'epoch{epoch:02d}_dev.json'),value);return value
        progress('epoch0');assess(0)
        baseline_history=read(BASE/'RGB_DEPTH_PARTITION/history.json')
        for epoch in range(1,13):
            begin=time.monotonic();online=[];objectives=[];order=read(OUT/'orders.json')[str(epoch)]
            assert len(order)==48 and fingerprint(order)==baseline_history[epoch-1]['order_sha256']
            for i,clip in enumerate(order):
                progress('train',epoch=epoch,clip=i+1,updates=result['updates']);fs=load_clip(trainrows[clip['video_id']],clip,c)
                record=update_once(model,opt,fs,frame_bce_weight=0.);del fs
                assert all(g['finite'] for g in record['gradients'].values());record.pop('gradients')
                result['updates']+=1;online.append(record['terms']);objectives.append(record['loss'])
                with (OUT/'updates.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'clip':clip,'update':result['updates'],'frame_bce_weight':0.,**record})+'\n')
            value=assess(epoch);assert state_hash(dict(model.local.encoder.named_buffers()))==bn
            improved=value['metrics']['dice']>best
            if improved:best,best_epoch=value['metrics']['dice'],epoch
            components=np.mean(online,axis=0);history.append({'epoch':epoch,'updates':result['updates'],'terms':components.tolist(),'optimization_loss':float(np.mean(objectives)),
                'spatial_loss':float(components[1:].sum()),'unweighted_frame_BCE':float(components[0]),'dev':value['metrics'],'order_sha256':fingerprint(order),'epoch_seconds':time.monotonic()-begin,'best_epoch':best_epoch})
            save(OUT/'history.json',history)
            state={'state_dict':model.state_dict(),'epoch':epoch,'updates':result['updates'],'condition':'FRAME_BCE_OFF','model_mode':'RGB_DEPTH_PARTITION','frame_bce_weight':0.,'identity':identity['identity'],'initial_model_sha256':identity['initial_model_sha256']}
            tmp=OUT/'last.pt.tmp';torch.save(state,tmp);tmp.replace(OUT/'last.pt')
            if improved:shutil.copyfile(OUT/'last.pt',OUT/'best.pt')
            result['last_epoch']=epoch;save(OUT/'worker_result.json',{**result,'status':'RUNNING'});print('EPOCH_COMPLETE',history[-1],flush=True)
        progress('last_dev_AP');last=assess(12,True);save(OUT/'last_dev.json',last)
        progress('last_training_fit');training=train_eval(model,list(trainrows.values()),m['clips'],c,progress);save(OUT/'last_train.json',training)
        checkpoint=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
        progress('best_dev_AP',epoch=best_epoch);bv=last if best_epoch==12 else assess(best_epoch,True);save(OUT/'best_dev.json',bv)
        result.update(status='COMPLETE',exit_code=0,stop_reason='12_EPOCHS_576_UPDATES',best_epoch=best_epoch,last_dev=last['metrics'],best_dev=bv['metrics'],last_train=training['metrics'],BN_unchanged=True)
    except BaseException as error:
        result.update(status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED',stop_reason=str(error),error=traceback.format_exc());(OUT/'error.txt').write_text(result['error']);print(result['error'],flush=True)
    finally:
        result.update(wall_s=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save(OUT/'worker_result.json',result)
    return result['exit_code']


def table(headers,rows):
    def fmt(x):return '未记录' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join('---' for _ in headers)+'|']+['|'+'|'.join(fmt(x) for x in r)+'|' for r in rows])


def report():
    base=read(OUT/'baseline.json');result=read(OUT/'worker_result.json') if (OUT/'worker_result.json').exists() else {'status':'FAILED','updates':0};exit=read(OUT/'exit.json')
    if exit['timed_out']:
        result.update(status='TIMEOUT',stop_reason='FOUR_HOUR_PARENT_LIMIT')
        if (OUT/'updates.jsonl').exists():result['updates']=max([json.loads(x)['update'] for x in (OUT/'updates.jsonl').read_text().splitlines() if x.strip()],default=0)
    complete=result.get('status')=='COMPLETE' and result.get('updates')==576 and exit['exit_code']==0
    summary={'status':'COMPLETE' if complete else result.get('status','FAILED'),'result':result,'exit':exit,'identity':read(OUT/'identity.json'),'baseline':base,'paired_dev':[],'paired_train':[]}
    lines=['# ANAKIN帧BCE关闭消融','状态：'+summary['status'],'唯一改动：帧BCE权重1→0；pixel BCE/soft Dice各1，未重归一化。完整RGB_DEPTH_PARTITION及帧头/像素消息保留，原参数分组/初始化/seed/BN/Dropout/BF16/clip1不变。',
           '原8对训练16视频48段，每轮1152帧次，固定12轮576步；直接复制基线manifest及orders，原4对完整开发924帧不变。未使用扩量清单、权重或新缓存。',
           '执行提交：'+summary['identity']['head'],'预检：'+json.dumps(read(OUT/'preflight.json') if (OUT/'preflight.json').exists() else {'status':'未完成'},ensure_ascii=False)]
    if complete:
        a=base['last']['last_dev'];b=result['last_dev'];keys=['dice','iou','pixel_ap','GT_recall','real_fpr'];delta={k:b[k]-a[k] for k in keys};summary['difference']=delta
        lines+=['','## 第576步开发主比较',table(['条件','Dice','IoU','pixel AP','GT召回','real FPR'],[[name]+[v[k] for k in keys] for name,v in [('原帧BCE_ON基线',a),('FRAME_BCE_OFF',b),('OFF−ON',delta)]])]
        aa=read(BASE/'RGB_DEPTH_PARTITION/last_dev.json')['records'];bb=read(OUT/'last_dev.json')['records']
        for old in aa:
            if old['role']!='fake':continue
            new=next(r for r in bb if r['video_id']==old['video_id']);real_a=next(r for r in aa if r['pair_id']==old['pair_id'] and r['role']=='real');real_b=next(r for r in bb if r['pair_id']==old['pair_id'] and r['role']=='real')
            summary['paired_dev'].append({'pair':old['pair_id'],'on_Dice':old['dice'],'off_Dice':new['dice'],'delta_Dice':new['dice']-old['dice'],'on_IoU':old['iou'],'off_IoU':new['iou'],'delta_IoU':new['iou']-old['iou'],'on_recall':old['gt_recall'],'off_recall':new['gt_recall'],'on_real_FPR':real_a['empty_fpr'],'off_real_FPR':real_b['empty_fpr']})
        pp=summary['paired_dev'];lines+=['','## 逐开发pair（全部4对）',table(list(pp[0]),[list(r.values()) for r in pp])]
        dd=[r['delta_Dice'] for r in pp];summary['improved_pairs']=sum(x>0 for x in dd);summary['declined_pairs']=sum(x<0 for x in dd);summary['tied_pairs']=sum(x==0 for x in dd)
        rest=[r['delta_Dice'] for r in pp if r['pair']!='an:4rvro9'];summary['delta_Dice_without_4rvro9']=float(np.mean(rest))
        lines.append(f"开发Dice改善{summary['improved_pairs']}/4，持平{summary['tied_pairs']}，下降{summary['declined_pairs']}；其他3对（不含4rvro9）平均ΔDice {summary['delta_Dice_without_4rvro9']:+.6f}，仅作敏感性说明，不改变主表。")
        ta=base['train_fit']['train'];tb=result['last_train'];tkeys=['dice','iou','GT_recall','pred_area','real_fpr','fake_nonedit_FPR'];td={k:tb[k]-ta[k] for k in tkeys};summary['train_difference']=td
        lines+=['','## 原48段同口径训练回测',table(['条件','Dice','IoU','GT召回','fake预测面积','real FPR','fake非编辑区FPR'],[[name]+[v[k] for k in tkeys] for name,v in [('原基线',ta),('FRAME_BCE_OFF',tb),('OFF−ON',td)]]),
                '训练指标按视频聚合再fake宏平均；训练只覆盖反向使用的原48段，开发覆盖完整视频。fake非编辑区FPR分母是GT负像素数，每帧→视频→fake宏平均，不能以real FPR代替。']
        aa=read(FIT/'RGB_DEPTH_PARTITION_last/evaluation.json')['records'];bb=read(OUT/'last_train.json')['records']
        for old in aa:
            if old['role']!='fake':continue
            new=next(r for r in bb if r['video_id']==old['video_id']);real=next(r for r in bb if r['pair_id']==old['pair_id'] and r['role']=='real')
            summary['paired_train'].append({'pair':old['pair_id'],'on_Dice':old['dice'],'off_Dice':new['dice'],'delta_Dice':new['dice']-old['dice'],'off_IoU':new['iou'],'off_recall':new['gt_recall'],'off_pred_area':new['pred_area'],'off_fake_nonedit_FPR':new['fake_nonedit_FPR'],'off_real_FPR':real['empty_fpr']})
        tr=summary['paired_train'];lines+=['',table(list(tr[0]),[list(r.values()) for r in tr])]
        base_records=read(BASE/'RGB_DEPTH_PARTITION/last_dev.json')['records'];z=torch.tensor([x for r in base_records for x in r['scores']],dtype=torch.float32);y=torch.tensor([x for r in base_records for x in r['labels']],dtype=torch.float32);bp=float(F.binary_cross_entropy_with_logits(z[y.bool()],y[y.bool()]))
        lines+=['','## 帧指标仅诊断：OFF未接受帧BCE监督',table(['条件','开发帧AUROC','开发帧AP','开发正GT帧BCE'],[['基线',a['frame']['auroc'],a['frame']['ap'],bp],['OFF（未帧监督）',b['frame']['auroc'],b['frame']['ap'],b['positive_frame_bce']]]),
                '帧头仍计算并留在原AdamW参数组；0权重产生的零梯度是预期，不作为路径缺失，不把该头指标当正式检测能力或消融胜负标准。',
                '## best补充，不替代第12轮',json.dumps({'baseline_best_epoch':base['last']['best_epoch'],'baseline_best_dev':base['last']['best_dev'],'OFF_best_epoch':result['best_epoch'],'OFF_best_dev':result['best_dev']},ensure_ascii=False)]
        summary['conclusion']=f"训练Dice变化{td['dice']:+.6f}；开发Dice变化{delta['dice']:+.6f}，改善{summary['improved_pairs']}/4，real误报变化{delta['real_fpr']:+.6f}。"
        if td['dice']>0 and delta['dice']>0:summary['conclusion']+='当前配置下帧监督可能拖累定位，值得研究任务协调；仍须看多视频方向及误报，不确定具体层的梯度冲突。'
        else:summary['conclusion']+='未同时改善训练和开发定位，不宣称帧BCE普遍拖累定位。'
        lines+=['',summary['conclusion']]
        for key in ['paired_dev','paired_train']:
            with (OUT/(key+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(summary[key][0]));w.writeheader();w.writerows(summary[key])
    else:lines.append('本条件未完成576步，不判主比较胜负。保留完整轮、部分更新与真实退出记录，不重试。')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bh=read(BASE/'RGB_DEPTH_PARTITION/history.json');nh=read(OUT/'history.json') if (OUT/'history.json').exists() else []
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for h,name in [(bh,'frame BCE ON'),(nh,'frame BCE OFF')]:
        x=[r['updates'] for r in h]
        axs[0,0].plot(x,[r['dev']['dice'] for r in h],label=name)
        axs[0,1].plot(x,[r['terms'][1] for r in h],label=name)
        axs[1,0].plot(x,[r['terms'][2] for r in h],label=name)
        axs[1,1].plot(x,[r['terms'][0] for r in h],label=name)
    for ax,title in zip(axs.flat,['Development Dice','Pixel BCE (raw)','Soft Dice loss (raw)','Frame BCE (unweighted; OFF diagnostic)']):ax.set(xlabel='Updates',title=title);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'curves.png');plt.close(fig)
    lines+=['','![同更新数曲线](curves.png)','评价记录中的metrics.loss沿用原三项诊断和，不是OFF优化目标，不据此比较总loss。曲线分开比较两项原始空间loss，保留未加权frame BCE；新优化目标是pixel BCE+soft Dice，没有与旧三项总loss直接比大小。逐次更新日志记录原始terms和实际loss，逐轮history记录optimization_loss/spatial_loss/unweighted_frame_BCE。',
            '资源：'+json.dumps({k:result.get(k) for k in ['updates','last_epoch','wall_s','peak_allocated','peak_reserved','stop_reason']},ensure_ascii=False),'产物：'+str(OUT),'本轮只运行一个新条件，不重训基线、不追加其他权重/模型/seed/数据。计算与发布状态独立。']
    text='\n\n'.join(lines)+'\n';save(OUT/'summary.json',summary);(OUT/'report.md').write_text(text);(REPO/DOC).write_text(text.replace('](curves.png)',']('+str(OUT/'curves.png')+')'))
    final={'status':summary['status'],'stage':'report_complete','exit_code':exit['exit_code'],'pid':os.getpid()};save(OUT/'status.json',final);save(OUT/'progress.json',final)


def publish():
    try:
        git('add',DOC);git('commit','-m','Report fixed-budget ANAKIN frame BCE ablation')
        head=git('rev-parse','HEAD');subprocess.run(['git','push','origin',BRANCH],cwd=REPO,check=True,timeout=120)
        save(OUT/'publication.json',{'status':'PUSHED','head':head});remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        save(OUT/'publication.json',{'status':'COMPLETE' if remote==head else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','head':git('rev-parse','HEAD'),'error':traceback.format_exc()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true');a=parser.parse_args()
    if a.worker:return worker()
    if git('branch','--show-current')!=BRANCH:raise ValueError('WRONG_BRANCH')
    if OUT.exists():raise ValueError('EXISTING_RUN_INSPECT_NO_RETRY')
    if git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice'):raise ValueError('UNCOMMITTED_SOURCE')
    try:
        prepare();sys.stdout=(OUT/'parent.log').open('a',buffering=1);sys.stderr=sys.stdout
        start=time.monotonic()
        with (OUT/'worker.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--worker'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(OUT/'process.json',{'worker_pid':child.pid,'parent_pid':os.getpid()});timeout=False
            try:code=child.wait(timeout=LIMIT)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        save(OUT/'exit.json',{'exit_code':code,'timed_out':timeout,'wall_s':time.monotonic()-start});report();publish()
        return 0 if read(OUT/'summary.json')['status']=='COMPLETE' else 2
    except BaseException:
        if OUT.exists():
            detail=traceback.format_exc();(OUT/'parent_error.txt').write_text(detail)
            if not (OUT/'summary.json').exists():save(OUT/'status.json',{'status':'FAILED','stage':'identity_or_parent_error','error':detail})
        raise


if __name__=='__main__':raise SystemExit(main())
