"""One complete-B fit diagnostic; formal run is strictly a read-only input.

CLI -> existing make/update/evaluate -> independent evidence/report. The old
fitting() is A-only; a thin loop avoids changing its formal training behavior.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

from .experiment import CONDITIONS,REPO,git,make,optimizer,update,evaluate,load_clip,grad_info,state_hash
from .data import save
from scripts.prepare_v8_assets import digest

SOURCE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1')
OUTPUT=SOURCE.parent/'pvst_rgbd_slice_b_fit_check_v1'
DOC='docs/pvst/rgbd_slice_b_fit_check_v1.md'


def separate_output(source,out):
    source=source.resolve();out=out.resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('DIAGNOSTIC_OUTPUT_MUST_BE_SEPARATE')


def inputs(source):
    c=json.loads((source/'config.json').read_text());m=json.loads((source/'manifest.json').read_text());a=json.loads((source/'fit/result.json').read_text());h=json.loads((source/'fit/history.json').read_text())
    if Path(c['output']).resolve()!=source.resolve():raise ValueError('FORMAL_GEOMETRY_ROOT')
    rr=[r for r in m['rows'] if r['split']=='train' and r['pair_id']==a['pair']];cc=a['clips']
    if len(rr)!=2 or {r['role'] for r in rr}!={'real','fake'} or len(cc)!=2:raise ValueError('FIXED_PAIR_IDENTITY')
    if [r['video_id'] for r in rr]!=[x['video_id'] for x in cc]:raise ValueError('ORIGINAL_ALTERNATING_ORDER')
    if [r['role'] for r in rr]!=['fake','real']:raise ValueError('SAVED_A_ORDER_NOT_FAKE_REAL')
    for clip in cc:
        if clip not in m['clips'] or clip['end']-clip['start']!=24:raise ValueError('CLIP_MISMATCH')
        r=next(r for r in rr if r['video_id']==clip['video_id']);pts=r['pts'][clip['start']:clip['end']]
        if len(pts)!=24 or not all(y['pts_s']>x['pts_s'] for x,y in zip(pts,pts[1:])):raise ValueError('PTS_MISMATCH')
    if len(h)!=a['updates'] or [x['update'] for x in h]!=list(range(1,len(h)+1)):raise ValueError('A_HISTORY_UPDATES')
    if [x['update'] for x in h if 'monitor' in x]!=list(range(10,len(h)+1,10)):raise ValueError('A_EVALUATION_FREQUENCY')
    protocol={'seed':c['seed'],'max_updates':150,'budget_seconds':2700,'evaluate_every':10,'goal_dice':.8,'goal_real_fpr':.05,
              'clear_dice':c['fit_clear_dice'],'clear_recall':c['fit_clear_recall'],'clear_real_fpr':.05,'order':['fake','real'],'encoder_lr':c['encoder_lr'],'new_lr':c['new_lr'],'weight_decay':.01,'clip_norm':1,'terms':'frame BCE + pixel BCE + Dice loss; frame/clip mean; empty GT epsilon1',
              'budget_clock':'as original A: initial evaluation + updates + intermediate evaluations; model/load before clock; final evaluation after stopping',
              'configuration_difference':[]}
    for key,value in [('fit_updates',150),('fit_seconds',2700),('fit_goal_dice',.8),('fit_goal_real_fpr',.05),('weight_decay',.01)]:
        if c.get(key)!=value:protocol['configuration_difference'].append({'key':key,'saved':c.get(key),'actual_A_code':value})
    return c,m,a,h,rr,cc,protocol


def complete_b(c,source):
    model=make(c,source,CONDITIONS[1])
    if not model.relation or any(not hasattr(model,name) for name in ['pool','time','depth','pixel_context','frame_context']):raise ValueError('NOT_COMPLETE_B')
    common=json.loads((source/'initialization.json').read_text())['common_sha256']
    if state_hash(model.local.state_dict())!=common:raise ValueError('COMMON_WEIGHTS_CHANGED')
    if any(torch.count_nonzero(p) for module in [model.pixel_context,model.frame_context] for p in module.parameters()):raise ValueError('RESIDUAL_NOT_ORIGINAL_ZERO_INITIALIZATION')
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('UNAUTHORIZED_FREEZING')
    return model


def source_snapshot(source):
    # Metadata check covers all formal files without loading trained checkpoints.
    return {str(p.relative_to(source)):[p.stat().st_size,p.stat().st_mtime_ns] for p in source.rglob('*') if p.is_file()}


def history_review(source):
    result={}
    for condition in CONDITIONS:
        folder=source/condition;history=json.loads((folder/'history.json').read_text());saved=list(csv.DictReader((folder/'history.csv').open()))
        if len(saved)!=len(history):raise ValueError('FORMAL_CSV_JSON_LENGTH')
        keys=['frame_BCE','pixel_BCE','dice_loss']
        for row,line in zip(history,saved):
            if int(line['epoch'])!=row['epoch']:raise ValueError('FORMAL_CSV_EPOCH')
            for key in keys:
                if float(line[key])!=row[key]:raise ValueError('FORMAL_CSV_LOSS')
            for key,value in [('dev_dice',row['dev']['dice']),('dev_frame_auroc',row['dev']['frame']['auroc']),('train_monitor_clips_dice',row['train_monitor_clips']['dice'])]:
                if float(line[key])!=value:raise ValueError('FORMAL_CSV_METRICS')
        summary=json.loads((folder/'summary.json').read_text());best=next(r for r in history if r['epoch']==summary['best_epoch']);auc=max(history,key=lambda r:r['dev']['frame']['auroc']);peak=max(history,key=lambda r:r['train_monitor_clips']['dice'])
        result[condition]={'losses':{k:{'first':history[0][k],'last':history[-1][k],'minimum':min(r[k] for r in history)} for k in keys},
                           'spatial_best_epoch':best['epoch'],'spatial_best_dev':best['dev'],'spatial_best_monitor':best['train_monitor_clips'],
                           'highest_saved_dev_frame_AUROC':{'epoch':auc['epoch'],'dev':auc['dev'],'monitor':auc['train_monitor_clips']},
                           'highest_monitor_dice':{'epoch':peak['epoch'],'metrics':peak['train_monitor_clips']},
                           'epochs':[{'epoch':r['epoch'],**{k:r[k] for k in keys},'dev_dice':r['dev']['dice'],'dev_frame_auroc':r['dev']['frame']['auroc'],'monitor_dice':r['train_monitor_clips']['dice'],'monitor_GT_recall':r['train_monitor_clips']['GT_recall'],'monitor_real_fpr':r['train_monitor_clips']['real_fpr'],'monitor_frame_auroc':r['train_monitor_clips']['frame']['auroc']} for r in history]}
    return result


class ContextStats:
    """Scalar-only hooks on the two residual projections; no tensor retention."""
    def __init__(self,model):
        self.active=False;self.values={};self.handles=[]
        for name in ['pixel_context','frame_context']:
            def collect(module,args,output,key=name):
                if self.active:
                    a=output.detach().float();s=self.values.setdefault(key,{'absolute_sum':0.,'elements':0,'calls':0})
                    s['absolute_sum']+=float(a.abs().sum());s['elements']+=a.numel();s['calls']+=1
            self.handles.append(getattr(model,name).register_forward_hook(collect))
    def start(self):self.values={};self.active=True
    def finish(self):
        self.active=False
        return {k:{'mean_absolute':s['absolute_sum']/max(1,s['elements']),'elements':s['elements'],'calls':s['calls'],'scope':'occupied-slice projection elements, before native broadcast'} for k,s in self.values.items()}
    def close(self):
        for h in self.handles:h.remove()


def projection_norms(model):return {name:float(getattr(model,name).weight.detach().float().norm()) for name in ['pixel_context','frame_context']}


def publish(out):
    try:
        git('add',DOC);git('commit','-m','Report complete B fixed-pair fitting diagnostic and history review')
        head=git('rev-parse','HEAD');git('push','origin','pvst-rgbd-slice-relations');remote=git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/pvst-rgbd-slice-relations').split()[0]
        save(out/'publication.json',{'status':'COMPLETE' if head==remote else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(out/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'head':git('rev-parse','HEAD')})


def report(out,result,review,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=result['history'];measured=[r for r in history if 'monitor' in r]
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for i,key in enumerate(['frame_BCE','pixel_BCE','dice_loss']):axes[0].plot([r['update'] for r in history],[r['terms'][i] for r in history],label=key)
    for name,h in [('A',a['history']),('B',history)]:
        hh=[r for r in h if 'monitor' in r];axes[1].plot([r['update'] for r in hh],[r['monitor']['dice'] for r in hh],label=name)
    for r in measured:axes[2].scatter(r['update'],r['monitor']['GT_recall'],color='C0');axes[2].scatter(r['update'],r['monitor']['real_fpr'],color='C1')
    axes[0].legend();axes[1].legend();axes[1].set_title('Fixed fake clip Dice');axes[2].set_title('B recall (blue) / real FPR (orange)');fig.tight_layout();fig.savefig(out/'curves.png');plt.close(fig)
    b=result.get('final',{}).get('metrics');av=a['final']['metrics'];lines=['# 完整B固定pair拟合诊断与正式history审阅',f'计算状态：{result["status"]}，退出码{result["exit_code"]}。拟合判断：{result.get("fit_outcome","未完成")}。',
        '固定an:7ogg04；fake cb41b3d004b6640f6c3b5bb1、real c26e3fc1fdc6aabda308ae84，两段均[0,24)，PTS 0–0.92秒。fake→real交替；原make(B)+common_initial，不加载正式best/last或拟合A，不重提几何。',
        '原A实际协议：150次/约2700秒上限、每10次评价；主要目标Dice≥.8且real FPR≤.05；次级判据Dice≥.6、GT召回≥.6且FPR≤.05，分别报告。像素AP原A未计算，本轮不补。所有分项损失/优化器/BN/Dropout保持原实现。',
        '|固定片段诊断|更新数|fake Dice|IoU|GT召回|real FPR|预测前景面积(fake)|', '|---|---:|---:|---:|---:|---:|---:|',
        '|原A|'+ '|'.join(map(str,[a['updates'],av['dice'],av['iou'],av['GT_recall'],av['real_fpr'],av['pred_area']]))+'|']
    if b:lines.append('|完整B|'+ '|'.join(map(str,[result['updates'],b['dice'],b['iou'],b['GT_recall'],b['real_fpr'],b['pred_area']]))+'|')
    lines+=['A拟合用时'+str(a['seconds'])+'秒；B分阶段用时：'+json.dumps(result['timing'],ensure_ascii=False)+'。此为不同模型的诊断成本，不是等更新数速度比。',
            '|分支|第1次梯度范数|第2次|最后一次采样|采样梯度存在且有限|','|---|---:|---:|---:|---|']
    sampled=[r for r in history if r.get('gradients')]
    if sampled:
        for name in sampled[0]['gradients']:
            get=lambda r:r['gradients'].get(name,{})
            chosen=[sampled[0],sampled[1] if len(sampled)>1 else sampled[0],sampled[-1]]
            lines.append('|'+ '|'.join([name,*[str(get(r).get('norm')) for r in chosen],str(all(get(r).get('present') and get(r).get('finite') for r in sampled))])+'|')
    lines+=['最终两个残差投影权重范数：'+json.dumps(result.get('final_projection_weight_norms'),ensure_ascii=False)]
    for r in sampled[:2]:lines.append('第'+str(r['update'])+'次训练投影输出：'+json.dumps(r.get('context_outputs'),ensure_ascii=False))
    lines.append('最终eval投影输出：'+json.dumps(result.get('final_context_outputs'),ensure_ascii=False))
    for name,value in review.items():
        lines += [f'## 正式{name}（原记录只读）','三项在线训练loss：'+json.dumps(value['losses'],ensure_ascii=False),
                  '监测最高：epoch'+str(value['highest_monitor_dice']['epoch'])+'，Dice='+str(value['highest_monitor_dice']['metrics']['dice']),
                  '历史开发帧AUROC最高（不替换正式checkpoint）：epoch'+str(value['highest_saved_dev_frame_AUROC']['epoch'])+'，AUROC='+str(value['highest_saved_dev_frame_AUROC']['dev']['frame']['auroc'])+'，该轮开发Dice='+str(value['highest_saved_dev_frame_AUROC']['dev']['dice']),
                  '正式空间best轮次'+str(value['spatial_best_epoch'])+'，该轮监测Dice='+str(value['spatial_best_monitor']['dice'])+'、开发帧AUROC='+str(value['spatial_best_dev']['frame']['auroc'])+'。']
    lines += ['B监测Dice第11轮.7654→第12轮.1263→第13轮0，后期反复低值；不是始终不能拟合，也非单调下降。三项在线loss总体下降不代表这个固定pair定位稳定。监测只有1pair/两条24帧片段，不代表全训练集。',
              'B在线frame BCE从1.0995降至.4026，开发frame AUROC第6轮.6539高于空间best第12轮.4717，随后未保持；A最高开发frame AUROC第6轮.5244，空间best是第5轮.5144。监测帧AUROC两者早期达到1，但固定阈值检测仍可能偏向同一类。历史检测与定位最优不同步，不能替换best或拼成绩。',
              '正式history未保存逐轮关系残差/分支梯度，能确认随训练的定位波动，不能仅凭时间顺序认定关系支路导致退化。']
    if result.get('goal_reached'):
        lines+=['## 判断','完整B达到与原A相同的主要拟合目标，排除了“完整B没有基本小样本拟合能力”的解释。结合正式记录，当前证据最支持多pair训练中空间优化不稳定，而不是直接归因于相机运动或三维无效。',
                '唯一优先下一步：定位多pair训练第11→12轮监测定位骤降的更新干扰，先查空间优化与残差变化，不扩量。']
    elif result.get('clear_fit'):
        lines+=['## 判断','B明显改善但未达到主要目标，不能与A的主要目标通过等同。结合梯度与残差记录，优先检查关系接入后的空间优化稳定性。','唯一优先下一步：针对本固定pair定位学习停滞的位置检查空间梯度与残差贡献，不扩量。']
    elif b:
        lines+=['## 判断','B未达到原明显拟合判据。直接证据是本表的覆盖/前景面积和梯度/残差记录；低Dice不等于程序FAILED。','唯一优先下一步：定位固定pair中空间梯度与上下文残差的失衡，不启动完整重训。']
    else:lines+=['## 执行错误',str(result.get('error')),'唯一优先下一步：依据真实错误修复执行问题，当前不能判断B拟合能力。']
    lines+=['正式文件未变：'+str(result.get('formal_unchanged'))+'；本轮执行源码：'+result['execution_head'],
            '产物：`'+str(out)+'`，固定样例samples/index.html，机器摘要result.json、formal_history_review.json。检查权重不进入正式模型，本轮到此停止。']
    text='\n\n'.join(lines)+'\n';(out/'report.md').write_text(text);(REPO/DOC).write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args();source=args.source.resolve();out=args.output.resolve();separate_output(source,out)
    if git('branch','--show-current')!='pvst-rgbd-slice-relations':raise ValueError('WRONG_BRANCH')
    if git('status','--porcelain','--','research_tools/rgbd_slice/b_fit_check.py'):raise ValueError('UNCOMMITTED_DIAGNOSTIC_SOURCE')
    c,m,a,ah,rr,cc,protocol=inputs(source);protected=['config.json','manifest.json','fit/result.json','fit/history.json','initialization.json','common_initial.pt']
    hashes={name:digest(source/name) for name in protected};code={str(p.relative_to(REPO)):digest(p) for p in (REPO/'research_tools/rgbd_slice').glob('*.py')};code['PairRelation']=digest(REPO/'research_tools/depth_slice/model.py')
    identity=hashlib.sha256(json.dumps({'source':str(source),'hashes':hashes,'code':code,'protocol':protocol},sort_keys=True).encode()).hexdigest()
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())['identity']!=identity:raise ValueError('DIAGNOSTIC_IDENTITY_CHANGED')
        if (out/'result.json').exists() and json.loads((out/'result.json').read_text())['status']=='COMPLETE':print('REUSE_COMPLETE',out);return
        raise ValueError('EXISTING_INCOMPLETE_RUN_NO_AUTO_RETRY')
    out.mkdir(parents=True,exist_ok=True);head=git('rev-parse','HEAD');save(out/'identity.json',{'identity':identity,'execution_head':head,'source_hashes':hashes,'code':code});save(out/'config.json',{'formal_config':c,'source':str(source),'diagnostic_output':str(out),'actual_A_protocol':protocol,'sample_display_only_change':'also save real clip previews; numeric evaluation unchanged'})
    save(out/'selection.json',{'pair':a['pair'],'rows':rr,'clips':cc,'clip_pts':{r['video_id']:r['pts'][x['start']:x['end']] for r,x in zip(rr,cc)}})
    snapshot=source_snapshot(source);review=history_review(source);save(out/'formal_history_review.json',review)
    started=time.monotonic();history=[];timing={'model_construct_s':0.,'data_load_s':0.,'initial_eval_s':0.,'updates_s':0.,'intermediate_eval_s':0.,'final_eval_s':0.};model=None;collector=None
    result={'status':'FAILED','exit_code':1,'history':history,'updates':0,'timing':timing,'execution_head':head};exit_code=1
    def progress(stage,**kw):save(out/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'wall_s':time.monotonic()-started,**kw});print(stage,kw,flush=True)
    try:
        torch.set_num_threads(4);progress('construct_complete_B');t=time.monotonic();model=complete_b(c,source);opt=optimizer(model,c);timing['model_construct_s']=time.monotonic()-t
        save(out/'model_identity.json',{'condition':CONDITIONS[1],'common_sha256':state_hash(model.local.state_dict()),'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'initial_projection_norms':projection_norms(model),'gpu':torch.cuda.get_device_name(),'gpu_free_total':torch.cuda.mem_get_info()})
        t=time.monotonic();fs=[load_clip(r,x,c) for r,x in zip(rr,cc)];timing['data_load_s']=time.monotonic()-t
        for row,clip,ff in zip(rr,cc,fs):
            if [f['frame_index'] for f in ff]!=list(range(clip['start'],clip['end'])) or [f['pts'] for f in ff]!=[p['pts_s'] for p in row['pts'][clip['start']:clip['end']]]:raise ValueError('LOADED_FRAME_IDENTITY')
        eval_c={**c,'sample_video_ids':[r['video_id'] for r in rr]};collector=ContextStats(model);bn=state_hash(dict(model.local.encoder.named_buffers()));torch.cuda.reset_peak_memory_stats()
        fit_clock=time.monotonic();t=time.monotonic();initial=evaluate(model,rr,eval_c,progress,clip_list=cc,samples=out/'initial_samples',pixel_ap=False);timing['initial_eval_s']=time.monotonic()-t;save(out/'initial.json',initial);goal=False
        for i in range(1,151):
            if time.monotonic()-fit_clock>=2700:break
            sample=i in [1,2] or i%10==0
            if sample:collector.start()
            progress('B_fixed_pair_fit',update=i,role=rr[(i-1)%2]['role']);record=update(model,opt,fs[(i-1)%2],diagnose=sample);timing['updates_s']+=record['seconds'];record.update(update=i,role=rr[(i-1)%2]['role'],total_loss=sum(record['terms']),projection_weight_norms=projection_norms(model))
            if sample:record['context_outputs']=collector.finish()
            history.append(record);result['updates']=i
            if i%10==0 or i==150 or time.monotonic()-fit_clock>=2700:
                t=time.monotonic();observed=evaluate(model,rr,eval_c,progress,clip_list=cc,samples=out/'samples' if i%25==0 else None,pixel_ap=False);timing['intermediate_eval_s']+=time.monotonic()-t;record['monitor']=observed['metrics'];save(out/'history.json',history)
                goal=observed['metrics']['dice']>=.8 and observed['metrics']['real_fpr']<=.05;print('B_FIT_MONITOR',i,json.dumps(observed['metrics']),flush=True)
                if goal:break
        if history and not history[-1].get('gradients'):history[-1]['gradients']=grad_info(model);history[-1]['gradient_scope']='post-clipping last-update gradients'
        result['fit_budget_elapsed_s']=time.monotonic()-fit_clock;t=time.monotonic();collector.start();final=evaluate(model,rr,eval_c,progress,clip_list=cc,samples=out/'samples',pixel_ap=False);result['final_context_outputs']=collector.finish();timing['final_eval_s']=time.monotonic()-t
        metrics=final['metrics'];clear=metrics['dice']>=c['fit_clear_dice'] and metrics['GT_recall']>=c['fit_clear_recall'] and metrics['real_fpr']<=.05
        if state_hash(dict(model.local.encoder.named_buffers()))!=bn:raise ValueError('BN_STATISTICS_CHANGED')
        result.update(status='COMPLETE',exit_code=0,fit_outcome='MAIN_GOAL_REACHED' if goal else 'CLEAR_FIT_BELOW_MAIN_GOAL' if clear else 'FIT_NOT_ESTABLISHED',goal_reached=goal,clear_fit=clear,initial=initial,final=final,initial_to_final_dice=metrics['dice']-initial['metrics']['dice'],final_projection_weight_norms=projection_norms(model),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        exit_code=0
    except BaseException as error:
        detail=traceback.format_exc();(out/'error.txt').write_text(detail);print(detail,flush=True);result.update(status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'FAILED',error=detail)
    finally:
        if collector:collector.close()
        result['formal_unchanged']=source_snapshot(source)==snapshot;result['wall_s']=time.monotonic()-started
        if not result['formal_unchanged']:result.update(status='FAILED',exit_code=1,error='FORMAL_SOURCE_FILES_CHANGED');exit_code=1
        save(out/'history.json',history);save(out/'result.json',result)
        flat=[]
        for r in history:flat.append({'update':r['update'],'role':r['role'],'total_loss':r['total_loss'],'frame_BCE':r['terms'][0],'pixel_BCE':r['terms'][1],'dice_loss':r['terms'][2],'update_s':r['seconds'],**{k:r.get('monitor',{}).get(k) for k in ['dice','iou','GT_recall','real_fpr','pred_area']},**{k:r['projection_weight_norms'][k] for k in ['pixel_context','frame_context']}})
        with (out/'history.csv').open('w',newline='') as f:
            if flat:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        report(out,result,review,a);save(out/'progress.json',{'status':result['status'],'stage':'report_complete','fit_outcome':result.get('fit_outcome'),'pid':os.getpid(),'exit_code':exit_code});(out/'exit_code.txt').write_text(str(exit_code)+'\n');publish(out)
    raise SystemExit(exit_code)


if __name__=='__main__':main()
