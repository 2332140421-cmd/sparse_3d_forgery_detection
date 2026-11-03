"""PVST v0.3 bounded pipeline: scale -> resources -> exactly one epoch/condition.

Immediate CLI caller; emits independent v0.3 checkpoints/metrics/report. Reuses
PVST, Video, resource worker and frozen evaluator; never calls v0.2 train.main.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from scripts.prepare_v8_assets import digest
from research_tools.v8.local_geometry_pilot.runner import atomic_torch,rng_state,restore_rng
from .data import Video,save,select_clips
from .probe import select_v03_scale,parameters,losses
from .train import build,initialization_hash
from .evaluate import evaluate
from .compact_check import worker,host_memory

REPO=Path(__file__).resolve().parents[2]
BRANCH='pvst-point-voxel-v03'


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def clip_terms(predictions,ground_truth,labels,epsilon):
    if not (len(predictions)==len(ground_truth)==len(labels)>0):raise ValueError('TARGET_SUPERVISION_COVERAGE')
    return torch.stack([torch.stack(losses(p,g,y,epsilon)) for p,g,y in zip(predictions,ground_truth,labels)]).mean(0)


def paired_order(clips,seed):
    rng=random.Random(seed);groups=[[i for i,c in enumerate(clips) if c['dataset']==d] for d in ['ActivityForensics','ANAKIN']]
    for group in groups:rng.shuffle(group)
    return [[g[i] for g in groups if i<len(g)] for i in range(max(map(len,groups)))]


def protocol(config,out):
    source=Path(config['asset_root']);metadata={p.parent.name:digest(p) for p in Path(config['cache_root']).glob('*/complete.json')}
    return {'config':config,'manifest_sha256':digest(source/'manifest.json'),'scale_sha256':digest(source/'scale.json'),
        'clips_sha256':digest(out/'clips.json'),'scale_selection_sha256':digest(out/'scale_selection.json'),
        'cache_metadata_sha256':metadata,'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'source_commit':git('rev-parse','HEAD')}


def freeze_inputs(c,out,progress):
    source=Path(c['asset_root']);rows=json.loads((source/'manifest.json').read_text())['rows'];scale=json.loads((source/'scale.json').read_text())['s']
    assert len(rows)==48 and sum(r['actual_frames'] for r in rows)==22074
    # Headers/identity only. Missing/incompatible caches are errors, never extraction.
    for r in rows:
        p=Path(c['cache_root'])/r['video_id'];meta=json.loads((p/'complete.json').read_text())
        if meta['video_sha256']!=r['video_sha256'] or meta['prepared_frames']!=r['actual_frames'] or meta['tokens']!='no LoRA; 448 letterbox; image patch tokens only':raise ValueError('CACHE_INCOMPATIBLE:'+r['video_id'])
        for name,shape,dtype in [('rgb',(r['actual_frames'],*r['source_hw'],3),np.uint8),('tokens',(r['actual_frames'],1024,32,32),np.float16)]:
            a=np.load(p/(name+'.npy'),mmap_mode='r')
            if a.shape!=shape or a.dtype!=dtype:raise ValueError('CACHE_HEADER:'+r['video_id']+name)
    for name in ['manifest.json','scale.json']:
        p=out/name
        if p.exists() and digest(p)!=digest(source/name):raise ValueError('FROZEN_INPUT_CHANGED')
        if not p.exists():p.write_bytes((source/name).read_bytes())
    save(out/'config.json',c)
    counts=select_v03_scale(c,out,progress)
    if counts['selected_h'] is None:return c,rows,scale,counts,[]
    c={**c,'voxel_sizes':{**c['voxel_sizes'],'XYZ_PVST':counts['selected_h']}}
    clipfile=out/'clips.json'
    if clipfile.exists():clips=json.loads(clipfile.read_text())['clips']
    else:
        clips=select_clips(rows,out,c,scale);save(clipfile,{'seed':c['seed'],'manifest_sha256':digest(out/'manifest.json'),'clips':clips,'order':paired_order(clips,c['seed'])})
    sources={d:sorted({r['source_id'] for r in rows if r['split']=='train' and r['dataset']==d})[:1] for d in ['ActivityForensics','ANAKIN']}
    c['monitor_video_ids']=[r['video_id'] for r in rows if r['split']=='train' and r['source_id'] in sources[r['dataset']]]
    c['sample_video_ids']=[next(r['video_id'] for r in rows if r['split']=='dev' and r['dataset']==d and r['role']==role) for d in ['ActivityForensics','ANAKIN'] for role in ['fake','real']]
    save(out/'config.json',c);return c,rows,scale,counts,clips


def resource_checks(c,out,counts,identity,progress):
    records=[]
    for w in counts['representatives']:
        for condition in ['XYZ_PVST','UV_PVST']:
            for offload in [False,True]:
                spec={'config':c,'window':w,'condition':condition,'offload':offload,'identity':identity}
                name=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12];p=out/f'resource_{name}.json'
                if not p.exists():
                    save(out/f'spec_{name}.json',spec);progress('resource',condition=condition,window=w,offload=offload)
                    with (out/f'resource_{name}.log').open('w') as log:
                        subprocess.run([sys.executable,'-u','-m','research_tools.pvst.v03','--worker',str(out/f'spec_{name}.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
                saved=json.loads(p.read_text())
                if saved['spec']!=spec:raise ValueError('RESOURCE_IDENTITY_CHANGED')
                r=saved['result'];records.append(r)
                if r['status']=='PASS':break
                if r['status']!='RESOURCE_BLOCKED' or offload:save(out/'resources.json',{'status':r['status'],'records':records});return False,records
                if host_memory()['available_bytes']<16*2**30:save(out/'resources.json',{'status':'RESOURCE_BLOCKED','records':records,'reason':'host available <16GiB'});return False,records
    # One short profile only if backward remains disproportionately expensive.
    if any(sum(u['backward_s'] for u in r['updates'])>30 for r in records if r['status']=='PASS'):
        r=max((r for r in records if r['status']=='PASS'),key=lambda r:sum(u['backward_s'] for u in r['updates']))
        path=out/'short_profile.json'
        if not path.exists():
            spec={'config':c,'window':r['window'],'condition':r['condition'],'offload':r['offload'],'profile':True,'identity':identity}
            save(out/'spec_short_profile.json',spec)
            with (out/'short_profile.log').open('w') as log:subprocess.run([sys.executable,'-u','-m','research_tools.pvst.v03','--worker',str(out/'spec_short_profile.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
            (out/'resource_short_profile.json').replace(path)
    save(out/'resources.json',{'status':'PASS','records':records});return True,records


def train_one(c,out,rows,scale,clips,condition,identity,progress):
    folder=out/condition;folder.mkdir(exist_ok=True);summaryfile=folder/'summary.json'
    if summaryfile.exists():return json.loads(summaryfile.read_text())
    model=build(c,condition);model.check_bn();initial=initialization_hash(model)
    if sum(p.numel() for p in model.parameters())!=17821672:raise ValueError('PARAMETER_COUNT_CHANGED')
    opt=torch.optim.AdamW(parameters(c,model),weight_decay=c['weight_decay']);order=paired_order(clips,c['seed']);updates=[];position=0
    checkpoint=folder/'resume.pt'
    def state(epoch,pos):
        return {'identity':identity,'condition':condition,'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
            'clip_order':order,'completed_update_position':pos,'checkpoint_epoch':epoch,'updates':updates,'initial_parameters_sha256':initial}
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['clip_order']!=order:raise ValueError('RESUME_IDENTITY')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng']);updates=cp['updates'];position=cp['completed_update_position'];del cp
    else:
        atomic_torch(folder/'epoch0.pt',state(0,0));atomic_torch(checkpoint,state(0,0))
    dev=[r for r in rows if r['split']=='dev'];monitor=[r for r in rows if r['video_id'] in c['monitor_video_ids']];lookup={r['video_id']:r for r in rows}
    def evaluation(epoch):
        model.eval();sha=digest(folder/f'epoch{epoch}.pt')
        for split,rr in [('dev',dev),('monitor',monitor)]:
            path=folder/f'epoch{epoch}_{split}.json'
            if path.exists():
                if json.loads(path.read_text())['checkpoint_sha256']!=sha:raise ValueError('EVALUATION_CHECKPOINT_CHANGED')
                continue
            value=evaluate(model,rr,out,c,scale,samples=folder/f'epoch{epoch}_samples'/split,progress=progress)
            value.update(epoch=epoch,checkpoint_sha256=sha,scope='complete dev videos' if split=='dev' else 'fixed four complete training monitor videos');save(path,value)
    if any(not (folder/f'epoch0_{split}.json').exists() for split in ['dev','monitor']):
        current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};saved_rng=rng_state()
        zero=torch.load(folder/'epoch0.pt',map_location='cpu',weights_only=False);model.load_state_dict(zero['model']);del zero
        evaluation(0);model.load_state_dict(current);restore_rng(saved_rng);del current
    for i in range(position,len(order)):
        progress('training',condition=condition,epoch=1,update=i+1,updates_total=len(order));model.train();torch.cuda.reset_peak_memory_stats();tick=time.monotonic();opt.zero_grad(set_to_none=True);terms=[];n=len(order[i])
        for index in order[i]:
            clip=clips[index];r=lookup[clip['video_id']];v=Video(out,r,condition,scale,c);left,end=clip['input_range'];start=clip['target_range'][0]
            try:
                fs=v.frames(left,end,'cuda');gt=[v.gt(j) for j in range(start,end)];labels=[v.label(j) for j in range(start,end)]
                if labels!=clip['frame_labels'] or [f['pts'] for f in fs]!=clip['input_pts_s']:raise ValueError('CLIP_LABEL_OR_PTS_CHANGED')
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(fs,start-left);values=clip_terms(pred,gt,labels,c['dice_epsilon']);loss=values[0]/n
                if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
                loss.backward();terms.append(values.detach().float().cpu().numpy()/n);del fs,pred,loss,values
            finally:v.close()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],error_if_nonfinite=True));opt.step();model.check_bn();torch.cuda.synchronize()
        loss,lf,lb,ld=np.sum(terms,axis=0).tolist();item={'update':i+1,'loss':loss,'frame_BCE':lf,'pixel_BCE':lb,'Dice_loss':ld,'lr':[g['lr'] for g in opt.param_groups],'gradient_norm':norm,'update_s':time.monotonic()-tick,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'clip_indices':order[i],
            'clips_processed':sum(len(x) for x in order[:i+1]),'target_frames_processed':sum(clips[j]['target_range'][1]-clips[j]['target_range'][0] for group in order[:i+1] for j in group)}
        updates.append(item);save(folder/'history.json',updates);atomic_torch(checkpoint,state(1,i+1));print('UPDATE',condition,json.dumps(item),flush=True)
    if not (folder/'epoch1.pt').exists():atomic_torch(folder/'epoch1.pt',state(1,len(order)))
    evaluation(1)
    summary={'condition':condition,'epochs':1,'updates':len(updates),'train_update_s':sum(u['update_s'] for u in updates),'initial_parameters_sha256':initial,
        'peak_allocated_bytes':max(u['peak_allocated_bytes'] for u in updates),'peak_reserved_bytes':max(u['peak_reserved_bytes'] for u in updates),'identity':identity}
    save(summaryfile,summary);del model,opt;torch.cuda.empty_cache();return summary


def report(out,c,status,resources,summaries,error=None):
    counts=json.loads((out/'scale_selection.json').read_text()) if (out/'scale_selection.json').exists() else {}
    clips=json.loads((out/'clips.json').read_text())['clips'] if (out/'clips.json').exists() else []
    lines=['# PVST v0.3 单轮贯通报告','',f'计算状态：{status}；模型完成{len(summaries)}/2；每条件最多一轮，不是最终收敛结论。',
        'v0.3改变XYZ结构分支统一体素尺度及片段监督/更新方式，不与v0.2全视频训练宣称等价。扩大体素降低块级结构空间分辨率，点旁路不能保证精度不下降。',
        '逐点9维坐标→geo128/局部卷积→VFE及fine256/coarse512；geo点旁路与逐像素视觉128和所属时空块256进入原像素解码器。全部像素有预测，全部有效点归块，无随机删点/块或邻接截断。几何为dense256场回映原生，并非新增独立高分辨率观测。',
        f'冻结体素尺度：{c["voxel_sizes"]}；候选预算统计：'+json.dumps(counts.get('candidates'),ensure_ascii=False),
        f'复用48视频22,074帧（训练32、开发16），新增前端调用0；训练片段{len(clips)}，去重目标帧'+str(sum(x['target_range'][1]-x['target_range'][0] for x in clips))+'；每视频至多4个互不重叠目标段，历史不监督。',
        '训练每更新AF/AN顺序反向后对参与片段等权平均；片段内frame BCE及AN逐帧pixel BCE/Dice分别取目标帧均值，无旧全视频分母/类别权重。',
        '', '|条件|checkpoint|集合|AF video AUROC/AP/F1/ACC|AF frame AUROC/AP/F1|AN Dice/IoU/pixel AP/real FPR|', '|---|---|---|---|---|---|']
    metrics={}
    for summary in summaries:
        condition=summary['condition'];metrics[condition]={}
        for epoch in [0,1]:
            for split in ['dev','monitor']:
                d=json.loads((out/condition/f'epoch{epoch}_{split}.json').read_text());ds=d['datasets'];af=ds['ActivityForensics'];an=ds['ANAKIN']
                lines.append('|'+ '|'.join([condition,f'epoch{epoch}',split,' / '.join(str(af['video'][k]) for k in ['auroc','ap','f1','acc']),' / '.join(str(af['frame'][k]) for k in ['auroc','ap','f1']),' / '.join(str(an[k]) for k in ['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr'])])+'|');metrics[condition][f'epoch{epoch}_{split}']={k:d[k] for k in ['epoch','checkpoint_sha256','datasets','elapsed_s','voxel_summary']}
    lines+=['','资源记录（非profiler；两更新/两微批；权重丢弃）：']
    for r in resources:
        lines.append(json.dumps({k:r.get(k) for k in ['condition','window','offload','status','fine_nodes','coarse_nodes','native_hw','input_s','total_s','eval_s','peak_allocated_bytes','peak_reserved_bytes','host_before','host_after','updates','phase','error','trace']},ensure_ascii=False))
    if not summaries:lines+=['','尚无一轮训练或评价指标；资源/尺度未通过则停止，不用探针loss代替正式成绩。']
    if summaries:
        costs={s['condition']:{'training_s':s['train_update_s'],'evaluation_s':sum(v['elapsed_s'] for v in metrics[s['condition']].values()),'30_epoch_training_only_range_s':[s['train_update_s']*25,s['train_update_s']*35]} for s in summaries}
        lines+=['','成本：'+json.dumps(costs,ensure_ascii=False),'25–35倍单轮更新墙时仅作30轮训练成本范围，未含每轮完整开发/monitor评价；评价单列。无足够轮次判断best/收敛。']
    lines+=['','资源简表（前向/反向均合计4微批；吞吐按32次目标帧处理计，不是独立帧数）：','|条件|视频/目标|卸载|结果|fine/coarse|allocated/reserved GiB|前向/反向秒|目标帧/秒|','|---|---|---|---|---|---|---|---|']
    for r in resources:
        u=r['updates'];t=sum(x['total_s'] for x in u)
        lines.append('|'+ '|'.join([r['condition'],r['window']['video_id']+str(r['window']['target_range']),str(r['offload']),r['status'],str(r.get('fine_nodes'))+'/'+str(r.get('coarse_nodes')),f"{r['peak_allocated_bytes']/2**30:.2f}/{r['peak_reserved_bytes']/2**30:.2f}",str(round(sum(x['forward_s'] for x in u),2))+'/'+str(round(sum(x['backward_s'] for x in u),2)),f'{32/t:.4f}' if t else '未完成'])+'|')
    if counts.get('windows'):
        for split in ['train','dev']:
            ww=[w for w in counts['windows'] if w['split']==split]
            lines+=['冻结XYZ尺度 '+split+'窗口节点min/median/p95/max：'+json.dumps({k:[float(x) for x in np.percentile([w[k] for w in ww],[0,50,95,100])] for k in ['fine_nodes','coarse_nodes']})]
    if (out/'short_profile.json').exists():
        profile=json.loads((out/'short_profile.json').read_text())['result'];lines+=['独立短profiler（不用于吞吐）：',profile.get('profiler_table',profile.get('error','未记录'))]
    lines+=['','坐标计数秒：'+str(counts.get('seconds','未记录')),'资源case总秒：'+str(sum(r['total_s'] for r in resources)),
        '4090建议：'+('本轮资源及两条件各一轮贯通，结合实测成本决定后续预算，不自动继续。' if status=='COMPLETE' else '当前未通过全部验收，不启动或追加训练；具体阻塞见下。'),
        '本轮输出目录：'+str(out),'完整视频评价使用固定logit>=0，正GT帧先视频内均值再视频均值，空GT/real误报分列；AF无空间GT不报像素定位。固定图25/50/75% PTS、统一概率0–1；不按模型峰值挑帧。']
    if (out/'scale_budget_audit.json').exists():
        audit=json.loads((out/'scale_budget_audit.json').read_text());lines+=['','## 训练坐标预算审计','|XYZ h|fine median/p95/max|coarse median/p95/max|','|---|---|---|']
        for r in audit['XYZ']:
            lines.append('|'+str(r['h'])+'|'+' / '.join(f'{v:,.1f}' for v in r['distribution']['fine'][1:])+'|'+' / '.join(f'{v:,.1f}' for v in r['distribution']['coarse'][1:])+'|')
        lines+=['','最粗1/8最大窗口：'+json.dumps(audit['XYZ'][-1]['fine_max_window'],ensure_ascii=False),'UV原1/32精确原生UV格点占据计数（坐标解析计数，不是GPU实测）：'+json.dumps(audit['UV'],ensure_ascii=False)]
    if (out/'model_identity.json').exists():
        lines+=['','CPU模型身份核对（无前向/训练）：'+(out/'model_identity.json').read_text()]
    if (out/'correctness.txt').exists():
        lines+=['','## 必要正确性','```',(out/'correctness.txt').read_text(),'```']
    if not counts.get('selected_h'):
        lines+=['','三个授权候选均超训练坐标预算；未冻结XYZ尺度，未读取开发坐标选尺度，未运行GPU资源探针/一轮训练/epoch0或epoch1评价。不能提供未测显存、吞吐、分数或30轮成本。未扩展到更粗尺度、删点或改变其他规格。旧4090失败不是本轮重新发生的OOM。']
    if error:lines+=['','错误：','```',error,'```']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n');save(out/'result_summary.json',{'status':status,'formal_models_completed':len(summaries),'source_commit':git('rev-parse','HEAD'),'config':c,'summaries':summaries,'metrics':metrics,'scale_candidates':counts.get('candidates'),'budget_audit':json.loads((out/'scale_budget_audit.json').read_text()) if (out/'scale_budget_audit.json').exists() else None,'clips':len(clips),'resource_seconds':sum(r['total_s'] for r in resources),'count_seconds':counts.get('seconds'),'error':error})


def publish(out,status):
    if git('branch','--show-current')!=BRANCH or git('status','--porcelain'):raise ValueError('PUBLISH_REQUIRES_CLEAN_SOURCE')
    paths=[]
    for name,source in [('results_v03.md','report_zh.md'),('results_v03.json','result_summary.json')]:
        p=REPO/'docs/pvst'/name;p.write_bytes((out/source).read_bytes());paths.append(str(p.relative_to(REPO)))
    p=REPO/'docs/pvst/current_status.md';s=p.read_text();s='\n'.join(f'状态{status}（v0.3；仅两条件各一轮上限；见[结果](results_v03.md)）' if line.startswith('状态') else line for line in s.split('\n'));p.write_text(s);paths.append(str(p.relative_to(REPO)))
    subprocess.run(['git','add','--',*paths],check=True);subprocess.run(['git','commit','-m','Record PVST v03 bounded run outcome'],check=True);head=git('rev-parse','HEAD');save(out/'publication.json',{'status':'COMMITTED','commit':head})
    subprocess.run(['git','push','origin',BRANCH],check=True);remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if head!=remote:raise ValueError('REMOTE_HEAD_MISMATCH')
    save(out/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('docs/pvst/config_v03.json'));parser.add_argument('--worker',type=Path);a=parser.parse_args()
    if a.worker:
        spec=json.loads(a.worker.read_text());c=spec['config'];out=Path(c['output']);result=worker(c,out,spec['window'],spec['condition'],spec['offload'],c['attention_chunk'],profile=spec.get('profile',False),source=Path(c['asset_root']))
        save(out/a.worker.name.replace('spec_','resource_'),{'spec':spec,'result':result});return
    c=json.loads(a.config.read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();resources=[];summaries=[];status='FAILED';error=None
        def progress(stage,**kw):
            save(out/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'unix_time':time.time(),**kw});print(stage,kw,flush=True)
        try:
            torch.set_num_threads(4)
            if git('branch','--show-current')!=BRANCH or git('status','--porcelain'):raise ValueError('RUN_REQUIRES_COMMITTED_SOURCE')
            c,rows,scale,counts,clips=freeze_inputs(c,out,progress)
            if counts['selected_h'] is None:status='RESOURCE_BLOCKED';raise RuntimeError('NO_AUTHORIZED_SCALE_WITHIN_BUDGET')
            identity=protocol(c,out);p=out/'protocol.json'
            if p.exists() and json.loads(p.read_text())!=identity:raise ValueError('PROTOCOL_CHANGED')
            save(p,identity);passed,resources=resource_checks(c,out,counts,identity,progress)
            if not passed:status='RESOURCE_BLOCKED';raise RuntimeError('RESOURCE_REPRESENTATIVE_FAILED')
            c={**c,'offload_saved_tensors':any(r['offload'] and r['status']=='PASS' for r in resources)};save(out/'execution_config.json',c)
            for condition in c['conditions']:summaries.append(train_one(c,out,rows,scale,clips,condition,identity,progress))
            if len({s['initial_parameters_sha256'] for s in summaries})!=1:raise ValueError('INITIALIZATION_NOT_MATCHED')
            status='COMPLETE'
        except BaseException:
            error=traceback.format_exc();print(error,flush=True)
        report(out,c,status,resources,summaries,error);save(out/'progress.json',{'status':status,'stage':'report_complete','formal_models_completed':len(summaries),'pid':os.getpid(),'elapsed_s':time.monotonic()-started});(out/'exit_code.txt').write_text('0\n' if status=='COMPLETE' else '1\n')
        try:publish(out,status)
        except BaseException:save(out/'publication.json',{'status':'FAILED','calculation_status':status,'error':traceback.format_exc()})
        if status!='COMPLETE':raise SystemExit(1)

if __name__=='__main__':main()
