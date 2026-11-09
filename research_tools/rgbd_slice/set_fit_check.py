"""Bounded two-pair diagnostic; reuse native loaders, optimizer/loss and previews.

CLI parent owns only its worker process group and finalizes partial evidence.
No prepare/frontend/formal training entry is called.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .experiment import REPO,git,optimizer,forward,load_clip,state_hash,preview,seed_all
from .model import RGBDSetSlices,backbone
from .data import save
from research_tools.depth_slice.model import terms,slab_ids
from research_tools.v8.spatial_localization.metrics import spatial_frame
from scripts.prepare_v8_assets import digest

SOURCE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1')
OUT=SOURCE.parent/'pvst_set_to_set_fit_check_v1'
PAIRS=['an:7ogg04','an:728qx7']
DOC='docs/pvst/set_to_set_fit_check_v1.md'


def inputs():
    c=json.loads((SOURCE/'config.json').read_text())
    c.update(geometry_root=str(SOURCE/'geometry'),initialization_root=str(SOURCE))
    if [c[k] for k in ['seed','scale','h','d0','D']]!=[20260909,1.9888148307800293,.125,0,512]:raise ValueError('FROZEN_COORDINATE_CONFIG')
    manifest=json.loads((SOURCE/'manifest.json').read_text());rows=[]
    for pair in PAIRS:
        for role in ['fake','real']:
            match=[r for r in manifest['rows'] if r['pair_id']==pair and r['role']==role and r['split']=='train']
            if len(match)!=1 or match[0]['actual_frames']<24:raise ValueError('FIXED_TRAIN_IDENTITY:'+pair+role)
            rows+=match
    return c,rows


def build(c,out,*,rgb_only=False):
    seed_all(c['seed']);model=RGBDSetSlices(backbone(c,out/'official_load.json'),rgb_only=rgb_only)
    common=torch.load(SOURCE/'common_initial.pt',map_location='cpu',weights_only=True)
    model.local.load_state_dict(common,strict=True)
    h=state_hash(model.local.state_dict())
    if h!=json.loads((SOURCE/'initialization.json').read_text())['common_sha256']:raise ValueError('COMMON_INITIAL_IDENTITY')
    removed=[k for k in common if k.startswith('frame.')]
    # The old classifier is intentionally absent: new detection only reads GRU.
    del model.local.frame
    save(out/'initialization.json',{'common_sha256':h,'loaded_local_keys':list(common),'strict_local_load':True,
         'model_sha256':state_hash(model.state_dict()),
         'removed_old_classifier_keys':removed,'new_modules':['pool','sets','pixel_context'],
         'trained_checkpoint_loaded':False,'parameters':sum(p.numel() for p in model.parameters()),
         'module_parameters':{n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()}})
    return model.cuda()


def gradient_info(model):
    prefixes=['local.encoder','local.refine','pool','sets.participation.0','sets.participation.1','sets.read_members',
              'sets.interaction','sets.unit_readout','sets.current_readout','sets.global_readout','sets.gru','sets.frame_head','pixel_context']
    result={}
    for prefix in prefixes:
        gradients=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result[prefix]={'present':bool(gradients),'finite':all(bool(torch.isfinite(g).all()) for g in gradients),
                        'norm':float(torch.sqrt(sum((g.float().square().sum() for g in gradients),torch.zeros((),device='cuda'))))}
    return result


def update_once(model,opt,fs):
    model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
    output=forward(model,fs);parts=torch.stack([terms(p,f['gt']) for p,f in zip(output,fs)]).mean(0);loss=parts.sum()
    if not bool(torch.isfinite(loss)):raise ValueError('NONFINITE_LOSS')
    torch.cuda.synchronize();fwd=time.monotonic();loss.backward();torch.cuda.synchronize();bwd=time.monotonic()
    gradients=gradient_info(model);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    before=model.local.final[-1].weight.detach().clone();opt.step();torch.cuda.synchronize();end=time.monotonic()
    changed=not torch.equal(before,model.local.final[-1].weight)
    if not changed:raise ValueError('NO_ADAMW_PARAMETER_CHANGE')
    return {'terms':parts.detach().float().cpu().tolist(),'loss':float(loss.detach()),'gradient_norm':float(norm),
            'gradients':gradients,'forward_s':fwd-start,'backward_s':bwd-fwd,'adamw_and_checks_s':end-bwd,
            'seconds':end-start,'parameter_changed':changed,'optimizer_states':len(opt.state),
            'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}


@torch.no_grad()
def evaluate_fixed(model,rows,clips,out,number,samples=False):
    model.eval();records=[];start=time.monotonic()
    for row,fs in zip(rows,clips):
        pred=forward(model,fs);spatial=[];losses=[];recall=[];areas=[];bce=[]
        pts=np.array([f['pts'] for f in fs]);chosen={int(np.argmin(abs(pts-(pts[0]+x*(pts[-1]-pts[0]))))) for x in [.25,.5,.75]}
        for i,(p,f) in enumerate(zip(pred,fs)):
            z=p['pixel'].float().cpu().numpy();g=f['gt'].numpy();spatial.append(spatial_frame(z>=0,g));areas.append(float((z>=0).mean()))
            losses.append(terms(p,f['gt']).cpu().tolist());bce.append(float(F.binary_cross_entropy_with_logits(p['frame'].float(),p['frame'].new_tensor(float(g.any())))))
            if g.any():recall.append(float(((z>=0)&g).sum()/g.sum()))
            if samples and i in chosen:preview(out/f'samples_{number}'/f'{row["video_id"]}_{i}.jpg',f['rgb'].numpy(),g,z)
        records.append({'pair_id':row['pair_id'],'role':row['role'],'video_id':row['video_id'],'frames':len(fs),
                        'terms':np.mean(losses,0).tolist(),'frame_BCE':float(np.mean(bce)),
                        'dice':float(np.mean([x['dice'] for x,g in zip(spatial,[f['gt'] for f in fs]) if g.any()])) if recall else None,
                        'iou':float(np.mean([x['iou'] for x,g in zip(spatial,[f['gt'] for f in fs]) if g.any()])) if recall else None,
                        'GT_recall':float(np.mean(recall)) if recall else None,
                        'real_FPR':float(np.mean([x['empty_fpr'] for x in spatial])) if row['role']=='real' else None,
                        'pred_area':float(np.mean(areas)),'frame_logits':[float(p['frame']) for p in pred]})
        del pred
    result={'update':number,'records':records,'seconds':time.monotonic()-start}
    save(out/f'evaluation_{number:03d}.json',result)
    if samples:
        folder=out/f'samples_{number}';folder.mkdir(exist_ok=True)
        (folder/'index.html').write_text('<meta charset="utf-8"><p>固定PTS 25/50/75%；RGB | GT | 概率0–1 | logit≥0</p>'+''.join(f'<p>{p.name}</p><img width="100%" src="{p.name}">' for p in sorted(folder.glob('*.jpg'))))
    return result


@torch.no_grad()
def intervention(model,fs,out):
    model.eval();model.sets.capture=True;model.sets.observations=[]
    a=forward(model,fs);save(out/'participation.json',model.sets.observations)
    model.sets.capture=False;model.sets.zero_messages=True
    b=forward(model,fs);model.sets.zero_messages=False
    result={'scope':'first fixed training fake clip, same final weights/eval; zero returned relation message and relation g only',
            'frame_logit_mean_abs_difference':float(torch.stack([(x['frame']-y['frame']).abs() for x,y in zip(a,b)]).mean()),
            'pixel_logit_mean_abs_difference':float(torch.stack([(x['pixel'].float()-y['pixel'].float()).abs().mean() for x,y in zip(a,b)]).mean()),
            'pixel_probability_mean_abs_difference':float(torch.stack([(x['pixel'].float().sigmoid()-y['pixel'].float().sigmoid()).abs().mean() for x,y in zip(a,b)]).mean())}
    save(out/'intervention.json',result)


def worker(out,*,condition=None):
    start=time.monotonic();history=[];result={'status':'FAILED','updates':0,'exit_code':1}
    def progress(stage,**extra):
        save(out/'progress.json',{'status':'RUNNING','pid':os.getpid(),'stage':stage,'wall_s':time.monotonic()-start,**extra});print(stage,extra,flush=True)
    try:
        torch.set_num_threads(4);c,rows=inputs();save(out/'config.json',{**c,'max_updates':80,'worker_seconds':1800,'order':[(r['pair_id'],r['role']) for r in rows],**({'condition':condition} if condition is not None else {})})
        save(out/'launch.json',{'head':git('rev-parse','HEAD'),'gpu':torch.cuda.get_device_name(),'gpu_free_total':torch.cuda.mem_get_info(),'started_unix':time.time()})
        progress('model');t=time.monotonic();model=build(c,out,rgb_only=condition=='RGB_DEPTH_PARTITION');opt=optimizer(model,c);result['model_s']=time.monotonic()-t
        bn=state_hash(dict(model.local.encoder.named_buffers()));t=time.monotonic();clips=[];identities=[]
        for r in rows:
            progress('load',video=r['video_id'])
            if digest(Path(r['video_path']))!=r['video_sha256']:raise ValueError('VIDEO_SHA')
            if r['role']=='fake' and digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SHA')
            clip={'video_id':r['video_id'],'start':0,'end':24};fs=load_clip(r,clip,c)
            if [f['pts'] for f in fs]!=[p['pts_s'] for p in r['pts'][:24]] or [f['frame_index'] for f in fs]!=list(range(24)):raise ValueError('FRAME_PTS_IDENTITY')
            if any(b['pts']<=a['pts'] for a,b in zip(fs,fs[1:])):raise ValueError('NONMONOTONIC_PTS')
            meta=json.loads((Path(c['geometry_root'])/r['video_id']/'complete.json').read_text())
            if max(meta['output_hw'])<=256:raise ValueError('LOW_RES_GEOMETRY_FORBIDDEN')
            counts=[len(slab_ids(f['xyz'],f['valid'])[0]) for f in fs]
            identities.append({k:r[k] for k in ['pair_id','source_id','role','split','video_id','video_sha256','source_hw']})
            identities[-1].update(range=[0,24],pts=r['pts'][:24],geometry=meta,occupied_slices=counts,total_tokens=sum(counts),
                input_shapes={k:list(v.shape) for k,v in fs[0].items() if isinstance(v,torch.Tensor)})
            clips.append(fs)
        save(out/'selected_identity.json',identities);result['load_s']=time.monotonic()-t
        torch.cuda.reset_peak_memory_stats();progress('evaluation',update=0);evaluate_fixed(model,rows,clips,out,0,True)
        for i in range(1,81):
            if time.monotonic()-start>=1800:break
            j=(i-1)%4;progress('update',update=i,pair=rows[j]['pair_id'],role=rows[j]['role'])
            record=update_once(model,opt,clips[j]);record.update(update=i,pair_id=rows[j]['pair_id'],role=rows[j]['role'])
            history.append(record);result['updates']=i;save(out/'history.json',history)
            with (out/'history.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['update','pair_id','role','loss','frame_BCE','pixel_BCE','dice_loss','gradient_norm','forward_s','backward_s','adamw_and_checks_s','seconds'])
                for h in history:writer.writerow([h['update'],h['pair_id'],h['role'],h['loss'],*h['terms'],h['gradient_norm'],h['forward_s'],h['backward_s'],h['adamw_and_checks_s'],h['seconds']])
            if i%10==0:progress('evaluation',update=i);evaluate_fixed(model,rows,clips,out,i,i==80)
        last=len(history)
        if last!=80:progress('final_evaluation',update=last);evaluate_fixed(model,rows,clips,out,last,True)
        progress('relation_intervention');intervention(model,clips[0],out)
        if state_hash(dict(model.local.encoder.named_buffers()))!=bn:raise ValueError('BN_BUFFERS_CHANGED')
        if condition is not None and last==80:
            torch.save({'condition':condition,'updates':last,'state_dict':model.state_dict()},out/'final.pt')
        result.update(status='COMPLETE',exit_code=0,stop_reason='80_UPDATES' if last==80 else 'TIME_LIMIT',
                      BN_unchanged=True,optimizer_states=len(opt.state))
    except BaseException as e:
        result.update(status='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'FAILED',error=traceback.format_exc());(out/'error.txt').write_text(result['error']);print(result['error'],flush=True)
    finally:
        result.update(wall_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(out/'result.json',result);save(out/'progress.json',{**result,'stage':'worker_finished','pid':os.getpid()})
    raise SystemExit(result['exit_code'])


def report(out,exit_code,timed_out):
    result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'updates':len(json.loads((out/'history.json').read_text())) if (out/'history.json').exists() else 0}
    if timed_out:result.update(status='TIMEOUT',last_stage=json.loads((out/'progress.json').read_text()))
    result['actual_worker_exit_code']=exit_code;save(out/'result.json',result)
    evaluations=[json.loads(p.read_text()) for p in sorted(out.glob('evaluation_*.json'))];history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    lines=['# 自适应集合对集合：两个训练pair限定拟合诊断',f'状态：{result.get("status","FAILED")}；真实worker退出码{exit_code}；已完成更新{result["updates"]}/80。',
           '只用训练an:7ogg04、an:728qx7各fake/real的原始[0,24)，四段共96帧；不是开发/测试性能。四段按pair1 fake→real→pair2 fake→real循环，每段一次更新，最多1800秒。',
           '方法与张量见ADR-0036及[设计](set_to_set_design_v1.md)。完整DFormer-S/原生解码/SlabPool复用；8个关系单元×A/B×4诱导状态512；两块8head联合交互；关系readout→512因果GRU→帧头，无旧global分类捷径；像素路径C+关系消息→零初始化512→32投影。',
           '各次确定性评价（logit≥0；空GT不填假标；正GT帧Dice/IoU/召回，real像素FPR）：','','|更新|pair|role|Dice|IoU|GT召回|real FPR|预测面积|frame BCE|','|---|---|---|---|---|---|---|---|---|']
    for ev in evaluations:
        for r in ev['records']:lines.append('|'+ '|'.join(str(v) for v in [ev['update'],r['pair_id'],r['role'],r['dice'],r['iou'],r['GT_recall'],r['real_FPR'],r['pred_area'],r['frame_BCE']])+'|')
    lines += ['','资源与停止记录：\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```']
    if history:
        lines+=['首／末更新三项loss（交替样本不同，不能直接当同一片段下降）：'+json.dumps([history[0]['terms'],history[-1]['terms']]),
                '最后更新剪裁前分支梯度：'+json.dumps(history[-1]['gradients'],ensure_ascii=False),
                '代表性首／末前向、反向、AdamW及检查耗时：'+json.dumps([{k:h[k] for k in ['forward_s','backward_s','adamw_and_checks_s','seconds']} for h in [history[0],history[-1]]])]
    if (out/'intervention.json').exists():lines+=['固定关系消息干预：'+(out/'intervention.json').read_text(),'参与与状态记录：participation.json；独立sigmoid允许重叠，是否实际退化需结合该记录，不能由容量宣称已学到关系。']
    lines+=['固定样例：'+('samples_80/index.html' if (out/'samples_80').exists() else '最近完成的samples_*/index.html')+'；RGB | GT | 概率0–1 | 二值预测。',
            '只判断本次执行和两个训练pair拟合进展；关系被消费不证明其泛化增益，Dice改善也不能单独证明关系有效。旧A/B/NORM的样本/预算不同，不作胜负比较。',
            '完整产物：`'+str(out)+'`。本轮结束，不追加正式训练、其他seed、消融或数据。']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text);(REPO/DOC).write_text(text)
    if history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,3))
        for j,k in enumerate(['frame BCE','pixel BCE','Dice loss']):ax[0].plot([h['update'] for h in history],[h['terms'][j] for h in history],label=k)
        for pair in PAIRS:ax[1].plot([e['update'] for e in evaluations],[next(r['dice'] for r in e['records'] if r['pair_id']==pair and r['role']=='fake') for e in evaluations],label=pair)
        for a in ax:a.legend()
        fig.tight_layout();fig.savefig(out/'curves.png');plt.close(fig)
    page=REPO/'docs/pvst/current_status.md';old=page.read_text().split('<!-- set-to-set-fit -->')[0];page.write_text(old+'\n<!-- set-to-set-fit -->\n\n## 集合对集合限定拟合\n\n'+str(result.get('status','FAILED'))+'；更新'+str(result['updates'])+'/80，见[报告](set_to_set_fit_check_v1.md)。无正式训练或开发/测试评分。\n')
    try:
        git('add',DOC,'docs/pvst/current_status.md');git('commit','-m','Report bounded set-to-set training-pair fitting diagnostic')
        head=git('rev-parse','HEAD');subprocess.run(['git','push','origin','pvst-set-to-set-relations'],cwd=REPO,check=True,timeout=120)
        remote=git('ls-remote','origin','refs/heads/pvst-set-to-set-relations').split()[0]
        save(out/'publication.json',{'status':'PUSHED' if head==remote else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(out/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'head':git('rev-parse','HEAD')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:return worker(OUT)
    if git('branch','--show-current')!='pvst-set-to-set-relations':raise ValueError('WRONG_BRANCH')
    if git('status','--porcelain','--','research_tools/rgbd_slice','tests/test_rgbd_set_relations.py'):raise ValueError('UNCOMMITTED_EXECUTION_SOURCE')
    if OUT.exists() and any(OUT.iterdir()):raise ValueError('EXISTING_RUN_REQUIRES_REVIEW_NO_RETRY')
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'identity.json',{'head':git('rev-parse','HEAD'),'source_manifest_sha256':digest(SOURCE/'manifest.json'),
          'code':{str(p.relative_to(REPO)):digest(p) for p in (REPO/'research_tools/rgbd_slice').glob('*.py')},'limit_s':1800})
    with (OUT/'worker.log').open('w') as log:
        process_start=time.monotonic()
        child=subprocess.Popen([sys.executable,'-m','research_tools.rgbd_slice.set_fit_check','--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=REPO)
        timed_out=False
        try:code=child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    save(OUT/'exit.json',{'worker_exit_code':code,'timed_out':timed_out,'worker_process_wall_s':time.monotonic()-process_start});report(OUT,code,timed_out)


if __name__=='__main__':main()
