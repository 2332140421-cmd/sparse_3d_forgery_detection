"""Independent prepare/resource/train/evaluate runner for ADR-0043.

The single script persists stages, real exits and checkpoints. It does not
resume failed training, retry OOM or alter configuration after dev results.
"""
import argparse
import csv
from collections import deque
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import time
import traceback
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from research_tools.rgbd_slice.experiment import state_hash
from research_tools.v8.spatial_localization.metrics import spatial_frame,classification
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import numeric_members,mask_array,decode
from .data import OUT,REPO,CONFIG,Cache,read,save,progress,prepare
from .model import XYZPilot,history_indices


def csvsave(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def terms(out,target):
    gt=target['gt_work'].float();z=out['pixel'].float()
    pixel=F.binary_cross_entropy_with_logits(z,gt)
    p=z.sigmoid();dice=1-(2*(p*gt).sum()+1)/(p.sum()+gt.sum()+1)
    frame=F.binary_cross_entropy_with_logits(out['frame'].float(),z.new_tensor(float(target['frame_positive'])))
    return torch.stack((pixel,dice,frame))


def seed():random.seed(20260909);np.random.seed(20260909);torch.manual_seed(20260909);torch.cuda.manual_seed_all(20260909)
def eligible(row,i):return not(row['pair_id']=='an:40y1x9' and row['role']=='fake' and 6<=i<10)


def update(model,opt,cache,row,i,temporal,diagnose=False):
    start=time.monotonic();model.train();opt.zero_grad(set_to_none=True);fs=cache.window(row,i,temporal);assert fs[-1]['supervised']
    torch.cuda.synchronize();load_done=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(fs,temporal)
    t=terms(out,fs[-1]);loss=t.sum();assert torch.isfinite(loss)
    torch.cuda.synchronize();fw=time.monotonic();loss.backward();torch.cuda.synchronize();bw=time.monotonic()
    grads=None
    if diagnose:
        grads={}
        for prefix in ['geometry','point','score','value','space','context','time','pixel','frame']:
            gs=[p.grad.float() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            grads[prefix]={'present':bool(gs),'finite':all(bool(torch.isfinite(g).all()) for g in gs),'norm':float(torch.sqrt(sum((g.square().sum() for g in gs),torch.tensor(0.,device='cuda'))))}
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    before=model.pixel[-1].weight.detach().clone() if diagnose else None
    opt.step();torch.cuda.synchronize();end=time.monotonic()
    if diagnose:assert not torch.equal(before,model.pixel[-1].weight) and len(opt.state)>0
    return {'loss':float(loss),'pixel_BCE':float(t[0]),'soft_Dice':float(t[1]),'frame_BCE':float(t[2]),'gradient_norm':float(norm),'load_seconds':load_done-start,'forward_seconds':fw-load_done,'backward_seconds':bw-fw,'optimizer_seconds':end-bw,'seconds':end-start,'target_blocks':out['blocks'],'gradients':grads,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}


def orders(rows,c):
    train=[r for r in rows if r['split']=='train'];pairs=c['train'];result=[];rng=random.Random(c['seed'])
    for cycle in range((c['updates']+13)//14):
        ids=pairs.copy();random.Random(c['seed']+cycle).shuffle(ids)
        for pair in ids:
            for role in ['real','fake']:
                row=next(r for r in train if r['pair_id']==pair and r['role']==role)
                target=rng.choice([i for i in range(row['actual_frames']) if eligible(row,i)])
                result.append({'video_id':row['video_id'],'pair_id':pair,'role':role,'target':target,'history':history_indices(row['pts'],target,c['history_slots'],c['history_step_seconds'])})
    return result[:c['updates']]


@torch.no_grad()
def evaluate(model,cache,rows,c,condition,reverse=False):
    start=time.monotonic();model.eval();video_records=[];frame_records=[];predictions={};all_y=[];all_z=[]
    for row in rows:
        ring={};metrics=[];labels=[];scores=[];fixed={0,row['actual_frames']//2,row['actual_frames']-1}
        zfile=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None;members=numeric_members(zfile.namelist()) if zfile else None
        for i in range(row['actual_frames']):
            f=cache.get(row,i)
            with torch.autocast('cuda',dtype=torch.bfloat16):e=model.encode(f)
            ring[i]={'C':e['C'].cpu(),'position':e['position'].cpu()}
            slots=history_indices(row['pts'],i,c['history_slots'],c['history_step_seconds']) if condition=='XYZ_TEMPORAL' else [i]
            ids=[j for j in slots if j is not None];actual=ids[:-1][::-1]+[i] if reverse else ids
            contexts=[{k:v.cuda() for k,v in ring[j].items()} for j in actual]
            delta=[row['pts'][j]-row['pts'][i] for j in ids]
            assert sorted(actual[:-1])==sorted(ids[:-1]) and actual[-1]==ids[-1]==i
            with torch.autocast('cuda',dtype=torch.bfloat16):context,frame=model.read(e,contexts,delta);pixel=model.decode(f,e,context)
            if eligible(row,i):
                gt=mask_array(zfile.read(members[i][1]))[1] if zfile else np.zeros(row['native_hw'],bool)
                assert int(gt.sum())==f['native_GT_pixels'] and list(gt.shape)==row['native_hw']
                native=F.interpolate(pixel.float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy();assert np.isfinite(native).all()
                overlap=spatial_frame(native>=0,gt);ap=float(average_precision_score(gt.ravel(),native.ravel())) if gt.any() else None
                score=float(frame.float());label=int(gt.any());assert np.isfinite(score)
                metrics.append({'AP':ap,'Dice':overlap['dice'],'IoU':overlap['iou'],'FPR':overlap['empty_fpr'],'recall':int(((native>=0)&gt).sum())/int(gt.sum()) if gt.any() else None})
                labels.append(label);scores.append(score);all_y.append(label);all_z.append(score)
                frame_records.append({'condition':condition,'reverse':reverse,'video_id':row['video_id'],'pair_id':row['pair_id'],'role':row['role'],'frame_index':i,'pts':row['pts'][i],'history_ids':ids,'content_ids':actual,'native_pixels':gt.size,'GT_pixels':int(gt.sum()),'frame_label':label,'frame_logit':score,'frame_BCE':float(F.softplus(torch.tensor(score,dtype=torch.float32))-label*score),'pixel_AP':ap,'Dice':overlap['dice'],'IoU':overlap['iou'],'real_FPR':overlap['empty_fpr'] if row['role']=='real' else None})
                if row['role']=='fake' and i in fixed:predictions[(row['video_id'],i)]=native.astype(np.float16)
            cutoff=row['pts'][i]-(c['history_slots']-1)*c['history_step_seconds']-.2
            for j in list(ring):
                if row['pts'][j]<cutoff:del ring[j]
            del f,e,pixel,context,contexts
        if zfile:zfile.close()
        avg=lambda name:float(np.mean([r[name] for r in metrics if r[name] is not None])) if any(r[name] is not None for r in metrics) else None
        video_records.append({'condition':condition,'reverse':reverse,'pair_id':row['pair_id'],'video_id':row['video_id'],'role':row['role'],'frames':len(metrics),'pixel_AP':avg('AP'),'Dice':avg('Dice'),'IoU':avg('IoU'),'GT_recall':avg('recall'),'real_FPR':avg('FPR') if row['role']=='real' else None,'video_logit':float(np.mean(scores)),'frame_metrics':classification(labels,scores)})
        print('evaluated',condition,reverse,row['video_id'],len(metrics),flush=True)
    macro={k:float(np.mean([v[k] for v in video_records if v['role']=='fake' and v[k] is not None])) for k in ['pixel_AP','Dice','IoU','GT_recall']}
    macro['real_FPR']=float(np.mean([v['real_FPR'] for v in video_records if v['role']=='real']))
    macro['frame_metrics']=classification(all_y,all_z);macro['video_metrics']=classification([int(v['role']=='fake') for v in video_records],[v['video_logit'] for v in video_records])
    return {'macro':macro,'videos':video_records,'frames':frame_records,'seconds':time.monotonic()-start},predictions


def previews(rows,all_preds):
    folder=OUT/'preview';folder.mkdir(exist_ok=True);html=['<!doctype html><meta charset="utf-8"><h1>预先固定首/中/末时刻：RGB / GT / A / B</h1><p>同一色阶0–1，概率图只作展示；原生全画面评分。猫为跨编辑类型，7ogeg4为运动程度未确认参照。</p>']
    for row in rows:
        if row['role']!='fake':continue
        fixed={0,row['actual_frames']//2,row['actual_frames']-1};_,rgb=decode(Path(row['video_path']),fixed)
        with zipfile.ZipFile(row['mask_zip']) as z:
            members=numeric_members(z.namelist())
            for i in sorted(fixed):
                gt=mask_array(z.read(members[i][1]))[1];panels=[np.array(rgb[i]),np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)]
                for name in ['XYZ_CURRENT','XYZ_TEMPORAL']:
                    logit=all_preds[name][(row['video_id'],i)].astype(np.float32);prob=1/(1+np.exp(-np.clip(logit,-40,40)))
                    panels.append(np.repeat((prob*255).astype(np.uint8)[...,None],3,-1))
                images=[]
                for a in panels:
                    factor=min(1.,480/a.shape[1]);images.append(np.array(Image.fromarray(a).resize((round(a.shape[1]*factor),round(a.shape[0]*factor)))))
                file=f'{row["video_id"]}_{i:05d}.png';Image.fromarray(np.concatenate(images,axis=1)).save(folder/file)
                html.append(f'<h3>{row["pair_id"]} 原生帧{i} PTS {row["pts"][i]:.6f}</h3><img style="max-width:100%" src="{file}">')
    (folder/'index.html').write_text('\n'.join(html))


def report(s):
    r=s['results'];a=r['XYZ_CURRENT']['evaluation'];b=r['XYZ_TEMPORAL']['evaluation'];rows=[]
    text=['# PVST等尺寸XYZ块：当前空间 vs 加入历史','状态：'+s['status']+'；末次576步主比较，未调参/选best。', '|指标|XYZ_CURRENT|XYZ_TEMPORAL|B−A|','|---|---:|---:|---:|']
    for k in ['pixel_AP','Dice','IoU','GT_recall','real_FPR']:text.append(f"|{k}|{a['macro'][k]:.8f}|{b['macro'][k]:.8f}|{b['macro'][k]-a['macro'][k]:+.8f}|")
    text+=['\n|开发来源|A AP|B AP|B−A|A Dice|B Dice|配对real FPR A/B|','|---|---:|---:|---:|---:|---:|---|']
    for x in a['videos']:
        if x['role']!='fake':continue
        y=next(v for v in b['videos'] if v['video_id']==x['video_id']);ar=next(v for v in a['videos'] if v['pair_id']==x['pair_id'] and v['role']=='real');br=next(v for v in b['videos'] if v['pair_id']==x['pair_id'] and v['role']=='real')
        rows.append({'pair':x['pair_id'],'A_AP':x['pixel_AP'],'B_AP':y['pixel_AP'],'delta':y['pixel_AP']-x['pixel_AP']})
        text.append(f"|{x['pair_id']}|{x['pixel_AP']:.8f}|{y['pixel_AP']:.8f}|{y['pixel_AP']-x['pixel_AP']:+.8f}|{x['Dice']:.8f}|{y['Dice']:.8f}|{ar['real_FPR']:.8f}/{br['real_FPR']:.8f}|")
    csvsave(OUT/'comparison.csv',rows)
    text+=['改善'+str(sum(v['delta']>0 for v in rows))+'/4，下降'+str(sum(v['delta']<0 for v in rows))+'/4；四来源始终全部保留。','7e2e57猫添加是跨编辑类型；95q8g9骑马人物移除和4xd3r9多人过街移除为已明确运动内容；7ogeg4为站立对象移除参照，运动程度未获人工确认。逐来源不能以宏均值替代。','## 结构/数据/评价对应','通用原版DFormer-S冻结eval，仅官方通用预训练权重，不加载旧伪造训练last/common-C；新多尺度投影、几何条件调制、块内学习式softmax、空间及跨候选关系、像素/帧头训练。块为统一XYZ floor、不是深度整片或图像格；保留全部工作网格有效点/占据块，模型无GT选块、无共有键或追踪前提。','单一训练尺度与边长：'+json.dumps({k:s['config'][k] for k in ['scale','h','D','history_slots','history_step_seconds','work_long_edge']})+'。原点0，无米制/世界对齐保证；单目帧间尺度漂移仍是限制。工作长边384，等比resize、右/下pad32；输出bilinear恢复原生，全画面评分，不按几何valid裁评价像素。几何无效处仍有视觉局部旁路。','固定7/4来源，无测试访问。40y1x9 fake6–9不作为目标/标签，可进入无标签历史；real原零GT合法。开发所有原生帧各计一次，AP仅fake正GT帧：逐帧全画面AP→视频均值→4视频宏平均，real不混入AP，固定logit≥0的FPR单列。','训练采用同初始化/576目标顺序、AdamW1e-4 wd.01 clip1；pixelBCE+softDice+frameBCE各1。工作GT nearest；帧标签按原生GT非空，不因下采样消失改变帧标签。每步目标一帧，过去7个PTS槽是约0.7秒；不足历史显式None不补未来，真实相对PTS进入注意力key。A只当前，B允许当前和过去全部跨位置块。','## 一次资源验收与成本',json.dumps(s['resource'],ensure_ascii=False),'准备成本：'+json.dumps(s['preparation'],ensure_ascii=False),'训练/评价成本：'+json.dumps({k:{n:v[n] for n in ['updates','train_seconds','peak_allocated','peak_reserved']} for k,v in r.items()},ensure_ascii=False),'## 固定历史倒序（不训练）','B倒序评价：'+json.dumps(s['reverse_summary'],ensure_ascii=False)+'。仅重新绑定历史内容到原时间槽，当前/历史集合及次数不变。顺序变化不预设提高异常分数。','帧/视频指标仅辅助，开发来源少且无独立内容编辑起止真值，不称已验证时间边界定位。A/B只回答本模型历史关系增量；不能单独证明三维优于二维，负结果也不否定全部时空关系假设。旧结果不与本协议混表。','固定首/中/末RGB/GT/A/B：preview/index.html。','执行身份：'+json.dumps(s['identity'],ensure_ascii=False),'本轮停止，不扩数据、seed或架构。完整产物：'+str(OUT)]
    result='\n\n'.join(text)+'\n';(OUT/'report.md').write_text(result);(REPO/'docs/pvst/xyz_temporal_pilot_v1.md').write_text(result)


def run():
    start=time.monotonic();torch.set_num_threads(4);c=read(OUT/'config.json');assert c['h'] and c['scale']
    if (OUT/'run_identity.json').exists():raise RuntimeError('Existing run preserved, no duplicate/retry')
    assert read(CONFIG)==c,'EXECUTION_CONFIG_NOT_FROZEN_IN_GIT'
    identity={'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'config_sha256':digest(OUT/'config.json'),'manifest_sha256':digest(OUT/'manifest.json'),'prepared_identity':read(OUT/'identity.json'),'model_source_sha256':digest(REPO/'research_tools/xyz_pilot/model.py')}
    save(OUT/'run_identity.json',identity);rows=read(OUT/'manifest.json')['rows'];byid={r['video_id']:r for r in rows};cache=Cache(c)
    sequence=orders(rows,c);save(OUT/'orders.json',sequence)
    seed();model=XYZPilot(c['h']);initial={k:v.clone() for k,v in model.state_dict().items()};ih=state_hash(initial);torch.save(initial,OUT/'initial.pt');save(OUT/'initialization.json',{'sha256':ih,'parameters':sum(p.numel() for p in model.parameters()),'forensic_initialization':False});del model
    counts=read(OUT/'scale_selection.json')['node_counts'];candidates=[]
    for row in rows:
        if row['split']!='train':continue
        for i in range(row['actual_frames']):
            if eligible(row,i):
                ids=history_indices(row['pts'],i,c['history_slots'],c['history_step_seconds']);load=sum(counts[row['video_id']][str(c['h'])][j] for j in ids if j is not None)
                candidates.append((load,row['video_id'],i))
    load,vid,i=max(candidates);progress('resource_update',video=vid,target=i,window_blocks=load)
    model=XYZPilot(c['h']).cuda();model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);torch.cuda.reset_peak_memory_stats()
    free,total=torch.cuda.mem_get_info();resource_result=update(model,opt,cache,byid[vid],i,True,True);resource_result.update(status='PASS',target_video=vid,target_frame=i,window_blocks=load,free_before=free,total_gpu=total,optimizer_states=len(opt.state),initial_sha256=ih)
    assert all(g['present'] and g['finite'] for g in resource_result['gradients'].values())
    save(OUT/'resource.json',resource_result)
    print('ESTIMATE_SECONDS_A_B_TRAIN_FROM_B_STEP',resource_result['seconds']*c['updates']*2,'plus evaluation; resource-step estimate only',flush=True)
    del model,opt;torch.cuda.empty_cache();results={};all_predictions={};history=[]
    for condition,temporal in [('XYZ_CURRENT',False),('XYZ_TEMPORAL',True)]:
        seed();model=XYZPilot(c['h']).cuda();model.load_state_dict(initial);assert state_hash(model.state_dict())==ih
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        with (OUT/f'{condition}_updates.jsonl').open('w') as log:
            for step,item in enumerate(sequence,1):
                if time.monotonic()-begin>c['train_eval_hard_seconds']:raise TimeoutError(condition+'_TRAIN_EVAL_CAP')
                record=update(model,opt,cache,byid[item['video_id']],item['target'],temporal)
                record.update(condition=condition,step=step,video_id=item['video_id'],target=item['target']);log.write(json.dumps(record)+'\n');log.flush()
                if step%24==0:
                    history.append({k:record[k] for k in ['condition','step','loss','pixel_BCE','soft_Dice','frame_BCE','seconds','peak_allocated','peak_reserved']});csvsave(OUT/'history.csv',history);progress('train',condition=condition,step=step,elapsed=time.monotonic()-begin,remaining_estimate=(time.monotonic()-begin)/step*(c['updates']-step))
                if step%96==0:torch.save({'condition':condition,'updates':step,'state_dict':model.state_dict(),'initial_sha256':ih,'identity':identity},OUT/f'{condition}_last.pt')
        train_seconds=time.monotonic()-begin
        evaluated,pred=evaluate(model,cache,[r for r in rows if r['split']=='dev'],c,condition)
        assert len(evaluated['frames'])==sum(r['actual_frames'] for r in rows if r['split']=='dev')
        csvsave(OUT/f'{condition}_dev_videos.csv',evaluated['videos']);csvsave(OUT/f'{condition}_dev_frames.csv',evaluated['frames']);save(OUT/f'{condition}_evaluation.json',evaluated)
        results[condition]={'updates':576,'train_seconds':train_seconds,'evaluation':evaluated,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()};all_predictions[condition]=pred
        if temporal:
            reversed_eval,_=evaluate(model,cache,[r for r in rows if r['split']=='dev'],c,condition,True);save(OUT/'B_history_reversed.json',reversed_eval)
            normal=np.array([f['frame_logit'] for f in evaluated['frames']]);rev=np.array([f['frame_logit'] for f in reversed_eval['frames']])
            reverse_summary={'macro':reversed_eval['macro'],'macro_pixel_AP_delta':reversed_eval['macro']['pixel_AP']-evaluated['macro']['pixel_AP'],'max_abs_frame_logit_change':float(np.max(abs(rev-normal))),'mean_abs_frame_logit_change':float(np.mean(abs(rev-normal))),'frames':len(rev),'seconds':reversed_eval['seconds']}
        del model,opt;torch.cuda.empty_cache()
    a=results['XYZ_CURRENT']['evaluation']['frames'];b=results['XYZ_TEMPORAL']['evaluation']['frames'];assert [(x['video_id'],x['frame_index'],x['pts'],x['native_pixels'],x['GT_pixels']) for x in a]==[(x['video_id'],x['frame_index'],x['pts'],x['native_pixels'],x['GT_pixels']) for x in b]
    previews([r for r in rows if r['split']=='dev'],all_predictions)
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'config':c,'resource':resource_result,'preparation':read(OUT/'preparation.json'),'results':results,'reverse_summary':reverse_summary,'wall_seconds':time.monotonic()-start,'same_evaluation_frames_and_pixels':True,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};save(OUT/'summary.json',summary);report(summary);save(OUT/'status.json',{'status':'COMPLETE','stage':'report_complete','exit_code':0});(OUT/'exit_code.txt').write_text('0\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);args=p.parse_args()
    try:
        if args.stage=='prepare':prepare()
        else:run()
    except Exception as error:
        if OUT.exists():
            (OUT/f'{args.stage}_error.log').write_text(traceback.format_exc());save(OUT/'status.json',{'status':'OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'FAILED','stage':args.stage,'exit_code':1,'error':str(error)});(OUT/f'{args.stage}_exit_code.txt').write_text('1\n')
        raise
