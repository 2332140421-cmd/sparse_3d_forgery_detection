"""Bounded ADR-0044 paired experiment using existing generic observation caches.

Caller: python -m research_tools.xyz_pilot.joint_experiment. Produces one resource
update, two fixed 14700-update arms, prescribed evaluations and report. No prepare
or frontend path exists. Old experiment artifacts are read-only.
"""
import collections
import csv
import hashlib
import json
import math
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
from .data import Cache, OUT as SOURCE, REPO, read
from .model import history_indices
from .joint_model import XYZJointPilot
from .experiment import orders, eligible, terms, seed, csvsave
from research_tools.rgbd_slice.experiment import state_hash
from research_tools.v8.spatial_localization.metrics import classification, spatial_frame
from scripts.review_anakin import numeric_members, mask_array, decode

OUT = SOURCE.parent / 'pvst_xyz_joint_relation_pilot_v1'
CONFIG = REPO / 'docs/pvst/xyz_joint_relation_config_v1.json'
CONDITIONS = [('XYZ_JOINT_CURRENT', False), ('XYZ_JOINT_TEMPORAL', True)]


def save(path, value):
    # Atomic small state/checkpoint publication keeps interrupted files explicit.
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(stage, **fields):
    record = dict(status='RUNNING', stage=stage, **fields)
    save(OUT/'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def observation_targets(rows):
    result = {}
    for row in rows:
        if row['split'] != 'train':
            continue
        allowed = [i for i in range(row['actual_frames']) if eligible(row, i)]
        result[row['video_id']] = [allowed[int(f*(len(allowed)-1))] for f in [.10,.35,.65,.90]]
    return result


def coverage(sequence, rows):
    result = {}
    for row in rows:
        if row['split'] != 'train':
            continue
        chosen = [x['target'] for x in sequence if x['video_id'] == row['video_id']]
        possible = sum(eligible(row, i) for i in range(row['actual_frames']))
        result[row['video_id']] = dict(pair_id=row['pair_id'],role=row['role'],updates=len(chosen),
            unique_targets=len(set(chosen)),eligible_targets=possible,
            coverage=len(set(chosen))/possible)
    return dict(per_video=result,updates=len(sequence),
        real=sum(x['role']=='real' for x in sequence),fake=sum(x['role']=='fake' for x in sequence),
        unique_targets=sum(x['unique_targets'] for x in result.values()),
        eligible_targets=sum(x['eligible_targets'] for x in result.values()))


def checkpoint(path, model, optimizer, step, condition, identity, initial_hash):
    item = dict(condition=condition, updates=step, state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(), identity=identity, initial_sha256=initial_hash,
                random_state=dict(python=random.getstate(),numpy=np.random.get_state(),
                                  torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    temp = path.with_name(path.name+'.tmp')
    torch.save(item,temp);temp.replace(path)


def update(model,opt,cache,row,target,temporal,diagnose=False):
    model.train();opt.zero_grad(set_to_none=True)
    begin=time.monotonic()
    frames=cache.window(row,target,True)
    assert frames[-1]['supervised'] and eligible(row,target)
    torch.cuda.synchronize();loaded=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(frames,temporal)
    parts=terms(out,frames[-1]);loss=parts.sum()
    assert torch.isfinite(parts).all(),'NONFINITE_LOSS'
    torch.cuda.synchronize();forward=time.monotonic()
    loss.backward();torch.cuda.synchronize();backward=time.monotonic()
    gradients={}
    if diagnose:
        for prefix in ['geometry','point','score','value','layers','position','time','pixel','evidence']:
            gs=[p.grad.float() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            gradients[prefix]=dict(present=bool(gs),finite=all(bool(torch.isfinite(g).all()) for g in gs),
                                  norm=math.sqrt(sum(float(g.square().sum()) for g in gs)))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    before=model.evidence.weight.detach().clone() if diagnose else None
    opt.step();torch.cuda.synchronize();end=time.monotonic()
    if diagnose:
        assert not torch.equal(before,model.evidence.weight) and opt.state
    return dict(loss=float(loss),pixel_BCE=float(parts[0]),soft_Dice=float(parts[1]),
                frame_BCE=float(parts[2]),gradient_norm=float(norm),target_blocks=out['blocks'],
                window_blocks=out['window_blocks'],load_seconds=loaded-begin,
                forward_seconds=forward-loaded,backward_seconds=backward-forward,
                optimizer_seconds=end-backward,seconds=end-begin,gradients=gradients,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())


@torch.no_grad()
def evaluate(model,cache,rows,c,temporal,targets=None,capture=False):
    """Reuse native overlap/AP contract; each target's own cutoff window updates R.

    C0 may be reused during fixed-weight eval; trainable C0 is never cached across
    optimizer updates. History R is never reused as if it were causal at its PTS.
    """
    begin=time.monotonic();model.eval();videos=[];records=[];predictions={}
    for row in rows:
        wanted = list(range(row['actual_frames'])) if targets is None else targets[row['video_id']]
        wanted = [i for i in wanted if eligible(row,i)]
        wanted_set=set(wanted)
        needed=set()
        for target in wanted:
            needed.update(i for i in history_indices(row['pts'],target,c['history_slots'],c['history_step_seconds']) if i is not None)
        ring={};per_video=[]
        zfile=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        members=numeric_members(zfile.namelist()) if zfile else None
        for i in sorted(needed):
            f=cache.get(row,i)
            with torch.autocast('cuda',dtype=torch.bfloat16):encoded=model.encode(f)
            ring[i]={k:encoded[k].cpu() for k in ['C','position']}
            if i in wanted_set:
                slots=history_indices(row['pts'],i,c['history_slots'],c['history_step_seconds'])
                ids=[j for j in slots if j is not None]
                contexts=[{k:v.cuda() for k,v in ring[j].items()} for j in ids]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    relation,frame=model.read_window(contexts,[row['pts'][j]-row['pts'][i] for j in ids],ids,temporal)
                    pixel=model.decode(f,encoded,relation)
                gt=mask_array(zfile.read(members[i][1]))[1] if zfile else np.zeros(row['native_hw'],bool)
                assert list(gt.shape)==row['native_hw'] and int(gt.sum())==f['native_GT_pixels']
                native=F.interpolate(pixel.float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                assert np.isfinite(native).all() and torch.isfinite(frame)
                ap=float(average_precision_score(gt.ravel(),native.ravel())) if gt.any() else None
                overlap=spatial_frame(native>=0,gt);label=int(gt.any());score=float(frame.float())
                item=dict(video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],frame_index=i,
                          pts=row['pts'][i],history_ids=ids,native_pixels=gt.size,GT_pixels=int(gt.sum()),
                          frame_label=label,frame_logit=score,
                          frame_BCE=float(F.softplus(torch.tensor(score,dtype=torch.float32))-label*score),
                          pixel_AP=ap,constant_AP=float(gt.mean()) if gt.any() else None,
                          Dice=overlap['dice'],IoU=overlap['iou'],
                          GT_recall=float(((native>=0)&gt).sum()/gt.sum()) if gt.any() else None,
                          real_FPR=overlap['empty_fpr'] if row['role']=='real' else None)
                records.append(item);per_video.append(item)
                fixed=set(wanted) if targets is not None else {0,row['actual_frames']//2,row['actual_frames']-1}
                if capture and i in fixed:predictions[(row['video_id'],i)]=native.astype(np.float16)
                del contexts,relation,pixel,native
            cutoff=row['pts'][i]-(c['history_slots']-1)*c['history_step_seconds']-.2
            for j in list(ring):
                if row['pts'][j]<cutoff:del ring[j]
            del f,encoded
        if zfile:zfile.close()
        assert [x['frame_index'] for x in per_video]==sorted(wanted_set)
        avg=lambda key:float(np.mean([x[key] for x in per_video if x[key] is not None])) if any(x[key] is not None for x in per_video) else None
        videos.append(dict(video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],frames=len(per_video),
                           **{k:avg(k) for k in ['pixel_AP','constant_AP','Dice','IoU','GT_recall','real_FPR','frame_BCE']},
                           video_logit=avg('frame_logit'),frame_metrics=classification([x['frame_label'] for x in per_video],[x['frame_logit'] for x in per_video])))
    keys=[(x['video_id'],x['frame_index']) for x in records]
    assert len(keys)==len(set(keys))
    macro={key:float(np.mean([v[key] for v in videos if v['role']=='fake' and v[key] is not None]))
           for key in ['pixel_AP','constant_AP','Dice','IoU','GT_recall']}
    macro['real_FPR']=float(np.mean([v['real_FPR'] for v in videos if v['role']=='real']))
    macro['frame_BCE']=float(np.mean([x['frame_BCE'] for x in records]))
    macro['frame_metrics']=classification([x['frame_label'] for x in records],[x['frame_logit'] for x in records])
    macro['video_metrics']=classification([int(v['role']=='fake') for v in videos],[v['video_logit'] for v in videos])
    identity=[(x['video_id'],x['frame_index'],x['pts'],x['native_pixels'],x['GT_pixels']) for x in records]
    return dict(macro=macro,videos=videos,frames=records,seconds=time.monotonic()-begin,
                scored_frames=len(records),frame_identity_sha256=hashlib.sha256(json.dumps(identity).encode()).hexdigest()),predictions


def preview(rows,predictions,condition,group,output_root=None):
    folder=(OUT if output_root is None else output_root)/'preview'/group;folder.mkdir(parents=True,exist_ok=True)
    for row in rows:
        ids=sorted(i for vid,i in predictions if vid==row['video_id'])
        if not ids:continue
        _,rgb=decode(Path(row['video_path']),set(ids))
        z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        members=numeric_members(z.namelist()) if z else None
        for i in ids:
            gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool)
            arrays=[np.asarray(rgb[i]),np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)]
            p=1/(1+np.exp(-np.clip(predictions[(row['video_id'],i)].astype(np.float32),-40,40)))
            arrays.append(np.repeat((p*255).astype(np.uint8)[...,None],3,-1))
            panels=[]
            for a in arrays:
                factor=min(1.,480/a.shape[1]);panels.append(np.asarray(Image.fromarray(a).resize((round(a.shape[1]*factor),round(a.shape[0]*factor)))))
            Image.fromarray(np.concatenate(panels,axis=1)).save(folder/f'{condition}_{row["video_id"]}_{i:05d}.png')
        if z:z.close()


@torch.no_grad()
def order_check(model,cache,rows,c):
    model.eval();results=[]
    chosen=[(r,r['actual_frames']//2) for r in rows if r['role']=='fake']
    chosen.append((next(r for r in rows if r['role']=='fake'),0))
    for row,target in chosen:
        slots=history_indices(row['pts'],target,c['history_slots'],c['history_step_seconds'])
        ids=[j for j in slots if j is not None];memo={j:model.encode(cache.get(row,j)) for j in set(ids)}
        contexts=[memo[j] for j in ids];times=[row['pts'][j]-row['pts'][target] for j in ids]
        current=cache.get(row,target)
        def readout(order):
            content=[memo[j] for j in order]
            r,s=model.read_window(content,times,order,True)
            return model.decode(current,memo[target],r),s
        p1,s1=readout(ids);p2,s2=readout(ids)
        reversed_ids=list(reversed(ids[:-1]))+[target]
        p3,s3=readout(reversed_ids)
        results.append(dict(video_id=row['video_id'],target=target,pts=row['pts'][target],
            slots=ids,reversed_content=reversed_ids,unique_C0_encodes=len(memo),precision='FP32',
            repeat_pixel_max=float((p2-p1).abs().max()),repeat_frame_abs=float((s2-s1).abs()),
            reverse_pixel_max=float((p3-p1).abs().max()),reverse_frame_delta=float(s3-s1)))
    return results


def write_report(summary):
    a,b=[summary['results'][n]['evaluations']['14700']['dev'] for n,_ in CONDITIONS]
    lines=['# PVST 两层窗口联合关系：同帧与跨时刻配对实验',
           '状态COMPLETE；固定14700步末次主比较。没有选best、调参或增加来源。',
           '|开发指标|A同帧|B时空|B−A|','|---|---:|---:|---:|']
    for key in ['pixel_AP','constant_AP','Dice','IoU','GT_recall','real_FPR','frame_BCE']:
        x,y=a['macro'][key],b['macro'][key];lines.append(f'|{key}|{x:.8f}|{y:.8f}|{y-x:+.8f}|')
    lines+=['\n|来源|A AP|B AP|B−A|A/B Dice|配对real FPR A/B|','|---|---:|---:|---:|---|---|']
    comparison=[]
    for x in a['videos']:
        if x['role']!='fake':continue
        y=next(v for v in b['videos'] if v['video_id']==x['video_id'])
        ar=next(v for v in a['videos'] if v['pair_id']==x['pair_id'] and v['role']=='real')
        br=next(v for v in b['videos'] if v['pair_id']==x['pair_id'] and v['role']=='real')
        delta=y['pixel_AP']-x['pixel_AP'];comparison.append(dict(pair=x['pair_id'],A_AP=x['pixel_AP'],B_AP=y['pixel_AP'],delta=delta))
        lines.append(f"|{x['pair_id']}|{x['pixel_AP']:.8f}|{y['pixel_AP']:.8f}|{delta:+.8f}|{x['Dice']:.6f}/{y['Dice']:.6f}|{ar['real_FPR']:.6f}/{br['real_FPR']:.6f}|")
    csvsave(OUT/'comparison.csv',comparison)
    motion=[x for x in comparison if x['pair']!='an:7ogeg4']
    lines += [f"改善{sum(x['delta']>0 for x in comparison)}/4，下降{sum(x['delta']<0 for x in comparison)}/4；三个明确运动来源平均增量{np.mean([x['delta'] for x in motion]):+.8f}。",
              '7e2e57猫添加为跨编辑类型；95q8g9、4xd3r9为运动人物移除；7ogeg4仅为对象移除参照。',
              '## 训练观察子集（固定56帧，非全训练集重评分）',
              '|条件|更新|训练AP|训练Dice|训练real FPR|开发AP|开发real FPR|','|---|---:|---:|---:|---:|---:|---:|']
    for condition,_ in CONDITIONS:
        for step,ev in summary['results'][condition]['evaluations'].items():
            tr=ev['train']['macro'];dev=ev.get('dev',{}).get('macro',{})
            lines.append(f"|{condition}|{step}|{tr['pixel_AP']:.8f}|{tr['Dice']:.8f}|{tr['real_FPR']:.8f}|{dev.get('pixel_AP','未测量')}|{dev.get('real_FPR','未测量')}|")
    lines+=['## 检测（辅助，不等于编辑起止定位）','|条件|帧ACC|帧AUROC|帧AP|视频ACC|视频AUROC|视频AP|','|---|---:|---:|---:|---:|---:|---:|']
    for name,ev in [('A',a),('B',b)]:
        fm,vm=ev['macro']['frame_metrics'],ev['macro']['video_metrics']
        lines.append('|'+name+'|'+'|'.join(f'{v:.8f}' for v in [fm['acc'],fm['auroc'],fm['ap'],vm['acc'],vm['auroc'],vm['ap']])+'|')
    lines+=['## 实现与解释范围',
            '沿用通用DFormer四级视觉缓存和MoGe几何；不载入旧取证权重。XYZ全局scale、h=.5、长边384、padding/恢复、8槽/.1秒不变。块内可训练视觉几何调制与单头学习池化得到C0；两层128维4头联合注意力替代旧space/context，XYZ中心与相对PTS加到各层Q/K，V为归一化内容。保留全部有效块，无共有键前提。',
            'B窗口内双向、只输出窗口末目标；历史R不当作其原时刻因果输出。A按实际观测帧分组执行等价block-diagonal attention，所有窗口token仍更新；不同帧连接被屏蔽。两组相同参数/初始化/采样。重复槽保留相同实际观测身份，不冒充独立帧。',
            '当前R映射32维上下文与local32/RGB16共同定位，几何无效像素保留视觉路径；逐块e线性证据以logsumexp−logN汇聚帧logit，无均值MLP、无像素门控；视频均值帧logit。数量信息未额外输入，归一化池化的数量限制保留。',
            '像素AP为原生全画面正GT帧AP→fake视频均值→来源宏平均，real不混入。恒定预测AP=帧GT面积比例，按相同层级聚合。Dice/IoU/FPR固定logit≥0；AP@IoU未实现，不填0。开发824个唯一帧、全画面、不因几何无效裁评价。40y1x9 fake6–9仅可作无标签历史。',
            '本轮不同于旧模型和576步协议，新旧差值不能单独归因关系层、读出或预算。A/B只检验本新模型中跨时刻连接增量；单seed小增益不等于稳定收益，不证明三维优于二维。无独立起止真值，不声称验证时间边界。',
            '训练观察子集若改善而开发不改善，应记录泛化差距；若训练仍弱，表示、优化及监督尺度等解释仍待验证，不能单独归因数据量。',
            '## 采样、资源与顺序检查',json.dumps(summary['coverage'],ensure_ascii=False),
            '一次验收（权重已丢弃）：'+json.dumps(summary['resource'],ensure_ascii=False),
            '各条件成本：'+json.dumps({k:{n:v[n] for n in ['updates','training_seconds','evaluation_seconds','wall_seconds','peak_allocated','peak_reserved']} for k,v in summary['results'].items()},ensure_ascii=False),
            'FP32有限顺序检查：'+json.dumps(summary['order_check'],ensure_ascii=False),
            '先比较同输入重复差，再比较倒序效应；顺序变化不预设提高异常分数。这里只观察五个目标，不以梯度通路当学习有效证明。',
            '## 产物与执行', '执行身份：'+json.dumps(summary['identity'],ensure_ascii=False),
            '新增MoGe=0；冻结视觉编码=0。复用旧缓存只读，不改变旧实验。总墙钟秒：'+str(summary['wall_seconds']),
            '预览：preview/index.html；训练曲线：curves.png；完整配置/顺序/初始化/检查点含优化器与随机状态。独立入口：research_tools.xyz_pilot.joint_experiment。',
            '数据目录：'+str(OUT),'计算状态与Git发布状态分开，发布见publication.json。规定范围完成后停止。']
    text='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(text)


def figures(history,evaluations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for name,_ in CONDITIONS:
        hs=[x for x in history if x['condition']==name]
        axes[0].plot([x['step'] for x in hs],[x['loss'] for x in hs],label=name)
        ev=evaluations[name]['evaluations']
        axes[1].plot([int(s) for s in ev],[x['train']['macro']['pixel_AP'] for x in ev.values()],marker='o',label=name)
        de=[(int(s),x['dev']['macro']['pixel_AP']) for s,x in ev.items() if 'dev' in x]
        axes[2].plot([x[0] for x in de],[x[1] for x in de],marker='o',label=name)
    for ax,title in zip(axes,['Online total loss (100-step mean)','Fixed train monitor pixel AP','Dev pixel AP']):
        ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)
    html=['<!doctype html><meta charset="utf-8"><h1>预先固定目标：RGB / GT / 预测概率</h1><p>左右两图各自为RGB / GT / A或B概率，固定0–1色阶；仅预览缩放。无按成绩挑图。</p>']
    for group in ['train_monitor','dev']:
        html.append('<h2>'+group+'</h2>')
        folder=OUT/'preview'/group
        for path in sorted(folder.glob('XYZ_JOINT_CURRENT_*.png')):
            other=path.name.replace('XYZ_JOINT_CURRENT','XYZ_JOINT_TEMPORAL')
            html.append(f'<h3>{path.stem.removeprefix("XYZ_JOINT_CURRENT_")}</h3><img width="49%" src="{group}/{path.name}"><img width="49%" src="{group}/{other}">')
    (OUT/'preview'/'index.html').write_text('\n'.join(html))


def run():
    started=time.monotonic();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'run_identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':
            print('EXISTING_COMPLETE_REUSED',OUT,flush=True);return
        raise RuntimeError('Existing incomplete run preserved; no automatic retry')
    c=read(CONFIG);source_c=read(SOURCE/'config.json');manifest=read(SOURCE/'manifest.json');rows=manifest['rows']
    old_identity=read(SOURCE/'run_identity.json')
    assert digest(SOURCE/'manifest.json')==old_identity['manifest_sha256']
    assert digest(SOURCE/'config.json')==old_identity['config_sha256']
    for key in ['scale','h','work_long_edge','pad_multiple','train','dev','history_slots','history_step_seconds']:
        assert c[key]==source_c[key],key
    assert len(rows)==22 and len({r['source_id'] for r in rows if r['split']=='train'})==7
    assert {r['source_id'] for r in rows if r['split']=='train'}.isdisjoint({r['source_id'] for r in rows if r['split']=='dev'})
    for row in rows:
        files=list((SOURCE/'cache'/row['video_id']).glob('*.pt'))
        assert len(files)==row['actual_frames']
        assert all((SOURCE/'cache'/row['video_id']/f'{i:05d}.pt').exists() for i in range(row['actual_frames']))
    save(OUT/'config.json',c);save(OUT/'manifest.json',manifest)
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_sha256=digest(CONFIG),manifest_sha256=digest(SOURCE/'manifest.json'),source_run_identity=old_identity,
        source_sha256={name:digest(REPO/name) for name in ['research_tools/xyz_pilot/joint_model.py','research_tools/xyz_pilot/joint_experiment.py','research_tools/xyz_pilot/model.py','research_tools/xyz_pilot/data.py']})
    save(OUT/'run_identity.json',identity)
    sequence=orders(rows,c);assert sequence[:576]==read(SOURCE/'orders.json')
    save(OUT/'orders.json',sequence);cov=coverage(sequence,rows);save(OUT/'coverage.json',cov)
    targets=observation_targets(rows);save(OUT/'train_monitor_targets.json',targets)
    save(OUT/'sequence_identity.json',dict(sha256=digest(OUT/'orders.json'),train_monitor_sha256=digest(OUT/'train_monitor_targets.json'),same_prefix_576=True))
    cache=Cache(c);byid={r['video_id']:r for r in rows};train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    seed();model=XYZJointPilot(c['h']);initial={k:v.clone() for k,v in model.state_dict().items()};initial_hash=state_hash(initial)
    torch.save(initial,OUT/'initial.pt');save(OUT/'initialization.json',dict(sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),forensic_initialization=False,modules=str(model)))
    counts=read(SOURCE/'scale_selection.json')['node_counts'];candidates=[]
    for row in train:
        for i in range(row['actual_frames']):
            if eligible(row,i):
                ids=history_indices(row['pts'],i,c['history_slots'],c['history_step_seconds'])
                candidates.append((sum(counts[row['video_id']][str(c['h'])][j] for j in ids if j is not None),row['video_id'],i))
    total,vid,target=max(candidates);progress('resource',video_id=vid,target=target,window_blocks=total)
    model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    torch.cuda.reset_peak_memory_stats();free,gpu_total=torch.cuda.mem_get_info()
    resource_result=update(model,opt,cache,byid[vid],target,True,True)
    assert resource_result['window_blocks']==total
    assert all(x['present'] and x['finite'] for x in resource_result['gradients'].values())
    resource_result.update(status='PASS',video_id=vid,target=target,free_before=free,total_gpu=gpu_total,optimizer_states=len(opt.state),acceptance_weights_discarded=True)
    save(OUT/'resource.json',resource_result)
    print('RESOURCE_PASS; worst-window step estimate, NOT measured full-run wall:',resource_result['seconds']*c['updates']*2,flush=True)
    del model,opt;torch.cuda.empty_cache();results={};history=[]
    for condition,temporal in CONDITIONS:
        seed();model=XYZJointPilot(c['h']).cuda();model.load_state_dict(initial,strict=True)
        assert state_hash(model.state_dict())==initial_hash
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        torch.cuda.reset_peak_memory_stats();begin=time.monotonic();training_seconds=0.;eval_seconds=0.;evs={}
        progress('initial_monitor',condition=condition)
        observed,_=evaluate(model,cache,train,c,temporal,targets);eval_seconds+=observed['seconds']
        evs['0']=dict(train=observed);save(OUT/f'{condition}_step00000_train.json',observed)
        block=[]
        with (OUT/f'{condition}_updates.jsonl').open('w') as log:
            for step,item in enumerate(sequence,1):
                record=update(model,opt,cache,byid[item['video_id']],item['target'],temporal)
                training_seconds+=record['seconds'];record.update(condition=condition,step=step,video_id=item['video_id'],target=item['target'])
                log.write(json.dumps(record)+'\n');log.flush();block.append(record)
                if step%100==0 or step in c['evaluation_updates']:
                    row=dict(condition=condition,step=step,**{k:float(np.mean([x[k] for x in block])) for k in ['loss','pixel_BCE','soft_Dice','frame_BCE']},
                             seconds=sum(x['seconds'] for x in block),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                    history.append(row);csvsave(OUT/'history.csv',history);block=[]
                if step%500==0 or step in c['evaluation_updates']:
                    elapsed=time.monotonic()-begin
                    progress('training',condition=condition,step=step,training_seconds=training_seconds,elapsed=elapsed,remaining_training_estimate=training_seconds/step*(c['updates']-step))
                    checkpoint(OUT/f'{condition}_last.pt',model,opt,step,condition,identity,initial_hash)
                if step in c['evaluation_updates']:
                    progress('evaluation',condition=condition,step=step)
                    observed,pred=evaluate(model,cache,train,c,temporal,targets,capture=step==c['updates'])
                    evaluated,dpred=evaluate(model,cache,dev,c,temporal,capture=step==c['updates'])
                    eval_seconds+=observed['seconds']+evaluated['seconds']
                    assert evaluated['scored_frames']==824
                    evs[str(step)]=dict(train=observed,dev=evaluated)
                    for label,data in [('train',observed),('dev',evaluated)]:
                        save(OUT/f'{condition}_step{step:05d}_{label}.json',data)
                        csvsave(OUT/f'{condition}_step{step:05d}_{label}_videos.csv',data['videos'])
                        csvsave(OUT/f'{condition}_step{step:05d}_{label}_frames.csv',data['frames'])
                    save(OUT/f'{condition}_evaluations.json',evs)
                    if step==c['updates']:
                        preview(train,pred,condition,'train_monitor');preview(dev,dpred,condition,'dev')
        result=dict(updates=c['updates'],initial_sha256=initial_hash,training_seconds=training_seconds,evaluation_seconds=eval_seconds,
                    wall_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),evaluations=evs)
        results[condition]=result;save(OUT/f'{condition}_result.json',result)
        if temporal:
            progress('fp32_order_check',condition=condition)
            check=order_check(model,cache,dev,c);save(OUT/'order_check.json',check)
        del model,opt;torch.cuda.empty_cache()
    for step in c['evaluation_updates']:
        a,b=[results[n]['evaluations'][str(step)] for n,_ in CONDITIONS]
        for label in ['train','dev']:assert a[label]['frame_identity_sha256']==b[label]['frame_identity_sha256']
    summary=dict(status='COMPLETE',exit_code=0,identity=identity,config=c,coverage=cov,resource=resource_result,
                 results=results,order_check=check,wall_seconds=time.monotonic()-started,
                 peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 new_geometry_frames=0,new_frozen_visual_frames=0,same_scoring_frames_and_pixels=True)
    save(OUT/'summary.json',summary);figures(history,results);write_report(summary)
    save(OUT/'status.json',dict(status='COMPLETE',exit_code=0,stage='report_complete'))


def main():
    try:
        run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'error.log').write_text(traceback.format_exc())
        save(OUT/'status.json',dict(status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'FAILED',exit_code=1,error=str(error)))
        raise


if __name__=='__main__':main()
