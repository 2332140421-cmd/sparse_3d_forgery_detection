"""ADR-0049: full-video source-coverage comparison, one shared step budget.

Reuse generic observation extraction, localization model/update/evaluate. New
protocol alone assigns accepted candidates to train; historical assets are read-only.
"""
import collections
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from .data import OUT as OBS, REPO, CAND, ROOT, read, work_shape, extract_video, backbone
from .insource_model import XYZLocalization
from .model import route
from .insource_probe import OUT as INSOURCE, PartitionCache, partition_history, epoch_orders, update, evaluate
from .experiment import eligible, seed, csvsave
from .joint_experiment import save, digest, checkpoint, preview
from .readout_adaptation import add_area, METRICS
from research_tools.rgbd_slice.experiment import state_hash

OUT=ROOT/'pvst_xyz_source_expansion_v1'
REVIEW=CAND/'anakin_pvst_candidate_review_v3'
NEW=['an:4ymrn4','an:4wge37','an:71gz67','an:4xeqd7']
ARMS=['TRAIN7_FULL','TRAIN_EXPANDED_FULL']
MODE='XYZ_TEMPORAL'
QUOTE='人工审阅，镜头基本固定；伪造类型基本全是移除、新增原本画面中一部分运动的物体或人。'


def explicit_exclusions(value):
    """Only explicit per-ID negative mask evidence overrides batch permission."""
    found=set()
    if isinstance(value,list):
        for x in value:found|=explicit_exclusions(x)
    elif isinstance(value,dict):
        pair=value.get('pair',value.get('pair_id'))
        if pair:
            for k in ['mask','raw_mask','human_semantic_alignment','human_semantic_mask']:
                text=value.get(k)
                if isinstance(text,str) and ('不对应' in text or text.lower() in ['does_not_correspond','mismatch','incorrect']):found.add(pair)
        for x in value.values():
            if isinstance(x,(dict,list)):found|=explicit_exclusions(x)
    return found


def full_train_row(source):
    row=copy.deepcopy(source);n=row['actual_frames'];row['train_range']=[0,n]
    row['evaluation_range']=[0,0];row['evaluation_targets']=[];row['preview_targets']=[]
    row['train_targets']=[i for i in range(n) if eligible(row,i)]
    targets=row['train_targets'];row['monitor_targets']=[targets[int(f*(len(targets)-1))] for f in [1/3,2/3]]
    return row


def training_order(rows,budget,seed_value):
    n=sum(len(r['train_targets']) for r in rows)
    return [v for seq in epoch_orders(rows,math.ceil(budget/n),seed_value) for v in seq][:budget]


def coverage(sequence,rows):
    result=[]
    for row in rows:
        targets=[v['target'] for v in sequence if v['video_id']==row['video_id']];freq=collections.Counter(targets)
        result.append(dict(video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],presentations=len(targets),unique_targets=len(freq),eligible_targets=len(row['train_targets']),min_exposure=min(freq.values()),max_exposure=max(freq.values()),target_exposures=dict(sorted(freq.items()))))
    return dict(updates=len(sequence),real=sum(r['presentations'] for r in result if r['role']=='real'),fake=sum(r['presentations'] for r in result if r['role']=='fake'),per_video=result)


def progress(stage,**kw):
    r=dict(status='RUNNING',stage=stage,**kw);save(OUT/'status.json',r);print(json.dumps(r),flush=True)


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    c=read(INSOURCE/'config.json');manifest=read(OBS/'manifest.json');old=[full_train_row(r) for r in manifest['rows'] if r['split']=='train'];dev=[]
    for source in manifest['rows']:
        if source['split']!='dev':continue
        r=copy.deepcopy(source);n=r['actual_frames'];r.update(evaluation_range=[0,n],evaluation_targets=list(range(n)),preview_targets=[0,n//2,n-1]);dev.append(r)
    human_paths=sorted(set(REVIEW.glob('*human*.json'))|{Path('/media/njupt/8FCD3869E2FAB06F/chenyh/reports/human_review.json')})
    human=[dict(path=str(p),sha256=digest(p),raw=read(p)) for p in human_paths if p.is_file()]
    excluded=explicit_exclusions(human);inventory=read(REVIEW/'inventory.json');added=[];omitted=[]
    for pair in NEW:
        item=next(r for r in inventory if r['pair_id']==pair)
        if pair in excluded:omitted.append(dict(pair_id=pair,reason='explicit per-ID negative mask review; batch permission does not override'));continue
        if item['mask']['status']!='PASS' or item['split'] not in ['train','candidate_unassigned']:
            omitted.append(dict(pair_id=pair,reason='technical/role exclusion'));continue
        for role in ['real','fake']:
            m=item['media'][role];hw=list(m['sizes_hw'][0]);wh=work_shape(hw,c['work_long_edge'])
            assert digest(m['path'])==m['sha256'] and digest(item['mask']['path'])==item['mask']['mask_sha256']
            row=dict(pair_id=pair,source_id=item['source_id'],split='train',previous_role=item['split'],role=role,video_id=pair[3:]+'_'+role,video_path=m['path'],video_sha256=m['sha256'],mask_zip=item['mask']['path'],mask_sha256=item['mask']['mask_sha256'],index_base=item['mask']['index_base'],actual_frames=m['frame_count'],pts=[p['seconds'] for p in m['pts']],native_hw=hw,work_hw=list(wh),padded_hw=[math.ceil(x/32)*32 for x in wh],official_task=item['official_task'],human_approval='Current user batch permission; no fabricated per-ID description or mask approval')
            added.append(full_train_row(row))
    assert added,'No authorized technically valid new sources; no replacement'
    expanded=old+added;tr={r['source_id'] for r in expanded};de={r['source_id'] for r in dev}
    assert tr.isdisjoint(de) and len(tr)==len(expanded)//2 and len(de)==4
    assert len({r['video_sha256'] for r in expanded+dev if r['role']=='real'})==len(tr)+4
    # Read reserved identity metadata only, never its media.
    asset=ROOT.parent/'secondary_datasets/anakin';meta={r['video-id']:r for r in csv.DictReader((asset/'references/metadata.csv').open())}
    reserved={meta[i]['url'] for i in read(asset/'metadata_audit.json')['config_splits']['has-masks']['test']}
    assert all(r['source_id'].removeprefix('anakin:') not in reserved for r in expanded)
    n7=sum(len(r['train_targets']) for r in old);nexp=sum(len(r['train_targets']) for r in expanded);budget=10*nexp
    assert n7==1470 and sum(len(r['evaluation_targets']) for r in dev)==824
    c.update(sources=[r['pair_id'] for r in expanded if r['role']=='fake'],conditions=ARMS,updates_per_condition=budget,evaluation_steps=[budget//2,budget],evaluation_targets=824,targets_per_epoch={ARMS[0]:n7,ARMS[1]:nexp},split_rule='Full training source videos vs original source-disjoint dev4 full videos; old same-source holdout is now train')
    save(OUT/'config.json',c);save(OUT/'protocol.json',dict(original=old,added=added,expanded=expanded,dev=dev,N7=n7,Nexp=nexp,U=budget,omitted=omitted,source_manifest_sha256=digest(OBS/'manifest.json')))
    save(OUT/'human_review.json',dict(batch_quote=QUOTE,batch_source='current user prompt',training_authorized=True,per_ID_confirmation_not_inferred=True,existing_records=human,excluded=omitted))
    for arm,rows in [(ARMS[0],old),(ARMS[1],expanded)]:
        order=training_order(rows,budget,c['seed']);save(OUT/f'{arm}_orders.json',order);save(OUT/f'{arm}_coverage.json',coverage(order,rows))
    provider=read(OBS/'provider_config.json');assert digest(provider['dformer_weights'])==provider['dformer_weights_sha256']
    for code,revision in [('dformer_code','dformer_revision'),('moge_code','moge_revision')]:
        assert subprocess.check_output(['git','-C',provider[code],'rev-parse','HEAD'],text=True).strip()==provider[revision]
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_manifest_sha256=digest(OBS/'manifest.json'),old_initial_path=str(INSOURCE/'initial.pt'),old_initial_sha256=digest(INSOURCE/'initial.pt'),provider=dict(DFormer_weights_sha256=provider['dformer_weights_sha256'],DFormer_revision=provider['dformer_revision'],MoGe_weights_sha256=digest(provider['moge_weights']),MoGe_revision=provider['moge_revision']),protocol_sha256=digest(OUT/'protocol.json'),orders_sha256={a:digest(OUT/f'{a}_orders.json') for a in ARMS})
    save(OUT/'identity.json',identity);save(OUT/'provider_config.json',provider)
    return c,old,added,expanded,dev,identity,provider


def observations(old,added,dev,provider):
    start=time.monotonic();cache=OUT/'cache';cache.mkdir(exist_ok=True);reused=0
    for r in old+dev:
        source=OBS/'cache'/r['video_id'];assert (source/'complete.json').exists()
        assert all((source/f'{i:05d}.pt').is_file() for i in range(r['actual_frames']))
        (cache/r['video_id']).symlink_to(source,target_is_directory=True);reused+=r['actual_frames']
    plans=[]
    for r in added:
        folder=cache/r['video_id'];missing=[i for i in range(r['actual_frames']) if not(folder/f'{i:05d}.pt').exists()]
        plans.append(dict(video_id=r['video_id'],native_frames=r['actual_frames'],missing=missing))
    save(OUT/'observation_plan.json',dict(reused_old_unique_frames=reused,new_unique_frames=sum(len(p['missing']) for p in plans),per_video=plans))
    enc=backbone(provider,OUT/'official_encoder_load.json').cuda().eval().requires_grad_(False);before=state_hash(enc.state_dict());moge=None;records=[]
    for r,p in zip(added,plans):
        progress('prepare_observations',video_id=r['video_id'],missing=len(p['missing']))
        record,moge=extract_video(r,provider,enc,cache/r['video_id'],moge,missing=p['missing']);records.append(record);save(OUT/'observation_progress.json',records)
    assert state_hash(enc.state_dict())==before
    del enc,moge;torch.cuda.empty_cache()
    cost=dict(reused_old_unique_frames=reused,new_unique_frames=sum(p['native_frames'] for p in plans),new_MoGe_frames=sum(r['new_geometry_calls'] for r in records),reused_geometry_frames=sum(r['reused_geometry_frames'] for r in records),new_visual_frames=sum(r['new_visual_calls'] for r in records),seconds=time.monotonic()-start,new_cache_bytes=sum(p.stat().st_size for r in added for p in (cache/r['video_id']).glob('*.pt')),encoder_weights_buffers_unchanged=True,per_video=records)
    save(OUT/'observation_cost.json',cost);return cost


def largest_new_window(added,c):
    records=[];chosen=None;maximum=-1
    for row in added:
        counts=[]
        for i in range(row['actual_frames']):
            p=torch.load(OUT/'cache'/row['video_id']/f'{i:05d}.pt',map_location='cpu',weights_only=True)
            assert p['frame_index']==i and p['pts']==row['pts'][i]
            # Same CPU scale-once then GPU route as Cache.get/model.encode.
            _,keys,_=route((p['xyz_provider']/c['scale']).cuda(),p['valid'].cuda(),c['h'])
            counts.append(len(keys))
        for i in row['train_targets']:
            slots=partition_history(row,i,'train');n=sum(counts[j] for j in slots if j is not None)
            if n>maximum:maximum=n;chosen=(row,i,slots)
        records.append(dict(video_id=row['video_id'],blocks_per_frame=counts))
    row,i,slots=chosen;result=dict(video_id=row['video_id'],target=i,pts=row['pts'][i],window_indices=slots,window_blocks=maximum,per_video=records)
    save(OUT/'new_window_counts.json',result);return row,i,result


def fresh(c,initial):
    seed();m=XYZLocalization(c['h']);m.load_state_dict(initial,strict=True);m=m.cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);assert not opt.state
    assert state_hash(m.state_dict())==state_hash(initial)
    return m,opt


def persist_eval(stem,ev):
    add_area(ev);save(OUT/f'{stem}.json',ev);csvsave(OUT/f'{stem}_frames.csv',ev['frames']);csvsave(OUT/f'{stem}_videos.csv',ev['videos'])


def make_preview(rows,predictions):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    html=['<!doctype html><meta charset="utf-8"><h1>固定开发24目标：TRAIN7_FULL / TRAIN_EXPANDED_FULL</h1><p>每图RGB / 官方GT / 概率（统一灰阶0–1），下方logit≥0二值。仅预览缩放，不按成绩选帧。</p>']
    for arm in ARMS:
        assert len(predictions[arm])==24;preview(rows,predictions[arm],arm,'evaluation',output_root=OUT)
        for (vid,i),a in predictions[arm].items():
            h,w=a.shape;f=min(1.,480/w)
            Image.fromarray(((a>=0)*255).astype(np.uint8)).resize((round(w*f),round(h*f)),Image.Resampling.NEAREST).save(folder/'evaluation'/f'{arm}_{vid}_{i:05d}_binary.png')
    for row in rows:
        for i in row['preview_targets']:
            html.append(f'<h3>{row["pair_id"]}/{row["role"]} 帧{i} PTS {row["pts"][i]:.6f}s</h3><div style="display:flex">')
            for arm in ARMS:
                path=f'evaluation/{arm}_{row["video_id"]}_{i:05d}';html.append(f'<div style="width:50%">{arm}<a href="{path}.png"><img width="100%" src="{path}.png"></a><img width="33%" src="{path}_binary.png"></div>')
            html.append('</div>')
    (folder/'index.html').write_text('\n'.join(html))


def report(s):
    u=s['protocol']['U'];lines=['# PVST全视频训练来源扩充配对结果',f'计算COMPLETE，退出码0，两组各{u}次更新。主结果固定末次，不选择best。',
        '|指标|TRAIN7_FULL|TRAIN_EXPANDED_FULL|扩充−原7|','|---|---:|---:|---:|']
    for k in METRICS:
        a,b=[s['results'][arm]['evaluations'][str(u)]['dev']['macro'][k] for arm in ARMS];lines.append(f'|{k}|{a:.8f}|{b:.8f}|{b-a:+.8f}|')
    lines+=['\n|来源|原7 AP|扩充AP|增量|原7/扩充Dice|','|---|---:|---:|---:|---|'];table=[]
    for row in s['protocol']['dev']:
        if row['role']!='fake':continue
        vv=[next(v for v in s['results'][arm]['evaluations'][str(u)]['dev']['videos'] if v['video_id']==row['video_id']) for arm in ARMS]
        lines.append(f'|{row["pair_id"]}|{vv[0]["pixel_AP"]:.8f}|{vv[1]["pixel_AP"]:.8f}|{vv[1]["pixel_AP"]-vv[0]["pixel_AP"]:+.8f}|{vv[0]["Dice"]:.8f}/{vv[1]["Dice"]:.8f}|')
        for arm,v in zip(ARMS,vv):
            real=next(r for r in s['results'][arm]['evaluations'][str(u)]['dev']['videos'] if r['pair_id']==row['pair_id'] and r['role']=='real')
            table.append(dict(pair_id=row['pair_id'],condition=arm,**{k:real[k] if k=='real_FPR' else v[k] for k in METRICS}))
    csvsave(OUT/'source_comparison.csv',table)
    lines += ['\n## 训练与协议',
        '训练视频全部合法目标，不再使用前70%限制；旧同视频后30%本轮已参与训练，不能当未见评价。原开发4完整824帧/412正GT+412real，来源隔离不变，已多次查看，非最终测试。',
        '用户批次意见按原文保存在human_review.json，仅作为本轮授权，不伪造逐ID机位、编辑描述或mask确认；既有明确mask排除优先。新增仅指定技术PASS四对中的实际可用来源，不补其他6个待核。新增官方任务均为移除，不称已补足猫添加覆盖。',
        '两组XYZLocalization时空路径、8槽/.1秒、scale=12.880994319915771、h=.5、长边384、两层D128/4头不变。共同原initial.pt随机初值严格加载，AdamW全新状态；没有B10/B13继续训练。仅pixel BCE+soft Dice各1，BF16/FP32、clip1，无帧头/阈值搜索。',
        '同算法seed+cycle无放回循环，因集合不同，序列不可能逐项相同。扩充组完整10轮，原7循环至共同预算，尾轮部分。实际角色、来源、视频和逐目标曝光保存在各coverage.json；固定预算来源使用策略与重复次数共同变化，非严格独立样本数量效应。',
        '\n## 固定训练观察与评价','|条件|步数|共同28观察AP|共同Dice|新增来源观察AP|开发AP|开发Dice|','|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in s['results'].items():
        for step,e in r['evaluations'].items():
            extra=e['new_monitor']['macro']['pixel_AP'] if e['new_monitor'] else None
            lines.append(f'|{arm}|{step}|{e["common_monitor"]["macro"]["pixel_AP"]:.8f}|{e["common_monitor"]["macro"]["Dice"]:.8f}|{extra if extra is not None else "不适用"}|{e["dev"]["macro"]["pixel_AP"]:.8f}|{e["dev"]["macro"]["Dice"]:.8f}|')
    lines+=['观察集只用于固定拟合观察，不代表完整训练集；在线loss见history.csv，与固定观察分开。AP使用正GT fake帧原生FP32排序→视频→四源宏平均，real不混AP；阈值logit≥0，FPR和预测面积单列。',
        '\n## 成本、身份与终态',json.dumps(s['observation_cost'],ensure_ascii=False),json.dumps({a:{k:v for k,v in r.items() if k!='evaluations'} for a,r in s['results'].items()},ensure_ascii=False),
        f'执行HEAD {s["identity"]["execution_head"]}，训练主流程{s["wall_seconds"]:.2f}秒。新MoGe/视觉计算量以observation_cost.json为准，不写成0。',
        f'结果 {OUT}；固定24目标预览 {OUT}/preview/index.html。旧结果保留，计算与Git发布分开。完成后停止。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for arm in ARMS:
        h=[x for x in s['history'] if x['condition']==arm];ax.plot([x['updates'] for x in h],[x['loss'] for x in h],label=arm)
    ax.set_xlabel('Updates');ax.set_ylabel('Mean online pixel BCE + Dice');ax.legend();fig.tight_layout();fig.savefig(OUT/'loss_curve.png',dpi=140);plt.close(fig)


def run():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(2)
    c,old,added,expanded,dev,identity,provider=freeze();cost=observations(old,added,dev,provider)
    initial=torch.load(INSOURCE/'initial.pt',weights_only=True,map_location='cpu')['XYZ'];assert state_hash(initial)==read(INSOURCE/'initialization.json')['XYZ_sha256']
    ref=XYZLocalization(c['h']);ref.load_state_dict(initial,strict=True);del ref
    save(OUT/'initialization.json',dict(source=str(INSOURCE/'initial.pt'),key='XYZ',sha256=state_hash(initial),trained_weights=False,optimizer='fresh AdamW',parameters=sum(v.numel() for v in initial.values())));torch.save({'XYZ':initial},OUT/'initial.pt')
    cache=PartitionCache(c,cache_root=OUT/'cache');row,i,count=largest_new_window(added,c)
    progress('real_preflight',video_id=row['video_id'],target=i,window_blocks=count['window_blocks'])
    m,opt=fresh(c,initial);torch.cuda.reset_peak_memory_stats();check=update(m,opt,cache,row,i,MODE,True)
    assert check['window_blocks']==count['window_blocks']
    check.update(discarded=True,video_id=row['video_id'],target=i,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save(OUT/'preflight.json',check)
    del m,opt;torch.cuda.empty_cache();print('THROUGHPUT_ESTIMATE',check['seconds']*2*c['updates_per_condition'],'seconds; first real update only, not measured whole-run time',flush=True)
    results={};history=[];predictions={};byid={r['video_id']:r for r in expanded};u=c['updates_per_condition']
    for arm,rows in [(ARMS[0],old),(ARMS[1],expanded)]:
        m,opt=fresh(c,initial);begin=time.monotonic();torch.cuda.reset_peak_memory_stats();training_seconds=evaluation_seconds=0.;evs={};recent=[]
        sequence=read(OUT/f'{arm}_orders.json');progress('train',condition=arm,updates=0,U=u)
        with (OUT/f'{arm}_updates.jsonl').open('x') as log:
            for step,item in enumerate(sequence,1):
                rec=update(m,opt,cache,byid[item['video_id']],item['target'],MODE);training_seconds+=rec['seconds']
                rec.update(condition=arm,updates=step,cycle=item['epoch'],video_id=item['video_id'],target=item['target']);log.write(json.dumps(rec)+'\n');log.flush();recent.append(rec)
                if step%500==0 or step in c['evaluation_steps']:
                    h=dict(condition=arm,updates=step,**{k:float(np.mean([x[k] for x in recent])) for k in ['loss','pixel_BCE','soft_Dice']},seconds=sum(x['seconds'] for x in recent));recent=[];history.append(h);csvsave(OUT/'history.csv',history)
                    progress('train',condition=arm,updates=step,U=u,remaining_train_seconds=training_seconds/step*(u-step))
                if step in c['evaluation_steps']:
                    checkpoint(OUT/f'{arm}_step{step:06d}.pt',m,opt,step,arm,identity,state_hash(initial));progress('evaluate',condition=arm,updates=step)
                    common,_=evaluate(m,cache,old,MODE,monitor=True);persist_eval(f'{arm}_step{step}_common_monitor',common)
                    extra=None
                    if arm==ARMS[1]:extra,_=evaluate(m,cache,added,MODE,monitor=True);persist_eval(f'{arm}_step{step}_new_monitor',extra)
                    ev,images=evaluate(m,cache,dev,MODE,capture=step==u,capture_dtype=np.float32);persist_eval(f'{arm}_step{step}_dev',ev)
                    assert ev['scored_frames']==824
                    evs[str(step)]=dict(common_monitor=common,new_monitor=extra,dev=ev)
                    evaluation_seconds+=common['seconds']+ev['seconds']+(extra['seconds'] if extra else 0)
                    if step==u:predictions[arm]=images
        assert step==u
        results[arm]=dict(updates=step,training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,wall_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),initial_sha256=state_hash(initial),evaluations=evs)
        save(OUT/f'{arm}_result.json',results[arm]);del m,opt;torch.cuda.empty_cache()
    for step in c['evaluation_steps']:
        assert len({results[a]['evaluations'][str(step)]['dev']['frame_identity_sha256'] for a in ARMS})==1
        assert len({results[a]['evaluations'][str(step)]['common_monitor']['frame_identity_sha256'] for a in ARMS})==1
    make_preview(dev,predictions)
    s=dict(status='COMPLETE',exit_code=0,identity=identity,config=c,protocol=read(OUT/'protocol.json'),observation_cost=cost,preflight=check,results=results,history=history,wall_seconds=time.monotonic()-start)
    save(OUT/'summary.json',s);report(s);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


def main():
    if (OUT/'identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':print('EXISTING_COMPLETE_REUSED');return
        raise SystemExit('Existing incomplete run preserved, no automatic retry')
    try:run()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(e)));raise


if __name__=='__main__':main()
