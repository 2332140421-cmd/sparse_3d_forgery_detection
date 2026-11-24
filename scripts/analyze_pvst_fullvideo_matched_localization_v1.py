#!/usr/bin/env python3
"""CPU-only closeout of existing PVST matched-video results; no torch imports.

Reads JSON/CSV/log metadata only. Derived files stay in result/analysis; the
optional repository report is explicit. Never calls a training/evaluation entry.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

DEFAULT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_fullvideo_matched_localization_v1')
METRICS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area']
FIELDS=METRICS+['constant_AP']
GROUPS=['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照']
HIST=['an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4']
MODES=['FRAME_ONLY','FULL_VIDEO']


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def write_csv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def average(values):
    values=[v for v in values if v is not None]
    return statistics.fmean(values) if values else None

def difference(a,b,fields=METRICS):
    """a minus b; absent results remain absent, never fabricated as zero."""
    return {k:a.get(k)-b.get(k) if a.get(k) is not None and b.get(k) is not None else None for k in fields}

def directions(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return dict(up=sum(v>0 for v in values),down=sum(v<0 for v in values),same=sum(v==0 for v in values),
                median_absolute=statistics.median(abs(v) for v in values) if values else None)

def contribution(rows,key='pixel_AP'):
    valid=[r for r in rows if r.get(key) is not None];positive=[r for r in valid if r[key]>0]
    if not positive:return dict(largest_positive=None,positive_share=None,remaining_mean=None)
    largest=max(positive,key=lambda r:r[key]);remaining=[r[key] for r in valid if r['pair_id']!=largest['pair_id']]
    return dict(largest_positive=largest['pair_id'],delta=largest[key],
                positive_share=largest[key]/sum(r[key] for r in positive),
                contribution_to_full_mean=largest[key]/len(valid),remaining_mean=average(remaining))


def close(a,b):
    assert (a is None)==(b is None),(a,b)
    if a is not None:assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)


def validate_evaluation(ev,rows,training_observation=False):
    byid={r['video_id']:r for r in rows}
    target='train_targets' if training_observation else 'evaluation_targets'
    expected={(r['video_id'],i,r['pts'][i],math.prod(r['native_hw'])) for r in rows for i in r[target]}
    actual={(f['video_id'],f['frame_index'],f['pts'],f['native_pixels']) for f in ev['frames']}
    assert len(actual)==len(ev['frames'])==ev['scored_frames'] and actual==expected
    for f in ev['frames']:
        assert f['source_group_id']==byid[f['video_id']]['source_group_id']
        assert 0<=f['GT_pixels']<=f['native_pixels']
        if f['role']=='real':assert f['GT_pixels']==0
        if f['GT_pixels']==0:assert f['pixel_AP'] is None
        else:assert f['role']=='fake' and f['pixel_AP'] is not None
    for video in ev['videos']:
        fs=[f for f in ev['frames'] if f['video_id']==video['video_id']]
        assert len(fs)==video['frames']
        for k in FIELDS:close(video.get(k),average(f.get(k) for f in fs))
    for source in ev['sources']:
        vs=[v for v in ev['videos'] if v['source_group_id']==source['source_group_id']]
        for k in FIELDS:
            role='real' if k=='real_FPR' else 'fake'
            close(source.get(k),average(v.get(k) for v in vs if v['role']==role))
    for k in FIELDS:close(ev['macro'].get(k),average(s.get(k) for s in ev['sources']))
    return sorted((f['video_id'],f['frame_index'],f['pts'],f['GT_pixels'],f['native_pixels']) for f in ev['frames'])


def analyze(root,out):
    assert out.resolve().is_relative_to((root/'analysis').resolve()),'Derived analysis must stay under analysis/'
    out.mkdir(parents=True,exist_ok=True);inputs={}
    def load(name):
        p=root/name;inputs[name]=sha(p);return read(p)
    def loadcsv(name):
        p=root/name;inputs[name]=sha(p);return list(csv.DictReader(p.read_text().splitlines()))
    status=load('status.json');summary=load('summary.json');assert status['status']==summary['status']=='COMPLETE' and status['exit_code']==0
    p=load('protocol.json');cfg=load('config.json');compute=load('compute.json');initial=load('initialization.json');ident=load('identity.json')
    orders=load('orders.json');monitor=load('monitor.json');cats=load('categories.json')
    rows=p['rows'];dev=[r for r in rows if r['split']=='dev'];train=[r for r in rows if r['split']=='train'];rowmap={r['video_id']:r for r in rows}
    assert len(train)==178 and len(dev)==38 and len({r['source_group_id'] for r in train})==89 and len({r['source_group_id'] for r in dev})==19
    assert {r['source_group_id'] for r in train}.isdisjoint({r['source_group_id'] for r in dev})
    assert p['protected_test_sources']==94 and sum(len(r['train_targets']) for r in train)==22878
    assert sum(r['actual_frames'] for r in rows)==28034 and cfg['epochs']==3
    assert compute['identity']==ident and compute['resource']=='PASS'
    reference=load('frozen_reference.json');keys=validate_evaluation(reference,dev)
    models={'冻结二维epoch01':reference};all_evals={};curves=[];epoch_groups=[];resources={};best=[];comparisons=[]
    old_history=loadcsv('history.csv');old_primary=loadcsv('primary_epoch3.csv');loadcsv('group_metrics.csv')
    old_delta=loadcsv('FULL_VIDEO_minus_FRAME_ONLY_sources.csv');old_best=loadcsv('best_development_supplement.csv')
    for mode in MODES:
        complete=load(mode+'/complete.json');assert complete==summary['complete_modes'][mode]
        assert complete['status']=='COMPLETE' and complete['updates']==534 and complete['supervised_presentations']==68634
        assert complete['final_epoch']==3 and complete['identity']['protocol']==ident
        assert complete['identity']['query_chunk']==compute['query_chunk'] and complete['identity']['initial_sha256']==initial['state_sha256']
        updates=loadcsv(mode+'/updates.csv');assert len(updates)==534
        assert [r['video_id'] for r in updates]==[vid for seq in orders for vid in seq]
        counts=collections.Counter(r['video_id'] for r in updates);assert len(counts)==178 and set(counts.values())=={3}
        assert sum(int(r['supervised_frames']) for r in updates)==68634
        exposure=loadcsv(mode+'/exposure.csv');assert sum(int(x['supervised_frames']) for x in exposure)==68634
        for x in exposure:assert int(x['presentations'])==3 and int(x['supervised_frames'])==3*len(rowmap[x['video_id']]['train_targets'])
        eval_seconds=observation_seconds=0
        for epoch in [1,2,3]:
            ev=load(f'{mode}/epoch{epoch:02d}_dev.json');assert validate_evaluation(ev,dev)==keys
            obs=load(f'{mode}/epoch{epoch:02d}_monitor.json');validate_evaluation(obs,[rowmap[v] for v in monitor],True)
            loss=load(f'{mode}/epoch{epoch:02d}_loss.json')
            for k in ['loss','pixel_BCE','soft_Dice']:close(loss[k],average(float(r[k]) for r in updates if int(r['epoch'])==epoch))
            history=next(r for r in old_history if r['mode']==mode and int(r['epoch'])==epoch)
            for k in FIELDS:close(ev['macro'][k],float(history[k]));close(obs['macro'][k],float(history['train_observation_'+k]))
            curves.append(dict(mode=mode,epoch=epoch,updates=178*epoch,observation_frames=obs['scored_frames'],**loss,
                               **{'dev_'+k:v for k,v in ev['macro'].items()},**{'train_observation_'+k:v for k,v in obs['macro'].items()}))
            for group,metrics in ev['groups'].items():epoch_groups.append(dict(mode=mode,epoch=epoch,group=group,**metrics))
            all_evals[(mode,epoch)]=ev;eval_seconds+=ev['seconds'];observation_seconds+=obs['seconds']
        selected=max([1,2,3],key=lambda e:(all_evals[(mode,e)]['groups']['新增15开发来源']['pixel_AP'],-e))
        assert complete['best_epoch']==selected
        old=next(r for r in old_best if r['mode']==mode);assert int(old['selected_epoch'])==selected
        best.append(dict(mode=mode,selected_epoch=selected,updates=178*selected,**all_evals[(mode,selected)]['macro']))
        models[mode]=all_evals[(mode,3)]
        res=dict(train_update_seconds=sum(float(r['seconds']) for r in updates),dev_evaluation_seconds=eval_seconds,
                 training_observation_seconds=observation_seconds,peak_allocated_bytes=max(int(r['peak_allocated_bytes']) for r in updates),
                 peak_reserved_bytes=max(int(r['peak_reserved_bytes']) for r in updates))
        close(res['train_update_seconds'],summary['resources'][mode]['train_seconds']);resources[mode]=res
    primary=[];group_deltas=[]
    for group in GROUPS:
        for label,ev in models.items():
            row=ev['groups'][group];primary.append(dict(condition=label,group=group,**row))
            old_label='FROZEN_EPOCH01' if label=='冻结二维epoch01' else label
            saved=next(x for x in old_primary if x['condition']==old_label and x['group']==group)
            for k in FIELDS:close(row[k],float(saved[k]))
        for a,b in [('FULL_VIDEO','FRAME_ONLY'),('FRAME_ONLY','冻结二维epoch01'),('FULL_VIDEO','冻结二维epoch01')]:
            group_deltas.append(dict(comparison=a+'−'+b,group=group,**difference(models[a]['groups'][group],models[b]['groups'][group])))
    long=[];deltas=[];source_by={label:{s['pair_id']:s for s in ev['sources']} for label,ev in models.items()}
    for pair in sorted(source_by['FRAME_ONLY']):
        for label,values in source_by.items():long.append(dict(condition=label,**cats[pair],**values[pair]))
        ds=dict(pair_id=pair,**cats[pair],constant_AP=source_by['冻结二维epoch01'][pair]['constant_AP'])
        for a,b,tag in [('FULL_VIDEO','FRAME_ONLY','B_minus_A'),('FRAME_ONLY','冻结二维epoch01','A_minus_base'),('FULL_VIDEO','冻结二维epoch01','B_minus_base')]:
            ds.update({tag+'_'+k:v for k,v in difference(source_by[a][pair],source_by[b][pair]).items()})
        saved=next(r for r in old_delta if r['pair_id']==pair)
        for k in METRICS:close(ds['B_minus_A_'+k],float(saved[k]))
        deltas.append(ds)
    ab=[dict(pair_id=d['pair_id'],**{k:d['B_minus_A_'+k] for k in METRICS}) for d in deltas]
    direction={k:directions(ab,k) for k in ['pixel_AP','Dice','GT_recall']}
    obj=[r for r in ab if cats[r['pair_id']]['content_priority']=='object_edit_pending_human']
    direction['object_AP']=directions(obj,'pixel_AP')
    direction['AP_Dice_recall_all_positive']=sum(all(r[k]>0 for k in ['pixel_AP','Dice','GT_recall']) for r in ab)
    direction['object_all_positive']=sum(all(r[k]>0 for k in ['pixel_AP','Dice','GT_recall']) for r in obj)
    contributor=contribution(ab)
    epoch_delta=[dict(epoch=e,**difference(all_evals[('FULL_VIDEO',e)]['macro'],all_evals[('FRAME_ONLY',e)]['macro'])) for e in [1,2,3]]
    queue=load('queue_stats.json');assert len(queue)==216 and sum(q['T'] for q in queue)==28034
    extrema=load('queue_extrema.json');prepared=load('prepared.json');assert prepared['identity']==ident and prepared['status']=='COMPLETE'
    queue_ranges={}
    for split in ['train','dev']:
        qs=[q for q in queue if q['split']==split]
        queue_ranges[split]=dict(T_min=min(x['T'] for x in qs),T_max=max(x['T'] for x in qs),M_min=min(x['M'] for x in qs),M_max=max(x['M'] for x in qs),N_min=min(x['N_min'] for x in qs),N_max=max(x['N_max'] for x in qs))
        assert extrema[split]['max_M']['M']==queue_ranges[split]['M_max']
    resource_gate=load('resource_chunk0_PASS.json')
    calls=collections.Counter();unique=collections.defaultdict(set);inputs['frontend_calls.jsonl']=sha(root/'frontend_calls.jsonl')
    for line in (root/'frontend_calls.jsonl').open():
        event=json.loads(line);calls[event['kind']]+=1;unique[event['kind']].add((event['video_id'],event['frame_index']))
    assert dict(calls)==summary['new_frontend_calls']
    costs={p.name:load(p.name) for p in root.glob('*cost_*.json')}
    old_failures=[load(p.name) for p in root.glob('failure_*.json') if 'geometry_cost_' in p.read_text()]
    assert len(old_failures)==1 and old_failures[0]['exit_code']==1
    original_failed_seconds=old_failures[0]['seconds'];successful_seconds=status['seconds']
    logs=sorted((root/'invocations').glob('*.log'),key=lambda p:p.stat().st_mtime);latest=logs[-1]
    assert latest.with_suffix('.exit_code').read_text().strip()=='0'
    inputs[str(latest.relative_to(root))]=sha(latest)
    targets=[(r,i) for r in dev for i in r['preview_targets']];assert len(targets)==54
    missing_preview=[]
    for row,i in targets:
        stem=row['video_id']+f'_{i:05d}'
        for suffix in ['rgb','gt','FROZEN_EPOCH01_prob','FROZEN_EPOCH01_mask','FRAME_ONLY_prob','FRAME_ONLY_mask','FULL_VIDEO_prob','FULL_VIDEO_mask']:
            if not (root/'preview'/(stem+'_'+suffix+'.png')).is_file():missing_preview.append(stem+'_'+suffix)
    observed_labels=dict(fake=sum(f['role']=='fake' for f in reference['frames']),real=sum(f['role']=='real' for f in reference['frames']),
                         positive_fake=sum(f['role']=='fake' and f['GT_pixels']>0 for f in reference['frames']))
    result=dict(status='COMPLETE_CPU_ANALYSIS',analysis_date='2026-11-25',protocol=dict(train_sources=89,train_videos=178,dev_sources=19,dev_videos=38,
        legal_train_frames=22878,dev_frames=5152,protected_test_sources=94,updates_per_mode=534,supervised_presentations_per_mode=68634,
        equal_initialization=True,equal_orders_and_exposures=True,query_chunk=compute['query_chunk'],initial_sha256=initial['state_sha256'],labels=observed_labels),
        primary=primary,group_deltas=group_deltas,directions=direction,contribution=contributor,epoch_deltas=epoch_delta,best=best,
        resources=resources,queue_ranges=queue_ranges,extrema=extrema,cache=prepared,preparation_costs=costs,
        frontend_counts=dict(calls),unique_frontend_frames={k:len(v) for k,v in unique.items()},
        failed_call_seconds=original_failed_seconds,successful_resume_seconds=successful_seconds,
        combined_recorded_active_seconds=original_failed_seconds+successful_seconds,
        latest_success_log=str(latest),preview_targets=len(targets),missing_preview=missing_preview,
        new_training=0,new_model_forwards=0,new_checkpoint_loads=0,new_depth=0,new_feature_extraction=0,
        next_step='固定末次权重的关系context32置零依赖诊断，保留detail32与z0，不重训；本轮不执行')
    write_csv(out/'primary_groups.csv',primary);write_csv(out/'group_deltas.csv',group_deltas)
    write_csv(out/'sources_long.csv',long);write_csv(out/'source_deltas.csv',deltas)
    write_csv(out/'historical4.csv',[r for r in long if r['pair_id'] in HIST])
    write_csv(out/'epoch_training_and_development.csv',curves);write_csv(out/'epoch_groups.csv',epoch_groups)
    write_csv(out/'epoch_B_minus_A.csv',epoch_delta);write_csv(out/'best_development.csv',best)
    write_csv(out/'epoch_sources.csv',[dict(mode=mode,epoch=epoch,**cats[source['pair_id']],**source) for (mode,epoch),ev in all_evals.items() for source in ev['sources']])
    write_json(out/'summary.json',result)
    text=render_report(root,result,models,source_by,curves,cats)
    (out/'report.md').write_text(text)
    for name,h in inputs.items():assert sha(root/name)==h,('Original input changed',name)
    write_json(out/'input_identity.json',inputs)
    write_json(out/'checks.json',dict(input_files_unchanged=True,all_development_frames_same=True,
               frame_video_source_macro_recomputed_from_saved_metrics=True,orders_exposure_initial_identity_match=True,
               original_tables_crosschecked=True,preview_files_missing=missing_preview,no_model_or_checkpoint_operations=True))
    return result,text


def fmt(v):return '未提供' if v is None else f'{v:.8f}'

def render_report(root,s,models,by,curves,categories):
    lines=['# PVST完整视频匹配定位：结果分析','',
     '分析日期：2026-11-25。**两组均正常完成，不是新中断。** 最新调用退出0、各3轮/534更新，末轮均评价5152帧。旧failure仅为此前几何统计路径错误。',
     '本轮只读JSON、CSV、日志及预览文件身份，独立CPU汇总写入analysis/；新增训练、模型前向、检查点加载、MoGe和特征提取全部为0。原程序报告、summary、指标、日志、预测及失败记录不改。',
     '', '## 结论先行','',
     '完整视频原始块模型在本89训练/19开发队列已实际跑通；但本次3轮预算没有获得有实质幅度、多来源一致的跨时刻定位收益。FULL_VIDEO末轮宏AP比FRAME_ONLY仅高0.00048280，Dice低0.00144335、召回低0.00164638；两组宏AP都略低于冻结二维epoch01。不能只用微小AP正差宣布时空关系有效。',
     '对象16来源同样只有微小AP正差，Dice/召回下降；最大AP正增量由4qwdw4单来源提供，其他18来源平均反而下降。训练观察高分与开发中等成绩存在水平差距，但观察只含两个来源，不能据此认定新支路已经过拟合或找出唯一故障层。',
     '', '## 实际协议及核对','',
     '数据直接继承扩充protocol：89训练来源/178视频，全视频每轮22878合法监督帧（原始22882，40y1x9_fake[6,10)四帧仅上下文/输出）；19开发来源/38视频/5152帧，原4为824、新增15为4328。按source_group_id隔离，不是同视频70/30，也不是跨数据集或最终测试；94保护测试来源未参与。',
     f"开发记录：{s['protocol']['labels']['fake']}个fake帧（其中{s['protocol']['labels']['positive_fake']}个正GT、5个空GT）、{s['protocol']['labels']['real']}个real帧。官方标注技术核验后使用，未要求逐条人工语义验收。",
     '两组都是XYZ等尺寸块＋定位修正，冻结同一89来源二维epoch01观测；仅计划连接范围不同。完整单视频全部帧入C，FULL_VIDEO直接双向，不跨视频、不只读过去8槽。scale=12.880994319915771/h=.5/origin0，D128/2层4头/FF256；BCE+Dice等权，AdamW1e-4/wd.01/clip1、BF16/FP32。',
     '完成记录的初态hash与initialization.json一致，两组相同；逐视频updates/exposure CSV与共同三轮orders逐项匹配，每视频3次、每组534有效更新/68634合法帧呈现。每轮开发唯一身份/PTS/GT像素数完全相同。仅从记录核对，未重新载权逐张量验权。',
     f"初态hash：`{s['protocol']['initial_sha256']}`。compute实际Query chunk={s['protocol']['query_chunk']}，非重入checkpoint，资源预检记录显示SDPA Flash前向/反向，无Q1024备用。正式训练没有逐步profiler，不能将资源预检记录冒称每步后端实测。",
     '', '## 固定第3轮主表','',
     '所有指标/FPR/面积均为比例，不是百分数；例如real FPR .0023约为.23%。AP正GT fake帧→视频→原来源等权，real不混入AP；空GT AP不填0/1。Dice/IoU/召回/FPR固定logit≥0。常数AP是同层级GT正类面积比例，不是新的归一化主指标。',
     '', '|范围|条件|AP|Dice|IoU|召回|real FPR|fake背景FPR|预测面积|常数AP|',
     '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        for label,ev in models.items():lines.append('|'+group+'|'+label+'|'+'|'.join(fmt(ev['groups'][group][k]) for k in FIELDS)+'|')
    lines += ['', '### 三类差值（不省略二维起点）','',
      '|范围|比较|ΔAP|ΔDice|ΔIoU|Δ召回|Δreal FPR|Δfake背景FPR|Δ面积|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in s['group_deltas']:lines.append('|'+row['group']+'|'+row['comparison']+'|'+'|'.join(fmt(row[k]) for k in METRICS)+'|')
    lines += ['', 'FRAME_ONLY比冻结二维：AP略降、Dice略增、召回几乎不变，误报/面积略减；FULL_VIDEO比冻结二维：AP、Dice及召回都略降，误报略减。没有证据说明新增整条支路整体优于二维起点。AP变化也不能完全归结为全局置信度平移；本轮没有原生logit修正幅度统计或同权重依赖干预，不能进一步断言纯校准问题或具体层失效。',
      '', '## 逐来源方向与历史四来源','',
      '|来源|AP 冻结/A/B|Dice 冻结/A/B|召回 冻结/A/B|',
      '|---|---|---|---|']
    for pair in HIST:
        vals=[by[n][pair] for n in ['冻结二维epoch01','FRAME_ONLY','FULL_VIDEO']]
        lines.append('|'+pair+'|'+'|'.join('/'.join(f'{v[k]:.6f}' for v in vals) for k in ['pixel_AP','Dice','GT_recall'])+'|')
    c=s['contribution'];d=s['directions']
    lines += ['',
      f"19来源B−A：AP {d['pixel_AP']['up']}升/{d['pixel_AP']['down']}降，Dice {d['Dice']['up']}升/{d['Dice']['down']}降/{d['Dice']['same']}平，召回同为4升/10降/5平。AP差绝对值中位数{d['pixel_AP']['median_absolute']:.8f}。仅{d['AP_Dice_recall_all_positive']}个来源AP/Dice/召回同时数值上升；其中对象16来源仅{d['object_all_positive']}个。方向计数包含很小的浮点差，不等于实质改善计数。",
      f"对象组AP {d['object_AP']['up']}升/{d['object_AP']['down']}降；3个外观/背景参照AP都微升。最大正AP差来自{c['largest_positive']}：{c['delta']:+.8f}，占全部正增量{c['positive_share']:.2%}，对19源均值贡献{c['contribution_to_full_mean']:+.8f}，已超过净增量；其余18源平均{c['remaining_mean']:+.8f}。这只是影响度描述，主表仍保留全部19源，不剔除它。",
      '4qwdw4：冻结/A/B AP .505452/.477037/.496727，FULL_VIDEO主要是在恢复A的下降，仍未超过二维。最大正增量不是大面积背景，但它不能替其他对象编辑来源提供证据。',
      '猫7e2e57：AP约.818三者接近；A→B Dice .527374→.524686、召回.398552→.395678。两者相对冻结仅小幅召回变化，仍只覆盖约39%–40%的GT，不能称完整定位或时间带来恢复。',
      '骑马95q8g9：B−A AP −.00585858，Dice只有+.00010392，召回继续降；两者召回约.214，均略低于冻结.219，未恢复完整移除区域。',
      '过街4xd3r9：三者Dice/召回都0，AP .012805/.013363/.013219均低于常数AP .015999，固定阈值下仍未检出。7ogeg4的B AP/Dice/召回均略低于A和冻结。',
      '对象16的冻结/A/B AP约.326195/.325495/.325961，常数AP .020269；外观背景3的AP约.666052/.665159/.665730，常数AP .488997。大面积编辑确实抬高总体AP基础，不能把全部19均值当目标运动对象问题已解决；但猫等小面积高AP也说明不是所有排序能力都来自GT面积。',
      '', '### 全部19来源（完整保留）','',
      '|来源|动作|常数AP|AP 冻结/A/B|B−A AP|Dice 冻结/A/B|召回 冻结/A/B|',
      '|---|---|---:|---|---:|---|---|']
    for pair in sorted(by['FRAME_ONLY']):
        vals=[by[n][pair] for n in ['冻结二维epoch01','FRAME_ONLY','FULL_VIDEO']]
        lines.append('|'+pair+'|'+categories[pair]['official_action']+'|'+fmt(vals[0]['constant_AP'])+'|'+'/'.join(f"{v['pixel_AP']:.6f}" for v in vals)+'|'+f"{vals[2]['pixel_AP']-vals[1]['pixel_AP']:+.8f}"+'|'+'/'.join(f"{v['Dice']:.6f}" for v in vals)+'|'+'/'.join(f"{v['GT_recall']:.6f}" for v in vals)+'|')
    lines += ['', '## 三轮训练与开发','',
      '|模式|轮|在线BCE|在线Dice loss|总loss|训练观察AP/Dice|全部19开发AP/Dice|',
      '|---|---:|---:|---:|---:|---|---|']
    for h in curves:lines.append(f"|{h['mode']}|{h['epoch']}|{h['pixel_BCE']:.8f}|{h['soft_Dice']:.8f}|{h['loss']:.8f}|{h['train_observation_pixel_AP']:.8f}/{h['train_observation_Dice']:.8f}|{h['dev_pixel_AP']:.8f}/{h['dev_Dice']:.8f}|")
    lines += ['', '|轮|B−A AP|B−A Dice|B−A召回|','|---|---:|---:|---:|']
    for e in s['epoch_deltas']:lines.append(f"|{e['epoch']}|{e['pixel_AP']:+.8f}|{e['Dice']:+.8f}|{e['GT_recall']:+.8f}|")
    lines += ['', '训练观察固定为4dpnv4添加/复制与4qw154移除各real/fake，共4完整视频、2来源；不是网络模块、不是完整训练集或块特征独立能力。该观察集约.895 AP/.783 Dice，显著高于开发.38/.25，存在水平分化；但冻结底座本已用这些来源训练，缺少相同观察清单的初始z0计分，不能把高分全归给新增支路。',
      '第1→3轮，两模式来源AP方向都是11升8降，不能概括为普遍后期退化。A最大下降为4qwdw4（−.028344），其次4ny267（−.001724）；B为4qwdw4（−.008651）、7ogeg4（−.002580）、4nyv37（−.001848）、95q8g9（−.001624）。两者71g867均提高约.01076，但其召回仍低；退化不只落在大面积背景。A新增15宏AP为.350796→.349955→.349900；B为.350787→.351663→.351084。逐来源三轮表另存epoch_sources.csv，类别来自官方描述/自动预筛，不能视为人工确认。',
      '在线loss仅小幅下降，观察小幅改善；开发没有一致单调退化：A第2轮下降、第3轮部分恢复；B第2轮AP最高，第3轮AP小降而Dice/召回回升。不能仅因各自best为1/2轮就宣布过拟合，也不能用在线loss替代固定权重完整训练集指标（后者未测量）。分范围/动作/机位三轮完整表见epoch_groups.csv。',
      '', '### 开发选择附表（不替代末轮主表）','',
      '|模式|选择轮/更新|全部19 AP|Dice|召回|real FPR|fake背景FPR|','|---|---|---:|---:|---:|---:|---:|']
    for b in s['best']:lines.append(f"|{b['mode']}|{b['selected_epoch']}/{b['updates']}|{b['pixel_AP']:.8f}|{b['Dice']:.8f}|{b['GT_recall']:.8f}|{b['real_FPR']:.8f}|{b['fake_background_FPR']:.8f}|")
    lines += ['', '选择依据是新增15开发宏AP、同分早轮。FULL_VIDEO第2轮AP略高于冻结，但Dice/召回较低；第2轮与A第1轮更新预算不同，不能作等预算主对照，也不是独立测试成绩。',
      '', '## 资源、观测与失败恢复','',
      '|范围|T范围|M范围|单帧N范围|最大M病例|','|---|---|---|---|---|']
    for split,q in s['queue_ranges'].items():
        x=s['extrema'][split]['max_M'];lines.append(f"|{split}|{q['T_min']}–{q['T_max']}|{q['M_min']}–{q['M_max']}|{q['N_min']}–{q['N_max']}|{x['video_id']}：T={x['T']}，M={x['M']}|")
    lines += ['', '|模式|训练更新段秒|3轮开发评价秒|训练观察评价秒|峰值allocated/reserved GiB|','|---|---:|---:|---:|---|']
    for mode,r in s['resources'].items():lines.append(f"|{mode}|{r['train_update_seconds']:.3f}|{r['dev_evaluation_seconds']:.3f}|{r['training_observation_seconds']:.3f}|{r['peak_allocated_bytes']/2**30:.4f}/{r['peak_reserved_bytes']/2**30:.4f}|")
    lines += ['', '训练段时间不含其前的CPU缓存加载、其后的检查点持久化；评价时间包含加载和原生计分。不能简单相加冒称所有阶段墙钟，也不能由FULL_VIDEO训练段较短推断理论复杂度更低。',
      f"原失败调用{s['failed_call_seconds']:.6f}秒（约51分33秒）；成功续跑{s['successful_resume_seconds']:.6f}秒（约1小时55分25秒）；两次有记录活跃墙钟合计{s['combined_recorded_active_seconds']:.6f}秒（约2小时46分58秒），不含中间等待和更早代码交付小预检。旧failure保留，但不能据其存在判当前失败。",
      f"恢复几何调用：{s['preparation_costs'][next(k for k in s['preparation_costs'] if k.startswith('geometry_cost_'))]}。它只表示本次跳过已缓存路由，不清零累计MoGe。",
      f"累计本实验新增前端journal：{s['frontend_counts']}，唯一帧计数{s['unique_frontend_frames']}；旧几何复用3397、新MoGe24637，共28034；冻结视觉复用366、新增27668，共28034。无新增DFormer/额外BN。各次准备cost原记录见analysis/summary.json，未重新提取或重复哈希大缓存。",
      f"prepared记录新owned缓存{s['cache']['owned_cache_bytes']:,}字节，约{s['cache']['owned_cache_bytes']/2**30:.2f} GiB；referenced视觉{s['cache']['referenced_visual_bytes']:,}字节含旧引用，不再与owned简单相加。准备记录耗时{s['cache']['seconds']:.3f}秒，不把其局部cost和完整调用墙钟混为一谈。",
      '几何修复只改统计JSON路径，准备阶段只更新一个非模型源码摘要，旧identity/失败状态与累计成本证据在recovery/。原几何生成来源仍为失败提交d9998c3，成功续跑为dc17fc8；本轮不重做迁移。资源门另外有4次丢弃优化更新，旧交付smoke1次单列，均不混入两组各534正式更新。',
      '', '## 预览与唯一下一步','',
      f"已有固定预览：`{root/'preview/index.html'}`。54个预定目标、预期432张RGB/GT/z0/A/B概率及mask文件；缺项{s['missing_preview']}。本轮仅整理文件身份，未新增预测或人工视觉判断。PNG为展示，不从量化图重算AP或判断细小logit变化。",
      '**唯一建议：做固定末次权重的关系context32依赖诊断。** 当前三种结果过于接近，先检验新增关系证据是否实际影响定位读出：保持权重、所有观测、块编码/关系计算、detail32与z0不变，只将pixel_context投影后回填给修正头的完整context32置零（含投影bias），与正常输出比较原生delta/logit幅度及排序变化。不重训、不同时扩数据/调学习率/改结构。本轮不执行。',
      '这能区分读出对关系支路的当前依赖，但不能单独定位唯一故障层；即使依赖弱也不能证明C完全无信息。当前结果仅限冻结底座、简化块编码、logit修正、单seed和3轮预算，不否定所有PVST或时空关系，也不证明三维独立必要性/物理稳定规律。',
      '', '## 交付与复算','',
      f"原始程序报告仍为`{root/'report.md'}`，未覆盖；本轮派生分析在`{root/'analysis'}`。",
      '主要文件：primary_groups.csv、group_deltas.csv（三类比较）、sources_long.csv、source_deltas.csv、historical4.csv、epoch_training_and_development.csv、epoch_groups.csv、epoch_sources.csv、epoch_B_minus_A.csv、best_development.csv、summary.json、checks.json及input_identity.json。原始输入SHA前后相同。',
      '纯CPU入口：`python3 scripts/analyze_pvst_fullvideo_matched_localization_v1.py`；可显式加`--repo-report docs/pvst/pvst_fullvideo_matched_localization_v1.md`更新仓库说明。脚本不导入torch、不读取.pt、不调用训练/prepare/resume入口。',
      '本轮只提交CPU分析脚本、必要测试、轻量结果和中文报告；训练代码/权重/大缓存/预测不变。Git发布状态以最终回执为准。任务到此结束，不启动或监控新实验。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result-dir',type=Path,default=DEFAULT)
    parser.add_argument('--repo-report',type=Path)
    args=parser.parse_args()
    result,text=analyze(args.result_dir,args.result_dir/'analysis')
    if args.repo_report:args.repo_report.write_text(text)
    print(json.dumps(dict(status=result['status'],output=str(args.result_dir/'analysis'),directions=result['directions'],contribution=result['contribution']),ensure_ascii=False))


if __name__=='__main__':main()
