"""One source-isolated frozen test; reuse the existing single-video processor."""
import argparse
import csv
import fcntl
import hashlib
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from research_tools.v8.component_global_context_pilot.model import GlobalContextModel
from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file
from research_tools.v8.full_coverage_observation_field.pipeline import _metric
from research_tools.v8.full_video_streaming.inputs import metadata
from research_tools.v8.full_video_streaming.runner import tensors
from research_tools.v8.full_video_training.model import VideoTraining
from research_tools.v8.local_geometry_pilot.data import digest
from research_tools.v8.local_geometry_pilot.runner import csv_rows
from . import runner
from .cache import read_block
from .model import GlobalStreamingModel
from .evaluate import SEEDS

ROOT=Path('/root/autodl-tmp/data/sparse_3d_forgery_detection')
DERIVED=ROOT/'derived'
DATA=ROOT/'datasets/v7_core_candidates/activityforensics_charades_v1'
GROUPS=('COMPONENT_ADAPTED','COMPONENT_WINDOW','CELL_WINDOW')
HISTORY=(
 'v7_activityforensics_boundary_pooling_pilot_v1/scores/oof_window_scores.csv',
 'v7_activityforensics_periodic_requery_pilot_v1/scores/oof_window_scores.csv',
 'v7_activityforensics_fixed_grid_frozen_probe_v1/scores/window_scores.csv',
 'v7_activityforensics_paired_second_order_pilot_v1/metrics/per_window_signal.csv',
)


def used_sources(rows):
    # A missing placeholder is not proof of inference; scored or measured rows are.
    found=set()
    for r in rows:
        values=[v for k,v in r.items() if ('seed_2026' in k or k in ('O_SET','R_SET','K0_S_norm'))]
        if any(v not in ('','NA','None','null') and np.isfinite(float(v)) for v in values):
            found.add(r['source_id'])
    return found


def prepare(out):
    mapping=read_json(DATA/'paired/manifests/activityforensics_source_mapping.json')
    candidates=[r for r in mapping if r['activityforensics_split']=='test' and r['source_dataset']=='charades'
        and (DATA/'source/activityforensics/raw'/r['activityforensics_file']).is_file()
        and (DATA/'source/charades/videos'/f"{r['charades_source_id']}.mp4").is_file()]
    if len({r['charades_source_id'] for r in candidates})!=19:raise ValueError('LOCAL_19_CHANGED_REVIEW_REQUIRED')
    evidence={};used={};current=read_json(DERIVED/'v8_full_video_training_budget5_v1/input_manifest.json')['rows']
    direct_code=Path(__file__).parents[2]/'v7/periodic_requery_probe/runner.py'
    evidence[str(direct_code)]=sha256_file(direct_code)
    for path in HISTORY:
        p=DERIVED/path
        with p.open() as f: sources=used_sources(list(csv.DictReader(f)))
        evidence[str(p)]=sha256_file(p)
        for s in sources:used.setdefault(s,[]).append(path)
    for name in ('v8_component_global_context_pilot_v1','v8_full_video_training_budget5_v1'):
        p=DERIVED/name/'input_manifest.json';evidence[str(p)]=sha256_file(p)
        for r in read_json(p)['rows']:used.setdefault(r['source_id'],[]).append(name+'/input_manifest.json')
    reviews={r['charades_source_id']:r for r in read_json(DATA/'review/review_manifest.json')}
    with (DATA/'source/charades/metadata/Charades_v1_test.csv').open() as f: charades={r['id']:r for r in csv.DictReader(f)}
    audit=[];rows=[]
    order=sorted({r['charades_source_id'] for r in candidates},key=lambda s:hashlib.sha256(f'20260909:{s}'.encode()).hexdigest())
    for source in order:
        variants=sorted([r for r in candidates if r['charades_source_id']==source],key=lambda r:r['activityforensics_file'])
        r=variants[0];review=reviews[source];human=[k for k in ('notes','V7_CORE_DECISION','fake_looks_realistic','fake_has_obvious_game_or_cg_style') if review.get(k)]
        reason='HISTORICAL_MEASUREMENT_OR_TRAINING' if source in used else 'AMBIGUOUS_HUMAN_DEVELOPMENT' if human else ''
        info=charades[source]
        if not info['actions'].strip():reason=reason or 'NO_OFFICIAL_ACTIVITY_METADATA'
        if r['lineage_status']!='EXACT':reason=reason or 'LINEAGE_NOT_EXACT'
        audit.append({'source':source,'eligible':not bool(reason),'reason':reason or 'NO_PRIOR_TRAIN_OR_PERFORMANCE_USE_FOUND',
                      'evidence':json_text(sorted(set(used.get(source,[])))),'review_human_fields':json_text(human),
                      'review_use':'media/provenance package; no automatic detection or human decision fields',
                      'scene':info['scene'],'generator':r['generator']})
        if reason:continue
        for role in ('real','fake'):
            path=(DATA/'source/charades/videos'/f'{source}.mp4' if role=='real' else DATA/'source/activityforensics/raw'/r['activityforensics_file'])
            sha=sha256_file(path)
            row={'pair_id':'INDEPENDENT::'+source,'source_id':source,'role':role,'split':'test','video_label':int(role=='fake'),
                 'video_path':str(path),'video_sha256':sha,'video_bytes':path.stat().st_size,'official_split':'test',
                 'generator':r['generator'] if role=='fake' else 'REAL','scene':info['scene'],
                 'label_source':'official ActivityForensics variant + matched Charades original',
                 'selection':'all eligible local official-test sources; hash seed20260909; official variant lexicographic',
                 'policy':'ALL_NORMAL_DECODE_FRAMES_NO_DOWNSAMPLING'}
            row['video_id']=digest({k:row[k] for k in ('pair_id','source_id','role','video_sha256')})[:24]
            rows.append({**row,**metadata(path)})
    if not rows:raise ValueError('NO_ELIGIBLE_LOCAL_PAIRS')
    if {r['source_id'] for r in rows}&{r['source_id'] for r in current}:raise ValueError('SOURCE_LEAKAGE')
    csv_rows(out/'eligibility_audit.csv',audit)
    save_json(out/'eligibility_evidence.json',{'files':evidence,'review_manifest_sha256':sha256_file(DATA/'review/review_manifest.json'),
        'review_readme_sha256':sha256_file(DATA/'review/README.md'),'scope':'directly related V7/V8 training and scores only; not exhaustive server audit',
        'claim':'no prior training or performance-tuning use found; source-isolated frozen test, NOT strictly never touched'})
    save_json(out/'video_manifest.json',{'rows':rows,'selection_seed':20260909,'frozen_before_scoring_unix':time.time()})
    print(f'FROZEN {len(rows)} videos {len(rows)//2} sources; exclusions={len(audit)-len(rows)//2}',flush=True)
    return rows


def json_text(value):
    import json
    return json.dumps(value,ensure_ascii=False)


def model_records():
    result=[]
    for group in GROUPS:
        root=DERIVED/('v8_full_video_training_budget5_v1' if group=='COMPONENT_ADAPTED' else 'v8_component_global_context_pilot_v1')
        condition='GLOBAL_CELL' if group=='CELL_WINDOW' else 'GLOBAL_COMPONENT';epoch=5 if group=='COMPONENT_ADAPTED' else 100
        records=[r for r in read_json(root/'models/fold_models.json')['records'] if r['condition']==condition]
        if len(records)!=3 or {r['seed'] for r in records}!=set(SEEDS):raise ValueError('MODEL_SEEDS')
        for r in records:
            if r['epoch']!=epoch or r['status']!='COMPLETE' or sha256_file(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('MODEL_HASH_EPOCH')
            result.append({**r,'group':group,'expected_epoch':epoch})
    return result


def metrics(y,z):
    m=_metric(y,z);m['fpr']=m['fp']/(m['tn']+m['fp'])
    m['balanced_accuracy']=(m['recall']+m['tn']/(m['tn']+m['fp']))/2
    return m


def bootstrap(rows,z):
    sources=sorted({r['source_id'] for r in rows});y=np.array([r['video_label'] for r in rows])
    indices=[np.array([i for i,r in enumerate(rows) if r['source_id']==s]) for s in sources]
    rng=np.random.default_rng(20260909);draws=[]
    for _ in range(10000):
        ii=np.concatenate([indices[i] for i in rng.integers(0,len(sources),len(sources))])
        draws.append([roc_auc_score(y[ii],z[g][ii]) for g in GROUPS])
    draws=np.array(draws);point=np.array([roc_auc_score(y,z[g]) for g in GROUPS])
    own={g:{'auroc':float(point[i]),'ci95':np.quantile(draws[:,i],[.025,.975]).tolist()} for i,g in enumerate(GROUPS)}
    pairs={f'{GROUPS[a]}-{GROUPS[b]}':{'difference':float(point[a]-point[b]),
           'ci95':np.quantile(draws[:,a]-draws[:,b],[.025,.975]).tolist()} for a,b in ((0,1),(1,2))}
    return own,pairs


def finish(out,rows,summaries,records,identity,started):
    ts=time.monotonic();y=np.array([r['video_label'] for r in rows]);scores=[];mm=[];z={}
    for group in GROUPS:
        zz=np.array([[s['video_logits'][f'{group}__seed{seed}'] for s in summaries] for seed in SEEDS])
        if not np.isfinite(zz).all():raise ValueError('NONFINITE_SCORES')
        z[group]=zz.mean(0)
        for seed,values in list(zip(SEEDS,zz))+[('MEAN_VIDEO_LOGIT',z[group])]:
            mm.append({'group':group,'seed':seed,**metrics(y,values)})
            scores.extend({'group':group,'seed':seed,'source':r['source_id'],'video_id':r['video_id'],'role':r['role'],
                           'label':r['video_label'],'generator':r['generator'],'logit':float(v)} for r,v in zip(rows,values))
    own,pairs=bootstrap(rows,z);ev=out/'evaluation';ev.mkdir(exist_ok=True)
    csv_rows(ev/'video_scores.csv',scores);csv_rows(ev/'metrics.csv',mm)
    per=[]
    for source in sorted({r['source_id'] for r in rows}):
        ii=np.array([i for i,r in enumerate(rows) if r['source_id']==source])
        per.append({'source':source,**{g:float(roc_auc_score(y[ii],z[g][ii])) for g in GROUPS}})
    csv_rows(ev/'per_source_metrics.csv',per)
    result={'metrics':mm,'auroc_intervals':own,'paired_auroc':pairs,'bootstrap':{'unit':'source, real/fake kept together',
            'replicates':10000,'seed':20260909,'same_draws_all_groups':True},'threshold':0,'ensemble':'mean final independent-seed full-video area LSE logits'}
    save_json(ev/'summary.json',result)
    timing={g:sum(sum(v for k,v in s['costs'].get('model_cuda_s_by_key',{}).items() if k.startswith(g+'__')) for s in summaries) for g in GROUPS}
    costs={'download_s':0,'download_bytes':0,'reason':'all selected originals available locally',
        'decoded_frames':sum(s['decoded_frames'] for s in summaries),'frontend':{k:sum(s['costs'].get('frontend',{}).get(k,0) for s in summaries) for k in ('geometry_s','tracking_s','rgb_s','features_s','cache_hit','new_geometry')},
        'group_cuda_inference_s':timing,'model_wall_s':sum(s['costs']['model_s'] for s in summaries),
        'video_processing_wall_s':sum(s['costs']['wall_s'] for s in summaries),'evaluation_s':time.monotonic()-ts,
        'cache_bytes':sum(p.stat().st_size for p in (out/'inputs').rglob('*') if p.is_file()),
        'new_storage_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    save_json(out/'costs.json',costs)
    lines=['# V8 按source隔离的冻结测试','',f'COMPLETE；{len(rows)//2} source、{len(rows)}视频、{costs["decoded_frames"]}实际解码帧，九份冻结权重评分完整；无训练。',
       f'未发现参与既往训练或性能调参的记录；并非严格从未接触的测试。历史review仅媒体准备且人工作业字段为空。4个有历史测量/评分的source排除，全部合格本地{len(rows)//2}对入选，不为凑20额外下载。0HV07曾尝试前端但无有效结构，无训练/检测分数，不将其失败占位误判为性能使用。',
       '纠正此前准备报告：review登记或未执行计划不自动构成开发污染；本轮按实际评分/训练证据判断。核对范围与排除证据见eligibility_audit.csv及eligibility_evidence.json。',
       '室内日常活动场景按官方Charades scene/actions确认；未按fake观感或几何成功率筛选。全部fake为官方Vidu，训练只见wan/fcvg；这是同数据族独立source且未见生成器的联合迁移，不能分离两个因素，也不是跨数据集测试。',
       '三个seed分别全视频面积LSE后平均最终logit；阈值0。COMPONENT_WINDOW−CELL_WINDOW是同训练阶段动态分区对照；适配−窗口是适配影响。适配−CELL不称匹配消融。',
       '', '|模型组|seed|AUROC|AP|FPR|Recall|Balanced ACC|P|F1|ACC|TN/FP/FN/TP|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for m in mm:lines.append(f'|{m["group"]}|{m["seed"]}|'+ '|'.join(f'{m[k]:.6f}' for k in ('auroc','ap','fpr','recall','balanced_accuracy','precision','f1','accuracy'))+f'|{m["tn"]}/{m["fp"]}/{m["fn"]}/{m["tp"]}|')
    primary=own['COMPONENT_ADAPTED'];lines+=['','## 主要结果与配对',json_text(own),json_text(pairs),
       '自身AUROC及配对差CI均按source簇重采样10000次/20260909，同次抽样用于全部模型组。每source一真一假，逐source排序离散且不能把帧当样本。',
       '', '## 判断与限制',f'主方法AUROC={primary["auroc"]:.6f}，CI={primary["ci95"]}。保留已冻结方法及本次全部结果；不据此调参、换样本或追加训练。',
       ('该小子集有高于随机的初步排序证据，可将真正跨数据集验证作为下一次独立决策，当前不执行。' if primary['ci95'][0]>.5 else '该小子集不足以确认可靠迁移收益，暂不扩大实验；不据此宣布方法普遍无效。'),
       '测试集小、生成器集中、媒体曾被准备性接触，不能代表整个benchmark；无帧/像素定位真值。场景类别不等于独立拍摄地点。',
       '', '## 成本与可追溯',json_text(costs),
       'group_cuda_inference_s为CUDA events的设备时间，model_wall_s为全部模型墙钟，二者不混作总成本。当前权重、统计、grouping及前端版本hash见protocol.json；视频与身份见video_manifest.json。',
       '单视频处理与64帧原子恢复沿用原入口。前端一次、九份权重共享输入，hidden仅视频首帧初始化，块间保留。已有梯度/padding/连续状态测试只读引用；本轮只做两帧九模型加载/输出检查。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'final_status.json',{'status':'COMPLETE','identity':identity,'sources':len(rows)//2,'videos':len(rows),
        'decoded_frames':costs['decoded_frames'],'weights':9,'training_started':False,'elapsed_this_invocation_s':time.monotonic()-started})
    save_json(out/'progress.json',{'status':'COMPLETE','stage':'report','completed':len(rows),'total':len(rows),'updated_unix':time.time()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.prepare:
        if (out/'video_manifest.json').exists():raise ValueError('MANIFEST_ALREADY_FROZEN')
        prepare(out);return
    rows=read_json(out/'video_manifest.json')['rows'];records=model_records();root=DERIVED/'v8_component_global_context_pilot_v1'
    stats=read_json(root/'standardizer.json');group=read_json(root/'grouping_config.json');base=DERIVED/'v8_full_coverage_observation_field_identityfix_v1'
    adapted_protocol=read_json(DERIVED/'v8_full_video_training_budget5_v1/protocol.json')
    if adapted_protocol['standardizer_sha256']!=sha256_file(root/'standardizer.json') or adapted_protocol['grouping_sha256']!=sha256_file(root/'grouping_config.json'):
        raise ValueError('ADAPTED_STATISTICS_IDENTITY')
    frozen={str(root/n):sha256_file(root/n) for n in ('standardizer.json','grouping_config.json')}
    frozen.update({r['checkpoint']:r['checkpoint_sha256'] for r in records})
    bp=read_json(base/'protocol.json')
    frozen.update({bp['frontend']['geometry']['checkpoint']:sha256_file(bp['frontend']['geometry']['checkpoint']),
                   bp['frontend']['tracking']['checkpoint']:sha256_file(bp['frontend']['tracking']['checkpoint']),
                   bp['rgb_feature']['weights_path']:sha256_file(bp['rgb_feature']['weights_path'])})
    config={'manifest_sha256':sha256_file(out/'video_manifest.json'),'frozen_files':frozen,'groups':list(GROUPS),'seeds':list(SEEDS),
        'models':[{k:r[k] for k in ('group','condition','seed','checkpoint','checkpoint_sha256','expected_epoch')} for r in records],
        'frontend_protocol_sha256':sha256_file(base/'protocol.json'),'entry_sha256':sha256_file(__file__),
        'processor_sha256':sha256_file(runner.__file__),'threshold':0,'frames':'all native decode frames','block':64}
    identity=digest(config)
    if (out/'protocol.json').exists():
        if not args.resume or read_json(out/'protocol.json')['identity']!=identity:raise ValueError('RESUME_IDENTITY')
    else:save_json(out/'protocol.json',{**config,'identity':identity,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    signal.signal(signal.SIGTERM,lambda *_:setattr(runner,'STOP',True));started=time.monotonic();phase='load'
    save_json(out/'final_status.json',{'status':'RUNNING','pid':os.getpid(),'training_started':False,'started_unix':time.time()})
    try:
        models={};checks=[]
        sample=read_json(DERIVED/'v8_full_video_training_budget5_v1/input_manifest.json')['rows'][0]
        arrays,_=read_block(sample['blocks'][0]['path'],read_json(DERIVED/'v8_full_video_global_context_v1/protocol.json')['identity'],sample['video_sha256'])
        for r in records:
            cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            if cp['identity']!=r['identity'] or cp['epoch']!=r['expected_epoch']:raise ValueError('CHECKPOINT_IDENTITY')
            model=GlobalContextModel(r['condition'],stats).cuda().eval();model.load_state_dict(cp['model'],strict=True);model.requires_grad_(False)
            key=f'{r["group"]}__seed{r["seed"]}';stream=GlobalStreamingModel(model);state=None
            for a in arrays[:2]:state=stream.forward_step(tensors(a),state)['state']
            cpu={k:torch.from_numpy(np.stack([a[k] for a in arrays[:2]])) for k in arrays[0]}
            with torch.inference_mode(): other=VideoTraining(model).video(cpu,64,use_checkpoint=False)
            error=abs(float(other)-float(state['log_numerator']-state['total_area'].log()))
            if error>2e-5:raise ValueError('SHORT_LOADING_OUTPUT_CHECK')
            models[key]=stream;checks.append({'key':key,'epoch':cp['epoch'],'max_absolute_error':error})
        save_json(out/'loading_check.json',{'status':'PASS','frames':2,'checks':checks,'existing_tests_rerun':False})
        providers=runner.Providers(base,out,stats,group,identity);summaries=[];phase='continuous'
        for i,row in enumerate(rows):
            if sha256_file(row['video_path'])!=row['video_sha256']:raise ValueError('VIDEO_CHANGED')
            summaries.append(runner.process(row,out,DERIVED/'v8_full_video_streaming_pilot_v1',providers,models,identity,i,
                total_videos=len(rows),validate_first_block=False,model_timings=True))
        for path,sha in frozen.items():
            if sha256_file(path)!=sha:raise ValueError('FROZEN_ASSET_CHANGED')
        phase='evaluate_report';finish(out,rows,summaries,records,identity,started)
    except BaseException as e:
        save_json(out/'final_status.json',{'status':'STOPPED_SAFE' if isinstance(e,InterruptedError) else 'FAILED','phase':phase,
                  'error':str(e),'traceback':traceback.format_exc(),'training_started':False,'elapsed_this_invocation_s':time.monotonic()-started})
        raise


if __name__=='__main__':main()
