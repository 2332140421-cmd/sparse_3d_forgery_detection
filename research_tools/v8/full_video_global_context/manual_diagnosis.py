"""Join user-exported manual observations to saved video logits; no inference."""
import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from .manual_review import DERIVED, sha

GROUPS = ('COMPONENT_ADAPTED', 'COMPONENT_WINDOW', 'CELL_WINDOW')
SEEDS = ('20260909', '20260910', '20260911')
FIELDS = ('reviewed','indoor_scene','continuous_shot','camera_motion','subject_visibility',
          'observed_anomaly','anomaly_times','playback_problem','notes')


def join_reviews(records, mapping, frozen):
    by_id = {r['video_id']: r for r in mapping}
    if len(by_id) != len(mapping): raise ValueError('DUPLICATE_MAPPING_ID')
    reviewed = {}
    for r in records:
        v = r['video_id']; m = by_id.get(v)
        if v in reviewed: raise ValueError('DUPLICATE_REVIEW_IDENTITY')
        if not m or any(r[k] != m[k] for k in ('review_id','video_sha256','section')):
            raise ValueError('REVIEW_IDENTITY_MISMATCH')
        reviewed[v] = r
    if set(reviewed) != set(by_id): raise ValueError('MISSING_OR_EXTRA_REVIEW')
    result = []
    for f in frozen:
        m, r = by_id[f['video_id']], reviewed[f['video_id']]
        if m['section'] != 'test' or m['video_sha256'] != f['video_sha256']:
            raise ValueError('FROZEN_SHA_OR_SECTION_MISMATCH')
        result.append({**f, 'review_id': m['review_id'], **{k:r[k] for k in FIELDS}})
    return sorted(result, key=lambda r:r['review_id'])


def score_values(values):
    if set(values) != set(SEEDS) | {'MEAN_VIDEO_LOGIT'}: raise ValueError('SEED_SET_MISMATCH')
    z = [float(values[s]) for s in SEEDS]
    e = float(values['MEAN_VIDEO_LOGIT'])
    if not all(math.isfinite(v) for v in z+[e]): raise ValueError('NONFINITE_SCORE')
    if abs(sum(z)/3-e) > 1e-12: raise ValueError('ENSEMBLE_MISMATCH')
    return z,e


def explicit(r):
    return r['observed_anomaly'] in ('空间形态异常','时间运动异常','两者都有')


def suitable(r):
    return (explicit(r) and r['indoor_scene']=='室内' and r['continuous_shot']=='连续镜头'
            and r['camera_motion'] in ('基本稳定','轻微运动') and r['subject_visibility']=='主体清楚')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def responses(old, out, selected):
    """Read only the first two eligible misses, already saved frame responses."""
    import numpy as np
    table, descriptions = [], []
    for r in selected[:2]:
        directory = old/'responses'/r['video_id']; p = directory/'summary.json'
        if not p.is_file():
            descriptions.append(f'{r["review_id"]}：没有已保存summary，不补推理。'); continue
        summary = json.loads(p.read_text())
        if summary['video_sha256'] != r['video_sha256']: raise ValueError('RESPONSE_SHA_MISMATCH')
        models = summary['models']; ids = [models.index(f'COMPONENT_ADAPTED__seed{s}') for s in SEEDS]
        # These two selected records have simple explicit intervals; no guessed parsing.
        import re
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*', r['anomaly_times'])
        if not match:
            descriptions.append(f'{r["review_id"]}：时间文本无法无歧义解析，未猜测区间。'); continue
        lo,hi = map(float,match.groups()); times=[]; logits=[]
        for block in summary['blocks']:
            path = Path(block['response_path'])
            if sha(path) != block['response_sha256']: raise ValueError('RESPONSE_CONTENT_MISMATCH')
            with np.load(path,allow_pickle=False) as z:
                if 'frame_logits' not in z or 'pts_s' not in z:
                    descriptions.append(f'{r["review_id"]}：没有可用逐帧响应。'); break
                if z['frame_logits'].shape != (len(z['pts_s']),len(models)) or len(z['frame_indices']) != len(z['pts_s']):
                    raise ValueError('RESPONSE_FRAME_SHAPE_MISMATCH')
                for fi,t,values in zip(z['frame_indices'],z['pts_s'],z['frame_logits'][:,ids]):
                    if not np.isfinite(values).all(): raise ValueError('NONFINITE_FRAME_RESPONSE')
                    times.append(float(t)); logits.append(float(values.mean()))
                    table.append({'review_id':r['review_id'],'video_id':r['video_id'],'source_frame':int(fi),'pts_s':float(t),
                                  'within_user_interval':lo<=t<=hi,
                                  **{f'ADAPTED_seed{s}_frame_logit':float(v) for s,v in zip(SEEDS,values)},
                                  'mean_seed_frame_logit':float(values.mean())})
        if times:
            inside = [v for t,v in zip(times,logits) if lo<=t<=hi]
            outside = [v for t,v in zip(times,logits) if not lo<=t<=hi]
            descriptions.append(f'{r["review_id"]}：保存PTS {min(times):g}–{max(times):g}s，用户区间{lo:g}–{hi:g}s；'
                f'区间内{len(inside)}帧、区间外{len(outside)}帧；集成逐帧响应范围[{min(logits):.4f},{max(logits):.4f}]，'
                f'首/末{logits[0]:.4f}/{logits[-1]:.4f}。'
                + ('没有区间外对照，无法判断“异常时段相对非异常时段上升”。' if not outside else '仅描述已有响应，不证明因果或定位。'))
    if table: write_csv(out/'selected_frame_responses.csv', table)
    return descriptions


def run(review, bundle, old, out):
    out.mkdir(parents=True,exist_ok=True)
    mapping = json.loads((bundle/'review_manifest.json').read_text())
    if review.suffix.lower()=='.csv':
        records = list(csv.DictReader(review.open(encoding='utf-8-sig')))
    else:
        raw = json.loads(review.read_text(encoding='utf-8-sig'))
        if raw['package_id'] != mapping['package_id']: raise ValueError('PACKAGE_ID_MISMATCH')
        records = raw['records']
    frozen = json.loads((old/'video_manifest.json').read_text())['rows']
    rows = join_reviews(records,mapping['rows'],frozen)
    if len(rows)!=40: raise ValueError('TEST_COUNT_MISMATCH')
    scores = list(csv.DictReader((old/'evaluation/video_scores.csv').open()))
    if len(scores)!=480: raise ValueError('SCORE_COUNT_MISMATCH')
    by_key={}
    frozen_ids={r['video_id'] for r in rows}
    for s in scores:
        k=(s['video_id'],s['group'],s['seed'])
        if k in by_key or s['video_id'] not in frozen_ids: raise ValueError('SCORE_IDENTITY_DUPLICATE_OR_UNKNOWN')
        by_key[k]=s
    output=[]; confusion={g:Counter() for g in GROUPS}
    for r in rows:
        summary=json.loads((old/'responses'/r['video_id']/'summary.json').read_text())
        if summary['video_sha256']!=r['video_sha256']: raise ValueError('SAVED_RESPONSE_VIDEO_SHA_MISMATCH')
        item={k:r[k] for k in ('review_id','video_id','video_sha256','action_id','role','generator',*FIELDS)}
        for g in GROUPS:
            vv={}
            for s in (*SEEDS,'MEAN_VIDEO_LOGIT'):
                record=by_key[(r['video_id'],g,s)]
                if any(str(record[k])!=str(r[k]) for k in ('role','generator','action_id')) or int(record['label'])!=r['video_label']:
                    raise ValueError('SAVED_SCORE_LABEL_IDENTITY_MISMATCH')
                vv[s]=record['logit']
            z,e=score_values(vv)
            item.update({f'{g}_seed{s}_logit':v for s,v in zip(SEEDS,z)})
            category=('TP' if e>=0 else 'FN') if r['role']=='fake' else ('FP' if e>=0 else 'TN')
            item.update({f'{g}_ensemble_logit':e,f'{g}_prediction':int(e>=0),f'{g}_category':category})
            confusion[g][category]+=1
        item['ADAPTED_seed_direction']='混合正负' if min(z:= [item[f'COMPONENT_ADAPTED_seed{s}_logit'] for s in SEEDS])<0<=max(z) else ('全部非负' if min(z)>=0 else '全部负')
        output.append(item)
    original=json.loads((old/'evaluation/summary.json').read_text())
    for g in GROUPS:
        m=next(m for m in original['metrics'] if m['group']==g and m['seed']=='MEAN_VIDEO_LOGIT')
        assert all(confusion[g][c]==m[c.lower()] for c in ('TN','FP','FN','TP'))
    write_csv(out/'review_predictions.csv',output)
    shutil.copyfile(review,out/('manual_review_input.csv' if review.suffix.lower()=='.csv' else 'manual_review_input.json'))
    anomalies=[r for r in output if explicit(r)];fake=[r for r in anomalies if r['role']=='fake'];good=[r for r in fake if suitable(r)]
    misses=[r for r in fake if r['COMPONENT_ADAPTED_category']=='FN'];good_misses=[r for r in good if r['COMPONENT_ADAPTED_category']=='FN']
    ids=lambda rr:', '.join(r['review_id'] for r in rr) or '无'
    expected={'V028','V019','V035','V002','V021','V012','V024','V007','V030'}
    lines=['# DeepAction人工审核与已存预测诊断','',f'完整保留40测试视频，训练比较4条不纳入。审核来源：{review}；文件SHA {sha(review)}。',
        '按video_id连接并校验审核/观看包/冻结manifest/响应summary的SHA；review_id仅展示，标签/生成器取冻结manifest。',
        '没有训练、前端、模型前向、重评分、阈值调整或替换样本；阈值0，逐seed最终video logit均值复现，三组原混淆矩阵复现通过。',
        '',f'明确异常{len(anomalies)}条，官方fake{len(fake)}条；ADAPTED检出{len(fake)-len(misses)}、漏检{len(misses)}。',
        f'条件合适且明确异常的官方fake {len(good)}条：{ids(good)}；与用户所列9条一致：{set(r["review_id"] for r in good)==expected}。其中检出{len(good)-len(good_misses)}、漏检{len(good_misses)}。',
        f'这些条件合适漏检：{ids(good_misses)}。不能全部归因于超出场景约束。','',
        '|编号|官方生成器|ADAPTED|WINDOW|CELL|ADAPTED三个seed logit|seed方向|', '|---|---|---:|---:|---:|---|---|']
    for r in good:
        lines.append(f'|{r["review_id"]}|{r["generator"]}|'+ '|'.join(f'{r[g+"_ensemble_logit"]:.4f}' for g in GROUPS)+'|'+', '.join(f'{r[f"COMPONENT_ADAPTED_seed{s}_logit"]:.4f}' for s in SEEDS)+f'|{r["ADAPTED_seed_direction"]}|')
    lines+=['','## 漏检分布（计数而非因果）']
    for field in ('generator','camera_motion','subject_visibility'):
        for value in sorted({r[field] for r in fake}):
            total=[r for r in fake if r[field]==value];mm=[r for r in misses if r[field]==value]
            lines.append(f'- {field}={value}：明确异常fake {len(total)}，漏检{len(mm)}；{ids(mm)}。')
    lines+=['','## 其他模型是否检出这些漏检','|ADAPTED漏检编号|WINDOW类别|CELL类别|','|---|---|---|']
    lines += [f'|{r["review_id"]}|{r["COMPONENT_WINDOW_category"]}|{r["CELL_WINDOW_category"]}|' for r in misses]
    lines += ['',f'WINDOW检出其中{sum(r["COMPONENT_WINDOW_category"]=="TP" for r in misses)}条，CELL检出其中{sum(r["CELL_WINDOW_category"]=="TP" for r in misses)}条。',
              f'9条条件合适异常中，seed混合正负：{ids([r for r in good if r["ADAPTED_seed_direction"]=="混合正负"])}。全负也不代表模型一致正确，而是三个seed共同漏检。',
              '', '## 已保存响应的最小检查',*responses(old,out,good_misses),
              '逐帧三seed均值只用于观察既有响应，不重新进行视频LSE或覆盖保存的视频分数。没有空间bbox，不断言响应落在手或物体；未额外提帧或生成热图。',
              '', '## 解释边界',
              '场景超约束可以与漏检同时存在，但条件合适的明确异常也有漏检；现有证据不能把低召回唯一归于数据。',
              '训练的局部嵌回与本轮生成方式不同属于任务分布变化，不证明它是唯一原因。V019“不是全生成式”等notes原文保留，只是观感，不覆盖官方CogVideoX5B身份。',
              '逆光/泛光仅为用户补充的整体观察，没有逐视频字段，不能统计或关联漏检。',
              '明确异常是人工描述，不是可靠帧/像素真值。未观察异常不等于真实；不计算筛选后的AUROC、不将9条变成新正式测试集。',
              '模型在条件合适异常上漏检且部分seed方向分歧，是现有能力局限的证据；无法仅凭本次连接定位编码、聚合、训练分布或前端的具体责任。保留原40视频正式结果。']
    (out/'diagnosis.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=DERIVED/'v8_deepaction_manual_review_v1')
    p.add_argument('--experiment',type=Path,default=DERIVED/'v8_deepaction_cross_dataset_v1')
    p.add_argument('--output',type=Path,default=DERIVED/'v8_deepaction_manual_diagnosis_v1')
    a=p.parse_args();run(a.review,a.bundle,a.experiment,a.output)
