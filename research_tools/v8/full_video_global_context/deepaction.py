"""Small official DeepAction acquisition and action-cluster frozen evaluation."""
import argparse
import csv
import hashlib
import io
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import requests
from PIL import Image, ImageDraw

from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file
from research_tools.v8.full_video_streaming.inputs import metadata
from research_tools.v8.local_geometry_pilot.data import digest
from research_tools.v8.local_geometry_pilot.runner import csv_rows
from .independent_test import ROOT, GROUPS, SEEDS, metrics, evaluate_scores, processing_costs

REPO='faridlab/deepaction_v1'
REVISION='476932c3369424ca59dc1f8fb758cb3c13fe63fe'
DATA=ROOT/'datasets/deepaction_v1'
# Frozen by action semantics before viewing any media or model score.
INDOOR_ACTIONS=(3,8,9,12,13,16,17,18,19,20,21,22,23,25,26,27,28,29,30,32,34,35,38,39,42,47,48,49,50,51,52,54,55,56,57,58,59,62,64,67,68,69,70,71,72,73,75,76,78,79,80,82)


def ordered_actions():
    return sorted(INDOOR_ACTIONS,key=lambda i:hashlib.sha256(f'20260909:{i}'.encode()).hexdigest())


def initialize(out):
    out.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (out/'selection_protocol.json').exists():return
    url=f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true'
    r=requests.get(url,timeout=45);r.raise_for_status();info=r.json()
    if info['sha']!=REVISION:raise ValueError('OFFICIAL_REVISION')
    save_json(out/'official_file_list.json',info)
    for name in ('README.md','captions.csv'):
        r=requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}',timeout=45);r.raise_for_status()
        (out/name).write_bytes(r.content)
    save_json(out/'selection_protocol.json',{'repo':REPO,'revision':REVISION,'semantic_pool':list(INDOOR_ACTIONS),
        'order':ordered_actions(),'seed':20260909,'target_actions':20,'minimum_actions':10,'frame_budget':30000,
        'scene_rule':'real only: indoor everyday activity, continuous shot, fixed/light camera, no large cuts; uncertain excluded',
        'fake_rule':'accepted action rank alternates RunwayML/CogVideoX5B; lexicographic official file',
        'budget_rule':'largest fixed ordered accepted prefix within 30000 metadata frames; never reorder by length',
        'cluster':'action_id is semantic only, NOT common original-video lineage',
        'created_unix':time.time(),'official_api':url,'paper':'https://arxiv.org/abs/2412.00526'})
    save_json(out/'scene_review.json',{})


def official_file(out,generator,action):
    files=read_json(out/'official_file_list.json')['siblings']
    names=sorted(f['rfilename'] for f in files if f['rfilename'].startswith(f'{generator}/{action}/') and f['rfilename'].endswith('.mp4'))
    if not names:raise ValueError(f'OFFICIAL_MEDIA_MISSING:{generator}/{action}')
    return next(f for f in files if f['rfilename']==names[0])


def download(out,item):
    name=item['rfilename'];path=DATA/'media'/name
    if path.is_file():
        if sha256_file(path)!=item['lfs']['sha256']:raise ValueError('LOCAL_MEDIA_HASH')
        return path
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.download.part');started=time.monotonic()
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}?download=true'
    offset=tmp.stat().st_size if tmp.exists() else 0
    try:
        with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as r:
            r.raise_for_status()
            resumed=offset>0 and r.status_code==206
            if resumed and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('DOWNLOAD_RANGE_IDENTITY')
            with tmp.open('ab' if resumed else 'wb') as f:
                for block in r.iter_content(1024*1024):f.write(block)
        if tmp.stat().st_size!=item['size'] or sha256_file(tmp)!=item['lfs']['sha256']:raise ValueError('OFFICIAL_DOWNLOAD_HASH_SIZE')
        tmp.replace(path)
        record={'official_path':name,'url':url,'sha256':item['lfs']['sha256'],'bytes':path.stat().st_size,'elapsed_s':time.monotonic()-started,
                'resumed_prefix_bytes':offset if resumed else 0}
        # One record per file avoids parallel writers and repeated cost on resume.
        save_json(path.with_suffix('.acquisition.json'),record)
        print(f'DOWNLOADED {name} bytes={record["bytes"]}',flush=True)
        return path
    except BaseException as e:
        save_json(path.with_suffix(f'.failure.{time.time_ns()}.json'),{'path':name,'elapsed_s':time.monotonic()-started,'status':'DOWNLOAD_FAILED','error':f'{type(e).__name__}: {e}'})
        raise


def contact_sheet(out,action,path):
    dest=out/'scene_sheets'/f'{action}.jpg';dest.parent.mkdir(exist_ok=True)
    if dest.exists():return
    frames={};count=metadata(path)['metadata_frame_count']
    indices=np.linspace(0,count-1,12).round().astype(int);wanted=set(indices.tolist())
    with av.open(str(path)) as c:
        c.streams.video[0].thread_type='AUTO';c.streams.video[0].codec_context.thread_count=2
        for i,f in enumerate(c.decode(video=0)):
            if i in wanted:frames[i]=f.to_image().resize((320,180))
    if not frames:raise ValueError('EMPTY_REAL_VIDEO')
    if not wanted<=set(frames):raise ValueError('SCENE_REVIEW_METADATA_EOF')
    sheet=Image.new('RGB',(1280,636),'white');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(indices):
        x=(j%4)*320;y=(j//4)*212;sheet.paste(frames[i],(x,y));draw.text((x+4,y+181),f'action={action} frame={i}',fill='black')
    sheet.save(dest,quality=90)


def review_real(out,count):
    reviews=read_json(out/'scene_review.json');pending=[a for a in ordered_actions() if str(a) not in reviews][:count]
    items=[official_file(out,'Pexels',a) for a in pending]
    with ThreadPoolExecutor(max_workers=3) as pool:paths=list(pool.map(lambda i:download(out,i),items))
    for a,path in zip(pending,paths):contact_sheet(out,a,path)
    print('REVIEW_PENDING',pending,flush=True)


def prefix_within_budget(pairs,budget=30000):
    result=[];total=0
    for pair in pairs:
        n=sum(r['metadata_frame_count'] for r in pair)
        if total+n>budget:break
        result.append(pair);total+=n
    return result,total


def validate_manifest(rows):
    if len({r['source_id'] for r in rows})!=len(rows) or len({r['video_id'] for r in rows})!=len(rows):raise ValueError('MEDIA_IDENTITY_NOT_UNIQUE')
    for a in {r['action_id'] for r in rows}:
        pair=[r for r in rows if r['action_id']==a]
        if len(pair)!=2 or {r['role'] for r in pair}!={'real','fake'}:raise ValueError('ACTION_SEMANTIC_PAIR')
        for r in pair:
            expected=r['generator']=='Pexels'
            if (r['role']=='real')!=expected or r['video_label']!=int(not expected):raise ValueError('OFFICIAL_LABEL_IDENTITY')
            if r['generator'] not in ('Pexels','RunwayML','CogVideoX5B') or not r['official_path'].startswith(f'{r["generator"]}/{a}/') or r['official_revision']!=REVISION:raise ValueError('OFFICIAL_MEDIA_IDENTITY')


def freeze(out):
    if (out/'video_manifest.json').exists():raise ValueError('MANIFEST_ALREADY_FROZEN')
    reviews=read_json(out/'scene_review.json');accepted=[]
    for a in ordered_actions():
        r=reviews.get(str(a))
        if r is None:raise ValueError(f'EARLIER_ACTION_UNREVIEWED:{a}')
        if r['status']=='ACCEPT':accepted.append(a)
        elif r['status'] not in ('REJECT','UNCERTAIN'):raise ValueError('SCENE_STATUS')
        if len(accepted)==20:break
    captions={int(r['id']):r['prompt'] for r in csv.DictReader(io.StringIO((out/'captions.csv').read_text()))}
    pairs=[]
    for rank,a in enumerate(accepted):
        generator=('RunwayML','CogVideoX5B')[rank%2];pair=[]
        for g in ('Pexels',generator):
            item=official_file(out,g,a);path=download(out,item);sha=sha256_file(path);role='real' if g=='Pexels' else 'fake'
            row={'action_id':str(a),'prompt':captions[a],'generator':g,'role':role,'video_label':int(role=='fake'),
                 'source_id':f'DeepAction::{g}::{a}::{Path(item["rfilename"]).stem}',
                 'video_id':digest({'dataset':REPO,'revision':REVISION,'file':item['rfilename'],'sha256':sha})[:24],
                 'video_path':str(path),'video_sha256':sha,'video_bytes':path.stat().st_size,'official_path':item['rfilename'],
                 'official_revision':REVISION,'split':'test','label_source':'official Pexels real / official generator folder fake',
                 'lineage':'independent media identity; action semantics do not imply common original',**metadata(path)}
            pair.append(row)
        pairs.append(pair)
    selected,total=prefix_within_budget(pairs)
    if len(selected)<10:raise ValueError(f'BUDGET_INSUFFICIENT:{len(selected)} actions')
    rows=[r for p in selected for r in p]
    validate_manifest(rows)
    save_json(out/'budget_decision.json',{'target_actions':20,'acquired_actions':len(pairs),'selected_actions':len(selected),
        'all_acquired_frames':sum(r['metadata_frame_count'] for p in pairs for r in p),'selected_frames':total,'budget':30000,
        'selected_prefix':[p[0]['action_id'] for p in selected],'rule':'fixed semantic+scene ordering prefix, no length sorting, no frame truncation'})
    save_json(out/'video_manifest.json',{'rows':rows,'frozen_before_scoring_unix':time.time(),'dataset':REPO,'revision':REVISION,
        'scene_review_sha256':sha256_file(out/'scene_review.json'),'cluster_key':'action_id'})
    print(f'FROZEN actions={len(selected)} videos={len(rows)} frames={total}',flush=True)


def finish(out,rows,summaries,records,identity,started):
    ts=time.monotonic();result=evaluate_scores(out,rows,summaries,cluster_key='action_id');evaluation_s=time.monotonic()-ts
    costs=processing_costs(out,summaries,evaluation_s)
    downloads=[read_json(p) for p in (DATA/'media').rglob('*.acquisition.json')]
    failed=[read_json(p) for p in (DATA/'media').rglob('*.failure*.json')]
    costs.update(download_s_known_lower_bound=sum(r['elapsed_s'] for r in downloads+failed),
                 verified_media_bytes=sum(r['bytes'] for r in downloads),acquired_media_files=len(downloads),
                 download_cost_complete=False,
                 download_timing='sum recorded per-file attempts, parallel transfers overlap; first Pexels/17 failure timing was overwritten before unique failure logging, total wall/traffic UNKNOWN',
                 reason='only real scene-screen candidates and selected fake media; no whole-dataset download')
    save_json(out/'costs.json',costs)
    descriptive=[]
    for role in ('real','fake'):
        rr=[r for r in rows if r['role']==role]
        for field in ('fps','duration_s','metadata_frame_count'):
            values=[r[field] for r in rr];descriptive.append({'role':role,'field':field,'n':len(values),'min':min(values),'median':float(np.median(values)),'max':max(values)})
    csv_rows(out/'evaluation/media_summary.csv',descriptive)
    generator_metrics=[]
    for generator in ('RunwayML','CogVideoX5B'):
        actions={r['action_id'] for r in rows if r['generator']==generator}
        ii=[i for i,r in enumerate(rows) if r['action_id'] in actions]
        for g in GROUPS:
            z=np.array([np.mean([summaries[i]['video_logits'][f'{g}__seed{s}'] for s in SEEDS]) for i in ii])
            generator_metrics.append({'generator':generator,'group':g,'actions':len(actions),'real':len(actions),'fake':len(actions),
                'scope':'descriptive; real restricted to same action IDs, semantic matching only',**metrics(np.array([rows[i]['video_label'] for i in ii]),z)})
    csv_rows(out/'evaluation/per_generator_descriptive.csv',generator_metrics)
    lines=['# V8 DeepAction跨数据集冻结测试','',f'COMPLETE：{len(rows)//2}动作类别、{len(rows)}视频、{costs["decoded_frames"]}实际帧，九份冻结权重；没有训练。',
        f'官方revision：{REVISION}。出处：https://huggingface.co/datasets/{REPO}；论文：https://arxiv.org/abs/2412.00526。',
        '这是受限人体日常动作场景下的跨数据集及生成方式联合迁移，不代表完整DeepAction。Pexels为real，RunwayML/CogVideoX5B为fake，标签依据官方目录而非观感。',
        'action ID仅动作语义匹配，不宣称同源派生。CI为action-cluster bootstrap（10000次、seed20260909），同次类别采样保留所选real/fake并用于全部模型组。',
        '仅查看real作场景准入；选择顺序、生成器交替分配及预算规则在评分前冻结。fake毛刺/几何缺失不作为筛除依据。',
        '标准化/grouping/权重冻结；全原生帧、真实PTS、过去上下文、hidden跨块，全帧attention+GRU，面积LSE；各seed整视频LSE后平均video logit，阈值0。',
        '', '|组|seed|AUROC|AP|FPR|Recall|Balanced ACC|P|F1|ACC|TN/FP/FN/TP|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for m in result['metrics']:
        lines.append(f'|{m["group"]}|{m["seed"]}|'+ '|'.join(f'{m[k]:.6f}' for k in ('auroc','ap','fpr','recall','balanced_accuracy','precision','f1','accuracy'))+f'|{m["tn"]}/{m["fp"]}/{m["fn"]}/{m["tp"]}|')
    import json
    lines+=['','## CI与配对',json.dumps(result['auroc_intervals']),json.dumps(result['paired_auroc']),
        'ADAPTED−WINDOW是适配影响；COMPONENT_WINDOW−CELL_WINDOW是相同训练阶段组件对照；不把ADAPTED−CELL当匹配消融。',
        '', '## 媒体差异及成本',json.dumps(descriptive),json.dumps(generator_metrics),json.dumps(costs),
        '逐视频FPS/时长/分辨率/帧数/生成器见video_manifest.json，分辨率与生成器汇总见evaluation/media_details.csv；不用于测试后调参。',
        '生成方法少量样本仅描述，不声称整个生成器稳定；不与不同子集/训练协议的论文数字直接比。不计算无真值的帧/像素定位。',
        '下载成本为已记录逐文件尝试的耗时下界（并行有重叠），不是精确总下载墙钟；早期Pexels/17一次失败记录被覆盖，额外失败流量及该次耗时UNKNOWN。成功文件大小之和精确，失败重传不伪装成0成本。',
        '完整结果保留；无论下降与否均不换样本、调阈值或追加训练。当前架构与适配判断在验收后补充。']
    csv_rows(out/'evaluation/media_details.csv',[{k:r[k] for k in ('action_id','source_id','role','generator','fps','duration_s','source_hw','metadata_frame_count')} for r in rows])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'final_status.json',{'status':'COMPLETE','actions':len(rows)//2,'videos':len(rows),'decoded_frames':costs['decoded_frames'],
        'weights':9,'training_started':False,'identity':identity,'elapsed_this_invocation_s':time.monotonic()-started})
    save_json(out/'progress.json',{'status':'COMPLETE','stage':'report','completed':len(rows),'total':len(rows),'updated_unix':time.time()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('review','freeze'),required=True);p.add_argument('--count',type=int,default=20);args=p.parse_args()
    initialize(args.output)
    if args.stage=='review':review_real(args.output,args.count)
    else:freeze(args.output)


if __name__=='__main__':main()
