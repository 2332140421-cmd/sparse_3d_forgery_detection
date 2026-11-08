"""CPU input coverage/mixing and local raw-edit inventory; no model execution.

CLI -> existing Video.frame/slab_ids -> CSVs, six coverage figures and report.
Scene previews lack actual pool indices/counts, so a small statistics entry is
needed; no encoder, SlabPool or detector is instantiated or called.
"""
from collections import Counter
import csv
import html
import inspect
import io
import json
import os
from pathlib import Path
import random
import re
import time
import zipfile

import cv2
import numpy as np
import torch

from research_tools.rgbd_slice.data import Video
from research_tools.depth_slice.model import slab_ids
from research_tools.rgbd_slice.model import SlabPool
from scripts.prepare_v8_assets import digest,identities,real_path
from scripts.review_anakin import numeric_members,alignment

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection')
SOURCE=BASE/'derived/pvst_rgbd_slice_relations_v1'
FIT=BASE/'derived/pvst_set_to_set_fit_check_v1'
SCENE=BASE/'derived/pvst_sample_scene_review_v1'
OUT=BASE/'derived/pvst_edit_region_coverage_v1'
PAIRS=['an:7ogg04','an:728qx7']
REPO=Path(__file__).resolve().parents[1]
# Semantic primary classes for the eight actual descriptions, not keyword rules.
AN_CLASSES={
 'an:7ogg04':('B','移除一个人物；未描述改变保留人物的动作/姿态，不排除运动/遮挡关系'),
 'an:95qvg9':('B','移除背景行人，是主体存在的改变，不是替换整片背景'),
 'an:4wdo37':('B','移除奔跑持球者，动作描述指被移除主体，不等于修改其动作'),
 'an:40y1x9':('B','移除奔跑的狗；不是将狗的动作或形态改成另一种'),
 'an:728qx7':('C','在胸前添加纹身图案；无明确姿态/肢体形态修改描述'),
 'an:7e2557':('C','衬衫口袋胸针，主类为配饰外观；仍可能涉及局部对象/遮挡'),
 'an:7p8r14':('C','添加帽子，主类为配饰外观；仍可能涉及对象形态/遮挡'),
 'an:4z1827':('待确认','帽上蜘蛛可能是实体、装饰或图案，描述不足以区分B/C，不能直接认定动作改变')}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            w.writerows({k:('null' if v is None else v) for k,v in row.items()} for row in rows)


def counts(frame):
    """Native point-observation counts only; learned pooling is not evaluated."""
    g=frame['gt'];v=frame['valid'];q=frame['xyz']
    if g.dtype!=torch.bool or v.dtype!=torch.bool or g.shape!=v.shape or q.shape!=(*g.shape,3) or frame['rgb'].shape!=(*g.shape,3):raise ValueError('NATIVE_GT_VALID_RGB_SHAPE')
    if not torch.isfinite(q[v]).all():raise ValueError('NONFINITE_SELECTED_XYZ')
    points=torch.where(v.reshape(-1))[0]  # Exact current SlabPool selection.
    if not np.array_equal(points.numpy(),np.flatnonzero(v.numpy().reshape(-1))):raise ValueError('POOL_INDEX_MISMATCH')
    selected=torch.zeros(g.numel(),dtype=torch.bool);selected[points]=True
    if not torch.equal(selected.reshape_as(v),v):raise ValueError('POOL_SELECTION_NOT_VALID')
    ids,index=slab_ids(q,v);edited=g.reshape(-1)[points]
    total=torch.bincount(index,minlength=len(ids));positive=torch.bincount(index[edited],minlength=len(ids))
    if int(total.sum())!=len(points) or int(positive.sum())!=int((g&v).sum()):raise ValueError('SLICE_COUNT_CONSERVATION')
    slices=[{'absolute_slice_id':int(ids[i]),'edited_valid_pixels':int(positive[i]),'all_valid_pixels':int(total[i]),
             'nonedited_valid_pixels':int(total[i]-positive[i]),'edited_fraction':float(positive[i]/total[i])} for i in range(len(ids)) if positive[i]>0]
    gt=int(g.sum());inside=int(edited.sum())
    return {'GT_pixels':gt,'GT_selected_pixels':inside,'GT_excluded_pixels':gt-inside,
            'GT_pool_coverage':inside/gt if gt else None,'frame_valid_fraction':float(v.float().mean()),
            'selected_pixel_count':len(points),'occupied_slices':len(ids),'edited_slices':len(slices)},slices


def exclusion_path(video,frame,i):
    raw=np.array(video.valid[i]);q=np.array(video.xyz[i]);h,w=frame['valid'].shape
    gt=frame['gt'].numpy();actual=frame['valid'].numpy()
    if raw.shape==(h,w):
        if not np.array_equal(actual,raw):raise ValueError('NATIVE_VALID_RULE_CHANGED')
        return {'excluded_original_native_valid':int((gt&~actual).sum()),'excluded_resample_no_support':0,
                'excluded_resample_mixed_support':0,'excluded_resample_nonfinite':0}
    weight=cv2.resize(raw.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    native=cv2.resize(np.where(raw[...,None],q,0),(w,h),interpolation=cv2.INTER_LINEAR)
    finite=np.isfinite(native).all(-1);expected=(weight>=1-1e-6)&finite
    if not np.array_equal(actual,expected):raise ValueError('UPSAMPLE_VALID_RULE_CHANGED')
    parts={'excluded_original_native_valid':0,'excluded_resample_no_support':int((gt&(weight==0)).sum()),
           'excluded_resample_mixed_support':int((gt&(weight>0)&(weight<1-1e-6)).sum()),
           'excluded_resample_nonfinite':int((gt&(weight>=1-1e-6)&~finite).sum())}
    if sum(parts.values())!=int((gt&~actual).sum()):raise ValueError('EXCLUSION_PARTITION')
    return parts


def figures(frame,pair,i):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    valid=(frame['valid'].numpy()*255).astype(np.uint8)
    cv2.imwrite(str(folder/f'{pair[3:]}_{i}_valid.png'),valid)
    missing=(frame['gt']&~frame['valid']).numpy()
    prior=SCENE/'media'/pair.replace(':','_')
    if not (prior/f'fake_{i:03d}.png').exists() or not (prior/f'GT_{i:03d}.png').exists():raise FileNotFoundError('EXISTING_SCENE_PREVIEW')
    # RGB and GT remain references to the last review, not regenerated images.
    if missing.any():
        rgb=frame['rgb'].numpy().copy();rgb[missing]=np.round(.5*rgb[missing]+np.array([127.5,0,0])).astype(np.uint8)
        cv2.imwrite(str(folder/f'{pair[3:]}_{i}_excluded.png'),rgb[:,:,::-1])


def inventory(deadline):
    manifest=json.loads((SOURCE/'manifest.json').read_text());train=[r for r in manifest['rows'] if r['split']=='train']
    allowed={r['pair_id'][3:] for r in train}
    meta={r['video-id']:r for r in csv.DictReader((BASE/'secondary_datasets/anakin/references/metadata.csv').open()) if r['video-id'] in allowed}
    notes=json.loads((SCENE/'visual_notes.json').read_text());rows=[]
    for r in train:
        if r['role']!='fake':continue
        original=next(x for x in train if x['pair_id']==r['pair_id'] and x['role']=='real')
        official=meta[r['pair_id'][3:]]
        if ' '.join(official['task'].split())!=' '.join(r['task'].split()):raise ValueError('OFFICIAL_DESCRIPTION_MISMATCH')
        with zipfile.ZipFile(r['mask_zip']) as z:members=numeric_members(z.namelist())
        if digest(Path(r['mask_zip']))!=r['mask_sha256'] or alignment([n for n,_ in members],r['actual_frames'])!={'status':'aligned','index_base':r['index_base']}:raise ValueError('INVENTORY_MASK_REGISTRATION')
        category,basis=AN_CLASSES.get(r['pair_id'],('待确认','未逐项审阅的新描述，不按关键词自动分类'))
        rows.append({'dataset':'ANAKIN','pair_id':r['pair_id'],'source_id':r['source_id'],'video_id':r['video_id'],
           'category':category,'basis':basis,'official_description':official['task'],'original_activity_description':'',
           'split_status':'current RGBD train','fake_path':r['video_path'],'real_path':original['video_path'],
           'fake_exists':Path(r['video_path']).is_file(),'real_exists':Path(original['video_path']).is_file(),
           'temporal_labels':'frame edit presence from aligned mask; source start/end are long-video crop positions',
           'temporal_annotation':'','pixel_GT_status':'ZIP SHA and numeric coverage match registration; only current2 redecoded here',
           'mask_path':r['mask_zip'],'scene':notes.get(r['pair_id'],{}).get('camera_preliminary','待确认：无本轮认可的镜头资料'),
           'existing_preview':str(SCENE/'preview/index.html') if r['pair_id'] in notes else '',
           'current_training_approved':True})
    # Only source/split identity fields from registered protocols; never features.
    registry=[BASE/'derived/pvst_point_voxel_v02/manifest.json',REPO/'docs/v8/evidence/temporal_detection_core_manifest.json']
    official=identities(BASE);blocked={r['source_id'] for r in official if r['official_split']=='test'}
    registered_train=set()
    for path in registry:
        d=json.loads(path.read_text())
        blocked.update(r['source_id'] for r in d['rows'] if r['dataset']=='ActivityForensics' and r['split'] in ['dev','test','test_reserved'])
        registered_train.update(r['source_id'] for r in d['rows'] if r['dataset']=='ActivityForensics' and r['split']=='train')
    with zipfile.ZipFile(BASE/'metadata/Charades.zip') as z:
        original_desc={}
        for name in ['Charades/Charades_v1_train.csv','Charades/Charades_v1_test.csv']:
            original_desc.update({r['id']:r['descriptions'] for r in csv.DictReader(io.StringIO(z.read(name).decode()))})
    published=json.loads((BASE/'asset_preparation/official_files.json').read_text())
    nonmedia=[r['path'] for r in published['files'] if not r['path'].endswith('.mp4')]
    with zipfile.ZipFile(BASE/'metadata/annot.zip') as z:annotation_members=z.namelist()
    pixel_files=[p for p in nonmedia if any(w in p.lower() for w in ['mask','segmentation','pixel'])]
    if pixel_files:raise ValueError('NEW_AF_SPATIAL_ASSETS_REQUIRE_SEPARATE_VERIFICATION')
    afstats=Counter();afsources=set()
    for r in official:
        if time.monotonic()>deadline:
            table(OUT/'dataset_edit_inventory.csv',rows)
            raise TimeoutError('600_SECOND_CPU_BUDGET_INVENTORY')
        if r['official_split']!='train':continue
        operations=sorted(set(re.findall(r'_(add|delete)@',r['media_path'])))
        kind='+'.join(operations) or 'unparsed';afstats[kind]+=1;afsources.add(r['source_id'])
        src=r['source_id'].split(':',1)[1];real=real_path(BASE,r['scene_source'],src);fake=BASE/'official'/published['revision']/r['media_path']
        rows.append({'dataset':'ActivityForensics','pair_id':'af:'+r['source_id']+':'+r['video_id'],'source_id':r['source_id'],'video_id':r['video_id'],
           'category':'待确认','basis':'official activity add/delete tags; exact local action/pose/form and spatial extent not specified, cannot equate to confirmed classA',
           'official_description':'official activity operations: '+kind,'original_activity_description':original_desc.get(src,'') if r['scene_source']=='charades' else '',
           'split_status':('excluded: registered dev/test or official-test source overlap' if r['source_id'] in blocked else
                           '参考协议已登记train且无已查非训练冲突；当前RGBD划分待确认、未批准' if r['source_id'] in registered_train else
                           'official train；研究划分待确认（未在参考训练清单登记），不能立即训练'),
           'fake_path':str(fake),'real_path':str(real),'fake_exists':fake.is_file(),'real_exists':real.is_file(),
           'temporal_labels':'official edited-activity seconds; exact endpoint convention unspecified',
           'temporal_annotation':json.dumps(r['intervals_s']),'pixel_GT_status':'no spatial asset in verified official release listing/annotation ZIP; temporal only',
           'mask_path':'','scene':'待确认：本轮不读AF媒体；原片动作描述不证明固定机位',
           'existing_preview':'','current_training_approved':False})
    evidence={'registered_source_split_paths':[str(p) for p in registry],'registered_source_split_sha256':[digest(p) for p in registry],
       'official_test_sources_excluded':len({r['source_id'] for r in official if r['official_split']=='test'}),
       'all_nontrain_sources_excluded':len(blocked),'AF_train_edit_rows':sum(afstats.values()),'AF_train_source_count':len(afsources),
       'AF_operation_counts':dict(afstats),'AF_published_nonmedia_files':nonmedia,'AF_annotation_members':annotation_members,
       'scope':'only metadata and path existence; no AF media decoded; no V8 models/features/statistics read; source exclusions only from listed current-branch registries, no all-history nonoverlap claim'}
    save(OUT/'inventory_evidence.json',evidence)
    # Prioritize B edits with known relatively steady view, then unknown camera;
    # known moving scenes are retained in inventory, not selected by model score.
    an=[r for r in rows if r['dataset']=='ANAKIN' and r['category']=='B']
    picked=[];rng=random.Random(20260909)
    for ids in [['an:95qvg9'],['an:40y1x9'],['an:7ogg04','an:4wdo37']]:
        bucket=sorted([r for r in an if r['pair_id'] in ids],key=lambda x:x['pair_id']);rng.shuffle(bucket)
        picked.extend(bucket[:max(0,2-len(picked))])
    af=[r for r in rows if r['dataset']=='ActivityForensics' and not r['split_status'].startswith('excluded') and r['real_exists'] and r['fake_exists'] and r['source_id'].startswith('charades:') and r['source_id'] in registered_train]
    used=set()
    for operation in ['add','delete']:
        bucket=sorted([r for r in af if r['official_description']=='official activity operations: '+operation and r['source_id'] not in used],key=lambda x:x['video_id'])
        rng.shuffle(bucket)
        if bucket:picked.append(bucket[0]);used.add(bucket[0]['source_id'])
    save(OUT/'candidate_manifest.json',{'seed':20260909,'max_pairs':4,'rows':picked[:4],
       'selection_basis':'AN: classB subject movement edits, prior camera still-review incl uncertainty; AF: one add/one delete if raw pair exists, source registered train in consulted protocols and not blocked; deterministic sorted then seeded shuffle within equal tier; no scores/depth deviations',
       'AF_training_status':'metadata candidates only, not approved for current method; no pixel GT, cannot be current spatial supervision samples',
       'source_split_limit':'No complete cross-experiment history guarantee; unknown/unregistered sources remain subject to future split approval'})
    return rows,evidence


def summarize(frame_rows,slices):
    result=[]
    for pair in PAIRS:
        rr=[r for r in frame_rows if r['pair']==pair];ss=[s for s in slices if s['pair']==pair];cov=[r['GT_pool_coverage'] for r in rr if r['GT_pool_coverage'] is not None]
        gt=sum(r['GT_pixels'] for r in rr);inside=sum(r['GT_selected_pixels'] for r in rr);normal=sum(s['nonedited_valid_pixels'] for s in ss)
        result.append({'pair':pair,'frames':len(rr),'GT_pixels':gt,'GT_selected_pixels':inside,'GT_excluded_pixels':gt-inside,
            'total_coverage':inside/gt if gt else None,'nonempty_frame_coverage_min':min(cov) if cov else None,
            'nonempty_frame_coverage_median':float(np.median(cov)) if cov else None,'nonempty_frame_coverage_max':max(cov) if cov else None,
            'frames_with_excluded_GT':sum(r['GT_excluded_pixels']>0 for r in rr),'edited_slice_frame_units':len(ss),
            'coaggregated_normal_pixels':normal,'edited_fraction_in_edited_slices':inside/(inside+normal) if inside+normal else None,
            'exclusion_paths':{k:sum(r[k] for r in rr) for k in ['excluded_original_native_valid','excluded_resample_no_support','excluded_resample_mixed_support','excluded_resample_nonfinite']}})
    return result


def report(summary,frames,slices,inventory_rows):
    lines=['# 编辑区输入覆盖与训练编辑类型清点','', '仅CPU数组/身份统计；不构造或运行编码器、SlabPool或检测模型，不做注意力归因。原始视频、GT、valid和XYZ不改。','',
       '|pair|帧数|GT像素观测数|进入池化|未进入|总覆盖|非空GT帧min/median/max|有排除帧|','|---|---:|---:|---:|---:|---:|---|---:|']
    for r in summary['coverage']:lines.append('|'+ '|'.join(map(str,[r['pair'],r['frames'],r['GT_pixels'],r['GT_selected_pixels'],r['GT_excluded_pixels'],r['total_coverage'],[r['nonempty_frame_coverage_'+x] for x in ['min','median','max']],r['frames_with_excluded_GT']]))+'|')
    lines+=['','单位是24帧累计像素观测，不是跨帧独立物理点数。GT按原生网格统计，没有缩小GT到几何网格。',
            '输入路径：Video.frame原生valid→torch.where(valid.reshape(-1))→slab_ids floor(Z/.125)。选中XYZ有限性、全部索引和切片计数守恒逐帧核对。无额外深度范围/片号/每片点数排除；空片无观测可选，不另删有效像素。',
            '高分辨率根：`'+str(SOURCE/'geometry')+'`；same-size使用原valid，resize使用双线性valid权重≥1−1e−6及有限XYZ。排除路径拆分见summary，未修补valid。',
            '', '## 混合统计（不是注意力贡献）','', '|pair|编辑片—帧单元数|同片非编辑有效像素累计|编辑像素占这些片的总量比例|','|---|---:|---:|---:|']
    for r in summary['coverage']:lines.append(f'|{r["pair"]}|{r["edited_slice_frame_units"]}|{r["coaggregated_normal_pixels"]}|{r["edited_fraction_in_edited_slices"]}|')
    lines+=['','SlabPool为全点学习式4头加权池化；数量占比不等于注意力权重。相同深度片包含正常内容只能支持混合稀释风险，不能证明特征已丢失。当前拟合目录保存的训练权重文件：'+json.dumps(summary['fit_checkpoint_files'])+'；不以随机模型解释最终注意力。','',
            '## 固定帧切片表','']
    page=['<!doctype html><meta charset="utf-8"><title>编辑区输入覆盖</title><style>body{font-family:sans-serif;max-width:1400px;margin:24px auto}img{max-width:100%}td,th{border:1px solid #aaa;padding:4px}table{border-collapse:collapse}td img{max-width:320px}</style><h1>编辑区输入覆盖：仅数量统计</h1><p>原图／官方GT复用上轮。valid来自实际高分辨率Video.frame；没有模型预测或权重可视化。GT统计在原生网格，网页缩放仅显示。无排除时不着色。</p><p><a href="../frame_coverage.csv">48帧覆盖CSV</a> · <a href="../edited_slice_counts.csv">全部编辑切片CSV</a> · <a href="../dataset_edit_inventory.csv">元数据清点</a></p>']
    for pair in PAIRS:
        for i in [0,12,23]:
            matches=[f for f in frames if f['pair']==pair and f['frame_index']==i]
            if not matches:continue
            f=matches[0];top=sorted([s for s in slices if s['pair']==pair and s['frame_index']==i],key=lambda s:(-s['edited_valid_pixels'],s['absolute_slice_id']))[:10]
            title=f'{pair} frame{i} PTS={f["PTS_s"]:.9f}';lines+=['### '+title,'','按编辑像素数降序仅显示前10片；统计未截断，完整CSV含全部编辑片。','',
              '|绝对片号|编辑有效像素|全部有效像素|非编辑有效像素|编辑比例|','|---|---:|---:|---:|---:|']
            for s in top:lines.append('|'+ '|'.join(str(s[k]) for k in ['absolute_slice_id','edited_valid_pixels','all_valid_pixels','nonedited_valid_pixels','edited_fraction'])+'|')
            prior=SCENE/'media'/pair.replace(':','_');rgb=Path(os.path.relpath(prior/f'fake_{i:03d}.png',OUT/'preview'));gt=Path(os.path.relpath(prior/f'GT_{i:03d}.png',OUT/'preview'))
            missing=f'{pair[3:]}_{i}_excluded.png' if f['GT_excluded_pixels'] else str(rgb)
            page += [f'<h2>{title}</h2><p>GT {f["GT_pixels"]}；进入 {f["GT_selected_pixels"]}；未进入 {f["GT_excluded_pixels"]}'+('（无，不增强颜色）' if not f['GT_excluded_pixels'] else '；红色只标实际未进入像素，无扩张')+'</p>',
                '<table><tr><th>原图</th><th>官方GT（当前二值化）</th><th>实际valid</th><th>GT中未进入像素叠加</th></tr><tr>'+''.join(f'<td><a href="{html.escape(str(p))}"><img src="{html.escape(str(p))}"></a></td>' for p in [rgb,gt,f'{pair[3:]}_{i}_valid.png',missing])+'</tr></table>']
            page.append('<p>切片表仅显示按编辑像素数前10，全部统计见CSV。</p><pre>'+html.escape(json.dumps(top,ensure_ascii=False,indent=2))+'</pre>');lines+=['']
    lines+=['## 编辑类别（按官方原文语义，不按模型结果）','', '|pair|主类|官方描述|分类依据|','|---|---|---|---|']
    for r in inventory_rows:
        if r['dataset']=='ANAKIN':lines.append(f'|{r["pair_id"]}|{r["category"]}|{r["official_description"]}|{r["basis"]}|')
    lines+=['','类别数量：'+json.dumps(summary['AN_category_counts'],ensure_ascii=False),'AF官方训练编辑记录/来源数及操作统计：'+json.dumps(summary['AF_inventory'],ensure_ascii=False),
            'AF仅核对本地官方元数据、来源划分及文件存在。活动增删语义更贴活动修改，但未明确局部动作/姿态细节，主类待确认；没有经核验的像素GT，只能作为检测/时间候选。官方train不等于本方法批准训练清单。',
            '优先候选（最多4）身份、官方描述、路径、分层标签与划分限制见candidate_manifest.json；复用旧预览链接，不导出新视频。',
            '覆盖高不证明特征保留，深度图自然也不排除取证线索。本轮无法判断注意力是否关注编辑区或关系可分性；优先后续方向须依据本轮实际覆盖和任务差异，不能据此擅改训练集合。',
            f'耗时{summary["seconds"]:.3f}秒。执行基准HEAD：{summary["base_head"]}。独立产物：`{OUT}`。无训练/前端/GPU/下载；计算与Git发布分开。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(REPO/'docs/pvst/edit_region_coverage_v1.md').write_text(text)
    (OUT/'preview/index.html').write_text('\n'.join(page))


def main():
    import subprocess
    started=time.monotonic();deadline=started+600;torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    frame_rows=[];slices=[];summary={'status':'FAILED','base_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    c=json.loads((SOURCE/'config.json').read_text());c['geometry_root']=str(SOURCE/'geometry')
    m=json.loads((SOURCE/'manifest.json').read_text());rows=[r for r in m['rows'] if r['split']=='train' and r['pair_id'] in PAIRS]
    old=json.loads((FIT/'selected_identity.json').read_text());oldgt=json.loads((FIT/'GT_identity_supplement.json').read_text())
    if 'points=torch.where(valid.reshape(-1))[0]' not in inspect.getsource(SlabPool.forward):raise ValueError('POOL_SELECTION_SOURCE_CHANGED')
    save(OUT/'source_identity.json',{'source_manifest_sha256':digest(SOURCE/'manifest.json'),'code_sha256':{p:digest(REPO/p) for p in ['research_tools/rgbd_slice/data.py','research_tools/rgbd_slice/model.py','research_tools/depth_slice/model.py']},'geometry_root':c['geometry_root'],'pool_selection':'torch.where(valid.reshape(-1))[0]','slab_rule':inspect.getsource(slab_ids)})
    real_checks=[]
    for r in rows:
        before=next(x for x in old if x['video_id']==r['video_id']);gt_before=next(x for x in oldgt if x['video_id']==r['video_id'])
        if any(before[k]!=r[k] for k in ['pair_id','source_id','role','video_sha256','source_hw']) or before['pts']!=r['pts'][:24]:raise ValueError('FIXED_CLIP_IDENTITY')
        if gt_before['mask_sha256']!=r['mask_sha256'] or digest(Path(r['video_path']))!=r['video_sha256']:raise ValueError('FIXED_MEDIA_IDENTITY')
        if r['role']=='fake' and digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SHA')
        v=Video(r,c)
        try:
            if max(v.xyz.shape[1:3])<=256:raise ValueError('OLD_GEOMETRY_FORBIDDEN')
            if r['role']=='real':
                if v.zip is not None or any(v.gt(i).any() for i in range(24)):raise ValueError('REAL_GT_NOT_LEGITIMATE_ZERO')
                real_checks.append({'pair':r['pair_id'],'video_id':r['video_id'],'video_sha256':r['video_sha256'],'real_zero_GT':True,'fake_pair_mask_not_used':True});continue
            for i in range(24):
                if time.monotonic()>deadline:
                    table(OUT/'frame_coverage.csv',frame_rows);table(OUT/'edited_slice_counts.csv',slices)
                    raise TimeoutError('600_SECOND_CPU_BUDGET')
                f=v.frame(i)
                if f['frame_index']!=i or f['pts']!=r['pts'][i]['pts_s']:raise ValueError('FRAME_PTS')
                a,ss=counts(f);a.update(pair=r['pair_id'],video_id=r['video_id'],frame_index=i,PTS_s=f['pts'],native_hw=str(r['source_hw']),geometry_hw=str(list(v.xyz.shape[1:3])),**exclusion_path(v,f,i));frame_rows.append(a)
                slices.extend({'pair':r['pair_id'],'video_id':r['video_id'],'frame_index':i,'PTS_s':f['pts'],**s} for s in ss)
                if i in [0,12,23]:figures(f,r['pair_id'],i)
            print(r['pair_id'],'24 fake frames complete',flush=True)
            table(OUT/'frame_coverage.csv',frame_rows);table(OUT/'edited_slice_counts.csv',slices)
        finally:v.close()
    table(OUT/'frame_coverage.csv',frame_rows);table(OUT/'edited_slice_counts.csv',slices);save(OUT/'real_identity_checks.json',real_checks)
    inv,evidence=inventory(deadline);table(OUT/'dataset_edit_inventory.csv',inv)
    category_counts=Counter(r['category'] for r in inv if r['dataset']=='ANAKIN')
    summary.update(status='COMPLETE',coverage=summarize(frame_rows,slices),AN_category_counts={k:category_counts[k] for k in ['A','B','C','待确认']},
                   AF_inventory={k:evidence[k] for k in ['AF_train_edit_rows','AF_train_source_count','AF_operation_counts','all_nontrain_sources_excluded']},
                   fit_checkpoint_files=[str(p.name) for p in FIT.glob('*.pt')],seconds=time.monotonic()-started,model_frontend_GPU_download_calls=0)
    save(OUT/'summary.json',summary);report(summary,frame_rows,slices,inv);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        save(OUT/'execution_error.json',{'status':'TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED','error':str(error),'partial_CSVs_preserved':True})
        raise
