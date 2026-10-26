"""Fixed ANAKIN asset review, called by CLI; no models or training imports.

Produces a small selection manifest and external media/mask/HTML evidence.
Existing smoke labels describe visible anomalies, not published edit regions.
"""
import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import html
import io
import json
from pathlib import Path
import random
import re
import time
import zipfile

import av
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

from scripts.prepare_v8_assets import digest, verify

SEED = 20260909
REPO = 'AlexBlck/ANAKIN'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def select(root, manifest):
    refs=root/'references'
    info=json.loads((refs/'dataset_info.json').read_text())
    for name in ['metadata.csv','ANAKIN.py']:
        entry=next(e for e in info['siblings'] if e['rfilename']==name)
        check=verify(refs/name,{'size':entry['size'],'lfs':entry.get('lfs'),'blob_id':entry.get('blobId')})
        if check['status']!='VERIFIED':raise ValueError('PINNED_REFERENCE_INTEGRITY_FAILED:'+name)
    if manifest.exists() and json.loads(manifest.read_text())['revision']!=info['sha']:
        raise ValueError('SELECTION_REVISION_CONFLICT')
    rows=list(csv.DictReader((refs/'metadata.csv').open()))
    configs={}
    for config in ['all','no-full','has-masks','full-masks']:
        subset=[r for r in rows if (config not in ['all','full-masks'] or r['full-available']=='TRUE')
                and (config not in ['has-masks','full-masks'] or r['has-masks']=='TRUE')]
        ids=[r['video-id'] for r in subset];random.Random(47).shuffle(ids)
        a,b=int(len(ids)*.7),int(len(ids)*.8)
        configs[config]={'train':ids[:a],'validation':ids[a:b],'test':ids[b:]}
    train=set(configs['has-masks']['train'])
    # Description is only a pre-download eligibility filter, never visual approval.
    human=r'\b(man|men|woman|women|person|people|guy|girl|boy|player|runners?|skateboard|bike|neck|head|ear|hat|shirt|chest|arm|hair|face|badge|brooch|necklace|glasses|tie)\b'
    excluded=r'\b(text|letters|inscription|poster|graph|print|pattern|logo|emblem|picture|smoke|water|floating|flyboard|color|colour|background replacement)\b'
    eligible=[r for r in rows if r['video-id'] in train and r['has-masks']=='TRUE'
              and r['manipulation-type'] in ['splicing','inpainting']
              and re.search(human,r['task'],re.I) and not re.search(excluded,r['task'],re.I)]
    rng=random.Random(SEED);chosen=[]
    for kind in ['splicing','inpainting']:
        group=sorted([r for r in eligible if r['manipulation-type']==kind],key=lambda r:r['video-id'])
        rng.shuffle(group);chosen.extend(group[:10])
    chosen.sort(key=lambda r:r['video-id'])
    result={'repo':REPO,'revision':info['sha'],'selection_seed':SEED,
        'rules':'has-masks official train (loader seed47, CSV order); description regex human/body/interaction; exclude text/graphic/color/background/water/smoke terms; 10 splicing + 10 inpainting using independent selection seed; no visual replacement',
        'human_regex':human,'exclude_regex':excluded,'split_config':'has-masks',
        'metadata_sha256':digest(refs/'metadata.csv'),'loader_sha256':digest(refs/'ANAKIN.py'),
        'rows':[{**r,'official_split':'train','review_pool':'asset_review_pool',
                 'official_splits':{k:next((s for s,ids in d.items() if r['video-id'] in ids),None) for k,d in configs.items()},
                 'paths':{k:f'{k}/{r["video-id"]}.{ "zip" if k=="masks" else "mp4"}' for k in ['trimmed','edited','masks']}}
                for r in chosen]}
    save(manifest,result)
    urls=Counter(r['url'] for r in rows)
    save(root/'metadata_audit.json',{'revision':info['sha'],'rows':len(rows),'unique_ids':len({r['video-id'] for r in rows}),
        'mask_rows':sum(r['has-masks']=='TRUE' for r in rows),'types':dict(Counter(r['manipulation-type'] for r in rows)),
        'masked_types':dict(Counter(r['manipulation-type'] for r in rows if r['has-masks']=='TRUE')),
        'unique_urls':len(urls),'repeated_urls':{url:[{k:r[k] for k in ['video-id','task','start-time','end-time','manipulation-type']} for r in rows if r['url']==url] for url,n in urls.items() if n>1},
        'config_counts':{k:{s:len(v) for s,v in d.items()} for k,d in configs.items()},'config_splits':configs,
        'eligible':len(eligible),'selected':len(chosen)})


def download(root, manifest):
    selection=json.loads(manifest.read_text());info=json.loads((root/'references/dataset_info.json').read_text())
    entries={r['rfilename']:{'path':r['rfilename'],'size':r.get('size'),'lfs':r.get('lfs'),'blob_id':r.get('blobId')} for r in info['siblings']}
    start=time.time()
    def fetch(path):
        p=root/path;e=entries[path]
        if verify(p,e)['status']!='VERIFIED':
            hf_hub_download(REPO,path,repo_type='dataset',revision=selection['revision'],local_dir=root,token=False)
        check=verify(p,e)
        if check['status']!='VERIFIED':raise ValueError(f'INTEGRITY_FAILED:{path}:{check}')
        return {'path':path,**check,'local_sha256':digest(p)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,[p for r in selection['rows'] for p in r['paths'].values()]))
    save(root/'download_audit.json',{'status':'COMPLETE','revision':selection['revision'],'elapsed_s':time.time()-start,
                                   'bytes':sum(r['bytes'] for r in results),'files':results})


def numeric_members(names):
    pairs=[]
    for name in names:
        if name.endswith('/') or Path(name).suffix.lower() not in {'.png','.jpg','.jpeg','.bmp'}:continue
        numbers=re.findall(r'\d+',Path(name).stem)
        if len(numbers)!=1:raise ValueError('AMBIGUOUS_MASK_FRAME_NUMBER:'+name)
        pairs.append((int(numbers[0]),name))
    if len({i for i,n in pairs})!=len(pairs):raise ValueError('DUPLICATE_MASK_FRAME_NUMBER')
    return sorted(pairs)


def alignment(numbers, count):
    if numbers==list(range(count)):return {'status':'aligned','index_base':0}
    if numbers==list(range(1,count+1)):return {'status':'aligned','index_base':1}
    return {'status':'alignment_unresolved','index_base':None,'missing_annotation_not_normal':True}


def mask_timing(match, positive, pts):
    if match['status']!='aligned' or any(p['seconds'] is None for p in pts):
        return {'profile':'unresolved','positive_runs':None}
    indices=[i-match['index_base'] for i in positive];runs=[]
    for i in indices:
        if not runs or i!=runs[-1][-1]+1:runs.append([i])
        else:runs[-1].append(i)
    return {'profile':'all_nonempty' if len(indices)==len(pts) else 'mixed_empty_nonempty' if indices else 'all_empty',
        'positive_runs':[{'first_frame':r[0],'last_frame_inclusive':r[-1],
            'first_pts_s':pts[r[0]]['seconds'],'last_positive_pts_s':pts[r[-1]]['seconds']} for r in runs],
        'boundary_convention':'closed sampled PTS range, not an inferred continuous end-time'}


def mask_array(data):
    im=Image.open(io.BytesIO(data));im.load();a=np.asarray(im)
    if a.ndim==3:
        # Published binary-region masks can retain RGB compression/soft edges.
        # Explicit standard luminance conversion; original ZIP remains untouched.
        a=np.asarray(im.convert('L'))
    if a.ndim!=2 or a.dtype!=np.uint8:raise ValueError('MASK_ENCODING_UNRESOLVED')
    return a,a.astype(np.float32)/255 >= .5,im.mode


def decode(path, keep):
    pts=[];sizes=set();images={}
    with av.open(str(path)) as c:
        s=c.streams.video[0];s.thread_count=2;rate=str(s.average_rate);tb=str(s.time_base)
        for i,f in enumerate(c.decode(s)):
            pts.append({'ticks':f.pts,'seconds':float(f.pts*f.time_base) if f.pts is not None else None})
            sizes.add((f.height,f.width))
            if i in keep:images[i]=Image.fromarray(f.to_ndarray(format='rgb24'))
    return {'frame_count':len(pts),'sizes_hw':sorted(sizes),'time_base':tb,'average_rate':rate,'pts':pts},images


def review(root, manifest):
    started=time.time();selection=json.loads(manifest.read_text());out=root/'preview';out.mkdir(exist_ok=True);results=[];cards=[]
    scene_file=manifest.with_name('anakin_scene_review.json')
    scenes={r['video_id']:r for r in json.loads(scene_file.read_text())['rows']} if scene_file.exists() else {}
    for row in selection['rows']:
        vid=row['video-id'];r={'video_id':vid,'task':row['task'],'source_url':row['url'],'type':row['manipulation-type']}
        # Actual complete decode establishes count; only fixed preview frames retained.
        fake,_=decode(root/row['paths']['edited'],set());n=fake['frame_count']
        fixed={min(n-1,int(n*f)) for f in [.25,.5,.75]};masks={};modes=set();values=set();sizes=set();positive=[]
        with zipfile.ZipFile(root/row['paths']['masks']) as z:
            members=numeric_members(z.namelist());match=alignment([i for i,name in members],n)
            for num,name in members:
                a,b,mode=mask_array(z.read(name));modes.add(mode);values.update(np.unique(a).tolist());sizes.add(a.shape)
                if b.any():positive.append(num)
                masks[num]=b if num in {i+(match['index_base'] or 0) for i in fixed} else None
            first=positive[0] if positive else None
            if first is not None and match['status']=='aligned':
                fixed.add(first-match['index_base']);name=dict(members)[first];masks[first]=mask_array(z.read(name))[1]
        if len(sizes)!=1 or len(fake['sizes_hw'])!=1 or list(next(iter(sizes)))!=list(fake['sizes_hw'][0]):
            match={'status':'alignment_unresolved','index_base':None,'reason':'MASK_EDITED_DIMENSION_MISMATCH_NO_OFFICIAL_TRANSFORM'}
        r.update(edited=fake,alignment=match,mask={'count':len(members),'number_min':members[0][0] if members else None,
            'number_max':members[-1][0] if members else None,'modes':sorted(modes),'values':sorted(values),'sizes_hw':sorted(sizes),
            'foreground_rule':'PIL RGB-to-L luminance then uint8/255 >= 0.5; white edited-region interpretation checked by overlays; original ZIP unchanged',
            'positive_numbers':positive,'empty_count':len(members)-len(positive)})
        real,real_images=decode(root/row['paths']['trimmed'],fixed);_,fake_images=decode(root/row['paths']['edited'],fixed)
        r['trimmed']=real;r['scene_status']='pending_visual_review';r['preview_frames']=sorted(fixed)
        r['mask_timing']=mask_timing(match,positive,fake['pts'])
        r['scene_review']=scenes.get(vid,{'scene_status':'pending_visual_review'})
        r['scene_status']=r['scene_review']['scene_status']
        r['first_nonempty_gt_preview']=first
        comparisons=[]
        for i in sorted(fixed):
            images=[]
            if i in real_images:images.append(real_images[i])
            images.append(fake_images[i]);mask=masks.get(i+(match['index_base'] or 0))
            if match['status']=='aligned' and mask is not None:
                a=np.asarray(fake_images[i]).copy();a[mask]=(a[mask]*.55+np.array([255,0,0])*.45).astype('uint8');images.append(Image.fromarray(a))
            thumbs=[]
            for im in images:
                im=im.copy();im.thumbnail((320,200));thumbs.append(im)
            panel=Image.new('RGB',(320*len(thumbs),220),'white')
            for j,im in enumerate(thumbs):panel.paste(im,(320*j,20))
            panel.save(out/f'{vid}_{i}.jpg');comparisons.append(f'<p>frame {i}; PTS {fake["pts"][i]["seconds"]:.4f}s; real / edited / GT overlay</p><img width="960" src="{vid}_{i}.jpg">')
        save(out/f'{vid}.json',r);results.append(r)
        cards.append(f'<article id="{vid}"><h2>{vid}</h2><p>{html.escape(row["task"])}</p><p>官方编辑区域；不是可见异常标注。首个非空帧按GT选取。</p>'
            f'<p>空间对应：{match["status"]}。{html.escape(json.dumps(r["scene_review"],ensure_ascii=False))}</p>'
            f'<video controls width="450" src="../{row["paths"]["trimmed"]}"></video><video controls width="450" src="../{row["paths"]["edited"]}"></video>'+''.join(comparisons)+'</article>')
        print(vid,match['status'],n,len(members),flush=True)
    save(root/'media_audit.json',{'revision':selection['revision'],'verification_s':time.time()-started,'rows':results})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ANAKIN资产核验</title><h1>原始短片、编辑短片与公开编辑区域GT</h1><p>固定25%/50%/75%帧及首个非空GT帧；无模型输出。逐项实测与待确认事项见各video-id.json。</p>'+''.join(cards))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','download','review']);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    globals()[a.action](a.root,a.manifest)

if __name__=='__main__':main()
