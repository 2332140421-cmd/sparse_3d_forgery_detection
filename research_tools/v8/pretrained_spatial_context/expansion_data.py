"""This expansion only: fixed candidates, one original-video review, visual-only caches."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlsplit
import zipfile

import av
import numpy as np
from PIL import Image,ImageDraw
import torch

from scripts.prepare_v8_assets import digest
from scripts.review_anakin import numeric_members,alignment,mask_array
from research_tools.v8.spatial_localization.joint_data import REV,SEED,save,scan,canvas,fetch_anakin
from research_tools.v8.spatial_localization.data import content_mask,mask_to_canvas
from research_tools.v8.full_video_streaming.inputs import decode


def source_key(url):
    u=urlsplit(url);return u.netloc.lower().removeprefix('www.')+u.path.rstrip('/')


def inventory(config,out):
    root=Path(config['data_root'])/'secondary_datasets/anakin';refs=root/'references'
    info=json.loads((refs/'dataset_info.json').read_text());policy=json.loads(Path(config['selection_policy']).read_text())
    if info['sha']!=REV or digest(refs/'metadata.csv')!=policy['metadata_sha256'] or digest(refs/'ANAKIN.py')!=policy['loader_sha256']:raise ValueError('PINNED_METADATA_REQUIRED')
    metadata=list(csv.DictReader((refs/'metadata.csv').open()));splits=json.loads((root/'metadata_audit.json').read_text())['config_splits']['has-masks']
    old=json.loads((Path(config['original_run'])/'manifest.json').read_text())['rows'];old_by={r['pair_id'][3:]:r for r in old if r['role']=='fake'}
    selected=[r for r in metadata if r['video-id'] in old_by or (r['has-masks']=='TRUE' and r['manipulation-type'] in ['inpainting','splicing']
              and re.search(policy['human_regex'],r['task'],re.I) and not re.search(policy['exclude_regex'],r['task'],re.I))]
    selected.sort(key=lambda r:r['video-id']);random.Random(SEED).shuffle(selected)
    scenes={r['video_id']:r for r in json.loads(Path(config['scene_review']).read_text())['rows']}
    # Conservative reserved-test exclusion: every previously downloaded source, plus all previous manifests.
    seen={source_key(r['url']) for r in metadata if (root/'trimmed'/f'{r["video-id"]}.mp4').exists()}
    for p in (Path(config['data_root'])/'derived').glob('*/*manifest*.json'):
        if out.resolve() in p.resolve().parents:continue
        d=json.loads(p.read_text())
        for r in d.get('rows',[]):
            if r.get('dataset')=='ANAKIN':seen.add(source_key(r.get('original_url',r.get('source_id','anakin:').removeprefix('anakin:'))))
    rows=[]
    for i,r in enumerate(selected):
        vid=r['video-id'];scene=scenes.get(vid)
        rows.append({**r,'candidate_rank':i,'source_group':source_key(r['url']),
                     'official_split':next(s for s,ids in splits.items() if vid in ids),
                     'retained_split':old_by[vid]['split'] if vid in old_by else None,
                     'previously_downloaded':(root/'trimmed'/f'{vid}.mp4').exists(),
                     'test_unseen_source':source_key(r['url']) not in seen,'prior_scene_review':scene})
    result={'revision':REV,'seed':SEED,'target_pairs':{'train':60,'dev':20,'test_reserved':20},
            'rules':'retain original roles; same original URL/source never crosses split; metadata policy unchanged; original-only fixed25/50/75 review; no model scores; numeric mask frames and native dimensions exact; no repair',
            'area_bins':{'small':'mean positive-frame native mask fraction <0.01 (all-empty recorded separately)','medium':'0.01 <= fraction <0.10','large':'>=0.10'},
            'selection':'round-robin manipulation-type x area bin; fixed seed within each bin; test official-test first from unseen sources; then dev, then train; at most one editing version per source',
            'metadata_sha256':digest(refs/'metadata.csv'),'counts':{'metadata':len(metadata),'has_masks':sum(r['has-masks']=='TRUE' for r in metadata),
                'eligible_including_retained':len(rows),'masked_types':dict(Counter(r['manipulation-type'] for r in metadata if r['has-masks']=='TRUE')),
                'existing_triplets':sum(all((root/k/f'{r["video-id"]}{ext}').exists() for k,ext in [('trimmed','.mp4'),('edited','.mp4'),('masks','.zip')]) for r in metadata)},
            'rows':rows}
    dest=out/'candidate_order.json'
    if dest.exists():return json.loads(dest.read_text())  # Fixed before any new downloads or review.
    save(dest,result);return result


def original_previews(config,out):
    data=inventory(config,out);root=Path(config['data_root'])/'secondary_datasets/anakin'
    entries={e['rfilename']:e for e in json.loads((root/'references/dataset_info.json').read_text())['siblings']}
    pending=[r for r in data['rows'] if not r['retained_split'] and not (r['prior_scene_review'] and r['prior_scene_review']['scene_status'] in ['preliminary_match','not_matching'])]
    def fetch(r):
        name='trimmed/'+r['video-id']+'.mp4';fetch_anakin(root/name,entries[name],root,out)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,pending))
    directory=out/'scene_preview';directory.mkdir(exist_ok=True);scans={};panels=[]
    for r in pending:
        vid=r['video-id'];side=directory/(vid+'.json');preview=directory/(vid+'.jpg')
        if side.exists():scans[vid]=json.loads(side.read_text());panels.append((r,preview));continue
        thumbs=[];pts=[];hw=None
        with av.open(str(root/'trimmed'/f'{vid}.mp4')) as container:
            stream=container.streams.video[0];stream.thread_count=2;origin=float((stream.start_time or 0)*stream.time_base)
            for frame in container.decode(stream):
                size=[frame.height,frame.width]
                if hw and size!=hw:raise ValueError('VARIABLE_FRAME_SIZE:'+vid)
                hw=size
                if frame.pts is None:raise ValueError('MISSING_NATIVE_PTS:'+vid)
                pts.append({'pts_ticks':frame.pts,'time_base':str(frame.time_base),'pts_s':float(frame.pts*frame.time_base)-origin})
                im=frame.to_image();im.thumbnail((192,120));thumbs.append(im)
        if not pts or any(b['pts_s']<=a['pts_s'] for a,b in zip(pts,pts[1:])):raise ValueError('INVALID_NATIVE_PTS:'+vid)
        panel=Image.new('RGB',(576,146),'white');draw=ImageDraw.Draw(panel);draw.text((3,3),vid+' original: 25% / 50% / 75%',fill='black')
        for j,f in enumerate([.25,.5,.75]):panel.paste(thumbs[int(len(thumbs)*f)],(192*j,24))
        panel.save(preview)
        value={'actual_frames':len(pts),'source_hw':hw,'pts':pts};save(side,value);scans[vid]=value;panels.append((r,preview))
    for start in range(0,len(panels),10):
        batch=panels[start:start+10];sheet=Image.new('RGB',(576,146*len(batch)),'white')
        for j,(_,p) in enumerate(batch):sheet.paste(Image.open(p),(0,146*j))
        sheet.save(directory/f'sheet_{start//10:02d}.jpg')
    save(out/'scene_preview_index.json',{'rule':'one original-only quick review, fixed25/50/75; no model score or fake appearance',
        'rows':[{'video_id':r['video-id'],'task':r['task'],'preview':str(p)} for r,p in panels]})
    print(json.dumps({'candidates':len(data['rows']),'new_original_reviews':len(panels),'sheets':(len(panels)+9)//10}),flush=True)


def stratified_order(rows,seed=SEED):
    bins=defaultdict(list)
    for r in sorted(rows,key=lambda r:r['video-id']):bins[(r['manipulation-type'],r['area_group'])].append(r)
    rng=random.Random(seed)
    for group in sorted(bins):rng.shuffle(bins[group])
    result=[]
    while any(bins.values()):
        for group in sorted(bins):
            if bins[group]:result.append(bins[group].pop())
    return result


def allocate(rows,targets):
    selected=[];used=set()
    for r in rows:
        if r['retained_split']:
            if r['source_group'] in used:raise ValueError('RETAINED_SOURCE_CONFLICT')
            selected.append({**r,'split':r['retained_split']});used.add(r['source_group'])
    candidates=stratified_order([r for r in rows if not r['retained_split']])
    for split in ['test_reserved','dev','train']:
        count=sum(r['split']==split for r in selected)
        pool=[r for r in candidates if split!='test_reserved' or r['test_unseen_source']]
        if split=='test_reserved':pool=[r for r in pool if r['official_split']=='test']+[r for r in pool if r['official_split']!='test']
        for r in pool:
            if count>=targets[split]:break
            if r['source_group'] in used:continue
            selected.append({**r,'split':split});used.add(r['source_group']);count+=1
    return selected


def prepare(config,out):
    started=time.monotonic();data=inventory(config,out);root=Path(config['data_root'])/'secondary_datasets/anakin'
    decisions={r['video_id']:r for r in json.loads((out/'scene_review.json').read_text())['rows']}
    entries={e['rfilename']:e for e in json.loads((root/'references/dataset_info.json').read_text())['siblings']}
    accepted=[];rejected=[]
    for r in data['rows']:
        vid=r['video-id'];prior=r['prior_scene_review']
        scene={'scene_status':'retained_original_role','reason':'keep original train/dev identity'} if r['retained_split'] else prior if prior and prior['scene_status'] in ['preliminary_match','not_matching'] else decisions.get(vid)
        if not scene:raise ValueError('SCENE_REVIEW_MISSING:'+vid)
        if scene['scene_status'] not in ['preliminary_match','matching','retained_original_role']:
            rejected.append({'video_id':vid,'reason':'scene_not_matching','review':scene});continue
        accepted.append({**r,'scene_review':scene})
    def fetch(r):
        for k,ext in [('trimmed','.mp4'),('edited','.mp4'),('masks','.zip')]:
            name=f'{k}/{r["video-id"]}{ext}';fetch_anakin(root/name,entries[name],root,out)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,accepted))
    old_rows=json.loads((Path(config['original_run'])/'manifest.json').read_text())['rows'];old={r['video_id']:r for r in old_rows};old_pairs=defaultdict(list)
    for r in old_rows:old_pairs[r['pair_id'][3:]].append(r)
    for index,r in enumerate(accepted):
        vid=r['video-id'];review_path=out/'technical_review'/f'{vid}.json'
        if review_path.exists():result=json.loads(review_path.read_text())
        elif vid in old_pairs:
            rr=old_pairs[vid];fake=next(x for x in rr if x['role']=='fake')
            result={'status':'accepted','area_fraction':native_area(fake),'rows':rr,'reused_original_technical_acceptance':True}
            save(review_path,result)
        else:
            try:
                fake=scan(root/'edited'/f'{vid}.mp4');preview=out/'scene_preview'/f'{vid}.json'
                real=json.loads(preview.read_text()) if preview.exists() else scan(root/'trimmed'/f'{vid}.mp4')
                from research_tools.v8.spatial_localization.joint_data import mapping
                for m in [real,fake]:m.update(mapping256=mapping(m['source_hw'],256),mapping512=mapping(m['source_hw'],512))
                zpath=root/'masks'/f'{vid}.zip';sizes=set();areas=[]
                with zipfile.ZipFile(zpath) as z:
                    members=numeric_members(z.namelist());match=alignment([i for i,n in members],fake['actual_frames'])
                    for _,name in members:
                        raw,binary,_=mask_array(z.read(name));sizes.add(raw.shape)
                        if binary.any():areas.append(float(binary.mean()))
                if match['status']!='aligned' or sizes!={tuple(fake['source_hw'])}:raise ValueError('alignment_unresolved: frame numbering/count/native mask size')
                rr=[];mask_sha=digest(zpath)
                for role,kind,meta in [('real','trimmed',real),('fake','edited',fake)]:
                    path=root/kind/f'{vid}.mp4';sha=digest(path);source='anakin:'+r['url']
                    video_id=hashlib.sha256(f'ANAKIN:{source}:{role}:{sha}'.encode()).hexdigest()[:24]
                    rr.append({'dataset':'ANAKIN','video_id':video_id,'source_id':source,'role':role,'label':int(role=='fake'),
                        'video_path':str(path),'video_sha256':sha,'pair_id':'an:'+vid,'generator':r['manipulation-type'],
                        'original_url':r['url'],'official_revision':REV,'official_config':'has-masks','official_split':r['official_split'],
                        'annotation_source':'official masks/'+vid+'.zip','task':r['task'],'mask_zip':str(zpath),'mask_sha256':mask_sha,
                        'mask_members':members,'index_base':match['index_base'],'spatial_target':'edit_region',
                        'spatial_status':'edit_mask' if role=='fake' else 'known_background','train_monitor':False,**meta})
                result={'status':'accepted','area_fraction':float(np.mean(areas)) if areas else 0.,'all_empty':not bool(areas),'rows':rr}
            except (ValueError,av.error.FFmpegError) as e:result={'status':'rejected','reason':str(e)}
            save(review_path,result)
        if result['status']!='accepted':
            if r['retained_split']:raise ValueError('RETAINED_TECHNICAL_FAILURE:'+vid)
            rejected.append({'video_id':vid,'reason':result['reason']});continue
        fraction=result['area_fraction'];r.update(technical=result,area_fraction=fraction,area_group='small' if fraction<.01 else 'medium' if fraction<.1 else 'large')
        print(f'technical {index+1}/{len(accepted)} {vid} accepted',flush=True)
    valid=[r for r in accepted if 'technical' in r];chosen=allocate(valid,data['target_pairs']);rows=[]
    for r in chosen:
        for rr in r['technical']['rows']:
            rr={**rr,'split':r['split'],'source_group':r['source_group'],'area_fraction':r['area_fraction'],'area_group':r['area_group'],'scene_review':r['scene_review']}
            if rr['video_id'] in old:
                rr['cache_dir']=str(Path(config['source_run'])/'cache'/rr['video_id'])
                rr['feature_path']=str(Path(config['original_run'])/'features'/rr['video_id']/'patch.npy')
            else:rr['cache_dir']=str(out/'cache'/rr['video_id']);rr['feature_path']=str(out/'features'/rr['video_id']/'patch.npy')
            rows.append(rr)
    counts={s:sum(r['split']==s for r in chosen) for s in ['train','dev','test_reserved']}
    allowed=counts['train']>24 and counts['dev']>8
    manifest={'schema':'v8-spatial-visual-expansion-v1','revision':REV,'seed':SEED,'selection_rules':{k:data[k] for k in ['rules','area_bins','selection']},
              'targets':data['target_pairs'],'actual_pairs':counts,'training_allowed':allowed,'rows':rows}
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('FROZEN_MANIFEST_CHANGED')
    if not path.exists():save(path,manifest)
    for r in rows:
        if r['split']!='test_reserved':prepare_cache(r)
    report={'candidate_counts':data['counts'],'scene_passed':len(accepted),'technical_passed':len(valid),'actual_pairs':counts,
            'independent_sources':{s:len({r['source_group'] for r in rows if r['split']==s}) for s in counts},
            'frames':{s:sum(r['actual_frames'] for r in rows if r['split']==s) for s in counts},
            'types':{s:dict(Counter(r['generator'] for r in rows if r['split']==s and r['role']=='fake')) for s in counts},
            'exclusions':rejected,'target_shortfall':{s:data['target_pairs'][s]-counts[s] for s in counts},'training_allowed':allowed,'seconds':time.monotonic()-started}
    transport=[json.loads(line) for line in (out/'download_transport.jsonl').read_text().splitlines()] if (out/'download_transport.jsonl').exists() else []
    report.update(new_media_files=len(transport),new_media_bytes=sum(r['bytes'] for r in transport),original_preview_seconds=None,
        official_splits={s:dict(Counter(r['official_split'] for r in rows if r['split']==s and r['role']=='fake')) for s in counts},
        area_groups={s:dict(Counter(r['area_group'] for r in rows if r['split']==s and r['role']=='fake')) for s in counts})
    save(out/'asset_report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
    return rows


def native_area(r):
    areas=[]
    with zipfile.ZipFile(r['mask_zip']) as z:
        for _,name in r['mask_members']:
            a=mask_array(z.read(name))[1]
            if a.any():areas.append(float(a.mean()))
    return float(np.mean(areas)) if areas else 0.


def prepare_cache(r):
    directory=Path(r['cache_dir']);complete=directory/'complete.json'
    if complete.exists():
        meta=json.loads(complete.read_text())
        if meta['video_sha256']!=r['video_sha256']:raise ValueError('CACHE_VIDEO_IDENTITY')
        for k in ['rgb512','gt','content512']:
            a=np.load(directory/(k+'.npy'),mmap_mode='r')
            if a.shape[0]!=r['actual_frames'] and not (k=='content512' and a.shape==(512,512)):raise ValueError('CACHE_FRAME_IDENTITY')
        return
    directory.mkdir(parents=True,exist_ok=True);n=r['actual_frames']
    rgb=np.lib.format.open_memmap(directory/'rgb512.npy',mode='w+',dtype=np.uint8,shape=(n,512,512,3))
    gt=np.lib.format.open_memmap(directory/'gt.npy',mode='w+',dtype=bool,shape=(n,512,512))
    z=zipfile.ZipFile(r['mask_zip']) if r['role']=='fake' else None;count=0
    try:
        for i,bgr,info in decode(r['video_path']):
            if i>=n or abs(info['pts_s']-r['pts'][i]['pts_s'])>1e-8:raise ValueError('CACHE_PTS_IDENTITY')
            rgb[i]=canvas(bgr,r['mapping512'])[:,:,::-1]
            gt[i]=mask_to_canvas(mask_array(z.read(r['mask_members'][i][1]))[1],r['mapping512']) if z else False
            count+=1
    finally:
        if z:z.close()
    if count!=n:raise ValueError('CACHE_FULL_FRAME_COVERAGE')
    rgb.flush();gt.flush();np.save(directory/'content512.npy',content_mask(r['mapping512']))
    save(complete,{'schema':'v8-spatial-visual-expansion-v1','video_sha256':r['video_sha256'],'actual_frames':n,
                   'pts_sha256':hashlib.sha256(json.dumps(r['pts'],sort_keys=True).encode()).hexdigest(),
                   'fields':['rgb512','gt','content512'],'mapping512':r['mapping512'],'preprocess':'existing RGB512 letterbox; no geometry fields'})


class VisualVideo:
    def __init__(self,config,out,row,fit=None,context=False):
        if row['split']=='test_reserved':raise ValueError('RESERVED_TEST_MUST_NOT_ENTER_MODEL')
        self.row=row;self.base=self;directory=Path(row['cache_dir']);meta=json.loads((directory/'complete.json').read_text())
        if meta['video_sha256']!=row['video_sha256']:raise ValueError('VISUAL_CACHE_IDENTITY')
        if meta.get('schema')=='v8-spatial-visual-expansion-v1':
            pts_sha=hashlib.sha256(json.dumps(row['pts'],sort_keys=True).encode()).hexdigest()
            if meta['mapping512']!=row['mapping512'] or meta['pts_sha256']!=pts_sha:raise ValueError('VISUAL_CACHE_MAPPING_OR_PTS')
        self.arrays={k:np.load(directory/(k+'.npy'),mmap_mode='r') for k in ['rgb512','gt','content512']}
        if 'actual_frames' in row and any(a.shape[0]!=row['actual_frames'] for k,a in self.arrays.items() if k!='content512' or a.ndim>2):raise ValueError('VISUAL_CACHE_FRAME_COUNT')
        self.features=None
        if config.get('condition')=='FROZEN_VISUAL':self.features=np.load(row['feature_path'],mmap_mode='r')
        self.zip=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
    def chunk(self,start,end):
        a={k:torch.from_numpy(np.array(v[start:end] if v.ndim>2 else np.broadcast_to(v,(end-start,*v.shape)),copy=True)) for k,v in self.arrays.items()}
        if self.features is not None:a['features']=torch.from_numpy(np.array(self.features[start:end],copy=True)).float()
        return a
    def native_gt(self,i):return mask_array(self.zip.read(self.row['mask_members'][i][1]))[1] if self.zip else np.zeros(self.row['source_hw'],bool)
    def close(self):
        if self.zip:self.zip.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=['review','prepare'],required=True);a=p.parse_args()
    c=json.loads(a.config.read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    original_previews(c,out) if a.phase=='review' else prepare(c,out)


if __name__=='__main__':main()
