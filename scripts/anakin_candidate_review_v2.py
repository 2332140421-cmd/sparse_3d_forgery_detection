"""Register human evidence and prepare <=12 new source-isolated local candidates.

CPU CLI only. Reuses existing media, mask, camera and HTML helpers; outputs a
next-batch review and ten-frame boundary page. No model/frontend operations.
"""
import argparse
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import av
import numpy as np
from PIL import Image
from scripts import anakin_candidate_review as prior
from scripts.prepare_geometry_screen import save,csvsave,camera_check
from scripts.prepare_v8_assets import digest,verify
from scripts.review_anakin import decode,numeric_members,mask_array
from scripts.review_pvst_scenes import overlay,sized,write_video

DATA=prior.DATA;DERIVED=prior.DERIVED;A=DATA/'secondary_datasets/anakin'
PREVIOUS=prior.OUT;OUT=PREVIOUS.with_name('anakin_pvst_candidate_review_v2')
HUMAN=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/reports/human_review.json')
MAIN={'an:40y1x9','an:95qvg9','an:9l8w54','an:7e2e57','an:95q8g9'}
PENDING={'an:7ogo04','an:95qlg9'}


def read(p):return json.loads(p.read_text())


def group(pair):
    return 'main_candidate' if pair in MAIN else 'attached_object_reference' if pair=='an:4rvro9' else 'pending_verification' if pair in PENDING else 'moving_camera_reference'


def object_candidate(task):
    t=task.lower()
    # Targeted metadata queue, not a visual/model eligibility judgment.
    return any(w in t for w in ['remove','delete','add people','add a flying butterfly']) and not any(w in t for w in ['tattoo','necklace','badge','brooch','moving camera','shadow from the left hand'])


def source_role(roles):
    if any('test' in r for r in roles):return 'excluded_test'
    mapped={'dev' if r in ['dev','validation','val'] else r for r in roles}
    if 'train' in mapped and 'dev' in mapped:return 'excluded_role_conflict'
    return 'dev' if 'dev' in mapped else 'train' if 'train' in mapped else 'candidate_unassigned'


def register_human(old):
    assert HUMAN.exists(),'Expected user JSON missing; prompt fallback must be explicitly recorded instead'
    raw=read(HUMAN);assert {r['pair'] for r in raw}==set(read(PREVIOUS/'selection.json')['selected'])
    shutil.copyfile(HUMAN,OUT/'human_review_original.json')
    records=[]
    for r in raw:
        pair=r['pair'];original=next(x for x in old if x['pair_id']==pair)
        records.append({'pair_id':pair,'original_role':original['split'],'raw_camera':r['camera'],'raw_edit':r['edit'],'raw_mask':r['mask'],'group':group(pair),'human_semantic_mask':'corresponds' if r['mask']=='对应' else 'does_not_correspond','technical_alignment':original['mask']['status'],'source':str(HUMAN),'note':'human observation, not 3D temporal anomaly ground truth'})
    save(OUT/'human_review_registered.json',{'source':str(HUMAN),'source_sha256':digest(HUMAN),'records':records,'main_train_sources':3,'main_dev_sources':2,'not_scale_sufficient':True});csvsave(OUT/'human_review_registered.csv',records)


def inventory(old):
    meta=list(csv.DictReader((A/'references/metadata.csv').open()));byid={r['video-id']:r for r in meta};audit=read(A/'metadata_audit.json');info=read(A/'references/dataset_info.json')
    entries={e['rfilename']:{'size':e['size'],'lfs':e.get('lfs'),'blob_id':e['blobId']} for e in info['siblings']}
    assert verify(A/'references/metadata.csv',entries['metadata.csv'])['status']=='VERIFIED'
    test_urls={byid[i]['url'] for i in audit['config_splits']['has-masks']['test']}
    old_sources={r['source_id'] for r in old};roles=defaultdict(set);provenance=defaultdict(list);files={}
    for path in sorted(DERIVED.glob('*/manifest.json')):
        try:m=read(path)
        except (ValueError,OSError):continue
        for r in m.get('rows',[]):
            if r.get('dataset')=='ANAKIN' and r.get('source_id') and r.get('split'):
                roles[r['source_id']].add(r['split']);provenance[r['source_id']].append({'manifest':str(path),'role':r['split']})
        files[str(path)]=digest(path)
    excluded=[];pool=[];seen=set()
    for r in sorted(meta,key=lambda x:x['video-id']):
        sid='anakin:'+r['url'];id=r['video-id'];reason=None;role=source_role(roles[sid])
        if sid in old_sources:reason='source already in original20'
        elif r['url'] in test_urls or role=='excluded_test':reason='official or registered reserved test source; metadata only'
        elif role=='excluded_role_conflict':reason='registered train/dev role conflict; no reassignment'
        elif r['has-masks']!='TRUE' or not object_candidate(r['task']):reason='not targeted object edit metadata tier'
        elif not all((A/f'{k}/{id}.{ext}').is_file() for k,ext in [('trimmed','mp4'),('edited','mp4'),('masks','zip')]):reason='required local triplet missing; not downloaded because complete local pool suffices'
        elif sid in seen:reason='same source duplicate edit'
        if reason:excluded.append({'id':id,'source_id':sid,'reason':reason});continue
        seen.add(sid);pool.append({'metadata':r,'source_id':sid,'role':role,'role_provenance':provenance[sid]})
    save(OUT/'metadata_inventory.json',{'official_revision':info['sha'],'metadata_sha256':digest(A/'references/metadata.csv'),'role_manifest_sha256':files,'excluded':excluded,'complete_local_pool':[p['metadata']['video-id'] for p in pool]})
    return pool,entries,info


def prescreen(pool,entries):
    for p in pool:
        r=p['metadata'];id=r['video-id'];path=A/'trimmed'/f'{id}.mp4'
        assert verify(path,entries[f'trimmed/{id}.mp4'])['status']=='VERIFIED'
        # Exact complete decode count supplies deterministic uniform frame indexes.
        # Only five RGB stills are retained for conventional background matching.
        with av.open(str(path)) as c:n=sum(1 for _ in c.decode(video=0))
        keep=set(np.linspace(0,n-1,5,dtype=int).tolist());m,images=decode(path,keep)
        p['real_meta']=m;p['camera_auto']=camera_check(sorted(images.items()));print('camera',id,p['role'],p['camera_auto']['state'],flush=True)
    rng=random.Random(20260909);order=sorted(p['metadata']['video-id'] for p in pool);rng.shuffle(order);tie={id:i for i,id in enumerate(order)}
    key=lambda p:(not p['camera_auto']['stable_candidate'],p['camera_auto']['state'].startswith('可见'),p['role']!='train',tie[p['metadata']['video-id']])
    train=sorted([p for p in pool if p['role']!='dev'],key=key);dev=sorted([p for p in pool if p['role']=='dev'],key=key)
    selected=train[:9]+dev[:3]
    if len(selected)<12:selected += [p for p in train[9:]+dev[3:] if p not in selected][:12-len(selected)]
    save(OUT/'selection.json',{'seed':20260909,'rules':'source exclusions -> local complete triplets -> metadata object tier -> stable background tier -> preserve registered role, up to9 train/unassigned plus3 dev; ties fixed sorted-ID shuffle; no score, GT area or visible anomaly selection','selected':[p['metadata']['video-id'] for p in selected],'prescreen':pool})
    return selected


def prepare(p,entries,revision):
    row=p['metadata'];id=row['video-id'];media={};rr={}
    for role,folder in [('real','trimmed'),('fake','edited')]:
        path=A/folder/f'{id}.mp4';check=verify(path,entries[f'{folder}/{id}.mp4']);assert check['status']=='VERIFIED'
        m=p['real_meta'] if role=='real' else decode(path,set())[0]
        assert len(m['sizes_hw'])==1 and all(x['seconds'] is not None for x in m['pts'])
        assert all(b['seconds']>a['seconds'] for a,b in zip(m['pts'],m['pts'][1:]))
        with av.open(str(path)) as c:codec=c.streams.video[0].codec_context.name
        media[role]={'path':str(path),'sha256':check['sha256'],'video_id':f'anakin:{id}:{role}','codec':codec,**m}
        rr[role]={'video_path':str(path),'mask_zip':str(A/'masks'/f'{id}.zip'),'mask_sha256':verify(A/'masks'/f'{id}.zip',entries[f'masks/{id}.zip'])['sha256'],'source_hw':list(m['sizes_hw'][0])}
    assert verify(A/'masks'/f'{id}.zip',entries[f'masks/{id}.zip'])['status']=='VERIFIED'
    a,b=media['real'],media['fake'];clock=a['frame_count']==b['frame_count'] and all(abs(x['seconds']-y['seconds'])<1e-8 for x,y in zip(a['pts'],b['pts']))
    result={'pair_id':'an:'+id,'split':p['role'],'source_id':p['source_id'],'source_url':row['url'],'official_revision':revision,'official_task':row['task'],'generator':row['manipulation-type'],'edit_object':'运动人物/动物/其他对象增删（按官方文字，画面待确认）','priority':'新增对象编辑主组候选，未人工批准','relation_note':'运动衔接、形态或遮挡是否值得比较，交人工确认；不要求先看到不协调','human':None,'camera_auto':p['camera_auto'],'camera':p['camera_auto']['state']+'（自动预筛；尚未人工确认）','role_provenance':p['role_provenance'],'media':media,'mask':prior.mask_audit(rr['fake'],media['fake']),'equal_frame_pts':clock,'sync_note':'同源帧数/PTS网格一致，不证明像素内容配准' if clock else '真伪时间网格或长度不同，只按片内相对时间联动，不作严格逐帧配准','human_semantic_alignment':'PENDING'}
    result['mask']['path']=rr['fake']['mask_zip']
    if result['mask']['status']!='PASS':
        with zipfile.ZipFile(rr['fake']['mask_zip']) as z:result['mask']['published_mask_frames']=len(numeric_members(z.namelist()))
        result['mask']['decoded_fake_frames']=media['fake']['frame_count']
        result['mask']['note']='Published video/mask frame mismatch; no repair or overlay. Semantic qualification unknown.'
    prior.preview(result,rr)
    save(OUT/result['pair_id'].replace(':','_')/'identity.json',result)
    return result


def boundary(old):
    r=next(x for x in old if x['pair_id']=='an:40y1x9');folder=OUT/'40y1x9_boundary';folder.mkdir()
    path=Path(r['media']['fake']['path']);maskpath=Path(r['mask']['path']);keep=set(range(3,13));m,images=decode(path,keep);records=[];clips=[];pts=[]
    assert m['frame_count']==150
    with zipfile.ZipFile(maskpath) as z:
        members=numeric_members(z.namelist())
        for i in range(3,13):
            rgb=np.array(images[i]);gt=mask_array(z.read(members[i][1]))[1];p=m['pts'][i]['seconds'];assert gt.shape==rgb.shape[:2]
            for name,a in [('rgb',rgb),('mask',np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)),('overlay',overlay(rgb,gt))]:Image.fromarray(sized(a)).save(folder/f'{name}_{i:03d}.png')
            records.append({'frame_zero_based':i,'pts_s':p,'mask_pixels':int(gt.sum()),'mask_nonempty':bool(gt.any())});clips.append(sized(overlay(rgb,gt)));pts.append(p)
    assert all(not records[i-3]['mask_nonempty'] for i in range(6,10)) and records[5-3]['mask_nonempty'] and records[10-3]['mask_nonempty']
    write_video(folder/'boundary_overlay.mp4',clips,pts,m['average_rate']);save(folder/'frames.json',records)
    h=['<!doctype html><meta charset="utf-8"><title>40y1x9 边界待确认</title><h1>40y1x9：mask状态变化，内容边界待确认</h1><p>零基帧3–12；原始PTS逐帧列出。空档[6,10)不自动代表实际编辑起止。下方图像仅预览，帧号/PTS文字不写入原始输入。</p><video controls preload="none" src="boundary_overlay.mp4"></video>']
    for r in records:
        i=r['frame_zero_based'];h.append(f'<h3>帧{i}，实际PTS {r["pts_s"]:.6f}s，mask像素{r["mask_pixels"]}</h3>'+''.join(f'<img style="width:32%" src="{name}_{i:03d}.png">' for name in ['rgb','mask','overlay']))
    h.append('<p>请确认：空mask时狗是否仍有编辑；6/10的切换是否与画面对应。独立确认前不得作为内容编辑起止真值。</p><a href="../index.html">返回新候选</a>');(folder/'index.html').write_text('\n'.join(h))


def main(continue_existing=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic()
    if OUT.exists() and not continue_existing:raise RuntimeError('Existing output preserved; inspect before rerun')
    if not OUT.exists():OUT.mkdir(parents=True)
    old=read(PREVIOUS/'inventory.json')
    if continue_existing:
        assert not (OUT/'summary.json').exists()
        selection=read(OUT/'selection.json');pool=selection['prescreen'];selected=[next(p for p in pool if p['metadata']['video-id']==id) for id in selection['selected']]
        info=read(A/'references/dataset_info.json');entries={e['rfilename']:{'size':e['size'],'lfs':e.get('lfs'),'blob_id':e['blobId']} for e in info['siblings']}
        results=read(OUT/'prepared_partial.json')
    else:
        register_human(old);pool,entries,info=inventory(old);selected=prescreen(pool,entries);results=[]
    prior.OUT=OUT
    for p in selected:
        if 'an:'+p['metadata']['video-id'] in {r['pair_id'] for r in results}:continue
        r=prepare(p,entries,info['sha']);results.append(r);save(OUT/'prepared_partial.json',results);print('preview',r['pair_id'],flush=True)
    save(OUT/'inventory.json',results)
    csvsave(OUT/'inventory.csv',[{'pair_id':r['pair_id'],'source_id':r['source_id'],'role':r['split'],'official_task':r['official_task'],'camera_auto':r['camera'],'mask_technical':r['mask']['status'],'mask_semantic':'PENDING_HUMAN','time_profile':r['mask']['profile']} for r in results])
    prior.render(results,results);boundary(old)
    page=OUT/'index.html';text=page.read_text().replace('首批最多12对','本轮新增最多12对').replace('<h1>ANAKIN候选：人工确认前，不冻结新训练划分</h1>','<h1>ANAKIN第二批新增候选：等待人工确认</h1><p><a href="human_review_registered.json">首批人工结论（无需重填）</a> / <a href="40y1x9_boundary/index.html">40y1x9帧3–12边界查看</a>。首批主组仅3训练来源＋2开发来源，规模尚不充分。</p>');page.write_text(text)
    summary={'status':'COMPLETE','exit_code':0,'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'human_registered':12,'main_sources':5,'main_train':3,'main_dev':2,'prescreen_sources':len(pool),'new_preview_pairs':len(results),'technical_pass_pairs':sum(r['mask']['status']=='PASS' for r in results),'technical_pending':[r['pair_id'] for r in results if r['mask']['status']!='PASS'],'new_independent_sources':len({r['source_id'] for r in results}),'selected':[r['pair_id'] for r in results],'roles':{role:sum(r['split']==role for r in results) for role in ['train','dev','candidate_unassigned']},'download_bytes':0,'seconds':time.monotonic()-start,'models':0,'depth':0,'features':0,'boundary_state':'mask state change; content boundary pending'}
    save(OUT/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continue-existing',action='store_true');args=parser.parse_args();main(args.continue_existing)
