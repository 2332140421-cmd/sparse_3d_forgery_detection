"""Bounded CPU review of the registered ANAKIN train16/dev4 source union.

CLI only: exact media/mask identity, simple existing background matching and
12-pair HTML. No geometry, model, training, score or download entry is imported.
"""
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import av
import cv2
import numpy as np
from PIL import Image
from scripts.review_anakin import decode,numeric_members,alignment,mask_array
from scripts.review_pvst_scenes import overlay,sized,write_video
from scripts.prepare_geometry_screen import camera_check,save,csvsave,ranges
from scripts.prepare_v8_assets import digest

DATA=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection')
DERIVED=DATA/'derived'
OUT=DATA/'secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_candidate_review_v1'
BASE=DERIVED/'pvst_rgb_partition_anakin_dev_v1/manifest.json'
SCALE=DERIVED/'pvst_rgb_partition_anakin_scale_v1/manifest.json'
# Prior user oral review: explicitly identified four ANAKIN pairs. This is NOT
# the separate unmapped four-item natural-edit batch (which stays unmapped).
HUMAN={
 'an:7e2e57':{'camera':'固定','edit':'女人牵狗走向镜头，新增猫从前方经过；猫与背景融合不自然，痕迹明显'},
 'an:4rvro9':{'camera':'固定','edit':'给说话并做手势的男人加帽子；帽子随人物移动，略不协调，痕迹不明显'},
 'an:4wgx37':{'camera':'明显跟随主体','edit':'移除滑板男子及滑板并补背景；效果较真实，有轻微伪影'},
 'an:95q8g9':{'camera':'固定','edit':'移除左侧骑马男子并填背景；效果较真实，仅有细微不协调'}}
# Semantic inventory from the frozen descriptions, not prediction/GT area.
PRIORITY={'an:7ogg04':'人物移除','an:95qvg9':'背景行人移除','an:4wdo37':'奔跑持球人物移除','an:40y1x9':'奔跑动物移除','an:7ogo04':'人物移除','an:4wgm37':'雕塑反射中的多人物移除','an:9l8w54':'行走人物移除','an:95qlg9':'与球交互的人物移除','an:7e2e57':'行走动物增加','an:4wgx37':'运动主体移除','an:95q8g9':'骑手/马匹移除（官方描述）；人工画面描述另列'}


def read(p):return json.loads(p.read_text())


def merge_rows(a,b):
    merged={};source_split={}
    for r in a+b:
        key=(r['pair_id'],r['role'])
        if key in merged:
            old=merged[key]
            for f in ['split','source_id','video_sha256','video_id','actual_frames','source_hw','pts']:
                assert old[f]==r[f],(key,f)
        else:merged[key]=r
        if r['source_id'] in source_split:assert source_split[r['source_id']]==r['split'],'SOURCE_SPLIT_CONFLICT'
        source_split[r['source_id']]=r['split']
    pairs={}
    for (p,role),r in merged.items():pairs.setdefault(p,{})[role]=r
    assert all(set(v)=={'real','fake'} and v['real']['source_id']==v['fake']['source_id'] for v in pairs.values())
    assert len({v['real']['source_id'] for v in pairs.values()})==len(pairs),'SAME_SOURCE_MULTIPLE_PAIRS'
    return pairs


def mask_audit(row,meta):
    path=Path(row['mask_zip'])
    if not path.is_file():return {'status':'MISSING','profile':'unknown','reason':str(path)}
    assert digest(path)==row['mask_sha256'],'MASK_SHA'
    with zipfile.ZipFile(path) as z:
        members=numeric_members(z.namelist());match=alignment([i for i,_ in members],meta['frame_count'])
        if match['status']!='aligned':return {'status':'UNRESOLVED','profile':'unknown','alignment':match}
        areas=[]
        for num,name in members:
            a,gt,_=mask_array(z.read(name));assert list(gt.shape)==row['source_hw'],(row['pair_id'],num,'MASK_SIZE')
            areas.append(int(gt.sum()))
    positive=[i for i,a in enumerate(areas) if a>0];n=len(areas)
    return {'status':'PASS','profile':'all_edited' if len(positive)==n else 'partial' if positive else 'all_empty','mask_sha256':row['mask_sha256'],'index_base':match['index_base'],'frames':n,'positive_frames':len(positive),'areas':areas,'positive_runs_half_open':ranges(positive),'transition_frames':[i for i in range(1,n) if (areas[i]>0)!=(areas[i-1]>0)],'mapping':'official mask numeric index -> decoded fake frame index; size/count checked; no independent mask PTS exists','semantics':'official edited region, not visible 3D/temporal anomaly GT'}


def audit_pair(pair,rr):
    result={'pair_id':pair,'split':rr['fake']['split'],'source_id':rr['fake']['source_id'],'official_task':rr['fake']['task'],'generator':rr['fake']['generator'],'edit_object':PRIORITY.get(pair,'配饰/外观或描述待确认'),'priority':'对象增删/运动交互优先' if pair in PRIORITY else '较低：外观/配饰；不排除研究价值','relation_note':'对象存在、局部遮挡及运动衔接可能相关，是否可学未验证' if pair in PRIORITY else '外观附着与局部形态/遮挡需人工判断；不预判无关系信息','human':HUMAN.get(pair),'human_source':'用户此前按pair明确口述；人工可见线索，不是三维异常真值' if pair in HUMAN else None,'media':{}}
    sample={}
    for role,row in rr.items():
        assert digest(Path(row['video_path']))==row['video_sha256']
        fixed=set(np.linspace(0,row['actual_frames']-1,5,dtype=int).tolist())
        m,ims=decode(Path(row['video_path']),fixed)
        assert m['frame_count']==row['actual_frames'] and m['sizes_hw']==[tuple(row['source_hw'])]
        assert all(p['seconds'] is not None and abs(p['seconds']-q['pts_s'])<1e-8 for p,q in zip(m['pts'],row['pts']))
        assert all(b['seconds']>a['seconds'] for a,b in zip(m['pts'],m['pts'][1:]))
        with av.open(row['video_path']) as c:codec=c.streams.video[0].codec_context.name
        result['media'][role]={'path':row['video_path'],'video_id':row['video_id'],'sha256':row['video_sha256'],'codec':codec,**m}
        sample[role]=ims
    result['mask']=mask_audit(rr['fake'],result['media']['fake'])
    result['mask']['path']=rr['fake']['mask_zip']
    result['official_revision']=rr['fake'].get('official_revision')
    result['source_url']=rr['fake'].get('original_url')
    result['camera_auto']=camera_check(sorted(sample['real'].items()))
    result['camera']=HUMAN[pair]['camera']+'（明确人工记录）' if pair in HUMAN else result['camera_auto']['state']
    a,b=result['media']['real'],result['media']['fake']
    result['equal_frame_pts']=a['frame_count']==b['frame_count'] and all(abs(p['seconds']-q['seconds'])<1e-8 for p,q in zip(a['pts'],b['pts']))
    result['sync_note']='按已核对的帧号/PTS网格同步；同源不等于已证明逐像素内容配准' if result['equal_frame_pts'] else '真伪帧数或PTS不同；只作相对播放时间联动，不能视为严格同帧对应；分别显示时间轴'
    return result


def choose(records):
    dev=sorted([r for r in records if r['split']=='dev'],key=lambda r:r['pair_id'])
    train=[r for r in records if r['split']=='train']
    train.sort(key=lambda r:(r['mask']['profile']!='partial',r['pair_id'] not in PRIORITY,not r['camera_auto']['stable_candidate'],r['pair_id']))
    return train[:8]+dev[:4]


def preview(r,rr):
    folder=OUT/r['pair_id'].replace(':','_');folder.mkdir()
    for role in ['real','fake']:(folder/f'{role}.mp4').symlink_to(Path(rr[role]['video_path']))
    row=rr['fake'];m=r['media']['fake'];n=m['frame_count'];mask=r['mask']
    if mask['status']!='PASS':return
    fixed=sorted(set([0,n//2,n-1]+mask['transition_frames'][:4]));fixed_images={};over=[];maskvideo=[];pts=[]
    with zipfile.ZipFile(row['mask_zip']) as z,av.open(row['video_path']) as c:
        members=numeric_members(z.namelist())
        for i,f in enumerate(c.decode(video=0)):
            rgb=f.to_ndarray(format='rgb24');gt=mask_array(z.read(members[i][1]))[1]
            over.append(sized(overlay(rgb,gt)));maskvideo.append(sized(np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)));pts.append(float(f.pts*f.time_base))
            if i in fixed:
                for label,a in [('rgb',rgb),('mask',np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)),('overlay',overlay(rgb,gt))]:Image.fromarray(sized(a)).save(folder/f'{label}_{i:04d}.png')
    write_video(folder/'overlay.mp4',over,pts,m['average_rate']);write_video(folder/'mask.mp4',maskvideo,pts,m['average_rate'])
    r['fixed_frames']=[{'index':i,'fake_pts':pts[i]} for i in fixed]
    rects=''.join(f'<rect x="{i*900/n:.4f}" y="0" width="{900/n+.1:.4f}" height="24" fill="{"#d33" if v else "#ddd"}"/>' for i,v in enumerate(mask['areas']))
    (folder/'timeline.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 24">'+rects+'</svg>')
    save(folder/'identity.json',r)


def render(records,selected):
    h=['<!doctype html><meta charset="utf-8"><title>ANAKIN PVST候选人工核验</title><style>body{font:16px sans-serif;margin:24px;max-width:1600px}section{border-top:2px solid #888;padding:18px 0}.players{display:flex;gap:12px}.players video{width:48%}img{max-width:100%}.shots img{width:31%}textarea{display:block;width:95%;margin:8px}pre{white-space:pre-wrap}</style><h1>ANAKIN候选：人工确认前，不冻结新训练划分</h1><p>不显示模型成绩或热图。红色mask是官方编辑区域，不是三维异常真值。首批最多12对，按pair ID展示；未展示者仍保留inventory.csv。同步只按已核对的片内时间，不声称逐像素配准。原始媒体不改动，叠加/掩膜MP4只作缩放预览。</p><p>人工三项：相机是否固定或近固定；具体对象及运动/遮挡/形态；mask与时间轴是否对应画面。自然协调的编辑仍保留研究价值。</p><button id="export">导出人工JSON</button><p><a href="report.md">中文报告</a> / <a href="inventory.csv">完整初筛清单</a> / <a href="selection.json">固定首批选择规则</a></p>']
    for r in sorted(selected,key=lambda x:x['pair_id']):
        p=r['pair_id'].replace(':','_');ma=r['mask'];h.append(f'<section data-pair="{r["pair_id"]}"><h2>{r["pair_id"]} · 原{r["split"]}</h2><p>官方：{html.escape(r["official_task"])}</p><p>机位：{html.escape(r["camera"])}；自动初筛仅辅助：{html.escape(r["camera_auto"]["state"])}</p><p>{html.escape(r["edit_object"])}。{html.escape(r["relation_note"])}</p>')
        if r['human']:h.append('<p>已有明确人工记录：'+html.escape(r['human']['edit'])+'（'+r['human_source']+'）。官方描述与人工描述不一致时并列保留，不改写原文。</p>')
        h.append(f'<p>标注：{ma["status"]} / {ma["profile"]}；非空{ma.get("positive_frames","未知")}/{ma.get("frames","未知")}帧。{r["sync_note"]}</p><p>real/fake PTS及身份分别保存在<a href="{p}/identity.json">完整记录</a>；mask没有独立PTS，按官方编号对应。</p><button class="toggle">同步播放/暂停</button><select class="mode"><option value="fake.mp4">fake RGB</option><option value="overlay.mp4">fake＋官方mask</option><option value="mask.mp4">仅mask</option></select><span class="time"></span><div class="players"><video class="real" controls preload="none" src="{p}/real.mp4"></video><video class="fake" controls preload="none" src="{p}/fake.mp4"></video></div>')
        if ma['status']=='PASS':
            h.append(f'<p>编辑时间轴（红=该帧mask非空；灰=空，点击跳转；非独立动作起止真值）</p><img class="timeline" data-duration="{r["media"]["fake"]["pts"][-1]["seconds"]}" src="{p}/timeline.svg"><p>非空区间[起,止)帧：{ma["positive_runs_half_open"]}；画面内对象进出仍需人工确认，不自动等同mask起止。</p>')
            for f in r.get('fixed_frames',[]):
                i=f['index'];h.append(f'<p>固定fake帧{i} / PTS {f["fake_pts"]:.6f}s</p><div class="shots">'+''.join(f'<a href="{p}/{k}_{i:04d}.png"><img loading="lazy" src="{p}/{k}_{i:04d}.png"></a>' for k in ['rgb','mask','overlay'])+'</div>')
        for name,text in [('camera','1. 相机是否符合固定或近固定视角'),('edit','2. 实际修改哪个对象，涉及什么运动、遮挡或形态'),('mask','3. mask与编辑时间轴是否对应画面')]:h.append(f'<label>{text}<textarea data-field="{name}"></textarea></label>')
        h.append('</section>')
    h.append('''<script>
for(const s of document.querySelectorAll('section')){const r=s.querySelector('.real'),f=s.querySelector('.fake'),base=f.getAttribute('src').split('/').slice(0,-1).join('/');let lock=false;
s.querySelector('.toggle').onclick=()=>{if(r.paused){f.currentTime=Math.min(r.currentTime,Math.max(0,f.duration||0));Promise.all([r.play(),f.play()]).catch(()=>{})}else{r.pause();f.pause()}};
r.onseeking=()=>{if(!lock&&Math.abs(f.currentTime-r.currentTime)>.08)f.currentTime=Math.min(r.currentTime,f.duration||r.currentTime)};
f.onseeking=()=>{if(!lock&&Math.abs(r.currentTime-f.currentTime)>.08)r.currentTime=Math.min(f.currentTime,r.duration||f.currentTime)};
r.ontimeupdate=()=>{s.querySelector('.time').textContent=`real ${r.currentTime.toFixed(3)}s / fake ${f.currentTime.toFixed(3)}s`;if(!r.paused&&Math.abs(f.currentTime-r.currentTime)>.1)f.currentTime=Math.min(r.currentTime,f.duration||r.currentTime)};
s.querySelector('.mode').onchange=e=>{let t=f.currentTime;r.pause();f.pause();f.src=base+'/'+e.target.value;f.onloadedmetadata=()=>{f.currentTime=Math.min(t,f.duration)}};
const tl=s.querySelector('.timeline');if(tl)tl.onclick=e=>{r.pause();f.pause();let t=e.offsetX/tl.clientWidth*Number(tl.dataset.duration);r.currentTime=t;f.currentTime=t};
for(const a of s.querySelectorAll('textarea')){let k='pvst-candidate-'+s.dataset.pair+'-'+a.dataset.field;try{a.value=localStorage.getItem(k)||'';a.oninput=()=>localStorage.setItem(k,a.value)}catch(e){}}}
export.onclick=()=>{let data=Array.from(document.querySelectorAll('section')).map(s=>({pair:s.dataset.pair,...Object.fromEntries(Array.from(s.querySelectorAll('textarea')).map(a=>[a.dataset.field,a.value]))}));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='human_review.json';a.click()};
</script>'''.replace('export.onclick','document.getElementById("export").onclick'))
    (OUT/'index.html').write_text('\n'.join(h))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic();cv2.setNumThreads(2)
    if OUT.exists():raise RuntimeError('Existing output preserved; verify before reuse')
    OUT.mkdir(parents=True)
    a=read(BASE);b=read(SCALE);pairs=merge_rows(a['rows'],b['rows']);assert len(pairs)==20
    save(OUT/'protocol.json',{'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'manifests':{str(p):digest(p) for p in [BASE,SCALE]},'max_pairs':12,'selection':'audit all20; keep all4dev + first8train sorted(partial-time first, object-add/remove tier, stable-auto tier, stable pair ID); never model scores','media_scope':'full published trimmed source/edited clips, not full original Vimeo long video','human_review_source':'four explicitly identified ANAKIN pair statements from prior user conversation; unmapped natural-edit four-item batch remains untouched'})
    records=[]
    for pair,rr in sorted(pairs.items()):
        try:r=audit_pair(pair,rr)
        except Exception as e:r={'pair_id':pair,'split':rr['fake']['split'],'source_id':rr['fake']['source_id'],'status':'FAILED','error':repr(e)}
        records.append(r);save(OUT/'audit_partial.json',records);print('audit',pair,r.get('mask',{}).get('profile',r.get('error')),flush=True)
    complete=[r for r in records if 'mask' in r];selected=choose(complete)
    save(OUT/'selection.json',{'selected':[r['pair_id'] for r in selected],'remaining':[r['pair_id'] for r in records if r not in selected],'rule':read(OUT/'protocol.json')['selection']})
    for r in selected:preview(r,pairs[r['pair_id']]);print('preview',r['pair_id'],flush=True)
    save(OUT/'inventory.json',records)
    rows=[{'pair_id':r['pair_id'],'source_id':r['source_id'],'split':r['split'],'official_task':r.get('official_task'),'camera':r.get('camera'),'priority':r.get('priority'),'mask_status':r.get('mask',{}).get('status'),'time_profile':r.get('mask',{}).get('profile'),'positive_frames':r.get('mask',{}).get('positive_frames'),'frames':r.get('mask',{}).get('frames'),'preview_selected':r in selected,'error':r.get('error')} for r in records];csvsave(OUT/'inventory.csv',rows)
    render(records,selected)
    summary={'status':'COMPLETE','exit_code':0,'pairs':len(pairs),'sources':len({r['source_id'] for r in records}),'train_pairs':sum(r['split']=='train' for r in records),'dev_pairs':sum(r['split']=='dev' for r in records),'preview_pairs':len(selected),'selected':[r['pair_id'] for r in selected],'annotation_failed':[r['pair_id'] for r in records if r.get('mask',{}).get('status')!='PASS'],'partial_time_pairs':[r['pair_id'] for r in complete if r['mask']['profile']=='partial'],'all_edited_pairs':[r['pair_id'] for r in complete if r['mask']['profile']=='all_edited'],'strict_equal_pts_pairs':sum(r['equal_frame_pts'] for r in complete),'download_bytes':0,'seconds':time.monotonic()-start,'models':0,'depth':0}
    save(OUT/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
