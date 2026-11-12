"""Three fixed GRIP rigid/flipTemp asset controls, with unmapped human batch.

CLI caller only. Reuses the previous CPU decode and preview functions; adds codec,
PTS and known-GT-excluded background comparison. Pixel differences never create GT.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import html
import json
import os
from pathlib import Path
import subprocess
import time

import av
import numpy as np
from PIL import Image
import requests
from scipy.io import loadmat
from scripts.prepare_geometry_screen import ROOT,save,read,csvsave,overlay,ranges
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import decode

OUT=ROOT/'temporal_controls_v1'
GROUPS=[('04','Rolling Can'),('10','Tennis Ball'),('02','Fast Car')]


def download():
    plan=read(OUT/'download_plan.json');start=time.monotonic();records=[]
    def fetch(row):
        p=OUT/'assets'/row['name'];p.parent.mkdir(exist_ok=True)
        if p.exists():assert p.stat().st_size==row['size'];return {**row,'status':'REUSED','sha256':digest(p),'new_bytes':0}
        part=p.with_suffix(p.suffix+'.part');offset=part.stat().st_size if part.exists() else 0
        with requests.get(row['url'],stream=True,headers={'Range':f'bytes={offset}-'} if offset else {},timeout=(30,90)) as response:
            response.raise_for_status()
            if offset:assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
            with part.open('ab' if offset else 'xb') as f:
                for b in response.iter_content(1024*1024):
                    if b:f.write(b)
        assert part.stat().st_size==row['size'];part.rename(p)
        return {**row,'status':'DOWNLOADED','sha256':digest(p),'new_bytes':row['size']-offset}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,r):r for r in plan['files']}
        for future in as_completed(futures):
            try:r=future.result()
            except Exception as e:r={**futures[future],'status':'FAILED','error':repr(e)}
            records.append(r);save(OUT/'download_result.json',{'status':'RUNNING','files':records,'seconds':time.monotonic()-start});print(r['name'],r['status'],flush=True)
    save(OUT/'download_result.json',{'status':'COMPLETE' if all(r['status']!='FAILED' for r in records) else 'PARTIAL','files':records,'new_bytes':sum(r.get('new_bytes',0) for r in records),'seconds':time.monotonic()-start})


def metadata(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0];codec=s.codec_context
        return {'codec':codec.name,'pixel_format':codec.format.name,'codec_tag':codec.codec_tag,'profile':codec.profile,'width':s.width,'height':s.height,'rate':str(s.average_rate),'time_base':str(s.time_base),'container':c.format.name,'file_bytes':path.stat().st_size,'sha256':digest(path)}


def compare_known_background(a,b,mask):
    assert a.shape==b.shape and a.shape[:2]==mask.shape
    d=np.abs(a.astype(np.int16)-b.astype(np.int16));outside=~mask
    if not outside.any():return {'pixels':0,'nonzero_pixels':0,'max_abs':None,'mean_abs':None,'p95_abs':None}
    q=d[outside]
    return {'pixels':int(outside.sum()),'nonzero_pixels':int(np.count_nonzero(np.any(q!=0,axis=-1))),'max_abs':int(q.max()),'mean_abs':float(q.mean()),'p95_abs':float(np.quantile(q,.95))}


def timelines_equal(metas):
    a=metas[0]
    return all(m['frame_count']==a['frame_count'] and m['sizes_hw']==a['sizes_hw'] and len(m['pts'])==len(a['pts']) and all(x['seconds'] is not None and y['seconds'] is not None and abs(x['seconds']-y['seconds'])<1e-8 for x,y in zip(a['pts'],m['pts'])) for m in metas[1:])


def stream_checks(paths,gt_r,gt_f):
    """Compare all decoded frames outside supplied union labels, no new mask."""
    records=[];cs=[av.open(str(p)) for p in paths]
    try:
        decoders=[c.decode(c.streams.video[0]) for c in cs]
        for i,(o,r,f) in enumerate(zip(*decoders)):
            a,b,c=[x.to_ndarray(format='rgb24') for x in [o,r,f]];exclude=gt_r[:,:,i].astype(bool)|gt_f[:,:,i].astype(bool)
            record={'frame_index':i,'rigid_vs_flip':compare_known_background(b,c,exclude),'original_vs_rigid':compare_known_background(a,b,exclude),'original_vs_flip':compare_known_background(a,c,exclude),'excluded_known_union_pixels':int(exclude.sum())};records.append(record)
    finally:
        for c in cs:c.close()
    return records


def eligibility(coding_same,time_aligned,gt_aligned,background_equal,local_operation_documented):
    reasons=[]
    if not coding_same:reasons.append('原片/rigid/flipTemp编码或像素格式不同，存在编码混杂')
    if not time_aligned:reasons.append('完整帧数/原生尺寸/PTS未全部对齐')
    if not gt_aligned:reasons.append('官方GT维度或时间轴未核实')
    if not background_equal:reasons.append('两伪造版在官方并集GT外仍有实际像素差异，不能确认只改变局部时间关系')
    if not local_operation_documented:reasons.append('作者局部复制内容反转说明或对应官方文件身份尚未核实')
    return {'strict_temporal_control':not reasons,'qualification':'可用于严格时序对照' if not reasons else '只作定性观察','reasons':reasons,'target_only_pixel_GT':False,'note':'严格资格针对本轮仅局部时间变化的主张；并集GT始终不能作为实际编辑目标像素GT'}


def prepare():
    started=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    batch=read(OUT/'human_review_batch.json');assert batch['reviewed_items']==4 and batch['sample_ids'] is None
    csvsave(OUT/'human_review_batch.csv',[{'reviewed_items':4,'fixed_camera_count':3,'slight_camera_count':1,'edit':batch['edit'],'coherence':batch['coherence'],'sample_ids':'待用户补充','slight_movement_sample':'待用户补充','retention':'自然编辑样本，未确认三维时序异常'}])
    results=[]
    for number,name in GROUPS:
        folder=OUT/number;folder.mkdir(exist_ok=True)
        paths=[ROOT/'grip'/f'VIDEO_ORI_{number}.avi',ROOT/'grip'/f'VIDEO_FORG_rigid_{number}.avi',OUT/'assets'/f'VIDEO_FORG_flipTemp_{number}.avi']
        masks=[ROOT/'grip'/f'GT_rigid_{number}.mat',OUT/'assets'/f'GT_flipTemp_{number}.mat']
        if not all(p.exists() for p in paths+masks):results.append({'source_id':number,'name':name,'status':'MISSING','missing':[str(p) for p in paths+masks if not p.exists()],'qualification':'只作定性观察（资产未齐）'});continue
        known={x['path']:x['sha256'] for x in read(ROOT/'asset_identity.json')}
        for old in paths[:2]+masks[:1]:assert digest(old)==known[str(old.relative_to(ROOT))]
        gts=[];maskrecords=[]
        for p in masks:
            d={k:v for k,v in loadmat(p).items() if not k.startswith('__')};assert set(d)=={'GT'};g=d['GT'];assert g.ndim==3 and set(np.unique(g))<={0,1};gts.append(g);maskrecords.append({'path':str(p),'shape_HWT':list(g.shape),'values':np.unique(g).tolist(),'sha256':digest(p),'semantics':'复制源/克隆并集，源与目标未分离'})
        n=gts[0].shape[-1];fixed=sorted(set(np.linspace(0,n-1,5,dtype=int).tolist()));metas=[];images=[];codecs=[]
        for role,p in zip(['original','rigid','flipTemp'],paths):
            m,im=decode(p,set(fixed));metas.append(m);images.append(im);codecs.append(metadata(p));save(folder/f'{role}_identity.json',{'path':str(p),'decode':m,'media':codecs[-1]})
        aligned=all(g.shape==(metas[j+1]['sizes_hw'][0][0],metas[j+1]['sizes_hw'][0][1],metas[j+1]['frame_count']) for j,g in enumerate(gts)) and gts[0].shape==gts[1].shape
        clock=timelines_equal(metas);coding=len({(r['codec'],r['pixel_format'],r['profile']) for r in codecs})==1
        checks=stream_checks(paths,*gts) if aligned and clock else []
        background_equal=bool(checks) and all(r['rigid_vs_flip']['nonzero_pixels']==0 for r in checks)
        save(folder/'background_checks.json',{'no_GT_created':True,'excluded_mask':'provided GT_rigid OR GT_flipTemp; still includes source and target','records':checks})
        # Author definition + the fixed official variant identities document
        # the operation. Do not demand an obvious visual anomaly as an entry gate.
        # Exact source/target clock mapping is separately marked unavailable.
        qualified=eligibility(coding,clock,aligned,background_equal,True)
        for i in fixed:
            for role,im,g in [('original',images[0][i],None),('rigid',images[1][i],gts[0][:,:,i]),('flipTemp',images[2][i],gts[1][:,:,i])]:
                im.save(folder/f'{role}_{i:04d}.png')
                if g is not None:
                    Image.fromarray(g*255).save(folder/f'{role}_mask_{i:04d}.png')
                    if aligned:overlay(im,g.astype(bool)).save(folder/f'{role}_overlay_{i:04d}.png')
        preview=folder/'flipTemp.mp4'
        if not preview.exists():subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(paths[2]),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-vsync','0',str(preview)],check=True)
        pm,_=decode(preview,set());assert pm['frame_count']==n
        record={'source_id':number,'name':name,'status':'READY','both_treatments_are_fake':True,'same_source':'official identical numeric source ID; original SHA reused from previous asset identity','source_paths':[str(p) for p in paths],'media':codecs,'frames':[m['frame_count'] for m in metas],'full_pts_aligned':clock,'coding_same_all_three':coding,'GT_aligned':aligned,'mask_semantics':maskrecords,'target_only_GT_available':False,'local_flip_author_definition':'copy-moved clones can be played backwards in time; not a whole-original-video real/fake contrast','local_operation_exact_target_frame_mapping':'not released in GT; published operation provenance confirmed, exact per-target correspondence not recovered','operation_provenance':'author paper describes cloned content played backwards; only fixed official flipTemp and its GT downloaded; visible incoherence not an acceptance criterion','background_identical_outside_GT_union':background_equal,'background_different_frames':sum(r['rigid_vs_flip']['nonzero_pixels']>0 for r in checks),'fixed_frames':fixed,'fixed_fake_PTS':{str(i):[m['pts'][i]['seconds'] for m in metas] for i in fixed},'nonempty_mask_runs':{role:ranges([i for i in range(g.shape[-1]) if g[:,:,i].any()]) for role,g in zip(['rigid','flipTemp'],gts)},**qualified}
        save(folder/'record.json',record);results.append(record);print(number,'clock',clock,'codecs',[r['codec'] for r in codecs],'background_diff_frames',record['background_different_frames'],record['qualification'],flush=True)
    csvrows=[]
    for r in results:
        csvrows.append({'source_id':r['source_id'],'name':r['name'],'same_source':r.get('same_source','未核实'),'frames':str(r.get('frames')),'clock_aligned':r.get('full_pts_aligned'), 'codec_original_rigid_flip':'; '.join(x['codec']+'/'+x['pixel_format'] for x in r.get('media',[])),'background_identical_outside_union':r.get('background_identical_outside_GT_union'),'target_mask_separated':False,'qualification':r['qualification'],'reasons':'; '.join(r.get('reasons',r.get('missing',[])))})
    csvsave(OUT/'asset_controls.csv',csvrows);save(OUT/'summary.json',{'status':'PREVIEW_READY','execution_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'results':results,'strict_groups':sum(r.get('strict_temporal_control',False) for r in results),'human_review_mapping':'pending; no per-sample assignment','seconds':time.monotonic()-started,'models_run':0,'frontend_calls':0})
    render(results,batch)


def render(results,batch):
    cards=['<!doctype html><meta charset="utf-8"><title>GRIP rigid / local temporal flip</title><style>body{font:16px sans-serif;max-width:1700px;margin:24px}.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}video,img{width:100%}section{border-top:2px solid #aaa;padding-top:15px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}</style><h1>同源普通复制 vs 局部时间反转</h1><p>rigid和flipTemp都是伪造，原片仅为来源参照。无模型成绩或热图，播放器独立，未强制同步。官方GT是复制源／克隆并集，不是单独的实际编辑目标mask。</p><h2>已写回人工批次审阅（ID待补）</h2>', '<p>'+html.escape(batch['verbatim'])+'</p><p>保留为自然编辑样本，未确认为三维时序异常。四条编号及轻微移动对应项未知，没有依据顺序或自动机位结果分配到下方样本。<a href="human_review_batch.json">批次原样记录</a> / <a href="human_review_batch.csv">批次CSV</a></p>']
    for r in results:
        num=r['source_id'];cards.append(f'<section><h2>{num} {html.escape(r["name"])}</h2><p>{r["qualification"]}</p>')
        if r['status']!='READY':cards.append('<p>'+html.escape(str(r))+'</p></section>');continue
        cards.append('<p>'+html.escape('; '.join(r['reasons']))+'</p><p>编码（原片/rigid/flip）：'+html.escape('; '.join(x['codec']+'/'+x['pixel_format'] for x in r['media']))+f'。帧数{r["frames"]}；完整PTS对齐={r["full_pts_aligned"]}；两变体GT并集外像素完全相同={r["background_identical_outside_GT_union"]}。差异统计仅核查已有标签外背景，不生成新GT。</p>')
        links=' / '.join(f'<a href="{html.escape(os.path.relpath(path,OUT))}">{label}原AVI</a>' for label,path in zip(['原片','rigid','flipTemp'],r['source_paths']))
        cards.append('<p>'+links+'；下方H264只是兼容预览，编码核验依据原AVI，不依据预览副本。</p>')
        videos=[f'../preview/grip_{num}/real.mp4',f'../preview/grip_{num}/fake.mp4',f'{num}/flipTemp.mp4'];cards.append('<div class="row">')
        for label,v in zip(['原片来源（既有兼容预览）','rigid普通复制（伪造）','flipTemp局部反转（伪造）'],videos):cards.append(f'<div>{label}<video controls preload="none" src="{v}"></video></div>')
        cards.append('</div><p>下列固定0/25/50/75/100%帧按各自原生帧号与PTS展示。即使PTS相同，也不代表运动物体应相同；不把不同步视频标成同一内容时刻。</p>')
        for i in r['fixed_frames']:
            pts=r['fixed_fake_PTS'][str(i)];cards.append(f'<h3>固定帧{i}；原片/rigid/flip PTS={pts}秒</h3><div class="row">')
            for role,label in [('original','原片'),('rigid','rigid：RGB/并集GT'),('flipTemp','flipTemp：RGB/并集GT')]:
                file=f'{num}/{role}_{i:04d}.png' if role=='original' else f'{num}/{role}_overlay_{i:04d}.png';cards.append(f'<div>{label}<a href="{file}"><img loading="lazy" src="{file}"></a>')
                if role!='original':cards.append(f'<a href="{num}/{role}_mask_{i:04d}.png">官方原GT二值显示</a> / <a href="{num}/{role}_{i:04d}.png">无叠加RGB</a>')
                cards.append('</div>')
            cards.append('</div>')
        cards.append(f'<p><a href="{num}/record.json">身份/完整帧记录</a> / <a href="{num}/background_checks.json">背景核对</a></p></section>')
    cards.append('<p><a href="report.md">中文报告</a> / <a href="asset_controls.csv">三组资产表</a>。本轮不筛掉自然协调样本，不冻结新实验。</p>');(OUT/'index.html').write_text('\n'.join(cards))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download-only',action='store_true');args=parser.parse_args()
    if args.download_only:download()
    else:prepare()
