"""CPU-only scene evidence for current/candidate registered training pairs.

CLI -> verified RGB/GT/high-resolution cached geometry -> local HTML. Existing
fit previews include predictions, so cannot serve this score-blind review.
No model, runner, provider, download or CUDA calls occur here.
"""
import argparse
import csv
from fractions import Fraction
import html
import json
from pathlib import Path
import random
import time

import av
import cv2
import numpy as np

from research_tools.depth_slice.data import Video as RGBGTVideo
from research_tools.rgbd_slice.data import Video as RGBDVideo
from scripts.prepare_v8_assets import digest

BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection')
SOURCE=BASE/'derived/pvst_rgbd_slice_relations_v1'
PRIOR=BASE/'derived/pvst_set_to_set_fit_check_v1'
OUT=BASE/'derived/pvst_sample_scene_review_v1'
CURRENT=['an:7ogg04','an:728qx7']
FIXED=[0,4,8,12,16,23]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def clip_timing(pts,start,end):
    if not 0<=start<end<=len(pts):raise ValueError('CLIP_RANGE')
    selected=pts[start:end]
    if any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('PTS_NOT_INCREASING')
    # Presentation duration includes the last frame, not just first/last PTS span.
    last_step=pts[end]-pts[end-1] if end<len(pts) else pts[-1]-pts[-2]
    return {'range':[start,end],'frame_indices':list(range(start,end)),'pts_s':selected,
            'PTS_span_s':selected[-1]-selected[0],'presentation_duration_s':selected[-1]-selected[0]+last_step,
            'last_frame_duration_rule':'next original PTS, or final adjacent interval at video tail'}


def overlay(rgb,gt):
    result=rgb.copy();result[gt]=np.round(.6*rgb[gt]+.4*np.array([255,0,0])).astype(np.uint8)
    return result


def sized(rgb,width=640):
    h,w=rgb.shape[:2];factor=min(1.,width/w)
    ow=max(2,int(w*factor)//2*2);oh=max(2,int(h*factor)//2*2)
    return cv2.resize(rgb,(ow,oh),interpolation=cv2.INTER_AREA) if (h,w)!=(oh,ow) else rgb


def write_video(path,images,pts,fps):
    path.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=Fraction(fps));stream.width=images[0].shape[1];stream.height=images[0].shape[0]
        stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000000);stream.codec_context.thread_count=2
        stream.options={'crf':'21','preset':'fast'}
        for image,p in zip(images,pts):
            frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=round((p-pts[0])*1000000);frame.time_base=Fraction(1,1000000)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    # Check every preview frame and relative timestamp, not just container header.
    with av.open(str(path)) as container:
        actual=[float(f.pts*f.time_base) for f in container.decode(video=0)]
    if len(actual)!=len(pts) or any(abs(a-(p-pts[0]))>.001 for a,p in zip(actual,pts)):
        raise ValueError('PREVIEW_PTS_MISMATCH:'+str(path))


def candidates(train,metadata):
    # First metadata tier: local person/object edits explicitly describing running
    # or walking. No scene or model results enter this deterministic ordering.
    pool=[]
    for pair in sorted({r['pair_id'] for r in train}-set(CURRENT)):
        task=metadata[pair.removeprefix('an:')]['task'].lower()
        if any(word in task for word in ['running','walking']):pool.append(pair)
    random.Random(20260909).shuffle(pool)
    return pool[:2]


def media_identity(row):
    path=Path(row['video_path'])
    if digest(path)!=row['video_sha256']:raise ValueError('VIDEO_SHA:'+row['video_id'])
    pts=[];sizes=set()
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_count=2
        codec=stream.codec_context.name;pixel_format=stream.codec_context.format.name;fps=str(stream.average_rate)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('MISSING_PTS')
            pts.append(float(frame.pts*frame.time_base));sizes.add((frame.height,frame.width))
    if len(pts)!=row['actual_frames'] or sizes!={tuple(row['source_hw'])}:raise ValueError('NATIVE_IDENTITY')
    if any(abs(a-b['pts_s'])>1e-8 for a,b in zip(pts,row['pts'])):raise ValueError('MANIFEST_PTS')
    return {'codec':codec,'pixel_format':pixel_format,'fps':fps,'actual_frames':len(pts),'native_hw':row['source_hw'],
            'duration_s':clip_timing(pts,0,len(pts))['presentation_duration_s'],'pts_s':pts,'video_sha256':row['video_sha256'],
            'video_id':row['video_id'],'role':row['role'],'source_id':row['source_id'],'video_path':str(path),'full_decode':'PASS'}


def open_video(row,c):
    geom=SOURCE/'geometry'/row['video_id']/'complete.json'
    if geom.exists():
        video=RGBDVideo(row,c);meta=json.loads(geom.read_text())
        if max(meta['output_hw'])<=256:raise ValueError('LOW_RES_GEOMETRY_NOT_ALLOWED')
        return video,meta
    # Explicitly retain RGB/GT only if geometry was never prepared.
    return RGBGTVideo(row,c),None


def depth_rgb(video,number,lo,hi):
    frame=video.frame(number);depth=frame['xyz'][...,2].numpy();valid=frame['valid'].numpy()
    transform=lambda x:np.sign(x)*np.log1p(np.abs(x))
    low,high=transform(lo),transform(hi)
    display=np.clip((transform(depth)-low)/max(high-low,1e-12),0,1)
    image=cv2.applyColorMap((display*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
    image[~valid]=0
    return image


def sheet(path,images,labels,columns=3):
    tiles=[]
    for image,label in zip(images,labels):
        thumb=sized(image,320);tile=np.zeros((210,320,3),np.uint8);h,w=thumb.shape[:2]
        factor=min(1.,180/h);thumb=cv2.resize(thumb,(int(w*factor),int(h*factor)));h,w=thumb.shape[:2];tile[:h,:w]=thumb
        cv2.putText(tile,label,(4,200),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);tiles.append(tile)
    while len(tiles)%columns:tiles.append(np.zeros_like(tiles[0]))
    image=np.concatenate([np.concatenate(tiles[i:i+columns],1) for i in range(0,len(tiles),columns)],0)
    cv2.imwrite(str(path),image[:,:,::-1])


def prepare_pair(pair,rows,c,metadata,deadline):
    if time.monotonic()>deadline:raise TimeoutError('600_SECOND_MEDIA_BUDGET')
    folder=OUT/'media'/pair.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
    rr=[next(r for r in rows if r['pair_id']==pair and r['role']==role) for role in ['real','fake']]
    if any(r['source_id']!='anakin:'+metadata[pair.removeprefix('an:')]['url'] for r in rr):
        raise ValueError('PAIR_SOURCE_METADATA_CONFLICT')
    result={'pair_id':pair,'facts':{'official_metadata':metadata[pair.removeprefix('an:')],
            'split':'train','complete_original_means':'entire released trimmed/edited short video, not original Vimeo long video',
            'source_pair_not_pixel_registration':'same registered source; exact real/fake frame registration not established; independent PTS shown; no pixel differences',
            'preview_scaling':'only display/MP4 width<=640; original training sizes unchanged'},'clips':[]}
    videos=[];media=[];geometry=[]
    try:
        for r in rr:
            m=media_identity(r);media.append(m)
            if r['role']=='fake' and digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SHA')
            v,g=open_video(r,c);videos.append(v);geometry.append(g)
            link=folder/f'{r["role"]}_full.mp4'
            if not link.exists():link.symlink_to(Path(r['video_path']))
            if link.resolve()!=Path(r['video_path']).resolve():raise ValueError('ORIGINAL_LINK_CONFLICT')
        result['facts']['media']=media
        result['facts']['mask']={'path':rr[1]['mask_zip'],'sha256':rr[1]['mask_sha256'],'index_base':rr[1]['index_base'],
                'mapping':'numeric member sorting, exact coverage checked by registered loader; no resize/crop of GT; current mask_array >=.5',
                'real_GT':'legitimate zero target; inherited pair mask_zip field is provenance only, never a real per-frame mask',
                'target':'official edit region, not visible anomaly'}
        result['facts']['geometry']=[{'role':r['role'],'root':str(SOURCE/'geometry'/r['video_id']),'metadata':g} for r,g in zip(rr,geometry)]
        # Start at the actual trained clip; no content-dependent time selection.
        start,end=0,24
        ranges=[clip_timing(m['pts_s'],start,end) for m in media]
        lo,hi=np.inf,-np.inf
        for v,g in zip(videos,geometry):
            if g:
                for i in range(start,end):
                    if time.monotonic()>deadline:raise TimeoutError('MEDIA_BUDGET')
                    frame=v.frame(i);d=frame['xyz'][...,2].numpy();valid=frame['valid'].numpy()
                    if valid.any():lo=min(lo,float(d[valid].min()));hi=max(hi,float(d[valid].max()))
        depth_range=[lo,hi] if np.isfinite([lo,hi]).all() else None
        result['facts']['depth_display']={'range_normalized_Z':depth_range,'scale':c['scale'],
           'mapping':'signed log1p, common min/max from both roles all24 native mapped frames, TURBO; invalid black; not metres; no quantile clipping; input unchanged',
           'interpolation':'same native remapping as RGBD Video.frame; interpolated pixels are not independent depth measurements'}
        panels=[];panel_labels=[];clip_data=[]
        for role,v,m,timing in zip(['real','fake'],videos,media,ranges):
            images=[];overlays=[];areas=[];positive=0
            for i in range(start,end):
                if time.monotonic()>deadline:raise TimeoutError('MEDIA_BUDGET')
                rgb=np.array(v.rgb[i]);gt=v.gt(i);images.append(sized(rgb));areas.append(float(gt.mean()));positive+=bool(gt.any())
                if role=='fake':overlays.append(sized(overlay(rgb,gt)))
            write_video(folder/f'{role}_clip.mp4',images,timing['pts_s'],m['fps'])
            if role=='fake':write_video(folder/'fake_GT_overlay.mp4',overlays,timing['pts_s'],m['fps'])
            timing.update(role=role,positive_GT_frames=positive,GT_area_fraction_per_frame=areas,
                          GT_area_fraction_mean=float(np.mean(areas)))
            clip_data.append(timing)
        for i in FIXED:
            real=np.array(videos[0].rgb[i]);fake=np.array(videos[1].rgb[i]);gt=videos[1].gt(i)
            gt_image=np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)
            depth=depth_rgb(videos[1],i,*depth_range) if geometry[1] and depth_range else np.zeros_like(fake)
            components=[real,fake,gt_image,overlay(fake,gt),depth]
            for name,image in zip(['real','fake','GT','overlay','depth'],components):
                cv2.imwrite(str(folder/f'{name}_{i:03d}.png'),sized(image)[:,:,::-1])
                panels.append(image);panel_labels.append(f'{name} frame{i} PTS{media[0 if name=="real" else 1]["pts_s"][i]:.4f}')
        sheet(folder/'fixed_contact.jpg',panels,panel_labels,columns=5)
        whole=np.linspace(0,media[0]['actual_frames']-1,8,dtype=int).tolist()
        sheet(folder/'real_full_contact.jpg',[np.array(videos[0].rgb[i]) for i in whole],
              [f'real frame{i} PTS{media[0]["pts_s"][i]:.3f}' for i in whole],columns=4)
        result['clips']=clip_data;result['facts']['full_contact_indices']=whole
        result['facts']['viewing_scope']='ordered fixed stills/contact sheets only; complete video playback awaits human confirmation'
        result['preview_relative_dir']=str(folder.relative_to(OUT));result['status']='COMPLETE'
        save(OUT/'facts'/f'{pair.replace(":","_")}.json',result)
        return result
    finally:
        for v in videos:v.close()


def render(results):
    notes=json.loads((OUT/'visual_notes.json').read_text()) if (OUT/'visual_notes.json').exists() else {}
    text=['<!doctype html><meta charset="utf-8"><title>样本与研究场景核验</title><style>body{font-family:sans-serif;max-width:1500px;margin:24px auto}video{width:45%;max-height:420px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}td img{max-width:260px}pre{white-space:pre-wrap}section{border-top:2px solid #888;margin-top:24px}</style>',
          '<h1>样本与研究场景：人工快速查看</h1><p>优先基本固定机位、简单连续主体活动、局部人物／物体编辑及可靠GT。画面自然、深度平滑或模型不成功都不是排除条件。本页无模型预测/分数。</p><p>只作基于序列图的初审；没有声称已播放确认全段。完整视频请人工播放，拿不准标待定。红色为官方编辑区域，不是“可见异常”。全部缩放仅供预览，训练分辨率未变。</p>']
    csv_rows=[]
    for r in results:
        pair=r['pair_id'];d='../'+r['preview_relative_dir'];f=r['facts'];note=notes.get(pair,{})
        text += [f'<section><h2>{pair}</h2><p>{html.escape(f["official_metadata"]["manipulation-type"]+": "+f["official_metadata"]["task"])}</p>',
          '<p>source：'+html.escape(f['media'][0]['source_id'])+'；本轮只检查已登记训练样本。</p>',
          '<p>real完整发布短片／fake完整发布短片（不是源Vimeo长视频）</p>'+''.join(f'<video controls preload="none" src="{d}/{role}_full.mp4"></video>' for role in ['real','fake']),
          '<p>'+ ' ／ '.join(f'<a href="{d}/{role}_full.mp4">{role}原文件（可用VLC打开）</a>' for role in ['real','fake'])+'</p>',
          '<p>真实使用首24帧：real／fake；下方另有连续GT叠加</p>'+''.join(f'<video controls preload="none" src="{d}/{role}_clip.mp4"></video>' for role in ['real','fake']),
          f'<p>fake连续GT叠加：</p><video controls preload="none" src="{d}/fake_GT_overlay.mp4"></video>',
          '<p>同源关系已核对；未确证逐像素配准，以下为独立PTS、同原始帧索引展示，不计算像素差。</p>']
        for m,c in zip(f['media'],r['clips']):
            text.append(f'<p>{m["role"]}：{m["video_id"]}；原始{m["native_hw"]}，{m["actual_frames"]}帧，完整{m["duration_s"]:.6f}s；[0,24)实际PTS {c["pts_s"][0]:.9f}–{c["pts_s"][-1]:.9f}s，跨度{c["PTS_span_s"]:.6f}s，含末帧展示时长<b>{c["presentation_duration_s"]:.6f}s</b>；正GT {c["positive_GT_frames"]}/24，GT平均面积{c["GT_area_fraction_mean"]:.6%}。</p>')
            text.append('<details><summary>'+m['role']+'的全部原始帧索引／实际PTS</summary><pre>'+html.escape('\n'.join(f'{i}: {p:.9f} s' for i,p in zip(c['frame_indices'],c['pts_s'])))+'</pre></details>')
        text += ['<p>深度来自已有高分辨率geometry；两role共用signed log1p/TURBO范围 '+html.escape(str(f['depth_display']['range_normalized_Z']))+'（Z/固定s，非米）。无分位截断；无效点黑色，仅显示，不改输入。</p>',
                 '<p>几何状态：'+html.escape('；'.join(g['role']+(' 已有缓存' if g['metadata'] else ' 未准备几何，黑色占位不是观测') for g in f['geometry']))+'</p>',
                 '<table><tr><th>frame / 独立PTS</th><th>real</th><th>fake原图</th><th>官方GT（当前二值化）</th><th>GT叠加</th><th>已有几何深度</th></tr>']
        for i in FIXED:
            text.append(f'<tr><td>{i}<br>real {f["media"][0]["pts_s"][i]:.9f}<br>fake {f["media"][1]["pts_s"][i]:.9f}</td>'+''.join(f'<td><a href="{d}/{name}_{i:03d}.png"><img loading="lazy" src="{d}/{name}_{i:03d}.png"></a></td>' for name in ['real','fake','GT','overlay','depth'])+'</tr>')
        text+=['</table><details><summary>完整real等间隔8帧初审图（不代替播放）</summary>'+f'<img src="{d}/real_full_contact.jpg"></details>',
               '<h3>初步画面描述／待人工确认</h3><pre>'+html.escape(json.dumps(note or {'status':'待人工确认'},ensure_ascii=False,indent=2))+'</pre></section>']
        csv_rows.append({'pair_id':pair,'official_edit':f['official_metadata']['task'],'fact_split':'train','fact_real_duration':f['media'][0]['duration_s'],
            'fact_fake_duration':f['media'][1]['duration_s'],'fact_real_clip_duration':r['clips'][0]['presentation_duration_s'],
            'fact_fake_clip_duration':r['clips'][1]['presentation_duration_s'],'fact_positive_GT_frames':r['clips'][1]['positive_GT_frames'],
            'fact_GT_mean_area':r['clips'][1]['GT_area_fraction_mean'],'visual_preliminary':json.dumps(note,ensure_ascii=False),
            'human_confirmed':False})
    (OUT/'preview').mkdir(exist_ok=True);(OUT/'preview/index.html').write_text('\n'.join(text))
    with (OUT/'sample_review.csv').open('w',newline='') as stream:
        if csv_rows:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--add-candidates',action='store_true');parser.add_argument('--reason');parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    if args.render_only:
        order=json.loads((OUT/'selection.json').read_text())['pairs']
        return render([json.loads((OUT/'facts'/f'{pair.replace(":","_")}.json').read_text()) for pair in order if (OUT/'facts'/f'{pair.replace(":","_")}.json').exists()])
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    c=json.loads((SOURCE/'config.json').read_text());c['geometry_root']=str(SOURCE/'geometry')
    train=[r for r in json.loads((SOURCE/'manifest.json').read_text())['rows'] if r['split']=='train']
    allowed={r['pair_id'].removeprefix('an:') for r in train}
    metadata={r['video-id']:r for r in csv.DictReader((BASE/'secondary_datasets/anakin/references/metadata.csv').open()) if r['video-id'] in allowed}
    selection={'pairs':CURRENT,'rule':'fixed current pairs first; no predictions/results read','seed':20260909}
    selection.update(source_manifest_sha256=digest(SOURCE/'manifest.json'),
                     official_metadata_sha256=digest(BASE/'secondary_datasets/anakin/references/metadata.csv'))
    if args.add_candidates:
        if not args.reason:raise ValueError('EXPLICIT_CANDIDATE_REASON_REQUIRED')
        selection.update(pairs=CURRENT+candidates(train,metadata),candidate_reason=args.reason,
                         candidate_rule='registered train, metadata running/walking person/object edits; sorted IDs shuffled seed20260909, first2; no replacement',
                         candidate_metadata_pool=[{'pair_id':p,'task':metadata[p[3:]]['task']} for p in sorted({r['pair_id'] for r in train}-set(CURRENT)) if any(w in metadata[p[3:]]['task'].lower() for w in ['running','walking'])])
    if (OUT/'selection.json').exists():
        prior=json.loads((OUT/'selection.json').read_text())
        if len(prior['pairs'])>2 and selection['pairs']!=prior['pairs']:raise ValueError('CANDIDATES_ALREADY_FIXED')
    save(OUT/'selection.json',selection) # Before any candidate media access.
    prior_id=json.loads((PRIOR/'selected_identity.json').read_text());gt_id=json.loads((PRIOR/'GT_identity_supplement.json').read_text())
    for r in train:
        if r['pair_id'] in CURRENT:
            old=next(x for x in prior_id if x['video_id']==r['video_id']);oldgt=next(x for x in gt_id if x['video_id']==r['video_id'])
            if any(old[k]!=r[k] for k in ['pair_id','source_id','role','video_sha256','source_hw']) or old['range']!=[0,24] or old['pts']!=r['pts'][:24] or oldgt['mask_sha256']!=r['mask_sha256']:raise ValueError('PRIOR_IDENTITY_CONFLICT')
    results=[];error=None
    try:
        for pair in selection['pairs']:
            p=OUT/'facts'/f'{pair.replace(":","_")}.json'
            result=json.loads(p.read_text()) if p.exists() else prepare_pair(pair,train,c,metadata,started+600)
            results.append(result);print(pair,result['status'],flush=True)
    except Exception as exc:error=str(exc);raise
    finally:
        render(results);save(OUT/'status.json',{'status':'COMPLETE' if error is None else 'INCOMPLETE','error':error,'media_seconds_this_run':time.monotonic()-started,
            'processed_pairs':[r['pair_id'] for r in results],'GPU_model_frontend_download_calls':0,'publication':'PENDING'})


if __name__=='__main__':main()
