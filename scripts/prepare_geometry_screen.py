"""Bounded GRIP6/AINPAINT4 media and annotation review, no model imports.

Caller: this dataset screen CLI. Outputs external masks/media/HTML and small
candidate records. Existing decode/digest are reused; MAT union and PNG sequences
need only these dataset-specific adapters, not a downloader/data framework.
"""
import csv
import html
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

import av
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import decode,numeric_members

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/secondary_datasets/pvst_geometry_screen_v1')
GRIP_URL='https://www.grip.unina.it/download/prog/ForgedVideosDataset/Copymove/'
GRIP=[('02','Fast Car','additive'),('04','Rolling Can','additive'),('05','Falling Can','additive'),('10','Tennis Ball','additive'),('06','Walnuts','occlusive'),('14','Wall Frame','occlusive')]
SEED=20260909


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def read(p):return json.loads(p.read_text())
def csvsave(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def camera_check(images):
    """Classical background-border feature matching; insufficient != moving."""
    cv2.setRNGSeed(SEED);records=[];orb=cv2.ORB_create(nfeatures=2000)
    for (ia,a),(ib,b) in zip(images,images[1:]):
        aa=cv2.cvtColor(np.asarray(a),cv2.COLOR_RGB2GRAY);bb=cv2.cvtColor(np.asarray(b),cv2.COLOR_RGB2GRAY)
        h,w=aa.shape;mask=np.full((h,w),255,np.uint8);mask[int(h*.2):int(h*.8),int(w*.2):int(w*.8)]=0
        ka,da=orb.detectAndCompute(aa,mask);kb,db=orb.detectAndCompute(bb,mask)
        r={'from':ia,'to':ib,'reliable':False,'matches':0,'inliers':0}
        if da is not None and db is not None and len(db)>=2:
            pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance];r['matches']=len(good)
            if len(good)>=20:
                src=np.float32([ka[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good]);M,inlier=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000,confidence=.99,refineIters=10)
                if M is not None:
                    use=inlier[:,0].astype(bool);count=int(use.sum());span=np.ptp(src[use],axis=0)/[w,h] if count else [0,0]
                    r.update(inliers=count,inlier_fraction=count/len(good),coverage_xy=list(map(float,span)))
                    if count>=15 and count/len(good)>=.5 and min(span)>=.10:
                        scale=float(math.hypot(M[0,0],M[1,0]));angle=float(math.degrees(math.atan2(M[1,0],M[0,0])));shift=float(np.linalg.norm(M[:,2]));corners=np.array([[0,0],[w,0],[0,h],[w,h]],float);moved=corners@M[:,:2].T+M[:,2];maxmove=float(np.linalg.norm(moved-corners,axis=1).max())
                        r.update(reliable=True,translation_pixels=shift,rotation_degrees=angle,scale=scale,max_corner_displacement_fraction=maxmove/math.hypot(h,w))
        records.append(r)
    reliable=[r for r in records if r['reliable']]
    stable=bool(records) and len(reliable)==len(records) and all(r['max_corner_displacement_fraction']<=.03 and abs(r['rotation_degrees'])<=3 and .97<=r['scale']<=1.03 for r in reliable)
    state='较稳定候选（传统背景匹配，待人工确认）' if stable else ('可见全局运动迹象（非切镜定论）' if any(r['max_corner_displacement_fraction']>.03 or abs(r['rotation_degrees'])>3 or not .97<=r['scale']<=1.03 for r in reliable) else '不确定：可靠背景匹配不足')
    return {'state':state,'stable_candidate':stable,'intervals':records,'method':'5 uniform source frames; outer border outside centre60%; ORB2000/Hamming ratio.75/partial affine RANSAC3px; reliable>=20 matches,>=15 inliers,>=.5 fraction,xy span>=.1; stable thresholds maxcorner<=.03 diagonal,rotation<=3deg,scale.97..1.03; foreground leakage/parallax possible'}


def camera_text(camera):
    rs=[r for r in camera['intervals'] if r['reliable']]
    if not rs:return camera['state']+'；可靠背景间隔0/'+str(len(camera['intervals']))
    return camera['state']+f"；可靠间隔{len(rs)}/{len(camera['intervals'])}；最大平移{max(r['translation_pixels'] for r in rs):.2f}px、旋转{max(abs(r['rotation_degrees']) for r in rs):.2f}°、角点位移{max(r['max_corner_displacement_fraction'] for r in rs):.2%}对角线"


def motion_stats(masks):
    centers=[];areas=[];previous=None;ious=[];positive=[]
    for i,gt in enumerate(masks):
        yy,xx=np.where(gt);areas.append(len(xx))
        if len(xx):centers.append([float(xx.mean()),float(yy.mean())]);positive.append(i)
        if previous is not None:
            union=(gt|previous).sum()
            if union:ious.append(float((gt&previous).sum()/union))
        previous=gt
    c=np.asarray(centers);span=float(np.linalg.norm(c[:,None]-c[None,:],axis=-1).max()) if len(c)>1 else 0.
    return {'centroid_span_pixels':span,'centroid_span_fraction':span/math.hypot(*masks[0].shape),'mean_adjacent_mask_IoU':float(np.mean(ious)) if ious else None,'positive_frames':positive,'areas':areas}


def archive_key(name,container,variant,selected):
    parts=Path(name).parts
    for k,p in enumerate(parts):
        if p==container and len(parts)==k+4 and parts[k+1]==variant and parts[k+2] in selected and Path(parts[-1]).suffix.lower()=='.png':return parts[k+2],parts[-1]
    return None


def extract_selected(archive,container,variant,selected):
    dest=ROOT/'ainpaint'/'selected_frames'/container/variant;inventory=[];count=0
    with tarfile.open(archive,'r:') as tf:
        for member in tf:
            key=archive_key(member.name,container,variant,selected)
            if key and member.isfile():
                source,name=key;folder=dest/source;folder.mkdir(parents=True,exist_ok=True);p=folder/name
                if not p.exists():
                    with tf.extractfile(member) as src,p.open('xb') as out:
                        while chunk:=src.read(1024*1024):out.write(chunk)
                assert p.stat().st_size==member.size;count+=1;inventory.append({'member':member.name,'size':member.size,'relative_path':str(p.relative_to(ROOT))})
    save(ROOT/'ainpaint'/f'{container}_extracted.json',inventory);return dest,count


def pngs(folder):
    files=numeric_members([p.name for p in folder.glob('*.png')]);return [(i,folder/n) for i,n in files]


def video_from_png(files,out,rate=25):
    """PNG originals have no PTS; this is explicitly synthetic playback timing."""
    if out.exists():return
    with av.open(str(out),'w') as c:
        stream=c.add_stream('libx264',rate=rate);a=Image.open(files[0][1]).convert('RGB');stream.width=a.width;stream.height=a.height;stream.pix_fmt='yuv420p';stream.options={'crf':'18'}
        for i,p in files:
            frame=av.VideoFrame.from_image(Image.open(p).convert('RGB'))
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)


def ain_mask(image):
    # Author dataset.py final processed_masks: TF.to_tensor(mask_img) > 0.5.
    # Its earlier >0 is only the mask-aware crop envelope, not the target.
    return np.asarray(image.convert('L'),dtype=np.float32)/255 > .5


def overlay(image,gt):
    a=np.asarray(image).copy();a[gt]=(a[gt]*.55+np.array([255,70,0])*.45).astype(np.uint8);return Image.fromarray(a)


def ranges(indices):
    result=[]
    for i in indices:
        if result and result[-1][1]==i:result[-1][1]=i+1
        else:result.append([i,i+1])
    return result


def grip_review():
    result=[]
    for number,name,kind in GRIP:
        folder=ROOT/'preview'/('grip_'+number);folder.mkdir(parents=True,exist_ok=True)
        paths={role:ROOT/'grip'/file for role,file in [('real',f'VIDEO_ORI_{number}.avi'),('fake',f'VIDEO_FORG_rigid_{number}.avi'),('mask',f'GT_rigid_{number}.mat')]}
        if not all(p.exists() for p in paths.values()):
            result.append({'dataset':'GRIP','source_id':number,'status':'MISSING','name':name,'missing':[str(p) for p in paths.values() if not p.exists()]});continue
        arrays={k:v for k,v in loadmat(paths['mask']).items() if not k.startswith('__')};assert set(arrays)=={'GT'}
        gt=arrays['GT'];assert gt.ndim==3 and set(np.unique(gt))<={0,1};n=gt.shape[2];keep=sorted(set(np.linspace(0,n-1,5,dtype=int).tolist()));fixed=[0,n//2,n-1]
        meta={};ims={}
        for role in ['real','fake']:
            meta[role],ims[role]=decode(paths[role],set(keep+fixed));save(folder/f'{role}_identity.json',{'official_file':str(paths[role]),'sha256':digest(paths[role]),**meta[role]})
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(paths[role]),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-vsync','0',str(folder/f'{role}.mp4')],check=True) if not (folder/f'{role}.mp4').exists() else None
        aligned=meta['fake']['frame_count']==n and meta['fake']['sizes_hw']==[tuple(gt.shape[:2])]
        same_clock=meta['real']['pts']==meta['fake']['pts'] and meta['real']['frame_count']==n
        camera=camera_check([(i,ims['real'][i]) for i in keep if i in ims['real']]);masks=[gt[:,:,i].astype(bool) for i in range(n)];motion=motion_stats(masks)
        extra=next(iter(motion['positive_frames']),None)
        if extra is not None and extra not in fixed and not any(masks[i].any() for i in fixed):
            _,additional=decode(paths['fake'],{extra});ims['fake'].update(additional);fixed.append(extra)
        for i in fixed:
            ims['fake'][i].save(folder/f'fake_{i:04d}.png');Image.fromarray(gt[:,:,i]*255).save(folder/f'mask_{i:04d}.png')
            if aligned:overlay(ims['fake'][i],masks[i]).save(folder/f'overlay_{i:04d}.png')
        r={'dataset':'GRIP','source_id':number,'name':name,'status':'READY','revision':'author directory Last-Modified 2023-05-19; per-file SHA in identities','official_split':'not provided / review pool only','official_source':GRIP_URL,'edit_type':kind+' copy-move','object':name,'mask_semantics':'复制区域并集（原区与克隆均为1），未验证编辑目标掩膜','direct_pixel_GT':False,'alignment':'H×W×T exact decoded fake dimensions/count; MAT temporal order to decoded frame index; no independent mask timestamps' if aligned else 'alignment_unresolved','real_fake_sync':'帧数及PTS网格相同，但未证明内容逐像素注册' if same_clock else '未核实同步；分别记录PTS，独立播放','timing_type':'从复制并集mask导出帧号区间；不是已确认编辑目标区间','positive_frame_runs':ranges(motion['positive_frames']),'official_time_interval':None,'real_path':str(paths['real'].relative_to(ROOT)),'fake_path':str(paths['fake'].relative_to(ROOT)),'mask_path':str(paths['mask'].relative_to(ROOT)),'preview_folder':str(folder.relative_to(ROOT)),'frames':n,'native_hw':list(gt.shape[:2]),'real_frames':meta['real']['frame_count'],'PTS_real':meta['real']['pts'],'PTS_fake':meta['fake']['pts'],'camera':camera,'mask_motion':motion,'fixed_frames':fixed,'aligned_mask':aligned,'manual_scene_approval':False}
        save(folder/'record.json',r);result.append(r);print('GRIP reviewed',number,camera['state'],flush=True)
    return result


def ainpaint_review():
    rules=read(ROOT/'selection_rules.json');sources=rules['prescreen_sources_in_order'];archives=ROOT/'ainpaint'/'archives';records=[];result=[]
    if not all((archives/n).exists() for n in ['input_frames.tar','input_masks.tar','CoCoCo.tar']):
        mask_audit=[]
        if (archives/'input_masks.tar').exists():
            maskroot,_=extract_selected(archives/'input_masks.tar','input_masks','resized_432x240',set(sources))
            for source in sources:
                fs=pngs(maskroot/source);sizes=set();values=set()
                for i,path in fs:
                    with Image.open(path) as im:sizes.add(im.size);values.update(np.unique(np.array(im.convert('L'))).tolist())
                mask_audit.append({'source_id':source,'mask_frames':len(fs),'sizes_wh':sorted(sizes),'values':sorted(values),'real_fake_alignment':'not_checked_missing_media'})
            save(ROOT/'ainpaint'/'mask_only_audit.json',mask_audit)
        return [],{'status':'MISSING_ARCHIVES','missing':[n for n in ['input_frames.tar','input_masks.tar','CoCoCo.tar'] if not (archives/n).exists()],'planned_prescreen':len(sources),'RGB_prescreen_completed':0,'mask_only_sources':len(mask_audit),'reason':'real/fake archive incomplete after actual transport failures; no alternate method/variant or source substitution'}
    realroot,_=extract_selected(archives/'input_frames.tar','input_frames','432x240',set(sources));maskroot,_=extract_selected(archives/'input_masks.tar','input_masks','resized_432x240',set(sources))
    for order,source in enumerate(sources):
        record={'source_id':source,'seed_order':order,'status':'UNRESOLVED'}
        try:
            real=pngs(realroot/source);mask=pngs(maskroot/source)
            assert real and [i for i,p in real]==[i for i,p in mask] and [i for i,p in real]==list(range(len(real))), 'FRAME_IDENTITY_OR_COVERAGE'
            masks=[];values=set()
            for (_,rp),(_,mp) in zip(real,mask):
                with Image.open(rp) as a,Image.open(mp) as b:assert a.size==b.size==(432,240);v=np.array(b.convert('L'));values.update(np.unique(v).tolist());masks.append(ain_mask(b))
            keep=sorted(set(np.linspace(0,len(real)-1,5,dtype=int).tolist()));camera=camera_check([(i,Image.open(real[i][1]).convert('RGB')) for i in keep]);motion=motion_stats(masks);assert motion['positive_frames'],'NO_EDIT_MASK_FOREGROUND'
            record.update(status='VERIFIED_REAL_MASK',frames=len(real),mask_values=sorted(values),camera=camera,mask_motion=motion)
        except Exception as e:record.update(status='UNRESOLVED',reason=repr(e))
        records.append(record)
    save(ROOT/'ainpaint'/'prescreen24.json',records)
    stable=[r for r in records if r['status']=='VERIFIED_REAL_MASK' and r['camera']['stable_candidate']]
    ranked=sorted(stable,key=lambda r:(r['mask_motion']['centroid_span_fraction'],r['seed_order']));chosen=[]
    # Disjoint lower/higher relative displacement groups; fewer than4 stays fewer.
    for r in ranked[:min(2,len(ranked))]:chosen.append((r,'变化较少参照（相对本次稳定候选）'))
    remaining=[r for r in ranked if r not in [a for a,g in chosen]]
    for r in reversed(remaining[-2:]):chosen.append((r,'位移较明显候选（相对本次稳定候选）'))
    save(ROOT/'ainpaint'/'selected_before_fake_extraction.json',{'rules':rules,'stable_sources':len(stable),'selection':[{'source_id':r['source_id'],'group':g} for r,g in chosen],'no_visual_or_model_reselection':True})
    fakeroot,_=extract_selected(archives/'CoCoCo.tar','CoCoCo','432x240',{r['source_id'] for r,g in chosen})
    for record,group in chosen:
        source=record['source_id'];folder=ROOT/'preview'/('ainpaint_'+source);folder.mkdir(parents=True,exist_ok=True)
        real=pngs(realroot/source);fake=pngs(fakeroot/source);mask=pngs(maskroot/source);aligned=[i for i,p in fake]==[i for i,p in real]
        if not aligned:
            result.append({'dataset':'AINPAINT','source_id':source,'status':'ALIGNMENT_UNRESOLVED','name':source,'reason':'fake/real frame names differ; no replacement'});continue
        values=set();masks=[]
        for (_,fp),(_,mp) in zip(fake,mask):
            with Image.open(fp) as a,Image.open(mp) as b:assert a.size==b.size==(432,240);masks.append(ain_mask(b))
        for role,files in [('real',real),('fake',fake)]:video_from_png(files,folder/f'{role}.mp4');meta,_=decode(folder/f'{role}.mp4',set());assert meta['frame_count']==len(files);save(folder/f'{role}_preview_identity.json',meta)
        n=len(fake);fixed=[0,n//2,n-1]
        for i in fixed:
            im=Image.open(fake[i][1]).convert('RGB');im.save(folder/f'fake_{i:04d}.png');Image.open(mask[i][1]).save(folder/f'mask_{i:04d}.png');overlay(im,masks[i]).save(folder/f'overlay_{i:04d}.png')
        positive=record['mask_motion']['positive_frames']
        r={'dataset':'AINPAINT','source_id':source,'name':source,'status':'READY','revision':rules['HF_revision'],'official_split':'val (author code '+rules['official_code_revision']+')','official_source':'https://huggingface.co/datasets/Truebees/AINPAINT','edit_type':'CoCoCo object-removal inpainting; raw432x240','object':'待人工描述（仅按目录身份，不凭名称断言对象）','selection_group':group,'mask_semantics':'官方input_masks/resized_432x240编辑区域GT，非三维异常标签','direct_pixel_GT':True,'alignment':'real/fake/mask identical numbered PNG lists and432x240 at every frame; raw soft mask retained; uint8/255 >0.5 follows final official target conversion','real_fake_sync':'同一source及帧号对应；官方PNG没有PTS，播放器25fps是展示时钟，不是官方时间','timing_type':'从已对齐编辑mask导出帧号区间；官方秒级时间未提供','positive_frame_runs':ranges(positive),'official_time_interval':None,'real_path':str((realroot/source).relative_to(ROOT)),'fake_path':str((fakeroot/source).relative_to(ROOT)),'mask_path':str((maskroot/source).relative_to(ROOT)),'preview_folder':str(folder.relative_to(ROOT)),'frames':n,'native_hw':[240,432],'real_frames':n,'PTS_real':None,'PTS_fake':None,'preview_fps':25,'camera':record['camera'],'mask_motion':record['mask_motion'],'fixed_frames':fixed,'aligned_mask':True,'manual_scene_approval':False}
        save(folder/'record.json',r);result.append(r);print('AINPAINT prepared',source,group,flush=True)
    return result,{'status':'COMPLETE','prescreen_count':len(records),'stable_candidates':len(stable),'selected':len(result),'official_source_identity_limit':'deduplicated author clip IDs across techniques/variants; original long-source URLs not published in split; cannot assert independence of same-series different IDs'}


def render(records,ain):
    fields=['dataset','revision','source_id','name','status','official_split','official_source','edit_type','object','selection_group','real_path','fake_path','mask_path','frames','native_hw','alignment','real_fake_sync','mask_semantics','direct_pixel_GT','timing_type','positive_frame_runs','camera_state','preview_folder'];csvrows=[]
    for r in records:
        a={k:r.get(k,'') for k in fields};a['camera_state']=camera_text(r['camera']) if r.get('camera') else '未完成';a={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in a.items()};csvrows.append(a)
    csvsave(ROOT/'candidate_manifest.csv',csvrows);save(ROOT/'candidate_manifest.json',records)
    cards=['<!doctype html><meta charset="utf-8"><title>PVST候选数据人工审阅</title><style>body{font:16px sans-serif;max-width:1500px;margin:24px}video{width:100%;max-width:680px} .pair,.shots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.shots{grid-template-columns:repeat(3,minmax(0,1fr))}img{max-width:100%}textarea,select{display:block;width:90%;margin:8px 0}section{border-top:2px solid #aaa;padding:16px 0}</style><h1>局部编辑：结构／遮挡／运动衔接候选</h1><p>只供人工审阅，未冻结训练或评价集合。没有模型分数或热图。对象增删、mask位移本身不等于三维时序异常。自动机位只来自原片传统特征/RANSAC，匹配不足=不确定。播放器独立，不强制同步。</p><button id="export">导出人工填写CSV</button> <span id="save">填写保存在当前浏览器；请导出备份。</span>']
    for dataset in ['GRIP','AINPAINT']:
        cards.append(f'<h2>{dataset}</h2>')
        if dataset=='AINPAINT' and not any(r['dataset']=='AINPAINT' and r['status']=='READY' for r in records):
            cards.append('<p>本轮0/4对：两个必要媒体包下载连接中断、尚未通过完整SHA核验。仅mask包完整，不能当作已准备配对；固定24条val队列未完成原片机位初筛。未改用压缩版、其他方法或新来源凑数。详见中文报告及AINPAINT资产缺口。</p>')
        for r in [r for r in records if r['dataset']==dataset]:
            sid=dataset+'_'+r['source_id'];cards.append(f'<section data-id="{html.escape(sid)}"><h3>{html.escape(r.get("name",r["source_id"]))} · {html.escape(r["source_id"])}</h3>')
            if r['status']!='READY':cards.append('<p>'+html.escape(json.dumps(r,ensure_ascii=False))+'</p></section>');continue
            folder=r['preview_folder'];cards.append('<p>'+html.escape(r['mask_semantics'])+'；可直接用于编辑像素评价：'+('是（尚未人工场景确认）' if r['direct_pixel_GT'] else '否，仅定性／复制并集定位')+'</p><p>'+html.escape(camera_text(r['camera']))+'；'+html.escape(r['real_fake_sync'])+'</p><p>'+html.escape(r['timing_type'])+'；mask非空帧区间[起,止)：'+str(r['positive_frame_runs'])+'。全程mask非空不等于可验证编辑起止。</p>')
            cards.append(f'<p>播放器为H264兼容预览副本，原件未修改：<a href="{r["real_path"]}">原始real</a> / <a href="{r["fake_path"]}">原始fake</a> / <a href="{r["mask_path"]}">官方标注原件</a>。不强制同步播放。</p><div class="pair"><div>原片（完整发布短片／PNG全序列预览）<video controls preload="none" src="{folder}/real.mp4"></video></div><div>rigid / CoCoCo伪造<video controls preload="none" src="{folder}/fake.mp4"></video></div></div>')
            cards.append(f'<p>尺寸{r["native_hw"]}，帧数{r["frames"]}。'+('AINPAINT播放器25fps仅展示；未提供官方PTS。' if dataset=='AINPAINT' else '真实解码PTS见该对record.json，AVI原件保留；MP4仅兼容预览。')+'</p>')
            for i in r['fixed_frames']:
                timing=f'fake PTS {r["PTS_fake"][i]["seconds"]:.6f}s' if r['PTS_fake'] else '官方仅帧号，预览时钟'+f'{i/25:.3f}s'
                cards.append(f'<p>'+('固定首／中／末fake帧' if i in [0,r['frames']//2,r['frames']-1] else '额外首个非空官方GT帧（按标注选取，非模型）')+f'{i} · {timing}</p><div class="shots">')
                for name,title in [('fake','原画面'),('mask','官方标注（GRIP为复制并集）'),('overlay','标注叠加，不是三维异常真值')]:
                    file=f'{folder}/{name}_{i:04d}.png'
                    if (ROOT/file).exists():cards.append(f'<figure><a href="{file}"><img loading="lazy" src="{file}"></a><figcaption>{title}</figcaption></figure>')
                cards.append('</div>')
            cards.append('<label>1. 机位<select data-field="camera"><option value=""></option><option>固定</option><option>轻微移动</option><option>明显移动或切镜</option><option>不确定</option></select></label><label>2. 编辑改了什么；形态、遮挡、接触或运动衔接中有何值得比较的变化<textarea data-field="edit"></textarea></label><label>3. 变化协调性<select data-field="coherence"><option value=""></option><option>看起来不协调</option><option>看起来协调</option><option>不确定</option></select></label><p>合理的增删不自动算异常；不按肉眼明显程度筛掉样本。</p></section>')
    cards.append('<p><a href="candidate_manifest.csv">候选清单</a> · <a href="review_summary.md">中文报告</a> · <a href="selection_rules.json">预先选择规则</a></p>')
    cards.append('''<script>
const sections=[...document.querySelectorAll('section[data-id]')],key='pvst-geometry-screen-v1';
function collect(){return sections.filter(s=>s.querySelector('[data-field]')).map(s=>({sample:s.dataset.id,...Object.fromEntries([...s.querySelectorAll('[data-field]')].map(e=>[e.dataset.field,e.value]))}));}
try{const saved=JSON.parse(localStorage.getItem(key)||'[]');for(const s of sections){const r=saved.find(r=>r.sample===s.dataset.id);if(r)for(const e of s.querySelectorAll('[data-field]'))e.value=r[e.dataset.field]||'';}}catch(e){}
for(const s of sections)for(const e of s.querySelectorAll('[data-field]'))e.addEventListener('input',()=>{try{localStorage.setItem(key,JSON.stringify(collect()));}catch(e){document.querySelector('#save').textContent='本地保存不可用，请导出';}});
document.querySelector('#export').onclick=()=>{const fields=['sample','camera','edit','coherence'],q=x=>'"'+String(x??'').replaceAll('"','""')+'"';const text=[fields.map(q).join(','),...collect().map(r=>fields.map(k=>q(r[k])).join(','))].join('\\r\\n');const url=URL.createObjectURL(new Blob(['\\ufeff'+text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='geometry_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script>''')
    (ROOT/'index.html').write_text('\n'.join(cards))


def prepare():
    start=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=read(ROOT/'download_plan.json');checks=[]
    for spec in plan['files']:
        path=ROOT/spec['relative_path']
        if path.exists():
            assert path.stat().st_size==spec['size'];sha=digest(path)
            if spec.get('sha256'):assert sha==spec['sha256']
            checks.append({'path':spec['relative_path'],'bytes':path.stat().st_size,'sha256':sha,'verified':True})
    save(ROOT/'asset_identity.json',checks)
    (ROOT/'preview').mkdir(exist_ok=True);records=grip_review();ar,ain=ainpaint_review();records+=ar;render(records,ain)
    download=read(ROOT/'download_outcome.json');first=read(ROOT/'download_first_transfer.json');summary={'status':'PREVIEW_READY_WITH_AINPAINT_BLOCKED','execution_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'prepared_pairs':sum(r['status']=='READY' for r in records),'records':records,'ainpaint_screen':ain,'download_bytes':download['payload_bytes_saved'],'download_measurement':'saved payload incl incomplete ranges; wire overhead not measured','download_seconds':time.time()-first['started_unix'],'prepare_seconds':time.monotonic()-start,'no_models_or_frontends':True,'formal_split_changed':False}
    save(ROOT/'summary.json',summary);print('PREVIEW_READY',summary['prepared_pairs'],flush=True)


if __name__=='__main__':prepare()
