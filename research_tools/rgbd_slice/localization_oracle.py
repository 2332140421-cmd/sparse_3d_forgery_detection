"""CPU GT oracle for whole slabs and one fixed native-image subdivision.

CLI caller -> per-frame/per-video overlap limits. Existing model metrics cannot
answer representation capacity; no encoder/readout/model is constructed here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np
import torch
from PIL import Image,ImageDraw
from .data import Video,save
from research_tools.depth_slice.model import slab_ids
from scripts.prepare_v8_assets import digest

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
BASE=ROOT/'pvst_rgb_partition_anakin_dev_v1'
REF=ROOT/'pvst_slice_relation_readout_v1'
OUT=ROOT/'pvst_slice_localization_oracle_v1'
PAIRS={'an:7e2e57','an:4rvro9','an:4wgx37','an:95q8g9'}
REPO=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text())
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def csvsave(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def optimum(n,g):
    """All-prefix optimum; empty prefix included; stable density tie ordering."""
    n=np.asarray(n,dtype=np.int64);g=np.asarray(g,dtype=np.int64)
    assert n.shape==g.shape and np.all(n>0) and np.all((g>=0)&(g<=n))
    G=int(g.sum())
    if not G:return None,np.zeros(len(n),bool)
    order=np.argsort(-g.astype(np.float64)/n,kind='stable')
    tp=np.r_[0,np.cumsum(g[order])];area=np.r_[0,np.cumsum(n[order])]
    best=int(np.argmax(2.*tp/(G+area)))
    TP=int(tp[best]);P=int(area[best]);selected=np.zeros(len(n),bool);selected[order[:best]]=True
    return {'Dice':2*TP/(G+P),'IoU':TP/(G+P-TP),'TP':TP,'prediction_pixels':P,'selected_units':best},selected


def partitions(xyz,valid,gt):
    """Both partitions derive from the same original native slab inverse index."""
    ids,index=slab_ids(xyz,valid)
    inverse=index.numpy();v=valid.numpy();y,x=np.nonzero(v);H,W=v.shape;L=math.ceil(min(H,W)/16)
    ny=(H+L-1)//L;nx=(W+L-1)//L
    key=inverse*(ny*nx)+(y//L)*nx+x//L
    unique,grid=np.unique(key,return_inverse=True)
    assert np.array_equal(unique[grid]//(ny*nx),inverse)
    truth=gt.numpy()[v].astype(np.int64)
    output={}
    for name,ix in [('SLAB',inverse),('SLAB_GRID',grid)]:
        n=np.bincount(ix);g=np.bincount(ix[truth.astype(bool)],minlength=len(n))
        assert int(n.sum())==int(v.sum()) and int(g.sum())==int((gt&valid).sum())
        assert len(n)==0 or np.all(n>0)
        result,chosen=optimum(n,g)
        output[name]={'units':len(n),'result':result,'inverse':ix,'chosen':chosen}
    a,b=output['SLAB']['result'],output['SLAB_GRID']['result']
    if a is not None:
        assert b['Dice']+1e-12>=a['Dice'] and b['IoU']+1e-12>=a['IoU']
    return output,L


def preview(folder,row,i,f,parts):
    panels=[];v=f['valid'].numpy();gt=f['gt'].numpy();H,W=v.shape
    for name in ['GT','SLAB','SLAB_GRID']:
        if name=='GT':a=gt
        else:
            a=np.zeros((H,W),bool);q=parts[name];a[v]=q['chosen'][q['inverse']]
        pixels=np.zeros((H,W,3),np.uint8);pixels[~v]=[80,80,80];pixels[a]=255
        im=Image.fromarray(pixels);im.thumbnail((400,400));panel=Image.new('RGB',(400,430),'white');panel.paste(im,(0,30));ImageDraw.Draw(panel).text((5,5),name,fill='black');panels.append(panel)
    canvas=Image.new('RGB',(1200,430),'white')
    for j,im in enumerate(panels):canvas.paste(im,(400*j,0))
    filename=f"{row['pair_id'].replace(':','_')}_{i:04d}.png";canvas.save(folder/filename)
    return f"<p>{row['pair_id']} frame={i}, PTS={f['pts']:.9f}s; native={H}x{W}; preview only scaled. Gray: invalid geometry; GT is official full-image GT.</p><img src='{filename}' width='1200'>"


def main():
    started=time.monotonic();torch.set_num_threads(4)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert git('branch','--show-current')=='pvst-slice-localization-oracle'
    identity={'execution_head':git('rev-parse','HEAD'),'manifest_sha256':digest(BASE/'manifest.json'),'config_sha256':digest(BASE/'config.json'),'reference_summary_sha256':digest(REF/'summary.json'),'reference_cache_frames_sha256':digest(REF/'cache_frames.csv'),'partitions':['SLAB','SLAB_GRID'],'grid':'L=ceil(min(H,W)/16), native origin top-left','CUDA_VISIBLE_DEVICES':'','source_code':{p:digest(REPO/p) for p in ['research_tools/rgbd_slice/localization_oracle.py','research_tools/rgbd_slice/data.py','research_tools/depth_slice/data.py','research_tools/depth_slice/model.py','scripts/review_anakin.py']}}
    if OUT.exists():
        old=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if old.get('status')=='COMPLETE' and old.get('identity')==identity:print('REUSE COMPLETE');return
        raise ValueError('EXISTING_OUTPUT_IDENTITY_INSPECT_NO_OVERWRITE')
    OUT.mkdir();save(OUT/'identity.json',identity)
    manifest=read(BASE/'manifest.json');config=read(BASE/'config.json');base_id=read(BASE/'identity.json')
    assert identity['manifest_sha256']==base_id['frozen_manifest_sha256']==read(REF/'identity.json')['manifest_sha256']
    assert identity['config_sha256']==base_id['config_sha256']
    rows=[r for r in manifest['rows'] if r['split']=='dev' and r['role']=='fake']
    assert len(rows)==4 and {r['pair_id'] for r in rows}==PAIRS and sum(r['actual_frames'] for r in rows)==462
    ref={(r['video_id'],int(r['frame_index'])):r for r in csv.DictReader((REF/'cache_frames.csv').open()) if r['split']=='dev' and r['role']=='fake'}
    scored=read(REF/'CURRENT_ONLY/final_dev.json')['frame_records'];scored={(r['video_id'],r['frame_index']):r for r in scored if r['role']=='fake'}
    assert len(ref)==len(scored)==462 and set(ref)==set(scored)
    frame_rows=[];videos=[];seen=set();visuals=[];folder=OUT/'preview';folder.mkdir();assets=[]
    for row in rows:
        assert digest(Path(row['video_path']))==row['video_sha256'] and digest(Path(row['mask_zip']))==row['mask_sha256']
        asset=next(a for a in base_id['assets'] if a['video_id']==row['video_id'])
        for path,sha in asset['cache_sha256'].items():assert digest(Path(path))==sha
        assets.append({'video_id':row['video_id'],'pair_id':row['pair_id'],'video_sha256':row['video_sha256'],'mask_sha256':row['mask_sha256'],'source_hw':row['source_hw'],'cache_sha256':asset['cache_sha256']})
        v=Video(row,config);rr=[];ticks={0,row['actual_frames']//2,row['actual_frames']-1}
        try:
            for i in range(row['actual_frames']):
                f=v.frame(i);key=(row['video_id'],i);a=ref[key];b=scored[key];assert key not in seen;seen.add(key)
                assert f['pts']==row['pts'][i]['pts_s']==float(a['pts'])==b['pts'] and list(f['gt'].shape)==row['source_hw']
                valid=int(f['valid'].sum());G=int((f['gt']&f['valid']).sum());full=int(f['gt'].sum())
                assert valid==int(a['valid_pixels']) and G==int(a['valid_GT_pixels']) and full==int(a['GT_pixels'])
                parts,L=partitions(f['xyz'],f['valid'],f['gt']);assert parts['SLAB']['units']==int(a['slabs'])==b['slabs']
                entry={'pair_id':row['pair_id'],'video_id':row['video_id'],'frame_index':i,'pts':f['pts'],'H':row['source_hw'][0],'W':row['source_hw'][1],'L':L,'valid_pixels':valid,'GT_pixels':full,'valid_GT_pixels':G,'GT_coverage':G/full if full else None,'comparable':G>0}
                for name,q in parts.items():
                    entry[name+'_units']=q['units']
                    for k in ['Dice','IoU','TP','prediction_pixels','selected_units']:entry[name+'_'+k]=q['result'][k] if q['result'] else None
                frame_rows.append(entry);rr.append(entry)
                if i in ticks:visuals.append(preview(folder,row,i,f,parts))
                del f,parts
        finally:v.close()
        good=[x for x in rr if x['comparable']];total_GT=sum(x['GT_pixels'] for x in rr)
        result={'pair_id':row['pair_id'],'video_id':row['video_id'],'frames':len(rr),'comparable_frames':len(good),'G_zero_frames':len(rr)-len(good),'GT_coverage':float(np.mean([x['GT_coverage'] for x in rr if x['GT_coverage'] is not None])),'GT_coverage_pixel_weighted':sum(x['valid_GT_pixels'] for x in rr)/total_GT,'GT_pixels':total_GT,'valid_GT_pixels':sum(x['valid_GT_pixels'] for x in rr),'L':rr[0]['L'],'scene_record':'未知：已有sample_scene_review_v1未覆盖这四对，不依据本轮掩膜图推断机位或活动'}
        for name in ['SLAB','SLAB_GRID']:
            for metric in ['Dice','IoU']:result[name+'_'+metric]=float(np.mean([x[name+'_'+metric] for x in good])) if good else None
            result[name+'_units_mean']=float(np.mean([x[name+'_units'] for x in rr]))
        result['Dice_delta']=result['SLAB_GRID_Dice']-result['SLAB_Dice'] if good else None;result['IoU_delta']=result['SLAB_GRID_IoU']-result['SLAB_IoU'] if good else None
        prior=next(x for x in read(REF/'summary.json')['comparison'] if x['video_id']==row['video_id']);assert abs(prior['GT_coverage']-result['GT_coverage'])<1e-12
        videos.append(result);save(OUT/'status.json',{'status':'RUNNING','stage':'CPU_oracle','completed_videos':len(videos),'completed_frames':len(seen),'pid':os.getpid()});print(json.dumps(result,ensure_ascii=False),flush=True)
    assert seen==set(ref)
    macro={k:float(np.mean([r[k] for r in videos if r[k] is not None])) for k in ['SLAB_Dice','SLAB_IoU','SLAB_GRID_Dice','SLAB_GRID_IoU','Dice_delta','IoU_delta']}
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'assets':assets,'frames':len(seen),'videos':videos,'macro':macro,'elapsed_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'tests':'4 targeted tests passed before execution; per-frame conservation, reference identity/PTS/counts/coverage, subdivision monotonicity asserted','model_calls':0,'frontend_calls':0,'CUDA_calls':0}
    csvsave(OUT/'frames.csv',frame_rows);csvsave(OUT/'videos.csv',videos);save(OUT/'summary.json',summary)
    (folder/'index.html').write_text('<meta charset="utf-8"><h1>固定首／中／末帧：官方GT与整单元最优掩膜</h1><p>使用真实GT求得的表达上限，不是模型预测；灰色为几何无效区域。预览缩放不改变计分。</p>'+''.join(visuals))
    lines=['# PVST整片／片内固定方格定位表达上限','CPU、官方GT驱动的整单元最优解，不是模型成绩。', '|pair|帧|GT覆盖|SLAB Dice|GRID Dice|ΔDice|SLAB IoU|GRID IoU|ΔIoU|SLAB单元/帧|GRID单元/帧|L|','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in videos:lines.append('|'+ '|'.join(str(r[k]) for k in ['pair_id','comparable_frames','GT_coverage','SLAB_Dice','SLAB_GRID_Dice','Dice_delta','SLAB_IoU','SLAB_GRID_IoU','IoU_delta','SLAB_units_mean','SLAB_GRID_units_mean','L'])+'|')
    lines.extend(['宏平均：'+json.dumps(macro),'计分462唯一帧，所有视频保留；G=0单列且不补分。按g/n排序扫描含空前缀的所有前缀求精确最优Dice，同解IoU单调对应。有效范围与前次逐帧计数相同。','固定方格L=ceil(min(H,W)/16)，左上角原点，保留原h=.125、Z/s、valid、负数floor及边缘格，不是等体积三维体素。','执行HEAD '+identity['execution_head']+'；耗时 '+str(summary['elapsed_seconds'])+'秒；峰值RSS '+str(summary['peak_RSS_bytes']/1024**3)+'GiB；退出0。'])
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);(REPO/'docs/pvst/slice_localization_oracle_v1.md').write_text(report)
    save(OUT/'status.json',{'status':'COMPLETE','stage':'report_complete','exit_code':0});(OUT/'exit_code.txt').write_text('0\n')


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()})
        raise
