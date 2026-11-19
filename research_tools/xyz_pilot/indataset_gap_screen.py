"""Fixed localization weights on source-disjoint ANAKIN dev videos (ADR-0048).

Only wraps full-video evaluation rows, strict checkpoint loading, two cheap data
references and fixed previews. Never calls any source prepare/run or optimizer.
"""
import copy
import json
from pathlib import Path
import subprocess
import time
import traceback
import zipfile
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .data import OUT as OBSERVATIONS, REPO, read
from .insource_model import XYZLocalization, ImageOnly
from .insource_probe import OUT as SOURCE, PartitionCache, evaluate, partition_history
from .readout_adaptation import add_area, METRICS
from .joint_experiment import save, digest, preview
from .experiment import csvsave
from research_tools.rgbd_slice.experiment import state_hash
from research_tools.v8.spatial_localization.metrics import spatial_frame
from scripts.review_anakin import numeric_members, mask_array

OUT=SOURCE.parent/'pvst_xyz_indataset_gap_screen_v1'
ADAPT=SOURCE.parent/'pvst_xyz_readout_adaptation_v1'
DEV=['an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4']
CONDITIONS=[('I10','IMAGE_ONLY',SOURCE/'IMAGE_ONLY_epoch10.pt',10260),
            ('A10','XYZ_CURRENT',SOURCE/'XYZ_CURRENT_epoch10.pt',10260),
            ('B10','XYZ_TEMPORAL',SOURCE/'XYZ_TEMPORAL_epoch10.pt',10260),
            ('B13','XYZ_TEMPORAL',ADAPT/'B_FULL_CONTINUE_epoch13.pt',13338)]


def full_video_rows(manifest,training):
    rows=copy.deepcopy([r for r in manifest['rows'] if r['split']=='dev'])
    assert len(rows)==8 and {r['pair_id'] for r in rows}==set(DEV)
    assert not {r['source_id'] for r in rows}&{r['source_id'] for r in training['rows']}
    assert not {r['video_sha256'] for r in rows if r['role']=='real'}&{r['video_sha256'] for r in training['rows'] if r['role']=='real'}
    for r in rows:
        n=r['actual_frames'];r['evaluation_range']=[0,n];r['evaluation_targets']=list(range(n));r['preview_targets']=[0,n//2,n-1]
        assert partition_history(r,0,'evaluation')==[None]*7+[0]
    return rows


def load_model(path,mode,updates,c):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    expected='B_FULL_CONTINUE' if updates==13338 else mode
    assert cp['condition']==expected and cp['updates']==updates
    original=read(SOURCE/'run_identity.json')
    assert cp['identity']==(read(ADAPT/'identity.json') if updates==13338 else original)
    if updates==13338:assert cp['identity']['source_run_identity']==original and read(ADAPT/'config.json')['original']==c
    reference=XYZLocalization(c['h']);m=ImageOnly(reference) if mode=='IMAGE_ONLY' else reference
    m.load_state_dict(cp['state_dict'],strict=True);assert state_hash(m.state_dict())==state_hash(cp['state_dict'])
    assert not hasattr(m,'frame') and not hasattr(m,'evidence')
    return m.cuda().eval(),dict(path=str(path),sha256=digest(path),condition=expected,input_mode=mode,updates=updates,
        state_sha256=state_hash(cp['state_dict']),initial_sha256=cp['initial_sha256'],checkpoint_identity=cp['identity'])


def label_reference(gt,work_gt,valid):
    """Original nearest downsample then output-path bilinear restore, at .5."""
    hw=work_gt.shape
    expected=cv2.resize(gt.astype(np.uint8),(hw[1],hw[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
    assert np.array_equal(expected,work_gt),'CACHED_WORK_GT_MISMATCH'
    restored=F.interpolate(torch.from_numpy(work_gt.astype(np.float32))[None,None],size=gt.shape,mode='bilinear',align_corners=False)[0,0].numpy()>=.5
    overlap=spatial_frame(restored,gt)
    native_valid=cv2.resize(valid.astype(np.uint8),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
    g=int(gt.sum());wg=int(work_gt.sum())
    return dict(label_roundtrip_Dice=overlap['dice'],label_roundtrip_IoU=overlap['iou'],native_GT_pixels=g,
        work_GT_pixels=wg,native_GT_geometry_coverage=float((native_valid&gt).sum()/g) if g else None,
        work_GT_geometry_coverage=float((valid&work_gt).sum()/wg) if wg else None,
        GT_lost_at_work_grid=bool(g and not wg))


def audit_data(rows):
    records=[]
    for r in rows:
        if r['role']!='fake':continue
        with zipfile.ZipFile(r['mask_zip']) as z:
            members=numeric_members(z.namelist());assert len(members)==r['actual_frames']
            for i in r['evaluation_targets']:
                e=torch.load(OBSERVATIONS/'cache'/r['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
                assert e['frame_index']==i and e['pts']==r['pts'][i] and e['native_hw']==r['native_hw']
                gt=mask_array(z.read(members[i][1]))[1];assert list(gt.shape)==r['native_hw'] and int(gt.sum())==e['native_GT_pixels']
                records.append(dict(pair_id=r['pair_id'],video_id=r['video_id'],frame_index=i,pts=r['pts'][i],**label_reference(gt,e['gt_work'].numpy(),e['valid'].numpy())))
    videos=[]
    for pair in DEV:
        fs=[f for f in records if f['pair_id']==pair];v=dict(pair_id=pair,frames=len(fs),positive_GT_frames=sum(f['native_GT_pixels']>0 for f in fs),GT_lost_at_work_grid_frames=sum(f['GT_lost_at_work_grid'] for f in fs))
        for k in ['label_roundtrip_Dice','label_roundtrip_IoU','native_GT_geometry_coverage','work_GT_geometry_coverage']:
            vals=[f[k] for f in fs if f[k] is not None];v[k]=float(np.mean(vals)) if vals else None
        videos.append(v)
    csvsave(OUT/'data_reference_frames.csv',records);csvsave(OUT/'data_reference_sources.csv',videos)
    result=dict(frames=len(records),videos=videos,coverage_rule='Actual model work-grid valid nearest-resized to native grid, intersection with official GT / native GT; work-grid coverage separately recorded.',label_rule='Official native GT -> original cv2 nearest work GT (exact cached comparison) -> original output bilinear align_corners=False -> threshold >=0.5. Scale-loss reference, not model upper bound.')
    save(OUT/'data_reference.json',result);return result


def fixed_preview(rows,predictions):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    html=['<!doctype html><meta charset="utf-8"><title>PVST独立开发视频评价</title><h1>独立来源开发：固定24目标</h1><p>列依次I10、A10、B10、B13。各大图RGB/GT/概率（固定灰阶0–1），下图0.5二值。仅预览缩放，GT非三维异常真值；未按效果挑帧。</p>']
    for name,_,_,_ in CONDITIONS:
        assert len(predictions[name])==24
        preview(rows,predictions[name],name,'evaluation',output_root=OUT)
        for (vid,i),a in predictions[name].items():
            assert a.dtype==np.float32
            b=(a>=0).astype(np.uint8)*255;h,w=b.shape;f=min(1.,480/w)
            Image.fromarray(b).resize((round(w*f),round(h*f)),Image.Resampling.NEAREST).save(folder/'evaluation'/f'{name}_{vid}_{i:05d}_binary.png')
    for r in rows:
        for i in r['preview_targets']:
            html.append(f'<h3>{r["pair_id"]}/{r["role"]} 帧{i} PTS {r["pts"][i]:.6f}s</h3><div style="display:flex">')
            for name,_,_,_ in CONDITIONS:
                stem=f'evaluation/{name}_{r["video_id"]}_{i:05d}'
                html.append(f'<div style="width:25%">{name}<a href="{stem}.png"><img width="100%" src="{stem}.png"></a><img width="33%" src="{stem}_binary.png"></div>')
            html.append('</div>')
    (folder/'index.html').write_text('\n'.join(html))


def run():
    started=time.monotonic();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    c=read(SOURCE/'config.json');training=read(SOURCE/'protocol.json');manifest=read(OBSERVATIONS/'manifest.json')
    assert c['scale']==12.880994319915771 and c['h']==.5
    rows=full_video_rows(manifest,training);assert sum(len(r['evaluation_targets']) for r in rows)==824
    missing=[str(OBSERVATIONS/'cache'/r['video_id']/f'{i:05d}.pt') for r in rows for i in r['evaluation_targets'] if not(OBSERVATIONS/'cache'/r['video_id']/f'{i:05d}.pt').is_file()]
    if missing:save(OUT/'asset_gap.json',dict(missing=missing));raise FileNotFoundError(str(missing[:3]))
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_manifest_sha256=digest(OBSERVATIONS/'manifest.json'),source_protocol_sha256=digest(SOURCE/'protocol.json'),source_config_sha256=digest(SOURCE/'config.json'),source_initialization=read(SOURCE/'initialization.json'),new_training_updates=0,new_frontend_frames=0)
    save(OUT/'identity.json',identity);save(OUT/'config.json',c);save(OUT/'evaluation_manifest.json',dict(rows=rows,original_manifest=str(OBSERVATIONS/'manifest.json'),training_coverage='Only original seven train sources first70%, 1026 targets/epoch, ten epochs; dev full videos, no 70/30 split.'))
    data=audit_data(rows);assert data['frames']==412
    cache=PartitionCache(c);results={};preds={};cost={};checkpoints={}
    for name,mode,path,updates in CONDITIONS:
        save(OUT/'status.json',dict(status='RUNNING',stage='evaluation',condition=name));print('EVALUATE',name,flush=True)
        m,cp=load_model(path,mode,updates,c);before=state_hash(m.state_dict());torch.cuda.reset_peak_memory_stats()
        ev,images=evaluate(m,cache,rows,mode,capture=True,capture_dtype=np.float32);add_area(ev)
        assert ev['scored_frames']==824 and sum(f['GT_pixels']>0 for f in ev['frames'])==412
        assert before==state_hash(m.state_dict())
        save(OUT/f'{name}.json',ev);csvsave(OUT/f'{name}_frames.csv',ev['frames']);csvsave(OUT/f'{name}_videos.csv',ev['videos'])
        results[name]=ev;preds[name]=images;checkpoints[name]=cp
        cost[name]=dict(seconds=ev['seconds'],peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),weights_buffers_unchanged=True)
        del m;torch.cuda.empty_cache()
    assert len({x['frame_identity_sha256'] for x in results.values()})==1
    assert checkpoints['A10']['initial_sha256']==checkpoints['B10']['initial_sha256']==identity['source_initialization']['XYZ_sha256']
    assert checkpoints['I10']['initial_sha256']==identity['source_initialization']['IMAGE_ONLY_sha256']
    fixed_preview(rows,preds)
    comparisons=[]
    for pair in DEV:
        item={'pair_id':pair}
        for name in results:
            fake=next(v for v in results[name]['videos'] if v['pair_id']==pair and v['role']=='fake')
            real=next(v for v in results[name]['videos'] if v['pair_id']==pair and v['role']=='real')
            for k in METRICS:item[name+'_'+k]=real[k] if k=='real_FPR' else fake[k]
        for a,b,label in [('A10','I10','A_minus_I'),('B10','A10','B_minus_A'),('B13','B10','B13_minus_B10')]:
            for k in METRICS:item[label+'_'+k]=item[a+'_'+k]-item[b+'_'+k]
        comparisons.append(item)
    csvsave(OUT/'source_comparison.csv',comparisons)
    save(OUT/'summary.json',dict(status='COMPLETE',exit_code=0,identity=identity,config=c,checkpoints=checkpoints,results=results,data_reference=data,cost=cost,wall_seconds=time.monotonic()-started,preview_targets=24,training_updates=0,new_geometry_frames=0,new_visual_frames=0))
    save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


def main():
    if (OUT/'identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':
            s=read(OUT/'summary.json');assert all(digest(cp['path'])==cp['sha256'] for cp in s['checkpoints'].values());print('EXISTING_COMPLETE_REUSED');return
        raise SystemExit('Existing incomplete run preserved; inspect before retry')
    try:run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error)));raise


if __name__=='__main__':main()
