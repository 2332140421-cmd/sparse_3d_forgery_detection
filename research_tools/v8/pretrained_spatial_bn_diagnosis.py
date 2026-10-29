"""Fixed two-pair, best-checkpoint BN-statistics diagnosis; no training/providers."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from scripts.prepare_v8_assets import digest
from research_tools.v8.pretrained_spatial_context.assets import CachedVideo
from research_tools.v8.pretrained_spatial_context.model import SpatialModel,CONDITIONS
from research_tools.v8.spatial_localization.data import to_source
from research_tools.v8.spatial_localization.metrics import spatial_frame,to_source_rgb
from research_tools.v8.spatial_localization.joint_data import save

PAIRS=('an:95qlg9','an:40y1x9')
MODES=('RUNNING_STATS','BATCH_STATS')


def parameter_sha(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_bn_mode(model,mode):
    if mode not in MODES:raise ValueError('BN_MODE')
    model.eval().requires_grad_(False)
    layers={name:m for name,m in model.decoder.named_modules() if isinstance(m,nn.BatchNorm2d)}
    if not layers or any(not m.track_running_stats for m in layers.values()):raise ValueError('TRACKED_DECODER_BN_REQUIRED')
    if mode=='BATCH_STATS':
        for m in layers.values():m.train(True)  # Only BN, never the whole decoder/model.
    if any(m.training for m in model.modules() if m not in layers.values()):raise ValueError('NON_BN_MODULE_NOT_EVAL')
    return layers


def select(rows):
    selected=[]
    for pair in PAIRS:
        rr=[r for r in rows if r['pair_id']==pair]
        if len(rr)!=2 or {r['role'] for r in rr}!={'real','fake'} or any(r['split']!='train' for r in rr):
            raise ValueError('FIXED_TRAIN_PAIR_REQUIRED:'+pair)
        if any((r['source_hw'][0]>r['source_hw'][1])!=(pair==PAIRS[1]) for r in rr):raise ValueError('PAIR_ORIENTATION')
        selected.extend(rr)  # Preserve original manifest order inside each pair.
    return selected


def compare_existing(record,expected):
    differences={}
    for key in ['spatial_dice','spatial_iou','empty_spatial_fpr']:
        x,y=record[key],expected[key]
        if (x is None)!=(y is None):raise ValueError('EXISTING_METRIC_VALIDITY:'+key)
        differences[key]=None if x is None else x-y
    maximum=max((abs(v) for v in differences.values() if v is not None),default=0.)
    if maximum>1e-5:raise ValueError('RUNNING_STATS_REPRODUCTION_MISMATCH:'+str(differences))
    return {'differences':differences,'max_abs':maximum,'tolerance':1e-5}


@torch.no_grad()
def evaluate_video(model,config,source,out,row,fit):
    device=next(model.parameters()).device
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.monotonic();video=CachedVideo(config,source,row,fit,model.context)
    h=torch.zeros(256,128,device=device);active=torch.zeros(256,dtype=torch.bool,device=device)
    dices=[];ious=[];empty=[];cm=np.zeros((2,2),np.int64);inside_sum=outside_sum=0.
    inside_count=outside_count=0;kept={int(row['actual_frames']*f) for f in [.25,.5,.75]}
    out.mkdir(parents=True,exist_ok=True)
    try:
        for start in range(0,row['actual_frames'],2):
            a=video.chunk(start,min(start+2,row['actual_frames']))
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['amp']):
                h,active,z=model.chunk(a,h,active,start==0,row['mapping256'],row['mapping512'])
            if not torch.isfinite(z).all():raise ValueError('NONFINITE_LOGITS')
            for j,canvas in enumerate(z.float().cpu().numpy()):
                fi=start+j;gt=video.native_gt(fi)
                if gt is None:raise ValueError('MISSING_GT_NOT_BACKGROUND')
                native=to_source(canvas,row['mapping512']);prediction=native>=0
                scored=spatial_frame(prediction,gt);cm+=np.asarray(scored['confusion_matrix'])
                if gt.any():dices.append(scored['dice']);ious.append(scored['iou'])
                else:empty.append(scored['empty_fpr'])
                inside_sum+=float(native[gt].sum(dtype=np.float64));outside_sum+=float(native[~gt].sum(dtype=np.float64))
                inside_count+=int(gt.sum());outside_count+=int((~gt).sum())
                if fi in kept:
                    np.savez_compressed(out/f'{row["video_id"]}_{fi}.npz',logits=canvas.astype(np.float32),native_gt=gt,
                        rgb512=np.asarray(video.base.arrays['rgb512'][fi]),frame_index=fi,pts_s=row['pts'][fi]['pts_s'],
                        metadata=np.array(json.dumps({k:row[k] for k in ['video_id','pair_id','source_id','split','role','mapping256','mapping512']})))
    finally:video.close()
    if device.type=='cuda':torch.cuda.synchronize()
    tn,fp,fn,tp=[int(x) for x in cm.ravel()]
    return {**{k:row[k] for k in ['video_id','pair_id','role','split']},'frames':row['actual_frames'],
        'spatial_dice':float(np.mean(dices)) if dices else None,'spatial_iou':float(np.mean(ious)) if ious else None,
        'gt_pixel_recall':tp/(tp+fn) if tp+fn else None,'predicted_area_fraction':(tp+fp)/int(cm.sum()),
        'empty_spatial_fpr':float(np.mean(empty)) if empty else None,
        'gt_mean_logit':inside_sum/inside_count if inside_count else None,'outside_mean_logit':outside_sum/outside_count if outside_count else None,
        'positive_gt_frames':len(dices),'empty_gt_frames':len(empty),'pixel_confusion_matrix':cm.tolist(),'seconds':time.monotonic()-started}


def sample_index(out,rows):
    from PIL import Image,ImageDraw
    import matplotlib
    entries=[];details=[];directory=out/'comparison';directory.mkdir(exist_ok=True)
    for c in CONDITIONS:
        for r in rows:
            for fi in [int(r['actual_frames']*f) for f in [.25,.5,.75]]:
                arrays=[]
                for mode in MODES:
                    with np.load(out/c/mode/f'{r["video_id"]}_{fi}.npz',allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
                if not np.array_equal(arrays[0]['native_gt'],arrays[1]['native_gt']):raise ValueError('PAIRED_GT')
                gt=arrays[0]['native_gt'];zs=[to_source(a['logits'],r['mapping512']) for a in arrays]
                rgb=to_source_rgb(arrays[0]['rgb512'],r['mapping512']);gt_rgb=np.repeat(gt[...,None]*np.uint8(255),3,-1)
                binary=[np.repeat((z>=0)[...,None]*np.uint8(255),3,-1) for z in zs]
                heat=[(matplotlib.colormaps['coolwarm'](np.clip((z+6)/12,0,1))[...,:3]*255).astype(np.uint8) for z in zs]
                height=max(1,round(rgb.shape[0]*320/rgb.shape[1]));panel=Image.new('RGB',(1280,2*(height+28)),'white');draw=ImageDraw.Draw(panel)
                for ri,images in enumerate([[rgb,gt_rgb,*binary],[rgb,gt_rgb,*heat]]):
                    for col,a in enumerate(images):
                        title=['RGB','GT edit region',*MODES][col]+(' logit [-6,+6]' if ri and col>=2 else '')
                        draw.text((col*320+3,ri*(height+28)+5),title,fill='black')
                        resample=Image.Resampling.NEAREST if col==1 or (ri==0 and col>=2) else Image.Resampling.BILINEAR
                        panel.paste(Image.fromarray(a).resize((320,height),resample=resample),(col*320,ri*(height+28)+28))
                name=f'{c}_{r["video_id"]}_{fi}.png';panel.save(directory/name)
                entries.append('<h3>'+html.escape(f'{c} {r["pair_id"]} {r["role"]} frame{fi} PTS{float(arrays[0]["pts_s"]):.6f}')+'</h3><img loading="lazy" style="max-width:100%" src="comparison/'+name+'">')
                delta=zs[1]-zs[0];pa,pb=[z>=0 for z in zs];union=(pa|pb).sum()
                corr=float(np.corrcoef(zs[0].reshape(-1),zs[1].reshape(-1))[0,1]) if all(z.std()>0 for z in zs) else None
                details.append({'condition':c,'pair_id':r['pair_id'],'video_id':r['video_id'],'role':r['role'],'frame_index':fi,
                    'delta_mean':float(delta.mean()),'delta_std':float(delta.std()),'logit_correlation':corr,
                    'threshold_flip_fraction':float((pa!=pb).mean()),'between_mode_mask_iou':float((pa&pb).sum()/union) if union else None})
    save(out/'fixed_sample_differences.json',details)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>BN统计模式诊断：原图 | GT | RUNNING_STATS | BATCH_STATS</h1><p>固定25/50/75%；阈值0，热图绝对色阶[-6,+6]。无Dropout、无训练；当前批统计结果不替换正式成绩。</p>'+''.join(entries))


def report(out,records):
    keys=['condition','mode','pair_id','role','frames','spatial_dice','spatial_iou','gt_pixel_recall','predicted_area_fraction','paired_real_spatial_fpr','gt_mean_logit','outside_mean_logit','seconds']
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    fmt=lambda v:'—' if v is None else f'{v:.6f}'
    lines=['# V8预训练定位BatchNorm统计模式诊断','',
        '仅两个best、固定训练两pair/4视频480帧；无训练/恢复/前端或DINO提取，不读取开发媒体进行校准。',
        '整模型eval、Dropout关闭；BATCH_STATS仅decoder五层BN使用当前连续2帧统计，末块保留实际帧数。',
        'Dice/IoU按正GT帧平均；GT召回与GT内外logit按各视频原生像素pooled统计；面积包含全部帧。',
        '空GT保留背景统计、real无正GT的指标为null；先原图回映再以logit≥0阈值。对应real FP单列。','',
        '|模型|模式|pair/role|Dice|IoU|GT召回|预测面积|对应real FP|GT内logit|GT外logit|秒|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append('|'+ '|'.join([r['condition'],r['mode'],r['pair_id']+'/'+r['role']]+[fmt(r[k]) for k in keys[5:]])+'|')
    lines.extend(['','RUNNING_STATS复现差值、参数/文件完整性见results.json；固定样例见index.html。',
        '固定样例delta_std、相关系数和阈值翻转见fixed_sample_differences.json，仅描述该时刻，不作为全视频位置指标。',
        '四条训练视频上的效应不等于统计显著性或开发泛化，不据此替换原正式结果，也不自动校准/修改训练。'])
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('docs/v8/pretrained_spatial_context_config.json'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();config=json.loads(args.config.read_text())
    source=Path(config['output']);out=args.output.resolve()
    if source.resolve()==out or source.resolve() in out.parents or out in source.resolve().parents:raise ValueError('ISOLATED_OUTPUT_REQUIRED')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('NONEMPTY_DIAGNOSIS_OUTPUT; refuse accidental repeated execution')
    if config['chunk_frames']!=2:raise ValueError('FIXED_CHUNK2')
    torch.set_num_threads(4);started=time.monotonic()
    rows=select(json.loads((source/'manifest.json').read_text())['rows']);fit=json.loads((Path(config['source_run'])/'statistics.json').read_text())
    # Import the official decoder alone: no get_network(), visual backbone, or download.
    sys.path.insert(0,config['official_code'])
    from ainpaint.modules.context_decoder import HighResContextHead
    files={source/name for name in ['manifest.json','result_summary.json','epoch0.json','report_zh.md','progress.json','source_references.json']}
    files.add(Path(config['source_run'])/'statistics.json')
    for c in CONDITIONS:files.update(source/c/name for name in ['best.pt','summary.json','history.json'])
    for r in rows:
        folder=source/'features'/r['video_id'];meta=json.loads((folder/'complete.json').read_text())
        if meta['checkpoint_sha256']!=config['checkpoint_sha256'] or meta['video_sha256']!=r['video_sha256'] or meta['shape']!=[r['actual_frames'],1024,37,37] or meta['dtype']!='float16':raise ValueError('CACHED_FEATURE_IDENTITY')
        if meta['pts_sha256']!=hashlib.sha256(json.dumps(r['pts'],sort_keys=True).encode()).hexdigest():raise ValueError('CACHED_PTS')
        if digest(folder/'patch.npy')!=meta['sha256']:raise ValueError('CACHED_FEATURE_SHA')
        files.update(p for p in folder.iterdir() if p.is_file())
        files.update(p for p in (Path(config['source_run'])/'cache'/r['video_id']).iterdir() if p.is_file())
        files.add(Path(r['mask_zip']))
    before={str(p):digest(p) for p in sorted(files)}
    save(out/'input_identity.json',{'config':config,'rows':rows,'source_file_sha256':before,'diagnostic_code_sha256':digest(Path(__file__))})
    records=[];checks=[]
    for c,epoch in zip(CONDITIONS,[12,4]):
        summary=json.loads((source/c/'summary.json').read_text());previous={r['video_id']:r for r in summary['best']['train']['records']}
        shared=None
        for mode in MODES:
            cp=torch.load(source/c/'best.pt',map_location='cpu',weights_only=False)
            if cp['epoch']!=epoch or summary['best_epoch']!=epoch or cp['identity']['condition']!=c:raise ValueError('FIXED_BEST_CHECKPOINT')
            if cp['identity']['config']!=config:raise ValueError('CHECKPOINT_CONFIG_IDENTITY')
            model=SpatialModel(c,HighResContextHead(1024,1,norm='BN'),fit['statistics']);model.load_state_dict(cp['model'],strict=True)
            del cp
            model.cuda();layers=set_bn_mode(model,mode);sha=parameter_sha(model)
            if shared is not None and sha!=shared:raise ValueError('MODES_DIFFER_IN_WEIGHTS')
            shared=sha;counts={k:int(m.num_batches_tracked) for k,m in layers.items()}
            for r in rows:
                rec=evaluate_video(model,config,source,out/c/mode,r,fit);rec.update(condition=c,mode=mode)
                if mode=='RUNNING_STATS':rec['existing_evaluation_comparison']=compare_existing(rec,previous[r['video_id']])
                records.append(rec)
                print(json.dumps({k:rec[k] for k in ['condition','mode','pair_id','role','spatial_dice','seconds']}),flush=True)
            if sha!=parameter_sha(model) or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('PARAMETERS_NOT_FROZEN')
            after_counts={k:int(m.num_batches_tracked) for k,m in layers.items()}
            if mode=='RUNNING_STATS' and counts!=after_counts:raise ValueError('RUNNING_BN_BUFFERS_CHANGED')
            checks.append({'condition':c,'mode':mode,'parameters_sha256':sha,'parameters_unchanged':True,'dropout_and_non_BN_eval':True,
                'BN_num_batches_before':counts,'BN_num_batches_after':after_counts})
            del model;torch.cuda.empty_cache()
    for r in records:
        real=next(x for x in records if x['condition']==r['condition'] and x['mode']==r['mode'] and x['pair_id']==r['pair_id'] and x['role']=='real')
        r['paired_real_spatial_fpr']=real['empty_spatial_fpr']
    if any(digest(Path(p))!=sha for p,sha in before.items()):raise ValueError('SOURCE_FILES_CHANGED')
    sample_index(out,rows);report(out,records)
    save(out/'results.json',{'status':'COMPLETE','records':records,'checks':checks,'source_files_unchanged':True,
        'seconds':time.monotonic()-started,'BN_batch_buffers':'memory-only; discarded; no calibrated checkpoint saved'})


if __name__=='__main__':main()
