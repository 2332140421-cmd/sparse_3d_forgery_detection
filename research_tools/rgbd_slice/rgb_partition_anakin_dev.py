"""ADR-0038 fixed 8/4 ANAKIN comparison; no frontend or implicit resume.

Caller: CLI parent. Reuses build/update/evaluate; outputs independent frozen
identity, two bounded worker runs and a matched epoch12 report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
from torch.nn import functional as F

from .data import Video, save
from .experiment import REPO, evaluate, forward, git, load_clip, optimizer, state_hash
from .set_fit_check import SOURCE, build, update_once
from scripts.prepare_v8_assets import digest

OUT = SOURCE.parent / 'pvst_rgb_partition_anakin_dev_v1'
CONDITIONS = ('RGBD', 'RGB_DEPTH_PARTITION')
BRANCH = 'pvst-rgb-partition-anakin-dev'
DOC = 'docs/pvst/rgb_partition_anakin_dev_v1.md'
LIMIT = 4 * 3600


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def epoch_order(clips, seed, epoch):
    order = list(clips)
    random.Random(seed + epoch).shuffle(order)
    return order


def source_key(url):
    p = urlparse(url)
    host = (p.hostname or '').removeprefix('www.')
    if host in ('youtube.com', 'youtu.be'):
        return 'youtube:' + (parse_qs(p.query)['v'][0] if host == 'youtube.com' else p.path.strip('/'))
    return host + p.path.rstrip('/')


def validate_split(rows, clips):
    if len(rows) != 24 or len({r['video_id'] for r in rows}) != 24:
        raise ValueError('EXPECTED_24_UNIQUE_VIDEOS')
    by_split = {}
    for split, expected in [('train', 8), ('dev', 4)]:
        rr = [r for r in rows if r['split'] == split]
        pairs = {r['pair_id'] for r in rr}
        if len(rr) != 2 * expected or len(pairs) != expected:
            raise ValueError('PAIR_COUNTS:' + split)
        for pair in pairs:
            pair_rows = [r for r in rr if r['pair_id'] == pair]
            if (sorted(r['role'] for r in pair_rows) != ['fake', 'real']
                    or len({r['source_id'] for r in pair_rows}) != 1
                    or len({source_key(r['original_url']) for r in pair_rows}) != 1):
                raise ValueError('PAIR_SOURCE:' + pair)
        by_split[split] = rr
    for field in ('source_id', 'video_sha256', 'video_id'):
        if {r[field] for r in by_split['train']} & {r[field] for r in by_split['dev']}:
            raise ValueError('SOURCE_LEAKAGE:' + field)
    if ({source_key(r['original_url']) for r in by_split['train']}
            & {source_key(r['original_url']) for r in by_split['dev']}):
        raise ValueError('SOURCE_URL_LEAKAGE')
    if len(clips) != 48:
        raise ValueError('EXPECTED_48_FROZEN_CLIPS')
    coverage = {}
    for r in by_split['train']:
        cc = [c for c in clips if c['video_id'] == r['video_id']]
        expected = sorted(set(int(x) for x in np.linspace(0, r['actual_frames'] - 24, 3)))
        if sorted(c['start'] for c in cc) != expected or any(c['end'] - c['start'] != 24 for c in cc):
            raise ValueError('FROZEN_CLIPS:' + r['video_id'])
        coverage[r['video_id']] = len({i for c in cc for i in range(c['start'], c['end'])})
    if any(c['video_id'] not in coverage for c in clips):
        raise ValueError('NONTRAIN_CLIP')
    return {'train_videos': 16, 'train_pairs': 8, 'dev_videos': 8, 'dev_pairs': 4,
            'train_full_frames': sum(r['actual_frames'] for r in by_split['train']),
            'dev_frames': sum(r['actual_frames'] for r in by_split['dev']),
            'clips': len(clips), 'frame_presentations_per_epoch': sum(c['end'] - c['start'] for c in clips),
            'unique_training_clip_frames': sum(coverage.values()), 'unique_frames_per_video': coverage,
            'source_id_url_media_sha_isolation': True}


def prepare():
    started = time.monotonic()
    c = read(SOURCE / 'config.json')
    c.update(geometry_root=str(SOURCE / 'geometry'), initialization_root=str(SOURCE),
             epochs=12, conditions=list(CONDITIONS), output=str(OUT), worker_seconds=LIMIT,
             protocol='rgb-partition-anakin-dev-v1', patience=None, min_epochs=None)
    if [c[k] for k in ['seed', 'scale', 'h', 'd0', 'D', 'encoder_lr', 'new_lr']] != [20260909, 1.9888148307800293, .125, 0, 512, 1e-5, 1e-4]:
        raise ValueError('CONFIG_CHANGED')
    m = read(SOURCE / 'manifest.json')
    counts = validate_split(m['rows'], m['clips'])
    save(OUT / 'manifest.json', m)
    save(OUT / 'config.json', c)
    assets = []
    for r in m['rows']:
        print('VERIFY', r['video_id'], flush=True)
        if r['dataset'] != 'ANAKIN' or digest(Path(r['video_path'])) != r['video_sha256']:
            raise ValueError('VIDEO_IDENTITY:' + r['video_id'])
        if r['role'] == 'fake' and digest(Path(r['mask_zip'])) != r['mask_sha256']:
            raise ValueError('MASK_IDENTITY:' + r['video_id'])
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
            '-show_entries', 'frame=best_effort_timestamp_time,width,height', '-of', 'json', r['video_path']]))['frames']
        if len(probe) != r['actual_frames'] or len(r['pts']) != r['actual_frames']:
            raise ValueError('FRAME_COUNT:' + r['video_id'])
        for i, (a, b) in enumerate(zip(probe, r['pts'])):
            if abs(float(a['best_effort_timestamp_time']) - b['pts_s']) > 1e-6 or [a['height'], a['width']] != r['source_hw']:
                raise ValueError('FRAME_PTS_SIZE:' + r['video_id'] + ':' + str(i))
        if any(b['pts_s'] <= a['pts_s'] for a, b in zip(r['pts'], r['pts'][1:])):
            raise ValueError('NONMONOTONIC_PTS')
        folder = Path(c['geometry_root']) / r['video_id']
        meta = read(folder / 'complete.json')
        if (meta['frames'] != r['actual_frames'] or meta['pts'] != r['pts']
                or meta['video_sha256'] != r['video_sha256'] or meta['native_hw'] != r['source_hw']
                or max(meta['output_hw']) <= 256):
            raise ValueError('GEOMETRY_IDENTITY:' + r['video_id'])
        v = Video(r, c)
        try:
            if (v.xyz.shape != (r['actual_frames'], *meta['output_hw'], 3) or v.valid.shape != v.xyz.shape[:-1]
                    or v.xyz.dtype != np.float32 or v.valid.dtype != bool):
                raise ValueError('GEOMETRY_SHAPE')
            positives = 0
            for i in range(r['actual_frames']):
                if not np.isfinite(np.asarray(v.xyz[i])[v.valid[i]]).all():
                    raise ValueError('NONFINITE_VALID_GEOMETRY')
                positives += bool(v.gt(i).any())
        finally:
            v.close()
        # Full content hashes bind the actual read-only numerical caches, not only metadata.
        cache_files = [folder/'complete.json', folder/'xyz.npy', folder/'valid.npy',
                       Path(c['cache_root'])/r['video_id']/'rgb.npy', Path(c['cache_root'])/r['video_id']/'complete.json']
        assets.append({'video_id': r['video_id'], 'positive_gt_frames': positives,
                       'cache_sha256': {str(p): digest(p) for p in cache_files}})
    orders = {str(e): epoch_order(m['clips'], c['seed'], e) for e in range(1, 13)}
    save(OUT / 'orders.json', orders)
    identity = {'head': git('rev-parse', 'HEAD'), 'source_manifest_sha256': digest(SOURCE/'manifest.json'),
                'frozen_manifest_sha256': digest(OUT/'manifest.json'), 'config_sha256': digest(OUT/'config.json'),
                'common_initial_sha256': digest(SOURCE/'common_initial.pt'), 'orders_sha256': digest(OUT/'orders.json'),
                'code': {str(p.relative_to(REPO)): digest(p) for folder in ['research_tools/rgbd_slice', 'research_tools/depth_slice'] for p in (REPO/folder).glob('*.py')},
                'assets': assets, 'counts': counts, 'preparation_seconds': time.monotonic()-started}
    identity['identity'] = fingerprint(identity)
    save(OUT/'identity.json', identity)
    return c, m, identity


def augment_evaluation(result):
    for r in result['records']:
        z = torch.tensor(r['scores'], dtype=torch.float32)
        y = torch.tensor(r['labels'], dtype=torch.float32)
        if len(z) != r['frames'] or not torch.isfinite(z).all():
            raise ValueError('INVALID_EVALUATION_FRAME_COVERAGE')
        r['frame_bce_fp32'] = float(F.binary_cross_entropy_with_logits(z, y))
    # Frame mean, matching equal full native frames for the detection BCE diagnostic.
    result['metrics']['frame_bce_fp32'] = sum(r['frame_bce_fp32']*r['frames'] for r in result['records']) / sum(r['frames'] for r in result['records'])
    return result


def worker(condition):
    folder = OUT / condition
    started = time.monotonic()
    result = {'status': 'FAILED', 'condition': condition, 'updates': 0, 'last_epoch': 0, 'exit_code': 1}
    def progress(stage, **extra):
        if time.monotonic()-started >= LIMIT:
            raise TimeoutError('FOUR_HOUR_LIMIT')
        save(folder/'progress.json', {'status': 'RUNNING', 'pid': os.getpid(), 'stage': stage,
                                     'wall_s': time.monotonic()-started, **extra})
        print(stage, extra, flush=True)
    try:
        torch.set_num_threads(4)
        c, m, identity = read(OUT/'config.json'), read(OUT/'manifest.json'), read(OUT/'identity.json')
        save(folder/'config.json', {**c, 'condition':condition})
        save(folder/'launch.json', {'pid':os.getpid(),'head':git('rev-parse','HEAD'),
                                   'gpu':torch.cuda.get_device_name(),'free_total':torch.cuda.mem_get_info(),
                                   'started_unix':time.time()})
        if git('rev-parse','HEAD') != identity['head'] or digest(OUT/'manifest.json') != identity['frozen_manifest_sha256']:
            raise ValueError('EXECUTION_IDENTITY_CHANGED')
        for rel, sha in identity['code'].items():
            if digest(REPO/rel) != sha:
                raise ValueError('EXECUTION_SOURCE_CHANGED:' + rel)
        progress('model')
        model = build(c, folder, rgb_only=condition == 'RGB_DEPTH_PARTITION')
        init = read(folder/'initialization.json')
        if init['model_sha256'] != read(OUT/'preflight.json')['initial_model_sha256']:
            raise ValueError('UNMATCHED_INITIAL_MODEL')
        opt = optimizer(model, c)
        bn = state_hash(dict(model.local.encoder.named_buffers()))
        dev = [r for r in m['rows'] if r['split'] == 'dev']
        train = {r['video_id']: r for r in m['rows'] if r['split'] == 'train'}
        torch.cuda.reset_peak_memory_stats()
        def assess(epoch, ap=False):
            value = augment_evaluation(evaluate(model, dev, c, progress, pixel_ap=ap))
            if sum(r['frames'] for r in value['records']) != identity['counts']['dev_frames']:
                raise ValueError('DEV_FRAME_COUNT_CHANGED')
            save(folder/(f'epoch{epoch:02d}_dev.json' if not ap else f'epoch{epoch:02d}_dev_ap.json'), value)
            return value
        progress('epoch0');assess(0)
        history, best, best_epoch = [], -1., 0
        for epoch in range(1, 13):
            ep = time.monotonic();parts=[]
            order = read(OUT/'orders.json')[str(epoch)]
            for i, clip in enumerate(order):
                progress('train', epoch=epoch, clip=i+1, updates=result['updates'])
                fs = load_clip(train[clip['video_id']], clip, c)
                record = update_once(model, opt, fs)
                if not all(g['finite'] for g in record['gradients'].values()) or not np.isfinite(record['terms']).all():
                    raise ValueError('NONFINITE_UPDATE')
                del fs
                result['updates'] += 1;parts.append(record['terms'])
                # Keep training evidence light; full per-branch gradient logs are unnecessary here.
                record.pop('gradients')
                with (folder/'updates.jsonl').open('a') as f:
                    f.write(json.dumps({'epoch':epoch,'clip':clip,'update':result['updates'],**record})+'\n')
            value = assess(epoch)
            if state_hash(dict(model.local.encoder.named_buffers())) != bn:
                raise ValueError('BN_BUFFERS_CHANGED')
            losses = np.mean(parts, axis=0)
            improved = value['metrics']['dice'] > best
            if improved:best,best_epoch = value['metrics']['dice'],epoch
            row = {'epoch':epoch,'updates':result['updates'],'terms':losses.tolist(),'loss':float(losses.sum()),
                   'dev':value['metrics'],'epoch_seconds':time.monotonic()-ep,'cumulative_seconds':time.monotonic()-started,
                   'order_sha256':fingerprint(order),'best_epoch':best_epoch}
            history.append(row);save(folder/'history.json',history)
            state = {'state_dict':model.state_dict(),'epoch':epoch,'updates':result['updates'],
                     'condition':condition,'identity':identity['identity'],'initial_model_sha256':init['model_sha256']}
            tmp = folder/'last.pt.tmp';torch.save(state,tmp);tmp.replace(folder/'last.pt')
            if improved:
                tmp=folder/'best.pt.tmp';torch.save(state,tmp);tmp.replace(folder/'best.pt')
            result['last_epoch']=epoch
            save(folder/'result.json',{**result,'status':'RUNNING','best_epoch':best_epoch})
            print('EPOCH_COMPLETE',json.dumps(row),flush=True)
        progress('final_last_ap');last = assess(12,True)
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
        progress('final_best_ap',epoch=best_epoch)
        best_value=last if best_epoch==12 else assess(best_epoch,True)
        save(folder/'best_dev.json',best_value);save(folder/'last_dev.json',last)
        result.update(status='COMPLETE',exit_code=0,stop_reason='12_EPOCHS_576_UPDATES',best_epoch=best_epoch,
                      last_dev=last['metrics'],best_dev=best_value['metrics'],BN_unchanged=True,
                      initial_model_sha256=init['model_sha256'])
    except BaseException as error:
        result.update(status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'TIMEOUT' if isinstance(error,TimeoutError) else 'FAILED',
                      stop_reason=str(error),error=traceback.format_exc())
        (folder/'error.txt').write_text(result['error']);print(result['error'],flush=True)
    finally:
        result.update(wall_s=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        save(folder/'result.json',result);save(folder/'progress.json',{**result,'stage':'worker_finished','pid':os.getpid()})
    return result['exit_code']


def preflight(c,m):
    folder=OUT/'preflight';folder.mkdir()
    row=next(r for r in m['rows'] if r['split']=='train' and r['role']=='fake')
    clip=next(x for x in m['clips'] if x['video_id']==row['video_id'])
    fs=load_clip(row,clip,c);hashes=[];results=[]
    for condition in CONDITIONS:
        arm=folder/condition;arm.mkdir();model=build(c,arm,rgb_only=condition=='RGB_DEPTH_PARTITION').eval()
        hashes.append(read(arm/'initialization.json')['model_sha256'])
        with torch.no_grad():
            output=forward(model,fs)
            if len(output)!=24 or any(not torch.isfinite(p[k]).all() for p in output for k in ['frame','pixel']):
                raise ValueError('REAL_INPUT_PREFLIGHT')
        results.append({'condition':condition,'frames':24,'pixel_shape':list(output[0]['pixel'].shape),'finite':True})
        del output,model;torch.cuda.empty_cache()
    if len(set(hashes))!=1:raise ValueError('INITIALIZATION_MISMATCH')
    save(OUT/'preflight.json',{'status':'PASS','initial_model_sha256':hashes[0],'clip':clip,'results':results})


def run_arm(condition):
    folder=OUT/condition;folder.mkdir()
    with (folder/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--worker',condition],
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=REPO)
        save(folder/'process.json',{'pid':child.pid,'condition':condition})
        timeout=False;start=time.monotonic()
        try:code=child.wait(timeout=LIMIT)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        save(folder/'exit.json',{'exit_code':code,'timed_out':timeout,'wall_s':time.monotonic()-start})


def finish():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    identity=read(OUT/'identity.json');runs={};histories={}
    fig,ax=plt.subplots(figsize=(7,4))
    for condition in CONDITIONS:
        folder=OUT/condition
        result=read(folder/'result.json') if (folder/'result.json').exists() else {'status':'FAILED'}
        exit=read(folder/'exit.json')
        if exit['timed_out']:
            result.update(status='TIMEOUT',stop_reason='PARENT_FOUR_HOUR_LIMIT')
            if (folder/'updates.jsonl').exists():
                completed=[json.loads(line)['update'] for line in (folder/'updates.jsonl').read_text().splitlines() if line.strip()]
                result['updates']=max(completed,default=0)
        runs[condition]={'result':result,'exit':exit}
        if (folder/'history.json').exists():
            h=read(folder/'history.json');histories[condition]=h
            zero=read(folder/'epoch00_dev.json')['metrics']['dice']
            ax.plot([0]+[r['epoch'] for r in h],[zero]+[r['dev']['dice'] for r in h],marker='o',label=condition)
    ax.set(xlabel='Epoch',ylabel='Development fake-video macro Dice');ax.legend();fig.tight_layout();fig.savefig(OUT/'dev_dice.png');plt.close(fig)
    complete=all(r['result'].get('status')=='COMPLETE' and r['result'].get('updates')==576
                 and r['exit']['exit_code']==0 for r in runs.values())
    matched=complete and len({r['result']['initial_model_sha256'] for r in runs.values()})==1
    if matched:
        matched=all([r['order_sha256'] for r in histories[CONDITIONS[0]]]==[r['order_sha256'] for r in histories[name]] for name in CONDITIONS)
    status='COMPLETE' if matched else 'INCOMPLETE'
    summary={'status':status,'execution_head':identity['head'],'counts':identity['counts'],'runs':runs,'paired_fake':[]}
    lines=['# ANAKIN同数据集、来源隔离RGB-D分片匹配对照',f'状态：{status}。主比较固定第12轮；best单列；不是保留测试集结论。',
           '执行提交：'+identity['head'],'数据：'+json.dumps(identity['counts'],ensure_ascii=False),
           '共同初始化、原始48段、seed+epoch同序；完整原生像素/有效点、固定BN、BF16、三项等权loss。两组都使用相同深度分片，不能据此证明深度分片本身的增益。']
    keys=['dice','iou','pixel_ap','GT_recall','real_fpr','frame_bce_fp32']
    if matched:
        a,b=[runs[n]['result']['last_dev'] for n in CONDITIONS]
        diff={k:b[k]-a[k] for k in keys};summary['epoch12_difference']=diff
        lines+=['','|第12轮|Dice|IoU|pixel AP|GT召回|real FPR|FP32帧BCE|frame AUROC/AP|video AUROC/AP|','|---|---:|---:|---:|---:|---:|---:|---|---|']
        for name,v in [(CONDITIONS[0],a),(CONDITIONS[1],b)]:
            lines.append('|'+ '|'.join([name]+[f'{v[k]:.6f}' for k in keys]+[f"{v['frame']['auroc']:.6f}/{v['frame']['ap']:.6f}",f"{v['video']['auroc']:.6f}/{v['video']['ap']:.6f}"])+'|')
        lines.append('|新版−旧版|'+ '|'.join(f'{diff[k]:+.6f}' for k in keys)+f"|{b['frame']['auroc']-a['frame']['auroc']:+.6f}/{b['frame']['ap']-a['frame']['ap']:+.6f}|{b['video']['auroc']-a['video']['auroc']:+.6f}/{b['video']['ap']-a['video']['ap']:+.6f}|")
        aa,bb=[read(OUT/n/'last_dev.json')['records'] for n in CONDITIONS]
        lines+=['','|开发pair|旧Dice|新Dice|ΔDice|旧IoU|新IoU|ΔIoU|旧real FPR|新real FPR|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for x in aa:
            if x['role']!='fake':continue
            y=next(r for r in bb if r['video_id']==x['video_id'])
            ar=next(r for r in aa if r['pair_id']==x['pair_id'] and r['role']=='real');br=next(r for r in bb if r['pair_id']==x['pair_id'] and r['role']=='real')
            p={'pair_id':x['pair_id'],'old_dice':x['dice'],'new_dice':y['dice'],'delta_dice':y['dice']-x['dice'],
               'old_iou':x['iou'],'new_iou':y['iou'],'delta_iou':y['iou']-x['iou'],'old_real_fpr':ar['empty_fpr'],'new_real_fpr':br['empty_fpr']}
            summary['paired_fake'].append(p);lines.append('|'+ '|'.join([p['pair_id']]+[f'{v:.6f}' for k,v in p.items() if k!='pair_id'])+'|')
        dd=[p['delta_dice'] for p in summary['paired_fake']]
        summary['improved_pairs']=sum(v>0 for v in dd);summary['tied_pairs']=sum(v==0 for v in dd);summary['declined_pairs']=sum(v<0 for v in dd)
        lines+=['',f"Dice改善{summary['improved_pairs']}/4，持平{summary['tied_pairs']}，下降{summary['declined_pairs']}。"]
        if sum(dd)!=0:
            summary['largest_absolute_contribution_ratio']=max(abs(v) for v in dd)/abs(sum(dd))
            lines.append('最大单对绝对变化/总变化绝对值：'+str(summary['largest_absolute_contribution_ratio'])+'；该比率可能因正负抵消超过1。')
        lines.append(f"第12轮宏平均Dice变化{diff['dice']:+.6f}，real误报变化{diff['real_fpr']:+.6f}。只判断本4对开发视频；检测与空间指标分开判断。")
    else:lines.append('至少一组未完成12轮或身份不匹配，不给主比较胜负结论。已有同轮指标保留在history及逐视频文件。')
    lines+=['','## 各自best（不与第12轮主表拼接）']
    for name,r in runs.items():
        v=r['result'];lines.append(name+'：'+json.dumps({k:v.get(k) for k in ['status','updates','last_epoch','best_epoch','best_dev','wall_s','peak_allocated','peak_reserved','stop_reason']},ensure_ascii=False))
    lines+=['','![同轴开发Dice](dev_dice.png)','完整产物：'+str(OUT),'只读取原train/dev；不重提前端、不加载历史拟合权重、不访问保留测试。']
    save(OUT/'summary.json',summary);(OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (REPO/DOC).write_text('\n'.join(lines).replace('](dev_dice.png)',']('+str(OUT/'dev_dice.png')+')')+'\n')
    page=REPO/'docs/pvst/current_status.md'
    page.write_text(page.read_text()+'\n## ANAKIN来源隔离RGB分片对照\n\n'+status+'；见[报告](rgb_partition_anakin_dev_v1.md)。\n')
    save(OUT/'status.json',{'status':status,'stage':'report_complete','pid':os.getpid()})
    try:
        git('add',DOC,'docs/pvst/current_status.md');git('commit','-m','Report ANAKIN source-isolated RGB partition comparison')
        head=git('rev-parse','HEAD');subprocess.run(['git','push','origin',BRANCH],cwd=REPO,check=True,timeout=120)
        save(OUT/'publication.json',{'status':'PUSHED','head':head})
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        save(OUT/'publication.json',{'status':'COMPLETE' if remote==head else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'head':git('rev-parse','HEAD')})
    return 0 if status=='COMPLETE' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=CONDITIONS);args=parser.parse_args()
    if args.worker:return worker(args.worker)
    if git('branch','--show-current')!=BRANCH:raise ValueError('WRONG_BRANCH')
    if OUT.exists():
        raise ValueError('EXISTING_OUTPUT_NO_AUTOMATIC_RETRY:'+str(OUT))
    if git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice'):
        raise ValueError('UNCOMMITTED_EXECUTION_SOURCE')
    OUT.mkdir();sys.stdout=(OUT/'parent.log').open('a',buffering=1);sys.stderr=sys.stdout
    save(OUT/'status.json',{'status':'RUNNING','stage':'prepare','pid':os.getpid(),'head':git('rev-parse','HEAD')})
    try:
        torch.set_num_threads(4);c,m,_=prepare();preflight(c,m)
        for condition in CONDITIONS:
            save(OUT/'status.json',{'status':'RUNNING','stage':'training','condition':condition,'pid':os.getpid()})
            run_arm(condition)
        return finish()
    except BaseException:
        detail=traceback.format_exc();(OUT/'error.txt').write_text(detail)
        save(OUT/'status.json',{'status':'FAILED','stage':'prepare_or_parent_failed','exit_code':1,'error':detail});raise


if __name__=='__main__':raise SystemExit(main())
