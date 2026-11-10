"""ADR-0039: fixed-update independent-source expansion, existing RGB-only model.

CLI caller; emits one inventory and one bounded prepare/train run. Reuses native
RGB/GT loader, MoGe preparation, build/update/evaluate; no new model or registry.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import av
import numpy as np
import torch
from torch.nn import functional as F
from .data import Video, geometry, geometry_shape, save
from .experiment import REPO, evaluate, forward, git, load_clip, optimizer, state_hash
from .set_fit_check import SOURCE, build, update_once
from .rgb_partition_anakin_dev import source_key, fingerprint, epoch_order
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import alignment, numeric_members, mask_array

BASE=SOURCE.parent/'pvst_rgb_partition_anakin_dev_v1'
FIT=SOURCE.parent/'pvst_rgb_partition_train_fit_eval_v1'
OUT=SOURCE.parent/'pvst_rgb_partition_anakin_scale_v1'
BRANCH='pvst-rgb-partition-anakin-scale'
DOC='docs/pvst/rgb_partition_anakin_scale_v1.md'
PREP_SECONDS=3600
TRAIN_SECONDS=14400
BYTE_LIMIT=30*2**30


def read(p):return json.loads(Path(p).read_text())

def storage():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())

def select_candidates(rows,baseline,seed=20260909):
    """No test-media access. Only previously registered train roles may expand train."""
    blocked_sources={r['source_id'] for r in baseline}
    blocked_urls={source_key(r['original_url']) for r in baseline}
    blocked_hashes={r['video_sha256'] for r in baseline}
    reserved=[r for r in rows if r['split']=='test_reserved']
    blocked_sources|={r['source_id'] for r in reserved}
    blocked_urls|={source_key(r['original_url']) for r in reserved}
    blocked_hashes|={r['video_sha256'] for r in reserved}
    targets=Counter(r['generator'] for r in baseline if r['split']=='train' and r['role']=='fake')
    if sum(targets.values())!=8:raise ValueError('BASELINE_PAIR_COUNT')
    excluded=[];groups={k:[] for k in targets};seen=set()
    for r in sorted([r for r in rows if r['role']=='fake'],key=lambda x:x['pair_id']):
        reason=None
        pair=[x for x in rows if x['pair_id']==r['pair_id']]
        if r['source_id'] in blocked_sources or source_key(r['original_url']) in blocked_urls:reason='original_train_or_dev_or_registered_reserved_source'
        elif r['split']!='train':reason='not_registered_train; preserve historical dev roles'
        elif r['generator'] not in groups:reason='edit_category_not_in_original_8pair'
        elif len(pair)!=2 or {x['role'] for x in pair}!={'real','fake'}:reason='missing_unique_real_fake_pair'
        elif any(x['video_sha256'] in blocked_hashes for x in pair):reason='media_SHA_overlaps_excluded_sources'
        elif len({x['source_id'] for x in pair})!=1 or len({source_key(x['original_url']) for x in pair})!=1:reason='pair_source_conflict'
        elif source_key(r['original_url']) in seen:reason='duplicate_original_source; retain lowest_stable_pair_ID'
        if reason:excluded.append({'pair_id':r['pair_id'],'source_id':r['source_id'],'reason':reason})
        else:groups[r['generator']].append(r);seen.add(source_key(r['original_url']))
    rng=random.Random(seed)
    for kind in sorted(groups):rng.shuffle(groups[kind])
    return groups,dict(targets),excluded


def technical_pair(pair):
    for r in pair:
        if digest(Path(r['video_path']))!=r['video_sha256']:raise ValueError('VIDEO_SHA:'+r['video_id'])
        n=0
        with av.open(r['video_path']) as container:
            for f in container.decode(video=0):
                if n>=r['actual_frames'] or [f.height,f.width]!=r['source_hw'] or abs(float(f.pts*f.time_base)-r['pts'][n]['pts_s'])>1e-8:
                    raise ValueError('FRAME_SIZE_PTS:'+r['video_id'])
                n+=1
        if n!=r['actual_frames'] or n<24:raise ValueError('FRAME_COUNT:'+r['video_id'])
        if any(b['pts_s']<=a['pts_s'] for a,b in zip(r['pts'],r['pts'][1:])):raise ValueError('NONMONOTONIC_PTS')
        if r['role']=='fake':
            if digest(Path(r['mask_zip']))!=r['mask_sha256']:raise ValueError('MASK_SHA')
            with zipfile.ZipFile(r['mask_zip']) as z:
                members=numeric_members(z.namelist())
                if alignment([i for i,_ in members],n)!={'status':'aligned','index_base':r['index_base']}:raise ValueError('MASK_ALIGNMENT')
                if members!=[tuple(x) for x in r['mask_members']]:raise ValueError('MASK_MEMBERS_CHANGED')
                for _,name in members:
                    if list(mask_array(z.read(name))[1].shape)!=r['source_hw']:raise ValueError('MASK_NATIVE_SIZE')


def inventory():
    if OUT.exists():raise ValueError('EXISTING_OUTPUT; inspect before any restart')
    OUT.mkdir();c=read(BASE/'config.json');b=read(BASE/'manifest.json');identity=read(BASE/'identity.json');summary=read(BASE/'summary.json')
    assert identity['head']=='5ffe208bc9bcac16ccb0429ebc8659b6bea36267' and summary['status']=='COMPLETE'
    assert digest(BASE/'manifest.json')==identity['frozen_manifest_sha256']
    assert digest(BASE/'config.json')==identity['config_sha256']
    checkpoint=torch.load(BASE/'RGB_DEPTH_PARTITION/last.pt',map_location='cpu',weights_only=True)
    fit=next(x for x in read(FIT/'summary.json')['results'] if x['condition']=='RGB_DEPTH_PARTITION' and x['checkpoint']=='last')
    assert checkpoint['condition']=='RGB_DEPTH_PARTITION' and checkpoint['epoch']==12 and checkpoint['updates']==576
    assert checkpoint['identity']==identity['identity'] and checkpoint['initial_model_sha256']==fit['initial_model_sha256']
    assert digest(BASE/'RGB_DEPTH_PARTITION/last.pt')==fit['sha256']
    assert checkpoint['initial_model_sha256']==read(BASE/'RGB_DEPTH_PARTITION/initialization.json')['model_sha256']
    del checkpoint
    registry_path=REPO/'docs/v8/evidence/spatial_supervision_expansion_manifest.json';registry=read(registry_path)
    groups,targets,excluded=select_candidates(registry['rows'],b['rows']);selected=[];audit=[]
    # Save order and rules BEFORE media validation; technical failures alone may advance.
    save(OUT/'candidate_order.json',{'seed':20260909,'targets':targets,'order':{k:[r['pair_id'] for r in v] for k,v in groups.items()},'excluded':excluded,
        'rule':'metadata-only categories matching original 4 inpainting/4 splicing; stable pair-ID then seed shuffle by sorted category; only registered train; no scores/area/geometry selection; one source once'})
    for kind in sorted(groups):
        for fake in groups[kind]:
            pair=[copy.deepcopy(r) for r in registry['rows'] if r['pair_id']==fake['pair_id']]
            try:technical_pair(pair)
            except Exception as e:
                audit.append({'pair_id':fake['pair_id'],'status':'excluded_technical','reason':str(e)});continue
            selected.extend(pair);audit.append({'pair_id':fake['pair_id'],'status':'selected','category':kind})
            if sum(r['role']=='fake' and r['generator']==kind for r in selected)==targets[kind]:break
    save(OUT/'inventory.json',{'registry_sha256':digest(registry_path),'audit':audit,'excluded':excluded,'baseline_last':summary['runs']['RGB_DEPTH_PARTITION']['result'],'baseline_common_train_fit':fit,
                              'baseline_weight_sha256':fit['sha256'],'selected_pairs':len(selected)//2})
    if len(selected)!=16:
        blocked_report('ASSET_BLOCKED',f'合法新增来源只有{len(selected)//2}/8；不改变规模。');return False
    # Uniform roots with read-only links for original assets; only new native RGB and geometry consume storage.
    c.update(output=str(OUT),cache_root=str(OUT/'cache'),geometry_root=str(OUT/'geometry'),epochs=6,conditions=['RGB_DEPTH_PARTITION'],protocol='rgb-partition-anakin-scale-v1',patience=None,min_epochs=None)
    original_c=read(BASE/'config.json');(OUT/'cache').mkdir();(OUT/'geometry').mkdir()
    for r in b['rows']:
        (OUT/'cache'/r['video_id']).symlink_to(Path(original_c['cache_root'])/r['video_id'],target_is_directory=True)
        (OUT/'geometry'/r['video_id']).symlink_to(Path(original_c['geometry_root'])/r['video_id'],target_is_directory=True)
    estimates=[];new_rows=[]
    for r in selected:
        # No old geometry dependency: this metadata-only adapter satisfies RegisteredVideo frame-identity checks.
        r['dense_source']=str(OUT/'frame_identity'/r['video_id']);r['original_registered_split']=r['split'];r['split']='train'
        for k in ['cache_dir','feature_path','area_fraction','area_group']:r.pop(k,None)
        h,w=r['source_hw'];gh,gw=geometry_shape([h,w])[0];n=r['actual_frames']
        estimates.append(dict(video_id=r['video_id'],frames=n,native_hw=[h,w],geometry_hw=[gh,gw],rgb_bytes=n*h*w*3,geometry_bytes=n*gh*gw*13,reused=False))
        new_rows.append(r)
    # Original high-res protocol has numeric caches only for the registered original24 videos; inspect exact IDs before claiming new work.
    for e in estimates:
        old_geom=SOURCE/'geometry'/e['video_id'];old_rgb=Path(original_c['cache_root'])/e['video_id']
        if old_geom.exists() or old_rgb.exists():
            # Any unexpected existing cache must be audited explicitly, not silently regenerated or treated compatible.
            raise ValueError('UNEXPECTED_EXISTING_NEW_ID_CACHE_REQUIRES_AUDIT:'+e['video_id'])
    clips=list(b['clips'])
    for r in new_rows:
        starts=sorted(set(int(x) for x in np.linspace(0,r['actual_frames']-24,3)))
        if len(starts)!=3:raise ValueError('INSUFFICIENT_SPREAD_CLIPS')
        clips.extend(dict(video_id=r['video_id'],start=s,end=s+24) for s in starts)
    rows=b['rows']+new_rows
    if len(clips)!=96 or len({r['source_id'] for r in rows if r['split']=='train'})!=16:raise ValueError('EXPANSION_COUNTS')
    frozen={'rows':rows,'clips':clips,'original_clips':b['clips'],'new_video_ids':[r['video_id'] for r in new_rows]}
    save(OUT/'manifest.json',frozen);save(OUT/'config.json',c);save(OUT/'orders.json',{str(e):epoch_order(clips,c['seed'],e) for e in range(1,7)})
    history_cost=read(SOURCE/'execution_history/00_cpu_context_device_copy/geometry_cost.json')
    projected=sum(x['rgb_bytes']+x['geometry_bytes'] for x in estimates)
    frames=sum(x['frames'] for x in estimates);prior_rate=history_cost['seconds']/history_cost['new_frontend_frames']
    estimated_seconds=frames*prior_rate*3+120 # Explicit 3x allowance plus native decode/write/verification allowance; estimate, not guarantee.
    budget={'status':'PASS' if projected<BYTE_LIMIT and estimated_seconds<PREP_SECONDS else 'RESOURCE_BLOCKED','reused_original_videos':24,'reused_original_frames':sum(r['actual_frames'] for r in b['rows']),
            'new_videos':16,'new_frames':frames,'new_RGB_videos':16,'new_geometry_videos':16,'new_cache_bytes_estimate':projected,'new_cache_GiB_estimate':projected/2**30,
            'preparation_seconds_estimate':estimated_seconds,'estimate_basis':history_cost,'estimate_rule':'historical MoGe seconds/frame x new frames x3 plus120s; not guaranteed; parent hard cap3600s',
            'per_video':estimates,'limits':{'seconds':3600,'bytes':BYTE_LIMIT}}
    save(OUT/'budget.json',budget)
    save(OUT/'identity.json',{'baseline_identity':identity['identity'],'initial_model_sha256':fit['initial_model_sha256'],'baseline_checkpoint_sha256':fit['sha256'],
        'manifest_sha256':digest(OUT/'manifest.json'),'config_sha256':digest(OUT/'config.json'),'orders_sha256':digest(OUT/'orders.json'),'selection_sha256':digest(OUT/'candidate_order.json'),
        'inventory_head':git('rev-parse','HEAD'),'independent_train_frames':len({(x['video_id'],i) for x in clips for i in range(x['start'],x['end'])}),
        'frame_presentations_per_epoch':96*24,'total_frame_presentations':576*24})
    (OUT/'report.md').write_text('# ANAKIN来源扩量：已冻结，尚未训练\n\n'+json.dumps(budget,ensure_ascii=False,indent=2)+'\n')
    if budget['status']!='PASS':blocked_report('RESOURCE_BLOCKED','缓存估算超过本轮上限');return False
    save(OUT/'status.json',{'status':'FROZEN','stage':'inventory_complete'});return True


def blocked_report(status,reason):
    save(OUT/'summary.json',{'status':status,'reason':reason,'trained_updates':0})
    save(OUT/'status.json',{'status':status,'stage':'report_complete','reason':reason})
    text='# ANAKIN来源扩量未放行\n\n状态：'+status+'\n\n'+reason+'\n原8对基线复用；扩量未训练，无可比较的新指标。\n'
    (OUT/'report.md').write_text(text);(REPO/DOC).write_text(text)


def prepare_cache():
    c=read(OUT/'config.json');m=read(OUT/'manifest.json');started=time.monotonic();new=[r for r in m['rows'] if r['video_id'] in m['new_video_ids']]
    def progress(stage,**extra):
        if time.monotonic()-started>PREP_SECONDS or storage()>BYTE_LIMIT:raise RuntimeError('PREPARATION_BUDGET_EXCEEDED')
        save(OUT/'progress.json',{'status':'RUNNING','stage':stage,'seconds':time.monotonic()-started,**extra});print(stage,extra,flush=True)
    for r in new:
        progress('native_rgb',video=r['video_id']);folder=Path(c['cache_root'])/r['video_id'];folder.mkdir()
        meta_folder=Path(r['dense_source']);meta_folder.mkdir(parents=True)
        save(meta_folder/'complete.json',{'video_sha256':r['video_sha256'],'frames':[{'frame_index':i,'pts_s':p['pts_s']} for i,p in enumerate(r['pts'])],'purpose':'frame identity only, no old numeric geometry'})
        rgb=np.lib.format.open_memmap(folder/'rgb.npy',mode='w+',dtype=np.uint8,shape=(r['actual_frames'],*r['source_hw'],3));count=0
        with av.open(r['video_path']) as container:
            for f in container.decode(video=0):
                if count>=len(rgb) or abs(float(f.pts*f.time_base)-r['pts'][count]['pts_s'])>1e-8:raise ValueError('RGB_PTS')
                a=f.to_ndarray(format='rgb24')
                if list(a.shape[:2])!=r['source_hw']:raise ValueError('RGB_NATIVE_SIZE')
                rgb[count]=a;count+=1
        if count!=len(rgb):raise ValueError('RGB_FRAME_COUNT')
        rgb.flush();del rgb
        save(folder/'complete.json',{'video_sha256':r['video_sha256'],'frames':r['actual_frames'],'prepared_frames':count,'native_hw':r['source_hw'],'purpose':'native RGB only; no features'})
    geometry(c,OUT,new,progress)
    for r in new:
        progress('verify_new_cache',video=r['video_id']);v=Video(r,c)
        try:
            meta=read(Path(c['geometry_root'])/r['video_id']/'complete.json')
            if max(meta['output_hw'])<=256 or meta['moge_revision']!=c['moge_revision'] or meta['moge_weights_sha256']!=digest(Path(c['moge_weights'])):raise ValueError('HIGH_RES_PROVIDER_IDENTITY')
            for i in range(r['actual_frames']):
                if not np.isfinite(v.xyz[i][v.valid[i]]).all():raise ValueError('NONFINITE_VALID_GEOMETRY')
        finally:v.close()
    progress('prepared')
    save(OUT/'preparation_result.json',{'status':'PASS','wall_s':time.monotonic()-started,'new_cache_bytes':storage(),'new_frontend_frames':sum(r['actual_frames'] for r in new)})
    return 0


def enriched_eval(model,rows,c,progress,clips=None,ap=False):
    result=evaluate(model,rows,c,progress,clip_list=clips,pixel_ap=ap)
    scores=torch.tensor([z for r in result['records'] for z in r['scores']],dtype=torch.float32)
    labels=torch.tensor([y for r in result['records'] for y in r['labels']],dtype=torch.float32)
    losses=F.binary_cross_entropy_with_logits(scores,labels,reduction='none');m=result['metrics']
    m['frame_bce_fp32']=float(losses.mean());m['positive_frame_bce']=float(losses[labels.bool()].mean()) if labels.any() else None
    m['negative_frame_bce']=float(losses[~labels.bool()].mean()) if (~labels.bool()).any() else None
    return result


def train():
    started=time.monotonic();folder=OUT/'RGB_DEPTH_PARTITION';folder.mkdir();c=read(OUT/'config.json');m=read(OUT/'manifest.json');identity=read(OUT/'identity.json')
    result={'status':'FAILED','updates':0,'last_epoch':0,'exit_code':1}
    def progress(stage,**extra):
        if time.monotonic()-started>=TRAIN_SECONDS:raise TimeoutError('FOUR_HOUR_LIMIT')
        save(OUT/'progress.json',{'status':'RUNNING','stage':stage,'seconds':time.monotonic()-started,**extra});print(stage,extra,flush=True)
    try:
        torch.set_num_threads(4);save(folder/'launch.json',{'head':git('rev-parse','HEAD'),'gpu':torch.cuda.get_device_name(),'free_total':torch.cuda.mem_get_info(),'pid':os.getpid()})
        if digest(OUT/'manifest.json')!=identity['manifest_sha256'] or digest(OUT/'config.json')!=identity['config_sha256'] or digest(OUT/'orders.json')!=identity['orders_sha256']:raise ValueError('FROZEN_IDENTITY_CHANGED')
        model=build(c,folder,rgb_only=True);assert read(folder/'initialization.json')['model_sha256']==identity['initial_model_sha256']
        bn=state_hash(dict(model.local.encoder.named_buffers()));trainrows={r['video_id']:r for r in m['rows'] if r['split']=='train'};dev=[r for r in m['rows'] if r['split']=='dev']
        # True new-source24-frame forward, discarded; reset RNG via fresh build before formal training.
        row=next(r for r in m['rows'] if r['video_id']==m['new_video_ids'][0]);clip=next(x for x in m['clips'] if x['video_id']==row['video_id'])
        fs=load_clip(row,clip,c);model.eval()
        with torch.no_grad():
            output=forward(model,fs)
            if len(output)!=24 or any(not torch.isfinite(p[k]).all() for p in output for k in ['pixel','frame']):raise ValueError('PREFLIGHT_NONFINITE')
        save(OUT/'preflight.json',{'status':'PASS','clip':clip,'frames':24,'rgb_only':True,'initial_model_sha256':identity['initial_model_sha256']})
        del fs,output,model;torch.cuda.empty_cache();model=build(c,folder,rgb_only=True);opt=optimizer(model,c);torch.cuda.reset_peak_memory_stats()
        initial=enriched_eval(model,dev,c,progress);save(folder/'epoch00_dev.json',initial)
        history=[];best=-1.;best_epoch=0
        for epoch in range(1,7):
            t=time.monotonic();parts=[];order=read(OUT/'orders.json')[str(epoch)]
            if len(order)!=96:raise ValueError('EPOCH_CLIP_COUNT')
            for i,clip in enumerate(order):
                progress('train',epoch=epoch,clip=i+1,updates=result['updates']);fs=load_clip(trainrows[clip['video_id']],clip,c);record=update_once(model,opt,fs);del fs
                parts.append(record['terms']);result['updates']+=1;record.pop('gradients')
                with (folder/'updates.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'clip':clip,'update':result['updates'],**record})+'\n')
            value=enriched_eval(model,dev,c,progress);save(folder/f'epoch{epoch:02d}_dev.json',value)
            if state_hash(dict(model.local.encoder.named_buffers()))!=bn:raise ValueError('BN_BUFFERS_CHANGED')
            improved=value['metrics']['dice']>best
            if improved:best,best_epoch=value['metrics']['dice'],epoch
            terms=np.mean(parts,0);history.append({'epoch':epoch,'updates':result['updates'],'terms':terms.tolist(),'loss':float(terms.sum()),'dev':value['metrics'],'seconds':time.monotonic()-t,'order_sha256':fingerprint(order),'best_epoch':best_epoch})
            save(folder/'history.json',history)
            state={'state_dict':model.state_dict(),'epoch':epoch,'updates':result['updates'],'condition':'RGB_DEPTH_PARTITION','identity':identity,'initial_model_sha256':identity['initial_model_sha256']}
            tmp=folder/'last.pt.tmp';torch.save(state,tmp);tmp.replace(folder/'last.pt')
            if improved:shutil.copyfile(folder/'last.pt',folder/'best.pt')
            result['last_epoch']=epoch;save(folder/'result.json',{**result,'status':'RUNNING'});print('EPOCH_COMPLETE',history[-1],flush=True)
        progress('last_dev_AP');last=enriched_eval(model,dev,c,progress,ap=True);save(folder/'last_dev.json',last)
        progress('common_train_eval');oldrows=[r for r in m['rows'] if r['video_id'] in {x['video_id'] for x in m['original_clips']}]
        common=enriched_eval(model,oldrows,c,progress,clips=m['original_clips']);save(folder/'common_train_eval.json',common)
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['state_dict'],strict=True);del state
        progress('best_dev_AP',epoch=best_epoch);bv=last if best_epoch==6 else enriched_eval(model,dev,c,progress,ap=True);save(folder/'best_dev.json',bv)
        result.update(status='COMPLETE',exit_code=0,best_epoch=best_epoch,last_dev=last['metrics'],best_dev=bv['metrics'],common_train=common['metrics'],stop_reason='6_EPOCHS_576_UPDATES',BN_unchanged=True)
    except BaseException as e:
        result.update(status='OOM' if isinstance(e,torch.cuda.OutOfMemoryError) else 'TIMEOUT' if isinstance(e,TimeoutError) else 'FAILED',error=traceback.format_exc(),stop_reason=str(e));print(result['error'],flush=True)
    finally:
        result.update(wall_s=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save(folder/'result.json',result)
    return result['exit_code']


def stage(name,limit):
    start=time.monotonic();folder=OUT/name;folder.mkdir(exist_ok=True)
    with (folder/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--'+name],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(folder/'process.json',{'pid':child.pid,'started_unix':time.time()});timeout=False
        try:code=child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    save(folder/'exit.json',{'exit_code':code,'timed_out':timeout,'wall_s':time.monotonic()-start})
    return code,timeout


def finish():
    inv=read(OUT/'inventory.json');base=inv['baseline_last'];fit=inv['baseline_common_train_fit'];folder=OUT/'RGB_DEPTH_PARTITION'
    result=read(folder/'result.json') if (folder/'result.json').exists() else {'status':'FAILED','updates':0}
    exit=read(OUT/'train/exit.json')
    if exit['timed_out']:result.update(status='TIMEOUT',stop_reason='FOUR_HOUR_PARENT_LIMIT')
    complete=result.get('status')=='COMPLETE' and result.get('updates')==576 and exit['exit_code']==0
    summary={'status':'COMPLETE' if complete else result.get('status','FAILED'),'new_run':result,'train_exit':exit,
             'budget':read(OUT/'budget.json'),'preparation':read(OUT/'preparation_result.json'),'baseline_last':base,
             'execution':read(OUT/'execution.json'),'paired':[]}
    lines=['# ANAKIN固定576步训练来源扩量对照','状态：'+summary['status'],
           '原8对/48段训练12轮，扩量16对/96段训练6轮；均576次AdamW、13824帧次。原开发4对8视频924帧不变。两组匹配更新与输入帧次，实际墙钟不同。',
           '扩量固定新增4 inpainting/4 splicing，均来自历史登记train，排除原train/dev及保留test来源；未用面积/模型分数/几何异常挑选。新视频含1080p样本，未按资源负载替换。',
           '准备预算及实测：'+json.dumps(summary['budget'],ensure_ascii=False)+'\n'+json.dumps(summary['preparation'],ensure_ascii=False),
           '原缓存通过只读符号链接引用；只补新增16条native RGB/高分辨率MoGe。frame_identity只是帧身份适配元数据，不是旧256几何。']
    if complete:
        new=result['last_dev'];keys=['dice','iou','pixel_ap','GT_recall','real_fpr'];diff={k:new[k]-base['last_dev'][k] for k in keys};summary['difference']=diff
        lines+=['','|576步开发|Dice|IoU|pixel AP|GT召回|real FPR|','|---|---:|---:|---:|---:|---:|']
        for label,v in [('原8对',base['last_dev']),('扩量16对',new),('扩量−基线',diff)]:lines.append('|'+label+'|'+'|'.join(f'{v[k]:.6f}' for k in keys)+'|')
        oldrecords=read(BASE/'RGB_DEPTH_PARTITION/last_dev.json')['records'];newrecords=read(folder/'last_dev.json')['records']
        lines+=['','|开发pair|旧Dice|新Dice|ΔDice|旧IoU|新IoU|ΔIoU|旧召回|新召回|旧real FPR|新real FPR|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for old in oldrecords:
            if old['role']!='fake':continue
            newr=next(r for r in newrecords if r['video_id']==old['video_id']);ar=next(r for r in oldrecords if r['pair_id']==old['pair_id'] and r['role']=='real');br=next(r for r in newrecords if r['pair_id']==old['pair_id'] and r['role']=='real')
            p={'pair':old['pair_id'],'old_dice':old['dice'],'new_dice':newr['dice'],'delta_dice':newr['dice']-old['dice'],
               'old_iou':old['iou'],'new_iou':newr['iou'],'delta_iou':newr['iou']-old['iou'],'old_recall':old['gt_recall'],'new_recall':newr['gt_recall'],'old_real_FPR':ar['empty_fpr'],'new_real_FPR':br['empty_fpr']}
            summary['paired'].append(p);lines.append('|'+p['pair']+'|'+'|'.join(f'{v:.6f}' for k,v in p.items() if k!='pair')+'|')
        dd=[p['delta_dice'] for p in summary['paired']];summary['improved_pairs']=sum(d>0 for d in dd);summary['declined_pairs']=sum(d<0 for d in dd)
        summary['delta_without_largest_contributor']=(sum(dd)-max(dd,key=abs))/3
        oldtrain=fit['train'];common=result['common_train'];oldgap=oldtrain['dice']-base['last_dev']['dice'];newgap=common['dice']-new['dice']
        lines+=['',f"Dice改善{summary['improved_pairs']}/4，下降{summary['declined_pairs']}/4；剔除绝对变化最大一对的其他3对平均差{summary['delta_without_largest_contributor']:+.6f}（仅敏感性描述，主表保留4对）。",
                '## 原48个共同训练片段（同集合比较）',f"原基线训练Dice {oldtrain['dice']:.6f}、开发{base['last_dev']['dice']:.6f}、差距{oldgap:.6f}；扩量末轮共同片段Dice {common['dice']:.6f}、开发{new['dice']:.6f}、差距{newgap:.6f}。",
                '若差距缩小仅来自训练分数下降，不视为开发改善。实际是否受益看开发Dice/IoU/AP及逐pair，不凭差距一项判断。']
        summary['common_train_comparison']={'baseline':oldtrain,'expanded':common,'old_train_dev_gap':oldgap,'new_train_dev_gap':newgap}
        labels=[y for r in oldrecords for y in r['labels']];scores=[z for r in oldrecords for z in r['scores']];z=torch.tensor(scores,dtype=torch.float32);y=torch.tensor(labels,dtype=torch.float32);basepos=float(F.binary_cross_entropy_with_logits(z[y.bool()],y[y.bool()]))
        lines+=['','## 检测单列',f"原开发帧AUROC/AP {base['last_dev']['frame']['auroc']:.6f}/{base['last_dev']['frame']['ap']:.6f}，正GT帧FP32 BCE {basepos:.6f}；扩量{new['frame']['auroc']:.6f}/{new['frame']['ap']:.6f}，正GT帧BCE {new['positive_frame_bce']:.6f}。",
                '## best补充（不替代576步主表）',json.dumps({k:result[k] for k in ['best_epoch','best_dev']},ensure_ascii=False)]
        summary['conclusion']='固定576步下开发定位'+('提高' if diff['dice']>0 else '未提高')+f"；Dice改善{summary['improved_pairs']}/4。只反映更多来源替代片段重复的固定预算策略；不证明数据已充分，也不归因于单独样本量效应。"
        lines+=['',summary['conclusion']]
    else:lines+=['扩量组未完成，不能给576步主比较胜负；不自动重试。原8对基线保持原样。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));bh=read(BASE/'RGB_DEPTH_PARTITION/history.json');ax.plot([r['updates'] for r in bh if r['epoch']%2==0],[r['dev']['dice'] for r in bh if r['epoch']%2==0],marker='o',label='8 pairs baseline')
    if (folder/'history.json').exists():
        nh=read(folder/'history.json');ax.plot([r['updates'] for r in nh],[r['dev']['dice'] for r in nh],marker='o',label='16 pairs expanded')
    ax.set(xlabel='AdamW updates',ylabel='Development macro Dice');ax.legend();fig.tight_layout();fig.savefig(OUT/'dev_dice_by_updates.png');plt.close(fig)
    lines+=['','![同更新数开发Dice](dev_dice_by_updates.png)','资源：'+json.dumps({k:result.get(k) for k in ['wall_s','peak_allocated','peak_reserved','stop_reason','updates','last_epoch']},ensure_ascii=False),'执行代码：'+summary['execution']['head'],'产物：'+str(OUT),'计算状态与发布状态独立；不追加训练、模型、seed或规模。']
    save(OUT/'summary.json',summary);text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text)
    # Repository report is small; full per-video/cache detail stays in derived.
    (REPO/DOC).write_text(text.replace('](dev_dice_by_updates.png)',']('+str(OUT/'dev_dice_by_updates.png')+')'))
    save(OUT/'status.json',{'status':summary['status'],'stage':'report_complete','pid':os.getpid()})


def publish():
    try:
        git('add',DOC);git('commit','-m','Report fixed-budget ANAKIN independent-source expansion')
        head=git('rev-parse','HEAD');subprocess.run(['git','push','origin',BRANCH],cwd=REPO,check=True,timeout=120)
        save(OUT/'publication.json',{'status':'PUSHED','head':head})
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        save(OUT/'publication.json',{'status':'COMPLETE' if head==remote else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'head':git('rev-parse','HEAD')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');a=p.parse_args()
    if a.prepare:return prepare_cache()
    if a.train:return train()
    if git('branch','--show-current')!=BRANCH:raise ValueError('WRONG_BRANCH')
    if a.inventory:
        try:return 0 if inventory() else 2
        except Exception:
            if OUT.exists():blocked_report('ASSET_BLOCKED',traceback.format_exc())
            raise
    if read(OUT/'status.json')['status']!='FROZEN':raise ValueError('NOT_FRESH_FROZEN_RUN')
    if git('status','--porcelain','--','research_tools/rgbd_slice','research_tools/depth_slice'):raise ValueError('UNCOMMITTED_EXECUTION_SOURCE')
    save(OUT/'execution.json',{'head':git('rev-parse','HEAD'),'code_sha256':digest(Path(__file__)),'started_unix':time.time()})
    sys.stdout=(OUT/'parent.log').open('a',buffering=1);sys.stderr=sys.stdout
    code,timeout=stage('prepare',PREP_SECONDS)
    if code or timeout:
        blocked_report('PREPARATION_TIMEOUT' if timeout else 'PREPARATION_FAILED','新增缓存准备未完成；原始退出记录见prepare/exit.json。');publish();return 2
    code,timeout=stage('train',TRAIN_SECONDS);finish();publish()
    return 0 if read(OUT/'summary.json')['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
