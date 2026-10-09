"""All-frame two-video pilot: explicit frozen assets, atomic blocks, no training."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
import signal
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import read_json,sha256_file,write_json_atomic,write_npz_atomic
from research_tools.v8.full_coverage_observation_field.frontend import MogeV2,CoTrackerShortPrefix
from research_tools.v8.full_coverage_observation_field.pipeline import FrozenResNet18Spatial
from research_tools.v8.local_geometry_pilot.data import Dataset,digest
from research_tools.v8.local_geometry_pilot.model import LocalGeometryModel
from .inputs import Frontend,decode,video_manifest,patch
from .model import StreamingModel,state_to
from .validation import validate_window,validate_continuous

STOP=False


def save_pt(path,value):
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def tensors(a,device='cuda'):
    return {k:torch.from_numpy(v).unsqueeze(0).to(device) for k,v in a.items()}


def load_input(out,row,fi,identity):
    p=out/'inputs'/row['video_id']/f'{fi:06d}.npz';meta=read_json(p.with_suffix('.json'))
    if meta['identity']!=identity or meta['frame_index']!=fi or meta['video_sha256']!=row['video_sha256']:
        raise ValueError('INPUT_RESUME_IDENTITY')
    if sha256_file(p)!=meta['input_sha256']: raise ValueError('INPUT_CONTENT_HASH')
    raw=out/'frames'/row['video_id']/f'{fi:06d}.npz';rm=read_json(raw.with_suffix('.json'))
    if sha256_file(raw)!=rm['npz_sha256']: raise ValueError('RAW_CONTENT_HASH')
    with np.load(p,allow_pickle=False) as z: a={k:z[k].copy() for k in z.files if k not in ('tracks','visibility')}
    with np.load(raw,allow_pickle=False) as z: xyz=z['points_raw'];valid=z['validity'];rgb=z['rgb'].copy()
    m=meta['mapping'];content=np.zeros((256,256),bool);left,top=m['pad_xy'];h,w=m['content_hw'];content[top:top+h,left:left+w]=True
    cp=patch(content); vp=patch(valid)&cp&np.isfinite(patch(xyz)).all(-1)
    a.update(xyz=np.where(vp[...,None],patch(xyz),0.).astype(np.float32),valid=vp,content=cp)
    return a,meta,(fi,rgb[:,:,::-1].copy(),meta['pts_s'])


def checkpoint_block(directory,identity,state,records,block,cost,context):
    start=block[0]['frame_index'];end=block[-1]['frame_index']
    name=f'{start:06d}-{end:06d}'
    cells=np.stack([x.pop('cell_logits') for x in block])
    npz=directory/f'{name}.npz';js=directory/f'{name}.json'
    write_npz_atomic(npz,cell_logits=cells,frame_indices=np.array([x['frame_index'] for x in block],np.int64),
                     pts_s=np.array([x['pts_s'] for x in block],np.float64),frame_logits=np.array([x['frame_logit'] for x in block],np.float64))
    write_json_atomic(js,{'identity':identity,'records':block,'npz_sha256':sha256_file(npz)})
    records=records+[{'start':start,'end':end,'npz':str(npz),'json':str(js),'sha256':sha256_file(npz)}]
    save_pt(directory/'state.pt',{'identity':identity,'video_id':block[-1]['video_id'],'video_sha256':block[-1]['video_sha256'],
            'state':state_to(state,'cpu'),'next_frame_index':end+1,
            'previous_pts_s':block[-1]['pts_s'],'context_frame_indices':[x[0] for x in context],
            'records':records,'cost':cost,'true_video_first_frame':False})
    return records


def process_video(row,out,frontend,model,identity,block_size=16):
    directory=out/'responses'/row['video_id'];directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'state.pt';state=None;context=[];next_frame=0;records=[]
    cost={'decode_s':0.,'model_s':0.,'response_write_s':0.}; previous_active=None
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if c['identity']!=identity or c['video_id']!=row['video_id'] or c['video_sha256']!=row['video_sha256']: raise ValueError('STATE_IDENTITY_MISMATCH')
        state=state_to(c['state'],'cuda');next_frame=c['next_frame_index'];records=c['records'];cost=c['cost']
        if next_frame!=state['frame_count']: raise ValueError('STATE_PROGRESS_MISMATCH')
        expected=0
        for r in records:
            if r['start']!=expected or sha256_file(r['npz'])!=r['sha256']: raise ValueError('BLOCK_GAP_OR_HASH')
            m=read_json(r['json'])
            if m['identity']!=identity or len(m['records'])!=r['end']-r['start']+1: raise ValueError('BLOCK_RECORD_IDENTITY')
            expected=r['end']+1
        if expected!=next_frame: raise ValueError('BLOCK_STATE_GAP')
        for fi in c['context_frame_indices']:
            _,_,ctx=load_input(out,row,fi,identity);context.append(ctx)
        previous_active=state['previous_active'][0].cpu().numpy()
        print(f"RESUME {row['source_id']} {row['role']} next_frame={next_frame}",flush=True)
    stream=StreamingModel(model);block=[];iterator=iter(decode(row['video_path']));decoded=0
    start=time.monotonic(); validated=(out/'validation'/'continuous.json').exists()
    while True:
        ts=time.monotonic()
        try: fi,bgr,info=next(iterator)
        except StopIteration: break
        cost['decode_s']+=time.monotonic()-ts;decoded+=1
        if fi<next_frame: continue
        ip=out/'inputs'/row['video_id']/f'{fi:06d}.npz'
        if ip.exists():
            a,meta,ctx=load_input(out,row,fi,identity)
            if meta['pts_ticks']!=info['pts_ticks'] or abs(meta['pts_s']-info['pts_s'])>1e-9: raise ValueError('INPUT_PTS_MISMATCH')
        else:
            a,meta,ctx,tr=frontend.build(row,bgr,info,context,previous_active)
            ip.parent.mkdir(parents=True,exist_ok=True);ts=time.monotonic()
            write_npz_atomic(ip,**{k:v for k,v in a.items() if k not in ('xyz','valid','content')},tracks=tr['tracks'],visibility=tr['visibility'])
            meta.update(identity=identity,video_sha256=row['video_sha256'],input_sha256=sha256_file(ip))
            write_json_atomic(ip.with_suffix('.json'),meta);frontend.cost['write_s']+=time.monotonic()-ts
        f=tensors(a);torch.cuda.synchronize();ts=time.monotonic();r=stream.forward_step(f,state);torch.cuda.synchronize();cost['model_s']+=time.monotonic()-ts
        state=r['state']; context=(context+[ctx])[-2:];previous_active=a['area']>0
        block.append({'video_id':row['video_id'],'video_sha256':row['video_sha256'],'source_id':row['source_id'],'role':row['role'],**info,
                      'frame_logit':float(r['frame_logit']),'content_area':float(a['area'].sum()),
                      'component_count':int(len(np.unique(a['component_ids'][a['area']>0]))),'status':'RESPONSE_COMPLETE',
                      'cell_logits':r['cell_logits'][0].cpu().numpy()})
        if len(block)==block_size:
            ts=time.monotonic();cost['frontend']=frontend.cost.copy();records=checkpoint_block(directory,identity,state,records,block,cost,context)
            cost['response_write_s']+=time.monotonic()-ts;block=[]
            write_json_atomic(out/'progress.json',{'status':'RUNNING','video_id':row['video_id'],'role':row['role'],'stage':'all_frames','completed_frames':fi+1,'metadata_total':row['metadata_frame_count'],'updated_unix':time.time()})
            print(f"FRAME {row['source_id']} {row['role']} {fi+1}/{row['metadata_frame_count']} elapsed_s={time.monotonic()-start:.1f}",flush=True)
            if STOP: raise InterruptedError('SAFE_TERM_AFTER_BLOCK')
        if not validated and fi==63:
            frames=[tensors(load_input(out,row,j,identity)[0]) for j in range(64)]
            check=validate_continuous(model,frames)
            check.update(video_id=row['video_id'],native_frame_indices=list(range(64)),input='same saved native consecutive frame inputs')
            write_json_atomic(out/'validation'/'continuous.json',check);del frames;validated=True
            print('CONTINUOUS_CHUNK_RESTORE_CAUSALITY_PASS',flush=True)
    if block:
        ts=time.monotonic();cost['frontend']=frontend.cost.copy();records=checkpoint_block(directory,identity,state,records,block,cost,context);cost['response_write_s']+=time.monotonic()-ts
    if decoded==0 or state is None or state['frame_count']!=decoded: raise ValueError('EOF_FRAME_RESPONSE_COUNT')
    write_json_atomic(directory/'summary.json',{'status':'COMPLETE','video':row,'decoded_frame_count':decoded,
                 'response_frame_count':state['frame_count'],'video_logit':float(state['log_numerator']-state['total_area'].log()),
                 'metadata_frame_count_difference':decoded-row['metadata_frame_count'],'cost':cost,'blocks':records,
                 'this_invocation_s':time.monotonic()-start,'timestamp_source':'NATIVE_PTS', 'identity':identity})
    return read_json(directory/'summary.json')


def report(out,manifest,selected,summaries,initialization_s,peak,identity):
    rows=[];all_records=[];components=[];sizes=[];missing=0;pixel_count=0;no_history=0;active_count=0
    for s in summaries:
        video=s['video'];vid=video['video_id']
        rr=[]
        for b in s['blocks']: rr.extend(read_json(b['json'])['records'])
        assert [r['frame_index'] for r in rr]==list(range(s['decoded_frame_count']))
        for r in rr:
            a,m,_=load_input(out,video,r['frame_index'],identity)
            active=a['area']>0; comps,cnt=np.unique(a['component_ids'][active],return_counts=True)
            assert cnt.max()<=4 and a['content'].sum()==a['area'].sum() and np.isfinite(r['frame_logit'])
            components.append(len(comps));sizes.extend(cnt.tolist());missing+=int((a['content']&~a['valid']).sum());pixel_count+=int(a['content'].sum())
            no_history+=int((active&(a['association_weight'].sum(-1)==0)).sum());active_count+=int(active.sum())
        all_records+=rr
        rows.append({'source':video['source_id'],'role':video['role'],'metadata_frames':video['metadata_frame_count'],
                     'decoded':s['decoded_frame_count'],'responses':s['response_frame_count'],'video_logit':s['video_logit'],**s['cost']['frontend']})
    with (out/'frame_responses.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
    disk=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    total_frames=sum(r['metadata_frame_count'] for r in manifest);actual=sum(s['decoded_frame_count'] for s in summaries)
    front=sum(sum(s['cost']['frontend'][k] for k in ('geometry_s','rgb_s','tracking_s','features_s')) for s in summaries)
    model_seconds=sum(s['cost']['model_s'] for s in summaries)
    storage={'geometry':0,'rgb_and_frozen_rgb_feature':0,'cell_features_and_tracker':0,'cell_responses':0}
    for folder,category in (('frames','geometry'),('inputs','cell_features_and_tracker'),('responses','cell_responses')):
        for path in (out/folder).rglob('*.npz'):
            with zipfile.ZipFile(path) as z:
                for entry in z.infolist():
                    key=entry.filename.removesuffix('.npy')
                    target='rgb_and_frozen_rgb_feature' if folder=='frames' and key in ('rgb','rgb_map') else category
                    storage[target]+=entry.compress_size
    estimate={'estimated_total_metadata_frames':total_frames,'measured_pilot_frames':actual,
              'estimated_frontend_seconds':front/actual*total_frames,'estimated_model_seconds':model_seconds/actual*total_frames,
              'estimated_disk_bytes':disk/actual*total_frames,'measured_disk_bytes':disk,'disk_available_bytes':os.statvfs(out).f_bavail*os.statvfs(out).f_frsize,
              'measured_compressed_payload_bytes':storage,'estimated_compressed_payload_bytes':{k:v/actual*total_frames for k,v in storage.items()},
              'limitations':'metadata lengths, selected pair geometry complexity/cache-hit fraction; no full dataset execution'}
    coverage={'content_assignment_and_aggregate_coverage':1.,'geometry_missing_content_pixels':missing,'content_pixels':pixel_count,
              'no_predecessor_cell_times':no_history,'active_cell_times':active_count,
              'component_count_quantiles':np.quantile(components,[0,.25,.5,.75,1]).tolist(),'component_size_counts':{str(i):sizes.count(i) for i in range(1,5)}}
    write_json_atomic(out/'coverage_cost.json',{'videos':rows,'coverage':coverage,'estimate':estimate,'initialization_s':initialization_s,'gpu_memory':peak})
    lines=['# V8 全视频连续帧工程 pilot','', '状态：COMPLETE；范围仅 two_video_engineering_pilot，无训练、无全量处理。',
           f"272预期的实际清单={len(manifest)}视频；确定性选择train source {selected[0]['source_id']}，按source/pair身份排序，无分数选例。",
           '', '| role | metadata frames | decoded | responses | video logit (engineering only) |','|---|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['role']} | {r['metadata_frames']} | {r['decoded']} | {r['responses']} | {r['video_logit']:.6f} |")
    lines += ['', '## 接口与验证','', 'forward_step保持冻结MEAN_REPLICATED seed20260909全部权重。只在真正视频首帧初始化，块边界继续hidden/previous_active及前两帧上下文。',
              'frame logit为格响应的面积加权LSE；video logit用log域累计所有帧及内容面积。零面积严格排除。',
              f"旧16帧窗口兼容：{read_json(out/'validation'/'window.json')}。",
              f"连续64个真实原始帧：{read_json(out/'validation'/'continuous.json')}。",
              '块长7/16/31、帧31保存恢复、未来扰动前缀测试共用相同缓存输入；不混入前端波动。',
              '原始PTS/tick/time_base与视频相对秒数均保存。重复工具端点只在tracker调用轴中标记，没有新增输出帧或PTS。',
              '', '## 支撑与成本','',f'覆盖：{coverage}',f'初始化秒：{initialization_s:.3f}；GPU峰值：{peak}。',
              f'逐视频阶段耗时和缓存命中：{rows}',f'全量估计（非实测）：{estimate}',
              '', '## 解释边界','', '冻结窗口模型直接接收原始连续帧，时间间隔、运动输入、历史长度、聚合范围均发生分布变化。视频响应仅为工程读出，不计算或宣称AUROC/定位性能。',
              '当前仍是cell hidden与近期多前驱关联，不是永久component轨迹或全帧component图；几何缺失与无前驱真实格均保留。',
              '下一次全视频训练仍需另行冻结弱监督目标、长序列梯度范围、采样/历史分布及视频级评价；本轮不实现或启动。',
              '旧模型、标准化及正式结果不变；只有所选两视频处理完成，其他视频仅有身份/元数据清单。']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text);(out/'review_summary.md').write_text(text)
    return actual


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();base,ref,out=map(lambda x:Path(x).resolve(),(args.baseline,args.reference,args.output))
    if base.name!='v8_full_coverage_observation_field_identityfix_v1' or ref.name!='v8_local_geometry_encoding_pilot_v1': raise ValueError('FROZEN_ASSET_ROOTS_REQUIRED')
    if out in (base,ref) or base in out.parents or ref in out.parents: raise ValueError('OUTPUT_ISOLATION')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'protocol.json').exists() and not args.resume: raise ValueError('RESUME_REQUIRED')
    def term(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,term);torch.set_num_threads(2)
    # Keep full-window and one-frame convolution paths within the frozen numerical tolerance.
    # No parameter/model/training change; TF32 batch-dependent roundoff was measured in smoke.
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA_REQUIRED')
    start=time.monotonic()
    try:
        manifest,selected=video_manifest(base)
        for row in selected:
            if sha256_file(row['video_path'])!=row['video_sha256']: raise ValueError('VIDEO_SHA_MISMATCH')
        records=read_json(ref/'models'/'fold_models.json')['records'];record=[r for r in records if r['condition']=='MEAN_REPLICATED' and r['seed']==20260909]
        if len(record)!=1 or record[0]['epoch']!=100 or record[0]['status']!='COMPLETE': raise ValueError('FROZEN_MODEL_RECORD')
        record=record[0];checkpoint=Path(record['checkpoint']);stats=read_json(ref/'standardizer.json');grouping=read_json(ref/'grouping_config.json')
        if sha256_file(checkpoint)!=record['checkpoint_sha256'] or sha256_file(ref/'standardizer.json')!=record['identity']['standardizer_sha256']: raise ValueError('MODEL_STAT_HASH')
        oldp=read_json(base/'protocol.json');moge_path=oldp['frontend']['geometry']['checkpoint'];tracker_path=oldp['frontend']['tracking']['checkpoint']
        frozen={}
        for root in (base,ref):
            for path in (root/'protocol.json',root/'models'/'fold_models.json',root/'grouping_config.json',root/'report.md'):
                frozen[str(path)]=sha256_file(path)
            for rec in read_json(root/'models'/'fold_models.json')['records']:
                frozen[rec['checkpoint']]=sha256_file(rec['checkpoint'])
            for path in (root/'models').glob('standardizer*.json'): frozen[str(path)]=sha256_file(path)
            for path in (root/'scores').glob('*.csv'): frozen[str(path)]=sha256_file(path)
        frozen[str(ref/'standardizer.json')]=sha256_file(ref/'standardizer.json')
        config={'schema':'v8-full-video-stream-v1','model':record,'baseline_manifest_sha':sha256_file(base/'data_manifest.json'),
                'stats_sha':sha256_file(ref/'standardizer.json'),'grouping_sha':sha256_file(ref/'grouping_config.json'),
                'frontend_protocol_sha':sha256_file(base/'protocol.json'),'moge_sha':sha256_file(moge_path),'tracker_sha':sha256_file(tracker_path),
                'rgb_sha':sha256_file(oldp['rgb_feature']['weights_path']),'decode':'PyAV native all frames; relative seconds=PTS*time_base-stream_start_time*time_base',
                'chunks':16,'selection':[r['video_id'] for r in selected], 'training_started':False,'frozen_file_hashes':frozen,
                'numeric_runtime':{'detector_cudnn_tf32':False,'frontend_cudnn_tf32':True,'matmul_tf32':False,'dtype':'float32 model; float64 area accumulator; frozen MoGe native use_fp16=True'},
                'code_sha':{name:sha256_file(Path(__file__).parent/name) for name in ('model.py','inputs.py','runner.py','validation.py')}}
        identity=digest(config)
        if (out/'protocol.json').exists() and read_json(out/'protocol.json')['identity']!=identity: raise ValueError('PROTOCOL_IDENTITY_MISMATCH')
        write_json_atomic(out/'protocol.json',{**config,'identity':identity,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
        write_json_atomic(out/'video_manifest.json',{'rows':manifest,'count':len(manifest),'policy':'ALL_DECODE_FRAMES'})
        write_json_atomic(out/'pilot_selection.json',{'videos':selected,'rule':'train; source then pair identity sort; both metadata frame_count>=63; no labels/scores selection'})
        write_json_atomic(out/'final_status.json',{'status':'RUNNING','pid':os.getpid(),'completion_scope':'two_video_engineering_pilot','training_started':False})
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if c['identity']!=record['identity']: raise ValueError('CHECKPOINT_IDENTITY')
        model=LocalGeometryModel('MEAN_REPLICATED',stats).cuda().eval();model.load_state_dict(c['model'],strict=True)
        (out/'validation').mkdir(exist_ok=True)
        if not (out/'validation'/'window.json').exists():
            im=read_json(ref/'inputs'/'manifest.json');ds=Dataset([im['rows'][0]],stats);item=ds[0]
            batch={k:v[None].cuda() for k,v in item.items() if isinstance(v,torch.Tensor) and k not in ('label','index')}
            check=validate_window(model,batch);check.update(window_id=im['rows'][0]['window_id']);write_json_atomic(out/'validation'/'window.json',check)
            del batch,ds
        ts=time.monotonic();moge=MogeV2(moge_path);tracker=CoTrackerShortPrefix(tracker_path);rgb=FrozenResNet18Spatial()
        initialization_s=time.monotonic()-ts;print(f"PROVIDERS_LOADED device={torch.cuda.get_device_name()} source={selected[0]['source_id']} initialization_s={initialization_s:.2f}",flush=True)
        write_json_atomic(out/'provider_runtime.json',{'tracker_internal_hw':list(tracker.model.interp_shape), 'tracker_step':tracker.model.step,
                    'tracker_predictor_sha':sha256_file(Path(__import__('cotracker.predictor',fromlist=['x']).__file__)),
                    'torch':torch.__version__,'device':torch.cuda.get_device_name(),'initialization_s':initialization_s})
        summaries=[]
        for row in selected:
            directory=out/'responses'/row['video_id'];sp=directory/'summary.json'
            if sp.exists():
                s=read_json(sp)
                if s['identity']!=identity or s['status']!='COMPLETE': raise ValueError('COMPLETED_VIDEO_IDENTITY')
            else:
                frontend=Frontend(base,out,moge,tracker,rgb,stats,grouping,identity)
                state_path=directory/'state.pt'
                if state_path.exists():
                    saved=torch.load(state_path,map_location='cpu',weights_only=False);frontend.cost.update(saved['cost'].get('frontend',{}))
                s=process_video(row,out,frontend,model,identity)
            row.update(processed=True,actual_decoded_frame_count=s['decoded_frame_count'],timestamp_source='NATIVE_PTS')
            summaries.append(s)
        write_json_atomic(out/'video_manifest.json',{'rows':manifest,'count':len(manifest),'policy':'ALL_DECODE_FRAMES'})
        peak={'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        actual=report(out,manifest,selected,summaries,initialization_s,peak,identity)
        for path,sha in frozen.items():
            if sha256_file(path)!=sha: raise ValueError(f'FROZEN_ASSET_CHANGED:{path}')
        write_json_atomic(out/'final_status.json',{'status':'COMPLETE','completion_scope':'two_video_engineering_pilot','manifest_video_count':len(manifest),
                    'selected_video_count':2,'completed_video_count':2,'decoded_frame_count':actual,'frontend_frame_count':actual,'response_frame_count':actual,
                    'training_started':False,'full_dataset_processed':False,'elapsed_this_invocation_s':time.monotonic()-start,'identity':identity})
        print('COMPLETE two_video_engineering_pilot',flush=True)
    except BaseException as e:
        write_json_atomic(out/'final_status.json',{'status':'STOPPED_SAFE' if isinstance(e,InterruptedError) else 'FAILED',
                           'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'elapsed_this_invocation_s':time.monotonic()-start,
                           'completion_scope':'two_video_engineering_pilot','training_started':False,'full_dataset_processed':False})
        raise


if __name__=='__main__': main()
