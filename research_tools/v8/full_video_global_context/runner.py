"""All frozen videos, one shared causal frontend, six independent recurrent states."""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.component_global_context_pilot.model import GlobalContextModel, CONDITIONS
from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file, write_npz_atomic
from research_tools.v8.full_coverage_observation_field.frontend import MogeV2, CoTrackerShortPrefix, letterbox
from research_tools.v8.full_coverage_observation_field.pipeline import FrozenResNet18Spatial
from research_tools.v8.local_geometry_pilot.data import Dataset, digest
from research_tools.v8.full_video_streaming.inputs import Frontend, decode, video_manifest
from research_tools.v8.full_video_streaming.runner import load_input, save_pt, tensors
from research_tools.v8.full_video_streaming.model import state_to
from .model import GlobalStreamingModel
from .cache import FIELDS, read_block, write_block
from .validation import validate
from .evaluate import SEEDS, window_addendum, evaluate

STOP=False
BLOCK=64


def progress(out, **fields):
    save_json(out/'progress.json',{'status':'RUNNING','updated_unix':time.time(),**fields})


class Providers:
    def __init__(self,base,out,stats,group,identity):
        self.base,self.out,self.stats,self.group,self.identity=base,out,stats,group,identity
        self.tools=None

    def frontend(self):
        if self.tools is None:
            p=read_json(self.base/'protocol.json');ts=time.monotonic()
            self.tools=(MogeV2(p['frontend']['geometry']['checkpoint']),
                        CoTrackerShortPrefix(p['frontend']['tracking']['checkpoint']),FrozenResNet18Spatial())
            save_json(self.out/'provider_runtime.json',{'initialization_s':time.monotonic()-ts,'device':torch.cuda.get_device_name(),
                      'torch':torch.__version__,'tracker_internal_hw':list(self.tools[1].model.interp_shape),
                      'tracker_step':self.tools[1].model.step,'past_context':'last two actual frames plus current; no future',
                      'tracker_queries':1024,'geometry':'MoGe native frozen FP16; detector float32 TF32 disabled'})
            print('CUDA_PROVIDERS_LOADED',flush=True)
        return Frontend(self.base,self.out,*self.tools,self.stats,self.group,self.identity,persist_raw=False)


def legacy_frame(old,row,fi,info,canvas):
    p=old/'inputs'/row['video_id']/f'{fi:06d}.npz'
    if not p.exists(): return None
    protocol=read_json(old/'protocol.json')
    arrays,meta,ctx=load_input(old,row,fi,protocol['identity'])
    if meta['pts_ticks']!=info['pts_ticks'] or abs(meta['pts_s']-info['pts_s'])>1e-9:
        raise ValueError('LEGACY_CONTINUOUS_PTS')
    if not np.array_equal(ctx[1],canvas[:,:,::-1]): raise ValueError('LEGACY_CONTINUOUS_RGB')
    if meta['tracker_time']['input_frame_indices']!=list(range(max(0,fi-2),fi+1)):
        raise ValueError('LEGACY_CONTINUOUS_CONTEXT')
    return {k:arrays[k] for k in FIELDS},{**meta,'reused_input_path':str(p),'reused_input_sha256':sha256_file(p)},ctx


def process(row,out,old,providers,models,identity,completed_videos,*,total_videos=272,validate_first_block=True,model_timings=False):
    directory=out/'responses'/row['video_id'];directory.mkdir(parents=True,exist_ok=True)
    sp=directory/'summary.json';state_path=directory/'state.pt'
    if sp.exists():
        saved=read_json(sp)
        if saved['identity']!=identity or saved['video_sha256']!=row['video_sha256'] or saved['status']!='COMPLETE':
            raise ValueError('COMPLETED_VIDEO_IDENTITY')
        return saved
    states={k:None for k in models};records=[];next_frame=0;frontend=None;context=[]
    costs={'decode_s':0.,'model_s':0.,'write_s':0.,'legacy_reused_frames':0,'compact_reused_frames':0,'new_frontend_frames':0}
    coverage={'content_pixels':0,'geometry_valid_pixels':0,'active_cell_times':0,'no_predecessor_cell_times':0,
              'component_count_histogram':[0]*257,'component_size_counts':[0]*5,'estimated_timestamp_frames':0}
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity or saved['video_sha256']!=row['video_sha256']: raise ValueError('VIDEO_STATE_IDENTITY')
        states={k:state_to(v,'cuda') for k,v in saved['states'].items()}
        if set(states)!=set(models): raise ValueError('STATE_MODEL_SET')
        records=saved['blocks'];next_frame=saved['next_frame'];costs=saved['costs'];coverage=saved['coverage']
        end=0
        for r in records:
            if r['start']!=end or sha256_file(r['response_path'])!=r['response_sha256']: raise ValueError('RESPONSE_BLOCK_GAP_HASH')
            read_block(r['input_path'],identity,row['video_sha256']);end=r['end']+1
        if end!=next_frame or any(s['frame_count']!=next_frame for s in states.values()): raise ValueError('STATE_FRAME_COUNT')
    start=time.monotonic();checkpoint_clock=start;buf=[];metas=[];cell=[];fl=[];decoded=0;existing=None
    iterator=iter(decode(row['video_path']))
    print(f"VIDEO_START {row['source_id']} {row['role']} next_frame={next_frame}",flush=True)

    def commit():
        nonlocal buf,metas,cell,fl,records,checkpoint_clock
        ts=time.monotonic();first,last=metas[0]['frame_index'],metas[-1]['frame_index']
        ip=out/'inputs'/row['video_id']/f'{first:06d}.npz'
        if not ip.exists(): write_block(ip,buf,metas,identity,row['video_sha256'])
        else:
            restored,_=read_block(ip,identity,row['video_sha256'])
            if len(restored)!=len(buf): raise ValueError('REUSED_BLOCK_COUNT')
        rp=directory/f'{first:06d}.npz'
        write_npz_atomic(rp,cell_logits=np.stack(cell).astype(np.float32),frame_logits=np.stack(fl).astype(np.float64),
                         frame_indices=np.array([m['frame_index'] for m in metas],np.int64),
                         pts_s=np.array([m['pts_s'] for m in metas],np.float64))
        records=records+[{'start':first,'end':last,'input_path':str(ip),'response_path':str(rp),'response_sha256':sha256_file(rp)}]
        if frontend: costs['frontend']=frontend.cost.copy()
        costs['write_s']+=time.monotonic()-ts
        costs['wall_s']=costs.get('wall_s',0.)+time.monotonic()-checkpoint_clock
        checkpoint_clock=time.monotonic()
        save_pt(state_path,{'identity':identity,'video_sha256':row['video_sha256'],
                'states':{k:state_to(v,'cpu') for k,v in states.items()},'next_frame':last+1,
                'blocks':records,'costs':costs,'coverage':coverage,'past_context_indices':[c[0] for c in context]})
        progress(out,stage='continuous_frontend_and_frozen_models',completed_videos=completed_videos,total_videos=total_videos,
                 source=row['source_id'],role=row['role'],video_id=row['video_id'],completed_frames=last+1,
                 metadata_frames=row['metadata_frame_count'],elapsed_video_this_invocation_s=time.monotonic()-start)
        print(f"BLOCK {row['source_id']} {row['role']} {last+1}/{row['metadata_frame_count']} videos={completed_videos}/{total_videos} elapsed_s={time.monotonic()-start:.1f}",flush=True)
        buf=[];metas=[];cell=[];fl=[]
        if STOP: raise InterruptedError('SAFE_TERM_AFTER_COMMITTED_BLOCK')

    while True:
        ts=time.monotonic()
        try: fi,bgr,info=next(iterator)
        except StopIteration: break
        costs['decode_s']+=time.monotonic()-ts;decoded+=1
        canvas,mapping=letterbox(bgr);ctx=(fi,canvas[:,:,::-1].copy(),info['pts_s'])
        if fi<next_frame:
            context=(context+[ctx])[-2:];continue
        first=fi-fi%BLOCK
        if fi==first:
            ip=out/'inputs'/row['video_id']/f'{first:06d}.npz'
            existing=read_block(ip,identity,row['video_sha256']) if ip.exists() else None
        if existing:
            aa,mm=existing;a,meta=aa[fi-first],mm[fi-first]
            if meta['frame_index']!=fi or abs(meta['pts_s']-info['pts_s'])>1e-9 or meta['pts_ticks']!=info['pts_ticks'] or meta['mapping']!=mapping:
                raise ValueError('COMPACT_DECODE_IDENTITY')
            costs['compact_reused_frames']+=1
        else:
            reused=legacy_frame(old,row,fi,info,canvas)
            if reused:
                a,meta,ctx=reused;costs['legacy_reused_frames']+=1
            else:
                if frontend is None:
                    frontend=providers.frontend()
                    if 'frontend' in costs: frontend.cost.update(costs['frontend'])
                previous_active=None if fi==0 else next(iter(states.values()))['previous_active'][0].cpu().numpy()
                a,meta,ctx,_=frontend.build(row,bgr,info,context,previous_active)
                a={k:a[k] for k in FIELDS};costs['new_frontend_frames']+=1
        f=tensors(a);torch.cuda.synchronize();ts=time.monotonic();rr=[]
        events=[]
        for key,stream in models.items():
            if model_timings:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            result=stream.forward_step(f,states[key]);states[key]=result['state'];rr.append(result)
            if model_timings:end.record();events.append((key,begin,end))
        torch.cuda.synchronize();costs['model_s']+=time.monotonic()-ts
        for key,begin,end in events:
            timing=costs.setdefault('model_cuda_s_by_key',{});timing[key]=timing.get(key,0.)+begin.elapsed_time(end)/1000
        cell.append(np.stack([r['cell_logits'][0].cpu().numpy() for r in rr]))
        fl.append(np.array([float(r['frame_logit']) for r in rr]))
        context=(context+[ctx])[-2:];buf.append(a);metas.append({**meta,'video_sha256':row['video_sha256']})
        active=a['area']>0;_,sizes=np.unique(a['component_ids'][active],return_counts=True)
        if sizes.max()>4: raise ValueError('MAX_COMPONENT_SIZE')
        coverage['component_count_histogram'][len(sizes)]+=1
        for size in sizes: coverage['component_size_counts'][int(size)]+=1
        coverage['content_pixels']+=int(a['content'].sum());coverage['geometry_valid_pixels']+=int((a['valid']&a['content']).sum())
        coverage['active_cell_times']+=int(active.sum())
        coverage['no_predecessor_cell_times']+=int((active&(a['association_weight'].sum(-1)==0)).sum())
        coverage['estimated_timestamp_frames']+=int(info['timestamp_source']!='NATIVE_PTS')
        if fi==0:
            import cv2
            display=out/'display';display.mkdir(exist_ok=True);cv2.imwrite(str(display/f"{row['video_id']}__first.png"),canvas)
        if len(buf)==BLOCK:
            if validate_first_block and not (out/'native_continuous_validation.json').exists():
                batch={k:torch.from_numpy(np.stack([a[k] for a in buf]))[None].cuda() for k in FIELDS}
                checks={key:validate(stream.model,batch) for key,stream in models.items() if key.endswith(str(SEEDS[0]))}
                save_json(out/'native_continuous_validation.json',{'status':'PASS','video_id':row['video_id'],
                          'native_source_frames':[m['frame_index'] for m in metas],
                          'checks':checks,'input':'actual first 64 consecutive native frames, frozen history/graph'})
                del batch
            commit()
    if buf: commit()
    if not decoded or any(s is None or s['frame_count']!=decoded for s in states.values()): raise ValueError('VIDEO_EOF_COUNT')
    result={'status':'COMPLETE','identity':identity,'video_id':row['video_id'],'video_sha256':row['video_sha256'],
            'source_id':row['source_id'],'role':row['role'],'split':row['split'],'decoded_frames':decoded,
            'frontend_frames':decoded,'response_frames_per_model':decoded,'models':list(models),
            'video_logits':{k:float(s['log_numerator']-s['total_area'].log()) for k,s in states.items()},
            'blocks':records,'costs':costs,'coverage':coverage,'metadata_frame_difference':decoded-row['metadata_frame_count'],
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    save_json(sp,result)
    print(f"VIDEO_COMPLETE {row['source_id']} {row['role']} decoded={decoded}",flush=True)
    return result


def finish(out,manifest,summaries,identity):
    ts=time.monotonic();result=evaluate(out,manifest,summaries);elapsed=time.monotonic()-ts
    count=sum(s['decoded_frames'] for s in summaries)
    disk=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    costs={k:sum(s['costs'].get(k,0) for s in summaries) for k in ('decode_s','model_s','write_s','wall_s','legacy_reused_frames','compact_reused_frames','new_frontend_frames')}
    costs['frontend']={k:sum(s['costs'].get('frontend',{}).get(k,0) for s in summaries) for k in ('geometry_s','tracking_s','rgb_s','features_s','write_s','new_geometry','cache_hit')}
    coverage={k:sum(s['coverage'][k] for s in summaries) for k in ('content_pixels','geometry_valid_pixels','active_cell_times','no_predecessor_cell_times','estimated_timestamp_frames')}
    coverage['component_count_histogram']=np.sum([s['coverage']['component_count_histogram'] for s in summaries],axis=0).tolist()
    coverage['component_size_counts']=np.sum([s['coverage']['component_size_counts'] for s in summaries],axis=0).tolist()
    coverage['content_assignment_coverage']=1.
    stats={'decoded_frames':count,'cache_bytes':disk,'available_bytes':os.statvfs(out).f_bavail*os.statvfs(out).f_frsize,
           'evaluation_s':elapsed,'gpu_peak_allocated_bytes':max(s['gpu_peak_allocated_bytes'] for s in summaries),
           'gpu_peak_reserved_bytes':max(s['gpu_peak_reserved_bytes'] for s in summaries),'costs':costs,'coverage':coverage}
    save_json(out/'coverage_cost.json',stats)
    lines=['# V8 全视频连续帧冻结评价','', '**窗口训练模型的全视频连续帧冻结评价，不是全视频训练完成。**','',
           f'状态COMPLETE；{len(manifest)}视频、{len({r["source_id"] for r in manifest})} source、{count}实际解码帧；六模型每帧评分完整。',
           '全部正常解码帧顺序处理，原生PTS与估计来源逐帧保存。无训练、统计重拟合或阈值调整。',
           '', '| split | condition | seed | pooled video AUROC | AP | P | R | F1 | ACC | TN/FP/FN/TP |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for m in result['metrics']:
        lines.append(f"| {m['split']} | {m['condition']} | {m['seed']} | "+' | '.join('NA' if m[k] is None else f'{m[k]:.6f}' for k in ('auroc','ap','precision','recall','f1','accuracy'))+f" | {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']} |")
    lines+=['','## 主要比较',str(result['paired_comparison']),str(result['source_directions']),
            '每seed独立完成整视频面积LSE后平均最终video logit；主要指标pooled video AUROC。source配对bootstrap10000次/20260909；逐source一真一假AUROC仅0/.5/1，是配对排序。',
            '', '## 与窗口实验的关系',
            '旧窗口macro为COMPONENT 0.896825、CELL 0.904762；不是同一输入/历史/聚合单位，不能直接作提升差。旧窗口辅助CI另存window_auxiliary_bootstrap.json，不改变旧主指标。',
            '验证source已多轮开发使用，不是sealed test；未提供帧/像素真值，不计算定位准确率。',
            '', '## 输入与资源',str(stats),
            '一次共享前端；新帧不长期保存dense XYZ/depth/RGB。每64帧NPZ保存float32 cell输入/mu/状态/关联及无损packbits像素mask，可重新训练patch encoder。六模型hidden分别保存，块边界不重置。',
            '模型调用真实GlobalContextModel.spatial/region_context，不调用旧_spatial_content；GLOBAL_CELL使用singleton，COMPONENT使用动态分区。',
            '', '## 下一阶段（未执行）',
            '应先冻结视频级弱监督训练及长序列梯度截断方案；所需训练存储可复用本轮紧凑输入。推理成本不是训练成本，需另测梯度显存与吞吐；不自动训练或改架构。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'final_status.json',{'status':'COMPLETE','identity':identity,'completed_videos':len(summaries),
              'decoded_frames':count,'frontend_frames':count,'response_frames_per_model':count,'completed_models':6,
              'training_started':False,'scope':'all-video continuous frozen evaluation'})
    save_json(out/'progress.json',{'status':'COMPLETE','stage':'report','completed':len(summaries),'total':272,'updated_unix':time.time()})


def main():
    p=argparse.ArgumentParser()
    for key in ('baseline','reference','models','legacy','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    base,ref,modelroot,old,out=[Path(getattr(args,k)).resolve() for k in ('baseline','reference','models','legacy','output')]
    if (base.name,ref.name,modelroot.name,old.name)!=('v8_full_coverage_observation_field_identityfix_v1','v8_local_geometry_encoding_pilot_v1','v8_component_global_context_pilot_v1','v8_full_video_streaming_pilot_v1'):
        raise ValueError('FROZEN_V8_ASSET_ROOTS')
    if out in (base,ref,modelroot,old) or any(r in out.parents for r in (base,ref,modelroot,old)): raise ValueError('OUTPUT_ISOLATION')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'protocol.json').exists() and not args.resume: raise ValueError('RESUME_REQUIRED')
    def term(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,term);torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA_REQUIRED')
    started=time.monotonic();phase='plan'
    save_json(out/'final_status.json',{'status':'RUNNING','pid':os.getpid(),'training_started':False,'started_unix':time.time()})
    try:
        manifest,pilot=video_manifest(base)
        if len(manifest)!=272 or len({r['source_id'] for r in manifest if r['split']=='train'})!=122 or len({r['source_id'] for r in manifest if r['split']=='validation'})!=14:
            raise ValueError('VIDEO_QUEUE_COUNTS')
        if len({(r['pair_id'],r['source_id'],r['role'],r['video_sha256']) for r in manifest})!=272: raise ValueError('VIDEO_QUEUE_UNIQUE')
        for r in manifest:
            if sha256_file(r['video_path'])!=r['video_sha256']: raise ValueError(f"VIDEO_HASH:{r['video_id']}")
        if pilot[0]['source_id']!='1LARL': raise ValueError('FROZEN_FIRST_PAIR')
        pilot_ids={r['video_id'] for r in pilot};ordered=pilot+[r for r in manifest if r['video_id'] not in pilot_ids]
        stats=read_json(modelroot/'standardizer.json');group=read_json(modelroot/'grouping_config.json')
        if stats!=read_json(ref/'standardizer.json') or group!=read_json(ref/'grouping_config.json'): raise ValueError('FROZEN_STATISTICS')
        previous=read_json(old/'protocol.json');bp=read_json(base/'protocol.json')
        if previous['stats_sha']!=sha256_file(ref/'standardizer.json') or previous['grouping_sha']!=sha256_file(ref/'grouping_config.json'):
            raise ValueError('LEGACY_INPUT_STATISTICS')
        providers_hash={'moge':sha256_file(bp['frontend']['geometry']['checkpoint']),
                        'tracker':sha256_file(bp['frontend']['tracking']['checkpoint']),'rgb':sha256_file(bp['rgb_feature']['weights_path'])}
        if any(previous[k+'_sha']!=v for k,v in providers_hash.items()): raise ValueError('LEGACY_PROVIDER_WEIGHTS')
        records=read_json(modelroot/'models/fold_models.json')['records']
        if {(r['condition'],r['seed']) for r in records}!={(c,s) for c in CONDITIONS for s in SEEDS} or len(records)!=6: raise ValueError('SIX_MODELS_REQUIRED')
        frozen={str(modelroot/name):sha256_file(modelroot/name) for name in ('protocol.json','standardizer.json','grouping_config.json','models/fold_models.json','scores/window_scores.csv')}
        for r in records:
            if r['epoch']!=100 or r['status']!='COMPLETE' or sha256_file(r['checkpoint'])!=r['checkpoint_sha256']: raise ValueError('MODEL_RECORD_HASH_EPOCH')
            frozen[r['checkpoint']]=r['checkpoint_sha256']
        code=[Path(__file__).parent/f for f in ('runner.py','model.py','cache.py','validation.py','evaluate.py')]
        code += [Path(__file__).parents[1]/'full_video_streaming'/f for f in ('inputs.py','model.py')]
        config={'schema':'v8-full-video-global-context-v1','videos':[{k:r[k] for k in ('video_id','video_sha256','pair_id','source_id','role','split')} for r in manifest],
                'frozen_files':frozen,'providers_sha':providers_hash,'block_size':BLOCK,'frontend_protocol_sha':sha256_file(base/'protocol.json'),
                'code_sha':{str(f):sha256_file(f) for f in code},'legacy_identity':previous['identity'],
                'frame_policy':'all normal native decode frames, no sampling','ensemble':'mean final seed video logits after independent area LSE',
                'threshold':0,'bootstrap':{'unit':'source','replicates':10000,'seed':20260909},'training_started':False}
        identity=digest(config)
        if (out/'protocol.json').exists():
            if read_json(out/'protocol.json')['identity']!=identity: raise ValueError('PROTOCOL_RESUME_IDENTITY')
        else: save_json(out/'protocol.json',{**config,'identity':identity,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
        save_json(out/'video_manifest.json',{'count':272,'rows':manifest,'processing_order':[r['video_id'] for r in ordered]})
        phase='previous_window_addendum';progress(out,stage=phase);window_addendum(modelroot,out)
        models={};phase='saved_model_validation'
        windowrow=read_json(ref/'inputs/manifest.json')['rows'][0]
        ds=Dataset([windowrow],stats);item=ds[0]
        batch={k:v[None].cuda() for k,v in item.items() if isinstance(v,torch.Tensor) and k not in ('label','index','xyz')}
        checks={}
        for r in records:
            cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            if cp['identity']!=r['identity'] or cp['epoch']!=100: raise ValueError('CHECKPOINT_IDENTITY_EPOCH')
            key=f"{r['condition']}__seed{r['seed']}";model=GlobalContextModel(r['condition'],stats).cuda().eval()
            model.load_state_dict(cp['model'],strict=True);model.requires_grad_(False);models[key]=GlobalStreamingModel(model)
            if r['seed']==SEEDS[0]: checks[r['condition']]=validate(model,batch)
        save_json(out/'validation.json',{'checks':checks,'window_id':windowrow['window_id'],'status':'PASS'})
        del ds,batch,item
        providers=Providers(base,out,stats,group,identity);summaries=[];phase='continuous'
        for i,row in enumerate(ordered):
            s=process(row,out,old,providers,models,identity,i);summaries.append(s)
            if i==1:
                n=sum(x['decoded_frames'] for x in summaries);cache_bytes=sum(p.stat().st_size for p in (out/'inputs').rglob('*') if p.is_file())
                model_s=sum(x['costs']['model_s'] for x in summaries)
                total=sum(r['metadata_frame_count'] for r in manifest)
                projected=cache_bytes/n*total + total*6*256*4 + 512*2**20
                available=os.statvfs(out).f_bavail*os.statvfs(out).f_frsize
                save_json(out/'pilot_cost_projection.json',{'source':'1LARL','actual_frames':n,'six_model_s':model_s,
                          'new_cache_bytes':cache_bytes,'bytes_per_frame':cache_bytes/n,'metadata_total_frames':total,
                          'estimated_cache_and_responses_bytes':projected,'available_bytes':available,
                          'old_continuous_inputs_reused':True,'new_frontend_cost_reference':'legacy pilot measured costs; current pair reused, not remeasured frontend',
                          'elapsed_this_invocation_s':time.monotonic()-started,'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
                          'reserved_peak_bytes':torch.cuda.max_memory_reserved()})
                if projected>available: raise OSError(f'INSUFFICIENT_SPACE: estimated={projected}, available={available}')
        by_id={s['video_id']:s for s in summaries};summaries=[by_id[r['video_id']] for r in manifest]
        for path,sha in frozen.items():
            if sha256_file(path)!=sha: raise ValueError(f'FROZEN_ASSET_CHANGED:{path}')
        phase='evaluate_report';finish(out,manifest,summaries,identity)
        print('COMPLETE all 272 videos, six frozen models, evaluation and report',flush=True)
    except BaseException as e:
        save_json(out/'final_status.json',{'status':'STOPPED_SAFE' if isinstance(e,InterruptedError) else 'FAILED',
                  'phase':phase,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),
                  'elapsed_this_invocation_s':time.monotonic()-started,'training_started':False})
        raise


if __name__=='__main__': main()
