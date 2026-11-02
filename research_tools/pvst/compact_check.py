"""Limited resource audit, never train.main: failure -> counts -> <=4 windows.

Supports only the current PVST experiment. Window smoke losses/updates are not
formal full-video updates, no weights are saved, original run remains read-only.
"""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from scripts.prepare_v8_assets import digest
from research_tools.v8.local_geometry_pilot.runner import seed_all
from .data import Video,decoder_only,save
from .model import PVST
from .probe import parameters,losses,model_trace,scan_windows

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_point_voxel_v02')
OUTPUT=SOURCE.parent/'pvst_compact_execution_check_v1'
FAILED='cb41b3d004b6640f6c3b5bb1'


def host_memory():
    fields={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    return {'available_bytes':fields['MemAvailable'],'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def identity(config):
    return {'config':config,'manifest_sha256':digest(SOURCE/'manifest.json'),'scale_sha256':digest(SOURCE/'scale.json'),
        'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'cache_metadata_sha256':{p.parent.name:digest(p) for p in (SOURCE/'cache').glob('*/complete.json')}}


def window_loss(model,frames,start,gt,labels,epsilon):
    pred=model(frames,start)
    return torch.stack([losses(p,g,y,epsilon)[0] for p,g,y in zip(pred,gt,labels)]).mean()


def worker(config,out,window,condition,offload,chunk,profile=False):
    seed_all(config['seed']);torch.set_num_threads(4)
    c={**config,'attention_chunk':chunk,'offload_saved_tensors':offload}
    model=PVST(condition,decoder_only(c),c).cuda().train();opt=torch.optim.AdamW(parameters(c,model),weight_decay=c['weight_decay'])
    count=sum(p.numel() for p in model.parameters());keys_sha=hashlib.sha256('\n'.join(model.state_dict()).encode()).hexdigest()
    if count!=17821672 or keys_sha!='31beab46b45b34ed53ff89a70293a0e9e93abb86721dc51902e9c4c1808af665':raise ValueError('MODEL_IDENTITY_CHANGED')
    rows=json.loads((SOURCE/'manifest.json').read_text())['rows'];row=next(r for r in rows if r['video_id']==window['video_id']);scale=json.loads((SOURCE/'scale.json').read_text())['s']
    left,end=window['input_range'];start=window['target_range'][0];video=Video(SOURCE,row,condition,scale)
    result={'window':window,'condition':condition,'offload':offload,'attention_chunk':chunk,'ffn_chunk':c['ffn_chunk'],'profiled':profile,
        'parameters':count,'state_dict_keys_sha256':keys_sha,'host_before':host_memory(),'updates':[],'loss_scope':'window-only loss, two microbatches per smoke update; NOT formal video update; weights discarded'}
    tick=time.monotonic();torch.cuda.reset_peak_memory_stats();phase='input';prof=None
    try:
        if profile:
            prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
            prof.start()
        with torch.profiler.record_function('audit.data_read_transfer_geometry'):
            frames=video.frames(left,end,'cuda');gt=[video.gt(i) for i in range(start,end)];labels=[video.label(i) for i in range(start,end)]
        torch.cuda.synchronize();result['input_s']=time.monotonic()-tick
        fine=sum(len(f['voxel_meta']['coords']) for f in frames);coarse=sum(len(f['voxel_meta']['coarse_coords']) for f in frames)
        result.update(fine_nodes=fine,coarse_nodes=coarse,native_hw=row['source_hw'],target_frames=end-start,context_frames=start-left)
        if condition=='XYZ_PVST' and 'fine_nodes' in window and (fine,coarse)!=(window['fine_nodes'],window['coarse_nodes']):raise ValueError('COUNT_MISMATCH')
        steps=1 if profile else 2;micros=1 if profile else 2
        for step in range(steps):
            opt.zero_grad(set_to_none=True);forward_ms=backward_ms=0.;step_start=time.monotonic();accumulated=[]
            for micro in range(micros):
                accumulated.append(any(p.grad is not None for p in model.parameters()))
                events=[torch.cuda.Event(enable_timing=True) for _ in range(3)];events[0].record();phase='forward'
                with torch.profiler.record_function('audit.forward'),torch.autocast('cuda',dtype=torch.bfloat16):loss=window_loss(model,frames,start-left,gt,labels,c['dice_epsilon'])/micros
                events[1].record();phase='backward'
                with torch.profiler.record_function('audit.backward_recomputation'):loss.backward()
                events[2].record();events[2].synchronize();forward_ms+=events[0].elapsed_time(events[1]);backward_ms+=events[1].elapsed_time(events[2]);del loss
            phase='optimizer';opt_start=time.monotonic()
            paths={'geometry':model.geo[0].weight,'VFE':model.vfe[0].weight,'fine_qkv':model.fine_time.layers[0].qkv.weight,'coarse_qkv':model.coarse_time.layers[0].qkv.weight,'decoder':model.decoder.proj.weight}
            grads={k:float(v.grad.abs().sum()) if v.grad is not None else 0. for k,v in paths.items()}
            if step==1 and any(v<=0 for v in grads.values()):raise ValueError('PATH_NOT_LEARNING:'+str(grads))
            with torch.profiler.record_function('audit.optimizer'):
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],error_if_nonfinite=True));opt.step()
            torch.cuda.synchronize();model.check_bn()
            result['updates'].append({'update':step+1,'microbatches':micros,'had_prior_gradients':accumulated,'forward_s':forward_ms/1000,'backward_s':backward_ms/1000,
                'optimizer_s':time.monotonic()-opt_start,'total_s':time.monotonic()-step_start,'path_gradients':grads,'gradient_norm':norm,'adam_initialized':bool(opt.state)})
        if profile:
            prof.stop();stats=prof.key_averages();result['profiler_table']=stats.table(sort_by='self_cuda_time_total',row_limit=35)
            result['profile_scopes']={e.key:{'cpu_total_us':e.cpu_time_total,'device_total_us':e.device_time_total} for e in stats if e.key.startswith(('audit.','pvst.'))}
            prof=None
        else:
            phase='eval';model.eval();t=time.monotonic()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(frames,start-left)
            for p in pred:
                if p['pixel_logits'].shape!=tuple(row['source_hw']) or not torch.isfinite(p['pixel_logits']).all() or not torch.isfinite(p['frame_logit']):raise ValueError('EVAL_NATIVE_SHAPE_OR_FINITE')
            torch.cuda.synchronize();result['eval_s']=time.monotonic()-t;del pred
        result.update(status='PASS',total_s=time.monotonic()-tick)
    except BaseException as e:
        result.update(status='RESOURCE_BLOCKED' if isinstance(e,torch.OutOfMemoryError) else 'FAILED',phase=phase,error=str(e),traceback=traceback.format_exc(),trace=model_trace(model),total_s=time.monotonic()-tick)
    finally:
        if prof is not None:prof.stop()
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),host_after=host_memory())
        video.close()
    return result


def run_case(config,out,window,condition,offload=False,chunk=512,profile=False):
    spec={'window':window,'condition':condition,'offload':offload,'chunk':chunk,'profile':profile,'identity':identity(config)}
    name=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12];path=out/f'case_{name}.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['spec']!=spec:raise ValueError('RESOURCE_CASE_IDENTITY')
        return previous['result']
    specpath=out/f'spec_{name}.json';save(specpath,spec)
    print('CASE_START',condition,window['video_id'],window['target_range'],'chunk',chunk,'offload',offload,'profile',profile,flush=True)
    with (out/f'case_{name}.log').open('w') as log:
        p=subprocess.run([sys.executable,'-u','-m','research_tools.pvst.compact_check','--worker',str(specpath),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
    if not path.exists():raise RuntimeError(f'WORKER_EXIT_{p.returncode}: see case_{name}.log')
    result=json.loads(path.read_text())['result'];print('CASE_END',result['status'],round(result['total_s'],2),'s',round(result['peak_allocated_bytes']/2**30,2),'GiB',flush=True)
    return result


def estimates(scan,results):
    """Illustrative workload envelopes, not validated bounds or precise ETAs."""
    answer={};windows=[w for w in scan['windows'] if w['split']=='train']
    for condition in ['UV_PVST','XYZ_PVST']:
        rr=[r for r in results if r['condition']==condition and r['status']=='PASS' and r['context_frames']==16 and not r['profiled']]
        if not rr:answer[condition]={'epoch_s':None,'reason':'no successful full-history representative'};continue
        def load(w,uv=False):
            n=w['input_range'][1]-w['input_range'][0];h,wd=w['native_hw']
            if uv:
                def bins(size,coarse):
                    v=np.floor((2*(np.arange(size,dtype=np.float32)+.5)/size-1)/(1/32)).astype(np.int64)
                    return len(np.unique(v//2 if coarse else v))
                f=bins(h,False)*bins(wd,False)*n;c=bins(h,True)*bins(wd,True)*n
            else:f=w['fine_nodes'];c=w['coarse_nodes']
            return np.array([h*wd*n,f,c],float)
        projections=[]
        for w in windows:
            costs=[]
            for r in rr:
                ratios=load(w,condition=='UV_PVST')/load(r['window'],condition=='UV_PVST')
                measured=sum(u['forward_s']+u['backward_s'] for u in r['updates'])/sum(u['microbatches'] for u in r['updates'])
                costs.extend([measured*min(1.,float(ratios.min())),measured*max(1.,float(ratios.max()))])
            projections.append((min(costs),max(costs)))
        answer[condition]={'epoch_training_s_envelope':[sum(x[0] for x in projections),sum(x[1] for x in projections)],
            'basis':'each measured full-history representative; native input pixels, fine and coarse nodes ratios, min/max with fixed-overhead floor/ceiling; illustrative, NOT guaranteed bounds; input IO/epoch evaluation extra',
            'routine_eval':'not extrapolated from first8; representative eval recorded separately'}
    return answer


def write_report(out,summary):
    lines=['# PVST紧凑执行限定验收','', '状态：'+summary['status']+'；正式训练仍0/2，本入口没有正式训练调用。',
        '所有窗口更新仅资源探针，权重丢弃；旧运行和协议只读。完整数学设计、数据、监督和感受野不变。',
        '实现：帧级[T,81,4]共享关系；排序整数键及query按需精确邻居；FFN块内残差；不保存全层展开图。',
        '', '|条件|video|目标/上下文|fine/coarse|卸载|chunk|allocated/reserved GiB|前向/反向秒每micro|总秒|RSS峰值GiB|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary['results']:
        u=r['updates'];n=sum(x['microbatches'] for x in u);f=sum(x['forward_s'] for x in u)/n if n else None;b=sum(x['backward_s'] for x in u)/n if n else None
        lines.append('|'+ '|'.join(map(str,[r['condition'],r['window']['video_id'],[r['window']['target_range'],r['window']['context_range']],[r.get('fine_nodes'),r.get('coarse_nodes')],r['offload'],r['attention_chunk'],[round(r['peak_allocated_bytes']/2**30,3),round(r['peak_reserved_bytes']/2**30,3)],[f,b],round(r['total_s'],3),round(r['host_after']['rss_peak_bytes']/2**30,3)]))+'|')
    lines+=['','阶段与异常（如有）：'+json.dumps(summary.get('blocking'),ensure_ascii=False),
        '','估算：'+json.dumps(summary.get('estimates'),ensure_ascii=False),
        '','已测窗口不证明所有窗口可运行；吞吐采用非profiler实测。节点计数不能独立预测峰值。',
        '目标帧/秒、各micro耗时、optimizer/eval时间、梯度累积状态、host available/RSS及操作trace见result_summary.json及case文件。']
    if (out/'correctness.txt').exists():lines+=['','正确性：', '```text',(out/'correctness.txt').read_text(),'```']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--worker',type=Path);p.add_argument('--failure-only',action='store_true');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'docs/pvst/config_v02.json').read_text());config={**config,'execution_revision':'compact_v1','attention_chunk':512,'ffn_chunk':8192}
    if a.worker:
        spec=json.loads(a.worker.read_text())
        if spec['identity']!=identity(config):raise ValueError('WORKER_CODE_OR_INPUT_CHANGED')
        result=worker(config,out,spec['window'],spec['condition'],spec['offload'],spec['chunk'],spec['profile'])
        target=out/('case_'+a.worker.stem.removeprefix('spec_')+'.json');save(target,{'spec':spec,'result':result});return
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'result_summary.json').exists():
        old=json.loads((out/'result_summary.json').read_text())
        if old['status']=='COMPLETE' and old['identity']==identity(config):print('ALREADY_COMPLETE');return
    if digest(Path(config['checkpoint']))!=config['checkpoint_sha256']:raise ValueError('PINNED_CHECKPOINT_CHANGED')
    started=time.monotonic();summary={'status':'RUNNING','formal_models_completed':0,'results':[],'identity':identity(config),'host':host_memory()};save(out/'identity.json',summary['identity'])
    def progress(stage,**kw):save(out/'progress.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'unix_time':time.time(),**kw});print('STAGE',stage,kw,flush=True)
    failed={'video_id':FAILED,'target_range':[0,8],'context_range':[0,0],'input_range':[0,8],'reasons':['known_failure']};offload=False
    try:
        progress('direct_failure_window')
        r=run_case(config,out,failed,'XYZ_PVST');summary['results'].append(r)
        if r['status']=='RESOURCE_BLOCKED':
            if host_memory()['available_bytes']<16*2**30:raise RuntimeError('HOST_MEMORY_NOT_ENOUGH_FOR_OFFLOAD')
            offload=True;progress('direct_failure_offload_fallback');r=run_case(config,out,failed,'XYZ_PVST',True);summary['results'].append(r)
        if r['status']!='PASS':summary['blocking']=r;raise RuntimeError('KNOWN_FAILURE_WINDOW_NOT_PASSED')
        if a.failure_only:summary['status']='FAILURE_WINDOW_PASS';return
        progress('coordinate_count');scan=scan_windows(config,out,progress,source=SOURCE);summary['coordinate_count_s']=scan['seconds']
        representatives=scan['representatives'];summary['representatives']=representatives
        # One GPU-only pass. If a representative OOMs, one scoped-offload fallback
        # is allowed; recheck matching conditions, never reduce target8/halo16.
        for attempt in range(2 if not offload else 1):
            blocked=False
            for w in representatives:
                for condition in ['XYZ_PVST','UV_PVST']:
                    progress('representative',condition=condition,video=w['video_id'],target=w['target_range'],offload=offload)
                    r=run_case(config,out,w,condition,offload);summary['results'].append(r)
                    if r['status']!='PASS':summary['blocking']=r;blocked=True;break
                if blocked:break
            if not blocked:break
            if offload or r['status']!='RESOURCE_BLOCKED' or host_memory()['available_bytes']<16*2**30:raise RuntimeError('REPRESENTATIVE_NOT_PASSED')
            offload=True
        summary['selected_offload']=offload;summary['selected_chunk']=512
        # At most ONE larger-query short test, same full-history XYZ representative.
        w=next(w for w in representatives if w['context_range'][1]-w['context_range'][0]==16)
        if not offload:
            progress('one_larger_chunk_test');larger=run_case(config,out,w,'XYZ_PVST',False,2048);summary['results'].append(larger)
            base=next(r for r in summary['results'] if r['condition']=='XYZ_PVST' and r['window']['video_id']==w['video_id'] and r['window']['target_range']==w['target_range'] and not r['offload'])
            if larger['status']=='PASS' and larger['total_s']<base['total_s'] and larger['peak_allocated_bytes']<22*2**30:summary['selected_chunk']=2048
        summary['estimates']=estimates(scan,[r for r in summary['results'] if r['offload']==offload and r['attention_chunk']==512])
        w=next(w for w in representatives if 'original_high_resolution' in w['reasons']);progress('short_profiler')
        profile=run_case(config,out,w,'XYZ_PVST',offload,summary['selected_chunk'],True);save(out/'profiler_summary.json',profile)
        summary['status']='COMPLETE' if profile['status']=='PASS' else 'RESOURCE_BLOCKED';summary['profile_status']=profile['status']
    except BaseException as e:
        summary['status']='RESOURCE_BLOCKED' if 'NOT_PASSED' in str(e) or 'HOST_MEMORY' in str(e) else 'FAILED';summary['error']=str(e);summary['traceback']=traceback.format_exc()
    finally:
        summary['elapsed_s']=time.monotonic()-started;save(out/'result_summary.json',summary);write_report(out,summary)
        save(out/'progress.json',{'status':summary['status'],'stage':'resource_report_complete','pid':os.getpid(),'unix_time':time.time(),'formal_models_completed':0})
        (out/'exit_code.txt').write_text('0\n' if summary['status'] in ['COMPLETE','FAILURE_WINDOW_PASS'] else '1\n')
    if summary['status'] not in ['COMPLETE','FAILURE_WINDOW_PASS']:raise SystemExit(1)

if __name__=='__main__':main()
