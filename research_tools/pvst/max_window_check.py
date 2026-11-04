"""One fixed XYZ window, one update, parent-owned 900s deadline; no training CLI.

Research caller: this bounded resource CLI. Outputs stage evidence, coordinate
statistics and fixed plots. Existing resource worker runs two updates/microbatches
and has no hard parent deadline, so this thin one-update entry reuses its model,
Video, optimizer groups, clip loss and trace helpers rather than its loop.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
OUT=BASE/'pvst_v03_max_window_check_v1'
SPARSE_OUT=BASE/'pvst_v03_sparse_max_window_check_v1'
BRIDGE_OUT=BASE/'pvst_csr_values_bridge_check_v1'/'full_window'
HEAD_OUT=BASE/'pvst_head_output_offload_check_v1'/'full_window'
VIDEO='782426ea0330c7fea684953c'
EXPECTED=(2970629,1488042)


def read(p):return json.loads(p.read_text())

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def terminate_group(process,grace=5):
    # Only our new-session process group. Never touch a GPU-wide process list.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=grace)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    return process.wait()


def bounded_process(command,seconds,log,sample=None):
    start=time.monotonic();timed_out=False;samples=[]
    with log.open('w') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(log.parent/'launch.json',{'pid':p.pid,'deadline_s':seconds,'unix_time':time.time(),'command':command})
        while p.poll() is None:
            remaining=seconds-(time.monotonic()-start)
            if remaining<=0:timed_out=True;terminate_group(p);break
            try:p.wait(timeout=min(5,remaining))
            except subprocess.TimeoutExpired:
                if sample and seconds-(time.monotonic()-start)>5:
                    samples.append(sample(p.pid));save(log.parent/'gpu_samples.json',samples)
        code=p.wait()
    return {'pid':p.pid,'returncode':code,'termination_signal':-code if code<0 else None,'timed_out':timed_out,'wall_s':time.monotonic()-start,'timeout_s':seconds,'samples':samples}


def host():
    import resource
    available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    current=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))
    return {'available_bytes':available,'rss_bytes':current,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def gpu_sample(pid):
    def query(fields,kind='gpu'):
        try:return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=2).strip()
        except (OSError,subprocess.SubprocessError):return 'unavailable'
    return {'unix_time':time.time(),'worker_pid':pid,'gpu_total_free_used_util':query('memory.total,memory.free,memory.used,utilization.gpu'),'compute_pid_mib':query('pid,used_gpu_memory','compute-apps')}


def member_stats(count):
    count=np.asarray(count,np.int64);values,n=np.unique(count,return_counts=True)
    return {'nodes':len(count),'points':int(count.sum()),'singleton_nodes':int((count==1).sum()),'singleton_fraction':float((count==1).mean()) if len(count) else None,
        'count_quantiles_exact':np.percentile(count,[0,25,50,75,95,99,100]).tolist() if len(count) else [],'histogram':dict(zip(map(str,values),map(int,n)))}



def candidate_layers(model,require_execution=False,require_offload=False):
    from . import sparse_attention
    import types
    head=sparse_attention.sparse_head
    if not isinstance(head,types.FunctionType) or head.__module__!=sparse_attention.__name__:raise ValueError('SPARSE_HEAD_WRAPPED_OR_REPLACED')
    evidence=[]
    for name,width,nodes in [('fine_time',256,EXPECTED[0]),('coarse_time',512,EXPECTED[1])]:
        module=getattr(model,name)
        if len(module.layers)!=3:raise ValueError('TEMPORAL_LAYER_COUNT')
        for index,(layer,dilation) in enumerate(zip(module.layers,[1,2,4])):
            if not layer.sparse_edge_attention or layer.dim!=width or layer.qkv_chunk!=8192 or layer.chunk!=512 or layer.ffn_chunk!=8192:raise ValueError('CANDIDATE_NOT_ENABLED_ALL_SIX')
            if require_offload and not layer.head_output_offload:raise ValueError('HEAD_OFFLOAD_NOT_ENABLED_ALL_SIX')
            row={'layer':name+'.'+str(index),'width':width,'dilation':dilation,'enabled':True,'head_source_sha256':sha(Path(sparse_attention.__file__))}
            if require_execution:
                stat=getattr(layer,'edge_stats',None)
                if not stat or stat.get('completed_heads')!=layer.heads or stat.get('nodes')!=nodes or stat.get('dilation')!=dilation or stat.get('head_impl')!=head.__module__+'.'+head.__name__:raise ValueError('MISSING_ACTUAL_SPARSE_EXECUTION:'+row['layer'])
                if stat.get('backend')!=sparse_attention.BACKEND or stat.get('softmax_impl')!='research_tools.pvst.sparse_attention.CSRSoftmax' or stat.get('values_bridge')!='research_tools.pvst.sparse_attention.CSRFromValues':raise ValueError('SPARSE_BACKEND_MISMATCH')
                row.update(stat)
                if require_offload:
                    storage=getattr(layer,'head_storage',{})
                    if not storage.get('enabled') or storage.get('stack_device')!='cpu' or storage.get('gpu_list_bytes')!=0 or len(storage.get('heads',[]))!=layer.heads:raise ValueError('MISSING_ACTUAL_HEAD_OFFLOAD:'+row['layer'])
                    if any(h['stored_device']!='cpu' or h['source_device']!='cuda:0' or h['dtype']!='torch.bfloat16' or not h['requires_grad'] or h['grad_fn']!='ToCopyBackward0' for h in storage['heads']):raise ValueError('HEAD_OFFLOAD_GRADIENT_OR_DTYPE')
                    row['head_storage']=storage
            evidence.append(row)
    return evidence


def worker(out):
    started=time.monotonic();spec=read(out/'identity.json');c=spec['config'];row=spec['row'];scale=spec['global_scale'];stages=[];phase='worker_start';m=None;v=None;done=threading.Event();lock=threading.Lock();trace={}
    # Timer includes Python imports, construction, input and the single update.
    import torch
    from .train import build
    from .data import Video
    from .probe import parameters,model_trace
    from .v03 import clip_terms
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    def snapshot():
        return {'phase':phase,'elapsed_s':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'host':host(),'trace':model_trace(m) if m else {}}
    def boundary(name,seconds=None,**kw):
        event={**snapshot(),'stage':name,'stage_s':seconds,**kw};stages.append(event)
        with lock:save(out/'stages.json',stages);save(out/'live.json',event)
        print('STAGE',name,event['elapsed_s'],flush=True)
    def monitor():
        while not done.wait(5):
            with lock:save(out/'live.json',snapshot())
    boundary('worker_start');thread=threading.Thread(target=monitor,daemon=True);thread.start();result={}
    try:
        tick=time.monotonic();phase='model_build';m=build(c,'XYZ_PVST').train();opt=torch.optim.AdamW(parameters(c,m),weight_decay=c['weight_decay'])
        if sum(p.numel() for p in m.parameters())!=17821672:raise ValueError('PARAMETER_COUNT_CHANGED')
        keys=hashlib.sha256('\n'.join(m.state_dict()).encode()).hexdigest()
        if keys!='31beab46b45b34ed53ff89a70293a0e9e93abb86721dc51902e9c4c1808af665':raise ValueError('STATE_DICT_CHANGED')
        if c.get('sparse_edge_attention'):
            if os.environ.get('CUDA_LAUNCH_BLOCKING') not in [None,'','0'] or any(mod._forward_hooks or mod._forward_pre_hooks or mod._backward_hooks for mod in m.modules()):raise ValueError('DIAGNOSTIC_HOOK_OR_BLOCKING_ENABLED')
            result['candidate_build_evidence']=candidate_layers(m,require_offload=c.get('head_output_offload',False))
        m.check_bn();torch.cuda.synchronize();boundary('model_built',time.monotonic()-tick,parameters=17821672,candidate_layers=result.get('candidate_build_evidence'))
        phase='data_geometry';tick=time.monotonic();v=Video(Path(c['asset_root']),row,'XYZ_PVST',scale,c);fs=v.frames(104,128,'cuda')
        fine=sum(len(f['voxel_meta']['coords']) for f in fs);coarse=sum(len(f['voxel_meta']['coarse_coords']) for f in fs)
        result.update(fine_nodes=fine,coarse_nodes=coarse)
        if (fine,coarse)!=EXPECTED:raise ValueError('WINDOW_OCCUPANCY_MISMATCH:'+str((fine,coarse)))
        if [f['pts'] for f in fs]!=[r['pts_s'] for r in row['pts'][104:128]]:raise ValueError('PTS_MISMATCH')
        if not c.get('sparse_edge_attention'):
            occupancy=[]
            for i,f in enumerate(fs,104):
                meta=f['voxel_meta'];counts=meta['count'].cpu().numpy();parent=meta['parent'].cpu().numpy();cc=np.bincount(parent,weights=counts,minlength=len(meta['coarse_coords'])).astype(np.int64)
                record={'frame':i,'fine':member_stats(counts),'coarse':member_stats(cc)}
                if i in [120,124,127]:
                    coords=meta['coords'];record['sampled_fine_centers']=(coords[::max(1,len(coords)//10000)].float().cpu().numpy()*.125+.0625).tolist()
                occupancy.append(record)
            save(out/'occupancy.json',occupancy)
        gt=[v.gt(i) for i in range(120,128)];labels=[v.label(i) for i in range(120,128)]
        torch.cuda.synchronize();boundary('data_loaded',time.monotonic()-tick,fine_nodes=fine,coarse_nodes=coarse,q_dtype=str(fs[0]['q'].dtype),token_dtype=str(fs[0]['tokens'].dtype),rgb_dtype=str(fs[0]['rgb'].dtype),target_labels=labels,spatial_gt='zero mask for ANAKIN real' if row['role']=='real' else 'actual masks')
        opt.zero_grad(set_to_none=True);phase='forward';tick=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=m(fs,16);terms=clip_terms(pred,gt,labels,c['dice_epsilon']);loss=terms[0]
        if not torch.isfinite(terms).all():raise ValueError('NONFINITE_LOSS')
        if len(pred)!=8 or any(tuple(p['pixel_logits'].shape)!=(720,1280) for p in pred):raise ValueError('NATIVE_OUTPUT_COVERAGE')
        result['loss_components']=dict(zip(['total','frame_BCE','pixel_BCE','soft_Dice_loss'],terms.float().tolist()))
        if c.get('sparse_edge_attention'):result['forward_edge_evidence']=candidate_layers(m,True,require_offload=c.get('head_output_offload',False))
        torch.cuda.synchronize();boundary('forward_complete',time.monotonic()-tick,finite_loss=True,output_dtype=str(pred[0]['pixel_logits'].dtype),loss_components=result['loss_components'],candidate_layers=result.get('forward_edge_evidence'))
        phase='backward';tick=time.monotonic();loss.backward();torch.cuda.synchronize();boundary('backward_complete',time.monotonic()-tick)
        phase='gradient_clip_update';tick=time.monotonic();paths={'geometry':m.geo[0].weight,'VFE':m.vfe[0].weight,'fine_qkv':m.fine_time.layers[0].qkv.weight,'coarse_qkv':m.coarse_time.layers[0].qkv.weight,'decoder':m.decoder.proj.weight,'pixel_output':m.pixel_out.weight,'frame_output':m.frame_residual[-1].weight}
        gradients={k:{'exists':p.grad is not None,'finite':bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,'norm':float(p.grad.float().norm()) if p.grad is not None else None} for k,p in paths.items()}
        selected={'pixel_output':m.pixel_out.weight,'frame_visual_output':m.frame_visual[-1].weight,'decoder_predictor':m.decoder.predictor.weight}
        before={k:np.asarray(p.tolist()) for k,p in selected.items()}
        norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),c['clip_norm'],error_if_nonfinite=True));opt.step();m.check_bn();torch.cuda.synchronize()
        changes={k:float(np.linalg.norm(np.asarray(p.tolist())-before[k])) for k,p in selected.items()}
        changed=any(value>0 for value in changes.values())
        if not changed or not opt.state:raise ValueError('NO_REAL_ADAM_UPDATE')
        result.update(parameter_update_verified=changed,parameter_delta_norms=changes)
        boundary('update_complete',time.monotonic()-tick,gradients=gradients,gradient_norm=norm,optimizer_state_parameters=len(opt.state),parameter_delta_norms=changes,parameter_update_verified=changed)
        result.update(status='PASS',updates=1,gradients=gradients,optimizer_state_parameters=len(opt.state),weights='discarded; no checkpoint saved')
    except BaseException as e:
        tb=e.__traceback__;context=[]
        while tb is not None:
            frame=tb.tb_frame
            if frame.f_code.co_name=='sparse_head':
                loc=frame.f_locals;q0=loc.get('q');col0=loc.get('col')
                context.append({'function':'sparse_head','head':loc.get('head'),'q_shape':list(q0.shape) if q0 is not None else None,'edges':len(col0) if col0 is not None else None,'dtype':str(q0.dtype) if q0 is not None else None})
            tb=tb.tb_next
        result['failure_operator_context']=context
        del tb
        result.update(status='OOM' if isinstance(e,torch.OutOfMemoryError) else 'FAILED',updates=0,error=str(e),traceback=traceback.format_exc(),failure_phase=phase)
    finally:
        done.set();thread.join(timeout=6)
        with lock:result.update(snapshot(),stages=stages);save(out/'worker_result.json',result)
        if v:v.close()
    return 0 if result['status']=='PASS' else 2 if result['status']=='OOM' else 1


def summarize_hist(records,key):
    h={}
    for r in records:
        for v,n in r[key]['histogram'].items():h[int(v)]=h.get(int(v),0)+n
    nodes=sum(h.values());return {'nodes':nodes,'points':sum(v*n for v,n in h.items()),'singleton_nodes':h.get(1,0),'singleton_fraction':h.get(1,0)/nodes if nodes else None,'histogram':dict(sorted(h.items())),'note':'per-frame voxel memberships concatenated; not a cross-time spatial union'}


def coordinate_audit(out,spec):
    from .data import native_dense
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    row=spec['row'];scale=spec['global_scale'];rng=np.random.default_rng(20260909);stats=[];samples=[];occupancy=read(out/'occupancy.json') if (out/'occupancy.json').exists() else [];lookup={r['frame']:r for r in occupancy}
    rgb=np.load(Path(spec['config']['cache_root'])/VIDEO/'rgb.npy',mmap_mode='r');preview={}
    for i in range(104,128):
        q,valid=native_dense(row,i);p=q[valid];finite=np.isfinite(p).all(1);p=p[finite];normalized=p/scale
        item={'frame':i,'pts_s':row['pts'][i]['pts_s'],'valid_pixels':int(valid.sum()),'finite_valid_points':len(p),'nonfinite_valid_points':int((~finite).sum()),'valid_fraction':float(valid.mean()),'raw_min':p.min(0).tolist(),'raw_max':p.max(0).tolist(),'normalized_min':normalized.min(0).tolist(),'normalized_max':normalized.max(0).tolist()}
        sample=p[rng.choice(len(p),min(8192,len(p)),replace=False)];samples.append(sample)
        item['sample_quantiles_raw']=np.percentile(sample,[0,1,5,25,50,75,95,99,100],axis=0).tolist();item['sample_quantiles_normalized']=np.percentile(sample/scale,[0,1,5,25,50,75,95,99,100],axis=0).tolist()
        if i not in lookup:
            # Only if worker could not finish loading: derive missing frame once.
            coords,count=np.unique(np.floor(normalized/.125).astype(np.int64),axis=0,return_counts=True);parent,cc=np.unique(coords//2,axis=0,return_inverse=True);coarse=np.bincount(cc,weights=count).astype(np.int64)
            lookup[i]={'frame':i,'fine':member_stats(count),'coarse':member_stats(coarse)}
            if i in [120,124,127]:lookup[i]['sampled_fine_centers']=(coords[::max(1,len(coords)//10000)]*.125+.0625).tolist()
        item['fine']=lookup[i]['fine'];item['coarse']=lookup[i]['coarse'];stats.append(item)
        if i in [120,124,127]:preview[i]=(np.array(rgb[i]),q[...,2]/scale,valid)
    combined=np.concatenate(samples);normalized_sample=combined/scale
    total={'finite_valid_points':sum(s['finite_valid_points'] for s in stats),'valid_pixels':sum(s['valid_pixels'] for s in stats),'nonfinite_valid_points':sum(s['nonfinite_valid_points'] for s in stats),'raw_min':np.min([s['raw_min'] for s in stats],axis=0).tolist(),'raw_max':np.max([s['raw_max'] for s in stats],axis=0).tolist(),'normalized_min':np.min([s['normalized_min'] for s in stats],axis=0).tolist(),'normalized_max':np.max([s['normalized_max'] for s in stats],axis=0).tolist(),'quantile_percentiles':[0,1,5,25,50,75,95,99,100],'sample_size':len(combined),'sample_quantiles_raw':np.percentile(combined,[0,1,5,25,50,75,95,99,100],axis=0).tolist(),'sample_quantiles_normalized':np.percentile(normalized_sample,[0,1,5,25,50,75,95,99,100],axis=0).tolist(),'fine':summarize_hist(list(lookup.values()),'fine'),'coarse':summarize_hist(list(lookup.values()),'coarse')}
    assert (total['fine']['nodes'],total['coarse']['nodes'])==EXPECTED
    for i,(image,depth,valid) in preview.items():
        fig,axes=plt.subplots(1,4,figsize=(17,3.5));axes[0].imshow(image);axes[0].set_title(f'Original frame {i}')
        axes[1].imshow(np.where(valid,np.log10(np.clip(depth,.01,1e4)),np.nan),vmin=-2,vmax=4,cmap='viridis');axes[1].set_title('log10 normalized Z; display .01..1e4')
        centers=np.array(lookup[i]['sampled_fine_centers']);limits=np.percentile(normalized_sample[:,:2],[.5,99.5],axis=0)
        axes[2].scatter(centers[:,0],centers[:,1],s=.2);axes[2].set_xlim(*limits[:,0]);axes[2].set_ylim(*limits[:,1]);axes[2].set_title('Sampled fine voxel XY centers');axes[2].set_xlabel('X');axes[2].set_ylabel('Y')
        hist=lookup[i]['fine']['histogram'];axes[3].loglog([int(k) for k in hist],[hist[k] for k in hist],'.');axes[3].set_xlabel('Points per fine voxel');axes[3].set_ylabel('Voxels');fig.tight_layout();fig.savefig(out/f'frame_{i}.jpg',dpi=120);plt.close(fig)
    (out/'index.html').write_text('<meta charset="utf-8"><title>PVST fixed coordinate preview</title><p>固定帧120/124/127。原图 | 归一化Z对数显示(.01..1e4，仅显示截断，模型不截断) | 抽样fine中心XY投影(统一抽样分位显示范围，Z维折叠) | 每块点数。无人体分割，不将近处等同人体。</p>'+''.join(f'<h3>{i}</h3><img style="max-width:100%" src="frame_{i}.jpg">' for i in preview))
    result={'frames':stats,'total':total,'quantiles':'approximate: fixed seed20260909, up to8192 uniform valid finite points/frame; pooled frame-stratified samples, not exact global weighted quantiles','range_and_counts':'exact, per-frame streaming; no point filtering beyond existing validity','global_scale':scale};save(out/'coordinates.json',result);return result


def identity(sparse=False):
    config=read(ROOT/'docs/pvst/config_v03.json');source=Path(config['asset_root']);row=next(r for r in read(source/'manifest.json')['rows'] if r['video_id']==VIDEO)
    matches=[r for r in read(source/'manifest.json')['rows'] if r['video_id']==VIDEO]
    if len(matches)!=1 or (row['dataset'],row['split'],row['role'],row['pair_id'],row['source_hw'])!=('ANAKIN','train','real','an:4wdo37',[720,1280]):raise ValueError('WINDOW_IDENTITY')
    if len(row['pts'])!=row['actual_frames'] or not np.all(np.diff([p['pts_s'] for p in row['pts'][104:128]])>0):raise ValueError('PTS_IDENTITY')
    scale=read(source/'scale.json')['s'];local={**config,'voxel_sizes':{**config['voxel_sizes'],'XYZ_PVST':.125},'offload_saved_tensors':True}
    if sparse:
        local['sparse_edge_attention']=True
        expected={'seed':20260909,'target_frames':8,'time_halo':16,'attention_chunk':512,'qkv_chunk':8192,'ffn_chunk':8192,'point_chunk':32768,'tile_size':256,'amp':True,'amp_dtype':'bfloat16'}
        if any(local.get(k)!=v for k,v in expected.items()) or config['voxel_sizes']['XYZ_PVST'] is not None:raise ValueError('LOCAL_RESOURCE_CONFIGURATION_CHANGED')
    p=Path(config['cache_root'])/VIDEO;meta=read(p/'complete.json')
    if meta['video_sha256']!=row['video_sha256'] or meta['prepared_frames']!=row['actual_frames'] or meta['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('CACHE_IDENTITY')
    for name,shape,dtype in [('rgb',(row['actual_frames'],*row['source_hw'],3),np.uint8),('tokens',(row['actual_frames'],1024,32,32),np.float16)]:
        a=np.load(p/(name+'.npy'),mmap_mode='r')
        if a.shape!=shape or a.dtype!=dtype:raise ValueError('CACHE_HEADER')
    paths=[source/'manifest.json',source/'scale.json',p/'complete.json',Path(row['dense_source'])/'complete.json',ROOT/'docs/pvst/config_v03.json']
    if sha(Path(config['checkpoint']))!=config['checkpoint_sha256']:raise ValueError('OFFICIAL_CHECKPOINT_IDENTITY')
    return {'config':local,'row':row,'global_scale':scale,'input_range':[104,128],'target_range':[120,128],'expected_nodes':EXPECTED,'deadline_s':900,'code_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},'assets_sha256':{str(p):sha(p) for p in paths},'formal_scale_unchanged':config['voxel_sizes']['XYZ_PVST'] is None}


def make_report(out,result,coords):
    total=coords['total'];stages=result.get('worker',{}).get('stages',[]);live=result.get('live',{});w=result.get('worker',{})
    lines=['# PVST v0.3 最大窗口900秒限定资源检查','',f'2026-11-04；结果 **{result["status"]}**；只检查XYZ_PVST、1/8、外层checkpoint CPU卸载；不是正式尺度冻结，正式模型仍0/2。',
        f'视频{VIDEO}，ANAKIN real / an:4wdo37；720×1280，历史[104,120)、目标[120,128)。真实空间GT为全零；v0.3逐目标帧均值frame BCE+pixel BCE+原softDice，历史不监督。',
        '局部配置：seed20260909/BF16；attention512、QKV8192、FFN8192，全部点/像素/邻接、128/256/512、24输入帧保持。只更新一次，零初始化导致部分上游梯度为0不作为失败。无重试、无其他视频/条件、无正式训练、无检测或定位成绩。',
        '',f'实际重新加载节点：fine={w.get("fine_nodes","加载未完成")}，coarse={w.get("coarse_nodes","加载未完成")}；坐标审计总计{total["fine"]["nodes"]}/{total["coarse"]["nodes"]}。',
        f'父进程墙时{result["process"]["wall_s"]:.3f}秒；真实returncode={result["process"]["returncode"]}，signal={result["process"]["termination_signal"]}。最后阶段：{live.get("phase",w.get("phase"))}。',
        '|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|主机RSS/可用GiB|','|---|---:|---:|---:|---:|']
    for s in stages:lines.append('|'+ '|'.join([s['stage'],str(s.get('stage_s')),f'{s["elapsed_s"]:.3f}',f'{s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f}',f'{s["host"]["rss_bytes"]/2**30:.3f}/{s["host"]["available_bytes"]/2**30:.3f}'])+'|')
    lines+=['','终态/最后存活采样：','```json',json.dumps({k:live.get(k) for k in ['phase','elapsed_s','peak_allocated_bytes','peak_reserved_bytes','host','trace']},ensure_ascii=False,indent=2),'```',
        '启动GPU（MiB；total/free/used/util）：'+json.dumps(result['gpu_start'],ensure_ascii=False),'关键分支梯度（未完成反向则未记录）：'+json.dumps(w.get('gradients'),ensure_ascii=False),
        '', '## 坐标分布','',f'有限有效点总数{total["finite_valid_points"]}；非有限有效点{total["nonfinite_valid_points"]}。全量精确范围：',
        '|坐标|原始min|max|归一化min|max|','|---|---:|---:|---:|---:|']
    for j,k in enumerate('XYZ'):lines.append(f'|{k}|{total["raw_min"][j]}|{total["raw_max"][j]}|{total["normalized_min"][j]}|{total["normalized_max"][j]}|')
    lines+=['','归一化仅除以既有训练全局尺度'+str(coords['global_scale'])+'，无逐帧归一化；range/count精确，坐标/深度分位数为固定seed每帧最多8192有效点抽样近似。',
        '|尺度|24帧节点合计|成员点合计|singleton比例|','|---|---:|---:|---:|']
    for k in ['fine','coarse']:lines.append(f'|{k}|{total[k]["nodes"]}|{total[k]["points"]}|{total[k]["singleton_fraction"]:.6f}|')
    lines+=['','以上为每帧占据集合的计数合计，不是把24帧合并后的空间去重集合。逐帧范围、近似分位数、fine/coarse点数分布及精确hist见coordinates.json。','', '|帧|有效点|fine/coarse|fine/coarse singleton比例|','|---|---:|---:|---:|']
    for f in coords['frames']:lines.append(f'|{f["frame"]}|{f["finite_valid_points"]}|{f["fine"]["nodes"]}/{f["coarse"]["nodes"]}|{f["fine"]["singleton_fraction"]:.4f}/{f["coarse"]["singleton_fraction"]:.4f}|')
    lines+=['','固定预览绝对入口：`'+str(out/'index.html')+'`。深度使用统一归一化Z的log10色阶，.01..1e4只在显示时截断；占据XY图为抽样中心、Z维折叠，显示范围只影响图。没有人体mask，不把近处等同人体或声称精确人体/背景统计。',
        '', '## 结论','']
    conclusions={'PASS':'该单高负载窗口完成一次前向、反向及AdamW更新；不代表正式尺度冻结或完整学习路径验收。仍不启动两条件训练。','OOM':'真实CUDA OOM，未通过当前4090容量验收；失败算子/shape见下。停止，不扩大尺度或改模型。','TIMEOUT':'900秒内未完成，父进程已终止本次进程组。不能把超时当作显存容量证明；当前单窗口执行速度不宜直接放行正式训练。','FAILED':'身份、数据或其他执行错误；不可笼统归因为资源不足。'}
    lines+= [conclusions[result['status']], '与旧版1/64不同视频窗口或首8帧的计时不直接比较加速比；本次重新加载节点与同一窗口此前计数核对。旧v0.3预算门槛结果/协议/退出码和正式配置null保持不变。', '原缓存复用，无下载/前端/冻结特征提取，无模型checkpoint保存；资源检查结果与正式模型0/2分开。', 'worker错误：','```',w.get('traceback',w.get('error','无')),'```', 'GPU样本/终止证据见gpu_samples.json、stages.json/live.json、result.json与worker.log；未完成阶段耗时不填0。']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')



def make_sparse_report(out,result,spec):
    """Report saved execution evidence only; no coordinate audit or model call."""
    w=result.get('worker',{});live=result.get('live',{});stages=w.get('stages',[])
    last=w if w.get('status') else live
    lines=['# PVST v0.3 修复后稀疏attention最大窗口资源检查','',
        '状态：**'+result['status']+'**；正式模型仍0/2，本轮无正式训练、评价、前端或单层重复测试。',
        '源码：`'+spec['source_commit']+'`。视频'+VIDEO+'；ANAKIN/train/real/an:4wdo37；720×1280；历史[104,120)、目标[120,128)。',
        '局部XYZ h=1/8、coarse=1/4；seed20260909；外层BF16、逐head FP32 CSR sampled_addmm → CSRSoftmax → CSRFromValues → CSR mm；保留逻辑填充列修复。',
        'attention512 / QKV8192 / FFN8192 / point32768 / tile256；原外层checkpoint CPU卸载。完整点、像素、邻居及128/256/512宽度不变；正式XYZ尺度仍null。',
        '真实损失为8目标帧均值frame BCE + pixel BCE + soft Dice loss；ANAKIN real使用既有全零空间GT。损失仅作执行检查。',
        '',f'实际fine/coarse：{w.get("fine_nodes","未完成加载")}/{w.get("coarse_nodes","未完成加载")}；预期2,970,629/1,488,042。',
        f'父进程墙时：{result["process"]["wall_s"]:.3f}秒；returncode={result["process"]["returncode"]}；signal={result["process"]["termination_signal"]}。',
        '最后阶段：'+str(last.get('phase'))+'；实际更新次数：'+str(w.get('updates',0))+'；参数改变核对：'+str(w.get('parameter_update_verified',False))+'。',
        '', '|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|当前 allocated/reserved GiB|RSS/主机可用 GiB|',
        '|---|---:|---:|---:|---:|---:|']
    for t in stages:
        lines.append(f'|{t["stage"]}|{t.get("stage_s")}|{t["elapsed_s"]:.3f}|{t["peak_allocated_bytes"]/2**30:.3f}/{t["peak_reserved_bytes"]/2**30:.3f}|{t["allocated_bytes"]/2**30:.3f}/{t["reserved_bytes"]/2**30:.3f}|{t["host"]["rss_bytes"]/2**30:.3f}/{t["host"]["available_bytes"]/2**30:.3f}|')
    lines+=['','六层构建核对：'+str(len(w.get('candidate_build_evidence',[])))+'/6；完成前向的运行证据如下。未完成层不冒称通过。',
        '|层|dilation|节点|有效边E|平均度|拓扑MiB|完成head|逻辑填充列|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in w.get('forward_edge_evidence',[]):
        lines.append(f'|{t["layer"]}|{t["dilation"]}|{t["nodes"]}|{t["edges"]}|{t["average_degree"]:.3f}|{t["topology_bytes"]/2**20:.3f}|{t["completed_heads"]}|{t["logical_padding_columns"]}|')
    evidence={k:w.get(k) for k in ['loss_components','gradients','parameter_delta_norms','optimizer_state_parameters']}
    lines+=['','梯度、损失及更新证据：','```json',json.dumps(evidence,ensure_ascii=False,indent=2),'```',
        '末次状态与失败位置（已观测峰值不是未完成更新的峰值上界）：','```json',json.dumps({k:last.get(k) for k in ['phase','elapsed_s','peak_allocated_bytes','peak_reserved_bytes','host','trace']},ensure_ascii=False,indent=2),'```',
        '错误与调用栈：','```',w.get('traceback',w.get('error','无；若被父进程终止，以退出信号及最后阶段为准')),'```',
        '启动/结束GPU：','```json',json.dumps({'start':result['gpu_start'],'after':result['gpu_after'],'worker_gpu_released':result['worker_gpu_released']},ensure_ascii=False,indent=2),'```',
        '', '同窗口旧query SDPA：前向61.7257秒；反向至少827.776秒仍未完成；900秒TIMEOUT，AdamW未执行。当前逐head FP32核心的精度路径不同，数学目标保持，不宣称逐位数值相同，也不以旧下界计算精确完整加速比。']
    conclusion={'PASS':'本次最大窗口完成一次真实更新；仅证明此窗口与本次共享显存环境可执行，不等于全队列验收或尺度冻结。根据峰值余量和更新成本再决定是否授权两条件各一轮，本轮不启动。',
        'OOM':'本次真实CUDA OOM，停止；不建议直接进入两条件各一轮。先根据明确失败算子与内存证据决定下一步，本轮不继续修复。',
        'TIMEOUT':'900秒内未完成，停止；这是执行时间阻塞，不能作为容量不足证明。不建议直接进入两条件各一轮。',
        'FAILED':'其他执行错误，停止；不建议进入两条件各一轮，先解决报告中的具体错误。'}
    lines+=['',conclusion[result['status']], '旧日志/退出码/缓存及正式配置保持原样；新模型和优化器丢弃，无checkpoint保存。计算状态与publication.json中的Git发布状态分开。',
        '独立产物目录：`'+str(out)+'`；resource_curve.png为约5秒设备采样；result.json、stages.json、live.json保留完整证据。']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=[]
    for v in result['process']['samples']:
        try:
            values=[float(x) for x in v['gpu_total_free_used_util'].splitlines()[0].split(',')]
            points.append((v['unix_time'],values[2]/1024,values[3]))
        except (ValueError,IndexError):pass
    fig,ax=plt.subplots(figsize=(8,3));right=ax.twinx()
    if points:
        a=np.array(points);x=a[:,0]-a[0,0];ax.plot(x,a[:,1],label='device used GiB');right.plot(x,a[:,2],color='orange',label='GPU utilization %')
    ax.set(xlabel='seconds from first sample',ylabel='device used GiB');right.set(ylabel='GPU utilization %',ylim=(0,105));fig.tight_layout();fig.savefig(out/'resource_curve.png',dpi=120);plt.close(fig)


def publish(out,result,sparse=False):
    files=[]
    prefix='sparse_max_window_check_v1' if sparse else 'max_window_check_v1'
    for name,source in [(prefix+'.md','report_zh.md'),(prefix+'.json','result_small.json')]:
        p=ROOT/'docs/pvst'/name;p.write_bytes((out/source).read_bytes());files.append(str(p.relative_to(ROOT)))
    p=ROOT/'docs/pvst/current_status.md';s=p.read_text();s+='\n2026-11-04'+('修复后稀疏' if sparse else '')+'单窗口独立资源检查：'+result['status']+'；正式模型仍0/2，旧预算结论不变，见[900秒报告]('+prefix+'.md)。\n';p.write_text(s);files.append(str(p.relative_to(ROOT)))
    subprocess.run(['git','add','--',*files],check=True);subprocess.run(['git','commit','-m','Record bounded PVST v03 maximum window check'],check=True)
    head=git('rev-parse','HEAD');save(out/'publication.json',{'status':'COMMITTED','commit':head});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
    remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
    if head!=remote:raise ValueError('REMOTE_HEAD_MISMATCH')
    save(out/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote})


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--sparse',action='store_true');p.add_argument('--values-bridge',action='store_true');p.add_argument('--head-output-offload',action='store_true');a=p.parse_args()
    if a.head_output_offload:a.values_bridge=True
    if a.values_bridge:a.sparse=True
    if a.sparse:OUT=SPARSE_OUT
    if a.values_bridge:OUT=BRIDGE_OUT
    if a.head_output_offload:OUT=HEAD_OUT
    if a.worker:raise SystemExit(worker(OUT))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);spec=identity(sparse=a.sparse)
        if a.head_output_offload:spec['config']['head_output_offload']=True
        if (OUT/'identity.json').exists():
            previous=read(OUT/'identity.json');previous.pop('source_commit',None)
            if previous!=json.loads(json.dumps(spec)):raise ValueError('CHECK_IDENTITY_CHANGED')
            if (OUT/'result.json').exists():print('REUSE',read(OUT/'result.json')['status']);return
            raise RuntimeError('PRIOR_LAUNCH_WITHOUT_RESULT: inspect; never auto retry')
        dirty=git('status','--porcelain','--','research_tools/pvst','tests/test_pvst_max_window.py','docs/pvst/config_v03.json')
        if dirty:raise ValueError('RESOURCE_SOURCE_NOT_COMMITTED')
        spec['source_commit']=git('rev-parse','HEAD');save(OUT/'identity.json',spec)
        save(OUT/'progress.json',{'status':'RUNNING','stage':'bounded_worker','pid':os.getpid()})
        startgpu=gpu_sample(None);command=[sys.executable,'-u','-m','research_tools.pvst.max_window_check','--worker']+(['--head-output-offload'] if a.head_output_offload else ['--values-bridge'] if a.values_bridge else ['--sparse'] if a.sparse else [])
        proc=bounded_process(command,900,OUT/'worker.log',gpu_sample)
        w=read(OUT/'worker_result.json') if (OUT/'worker_result.json').exists() else {};live=read(OUT/'live.json') if (OUT/'live.json').exists() else {}
        if not w and (OUT/'stages.json').exists():w={'stages':read(OUT/'stages.json')}
        status='TIMEOUT' if proc['timed_out'] else w.get('status','FAILED')
        if status=='PASS' and proc['returncode']!=0:status='FAILED'
        result={'status':status,'worker':w,'live':w if w.get('status') else live,'process':proc,'gpu_start':startgpu,'gpu_after':gpu_sample(proc['pid']),'source_commit':spec['source_commit'],'formal_models_completed':0}
        result['worker_gpu_released']=str(proc['pid']) not in [line.split(',')[0].strip() for line in result['gpu_after']['compute_pid_mib'].splitlines()]
        if status=='PASS' and not result['worker_gpu_released']:status='FAILED';result['status']=status
        save(OUT/'result.json',result);(OUT/'exit_code.txt').write_text(str(proc['returncode'])+'\n')
        coords=None
        if a.sparse:make_sparse_report(OUT,result,spec)
        else:coords=coordinate_audit(OUT,spec);make_report(OUT,result,coords)
        save(OUT/'result_small.json',{'status':status,'source_commit':spec['source_commit'],'video_id':VIDEO,'config':spec['config'],'process':{k:v for k,v in proc.items() if k!='samples'},'gpu_start':startgpu,'gpu_after':result['gpu_after'],'worker':{k:v for k,v in w.items() if k not in ['traceback']},'last_live':result['live'],'coordinate_total':coords['total'] if coords else None,'formal_models_completed':0})
        save(OUT/'progress.json',{'status':status,'stage':'report_complete','formal_models_completed':0})
        try:
            if not a.values_bridge:publish(OUT,result,sparse=a.sparse)
        except BaseException:save(OUT/'publication.json',{'status':'FAILED','calculation_status':status,'error':traceback.format_exc()})

if __name__=='__main__':main()
