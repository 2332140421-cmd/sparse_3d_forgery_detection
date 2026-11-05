"""One fixed 4+4 resource probe; no formal training or scale selection.

Caller: this CLI / existing resource worker. Output: stages and bounded report.
Reuses Video, build, clip_terms, optimizer groups and bounded_process. Separate
entry is necessary to keep this local 4+4 configuration out of formal protocols.
"""
import argparse
import hashlib
import sys
import time
import traceback
from pathlib import Path
from .max_window_check import (BASE,ROOT,read,save,sha,git,identity,bounded_process,
                               gpu_sample,host,candidate_layers)
OUT=BASE/'pvst_target_microbatch_check_v1'


def batches():
    return [{'name':'A','input_range':[104,124],'history_range':[104,120],'target_range':[120,124]},
            {'name':'B','input_range':[108,128],'history_range':[108,124],'target_range':[124,128]}]


def microbatch_worker(out):
    import torch
    import numpy as np
    from .train import build
    from .data import Video
    from .probe import parameters,model_trace
    from .v03 import clip_terms
    torch.set_num_threads(4);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    spec=read(out/'identity.json');c=spec['config'];row=spec['row'];m=opt=v=None
    phase='start';u=0;mb=None;stages=[]
    result={'status':'FAILED','updates':0,'optimizer_step_calls':0,'update_records':[],'microbatches':[]}
    def snapshot():
        return {'elapsed_s':time.monotonic()-started,'allocated_bytes':torch.cuda.memory_allocated(),
                'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'host':host()}
    def stage(label,tick=None,**kw):
        item={**snapshot(),'stage':label,'update':u,'microbatch':mb,'stage_s':None if tick is None else time.monotonic()-tick,**kw}
        stages.append(item);save(out/'stages.json',stages);save(out/'live.json',item)
        print('STAGE',label,'update',u,mb,item['elapsed_s'],flush=True);return item
    try:
        # Fail before construction if source/config/assets drifted since launch identity.
        for name,h in spec['code_sha256'].items():
            if sha(Path(__file__).parent/name)!=h:raise ValueError('EXECUTION_SOURCE_CHANGED:'+name)
        for name,h in spec['assets_sha256'].items():
            if sha(Path(name))!=h:raise ValueError('ASSET_IDENTITY_CHANGED:'+name)
        plan=batches();targets=[i for x in plan for i in range(*x['target_range'])]
        if targets!=list(range(120,128)):raise ValueError('TARGET_COVERAGE')
        stage('worker_start');phase='model_build';tick=time.monotonic();m=build(c,'XYZ_PVST').train()
        opt=torch.optim.AdamW(parameters(c,m),weight_decay=c['weight_decay'])
        if sum(p.numel() for p in m.parameters())!=17821672:raise ValueError('PARAMETER_COUNT_CHANGED')
        if hashlib.sha256('\n'.join(m.state_dict()).encode()).hexdigest()!='31beab46b45b34ed53ff89a70293a0e9e93abb86721dc51902e9c4c1808af665':raise ValueError('STATE_DICT_KEYS_CHANGED')
        candidate_layers(m,require_offload=True,require_stream=True,require_split=True);m.check_bn()
        torch.cuda.synchronize();stage('model_built',tick,parameters=17821672)
        # Same-window original counts are read-only historical identity, not rerun24 input.
        previous=read(BASE/'pvst_tensor_lifetime_check_v1/max_window/worker_result.json')
        prior_spec=read(BASE/'pvst_tensor_lifetime_check_v1/max_window/identity.json')
        if (prior_spec['row']['video_sha256']!=row['video_sha256'] or prior_spec['global_scale']!=spec['global_scale'] or prior_spec['input_range']!=[104,128] or previous['fine_nodes']!=2970629 or previous['coarse_nodes']!=1488042):raise ValueError('ORIGINAL_WINDOW_IDENTITY')
        result['original_window_nodes']=[2970629,1488042]
        for u in [1,2]:
            update_start=time.monotonic();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();weighted=np.zeros(4)
            for part in plan:
                mb=part['name'];lo,hi=part['input_range'];start,end=part['target_range'];phase='load';tick=time.monotonic()
                stage('before_microbatch_load')
                v=Video(Path(c['asset_root']),row,'XYZ_PVST',spec['global_scale'],c);fs=v.frames(lo,hi,'cuda')
                if [f['pts'] for f in fs]!=[p['pts_s'] for p in row['pts'][lo:hi]]:raise ValueError('PTS_MISMATCH')
                fine=sum(len(f['voxel_meta']['coords']) for f in fs);coarse=sum(len(f['voxel_meta']['coarse_coords']) for f in fs)
                gt=[v.gt(i) for i in range(start,end)];labels=[v.label(i) for i in range(start,end)]
                if any(g is None or bool(g.any()) for g in gt) or any(labels):raise ValueError('EXPECTED_LEGAL_REAL_ZERO_GT')
                torch.cuda.synchronize();loaded=stage('data_loaded',tick,**part,fine_nodes=fine,coarse_nodes=coarse,native_hw=row['source_hw'])
                phase='forward';tick=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=m(fs,start-lo);terms=clip_terms(pred,gt,labels,c['dice_epsilon']);loss=terms[0]*((end-start)/8)
                if len(pred)!=4 or any(list(p['pixel_logits'].shape)!=row['source_hw'] for p in pred):raise ValueError('NATIVE_COVERAGE')
                if not bool(torch.isfinite(terms).all()):raise ValueError('NONFINITE_LOSS')
                values=terms.detach().float().tolist();weighted+=np.array(values)*.5
                edges=candidate_layers(m,True,require_offload=True,require_stream=True,require_split=True,expected_nodes=(fine,coarse))
                torch.cuda.synchronize();forward=stage('forward_complete',tick,loss_components=values,weight=.5,edge_evidence=edges)
                phase='backward';tick=time.monotonic();loss.backward();torch.cuda.synchronize();backward=stage('backward_complete',tick)
                # No learned state passed across groups. Clear reader's GPU cache too.
                del pred,terms,loss,fs,gt,labels;v.close();v=None
                torch.cuda.synchronize();released=stage('graph_and_input_released')
                result['microbatches'].append({'update':u,**part,'weight':.5,'fine_nodes':fine,'coarse_nodes':coarse,
                  'loss_components':values,'load_s':loaded['stage_s'],'forward_s':forward['stage_s'],'backward_s':backward['stage_s'],
                  'after_release':released,'edge_evidence':edges})
                save(out/'partial.json',result)
            mb=None;phase='gradient_check_clip';tick=time.monotonic()
            paths={'geometry':m.geo[0].weight,'VFE':m.vfe[0].weight,'fine_qkv':m.fine_time.layers[0].qkv.weight,'coarse_qkv':m.coarse_time.layers[0].qkv.weight,'decoder':m.decoder.proj.weight,'pixel_output':m.pixel_out.weight,'frame_output':m.frame_residual[-1].weight}
            grads={k:{'exists':p.grad is not None,'finite':bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,'norm':float(p.grad.float().norm()) if p.grad is not None else None} for k,p in paths.items()}
            if any(not x['exists'] or not x['finite'] for x in grads.values()):raise ValueError('KEY_GRADIENT_MISSING_NONFINITE')
            norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),c['clip_norm'],error_if_nonfinite=True));torch.cuda.synchronize();clipped=stage('gradient_clip_complete',tick,gradients=grads,gradient_norm=norm)
            selected={'pixel_output':m.pixel_out.weight,'frame_visual':m.frame_visual[-1].weight,'decoder_predictor':m.decoder.predictor.weight}
            before={k:p.detach().cpu().clone() for k,p in selected.items()};phase='AdamW';tick=time.monotonic();opt.step();result['optimizer_step_calls']+=1
            torch.cuda.synchronize();step=stage('AdamW_complete',tick,optimizer_state_parameters=len(opt.state))
            changes={k:float((p.detach().cpu()-before[k]).norm()) for k,p in selected.items()}
            if not any(x>0 for x in changes.values()) or not opt.state:raise ValueError('NO_VERIFIED_UPDATE')
            result['updates']=u;m.check_bn()
            record={'update':u,'weighted_loss_components':weighted.tolist(),'gradient_norm':norm,'gradients':grads,'clip_s':clipped['stage_s'],'AdamW_s':step['stage_s'],
                    'parameter_delta_norms':changes,'optimizer_state_parameters':len(opt.state),'update_wall_s':time.monotonic()-update_start,**snapshot()}
            result['update_records'].append(record);stage('update_complete',update_wall_s=record['update_wall_s']);save(out/'partial.json',result)
        result['status']='PASS'
    except BaseException as e:
        result.update(status='OOM' if isinstance(e,torch.OutOfMemoryError) else 'FAILED',error=str(e),traceback=traceback.format_exc(),failure_phase=phase,failure_microbatch=mb,failure_update=u,trace=model_trace(m) if m is not None else {})
    finally:
        result.update(snapshot(),stages=stages,optimizer_state_parameters=len(opt.state) if opt else 0,weights='discarded; no checkpoint')
        save(out/'worker_result.json',result)
        if v:v.close()
    return 0 if result['status']=='PASS' else 2 if result['status']=='OOM' else 1


def report(r):
    w=r.get('worker',{});lines=['# PVST v0.3 目标帧4＋4微批资源验收','',
        '**状态：'+r['status']+'；正式模型0/2，不启动训练。**','',
        '只改变目标帧执行：A历史[104,120)、目标[120,124)；B历史[108,124)、目标[124,128)。每组20帧，逻辑更新40帧次、24唯一输入帧；历史重算，不复用可学习特征、不提前step。',
        '三层因果最大回溯2+4+8=14帧，加帧级visual的2帧，完整16帧。context读写仅同帧，BN固定、LayerNorm逐位置，XYZ为冻结训练全局scale；真实PTS与绝对q保持。没有窗口统计或未来帧依赖。',
        'losses逐帧计算：frame BCE标量、pixel BCE原生全像素均值、softDice每帧全像素sum/epsilon1；缺GT仅空间项0，不改变frame分母。clip_terms对目标帧取均值，本AN real8帧均有合法零GT，各分项每微批乘4/8。未来AF/AN外层配对权重仍须保留。',
        '诊断关闭所有Dropout（含MHA与attention内部），两份诊断模型同非零输出头验证上游梯度；资源验收恢复原Dropout与规定初始化。分组改变随机轨迹，不称逐步完全相同。','',
        '执行源码：`'+r['source_commit']+'`；冻结配置XYZ仍null；1/8只作局部资源配置。新增前端调用0。','',
        '## 小规模语义核验','```json',__import__('json').dumps(r.get('semantics',{}),ensure_ascii=False,indent=2),'```','',
        '## 实际微批阶段','|更新/微批/阶段|耗时秒|allocated/reserved GiB|已观测峰值A/R GiB|RSS GiB|','|---|---:|---:|---:|---:|']
    for x in w.get('stages',[]):
        lines.append(f"|{x['update']}/{x['microbatch']}/{x['stage']}|{x['stage_s']}|{x['allocated_bytes']/2**30:.3f}/{x['reserved_bytes']/2**30:.3f}|{x['peak_allocated_bytes']/2**30:.3f}/{x['peak_reserved_bytes']/2**30:.3f}|{x['host']['rss_bytes']/2**30:.3f}|")
    lines+=['','## 结果与边界',f"成功更新{w.get('updates',0)}次，step成功调用{w.get('optimizer_step_calls',0)}次；原24帧联合检查不重跑。",'```json',__import__('json').dumps({'update_records':w.get('update_records',[]),'microbatches':[{k:v for k,v in x.items() if k not in ['edge_evidence','after_release']} for x in w.get('microbatches',[])],'error':w.get('error'), 'failure_phase':w.get('failure_phase'),'failure_microbatch':w.get('failure_microbatch'),'process':r.get('process'),'host':w.get('host')},ensure_ascii=False,indent=2),'```',
        '本轮只验收规定XYZ窗口，不追加UV、非空GT或其他样本。'+('两次更新通过，只代表本窗口可执行；不自动启动正式训练。' if w.get('updates')==2 else '尚未完成两次更新；下一步仅依据本轮实际失败阶段/算子和资源证据决定，不自动缩小微批或改算子。'),
        '旧24帧失败峰值为未完成更新的观测值，不能拿旧失败耗时当作完整更新时间或计算总加速比。全部原节点/边、宽度、尺度、输入像素、损失及缓存保持。',
        '产物：`'+str(OUT)+'`。计算状态与publication分开。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/target_microbatch_check_v1.md').write_text(text)
    summary={k:v for k,v in r.items() if k!='worker'};summary['worker']={k:v for k,v in w.items() if k not in ['stages','traceback','trace']}
    save(OUT/'summary.json',summary);save(ROOT/'docs/pvst/target_microbatch_check_v1.json',summary)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+f"\n目标帧4＋4微批资源检查：{r['status']}，成功更新{w.get('updates',0)}次；正式模型0/2，未训练。见[报告](target_microbatch_check_v1.md)。\n")


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['semantics','max_window']);args=p.parse_args()
    if args.worker=='semantics':
        import torch
        from .csr_values_bridge_check import load_test
        sys.path.insert(0,str(ROOT/'tests'))
        test=load_test('test_pvst_target_microbatch.py')
        result={'status':'PASS','comparisons':[test.compare_microbatch('cuda',dtype) for dtype in [torch.float32,torch.bfloat16]]}
        save(OUT/'semantics.json',result);return
    if args.worker=='max_window':
        from .max_window_check import worker
        raise SystemExit(worker(OUT/'max_window'))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse; no retry')
    r={'status':'FAILED','source_commit':git('rev-parse','HEAD'),'formal_models_completed':0,'gpu_start':gpu_sample(None)}
    spec=identity(sparse=True);spec['config'].update(head_output_offload=True,head_qkv_stream=True,node_split=True);spec['target_microbatches']=batches();spec['updates_requested']=2;spec['source_commit']=r['source_commit']
    save(OUT/'identity.json',spec);folder=OUT/'semantics';folder.mkdir()
    try:
        process=bounded_process([sys.executable,'-u','-m',__spec__.name,'--worker','semantics'],300,folder/'worker.log',gpu_sample);save(folder/'process.json',process)
        if process['returncode']!=0 or process['timed_out']:raise RuntimeError('SEMANTIC_CHECK_FAILED; no maximum-window run')
        r['semantics']=read(OUT/'semantics.json');folder=OUT/'max_window';folder.mkdir();save(folder/'identity.json',spec)
        save(OUT/'progress.json',{'status':'RUNNING','stage':'max_window','source_commit':r['source_commit']})
        process=bounded_process([sys.executable,'-u','-m',__spec__.name,'--worker','max_window'],900,folder/'worker.log',gpu_sample);save(folder/'process.json',process);r['process']=process
        r['worker']=read(folder/'worker_result.json') if (folder/'worker_result.json').exists() else {'stages':read(folder/'stages.json') if (folder/'stages.json').exists() else [],**(read(folder/'partial.json') if (folder/'partial.json').exists() else {})}
        r['status']='TIMEOUT' if process['timed_out'] else r['worker'].get('status','FAILED')
        if r['status']=='PASS' and (process['returncode']!=0 or r['worker'].get('updates')!=2):r['status']='FAILED'
    except Exception as e:r.update(reason=str(e),traceback=traceback.format_exc())
    r['gpu_end']=gpu_sample(None);report(r);save(OUT/'progress.json',{'status':r['status'],'stage':'report_complete','updates':r.get('worker',{}).get('updates',0),'formal_models_completed':0})
    import subprocess
    try:
        subprocess.run(['git','add','--','docs/pvst/target_microbatch_check_v1.md','docs/pvst/target_microbatch_check_v1.json','docs/pvst/current_status.md'],check=True)
        subprocess.run(['git','commit','-m','Record fixed target microbatch resource acceptance'],check=True);head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':r['status']})
        subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True);remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if head!=remote:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':r['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':r['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
