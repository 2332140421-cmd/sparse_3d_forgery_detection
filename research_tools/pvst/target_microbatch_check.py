"""Fixed 4+4 or 2+2+2+2 resource probe; no formal training or scale selection.

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


def batches(group_size=4):
    if group_size not in [2,4]:raise ValueError('ONLY_FIXED_GROUP_SIZE_2_OR_4')
    return [{'name':('AB'[j] if group_size==4 else str(j+1)),
             'input_range':[start-16,start+group_size],'history_range':[start-16,start],
             'target_range':[start,start+group_size]}
            for j,start in enumerate(range(120,128,group_size))]


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
        plan=batches(spec.get('group_size',4))
        if spec['target_microbatches']!=plan:raise ValueError('MICROBATCH_PLAN_CHANGED')
        targets=[i for x in plan for i in range(*x['target_range'])]
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
        for u in range(1,spec.get('updates_requested',2)+1):
            update_start=time.monotonic();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();weighted=np.zeros(4)
            for part in plan:
                mb=part['name'];lo,hi=part['input_range'];start,end=part['target_range'];phase='load';tick=time.monotonic();weight=(end-start)/8
                stage('before_microbatch_load')
                v=Video(Path(c['asset_root']),row,'XYZ_PVST',spec['global_scale'],c);fs=v.frames(lo,hi,'cuda')
                if [f['pts'] for f in fs]!=[p['pts_s'] for p in row['pts'][lo:hi]]:raise ValueError('PTS_MISMATCH')
                fine=sum(len(f['voxel_meta']['coords']) for f in fs);coarse=sum(len(f['voxel_meta']['coarse_coords']) for f in fs)
                gt=[v.gt(i) for i in range(start,end)];labels=[v.label(i) for i in range(start,end)]
                if any(g is None or bool(g.any()) for g in gt) or any(labels):raise ValueError('EXPECTED_LEGAL_REAL_ZERO_GT')
                torch.cuda.synchronize();loaded=stage('data_loaded',tick,**part,fine_nodes=fine,coarse_nodes=coarse,native_hw=row['source_hw'],target_labels=labels,target_pts=[row['pts'][i]['pts_s'] for i in range(start,end)])
                phase='forward';tick=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=m(fs,start-lo);terms=clip_terms(pred,gt,labels,c['dice_epsilon']);loss=terms[0]*weight
                if len(pred)!=end-start or any(list(p['pixel_logits'].shape)!=row['source_hw'] for p in pred):raise ValueError('NATIVE_COVERAGE')
                if not bool(torch.isfinite(terms).all()):raise ValueError('NONFINITE_LOSS')
                values=terms.detach().float().tolist();weighted+=np.array(values)*weight
                edges=candidate_layers(m,True,require_offload=True,require_stream=True,require_split=True,expected_nodes=(fine,coarse))
                torch.cuda.synchronize();forward=stage('forward_complete',tick,loss_components=values,weight=weight,edge_evidence=edges)
                phase='backward';tick=time.monotonic();loss.backward();torch.cuda.synchronize();backward=stage('backward_complete',tick)
                # No learned state passed across groups. Clear reader's GPU cache too.
                del pred,terms,loss,fs,gt,labels;v.close();v=None
                torch.cuda.synchronize();released=stage('graph_and_input_released')
                result['microbatches'].append({'update':u,**part,'weight':weight,'fine_nodes':fine,'coarse_nodes':coarse,
                  'loss_components':values,'load_s':loaded['stage_s'],'forward_s':forward['stage_s'],'backward_s':backward['stage_s'],
                  'after_release':released,'edge_evidence':edges})
                save(out/'partial.json',result)
            if [x['name'] for x in result['microbatches'] if x['update']==u]!=[x['name'] for x in plan]:raise ValueError('INCOMPLETE_GROUPS_NO_UPDATE')
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
        contexts=[];tb=e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name=='sparse_head':
                loc=tb.tb_frame.f_locals
                contexts.append({'head':loc.get('head'),'q_shape':list(loc['q'].shape),'q_dtype':str(loc['q'].dtype)})
            tb=tb.tb_next
        del tb
        result['failure_operator_context']=contexts
        result.update(status='OOM' if isinstance(e,torch.OutOfMemoryError) else 'FAILED',error=str(e),traceback=traceback.format_exc(),failure_phase=phase,failure_microbatch=mb,failure_update=u,trace=model_trace(m) if m is not None else {})
    finally:
        result.update(snapshot(),stages=stages,optimizer_state_parameters=len(opt.state) if opt else 0,weights='discarded; no checkpoint')
        save(out/'worker_result.json',result)
        if v:v.close()
    return 0 if result['status']=='PASS' else 2 if result['status']=='OOM' else 1


def report(r):
    import json
    w=r.get('worker',{});size=r['group_size'];slug='target_microbatch_2_check_v1' if size==2 else 'target_microbatch_check_v1'
    groups=batches(size);edges=[]
    for x in w.get('stages',[]):
        for e in x.get('edge_evidence',[]):edges.append({'group':x['microbatch'],**{k:e[k] for k in ['layer','nodes','edges','topology_bytes']}})
    compact={k:v for k,v in w.items() if k not in ['stages','trace','traceback','error','microbatches']}
    compact['error']=w.get('error','').splitlines()[:1];compact['edge_counts']=edges
    compact['groups']=[{**part,'stages':[{k:v for k,v in x.items() if k not in ['edge_evidence']} for x in w.get('stages',[]) if x['microbatch']==part['name']]} for part in groups]
    summary={k:v for k,v in r.items() if k!='worker'};summary['worker']=compact
    if 'process' in summary:summary['process']={k:v for k,v in summary['process'].items() if k!='samples'}
    lines=['# PVST v0.3 目标帧'+('2＋2＋2＋2' if size==2 else '4＋4')+'资源验收','',
        '**状态：'+r['status']+'；真实更新'+str(w.get('updates',0))+'次；正式模型0/2，不启动训练。**','',
        '源码：`'+r['source_commit']+'`；产物：`'+str(OUT)+'`。',
        '沿用原模型/初始化/AdamW/Dropout和BF16外层、逐head FP32、CSR桥接、CPU卸载、head暂存、即时QKV、node_split/view清理。attention512/QKV8192/FFN8192/point32768/tile256不变。',
        '三层因果最大回溯14帧加visual两帧=16；同帧context、固定BN、逐位置LayerNorm、冻结全局坐标scale、真实PTS保持。',
        'frame BCE逐帧标量、pixel BCE逐帧原生像素均值、softDice逐帧全像素sum/epsilon1；clip_terms按目标帧均值。此AN real全部合法零GT，分项均以原8目标帧为分母，权重按目标数/8计算；缺GT语义、loss权重不变。以后AF/AN配对外层权重仍需保留。',
        '每组反向后释放图和Video GPU缓存，只保留梯度；所有组成功前不clip、不step、不zero_grad。未完成组不会计作更新。',
        '计划每更新'+str(sum(x['input_range'][1]-x['input_range'][0] for x in groups))+'帧次、24唯一输入、8唯一目标；重复历史重新计算。失败时实际加载组数见下，不能把计划量当作完成量。',
        '小图沿用同非零诊断头、全部Dropout关闭、训练/checkpoint路径与既定容差；真实资源恢复原Dropout，随机轨迹随分组变化，不称训练逐步一致。','',
        '## 语义核验','```json',json.dumps(r.get('semantics',{}),ensure_ascii=False,indent=2),'```','',
        '## 实测阶段','|组/阶段|秒|allocated/reserved GiB|峰值A/R GiB|RSS GiB|','|---|---:|---:|---:|---:|']
    for x in w.get('stages',[]):lines.append(f"|{x['microbatch']}/{x['stage']}|{x['stage_s']}|{x['allocated_bytes']/2**30:.3f}/{x['reserved_bytes']/2**30:.3f}|{x['peak_allocated_bytes']/2**30:.3f}/{x['peak_reserved_bytes']/2**30:.3f}|{x['host']['rss_bytes']/2**30:.3f}|")
    lines+=['','## 节点/边','|组/层|N|E|拓扑字节|','|---|---:|---:|---:|']
    for e in edges:lines.append(f"|{e['group']}/{e['layer']}|{e['nodes']}|{e['edges']}|{e['topology_bytes']}|")
    lines+=['','结果：成功step='+str(w.get('optimizer_step_calls',0))+'；优化器状态参数='+str(w.get('optimizer_state_parameters',0))+'。',
        '失败：'+str(compact['error'])+'；阶段='+str(w.get('failure_phase'))+'；组='+str(w.get('failure_microbatch')),
        '进程：'+str(summary.get('process')),
        '更新记录：'+json.dumps(w.get('update_records',[]),ensure_ascii=False),
        '主机内存：'+str(w.get('host')),
        '完整逐组记录/资源采样/原生错误保存在max_window。未完成的反向与完整更新时间不填0，不用旧4＋4失败耗时计算加速比。',
        '仅规定窗口资源检查，XYZ正式配置仍null，不改变节点门槛/模型完成数；不重跑前端、旧24帧/4＋4或额外样本。'+('本窗口一次完整更新通过，仍不自动放行正式训练。' if r['status']=='PASS' else '本轮停止，不重试、不追加逐帧微批或算子修改。')]
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/('docs/pvst/'+slug+'.md')).write_text(text)
    save(OUT/'summary.json',summary);save(ROOT/('docs/pvst/'+slug+'.json'),summary)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+f"\n目标帧组大小{size}资源验收：{r['status']}；真实更新{w.get('updates',0)}次，正式模型0/2，未训练。见[报告]({slug}.md)。\n")


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--group-size',type=int,choices=[2,4],default=4);p.add_argument('--worker',choices=['semantics','max_window']);args=p.parse_args()
    if args.group_size==2:OUT=BASE/'pvst_target_microbatch_2_check_v1'
    if args.worker=='semantics':
        import torch
        from .csr_values_bridge_check import load_test
        sys.path.insert(0,str(ROOT/'tests'))
        test=load_test('test_pvst_target_microbatch.py')
        result={'status':'PASS','comparisons':[test.compare_microbatch('cuda',dtype,args.group_size) for dtype in [torch.float32,torch.bfloat16]]}
        save(OUT/'semantics.json',result);return
    if args.worker=='max_window':
        from .max_window_check import worker
        raise SystemExit(worker(OUT/'max_window'))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse; no retry')
    r={'status':'FAILED','source_commit':git('rev-parse','HEAD'),'group_size':args.group_size,'formal_models_completed':0,'gpu_start':gpu_sample(None)}
    spec=identity(sparse=True);spec['config'].update(head_output_offload=True,head_qkv_stream=True,node_split=True);spec['group_size']=args.group_size;spec['target_microbatches']=batches(args.group_size);spec['updates_requested']=1 if args.group_size==2 else 2;spec['source_commit']=r['source_commit']
    save(OUT/'identity.json',spec);folder=OUT/'semantics';folder.mkdir()
    try:
        process=bounded_process([sys.executable,'-u','-m',__spec__.name,'--group-size',str(args.group_size),'--worker','semantics'],300,folder/'worker.log',gpu_sample);save(folder/'process.json',process)
        if process['returncode']!=0 or process['timed_out']:raise RuntimeError('SEMANTIC_CHECK_FAILED; no maximum-window run')
        r['semantics']=read(OUT/'semantics.json');folder=OUT/'max_window';folder.mkdir();save(folder/'identity.json',spec)
        save(OUT/'progress.json',{'status':'RUNNING','stage':'max_window','source_commit':r['source_commit']})
        process=bounded_process([sys.executable,'-u','-m',__spec__.name,'--group-size',str(args.group_size),'--worker','max_window'],900,folder/'worker.log',gpu_sample);save(folder/'process.json',process);r['process']=process
        r['worker']=read(folder/'worker_result.json') if (folder/'worker_result.json').exists() else {'stages':read(folder/'stages.json') if (folder/'stages.json').exists() else [],**(read(folder/'partial.json') if (folder/'partial.json').exists() else {})}
        r['status']='TIMEOUT' if process['timed_out'] else r['worker'].get('status','FAILED')
        if r['status']=='PASS' and (process['returncode']!=0 or r['worker'].get('updates')!=spec['updates_requested']):r['status']='FAILED'
    except Exception as e:r.update(reason=str(e),traceback=traceback.format_exc())
    r['gpu_end']=gpu_sample(None);report(r);save(OUT/'progress.json',{'status':r['status'],'stage':'report_complete','updates':r.get('worker',{}).get('updates',0),'formal_models_completed':0})
    import subprocess
    slug='target_microbatch_2_check_v1' if args.group_size==2 else 'target_microbatch_check_v1'
    try:
        subprocess.run(['git','add','--','docs/pvst/'+slug+'.md','docs/pvst/'+slug+'.json','docs/pvst/current_status.md'],check=True)
        subprocess.run(['git','commit','-m','Record fixed target microbatch resource acceptance'],check=True);head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':r['status']})
        subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True);remote=git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/pvst-point-voxel-v03').split()[0]
        if head!=remote:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':r['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':r['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
