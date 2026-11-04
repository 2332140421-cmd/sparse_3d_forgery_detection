"""Bounded reference-lifetime/copy evidence and pre-training acceptance only.

Callbacks are direct optional SparseLayer/sparse_head observers; they retain only
JSON-compatible metadata. No tensor registry, scheduling or lifecycle framework.
"""
import argparse
import copy
import functools
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from .max_window_check import BASE,ROOT,read,save,sha,git,bounded_process,gpu_sample,identity,worker,host
from .csr_values_bridge_check import load_test
OUT=BASE/'pvst_tensor_lifetime_check_v1'


def metadata(t):
    storage=t.untyped_storage()
    return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'requires_grad':t.requires_grad,'grad_fn':type(t.grad_fn).__name__ if t.grad_fn else None,'payload_bytes':t.numel()*t.element_size(),'storage_bytes':storage.nbytes(),'storage_id':str(storage.data_ptr())}


def configure_trace(model,out,context):
    """Only explicit conversions/cleanup boundaries; node hooks capture no Tensor."""
    lock=threading.RLock();counter=[0];life_count=[0];before={};pending={}
    def event(kind,record):
        with lock:
            counter[0]+=1;item={'kind':kind,'event':counter[0],**context(),**record}
            if kind=='lifetime':life_count[0]+=1
            if counter[0]<=1024 or (kind=='lifetime' and life_count[0]<=256):
                with (out/'tensor_events.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
            if kind=='copy_backward_pre':pending[record['copy_id']]=item
            elif kind=='copy_backward_post':pending.pop(record['copy_id'],None)
            if kind.startswith('copy_backward'):save(out/'pending_copies.json',list(pending.values()))
            return counter[0] if kind=='copy_forward' else None
    def conversion(stage,tensor,head,label):
        pairs={'Q_before_float':'Q','K_before_float':'K','V_before_float':'V','result_before_dtype':'result','head_output_before':'head_output','attended_before':'attended'}
        after={'Q_after_float':'Q','K_after_float':'K','V_after_float':'V','result_after_dtype':'result','head_output_after':'head_output','attended_after':'attended'}
        if stage in pairs:
            before[(label,head,pairs[stage])]=metadata(tensor);return
        if stage not in after:return
        source=before.pop((label,head,after[stage]),None)
        record={'layer':label,'head':head,'conversion':after[stage],'source':source,'target':metadata(tensor)}
        number=event('copy_forward',record)
        node=tensor.grad_fn
        if node is not None and type(node).__name__=='ToCopyBackward0' and source is not None and (source['device']!=record['target']['device'] or source['dtype']!=record['target']['dtype']):
            # Unique metadata-only closure; no node, module, storage or Tensor captured.
            item=copy.deepcopy(record);item['copy_id']=label+':'+str(head)+':'+stage+':'+str(number)
            node.register_prehook(lambda gradients,item=item:event('copy_backward_pre',item))
            node.register_hook(lambda inputs,outputs,item=item:event('copy_backward_post',item))
    def lifetime(stage,tensors,label):
        metas={name:metadata(t) for name,t in tensors.items()};unique={v['storage_id']:v['storage_bytes'] for v in metas.values() if v['device'].startswith('cuda')}
        event('lifetime',{'layer':label,'stage':stage,'tensors':metas,'unique_observed_storage_bytes':sum(unique.values()),'allocated_bytes':__import__('torch').cuda.memory_allocated(),'reserved_bytes':__import__('torch').cuda.memory_reserved(),'host':host()})
    for name in ['fine_time','coarse_time']:
        for index,layer in enumerate(getattr(model,name).layers):
            label=name+'.'+str(index);layer.copy_trace=functools.partial(conversion,label=label);layer.lifetime_trace=functools.partial(lifetime,label=label)


def regression():
    import torch
    check=load_test('test_pvst_tensor_lifetime.py').compare_cleanup
    r={'status':'PASS','comparisons':[]}
    for dtype in [torch.float32,torch.bfloat16]:
        for dim in [256,512]:r['comparisons'].append(check('cuda',dtype,dim))
    save(OUT/'regression.json',r)


def lifetime_probe():
    import torch
    from torch.utils.checkpoint import checkpoint
    from .temporal import SparseLayer
    torch.manual_seed(20260909);records=[]
    m=SparseLayer(512,8,512,8192).cuda().train()
    # The real projection/checkpoint boundary, no single-layer throughput sweep.
    for outer in [False,True]:
        phase=['forward']
        x=torch.randn(131072,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        def project(x):
            attended=x*1.5;values=[]
            for xb,ab in zip(x.split(512),attended.split(512)):
                args=(xb,ab)
                with torch.autocast('cuda',dtype=torch.bfloat16):y=checkpoint(m.project_block,*args,use_reentrant=False,preserve_rng_state=True)
                values.append(y)
            torch.cuda.synchronize()
            record={'outer_checkpoint':outer,'phase':phase[0],'source':metadata(attended),'view':metadata(ab),'allocated_before_name_delete':torch.cuda.memory_allocated()}
            del attended
            record['allocated_after_name_delete']=torch.cuda.memory_allocated()
            del xb,ab,args
            record['allocated_after_view_cleanup']=torch.cuda.memory_allocated();records.append(record)
            return torch.cat(values)
        y=checkpoint(project,x,use_reentrant=False,preserve_rng_state=True) if outer else project(x)
        phase[0]='backward';y.square().mean().backward();torch.cuda.synchronize()
        if x.grad is None or not torch.isfinite(x.grad).all():raise ValueError('LIFETIME_PROBE_GRADIENT')
        del y,x
        m.zero_grad(set_to_none=True)
    save(OUT/'lifetime_probe.json',{'status':'PASS','records':records,'note':'actual project_block, native splits and checkpoint; synthetic attended=x*1.5 only isolates storage ownership, not a model/performance result'})


def run_case(name,spec,timeout):
    out=OUT/name;out.mkdir()
    spec['source_commit']=git('rev-parse','HEAD');spec['lifetime_trace']=True
    save(out/'identity.json',spec);start=gpu_sample(None)
    process=bounded_process([sys.executable,'-u','-m','research_tools.pvst.tensor_lifetime_check','--worker',name],timeout,out/'worker.log',gpu_sample)
    w=read(out/'worker_result.json') if (out/'worker_result.json').exists() else {'stages':read(out/'stages.json') if (out/'stages.json').exists() else []}
    w['updates']=max(w.get('updates',0),sum(s['stage']=='update_complete' for s in w.get('stages',[])))
    status='TIMEOUT' if process['timed_out'] else w.get('status','FAILED')
    if status=='PASS' and (process['returncode']!=0 or w['updates']!=spec.get('updates_requested',1)):status='FAILED'
    result={'status':status,'source_commit':spec['source_commit'],'process':process,'worker':w,'gpu_start':start,'gpu_after':gpu_sample(process['pid']),'pending_copies':read(out/'pending_copies.json') if (out/'pending_copies.json').exists() else []}
    save(out/'result.json',result);(out/'exit_code.txt').write_text(str(process['returncode'])+'\n');return result


def positive_spec(config,condition):
    """Frozen train identity first; clips preferred, otherwise native stride8 pool."""
    from .data import Video
    asset=Path(config['asset_root']);rows=read(asset/'manifest.json')['rows'];lookup={r['video_id']:r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('DUPLICATE_FROZEN_VIDEO_ID')
    clipfile=BASE/'pvst_point_voxel_v03/clips.json'
    if clipfile.exists():candidates=read(clipfile)['clips'];origin=str(clipfile)
    else:
        candidates=[{'video_id':r['video_id'],'input_range':[start-16,start+8],'target_range':[start,start+8]} for r in rows if r['dataset']=='ANAKIN' and r['split']=='train' and r['role']=='fake' for start in range(16,r['actual_frames']-7,8)];origin='no frozen clips.json exists; deterministic stride8 asset-review pool, not a new formal training split'
    candidates=sorted(candidates,key=lambda c:(lookup[c['video_id']]['dataset'],lookup[c['video_id']]['split'],lookup[c['video_id']]['source_id'],lookup[c['video_id']]['pair_id'],lookup[c['video_id']]['video_sha256'],c['target_range']))
    for clip in candidates:
        row=lookup[clip['video_id']];lo,hi=clip['input_range'];start,end=clip['target_range']
        if row['dataset']!='ANAKIN' or row['split']!='train' or end-start!=8 or start-lo!=16:continue
        # GT selection uses UV reader so no XYZ-derived observations are read.
        v=Video(asset,row,'UV_PVST',1.,config)
        try:nonempty=[bool(v.gt(i).any()) for i in range(start,end)]
        finally:v.close()
        if not any(nonempty):continue
        if condition=='UV_PVST':scale=1.
        else:scale=read(asset/'scale.json')['s']
        return {'condition':condition,'config':config,'row':row,'global_scale':scale,'input_range':[lo,hi],'target_range':[start,end],'expected_nodes':None,'updates_requested':1,'require_nonempty_gt':True,'selection':origin,'target_nonempty_gt':nonempty,'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    raise ValueError('NO_FROZEN_TRAIN_NONEMPTY_FULL_CONTEXT_CLIP')


def brief(case):
    w=case['worker'];keep=['status','updates','optimizer_step_calls','update_records','fine_nodes','coarse_nodes','phase','update_index','elapsed_s','peak_allocated_bytes','peak_reserved_bytes','host','optimizer_state_parameters','failure_operator_context']
    return {**{k:v for k,v in case.items() if k not in ['worker','process']},'process':{k:v for k,v in case['process'].items() if k!='samples'},'worker':{k:w[k] for k in keep if k in w},'stages':[{k:v for k,v in s.items() if k not in ['trace','candidate_layers']} for s in w.get('stages',[])]}


def report(r):
    lines=['# PVST v0.3 无用view清理与训练前验收','', '**状态：'+r['status']+'；正式模型0/2，不启动训练。**',
        '源码仅在最后使用后删除projection循环xb/ab/args、已使用的分块列表和FFN尾部part/x；没有永久清理开关，没有绕过autograd。共享view的payload不等于底层storage，去重按storage_id。原CSR、逐headQKV、CPU head、split和外层卸载全部保留。',
        '显式转换仅增加元数据与ToCopy节点前/后hook，闭包不捕获Tensor；记录Q/K/V BF16→FP32、核心输出FP32→BF16、GPU head→CPU、CPU stack→GPU。pending_copies保存已开始但未完成的节点，不凭364MiB猜身份。checkpoint内部恢复未新增拷贝实现或自定义反向。复制事件日志上限1024，生命周期边界另限256条，并保留当前pending，非完整profiler/allocator history。',
        '## 正确性与存活证据','', '使用本分支固定父提交b7ab0b8的真实SparseLayer作为只读参考，原参数/键/数量/身份、dropout/RNG、尾块、checkpoint前反向与全部梯度核对；沿用原FP32/BF16容差，未放宽。','|dtype/宽度|输出max abs|输入梯度max abs|参数梯度max abs|','|---|---:|---:|---:|']
    for c in r.get('regression',{}).get('comparisons',[]):lines.append(f'|{c["dtype"]}/{c["dim"]}|{c["output"]["max_abs"]:.8g}|{c["gradients"]["input"]["max_abs"]:.8g}|{max(v["max_abs"] for k,v in c["gradients"].items() if k!="input"):.8g}|')
    lines+=['','单层存活检查只隔离真实project_block与checkpoint；attended=x*1.5用于明确存储归属，不作为模型效果或吞吐。无gc.collect/empty_cache。','```json',json.dumps(r.get('lifetime_probe'),ensure_ascii=False,indent=2),'```',
        '## 逐case真实更新','']
    for name,case in r.get('cases',{}).items():
        w=case['worker'];lines+=['### '+name,'', '**'+case['status']+'**；完成更新'+str(w.get('updates',0))+'；optimizer.step成功返回次数'+str(w.get('optimizer_step_calls',0))+'；returncode/signal='+str(case['process']['returncode'])+'/'+str(case['process']['termination_signal'])+'。',
            '|更新/阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/可用GiB|','|---|---:|---:|---:|---:|']
        for s in w.get('stages',[]):lines.append(f'|{s.get("update_index",0)}/{s["stage"]}|{s.get("stage_s")}|{s["elapsed_s"]:.3f}|{s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f}|{s["host"]["rss_bytes"]/2**30:.3f}/{s["host"]["available_bytes"]/2**30:.3f}|')
        lines+=['实际fine/coarse='+str(w.get('fine_nodes'))+'/'+str(w.get('coarse_nodes'))+'；worker至末态='+str(w.get('elapsed_s'))+'秒，父进程墙时='+str(case['process']['wall_s'])+'秒。',
            '每次更新loss/梯度/参数变化、pending转换及错误：','```json',json.dumps({'updates':w.get('update_records'),'pending_copies':case.get('pending_copies'),'error_first_line':w.get('error','').splitlines()[:1],'native_frames':[line for line in w.get('error','').splitlines() if ' at::native::' in line or 'Backward0::apply' in line][:8],'host':w.get('host')},ensure_ascii=False,indent=2),'```']
    lines+=['','## 结论边界','',
        '最大窗口固定XYZ video782426ea0330c7fea684953c、ANAKIN/train/real/an:4wdo37、720×1280、history[104,120)、target[120,128)、fine1/8/coarse1/4，节点2970629/1488042。seed20260909、BF16/逐head FP32、attention512/QKV8192/FFN8192/point32768/tile256，真实8帧均值frame BCE及合法AN real零GT pixel BCE/Dice不变。',
        '同一最大窗口最多两次正常zero_grad/重新前向反向更新，900秒总时限；第一次失败不做第二次。只有两次通过才补同一非空GT训练片段XYZ/UV各一次、各300秒。优先冻结clips，缺失则稳定身份排序冻结train队列stride8首合格片段，绝不读开发/测试标签。UV只走UV reader，scale1/32，不读XYZ派生输入。',
        '未完成阶段不填0；捕获OOM只落盘已有CPU证据，不继续CUDA计算。SIGABRT与计算状态分记，资源检查不计正式模型；不拿最大窗口时间冒充全队列平均。',
        '当前结论：'+('全部规定case通过，可认为具备进一步授权原定各一轮贯通的训练前条件，但只覆盖本轮case，不启动训练。' if r['status']=='PASS' else '尚未完成全部规定验收，不具备直接启动原定两条件训练贯通的依据；本轮停止，不追加优化。'),
        '源码`'+r['source_commit']+'`；产物`'+str(OUT)+'`；旧缓存/协议/错误/退出历史/config和用户JSON保持。计算与publication分列。错误/门槛：'+str(r.get('reason','无'))]
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/tensor_lifetime_check_v1.md').write_text(text)
    s={k:v for k,v in r.items() if k!='cases'};s['cases']={name:brief(case) for name,case in r.get('cases',{}).items()}
    save(OUT/'summary.json',s);save(ROOT/'docs/pvst/tensor_lifetime_check_v1.json',s)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+'\n无用view清理/训练前验收：'+r['status']+'；正式模型0/2，未训练。见[报告](tensor_lifetime_check_v1.md)。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['regression','lifetime_probe','max_window','gt_xyz','gt_uv']);a=p.parse_args()
    if a.worker:
        import torch
        torch.set_num_threads(4)
        if a.worker=='regression':regression()
        elif a.worker=='lifetime_probe':lifetime_probe()
        else:
            spec=read(OUT/a.worker/'identity.json')
            if spec.get('condition')=='UV_PVST':
                from . import data
                def forbidden(*args,**kwargs):raise RuntimeError('UV_MUST_NOT_READ_XYZ')
                data.native_dense=forbidden
            raise SystemExit(worker(OUT/a.worker))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse, no automatic retry')
    source=git('rev-parse','HEAD');save(OUT/'identity.json',{'source_commit':source,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'tests/test_pvst_tensor_lifetime.py']}})
    r={'status':'FAILED','source_commit':source,'cases':{},'small_gpu_wall_s':0.,'formal_models_completed':0}
    try:
        for name in ['regression','lifetime_probe']:
            folder=OUT/name;folder.mkdir();process=bounded_process([sys.executable,'-u','-m','research_tools.pvst.tensor_lifetime_check','--worker',name],300-r['small_gpu_wall_s'],folder/'worker.log');save(folder/'process.json',process);r['small_gpu_wall_s']+=process['wall_s']
            if process['returncode']!=0 or process['timed_out']:raise RuntimeError('SMALL_CHECK_FAILED:'+name)
            r[name]=read(OUT/(name+'.json'));save(OUT/'partial.json',r)
        spec=identity(sparse=True);spec['config'].update(head_output_offload=True,head_qkv_stream=True,node_split=True);spec['updates_requested']=2
        r['cases']['max_window']=run_case('max_window',spec,900);save(OUT/'partial.json',r)
        if r['cases']['max_window']['status']!='PASS':r['status']=r['cases']['max_window']['status']
        else:
            for name,condition in [('gt_xyz','XYZ_PVST'),('gt_uv','UV_PVST')]:
                selected=positive_spec(spec['config'],condition);r['cases'][name]=run_case(name,selected,300);save(OUT/'partial.json',r)
            r['status']='PASS' if all(c['status']=='PASS' for c in r['cases'].values()) else next(c['status'] for c in r['cases'].values() if c['status']!='PASS')
    except Exception as error:r['reason']=str(error);r['traceback']=__import__('traceback').format_exc()
    report(r);save(OUT/'progress.json',{'status':r['status'],'stage':'report_complete','formal_models_completed':0})
    try:
        subprocess.run(['git','add','--','docs/pvst/tensor_lifetime_check_v1.md','docs/pvst/tensor_lifetime_check_v1.json','docs/pvst/current_status.md'],check=True);subprocess.run(['git','commit','-m','Record tensor lifetime cleanup and pre-training acceptance'],check=True)
        head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':r['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':r['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':r['status'],'error':__import__('traceback').format_exc()})

if __name__=='__main__':main()
