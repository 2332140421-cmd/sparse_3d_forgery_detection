"""One confirmed SliceBackward fix, matched layer, one bounded candidate.

Reads the one original-HEAD diagnostic; never launches another diagnostic.
Reuses existing CUDA workers, timing, model, loss and 900s process deadline.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from .max_window_check import BASE,ROOT,bounded_process,read,save,sha,git
from .csr_values_bridge_check import load_test,benchmark,paired_check
OUT=BASE/'pvst_split_backward_check_v1'


def regression():
    import torch
    check=load_test('test_pvst_split_backward.py').compare_split
    r={'status':'PASS','comparisons':[]}
    for dtype in [torch.float32,torch.bfloat16]:
        for dim in [256,512]:r['comparisons'].append(check('cuda',dtype,dim))
    save(OUT/'regression.json',r)


def brief_full(f):
    w=f.get('worker',{})
    fields=['status','updates','fine_nodes','coarse_nodes','elapsed_s','phase','peak_allocated_bytes','peak_reserved_bytes','host','gradients','optimizer_state_parameters','parameter_update_verified','parameter_delta_norms','failure_operator_context','failure_phase']
    result={k:w[k] for k in fields if k in w};result['error_first_line']=w.get('error','').splitlines()[:1]
    result['stages']=[{k:v for k,v in s.items() if k not in ['trace','candidate_layers']} for s in w.get('stages',[])]
    result['layers']=[{k:v for k,v in row.items() if k in ['layer','nodes','edges','dilation','completed_heads','blocking_execution']} for row in w.get('forward_edge_evidence',[])]
    return {'status':f['status'],'source_commit':f['source_commit'],'process':{k:v for k,v in f['process'].items() if k!='samples'},'gpu_start':f.get('gpu_start'),'gpu_after':f.get('gpu_after'),'worker_gpu_released':f.get('worker_gpu_released'),'worker':result}


def report(r):
    diag=read(OUT/'diagnostic/result.json');dw=diag['worker'];native=[line for line in dw.get('error','').splitlines() if any(s in line for s in ['#19 ','#24 ','SliceBackward0::apply','slice_backward_wrapper'])]
    lines=['# PVST v0.3 SliceBackward取证与原生split修复','',
        '**候选结果：'+r['status']+'；正式模型0/2。** 原实现完整诊断1次，修复后完整验收至多1次；无正式训练/性能指标。',
        '## 已确认的申请来源','',
        '原HEAD `'+diag['source_commit']+'`、相同最大窗口/初始化/开关，仅启动前添加TORCH_SHOW_CPP_STACKTRACES=1，没有allocator history或全程profiler。原配置再次OOM，1.42GiB申请来自 **SliceBackward0 → slice_backward_wrapper → at::native::slice_backward → zeros → CUDA allocator malloc**。不是只根据申请量或最近FFN记录猜测。',
        '关键原生栈（完整证据留diagnostic/worker_result.json）：','```','\n'.join(native),'```',
        '确切slice的input_sizes/dtype未被C++异常文本打印，不能唯一指定fine/coarse或FFN/QKV/投影中的哪一个调用；两尺度完整BF16节点特征均约1.42GiB，属于形状线索。没有内存snapshot，驻留GPU张量不能逐个归类；已记录allocated/reserved、last trace、CPU head payload与当前head QKV shape，不把形状估算当精确驻留清单。',
        '对应安装PyTorch2.5.1+cu124的官方v2.5.1源码核对：[TensorShape.cpp slice_backward](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/TensorShape.cpp#L2381-L2385)建立input_sizes零梯度后回填切片；[derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.5.1/tools/autograd/derivatives.yaml)映射slice.Tensor；[FunctionsManual.cpp](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/autograd/FunctionsManual.cpp#L1979-L2047)的split反向汇总分块梯度，仍需要完整输入梯度，并非零内存。未升级依赖。',
        '小规模实际执行：[41,256]以11/7节点重复切片，分别4/6次完整尺寸slice_backward；一次split为SplitBackward0、无对应slice_backward。当前真实SparseLayer小图原路径共40次[41,256]反向分配，不将累计字节当作同时驻留峰值。CPU原始记录见small_original.json。',
        '## 修复与正确性','',
        '只增加默认False的node_split，三个已有大激活入口统一一次split：streamed_head的norm+QKV、attention投影/残差的两个输入、FFN。原qkv权重行切片、像素tile等保持；无全仓替换、CopySlices预分配、自定义autograd、原地修改多输出view或输入复制。共享小方法node_parts只供这三处调用并记录实际view grad_fn。',
        '逐head QKV、CPU head暂存、CSRSoftmax/CSRFromValues、全部节点/边/像素/时序/宽度、checkpoint/dropout/RNG、初始化/损失/参数身份和键不变。split仍可能暂存各块梯度并cat成完整输入梯度，不预填峰值减半或PASS。',
        '原容差FP32输出2e-5/2e-4、梯度2e-5/3e-3；BF16输出.04/.03、梯度.003/.08（atol/rtol），未放宽。',
        '|dtype/宽度|输出max abs|输入梯度max abs|参数梯度max abs|原/新全节点slice_backward次数|','|---|---:|---:|---:|---:|']
    for c in r.get('regression',{}).get('comparisons',[]):lines.append(f'|{c["dtype"]}/{c["dim"]}|{c["output"]["max_abs"]:.8g}|{c["gradients"]["input"]["max_abs"]:.8g}|{max(v["max_abs"] for k,v in c["gradients"].items() if k!="input"):.8g}|{c["proof"][0]["full_node_slice_backward_count"]}/{c["proof"][1]["full_node_slice_backward_count"]}|')
    lines+=['','小图完整SparseLayer、dropout开启、内外checkpoint及尾块，含全部norm/QKV/bias/projection/FFN参数梯度；参数身份/数量/state_dict与默认关闭复现核对。Dispatch检查前向/反向重算，不要求所有SliceBackward为零，仅禁止对应全节点激活的重复分配和旧[N,3D]/逻辑N×N稠密化。',
        '## 唯一匹配单层','',
        '实际N131072/E7475256、dim512/8head。两组均逐head QKV＋CPU head暂存＋CSR桥接；仅node_split不同，同seed/输入/权重/BF16/dropout/checkpoint，独立进程一次预热后同步计时，无搜索。',
        '|路径|前向秒|反向秒|peak allocated/reserved GiB|测量峰值RSS GiB|','|---|---:|---:|---:|---:|']
    for name in ['slice','split']:
        x=r.get(name)
        if x:lines.append(f'|{name}|{x["forward_s"]:.6f}|{x["backward_s"]:.6f}|{x["peak_allocated_bytes"]/2**30:.4f}/{x["peak_reserved_bytes"]/2**30:.4f}|{x["measurement_host"]["peak_rss_bytes"]/2**30:.4f}|')
    lines+=['','大图全部参数梯度核对，输出/输入梯度固定64行（每2048节点一行），不称全量输出比较；完整小图全量。实际分块view记录保留slice.json/split.json，SliceBackward/SplitBackward原生行为来自小图执行，不在大图计时中开profiler。小GPU累计墙时='+str(r.get('small_gpu_wall_s'))+'秒/300秒。',
        '## 完整窗口','',
        '两次均XYZ_PVST，video782426ea0330c7fea684953c、ANAKIN/train/real/an:4wdo37，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4；fine2970629/coarse1488042。seed20260909、attention512/QKV8192/FFN8192/point32768/tile256、BF16/逐head FP32、外层CPU卸载/CPU head/逐headQKV原样，候选仅增加node_split。正式XYZ=null。']
    cases=[('原实现诊断',diag)]
    if r.get('full_window'):cases.append(('修复后验收',r['full_window']))
    for name,f in cases:
        w=f.get('worker',{});p=f['process']
        lines+=['',name+'：**'+f['status']+'**；源码`'+f['source_commit']+'`；returncode/signal='+str(p['returncode'])+'/'+str(p['termination_signal'])+'。',
            '|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/主机可用GiB|','|---|---:|---:|---:|---:|']
        for s in w.get('stages',[]):lines.append(f'|{s["stage"]}|{s.get("stage_s")}|{s["elapsed_s"]:.3f}|{s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f}|{s["host"]["rss_bytes"]/2**30:.3f}/{s["host"]["available_bytes"]/2**30:.3f}|')
        lines+=['更新次数='+str(w.get('updates',0))+'；参数改变核对='+str(w.get('parameter_update_verified',False))+'；优化器状态参数数='+str(w.get('optimizer_state_parameters','未到达'))+'；worker至末态='+str(w.get('elapsed_s','未正常结束'))+'秒；父进程总墙时='+str(p['wall_s'])+'秒。','最后阶段='+str(w.get('phase'))+'；错误首行='+str(w.get('error','无').splitlines()[:1])+'。']
    f=r.get('full_window',{});w=f.get('worker',{})
    if f:
        lines+=['','六层实际split入口：','```json',json.dumps([{k:v for k,v in row.items() if k in ['layer','nodes','edges','completed_heads','blocking_execution']} for row in w.get('forward_edge_evidence',[])],ensure_ascii=False,indent=2),'```',
            '关键梯度、内存和失败上下文：','```json',json.dumps({k:w.get(k) for k in ['gradients','parameter_delta_norms','host','peak_allocated_bytes','peak_reserved_bytes','failure_operator_context']},ensure_ascii=False,indent=2),'```']
    lines+=['','诊断仅增加C++错误栈，符号化/错误收尾可能增加成本，不能把诊断墙时当无诊断吞吐。候选验收不启用CPP诊断环境或全程profiler。两个worker均原父进程900秒限时，不重试；未完成阶段不填0，失败观测峰值不冒充完整更新上界。',
        '结论：'+('规定最大窗口一次真实AdamW更新通过；仅此窗口，不代表全队列/正式尺度冻结，需结合成本和余量再判断训练，本轮不启动。' if r['status']=='PASS' else '未完成规定最大窗口一次更新，不建议直接正式训练；本轮停止，不再叠加其他优化。'),
        '错误/门槛：'+str(r.get('reason','无')),'产物：`'+str(OUT)+'`。原资产/协议/失败日志/退出码/config和用户无关JSON保留，权重丢弃无checkpoint，正式模型0/2；计算与Git发布分列。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/split_backward_check_v1.md').write_text(text)
    s={k:v for k,v in r.items() if k not in ['full_window','slice','split']}
    for name in ['slice','split']:
        if name in r:s[name]={k:v for k,v in r[name].items() if k in ['nodes','edges','forward_s','backward_s','baseline_allocated_bytes','peak_allocated_bytes','peak_reserved_bytes','measurement_host','forward_blocking_execution','backward_blocking_execution']}
    s['diagnostic']=brief_full(diag)
    if f:s['full_window']=brief_full(f)
    save(OUT/'summary.json',s);save(ROOT/'docs/pvst/split_backward_check_v1.json',s)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+'\n重复节点切片原生取证/split限定验收：'+r['status']+'；原诊断确认SliceBackward零缓冲OOM，正式模型0/2。见[报告](split_backward_check_v1.md)。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['regression','slice','split']);a=p.parse_args()
    if a.worker:
        import torch
        torch.set_num_threads(4)
        if a.worker=='regression':regression()
        else:benchmark(a.worker,head_output_offload=True,out=OUT,head_qkv_stream=True,node_split=a.worker=='split')
        return
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse; no retry')
    diag=read(OUT/'diagnostic/result.json');source=git('rev-parse','HEAD')
    save(OUT/'identity.json',{'source_commit':source,'diagnostic_source_commit':diag['source_commit'],'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'tests/test_pvst_split_backward.py']}})
    r={'status':'FAILED','source_commit':source,'full_window_entered':False,'small_gpu_wall_s':0.,'formal_models_completed':0,'processes':[]}
    try:
        if diag['status']!='OOM' or 'at::native::slice_backward' not in diag['worker'].get('error',''):r['status']='DIAGNOSIS_INCONCLUSIVE';raise RuntimeError('NO_CONFIRMED_SLICE_OOM')
        for name in ['regression','slice','split']:
            folder=OUT/name;folder.mkdir();proc=bounded_process([sys.executable,'-u','-m','research_tools.pvst.split_backward_check','--worker',name],max(0,300-r['small_gpu_wall_s']),folder/'worker.log');save(folder/'process.json',proc);r['small_gpu_wall_s']+=proc['wall_s'];r['processes'].append({'name':name,**proc})
            if proc['timed_out']:r['status']='TIMEOUT';raise RuntimeError('SMALL_TIMEOUT:'+name)
            if proc['returncode']!=0:raise RuntimeError('SMALL_FAILED:'+name)
            r[name]=read(OUT/(name+'.json'));save(OUT/'partial.json',r)
        r['paired_check']=paired_check(out=OUT,names=('slice','split'))
        if any(v['mode']!='split' or v['view_grad_fn']!='SplitBackward0' for v in r['split']['forward_blocking_execution'].values()):raise ValueError('NATIVE_SPLIT_NOT_USED')
        r['full_window_entered']=True;save(OUT/'partial.json',r)
        subprocess.run([sys.executable,'-u','-m','research_tools.pvst.max_window_check','--node-split'],check=True)
        r['full_window']=read(OUT/'full_window/result.json');r['status']=r['full_window']['status']
    except Exception as error:r['reason']=str(error);r['traceback']=traceback.format_exc()
    report(r);save(OUT/'progress.json',{'status':r['status'],'stage':'report_complete','formal_models_completed':0})
    try:
        subprocess.run(['git','add','--','docs/pvst/split_backward_check_v1.md','docs/pvst/split_backward_check_v1.json','docs/pvst/current_status.md'],check=True);subprocess.run(['git','commit','-m','Record native SliceBackward diagnosis and split acceptance'],check=True)
        head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':r['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':r['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':r['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
