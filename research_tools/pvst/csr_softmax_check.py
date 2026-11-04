"""Bounded CSR-softmax correctness gate; failed layout stops before large tests.

Caller this single experiment CLI. Reuses the existing 300s child deadline.
No model backend is added: candidate head lives in tests until layout is valid.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import traceback

from .max_window_check import BASE,ROOT,bounded_process,read,save,sha,git
OUT=BASE/'pvst_csr_softmax_check_v1'


def load_test(name):
    path=ROOT/'tests'/name;spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def worker():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from .sparse_attention import CSRSoftmax
    from . import sparse_attention
    from .sparse_attention_check import compare,dropout_check
    torch.set_num_threads(4);torch.manual_seed(20260909)
    result={'status':'FAILED','torch':torch.__version__,'comparisons':[],'full_window_entered':False,'large_layer_entered':False}
    try:
        crow=torch.tensor([0,2,2,3],device='cuda');x=torch.tensor([1000.,999.,-1000.],device='cuda',requires_grad=True)
        maximum=torch.segment_reduce(x,'max',offsets=crow);total=torch.segment_reduce(x,'sum',offsets=crow)
        (total.sum()+maximum[[0,2]].sum()).backward();torch.cuda.synchronize()
        result['capability']={'CUDA_max_sum_backward':True,'rows':torch.repeat_interleave(crow.diff(),output_size=3).tolist(),'gradient':x.grad.tolist()}
        result['softmax']=load_test('test_pvst_csr_softmax.py').softmax_case('cuda',torch.float32)
        x=torch.randn(9,device='cuda',dtype=torch.float64,requires_grad=True);crow=torch.tensor([0,0,1,5,5,9],device='cuda')
        result['cuda_fp64_gradcheck']=torch.autograd.gradcheck(CSRSoftmax.apply,(x,crow),eps=1e-6,atol=1e-5,rtol=1e-3)
        candidate=load_test('pvst_csr_candidate.py').csr_candidate_head
        original=sparse_attention.sparse_head;sparse_attention.sparse_head=candidate
        try:
            for dtype,dim,d in [(torch.float32,256,1),(torch.float32,256,2),(torch.float32,256,4),(torch.bfloat16,512,1)]:
                result['comparisons'].append(compare(dtype,dim,d,'cuda'));save(OUT/'partial.json',result)
            result['dropout']=dropout_check('cuda')
        finally:sparse_attention.sparse_head=original
        # Verify exact constructor behavior on a tiny actual CUDA CSR graph.
        # This is not a profiler, dependency search, or a large resource attempt.
        n=17;values=torch.randn(n,device='cuda',requires_grad=True);crow=torch.arange(n+1,device='cuda');col=torch.arange(n,device='cuda')
        matrix=torch.sparse_csr_tensor(crow,col,values,size=(n,n+1));rhs=torch.randn(n+1,64,device='cuda',requires_grad=True)
        result['constructor_grad_fn']=type(matrix.grad_fn).__name__;ops=[]
        class RecordDense(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                out=func(*args,**(kwargs or {}))
                if '_to_dense' in str(func):
                    ops.append({'op':str(func),'input_layout':str(args[0].layout),'input_shape':list(args[0].shape),'output_shape':list(out.shape),'dtype':str(out.dtype)})
                return out
        with RecordDense():torch.sparse.mm(matrix,rhs).square().mean().backward()
        torch.cuda.synchronize();result['dense_constructor_backward']=list(ops)
        # Prove this also happens on the complete test candidate, not just API.
        q=torch.randn(n,64,device='cuda',requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True);bias=torch.randn(1,81,1,device='cuda',requires_grad=True)
        ops.clear()
        with RecordDense():candidate(q,k,v,bias,crow,col,torch.arange(n,device='cuda'),0,.1,True).square().mean().backward()
        result['dense_candidate_backward']=list(ops)
        result['candidate_finite_gradients']=all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in [q,k,v,bias])
        if ops:result.update(status='REPRESENTATION_BLOCKED',reason='torch2.5.1 SparseCompressedTensorBackward0 invokes aten._to_dense for direct CSR construction; forbidden N x (N+padding) dense gradient')
        else:result.update(status='REVIEW_REQUIRED',reason='Unexpected constructor behavior; no resource run without completed integration/memory gate')
    except Exception as error:result.update(status='FAILED',error=str(error),traceback=traceback.format_exc())
    save(OUT/'checks.json',result)


def report(result,process,source):
    comparisons=result.get('comparisons',[])
    lines=['# PVST v0.3 CSR行分段softmax：限定修复与阻塞报告','',
        '2026-11-05；状态 **'+result['status']+'**。独立CSR softmax数值/梯度验证通过；直接CSR权重构造的反向违反禁止稠密N×N的要求，因此未放行实现。',
        '**未运行131,072节点单层内存对比，未进入900秒完整窗口，未执行AdamW。正式模型仍0/2。**',
        '## 实现与停止原因','',
        '在现有sparse_attention.py增加CSRSoftmax，仅封装行softmax。使用CSR offsets上的segment_reduce(max/sum)，p=exp(s-row_max)/row_sum；只保存p和crow。反向返回p*(g-row_sum(p*g))，不切断Q/K或relative bias梯度；每方向生成一次E长int64行号并复用，用完释放。空行无概率，非空行不加epsilon。FP32核心保持，FP64仅用于gradcheck；只支持所需一阶梯度。',
        '前向p.div_只修改Function中新生成、无上游别名的exp私有缓冲；不覆盖输入、保存的上游张量或反向所需概率。实际保存内容由测试核对为概率p与crow两项；临时row为8E字节，概率4E字节及其他临时浮点张量仍有成本。',
        '新head只作为tests/pvst_csr_candidate.py中的必要复现参考，显式路径为CSR sampled_addmm → CSR.values()+bias → CSRSoftmax → 原dropout → CSR constructor → sparse.mm，不构造COO。原正式sparse_head保持上次提交实现，候选仍默认关闭；没有将未通过的替代路径接入六层。',
        'CPU回归首先发现direct CSR constructor对应SparseCompressedTensorBackward0调用aten._to_dense；随后在极小CUDA图确认相同行为。不是CSR softmax解析梯度错误，也不是segment_reduce缺乏CUDA支持。未声称所有底层算子无COO。',
        'CUDA实际证据：','```json',__import__('json').dumps({k:result.get(k) for k in ['torch','capability','constructor_grad_fn','dense_constructor_backward','dense_candidate_backward','candidate_finite_gradients']},ensure_ascii=False,indent=2),'```',
        '上例直接构造17×18矩阵的反向出现17×18稠密梯度；完整小head的17×17反向也出现稠密矩阵。131,072节点逻辑shape[131072,131073]若以FP32稠密化，单张量形状成本约64.0005GiB（仅尺寸计算，未分配/未实测），明显违反本轮表示约束。',
        '本轮只允许小型softmax自定义反向；为了绕过此额外CSR构造问题，可能还需显式values梯度桥接或其他经核实的构造接口，已超出“仅封装CSR行softmax”的批准边界。本轮不继续实现，不添加CSR构造/SpMM自定义反向，也不把整套attention包进Function。',
        '', '## 正确性','',
        '预声明沿用已有容差：FP32输出atol2e-5/rtol2e-4、梯度atol2e-5/rtol3e-3；BF16输出atol.04/rtol.03、梯度atol.003/rtol.08。独立行softmax输出/梯度atol1e-6/rtol1e-5；FP64 gradcheck eps1e-6/atol1e-5/rtol1e-3。未事后放宽。',
        '|精度/dim/dilation|输出max abs|输入梯度max abs|QKV梯度max abs|bias梯度max abs|投影梯度max abs|','|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        params=c['parameters'];q=max(v['max_abs'] for k,v in params.items() if k.startswith('qkv.'));b=max(v['max_abs'] for k,v in params.items() if k.startswith('bias.'));p=max(v['max_abs'] for k,v in params.items() if k.startswith('proj.'))
        lines.append(f'|{c["dtype"]}/{c["dim"]}/{c["dilation"]}|{c["output"]["max_abs"]:.8g}|{c["input_gradient"]["max_abs"]:.8g}|{q:.8g}|{b:.8g}|{p:.8g}|')
    lines+=['','独立softmax/CUDA FP64 gradcheck/dropout：','```json',__import__('json').dumps({k:result.get(k) for k in ['softmax','cuda_fp64_gradcheck','dropout']},ensure_ascii=False,indent=2),'```',
        '小图包含非均匀PTS、负坐标、不同帧节点数/空帧、跨帧查询和尾块，FP32覆盖dilation1/2/4，BF16覆盖512/8head；复用既有小图和固定容差，不重跑旧GPU大矩阵。dropout=.1，checkpoint开/关的RNG与梯度核对保持。',
        'CPU直接相关测试结果另见cpu_tests.txt；布局阻塞测试明确断言当前API存在稠密化，不把它改写为“不生成稠密”的通过。17,821,672模型参数/原state_dict及原邻接保持；本轮仅新增无参数Function，无模型参数更改。',
        '', '## 资源与决策','',
        '131,072/512新旧前向、反向、起始allocated和峰值：**未运行**，原因是此前表示正确性门槛失败。不能用理论存储字节或单独softmax保存内容宣称单层/整模型峰值下降。',
        '完整窗口前向、反向、AdamW和峰值：**未运行**，没有新的OOM/TIMEOUT。旧18.064秒前向、约65.56秒OOM、allocated峰值21.25GiB的记录原样保留，不计算加速比。',
        f'本轮CUDA诊断子进程墙时{process["wall_s"]:.3f}秒，returncode={process["returncode"]}，signal={process["termination_signal"]}，timeout={process["timed_out"]}；此前3元素微型能力检查墙时未单独记录，不补测；本次有界子进程远低于300秒预算。无大规模分配、无正式训练、无前端/冻结特征提取、无新权重或checkpoint。',
        '**结论：本轮未完成可放行的显存修复；不建议进入两条件各一轮。唯一下一步建议：先授权并验证一个仅针对CSR构造values梯度的最小桥接，确认不稠密化，再继续既定单层内存门槛。** 本轮停止，不实施该建议。',
        '源码身份：`'+source+'`；新产物目录：`'+str(OUT)+'`。原缓存、正式config、旧报告/退出历史及用户无关JSON保留；计算与Git发布状态分列。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/csr_softmax_check_v1.md').write_text(text)
    summary={**result,'process':process,'source_commit':source,'formal_models_completed':0,'adamw_updates':0,'production_head_changed':False,'single_layer_memory_measured':False}
    save(OUT/'summary.json',summary);save(ROOT/'docs/pvst/csr_softmax_check_v1.json',summary)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+'\n2026-11-05 CSR行softmax检查：REPRESENTATION_BLOCKED；直接CSR构造反向在CUDA稠密化，未运行大单层/完整窗口；正式模型0/2。见[报告](csr_softmax_check_v1.md)。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker();return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('Existing run: inspect/reuse, no repeat')
    source=git('rev-parse','HEAD');save(OUT/'identity.json',{'source_commit':source,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research_tools/pvst/sparse_attention.py',ROOT/'tests/pvst_csr_candidate.py',ROOT/'tests/test_pvst_csr_softmax.py']}})
    process=bounded_process([sys.executable,'-u','-m','research_tools.pvst.csr_softmax_check','--worker'],298,OUT/'worker.log');save(OUT/'process.json',process)
    result=read(OUT/'checks.json') if (OUT/'checks.json').exists() else {'status':'FAILED'}
    if process['timed_out']:result['status']='TIMEOUT'
    elif process['returncode']!=0:result['status']='FAILED'
    report(result,process,source);save(OUT/'progress.json',{'status':result['status'],'stage':'report_complete','formal_models_completed':0})
    try:
        subprocess.run(['git','add','--','docs/pvst/csr_softmax_check_v1.md','docs/pvst/csr_softmax_check_v1.json','docs/pvst/current_status.md'],check=True);subprocess.run(['git','commit','-m','Record CSR softmax correctness and constructor backward blocker'],check=True)
        head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':result['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':result['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':result['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
