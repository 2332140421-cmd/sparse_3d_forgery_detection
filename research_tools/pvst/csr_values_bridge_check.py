"""One fixed-CSR bridge gate, one layer comparison, at most one real window.

Caller this CLI, outputs report/evidence. Reuses bounded_process, proven graph,
small layer parity and the real max_window_check; no new scheduling framework.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback
import time

from .max_window_check import BASE,ROOT,bounded_process,read,save,sha,git
OUT=BASE/'pvst_csr_values_bridge_check_v1'
BACKWARD_RATIO_LIMIT=1.25  # Frozen before measurements; no tuning/retry.


def load_test(name):
    path=ROOT/'tests'/name;spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def regression():
    import torch
    from .sparse_attention_check import compare,dropout_check
    test=load_test('test_pvst_csr_values_bridge.py')
    result={'status':'PASS','torch':torch.__version__,'bridge':test.bridge_check('cuda'),'layout':test.layout_check('cuda'),'comparisons':[]}
    for dtype,dim,d in [(torch.float32,256,1),(torch.float32,256,2),(torch.float32,256,4),(torch.bfloat16,512,1)]:result['comparisons'].append(compare(dtype,dim,d,'cuda'))
    result['dropout']=dropout_check('cuda')
    # Exercise the same logical padding branch on the complete small layer.
    from . import sparse_attention
    original=sparse_attention.intersection_padding;sparse_attention.intersection_padding=lambda rows,cols:1
    try:result['padded_layout']=test.layout_check('cuda')
    finally:sparse_attention.intersection_padding=original
    save(OUT/'regression.json',result)


def benchmark(kind):
    import torch
    import numpy as np
    from torch.utils.checkpoint import checkpoint
    from . import sparse_attention
    from .sparse_sigfpe import make_graph,validate_graph
    from .temporal import SparseLayer
    torch.manual_seed(20260909)
    if kind=='old':sparse_attention.sparse_head=load_test('pvst_coo_reference.py').coo_reference_head
    tick=time.monotonic();graph=make_graph(131072,'cuda');record=validate_graph(graph);torch.cuda.synchronize();record['graph_s']=time.monotonic()-tick
    model=SparseLayer(512,8,512,8192).cuda().train();model.qkv_chunk=8192;model.sparse_edge_attention=True
    x=torch.randn(131072,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    def run():
        model.zero_grad(set_to_none=True);x.grad=None
        with torch.autocast('cuda',dtype=torch.bfloat16):y=checkpoint(model,x,graph,1,use_reentrant=False,preserve_rng_state=True);loss=y.square().mean()
        return y,loss
    y,loss=run();loss.backward();torch.cuda.synchronize();del y,loss
    model.zero_grad(set_to_none=True);x.grad=None;torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    tick=time.monotonic();y,loss=run();torch.cuda.synchronize();forward=time.monotonic()-tick
    tick=time.monotonic();loss.backward();torch.cuda.synchronize();backward=time.monotonic()-tick
    if not torch.isfinite(x.grad).all() or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('NONFINITE_OR_MISSING_GRADIENT')
    record.update(status='PASS',kind=kind,dim=512,heads=8,head_dim=64,forward_s=forward,backward_s=backward,baseline_allocated_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),dropout=.1,outer_checkpoint=True,precision='BF16 outer / FP32 sparse core',softmax_saved='COO sparse softmax autograd' if kind=='old' else 'FP32 probabilities E + CSR crow; bridge saves crow/col, no values or dense matrix copy')
    # Timing/peak are captured before report snapshots. All parameter gradients
    # are compared, output/input gradients sampled at the same fixed 64 rows.
    arrays={'output_rows':y[::2048].float().cpu().tolist(),'input_gradient_rows':x.grad[::2048].float().cpu().tolist()}
    arrays.update({name:p.grad.float().cpu().numpy() for name,p in model.named_parameters()})
    np.savez(OUT/(kind+'_paired.npz'),**{k:np.asarray(v,np.float32) for k,v in arrays.items()})
    save(OUT/(kind+'.json'),record)


def paired_check():
    import numpy as np
    from .sparse_attention_check import TOLERANCES
    result={};tol=TOLERANCES['bf16']
    with np.load(OUT/'old_paired.npz') as a,np.load(OUT/'new_paired.npz') as b:
        for key in a.files:
            x,y=a[key],b[key];error=float(np.max(np.abs(x-y)));scale=float(np.max(np.abs(x)))
            result[key]={'max_abs':error,'reference_max_abs':scale,'relative_to_max':error/max(scale,1e-12)}
            atol,rtol=(tol['output_atol'],tol['output_rtol']) if key=='output_rows' else (tol['grad_atol'],tol['grad_rtol'])
            if not np.allclose(x,y,atol=atol,rtol=rtol):raise ValueError('PAIRED_MISMATCH:'+key)
    save(OUT/'paired_check.json',result);return result


def compact_full(full):
    w=full.get('worker',{});live=full.get('live',{})
    return {**{k:v for k,v in full.items() if k not in ['live','process','worker']},'process':{k:v for k,v in full['process'].items() if k!='samples'},'worker':{k:v for k,v in w.items() if k not in ['stages','trace','traceback']},'stages':[{k:v for k,v in s.items() if k not in ['trace','candidate_layers']} for s in w.get('stages',[])],'last_live':{k:v for k,v in live.items() if k not in ['stages','trace','candidate_build_evidence','forward_edge_evidence','traceback']}}


def report(result):
    lines=['# PVST v0.3 固定CSR values梯度桥接验收','',
        '2026-11-05；状态 **'+result['status']+'**；正式模型仍0/2，不启动训练或性能评价。',
        '## 修复与结构','',
        '原直接CSR构造的SparseCompressedTensorBackward0会_to_dense，再提取values梯度。新增CSRFromValues只封装固定结构构造：A[row_e,col_e]=v_e；其反向从真实CSR上游梯度返回对应边的grad_values。shape/crow/col均核对，不凭nnz假定顺序；CUDA结构核对用异步断言，避免每head主机同步。非空结构改变明确报错，不静默匹配或稠密化；本路径实测CSR结构完全一致。',
        'None梯度保留None；空CSR上游代表数学零梯度，明确返回原边上的零。支持空边/空行、单边、显式零值及无梯度输入，不把dropout后的零值边删除。没有改写SpMM、sampled_addmm或整套attention反向，只有CSR构造values的一阶桥接。',
        '现有CSRSoftmax保持不变；sampled_addmm输出仍直接可微values()，完整路径证明无需额外取值桥接。最终路径：CSR sampled_addmm → values+bias → CSRSoftmax → 原dropout → CSRFromValues → CSR sparse.mm。保留逐head FP32、外层BF16、SIGFPE无边填充列、全部邻接/节点/点/像素/宽度/PTS、初始化及损失。正式XYZ=null，候选仍默认关闭。',
        'CSRSoftmax仅保存p/crow；桥接仅保存crow/col引用和逻辑shape。临时行号及算术张量仍有成本；del局部变量不意味着autograd保存内容消失，节省以下列实测allocated为准。无显式COO构造，不承诺后端内部完全无稀疏布局转换。',
        '## 小图正确性与布局','',
        '沿用已有固定容差：FP32输出2e-5/2e-4，梯度2e-5/3e-3；BF16输出.04/.03，梯度.003/.08（atol/rtol）。桥接FP64对照1e-10/1e-9；gradcheck eps1e-6/atol1e-5/rtol1e-3。未事后放宽。',
        '桥接与SpMM：','```json',json.dumps(result.get('regression',{}).get('bridge'),ensure_ascii=False,indent=2),'```',
        '|精度/dim/dilation|输出max abs|输入梯度max abs|QKV梯度max abs|bias梯度max abs|投影梯度max abs|','|---|---:|---:|---:|---:|---:|']
    for c in result.get('regression',{}).get('comparisons',[]):
        p=c['parameters'];q=max(v['max_abs'] for k,v in p.items() if k.startswith('qkv.'));b=max(v['max_abs'] for k,v in p.items() if k.startswith('bias.'));proj=max(v['max_abs'] for k,v in p.items() if k.startswith('proj.'))
        lines.append(f'|{c["dtype"]}/{c["dim"]}/{c["dilation"]}|{c["output"]["max_abs"]:.8g}|{c["input_gradient"]["max_abs"]:.8g}|{q:.8g}|{b:.8g}|{proj:.8g}|')
    reg=result.get('regression',{})
    lines+=['','小图包含非均匀PTS、负坐标、不同帧节点/空帧、跨帧query与尾块；FP32覆盖三种dilation，BF16覆盖512/8head。TorchDispatch检查前向、完整反向及checkpoint重计算，拒绝_to_dense和逻辑N×N/N×(N+padding)张量，同时允许正常N×D特征。普通与强制逻辑padding两条小图均检查；没有重做旧SIGFPE定位。',
        '布局通过：'+str(reg.get('layout',{}).get('forward_backward_checkpoint_no_dense_logical'))+'；padding布局通过：'+str(reg.get('padded_layout',{}).get('forward_backward_checkpoint_no_dense_logical'))+'；dropout/checkpoint：'+json.dumps(reg.get('dropout'),ensure_ascii=False)+'。',
        '全部参数误差及相对参考最大幅度保留JSON；CPU直接相关检查见cpu_tests.txt。无新增模型参数/state_dict键。',
        '## 唯一规模单层内存与速度','',
        '实际131,072节点、dim512、8head、head_dim64，四帧真实已验证坐标图/非均匀PTS；旧COO候选仅作tests中的本次参考。两者独立进程、同seed/权重/输入/dropout.1/BF16/FP32/内外checkpoint，一次预热后各一次非profiler同步计时。',
        '|实现|实际N/E|前向秒|反向秒|起始allocated GiB|peak allocated/reserved GiB|','|---|---:|---:|---:|---:|---:|']
    for kind in ['old','new']:
        v=result.get(kind)
        if v:lines.append(f'|{kind}|{v["nodes"]}/{v["edges"]}|{v["forward_s"]:.6f}|{v["backward_s"]:.6f}|{v["baseline_allocated_bytes"]/2**30:.4f}|{v["peak_allocated_bytes"]/2**30:.4f}/{v["peak_reserved_bytes"]/2**30:.4f}|')
    lines+=['','大图一致性：全部参数梯度比较；输出及输入梯度比较固定每2048节点一行，共64行，不冒称大图逐元素全量。小图输出/输入/参数梯度均全量核对。配对误差：','```json',json.dumps(result.get('paired_check'),ensure_ascii=False,indent=2),'```',
        '放行门槛在执行前固定：peak allocated实际下降、反向≤旧1.25倍，且全部正确性/layout通过。小型GPU子进程累计墙时'+str(result.get('small_gpu_wall_s'))+'秒/290秒（为预先极小接口探针留约10秒预算，探针墙时未单列）。',
        '## 完整窗口','', '进入次数：'+str(int(result.get('full_window_entered',False)))+'；唯一900秒父进程限时，不重试。']
    full=result.get('full_window',{});w=full.get('worker',{});live=full.get('live',{})
    if full:
        lines+=['状态 **'+full['status']+'**；ANAKIN/train/real/an:4wdo37，video782426ea0330c7fea684953c；720×1280，历史[104,120)、目标[120,128)，fine1/8、coarse1/4。',
            '实际fine/coarse：'+str(w.get('fine_nodes'))+'/'+str(w.get('coarse_nodes'))+'；参数17,821,672及state_dict键hash由入口核对；8帧原生pixel输出、原frame BCE+AN real全零空间GT的pixel BCE/Dice损失，仅作执行检查。',
            '|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|当前allocated/reserved GiB|RSS/主机可用GiB|','|---|---:|---:|---:|---:|---:|']
        for s in w.get('stages',[]):lines.append(f'|{s["stage"]}|{s.get("stage_s")}|{s["elapsed_s"]:.3f}|{s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f}|{s["allocated_bytes"]/2**30:.3f}/{s["reserved_bytes"]/2**30:.3f}|{s["host"]["rss_bytes"]/2**30:.3f}/{s["host"]["available_bytes"]/2**30:.3f}|')
        lines+=['', '六层运行证据（全部通过才允许PASS）：','```json',json.dumps(w.get('forward_edge_evidence'),ensure_ascii=False,indent=2),'```',
            '梯度/更新/失败上下文：','```json',json.dumps({k:w.get(k) for k in ['gradients','optimizer_state_parameters','parameter_delta_norms','parameter_update_verified','failure_operator_context','phase','elapsed_s','host','peak_allocated_bytes','peak_reserved_bytes']},ensure_ascii=False,indent=2),'```',
            '实际更新次数：'+str(w.get('updates',0))+'；worker正常末态墙时：'+str(w.get('elapsed_s','未正常结束'))+'；父进程总墙时：'+str(full['process']['wall_s'])+'；returncode/signal：'+str(full['process']['returncode'])+'/'+str(full['process']['termination_signal'])+'。',
            'GPU开始/结束及释放：','```json',json.dumps({k:full.get(k) for k in ['gpu_start','gpu_after','worker_gpu_released']},ensure_ascii=False,indent=2),'```',
            '最后阶段：'+str(w.get('phase',live.get('phase')))+'。失败记录：','```',w.get('traceback',w.get('error','无；若超时，以父进程信号与live.json为准')),'```',
            '旧同窗口COO候选前向18.064秒、反向未完成，约65.56秒OOM、AdamW未执行，已观测allocated峰值21.25GiB。新旧都逐head FP32核心，但softmax计算次序不同，不保证训练逐位相同；不把旧失败时间当完整成本算精确加速比，不从单层直接外推整模型。']
    else:lines+=['未放行完整窗口；未测阶段不填0，详见门槛/错误。']
    if result['status']=='PASS':lines+=['','本窗口完成一次真实AdamW更新。是否值得进入两条件各一轮须结合本次完整成本与显存余量；只证明此窗口，不代表全队列或尺度冻结。本轮不自动训练。']
    else:lines+=['','不建议直接进入两条件各一轮。未达到完整更新放行条件；按本轮停止条件结束，不继续叠加优化或重试。已观测峰值不代表完整更新峰值上界。']
    lines+=['错误/门槛：'+str(result.get('reason','无')),'源码：`'+result['source_commit']+'`；产物：`'+str(OUT)+'`。缓存/前端/视觉特征均复用，不读测试集，不下载，不做坐标审计或预览；验收权重与优化器丢弃，无checkpoint。原报告、退出历史、正式config和用户无关JSON保持；计算与publication分列。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/csr_values_bridge_check_v1.md').write_text(text)
    compact={k:v for k,v in result.items() if k!='full_window'}
    if full:compact['full_window']=compact_full(full)
    save(OUT/'summary.json',compact);save(ROOT/'docs/pvst/csr_values_bridge_check_v1.json',compact)
    status=ROOT/'docs/pvst/current_status.md';status.write_text(status.read_text()+'\n2026-11-05 CSR values桥接限定验收：'+result['status']+'；正式模型仍0/2。见[报告](csr_values_bridge_check_v1.md)。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['regression','old','new']);a=p.parse_args()
    if a.worker:
        import torch
        torch.set_num_threads(4)
        if a.worker=='regression':regression()
        else:benchmark(a.worker)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse, never retry')
    spec={'source_commit':git('rev-parse','HEAD'),'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'tests/pvst_coo_reference.py',ROOT/'tests/test_pvst_csr_values_bridge.py']},'backward_ratio_limit':BACKWARD_RATIO_LIMIT,'small_gpu_limit_s':290}
    save(OUT/'identity.json',spec);result={'status':'FAILED','source_commit':spec['source_commit'],'full_window_entered':False,'small_gpu_wall_s':0.,'formal_models_completed':0,'processes':[]}
    try:
        for kind in ['regression','old','new']:
            folder=OUT/kind;folder.mkdir();process=bounded_process([sys.executable,'-u','-m','research_tools.pvst.csr_values_bridge_check','--worker',kind],max(0,290-result['small_gpu_wall_s']),folder/'worker.log');save(folder/'process.json',process);result['small_gpu_wall_s']+=process['wall_s'];result['processes'].append({'kind':kind,**process})
            if process['timed_out'] or process['returncode']!=0:raise RuntimeError('SMALL_CHECK_FAILED:'+kind+':'+str(process))
            result[kind]=read(OUT/(kind+'.json'));save(OUT/'partial.json',result)
        result['paired_check']=paired_check();old,new=result['old'],result['new']
        if new['peak_allocated_bytes']>=old['peak_allocated_bytes'] or new['backward_s']>old['backward_s']*BACKWARD_RATIO_LIMIT:raise RuntimeError('STOP_MEMORY_OR_BACKWARD_GATE')
        result['full_window_entered']=True;save(OUT/'partial.json',result)
        subprocess.run([sys.executable,'-u','-m','research_tools.pvst.max_window_check','--values-bridge'],check=True)
        result['full_window']=read(OUT/'full_window/result.json');result['status']=result['full_window']['status']
    except Exception as e:result['reason']=str(e);result['traceback']=traceback.format_exc()
    report(result);save(OUT/'progress.json',{'status':result['status'],'stage':'report_complete','formal_models_completed':0})
    files=['docs/pvst/csr_values_bridge_check_v1.md','docs/pvst/csr_values_bridge_check_v1.json','docs/pvst/current_status.md']
    try:
        subprocess.run(['git','add','--',*files],check=True);subprocess.run(['git','commit','-m','Record bounded CSR values bridge and full-window acceptance'],check=True)
        head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':result['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':result['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':result['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
