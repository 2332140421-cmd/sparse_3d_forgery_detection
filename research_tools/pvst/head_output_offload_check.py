"""Single storage-only comparison, then one existing 900s resource window.

This CLI reuses the CSR benchmark, parity snapshots and max_window worker.
Only head output placement differs; it never calls the old COO reference.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from .max_window_check import BASE,ROOT,bounded_process,read,save,sha,git
from .csr_values_bridge_check import load_test,benchmark,paired_check,compact_full
OUT=BASE/'pvst_head_output_offload_check_v1'


def regression():
    import torch
    compare=load_test('test_pvst_head_output_offload.py').compare_offload
    result={'status':'PASS','comparisons':[]}
    for dtype,dim,drop in [(torch.float32,256,False),(torch.float32,256,True),(torch.bfloat16,512,True)]:result['comparisons'].append(compare('cuda',dtype,dim,drop))
    save(OUT/'regression.json',result)


def report(result):
    lines=['# PVST v0.3 逐head输出CPU暂存资源验收','',
        '状态：**'+result['status']+'**；正式模型仍0/2。本轮只有存储路径变化，无训练、评价或前端/冻结视觉提取。',
        '新增默认关闭head_output_offload：GPU逐head原CSRSoftmax/CSRFromValues/SpMM及checkpoint → 保留autograd的同步to(cpu) → CPU按原head顺序stack/reshape → 完整attended一次同步回GPU。dtype不变，不用detach/no_grad，不改QKV/索引/初始化/损失/宽度/像素/点/节点/边/上下文。CPU列表没有GPU输出引用；成员变量只记标量/字符串，不记Tensor。',
        '原已验证外层saved-tensor CPU卸载、BF16/逐head FP32、SIGFPE padding、dropout和checkpoint RNG保持。正式config XYZ=null且无新开关字段，正式默认False。',
        '## 正确性','',
        '沿用FP32输出atol2e-5/rtol2e-4、梯度atol2e-5/rtol3e-3；BF16输出.04/.03、梯度.003/.08，不事后放宽。小图含非均匀PTS、空帧、负坐标和尾块，当前CSR＋桥接GPU拼接与CPU暂存逐项全量对比。',
        '|dtype/宽度/dropout|输出max abs|输入梯度max abs|全部参数梯度最大误差|','|---|---:|---:|---:|']
    for c in result.get('regression',{}).get('comparisons',[]):lines.append(f'|{c["dtype"]}/{c["dim"]}/{c["dropout"]}|{c["output"]["max_abs"]}|{c["gradients"]["input"]["max_abs"]}|{max(v["max_abs"] for k,v in c["gradients"].items() if k!="input")}|')
    lines+=['','参数量/state_dict键与权重保持；CPU测试核对head顺序/reshape、默认关闭与全层开关接入。CUDA检查CPU拷贝requires_grad=True、ToCopyBackward0、原dtype，输入/QKV/bias/投影/FFN全部梯度存在且正确；dropout开启时内外checkpoint复算与RNG一致。TorchDispatch拒绝_to_dense及逻辑N×N矩阵。',
        '## 匹配单层','',
        '同已验证131072节点/7475256边、512维/8head/64维head图与PTS。GPU拼接和CPU暂存使用相同CSR＋桥接、seed/权重/输入/dropout.1/BF16/内外checkpoint；独立进程、各一次预热与同步计时。没有使用旧COO作为对照，没有多规模或chunk搜索。',
        '|路径|前向秒|反向秒|peak allocated/reserved GiB|测量期峰值RSS GiB|实际CPU head payload MiB|stack设备|','|---|---:|---:|---:|---:|---:|---|']
    for name in ['gpu','cpu']:
        v=result.get(name)
        if v:
            st=v['forward_head_storage'];lines.append(f'|{name}|{v["forward_s"]:.6f}|{v["backward_s"]:.6f}|{v["peak_allocated_bytes"]/2**30:.4f}/{v["peak_reserved_bytes"]/2**30:.4f}|{v["measurement_host"]["peak_rss_bytes"]/2**30:.4f}|{st["cpu_bytes"]/2**20:.1f}|{st["stack_device"]}|')
    lines+=['','head_storage中CPU字节数为实际拷贝Tensor的numel×element_size累计，不能与CPU进程峰值RSS等同；CPU heads和CPU stack曾短暂同时存在。D2H/CPU stack/H2D耗时、每head后allocated/reserved和回传前后内存保留JSON，只记数值，不增加额外同步或逐head落盘。',
        '大图配对核对全部参数梯度，输出/输入梯度取固定每2048节点一行共64行；小图逐元素全量，未将大图抽样称全量。配对误差：','```json',json.dumps(result.get('paired_check'),ensure_ascii=False,indent=2),'```',
        '放行要求：正确性/layout及配对一致、真实CPU列表/CPU stack/dtype/梯度连接成立，无OOM；测量期峰值RSS相对GPU路径增加不超过预先记录1GiB安全界（payload128MiB、CPU stack另128MiB），超出则停止。若整层allocated峰值未下降，结合每head与拼接阶段解释，不以.cpu()调用冒充节省，也不另加其他优化。',
        '小型GPU子进程累计墙时：'+str(result.get('small_gpu_wall_s'))+'秒，限300秒。',
        '## 唯一完整窗口','', '进入次数：'+str(int(result.get('full_window_entered',False)))+'；现有父进程900秒限时，不重试。']
    full=result.get('full_window',{});w=full.get('worker',{});live=full.get('live',{})
    if full:
        lines+=['状态 **'+full['status']+'**。XYZ_PVST，ANAKIN/train/real/an:4wdo37，video782426ea0330c7fea684953c；720×1280；history[104,120)、target[120,128)，fine1/8/coarse1/4；实际fine/coarse='+str(w.get('fine_nodes'))+'/'+str(w.get('coarse_nodes'))+'。',
            '配置seed20260909/BF16/逐head FP32；attention512/QKV8192/FFN8192/point32768/tile256；外层saved-tensor CPU卸载及head_output_offload均True。参数17,821,672与state_dict键hash核对；原8目标帧均值frame BCE和本ANAKIN real合法全零GT pixel BCE/softDice。只用于执行检查，不报告效果指标。',
            '|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|当前allocated/reserved GiB|RSS/主机可用 GiB|','|---|---:|---:|---:|---:|---:|']
        for s in w.get('stages',[]):lines.append(f'|{s["stage"]}|{s.get("stage_s")}|{s["elapsed_s"]:.3f}|{s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f}|{s["allocated_bytes"]/2**30:.3f}/{s["reserved_bytes"]/2**30:.3f}|{s["host"]["rss_bytes"]/2**30:.3f}/{s["host"]["available_bytes"]/2**30:.3f}|')
        lines+=['','|层|dilation|N/E|CPU head payload GiB|stack设备|D2H/stack/H2D秒|回GPU前后allocated GiB|','|---|---:|---:|---:|---|---:|---:|']
        for row in w.get('forward_edge_evidence',[]):
            st=row['head_storage'];lines.append(f'|{row["layer"]}|{row["dilation"]}|{row["nodes"]}/{row["edges"]}|{st["cpu_bytes"]/2**30:.4f}|{st["stack_device"]}|{st["copy_to_cpu_s"]:.4f}/{st["stack_s"]:.4f}/{st["return_to_gpu_s"]:.4f}|{st["before_gpu_return"].get("allocated_bytes",0)/2**30:.4f}/{st["after_gpu_return"].get("allocated_bytes",0)/2**30:.4f}|')
        lines+=['','梯度、更新及失败上下文：','```json',json.dumps({k:w.get(k) for k in ['gradients','parameter_update_verified','optimizer_state_parameters','parameter_delta_norms','failure_operator_context','error','phase','elapsed_s','host','peak_allocated_bytes','peak_reserved_bytes']},ensure_ascii=False,indent=2),'```',
            '更新次数='+str(w.get('updates',0))+'；父进程总墙时='+str(full['process']['wall_s'])+'秒；returncode/signal='+str(full['process']['returncode'])+'/'+str(full['process']['termination_signal'])+'；GPU进程释放='+str(full.get('worker_gpu_released'))+'。',
            '最后阶段='+str(w.get('phase',live.get('phase')))+'。未完成阶段不填0；失败已观测峰值不当作完整更新峰值上界。完整错误和最后层/shape见full_window/worker_result.json、live.json与worker.log，5秒GPU采样/曲线保留独立数据目录。',
            '上次当前CSR＋桥接同窗口前向16.768秒、反向coarse dilation4第7head SpMM重算OOM，申请364MiB时剩298.12MiB；allocated/reserved峰值21.174/21.818GiB，AdamW0。旧反向未完成，不计算精确完整加速比；head输出1.06GiB等先验形状数不冒充实测释放。']
    else:lines+=['未运行完整窗口；停止原因见下，不以单层通过宣称完整模型可训练。']
    lines+=['','## 结论','']
    if result['status']=='PASS':lines+=['该规定窗口完成一次真实前向、反向、梯度裁剪和AdamW更新；优化器状态与参数改变均核对。结合本次传输/总耗时与显存余量，可判断是否值得由用户另行授权两条件各一轮，但不声称全部样本验收。本轮不自动启动训练。']
    else:lines+=['未完成完整更新，不建议直接进入两条件各一轮。按本轮限定停止，不自动改SpMM、索引类型、尺度或其他存储，不追加训练或资源测试。']
    lines+=['错误/门槛：'+str(result.get('reason','无')),'执行源码：`'+result['source_commit']+'`。本轮产物：`'+str(OUT)+'`。原报告/错误/退出码/协议/config和缓存不覆盖；用户无关JSON保留。验收权重丢弃，不保存checkpoint；计算状态与publication.json单列。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/head_output_offload_check_v1.md').write_text(text)
    summary={k:v for k,v in result.items() if k!='full_window'}
    if full:summary['full_window']=compact_full(full)
    save(OUT/'summary.json',summary);save(ROOT/'docs/pvst/head_output_offload_check_v1.json',summary)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+'\n逐head输出CPU暂存限定验收：'+result['status']+'；正式模型仍0/2。见[报告](head_output_offload_check_v1.md)。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['regression','gpu','cpu']);a=p.parse_args()
    if a.worker:
        import torch
        torch.set_num_threads(4)
        if a.worker=='regression':regression()
        else:benchmark(a.worker,head_output_offload=a.worker=='cpu',out=OUT)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse; no repeat')
    source=git('rev-parse','HEAD');save(OUT/'identity.json',{'source_commit':source,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'tests/test_pvst_head_output_offload.py']},'host_RSS_increase_limit_bytes':2**30})
    result={'status':'FAILED','source_commit':source,'small_gpu_wall_s':0.,'full_window_entered':False,'formal_models_completed':0,'processes':[]}
    try:
        for name in ['regression','gpu','cpu']:
            folder=OUT/name;folder.mkdir();proc=bounded_process([sys.executable,'-u','-m','research_tools.pvst.head_output_offload_check','--worker',name],max(0,300-result['small_gpu_wall_s']),folder/'worker.log');save(folder/'process.json',proc);result['small_gpu_wall_s']+=proc['wall_s'];result['processes'].append({'name':name,**proc})
            if proc['timed_out']:result['status']='TIMEOUT';raise RuntimeError('SMALL_CHECK_TIMEOUT:'+name)
            if proc['returncode']!=0:raise RuntimeError('SMALL_CHECK_FAILED:'+name)
            result[name]=read(OUT/(name+'.json'));save(OUT/'partial.json',result)
        result['paired_check']=paired_check(out=OUT,names=('gpu','cpu'))
        st=result['cpu']['forward_head_storage']
        if st['cpu_bytes']!=131072*512*2 or st['gpu_list_bytes']!=0 or st['stack_device']!='cpu':raise ValueError('OFFLOAD_STORAGE_NOT_REAL')
        if result['cpu']['measurement_host']['peak_rss_bytes']-result['gpu']['measurement_host']['peak_rss_bytes']>2**30:raise ValueError('ABNORMAL_HOST_RSS_INCREASE')
        result['full_window_entered']=True;save(OUT/'partial.json',result)
        subprocess.run([sys.executable,'-u','-m','research_tools.pvst.max_window_check','--head-output-offload'],check=True)
        result['full_window']=read(OUT/'full_window/result.json');result['status']=result['full_window']['status']
    except Exception as error:result['reason']=str(error);result['traceback']=traceback.format_exc()
    report(result);save(OUT/'progress.json',{'status':result['status'],'stage':'report_complete','formal_models_completed':0})
    try:
        subprocess.run(['git','add','--','docs/pvst/head_output_offload_check_v1.md','docs/pvst/head_output_offload_check_v1.json','docs/pvst/current_status.md'],check=True);subprocess.run(['git','commit','-m','Record head output CPU storage acceptance'],check=True)
        head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':result['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':result['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':result['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
