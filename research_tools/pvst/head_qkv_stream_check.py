"""One head-QKV execution check, matched layer and existing bounded window."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from .max_window_check import BASE,ROOT,bounded_process,read,save,sha,git
from .csr_values_bridge_check import load_test,benchmark,paired_check,compact_full
OUT=BASE/'pvst_head_qkv_stream_check_v1'


def regression():
    import torch
    compare=load_test('test_pvst_head_qkv_stream.py').compare_stream
    result={'status':'PASS','comparisons':[]}
    for dtype in [torch.float32,torch.bfloat16]:
        for dim in [256,512]:result['comparisons'].append(compare('cuda',dtype,dim,True))
    save(OUT/'regression.json',result)


def compact(result):
    s={k:v for k,v in result.items() if k!='full_window'}
    # Keep scalar/error/shape proof, leave repetitive per-head copy samples in raw data.
    s=json.loads(json.dumps(s))
    for name in ['full','stream']:
        if name in s:
            for key in ['forward_head_storage','backward_head_storage']:
                st=s[name][key];heads=st.pop('heads',[]);st['stored_head_count']=len(heads);st['all_CPU']=all(h['stored_device']=='cpu' for h in heads)
    if result.get('full_window'):
        s['full_window']=compact_full(result['full_window'])
        # compact_full retains raw dictionaries: copy before reducing report data.
        s['full_window']=json.loads(json.dumps(s['full_window']))
        for row in s['full_window']['worker'].get('forward_edge_evidence',[]):
            st=row['head_storage'];heads=st.pop('heads',[]);st['stored_head_count']=len(heads);st['all_CPU']=all(h['stored_device']=='cpu' for h in heads)
    return s


def report(r):
    lines=['# PVST v0.3 逐head即时QKV执行验收','', '**状态：'+r['status']+'；正式模型0/2。** 无正式训练、前端提取或效果评价。',
        '实现：保留原qkv Linear及参数身份，当前head可微选择Q/K/V各64行，合为192行权重；每8192节点仍对完整D维执行原FP32累积LayerNorm，再F.linear。当前head分块归一化/投影和CSR attention处于同一非重入checkpoint，节点块另用原checkpoint；head作为显式参数传入，重计算不读取循环尾值。',
        '只物化当前[N,192]，不在head间缓存QKV、不复制完整x、不创建新的Parameter。当前head输出沿用同步CPU暂存/CPU stack/完整attended回GPU，保留CSRSoftmax、CSRFromValues、SIGFPE填充列、BF16外层/逐head FP32、dropout及外层saved-tensor CPU卸载。新开关默认False，只允许CSR＋CPU head路径，不静默回退；正式XYZ仍null。',
        '## 正确性','', '沿用FP32输出atol2e-5/rtol2e-4、梯度2e-5/3e-3，BF16输出.04/.03、梯度.003/.08，不放宽。小图全量比较输出/输入/全部参数梯度；参数身份、键和数量不变；各head原qkv参数行梯度范数非零。CPU单独检查完整D LayerNorm及Q/K/V选行/排列、尾块、关闭路径与默认值。',
        '|精度/宽度/dropout|输出max abs|输入梯度max abs|全部参数梯度max abs|候选[N,3D]事件数|','|---|---:|---:|---:|---:|']
    for c in r.get('regression',{}).get('comparisons',[]):lines.append(f'|{c["dtype"]}/{c["dim"]}/{c["dropout"]}|{c["output"]["max_abs"]:.8g}|{c["gradients"]["input"]["max_abs"]:.8g}|{max(v["max_abs"] for k,v in c["gradients"].items() if k!="input"):.8g}|{c["shape_evidence"][1]["full_shape_events"]}|')
    lines+=['', 'TorchDispatch覆盖前向、完整反向、内外checkpoint重算，禁止[N,3D]/[N,3,H,64]激活、_to_dense和逻辑N×N；允许正常N×D特征与[3D,D]参数。小图含负坐标、空帧、非均匀PTS和尾块。实际最大绝对/相对参考尺度及逐参数误差保留JSON。',
        '## 匹配单层','', '两组均当前CSR＋桥接＋CPU head输出暂存，仅head_qkv_stream不同；实际N131072/E7475256、dim512/8head/64。同seed/输入/权重/dropout.1/BF16/checkpoint，独立进程各一次预热后同步计时，无profiler/多规模搜索。',
        '|路径|前向秒|反向秒|peak allocated/reserved GiB|测量峰值RSS GiB|实际QKV形状/dtype|','|---|---:|---:|---:|---:|---|']
    for name in ['full','stream']:
        b=r.get(name)
        if b:
            q=b['forward_qkv_execution'];shape=q.get('full_shape') if name=='full' else q.get('heads',{}).get('0',{}).get('shape');dtype=q.get('dtype') if name=='full' else q.get('heads',{}).get('0',{}).get('dtype')
            lines.append(f'|{name}|{b["forward_s"]:.6f}|{b["backward_s"]:.6f}|{b["peak_allocated_bytes"]/2**30:.4f}/{b["peak_reserved_bytes"]/2**30:.4f}|{b["measurement_host"]["peak_rss_bytes"]/2**30:.4f}|{shape}/{dtype}|')
    lines+=['', '大图配对：全部参数梯度；输出/输入梯度固定每2048节点一行64行，不称全量输出比较。小图均逐元素核验。QKV实际形状/字节、前向与重算记录、起始allocated及主机资源保留机器结果。小型GPU累计墙时='+str(r.get('small_gpu_wall_s'))+'秒/300秒。',
        '## 唯一完整窗口','', '进入次数='+str(int(r.get('full_window_entered',False)))+'；现有父进程900秒worker限时，不重试。']
    f=r.get('full_window',{});w=f.get('worker',{});live=f.get('live',{})
    if f:
        lines+=['状态 **'+f['status']+'**。XYZ_PVST、ANAKIN/train/real/an:4wdo37、video782426ea0330c7fea684953c，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4。实际fine/coarse='+str(w.get('fine_nodes'))+'/'+str(w.get('coarse_nodes'))+'。',
            'seed20260909；原AdamW/初始化/真实8帧均值frame BCE、AN real合法全零GT pixel BCE/softDice；attention512/QKV8192/FFN8192/point32768/tile256；CPU saved-tensor卸载、CPU head暂存和head_qkv_stream均True，BF16/逐head FP32保持。17,821,672参数及原state_dict键hash由worker核对。',
            '|完成阶段|阶段秒|累计秒|peak allocated/reserved GiB|RSS/主机可用 GiB|','|---|---:|---:|---:|---:|']
        for st in w.get('stages',[]):lines.append(f'|{st["stage"]}|{st.get("stage_s")}|{st["elapsed_s"]:.3f}|{st["peak_allocated_bytes"]/2**30:.3f}/{st["peak_reserved_bytes"]/2**30:.3f}|{st["host"]["rss_bytes"]/2**30:.3f}/{st["host"]["available_bytes"]/2**30:.3f}|')
        lines+=['','|层|N/E|完成head|当前head QKV形状/bytes|完整[N,3D]|CPU head payload GiB|','|---|---:|---:|---|---|---:|']
        for row in w.get('forward_edge_evidence',[]):
            q=row['qkv_execution'];h=q['heads']['0'];lines.append(f'|{row["layer"]}|{row["nodes"]}/{row["edges"]}|{row["completed_heads"]}|{h["shape"]}/{h["bytes"]}|{q["full_activation_materialized"]}|{row["head_storage"]["cpu_bytes"]/2**30:.4f}|')
        lines+=['','更新次数='+str(w.get('updates',0))+'；真实参数改变='+str(w.get('parameter_update_verified',False))+'；AdamW状态参数数='+str(w.get('optimizer_state_parameters','未到达'))+'。',
            'worker至末态='+str(w.get('elapsed_s','未正常结束'))+'秒，父进程总墙时='+str(f['process']['wall_s'])+'秒；returncode/signal='+str(f['process']['returncode'])+'/'+str(f['process']['termination_signal'])+'；GPU释放='+str(f.get('worker_gpu_released'))+'。',
            '末态资源/关键梯度/失败上下文：','```json',json.dumps({k:w.get(k) for k in ['phase','peak_allocated_bytes','peak_reserved_bytes','host','gradients','parameter_delta_norms','failure_operator_context','error']},ensure_ascii=False,indent=2),'```',
            '最后阶段='+str(w.get('phase',live.get('phase')))+'；完整调用栈、当前层/head/QKV shape及5秒采样保留full_window/worker_result.json、live.json、worker.log。未完成耗时不填0；失败时已观测峰值不代表完整更新峰值上界。',
            '上轮全量QKV＋CPU head暂存同窗口：前向20.641秒，反向coarse第三层QKV cat重算OOM，申请4.25GiB时余625.31MiB；峰值allocated/reserved20.876/21.811GiB，AdamW0。旧反向未完成，不计算精确完整加速比，不从单层直接外推整模型。']
    if r['status']=='PASS':lines+=['','结论：规定最大窗口完成一次真实更新。仅此窗口通过，结合实际成本与余量判断4090执行代价，不宣称全队列已验收；本轮不自动训练。']
    else:lines+=['','结论：尚未完成规定窗口一次真实更新，不建议直接训练；按本轮停止条件结束，不追加SpMM/索引/尺度/其他优化。']
    lines+=['错误/门槛：'+str(r.get('reason','无')),'源码：`'+r['source_commit']+'`；产物：`'+str(OUT)+'`。冻结缓存复用，无下载/前端/视觉重提取、坐标审计/预览或效果指标，验收权重丢弃无checkpoint。原报告、协议、退出码、config和用户无关JSON保留，正式模型0/2；计算与发布分列。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/head_qkv_stream_check_v1.md').write_text(text)
    summary=compact(r);save(OUT/'summary.json',summary);save(ROOT/'docs/pvst/head_qkv_stream_check_v1.json',summary)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+'\n逐head即时QKV限定验收：'+r['status']+'；正式模型0/2。见[报告](head_qkv_stream_check_v1.md)。\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['regression','full','stream']);a=p.parse_args()
    if a.worker:
        import torch
        torch.set_num_threads(4)
        if a.worker=='regression':regression()
        else:benchmark(a.worker,head_output_offload=True,out=OUT,head_qkv_stream=a.worker=='stream')
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: inspect/reuse; no repeat')
    source=git('rev-parse','HEAD');save(OUT/'identity.json',{'source_commit':source,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'tests/test_pvst_head_qkv_stream.py']}})
    r={'status':'FAILED','source_commit':source,'full_window_entered':False,'small_gpu_wall_s':0.,'formal_models_completed':0,'processes':[]}
    try:
        for name in ['regression','full','stream']:
            folder=OUT/name;folder.mkdir();proc=bounded_process([sys.executable,'-u','-m','research_tools.pvst.head_qkv_stream_check','--worker',name],max(0,300-r['small_gpu_wall_s']),folder/'worker.log');save(folder/'process.json',proc);r['small_gpu_wall_s']+=proc['wall_s'];r['processes'].append({'name':name,**proc})
            if proc['timed_out']:r['status']='TIMEOUT';raise RuntimeError('SMALL_CHECK_TIMEOUT:'+name)
            if proc['returncode']!=0:raise RuntimeError('SMALL_CHECK_FAILED:'+name)
            r[name]=read(OUT/(name+'.json'));save(OUT/'partial.json',r)
        r['paired_check']=paired_check(out=OUT,names=('full','stream'))
        q=r['stream']['forward_qkv_execution']
        if q.get('full_activation_materialized') or set(q.get('heads',{}))!={str(i) for i in range(8)} or any(h['shape']!=[131072,192] for h in q['heads'].values()):raise ValueError('STREAM_STORAGE_NOT_ELIMINATED')
        r['full_window_entered']=True;save(OUT/'partial.json',r)
        subprocess.run([sys.executable,'-u','-m','research_tools.pvst.max_window_check','--head-qkv-stream'],check=True)
        r['full_window']=read(OUT/'full_window/result.json');r['status']=r['full_window']['status']
    except Exception as error:r['reason']=str(error);r['traceback']=traceback.format_exc()
    report(r);save(OUT/'progress.json',{'status':r['status'],'stage':'report_complete','formal_models_completed':0})
    try:
        subprocess.run(['git','add','--','docs/pvst/head_qkv_stream_check_v1.md','docs/pvst/head_qkv_stream_check_v1.json','docs/pvst/current_status.md'],check=True);subprocess.run(['git','commit','-m','Record head-streamed QKV maximum-window acceptance'],check=True)
        head=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':head,'calculation_status':r['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote','origin','refs/heads/pvst-point-voxel-v03').split()[0]
        if head!=remote:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'calculation_status':r['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':r['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
