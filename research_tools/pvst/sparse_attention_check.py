"""Bounded single-layer gate; no training or automatic full-window retry."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .max_window_check import bounded_process,save,sha,git
BASE=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
OUT=BASE/'pvst_sparse_attention_check_v1'
# Frozen before tests. Relative scale is max-absolute error / max reference magnitude.
TOLERANCES={'fp32':{'output_atol':2e-5,'output_rtol':2e-4,'grad_atol':2e-5,'grad_rtol':3e-3},
             'bf16':{'output_atol':.04,'output_rtol':.03,'grad_atol':.003,'grad_rtol':.08}}
BACKWARD_RATIO_GATE=.8


def small_graph(device):
    import torch
    from .temporal import Neighborhood
    dense=torch.cartesian_prod(torch.arange(-1,2),torch.arange(-1,2),torch.arange(-1,2)).to(device)
    return Neighborhood([dense[:8],dense[:0],dense[::5],dense],[0.,.031,.10,.187])


def compare(dtype,dim,dilation,device):
    import torch
    from .temporal import SparseLayer
    from .sparse_attention import edge_topology
    torch.manual_seed(20260909);graph=small_graph(device);a=SparseLayer(dim,dim//64,7,11).to(device).train();a.qkv_chunk=7
    for mod in a.modules():
        if isinstance(mod,torch.nn.Dropout):mod.p=0
    b=copy.deepcopy(a);b.sparse_edge_attention=True
    assert list(a.state_dict())==list(b.state_dict()) and sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    base=torch.randn(len(graph.coords),dim,device=device,dtype=dtype);x=base.clone().requires_grad_();y=base.clone().requires_grad_()
    crow,col,relation=edge_topology(graph,dilation,7);reference,frame=graph.query(0,len(x),dilation)
    expected=[]
    for row in range(len(x)):
        slots=(reference[row]>=0).nonzero()[:,0]
        pairs=sorted([(int(reference[row,s]),int(frame[row])*81+int(s)) for s in slots])
        actual=list(zip(col[crow[row]:crow[row+1]].tolist(),relation[crow[row]:crow[row+1]].tolist()))
        assert pairs==actual and len({v[0] for v in actual})==len(actual)
    amp=dtype==torch.bfloat16
    with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):aa=a(x,graph,dilation);bb=b(y,graph,dilation)
    ga=torch.autograd.grad(aa.square().mean(),[x,*a.parameters()]);gb=torch.autograd.grad(bb.square().mean(),[y,*b.parameters()])
    def error(u,v):
        absolute=float((u-v).abs().max());scale=float(u.abs().max());return {'max_abs':absolute,'reference_max_abs':scale,'relative_to_max':absolute/max(scale,1e-12)}
    tol=TOLERANCES['bf16' if amp else 'fp32'];record={'dtype':str(dtype),'dim':dim,'dilation':dilation,'nodes':len(x),'edges':len(col),'output':error(aa,bb),'input_gradient':error(ga[0],gb[0]),'parameters':{n:error(u,v) for (n,p),u,v in zip(a.named_parameters(),ga[1:],gb[1:])}}
    torch.testing.assert_close(aa,bb,atol=tol['output_atol'],rtol=tol['output_rtol'])
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=tol['grad_atol'],rtol=tol['grad_rtol'])
    return record


def dropout_check(device):
    import torch
    from . import temporal
    graph=small_graph(device);a=temporal.SparseLayer(256,4,7,11).to(device).train();a.sparse_edge_attention=True;a.qkv_chunk=7;b=copy.deepcopy(a)
    base=torch.randn(len(graph.coords),256,device=device);x=base.clone().requires_grad_();y=base.clone().requires_grad_();original=temporal.checkpoint
    torch.manual_seed(71);aa=a(x,graph,1);aa.square().mean().backward()
    try:
        temporal.checkpoint=lambda function,*args,**kw:function(*args)
        torch.manual_seed(71);bb=b(y,graph,1);bb.square().mean().backward()
    finally:temporal.checkpoint=original
    torch.testing.assert_close(aa,bb,atol=2e-6,rtol=2e-5)
    for p,q in zip(a.parameters(),b.parameters()):
        assert torch.isfinite(p.grad).all();torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=3e-4)
    return {'dropout':.1,'max_output_error':float((aa-bb).abs().max()),'finite_gradients':True,'checkpoint_rng':'PASS'}


def benchmark(n,dim,candidate,profile=False):
    import torch
    from torch.utils.checkpoint import checkpoint
    from .temporal import Neighborhood,SparseLayer
    from .sparse_attention import edge_topology
    torch.manual_seed(20260909);side=16 if n==16384 else 32
    coords=torch.cartesian_prod(torch.arange(side),torch.arange(side),torch.arange(16)).cuda()
    tick=time.monotonic();graph=Neighborhood([coords]*4,[0.,.031,.10,.187]);torch.cuda.synchronize();compact_s=time.monotonic()-tick
    tick=time.monotonic();crow,col,relation=edge_topology(graph,1,512);torch.cuda.synchronize();edge_s=time.monotonic()-tick;edges=len(col);storage=sum(t.numel()*t.element_size() for t in [crow,col,relation]);del crow,col,relation
    model=SparseLayer(dim,dim//64,512,8192).cuda().train();model.qkv_chunk=8192;model.sparse_edge_attention=candidate
    x=torch.randn(n,dim,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    def run():
        model.zero_grad(set_to_none=True);x.grad=None
        with torch.autocast('cuda',dtype=torch.bfloat16):y=checkpoint(model,x,graph,1,use_reentrant=False,preserve_rng_state=True);loss=y.square().mean()
        return y,loss
    y,loss=run();loss.backward();torch.cuda.synchronize();del y,loss
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic();y,loss=run();torch.cuda.synchronize();forward=time.monotonic()-tick
    tick=time.monotonic();loss.backward();torch.cuda.synchronize();backward=time.monotonic()-tick
    result={'n':n,'dim':dim,'candidate':candidate,'dtype':'BF16; sparse core per-head FP32','dropout':.1,'outer_checkpoint':True,'edges':edges,'average_degree':edges/n,'topology_bytes':storage,'compact_topology_s':compact_s,'edge_topology_s':edge_s,'forward_s':forward,'backward_s':backward,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    del y,loss
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            y,loss=run();loss.backward();torch.cuda.synchronize()
        keys=prof.key_averages(group_by_input_shape=True);result['profiler']=[{'key':e.key,'count':e.count,'input_shapes':e.input_shapes,'cpu_total_us':e.cpu_time_total,'device_total_us':e.device_time_total,'self_device_memory_bytes':e.self_device_memory_usage} for e in keys if any(s in e.key.lower() for s in ['indexbackward','index_backward','new_zeros','zero_','sparse','slice','tocsr','toco','gather'])]
        result['profiler_table']=keys.table(sort_by='self_cuda_time_total',row_limit=18)
    return result


def worker():
    import torch
    torch.set_num_threads(4);result={'torch':torch.__version__,'tolerances':TOLERANCES,'backward_ratio_gate':BACKWARD_RATIO_GATE,'correctness':[],'benchmarks':[],'full_window_allowed':False}
    try:
        for dtype in [torch.float32,torch.bfloat16]:
            for dim in [256,512]:
                for dilation in [1,2,4]:
                    result['correctness'].append(compare(dtype,dim,dilation,'cuda'));save(OUT/'layer_results.json',result)
        result['dropout']=dropout_check('cuda');save(OUT/'layer_results.json',result)
        # First medium case is a stop gate, not a searched choice.
        for n,dim in [(16384,256),(16384,512),(65536,256),(65536,512)]:
            pair=[]
            for candidate in [False,True]:
                row=benchmark(n,dim,candidate,profile=n==16384 and dim==256);result['benchmarks'].append(row);pair.append(row);save(OUT/'layer_results.json',result)
            ratio=pair[1]['backward_s']/pair[0]['backward_s'];print('BENCHMARK',n,dim,ratio,flush=True)
            if ratio>BACKWARD_RATIO_GATE:
                result.update(status='STOP_PERFORMANCE',reason='candidate backward ratio exceeds predeclared 0.8 gate');save(OUT/'layer_results.json',result);return
        result.update(status='LAYER_PASS',full_window_allowed=True)
    except BaseException as e:result.update(status='FAILED',error=str(e),traceback=traceback.format_exc())
    save(OUT/'layer_results.json',result)



def finalize_layer(result,process):
    """Native signal cannot leave a partial success looking like a passed gate."""
    result={**result,'process':process}
    if process['timed_out']:result.update(status='TIMEOUT',full_window_allowed=False)
    elif process['returncode']!=0:result.update(status='FAILED',full_window_allowed=False,error='native worker exit '+str(process['returncode'])+'; no Python exception captured')
    elif result.get('status')!='LAYER_PASS':result['full_window_allowed']=False
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    if (OUT/'layer_results.json').exists():raise RuntimeError('Existing layer results: inspect/reuse; no automatic repeat')
    spec={'source_commit':git('rev-parse','HEAD'),'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'tolerances':TOLERANCES,'backward_ratio_gate':BACKWARD_RATIO_GATE}
    save(OUT/'layer_identity.json',spec)
    proc=bounded_process([sys.executable,'-u','-m','research_tools.pvst.sparse_attention_check','--worker'],290,OUT/'layer.log');save(OUT/'layer_process.json',proc)
    result=json.loads((OUT/'layer_results.json').read_text()) if (OUT/'layer_results.json').exists() else {}
    result=finalize_layer(result,proc);save(OUT/'layer_results.json',result)
    save(OUT/'progress.json',{'status':result.get('status','FAILED'),'stage':'layer_check_finished','formal_models_completed':0});print(result.get('status','FAILED'),flush=True)

if __name__=='__main__':main()
