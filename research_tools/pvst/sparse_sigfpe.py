"""Single-purpose SIGFPE reproducer, owned gdb child and shared 300s budget.

Immediate caller this CLI; stage JSONL/native stack survive native termination.
Reuse original benchmark graph/model/checkpoints; never call its full test suite.
"""
import argparse
import faulthandler
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .max_window_check import bounded_process,save,sha,git
OUT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_sparse_attention_sigfpe_v1')


def make_graph(n,device):
    import torch
    from .temporal import Neighborhood
    shapes={65536:(32,32,16),131072:(32,32,32)}
    dims=shapes[n];coords=torch.cartesian_prod(*(torch.arange(x,device=device) for x in dims));graph=Neighborhood([coords]*4,[0.,.031,.10,.187])
    if len(graph.coords)!=n:raise ValueError('ACTUAL_NODE_COUNT_MISMATCH')
    return graph


def validate_graph(graph):
    import torch
    from .sparse_attention import edge_topology
    crow,col,relation=edge_topology(graph,1,512);n=len(graph.coords);degree=crow.diff();rows=torch.repeat_interleave(torch.arange(n,device=col.device),degree)
    checks={'rowptr_length':len(crow)==n+1,'starts_zero':int(crow[0])==0,'monotone':bool((degree>=0).all()),'ends_nnz':int(crow[-1])==len(col),'col_range':bool(((col>=0)&(col<n)).all()),'sorted_unique':bool(((rows[1:]!=rows[:-1])|(col[1:]>col[:-1])).all()),'relation_range':bool(((relation>=0)&(relation<4*81)).all()),'relation_frame':bool((relation//81==torch.bucketize(rows,graph.boundaries,right=True)).all())}
    if not all(checks.values()):raise ValueError('CSR_INVALID:'+str(checks))
    if n==65536 and len(col)!=3658104:raise ValueError('ACTUAL_EDGE_COUNT_MISMATCH')
    return {'nodes':n,'edges':len(col),'checks':checks,'crow_dtype':str(crow.dtype),'col_dtype':str(col.dtype),'relation_dtype':str(relation.dtype),'logical_elements':n*n}


def diagnose():
    import torch
    from torch.utils.checkpoint import checkpoint
    from .temporal import SparseLayer
    from . import sparse_attention
    faulthandler.enable(all_threads=True);torch.set_num_threads(4);torch.manual_seed(20260909);tick=time.monotonic();phase='build'
    f=(OUT/'diagnostic_stages.jsonl').open('a',buffering=1)
    def log(stage,tensor=None,head=None):
        torch.cuda.synchronize();d={'stage':stage,'phase':phase,'head':head,'elapsed_s':time.monotonic()-tick}
        if tensor is not None:d.update(shape=list(tensor.shape),dtype=str(tensor.dtype),layout=str(tensor.layout),stride=list(tensor.stride()) if tensor.layout==torch.strided else None)
        f.write(json.dumps(d)+'\n');f.flush()
        if tensor is not None and tensor.grad_fn is not None and stage in ['sampled_addmm_complete','CSR_to_COO_complete','softmax_complete','COO_to_CSR_complete','sparse_mm_complete']:
            name=stage;h=head
            tensor.grad_fn.register_prehook(lambda grad,name=name,h=h:log('backward_pre:'+name,head=h))
            tensor.grad_fn.register_hook(lambda grad_in,grad_out,name=name,h=h:log('backward_post:'+name,head=h))
    original=sparse_attention.sparse_head;sparse_attention.sparse_head=functools.partial(original,trace=log)
    graph=make_graph(65536,'cuda');record=validate_graph(graph);save(OUT/'diagnostic_graph.json',record);log('graph_complete')
    model=SparseLayer(256,4,512,8192).cuda().train();model.qkv_chunk=8192;model.sparse_edge_attention=True
    model.qkv.register_forward_hook(lambda mod,args,y:log('norm_QKV_chunk_complete',y));x=torch.randn(65536,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    for iteration in ['warmup','measurement']:
        phase=iteration;model.zero_grad(set_to_none=True);x.grad=None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            y=checkpoint(model,x,graph,1,use_reentrant=False,preserve_rng_state=True);loss=y.square().mean()
        log('layer_forward_complete',y);log('backward_start');loss.backward();log('backward_complete');del y,loss
    save(OUT/'diagnostic_result.json',{'status':'NOT_REPRODUCED','graph':record});f.close()


def run_diagnostic():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'diagnostic_process.json').exists():raise RuntimeError('Existing diagnostic: reuse evidence, no retry')
    if not shutil.which('gdb'):raise RuntimeError('Installed gdb missing')
    save(OUT/'diagnostic_identity.json',{'source_commit':git('rev-parse','HEAD'),'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
    command=['gdb','-q','--batch','--return-child-result','-ex','set pagination off','-ex','set confirm off','-ex','set disable-randomization off','-ex','handle SIGFPE stop print pass','-ex','run','-ex','bt 24','-ex','handle SIGFPE nostop print pass','-ex','continue','--args',sys.executable,'-u','-m','research_tools.pvst.sparse_sigfpe','--diagnostic-worker']
    process=bounded_process(command,90,OUT/'native_stack.log');save(OUT/'diagnostic_process.json',process);save(OUT/'budget.json',{'limit_s':300,'spent_worker_wall_s':process['wall_s']})
    save(OUT/'progress.json',{'status':'DIAGNOSTIC_FINISHED','process':process,'formal_models_completed':0})



def minimal_worker():
    import torch
    faulthandler.enable(all_threads=True)
    index=torch.tensor([[0],[0]],device='cuda')
    a=torch.sparse_coo_tensor(index,torch.ones(1,device='cuda'),(65536,65536)).coalesce()
    b=torch.sparse_coo_tensor(index,torch.full((1,),2.,device='cuda'),(65536,65536)).coalesce()
    print('MINIMAL: CUDA sparse COO mul, N65536, E1 each, both coalesced, int64',flush=True)
    y=a*b;torch.cuda.synchronize();print('MINIMAL COMPLETE',y.values(),flush=True)


def run_minimal():
    stage=OUT/'minimal';stage.mkdir(exist_ok=True)
    if (stage/'process.json').exists():raise RuntimeError('Minimal evidence exists; do not repeat')
    budget=json.loads((OUT/'budget.json').read_text());remaining=budget['limit_s']-budget['spent_worker_wall_s']
    process=bounded_process([sys.executable,'-u','-m','research_tools.pvst.sparse_sigfpe','--minimal-worker'],min(20,remaining),stage/'worker.log')
    save(stage/'process.json',process);budget['spent_worker_wall_s']+=process['wall_s'];save(OUT/'budget.json',budget)


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic-worker',action='store_true');p.add_argument('--minimal',action='store_true');p.add_argument('--minimal-worker',action='store_true');a=p.parse_args()
    if a.diagnostic_worker:diagnose()
    elif a.minimal_worker:minimal_worker()
    elif a.minimal:run_minimal()
    else:run_diagnostic()

if __name__=='__main__':main()
