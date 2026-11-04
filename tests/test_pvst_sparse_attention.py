"""CPU small contracts; bounded CUDA parity/performance is the real gate."""
import torch
from research_tools.pvst.sparse_attention_check import compare,dropout_check,small_graph
from research_tools.pvst.sparse_attention import edge_topology


def test_sorted_edges_preserve_original_relation_and_no_duplicate():
    for dilation in [1,2,4]:
        graph=small_graph('cpu');crow,col,relation=edge_topology(graph,dilation,7)
        assert len(crow)==len(graph.coords)+1 and int(crow[-1])==len(col)==len(relation)
        for row in range(len(graph.coords)):
            values=col[crow[row]:crow[row+1]];assert bool((values[1:]>values[:-1]).all())
            assert len(values)>0  # self edge remains, missing edges not score0.


def test_fp32_output_and_all_parameter_gradients():
    compare(torch.float32,256,1,'cpu')


def test_sparse_dropout_checkpoint_rng_and_gradients():
    dropout_check('cpu')


def test_model_opt_in_preserves_keys_parameters_and_default():
    from test_pvst_v02 import FakeDecoder,CFG
    from research_tools.pvst.model import PVST
    torch.manual_seed(3);a=PVST('XYZ_PVST',FakeDecoder(),CFG)
    torch.manual_seed(3);b=PVST('XYZ_PVST',FakeDecoder(),{**CFG,'sparse_edge_attention':True})
    assert list(a.state_dict())==list(b.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert not any(l.sparse_edge_attention for t in [a.fine_time,a.coarse_time] for l in t.layers)
    assert all(l.sparse_edge_attention for t in [b.fine_time,b.coarse_time] for l in t.layers)


def test_actual_sparse_core_backward_never_materializes_dense_square():
    from torch.utils._python_dispatch import TorchDispatchMode
    from research_tools.pvst.sparse_attention import sparse_head
    class NoDenseSquare(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            for t in result if isinstance(result,(tuple,list)) else [result]:
                assert not (isinstance(t,torch.Tensor) and t.layout==torch.strided and tuple(t.shape)==(17,17)),str(func)
            return result
    q=torch.randn(17,64,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True);bias=torch.randn(1,81,1,requires_grad=True)
    crow=torch.arange(18);col=torch.arange(17);relation=torch.arange(17)
    with NoDenseSquare():
        y=sparse_head(q,k,v,bias,crow,col,relation,0,.1,True);y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in [q,k,v,bias])


def test_fatal_signal_never_passes_full_window_gate():
    from research_tools.pvst.sparse_attention_check import finalize_layer
    partial={'correctness':[{'passed':True}],'full_window_allowed':False}
    failed=finalize_layer(partial,{'timed_out':False,'returncode':-8})
    assert failed['status']=='FAILED' and not failed['full_window_allowed'] and failed['process']['returncode']==-8
    timed=finalize_layer({'status':'LAYER_PASS','full_window_allowed':True},{'timed_out':True,'returncode':-15})
    assert timed['status']=='TIMEOUT' and not timed['full_window_allowed']


def test_optional_stage_trace_does_not_modify_values_or_gradients():
    from research_tools.pvst.sparse_attention import sparse_head
    q=torch.randn(3,64,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True);bias=torch.randn(1,81,1,requires_grad=True)
    crow=torch.tensor([0,2,4,6]);col=torch.tensor([0,1,1,2,0,2]);relation=torch.arange(6);events=[]
    a=sparse_head(q,k,v,bias,crow,col,relation,0,0.,False)
    b=sparse_head(q,k,v,bias,crow,col,relation,0,0.,False,trace=lambda stage,tensor,head:events.append(stage))
    torch.testing.assert_close(a,b);ga=torch.autograd.grad(a.sum(),[q,k,v,bias],retain_graph=True);gb=torch.autograd.grad(b.sum(),[q,k,v,bias])
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    assert events[0]=='head_start' and events[-1]=='sparse_mm_complete' and 'softmax_start' in events
