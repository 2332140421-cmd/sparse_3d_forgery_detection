"""Fixed-structure CSR gradient bridge; only the real SpMM path is supported."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from research_tools.pvst.sparse_attention import CSRFromValues,sparse_head


class NoDenseLogical(TorchDispatchMode):
    def __init__(self,shapes):super().__init__();self.shapes=shapes;self.ops=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        assert '_to_dense' not in str(func),'TO_DENSE:'+str(func)
        out=func(*args,**(kwargs or {}))
        for t in out if isinstance(out,(list,tuple)) else [out]:
            assert not(isinstance(t,torch.Tensor) and t.layout==torch.strided and tuple(t.shape) in self.shapes),'DENSE_LOGICAL:'+str(func)
        if any(v in str(func) for v in ['sparse','values']):self.ops.append(str(func))
        return out


def bridge_check(device='cpu'):
    torch.manual_seed(20260909)
    crow=torch.tensor([0,2,2,3,5],device=device);col=torch.tensor([0,2,1,0,3],device=device)
    values=torch.tensor([.2,0.,-.5,0.,.1],device=device,dtype=torch.float64,requires_grad=True)
    rhs=torch.randn(6,3,device=device,dtype=torch.float64,requires_grad=True)
    a=CSRFromValues.apply(values,crow,col,(4,6));incoming=[]
    # Only the tiny diagnostic observes layout, never the full model worker.
    a.register_hook(lambda g:incoming.append({'layout':str(g.layout),'crow':g.crow_indices().tolist(),'col':g.col_indices().tolist()}))
    rows=torch.repeat_interleave(crow.diff(),output_size=len(col));dense=values.new_zeros(4,6).index_put((rows,col),values)
    with NoDenseLogical({(4,6),(6,4)}):out=torch.sparse.mm(a,rhs);ga=torch.autograd.grad(out.square().mean(),[values,rhs])
    ref=dense@rhs;gb=torch.autograd.grad(ref.square().mean(),[values,rhs])
    for x,y in [(out,ref),*zip(ga,gb)]:torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)
    def run(v,r):return torch.sparse.mm(CSRFromValues.apply(v,crow,col,(4,6)),r)
    assert torch.autograd.gradcheck(run,(values,rhs),eps=1e-6,atol=1e-5,rtol=1e-3)
    # Empty edges/rows and non-differentiable values remain legitimate routes.
    for rows_count in [0,3]:
        v=torch.empty(0,device=device,requires_grad=True);r=torch.randn(5,7,device=device,requires_grad=True)
        with NoDenseLogical({(rows_count,5)}):y=torch.sparse.mm(CSRFromValues.apply(v,torch.zeros(rows_count+1,device=device,dtype=torch.long),torch.empty(0,device=device,dtype=torch.long),(rows_count,5)),r);y.sum().backward()
        assert v.grad is not None and r.grad is not None and torch.isfinite(r.grad).all()
    constant=CSRFromValues.apply(values.new_ones(len(col)),crow,col,(4,6));assert not constant.requires_grad
    return {'incoming_gradient':incoming[0],'gradcheck':True,'output_max_abs':float((out-ref).abs().max()),'values_grad_max_abs':float((ga[0]-gb[0]).abs().max()),'V_grad_max_abs':float((ga[1]-gb[1]).abs().max()),'empty_edges_rows':True,'explicit_zero_edges':True,'padding_columns':2}


def layout_check(device='cpu'):
    from torch.utils.checkpoint import checkpoint
    from research_tools.pvst.sparse_attention_check import small_graph
    from research_tools.pvst.temporal import SparseLayer
    graph=small_graph(device);n=len(graph.coords);layer=SparseLayer(256,4,7,11).to(device).train();layer.qkv_chunk=7;layer.sparse_edge_attention=True
    x=torch.randn(n,256,device=device,requires_grad=True)
    guard=NoDenseLogical({(n,n),(n,n+1),(n+1,n)})
    with guard:
        out=checkpoint(layer,x,graph,1,use_reentrant=False,preserve_rng_state=True);out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in [x,*layer.parameters()])
    return {'logical_nodes':n,'forward_backward_checkpoint_no_dense_logical':True,'operators':sorted(set(guard.ops))}


def test_bridge_values_and_V_gradients_and_gradcheck():bridge_check()


def test_complete_layer_checkpoint_has_no_dense_logical():layout_check()


def test_same_nnz_wrong_structure_or_dense_gradient_is_rejected():
    import pytest
    crow=torch.tensor([0,1,2]);col=torch.tensor([0,1]);v=torch.randn(2,requires_grad=True)
    a=CSRFromValues.apply(v,crow,col,(2,3))
    with pytest.raises(RuntimeError,match='COLUMN_ORDER_CHANGED'):a.backward(torch.sparse_csr_tensor(crow,torch.tensor([1,2]),torch.ones(2),size=(2,3)))
    a=CSRFromValues.apply(v,crow,col,(2,3))
    with pytest.raises(RuntimeError,match='REQUIRES_CSR'):a.backward(torch.ones(2,3))
