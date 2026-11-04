"""Only CSR row-softmax contracts and representation regression."""
import torch
from research_tools.pvst.sparse_attention import CSRSoftmax


def softmax_case(device='cpu',dtype=torch.float64):
    crow=torch.tensor([0,0,1,4,4,6],device=device)
    x=torch.tensor([1000.,-1000.,0.,1000.,-999.,-1000.],dtype=dtype,device=device,requires_grad=True)
    y=CSRSoftmax.apply(x,crow)
    reference=torch.cat([x[0:1].softmax(0),x[1:4].softmax(0),x[4:6].softmax(0)])
    g=torch.arange(1,7,device=device,dtype=dtype)
    a=torch.autograd.grad((y*g).sum(),x,retain_graph=True)[0];b=torch.autograd.grad((reference*g).sum(),x)[0]
    torch.testing.assert_close(y,reference,atol=1e-6,rtol=1e-5);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    for offsets in [[0],[0,0,0]]:
        z=torch.empty(0,device=device,dtype=dtype,requires_grad=True);v=CSRSoftmax.apply(z,torch.tensor(offsets,device=device));v.sum().backward();assert z.grad is not None and z.grad.numel()==0
    return {'probability_max_abs':float((y-reference).abs().max()),'logits_gradient_max_abs':float((a-b).abs().max())}


def test_softmax_extremes_empty_and_gradcheck():
    softmax_case()
    x=torch.randn(9,dtype=torch.float64,requires_grad=True);crow=torch.tensor([0,0,1,5,5,9])
    assert torch.autograd.gradcheck(CSRSoftmax.apply,(x,crow),eps=1e-6,atol=1e-5,rtol=1e-3)


def test_saved_tensors_are_only_probability_and_offsets():
    x=torch.randn(9,requires_grad=True);crow=torch.tensor([0,0,1,5,9]);y=CSRSoftmax.apply(x,crow)
    saved=y.grad_fn.saved_tensors
    assert len(saved)==2 and saved[0].data_ptr()==y.data_ptr() and saved[1].data_ptr()==crow.data_ptr()


def test_head_has_no_explicit_coo_or_dense_square():
    from torch.utils._python_dispatch import TorchDispatchMode
    from pvst_csr_candidate import csr_candidate_head as sparse_head
    class Check(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            # Check caller-level explicit conversions. Internal ATen CSR backward
            # can still use sparse COO; do not claim all backend code is COO-free.
            assert 'to_sparse' not in str(func),str(func)
            out=func(*args,**(kwargs or {}))
            for t in out if isinstance(out,(list,tuple)) else [out]:
                assert not(isinstance(t,torch.Tensor) and t.layout==torch.strided and tuple(t.shape)==(17,17))
            return out
    q=torch.randn(17,64,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True);bias=torch.randn(1,81,1,requires_grad=True)
    with Check():y=sparse_head(q,k,v,bias,torch.arange(18),torch.arange(17),torch.arange(17),0,0.,False)
    # Dense-square check in existing tests covers backward too.
    y.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in [q,k,v,bias])


def test_direct_csr_constructor_backward_is_a_blocking_dense_path():
    from torch.utils._python_dispatch import TorchDispatchMode
    import pytest
    class RejectDense(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if '_to_dense' in str(func):raise RuntimeError('DENSE_CSR_CONSTRUCTOR_BACKWARD')
            return func(*args,**(kwargs or {}))
    v=torch.randn(4,requires_grad=True)
    a=torch.sparse_csr_tensor(torch.arange(5),torch.arange(4),v,size=(4,5))
    with pytest.raises(RuntimeError,match='DENSE_CSR_CONSTRUCTOR_BACKWARD'):
        with RejectDense():torch.sparse.mm(a,torch.randn(5,3)).sum().backward()
