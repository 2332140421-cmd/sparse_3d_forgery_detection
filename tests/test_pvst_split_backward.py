"""Only large-node blocking gradients; legitimate parameter slices remain."""
import copy
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint
from research_tools.pvst.sparse_attention_check import small_graph,TOLERANCES
from research_tools.pvst.temporal import SparseLayer


class Record(TorchDispatchMode):
    def __init__(self,n,d):super().__init__();self.n=n;self.d=d;self.slices=[];self.split_calls=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        assert '_to_dense' not in str(func)
        out=func(*args,**(kwargs or {}))
        if 'slice_backward' in str(func) and list(args[1])==[self.n,self.d]:self.slices.append({'input_sizes':list(args[1]),'dtype':str(out.dtype),'bytes':out.numel()*out.element_size()})
        if str(func)=='aten.split.Tensor':self.split_calls+=1
        for t in out if isinstance(out,(list,tuple)) else [out]:
            if isinstance(t,torch.Tensor) and t.layout==torch.strided:assert tuple(t.shape) not in [(self.n,self.n),(self.n,3*self.d),(self.n,self.n+1)]
        return out


def compare_split(device='cpu',dtype=torch.float32,dim=256):
    torch.manual_seed(20260909);graph=small_graph(device);n=len(graph.coords)
    a=SparseLayer(dim,dim//64,7,11).to(device).train();a.qkv_chunk=7;a.sparse_edge_attention=True;a.head_output_offload=True;a.head_qkv_stream=True
    b=copy.deepcopy(a);b.node_split=True;ids=[id(p) for p in b.parameters()]
    base=torch.randn(n,dim,device=device,dtype=dtype);xs=[base.clone().requires_grad_(),base.clone().requires_grad_()];ys=[];proof=[]
    for m,x in zip([a,b],xs):
        torch.manual_seed(71);r=Record(n,dim)
        with r:
            with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):y=checkpoint(m,x,graph,1,use_reentrant=False,preserve_rng_state=True)
            blocking=copy.deepcopy(m.blocking_execution);y.square().mean().backward()
        ys.append(y);proof.append({'full_node_slice_backward_count':len(r.slices),'slice_allocated_bytes_cumulative':sum(v['bytes'] for v in r.slices),'slice_examples':r.slices[:2],'native_split_calls':r.split_calls,'blocking':blocking})
    assert proof[0]['full_node_slice_backward_count']>0 and proof[1]['full_node_slice_backward_count']==0
    assert all(v['view_grad_fn']=='SplitBackward0' for v in proof[1]['blocking'].values())
    assert ids==[id(p) for p in b.parameters()] and list(a.state_dict())==list(b.state_dict()) and sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    tol=TOLERANCES['bf16' if dtype==torch.bfloat16 else 'fp32']
    def err(x,y):
        delta=float((x-y).abs().max());scale=float(x.abs().max());return {'max_abs':delta,'reference_max_abs':scale,'relative_to_max':delta/max(scale,1e-12)}
    torch.testing.assert_close(*ys,atol=tol['output_atol'],rtol=tol['output_rtol'])
    pairs=[('input',xs[0].grad,xs[1].grad),*[(name,p.grad,q.grad) for (name,p),q in zip(a.named_parameters(),b.parameters())]];errors={}
    for name,x,y in pairs:
        assert x is not None and y is not None and torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x,y,atol=tol['grad_atol'],rtol=tol['grad_rtol']);errors[name]=err(x,y)
    return {'dtype':str(dtype),'dim':dim,'dropout':.1,'output':err(*ys),'gradients':errors,'parameter_identity_keys_count_preserved':True,'proof':proof}


def test_split_fp32_widths_dropout_checkpoint_tail():
    for d in [256,512]:compare_split(dim=d)


def test_split_bf16_widths_dropout_checkpoint_tail():
    for d in [256,512]:compare_split(dtype=torch.bfloat16,dim=d)


def test_default_off_and_shared_native_split_view():
    a=SparseLayer(256,4);assert not a.node_split
    x=torch.randn(11,256,requires_grad=True);a.node_split=True;parts=list(a.node_parts(x,7,'test'))
    assert [len(p) for p in parts]==[7,4] and all(p.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() for p in parts)
    assert parts[0].grad_fn is parts[1].grad_fn
