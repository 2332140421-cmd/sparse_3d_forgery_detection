"""Head row mapping, complete-D norm and bounded activation rematerialization."""
import copy
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils._python_dispatch import TorchDispatchMode
from research_tools.pvst.temporal import SparseLayer
from research_tools.pvst.sparse_attention_check import small_graph,TOLERANCES


class Shapes(TorchDispatchMode):
    def __init__(self,n,dim,forbid):super().__init__();self.n=n;self.dim=dim;self.forbid=forbid;self.full=0;self.head=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        assert '_to_dense' not in str(func),str(func)
        out=func(*args,**(kwargs or {}))
        for t in out if isinstance(out,(tuple,list)) else [out]:
            if not isinstance(t,torch.Tensor) or t.layout!=torch.strided:continue
            shape=tuple(t.shape)
            assert shape not in [(self.n,self.n),(self.n,self.n+1)],'DENSE_LOGICAL:'+str(func)
            if shape in [(self.n,3*self.dim),(self.n,3,self.dim//64,64)]:
                self.full+=1;assert not self.forbid,'FULL_NODE_QKV:'+str(func)
            if shape==(self.n,192):self.head+=1
        return out


def compare_stream(device='cpu',dtype=torch.float32,dim=256,dropout=False):
    torch.manual_seed(20260909);graph=small_graph(device);n=len(graph.coords)
    a=SparseLayer(dim,dim//64,7,11).to(device).train();a.qkv_chunk=7;a.sparse_edge_attention=True;a.head_output_offload=True
    if not dropout:
        for mod in a.modules():
            if isinstance(mod,torch.nn.Dropout):mod.p=0.
    b=copy.deepcopy(a);b.head_qkv_stream=True
    ids=[id(p) for p in b.parameters()];keys=list(b.state_dict());count=sum(p.numel() for p in b.parameters())
    base=torch.randn(n,dim,device=device,dtype=dtype);inputs=[base.clone().requires_grad_(),base.clone().requires_grad_()];outputs=[];evidence=[]
    for m,x in zip([a,b],inputs):
        guard=Shapes(n,dim,m.head_qkv_stream);torch.manual_seed(71)
        with guard:
            with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):y=checkpoint(m,x,graph,1,use_reentrant=False,preserve_rng_state=True)
            forward=copy.deepcopy(m.qkv_execution);y.square().mean().backward()
        outputs.append(y);evidence.append({'full_shape_events':guard.full,'head_shape_events':guard.head,'forward':forward})
    assert evidence[0]['full_shape_events']>0 and evidence[1]['full_shape_events']==0 and evidence[1]['head_shape_events']>0
    assert ids==[id(p) for p in b.parameters()] and keys==list(b.state_dict())==list(a.state_dict()) and count==sum(p.numel() for p in a.parameters())
    tol=TOLERANCES['bf16' if dtype==torch.bfloat16 else 'fp32']
    def error(x,y):
        d=float((x-y).abs().max());s=float(x.abs().max());return {'max_abs':d,'reference_max_abs':s,'relative_to_max':d/max(s,1e-12)}
    torch.testing.assert_close(*outputs,atol=tol['output_atol'],rtol=tol['output_rtol'])
    pairs=[('input',inputs[0].grad,inputs[1].grad),*[(name,p.grad,q.grad) for (name,p),q in zip(a.named_parameters(),b.parameters())]];errors={}
    for name,x,y in pairs:
        assert x is not None and y is not None and torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x,y,atol=tol['grad_atol'],rtol=tol['grad_rtol']);errors[name]=error(x,y)
    row_gradients=[]
    for h in range(b.heads):
        row_gradients.append([float(b.qkv.weight.grad[base+h*64:base+(h+1)*64].norm()) for base in [0,dim,2*dim]])
    assert all(norm>0 for head in row_gradients for norm in head)
    return {'dtype':str(dtype),'dim':dim,'dropout':.1 if dropout else 0,'output':error(*outputs),'gradients':errors,'qkv_row_gradient_norms':row_gradients,'parameter_identity_preserved':True,'state_dict_unchanged':True,'shape_evidence':evidence}


def test_fp32_both_widths_and_all_gradients():
    for d in [256,512]:compare_stream(dim=d)


def test_bf16_dropout_nested_checkpoint_both_widths():
    for d in [256,512]:compare_stream(dtype=torch.bfloat16,dim=d,dropout=True)


def test_head_rows_use_complete_D_layer_norm():
    torch.manual_seed(1);m=SparseLayer(256,4,7,11);x=torch.randn(11,256,requires_grad=True)
    reference=m.norm_qkv(x).reshape(11,3,4,64)
    for head in range(4):
        ranges=[(base+head*64,base+(head+1)*64) for base in [0,256,512]]
        w=torch.cat([m.qkv.weight[a:b] for a,b in ranges]);b=torch.cat([m.qkv.bias[a:b] for a,b in ranges])
        actual=torch.cat([m.norm_qkv_head(x[i:i+7],w,b) for i in range(0,11,7)]).reshape(11,3,64)
        torch.testing.assert_close(actual,reference[:,:,head,:],atol=2e-6,rtol=2e-5)


def test_stream_requires_explicit_csr_and_cpu_output_and_defaults_off():
    import pytest
    from test_pvst_v02 import FakeDecoder,CFG
    from research_tools.pvst.model import PVST
    m=SparseLayer(256,4);assert not m.head_qkv_stream
    m.head_qkv_stream=True
    with pytest.raises(ValueError,match='STREAM_REQUIRES'):m(torch.randn(41,256),small_graph('cpu'),1)
    a=PVST('XYZ_PVST',FakeDecoder(),CFG);assert not any(l.head_qkv_stream for t in [a.fine_time,a.coarse_time] for l in t.layers)
    b=PVST('XYZ_PVST',FakeDecoder(),{**CFG,'head_qkv_stream':True});assert all(l.head_qkv_stream for t in [b.fine_time,b.coarse_time] for l in t.layers)
    assert list(a.state_dict())==list(b.state_dict())
