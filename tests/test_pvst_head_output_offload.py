"""Only storage placement, head order and complete gradient/RNG contracts."""
import copy
import torch
from torch.utils.checkpoint import checkpoint
from research_tools.pvst.sparse_attention_check import small_graph,TOLERANCES
from research_tools.pvst.temporal import SparseLayer
from research_tools.pvst.csr_values_bridge_check import load_test
NoDenseLogical=load_test('test_pvst_csr_values_bridge.py').NoDenseLogical


def compare_offload(device='cpu',dtype=torch.float32,dim=256,dropout=False):
    torch.manual_seed(20260909);graph=small_graph(device);n=len(graph.coords)
    a=SparseLayer(dim,dim//64,7,11).to(device).train();a.sparse_edge_attention=True;a.qkv_chunk=7
    if not dropout:
        for mod in a.modules():
            if isinstance(mod,torch.nn.Dropout):mod.p=0.
    b=copy.deepcopy(a);b.head_output_offload=True
    assert not a.head_output_offload and list(a.state_dict())==list(b.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    base=torch.randn(n,dim,device=device,dtype=dtype);xs=[base.clone().requires_grad_(),base.clone().requires_grad_()];ys=[];storage=[]
    for model,x in zip([a,b],xs):
        torch.manual_seed(71)
        with NoDenseLogical({(n,n),(n,n+1),(n+1,n)}):
            with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):y=checkpoint(model,x,graph,1,use_reentrant=False,preserve_rng_state=True)
            storage.append(copy.deepcopy(model.head_storage));y.square().mean().backward()
        ys.append(y)
    tol=TOLERANCES['bf16' if dtype==torch.bfloat16 else 'fp32']
    def error(x,y):
        diff=float((x-y).abs().max());scale=float(x.abs().max());return {'max_abs':diff,'reference_max_abs':scale,'relative_to_max':diff/max(scale,1e-12)}
    torch.testing.assert_close(*ys,atol=tol['output_atol'],rtol=tol['output_rtol'])
    pairs=[('input',xs[0].grad,xs[1].grad),*[(name,p.grad,q.grad) for (name,p),q in zip(a.named_parameters(),b.parameters())]]
    errors={}
    for name,x,y in pairs:
        assert x is not None and y is not None and torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x,y,atol=tol['grad_atol'],rtol=tol['grad_rtol']);errors[name]=error(x,y)
    cpu=storage[1]
    assert cpu['stack_device']=='cpu' and cpu['gpu_list_bytes']==0 and cpu['cpu_bytes']==n*dim*torch.tensor([],dtype=dtype).element_size()
    assert all(h['stored_device']=='cpu' and h['requires_grad'] and h['dtype']==str(dtype) for h in cpu['heads'])
    if device=='cuda':assert all(h['grad_fn']=='ToCopyBackward0' for h in cpu['heads'])
    return {'dtype':str(dtype),'dim':dim,'dropout':.1 if dropout else 0,'output':error(*ys),'gradients':errors,'state_dict_keys_unchanged':True,'parameters_unchanged':True,'no_dense_logical':True,'storage':storage}


def test_offload_complete_output_and_gradient_parity():compare_offload()


def test_dropout_and_outer_checkpoint_rng_parity():compare_offload(dtype=torch.bfloat16,dim=512,dropout=True)


def test_head_order_and_reshape(monkeypatch):
    from research_tools.pvst import sparse_attention
    def fake(q,k,v,bias,crow,col,relation,head,*args):return q*0+(head+1)
    monkeypatch.setattr(sparse_attention,'sparse_head',fake)
    graph=small_graph('cpu');a=SparseLayer(256,4,7,11).train();a.sparse_edge_attention=True;a.head_output_offload=True;a.qkv_chunk=7;inputs=[]
    hook=a.proj.register_forward_pre_hook(lambda mod,args:inputs.append(args[0]))
    a(torch.randn(len(graph.coords),256,requires_grad=True),graph,1);hook.remove()
    packed=torch.cat(inputs).reshape(len(graph.coords),4,64)
    torch.testing.assert_close(packed,torch.arange(1,5).float()[None,:,None].expand_as(packed))


def test_model_offload_default_and_state_dict():
    from test_pvst_v02 import FakeDecoder,CFG
    from research_tools.pvst.model import PVST
    torch.manual_seed(7);a=PVST('XYZ_PVST',FakeDecoder(),CFG)
    torch.manual_seed(7);b=PVST('XYZ_PVST',FakeDecoder(),{**CFG,'head_output_offload':True})
    assert list(a.state_dict())==list(b.state_dict())
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert not any(l.head_output_offload for t in [a.fine_time,a.coarse_time] for l in t.layers)
    assert all(l.head_output_offload for t in [b.fine_time,b.coarse_time] for l in t.layers)
