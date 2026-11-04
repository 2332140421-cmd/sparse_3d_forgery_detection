"""Only reference cleanup parity; baseline is this branch's frozen parent."""
import subprocess
import types
import torch
from torch.utils.checkpoint import checkpoint
from research_tools.pvst.temporal import SparseLayer
from research_tools.pvst.sparse_attention_check import small_graph,TOLERANCES
BASELINE='b7ab0b82afc4d04787b093a78c2fa6a98ed0286e'


def old_layer():
    module=types.ModuleType('research_tools.pvst._lifetime_reference');module.__package__='research_tools.pvst'
    code=subprocess.check_output(['git','show',BASELINE+':research_tools/pvst/temporal.py'],text=True)
    exec(compile(code,'<frozen parent temporal>','exec'),module.__dict__)
    return module.SparseLayer


def compare_cleanup(device='cpu',dtype=torch.float32,dim=256):
    torch.manual_seed(20260909);a=old_layer()(dim,dim//64,7,11).to(device).train()
    torch.manual_seed(20260909);b=SparseLayer(dim,dim//64,7,11).to(device).train()
    for m in [a,b]:m.sparse_edge_attention=True;m.qkv_chunk=7;m.head_qkv_stream=True;m.head_output_offload=True;m.node_split=True
    assert list(a.state_dict())==list(b.state_dict());ids=[id(p) for p in b.parameters()]
    graph=small_graph(device);base=torch.randn(len(graph.coords),dim,device=device,dtype=dtype);xs=[base.clone().requires_grad_(),base.clone().requires_grad_()];ys=[]
    for m,x in zip([a,b],xs):
        torch.manual_seed(71)
        with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):y=checkpoint(m,x,graph,1,use_reentrant=False,preserve_rng_state=True)
        y.square().mean().backward();ys.append(y)
    tol=TOLERANCES['bf16' if dtype==torch.bfloat16 else 'fp32'];torch.testing.assert_close(*ys,atol=tol['output_atol'],rtol=tol['output_rtol'])
    def err(x,y):
        d=float((x-y).abs().max());s=float(x.abs().max());return {'max_abs':d,'reference_max_abs':s,'relative_to_max':d/max(s,1e-12)}
    errors={}
    for name,x,y in [('input',xs[0].grad,xs[1].grad),*[(n,p.grad,q.grad) for (n,p),q in zip(a.named_parameters(),b.parameters())]]:
        assert x is not None and y is not None and torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x,y,atol=tol['grad_atol'],rtol=tol['grad_rtol']);errors[name]=err(x,y)
    assert ids==[id(p) for p in b.parameters()] and sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    return {'dtype':str(dtype),'dim':dim,'dropout':.1,'output':err(*ys),'gradients':errors,'parameter_identity_keys_count_unchanged':True}


def test_fp32_cleanup_and_recompute():
    for dim in [256,512]:compare_cleanup(dim=dim)


def test_bf16_cleanup_and_recompute():
    for dim in [256,512]:compare_cleanup(dtype=torch.bfloat16,dim=dim)


def test_view_storage_is_full_storage_not_view_payload():
    x=torch.randn(41,256,requires_grad=True);part=x.split(7)[-1]
    assert part.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
    assert part.untyped_storage().nbytes()==x.numel()*x.element_size()>part.numel()*part.element_size()


def test_observers_preserve_gradients_and_save_only_metadata(tmp_path):
    from types import SimpleNamespace
    from research_tools.pvst.tensor_lifetime_check import configure_trace
    import copy,json
    torch.manual_seed(7);a=SparseLayer(256,4,7,11).train();a.sparse_edge_attention=True;a.qkv_chunk=7;a.head_output_offload=True;a.head_qkv_stream=True;a.node_split=True
    b=copy.deepcopy(a);configure_trace(SimpleNamespace(fine_time=SimpleNamespace(layers=[b]),coarse_time=SimpleNamespace(layers=[])),tmp_path,lambda:{'phase':'unit','update_index':1})
    xs=[torch.randn(41,256,dtype=torch.bfloat16,requires_grad=True)];xs.append(xs[0].clone().requires_grad_());ys=[]
    for m,x in zip([a,b],xs):
        torch.manual_seed(71)
        with torch.autocast('cpu',dtype=torch.bfloat16):y=checkpoint(m,x,small_graph('cpu'),1,use_reentrant=False,preserve_rng_state=True)
        y.square().mean().backward();ys.append(y)
    torch.testing.assert_close(*ys,atol=2e-5,rtol=2e-4)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-3)
    records=[json.loads(line) for line in (tmp_path/'tensor_events.jsonl').read_text().splitlines()]
    assert any(r['kind']=='lifetime' for r in records)
    assert json.loads((tmp_path/'pending_copies.json').read_text())==[] if (tmp_path/'pending_copies.json').exists() else True
