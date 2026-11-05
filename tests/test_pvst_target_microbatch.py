"""Full PVST widths/paths; small native fields only for semantic comparison."""
import copy
import numpy as np
import torch
from torch import nn
from test_pvst_v02 import FakeDecoder,CFG,frames
from research_tools.pvst.model import PVST
from research_tools.pvst.v03 import clip_terms


def compare_microbatch(device='cpu',dtype=torch.float32):
    torch.set_num_threads(4);torch.manual_seed(20260909)
    c={**CFG,'voxel_sizes':{'XYZ_PVST':.125},'qkv_chunk':8192,'ffn_chunk':8192,
       'sparse_edge_attention':True,'head_output_offload':True,'head_qkv_stream':True,
       'node_split':True,'offload_saved_tensors':device=='cuda'}
    a=PVST('XYZ_PVST',FakeDecoder(),c).to(device).train()
    for mod in a.modules():
        if isinstance(mod,nn.Dropout):mod.p=0.
        if isinstance(mod,nn.MultiheadAttention):mod.dropout=0.
    # Diagnostic copies only: expose upstream paths masked by official zero heads.
    nn.init.normal_(a.pixel_out.weight,std=.01);nn.init.normal_(a.frame_residual[-1].weight,std=.01)
    b=copy.deepcopy(a);fs=frames(24,h=2,w=3)
    for i,f in enumerate(fs):
        f['q']=f['q']*.07;f['q'][...,2]+=.02*(i%3)
        f['pts']=3.+.061*i+.0003*i*i
        for k,v in list(f.items()):
            if isinstance(v,torch.Tensor):f[k]=v.to(device)
    gt=[np.zeros((2,3),bool) for _ in range(8)];labels=[False]*8
    def run(m,subset,start,truth,y):
        with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
            pred=m(subset,start);terms=clip_terms(pred,truth,y,1.)
        outputs=[{k:p[k].detach().float().cpu() for k in ['frame_logit','pixel_logits']} for p in pred]
        return outputs,terms
    outputs,terms=run(a,fs,16,gt,labels);reference_terms=terms.detach().float().cpu();terms[0].backward();del terms
    got=[];total=torch.zeros(4)
    for lo,hi,start,offset in [(0,20,16,0),(4,24,20,4)]:
        out,values=run(b,fs[lo:hi],start-lo,gt[offset:offset+4],labels[offset:offset+4])
        got+=out;total+=values.detach().float().cpu()*.5;(values[0]*.5).backward();del values
    # Frozen before measurement: existing halo tolerance, existing BF16 QKV tolerance.
    oa,orr=(2e-2,2e-2) if dtype==torch.bfloat16 else (2e-5,1e-5)
    ga,gr=(.2,.03) if dtype==torch.bfloat16 else (2e-5,3e-5)
    errors={}
    for key in ['frame_logit','pixel_logits']:
        x=torch.stack([o[key] for o in outputs]);y=torch.stack([o[key] for o in got])
        errors[key]=float((x-y).abs().max());torch.testing.assert_close(x,y,atol=oa,rtol=orr)
    errors['loss_components']=float((reference_terms-total).abs().max());torch.testing.assert_close(reference_terms,total,atol=oa,rtol=orr)
    grad_errors={};max_grad=0.;max_scaled=0.
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is None:continue
        x,y=p.grad.float(),q.grad.float();error=float((x-y).abs().max());max_grad=max(max_grad,error)
        max_scaled=max(max_scaled,error/max(float(x.abs().max()),1e-12))
        torch.testing.assert_close(x,y,atol=ga,rtol=gr,msg=lambda s:name+': '+s)
        if name in ['geo.0.weight','vfe.0.weight','fine_time.layers.0.qkv.weight','coarse_time.layers.0.qkv.weight','pixel_out.weight','frame_residual.6.weight']:
            assert p.grad.norm()>0;grad_errors[name]={'max_abs':error,'reference_norm':float(x.norm())}
    return {'status':'PASS','dtype':str(dtype),'errors':errors,'gradient_max_abs':max_grad,'gradient_max_abs_over_reference_max':max_scaled,'key_gradients':grad_errors,'tolerances':{'output_atol':oa,'output_rtol':orr,'gradient_atol':ga,'gradient_rtol':gr},'loss_terms_reference':reference_terms.tolist(),'loss_terms_microbatch':total.tolist(),'diagnostic_heads':'same nonzero weights only in diagnostic models; all Dropout and MHA dropout=0; train/checkpoint retained'}


def test_full_model_microbatch_fp32():compare_microbatch()


def test_fixed_partition_and_tail_mean():
    from research_tools.pvst.target_microbatch_check import batches
    plan=batches();assert [i for x in plan for i in range(*x['target_range'])]==list(range(120,128))
    assert all(x['target_range'][0]-x['input_range'][0]==16 for x in plan)
    p=torch.tensor(.2,requires_grad=True)
    pred=[{'frame_logit':p*(i+1),'pixel_logits':p.expand(2,3)*(i+1)} for i in range(7)]
    gt=[np.zeros((2,3),bool)]*7;y=[False]*7
    whole=clip_terms(pred,gt,y,1.);split=sum(clip_terms(pred[a:b],gt[a:b],y[a:b],1.)*((b-a)/7) for a,b in [(0,4),(4,7)])
    torch.testing.assert_close(whole,split)
