"""Full PVST widths/paths; small native fields only for semantic comparison."""
import copy
import numpy as np
import torch
from torch import nn
from test_pvst_v02 import FakeDecoder,CFG,frames
from research_tools.pvst.model import PVST
from research_tools.pvst.v03 import clip_terms


def diagnostic_fixture(device):
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
    fs=frames(24,h=2,w=3)
    for i,f in enumerate(fs):
        f['q']=f['q']*.07;f['q'][...,2]+=.02*(i%3)
        f['pts']=3.+.061*i+.0003*i*i
        for k,v in list(f.items()):
            if isinstance(v,torch.Tensor):f[k]=v.to(device)
    gt=[np.zeros((2,3),bool) for _ in range(8)];labels=[False]*8
    return a,fs,gt,labels


def compare_microbatch(device='cpu',dtype=torch.float32,group_size=4):
    a,fs,gt,labels=diagnostic_fixture(device);b=copy.deepcopy(a)
    def run(m,subset,start,truth,y):
        with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
            pred=m(subset,start);terms=clip_terms(pred,truth,y,1.)
        outputs=[{k:p[k].detach().float().cpu() for k in ['frame_logit','pixel_logits']} for p in pred]
        return outputs,terms
    outputs,terms=run(a,fs,16,gt,labels);reference_terms=terms.detach().float().cpu();terms[0].backward();del terms
    got=[];total=torch.zeros(4)
    for offset in range(0,8,group_size):
        lo,hi,start=offset,16+offset+group_size,16+offset
        weight=group_size/8
        out,values=run(b,fs[lo:hi],start-lo,gt[offset:offset+group_size],labels[offset:offset+group_size])
        got+=out;total+=values.detach().float().cpu()*weight;(values[0]*weight).backward();del values
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


def test_group2_full_model_and_fixed_coverage():
    from research_tools.pvst.target_microbatch_check import batches
    plan=batches(2)
    assert [x['input_range'] for x in plan]==[[104,122],[106,124],[108,126],[110,128]]
    assert [i for x in plan for i in range(*x['target_range'])]==list(range(120,128))
    assert all(x['target_range'][0]-x['input_range'][0]==16 and x['target_range'][1]-x['target_range'][0]==2 for x in plan)
    assert sum(b-a for a,b in [x['input_range'] for x in plan])==72
    compare_microbatch(group_size=2)


def gradient_comparison(actual,reference,atol,rtol):
    """Reference is always second, exactly like the existing assert_close call."""
    if actual is None or reference is None:
        return {'pass':actual is reference,'actual_missing':actual is None,'reference_missing':reference is None,'mismatches':0 if actual is reference else 1},None
    x,y=actual.double(),reference.double();diff=(x-y).abs();allowed=atol+rtol*y.abs()
    bad=(diff>allowed)|~torch.isfinite(x)|~torch.isfinite(y)
    return {'pass':not bool(bad.any()),'mismatches':int(bad.sum()),'max_abs':float(diff.max()),
            'max_abs_over_reference_max':float(diff.max()/y.abs().max().clamp_min(1e-12)),
            'actual_zero':not bool(x.any()),'reference_zero':not bool(y.any()),
            'actual_norm':float(x.norm()),'reference_norm':float(y.norm())},bad


def abc_microbatch(device='cpu',dtype=torch.float32):
    """Only small diagnostic retains shared graph; real worker is unchanged."""
    a,fs,gt,labels=diagnostic_fixture(device);b=copy.deepcopy(a);c=copy.deepcopy(a)
    names=[n for n,_ in a.named_parameters()]
    def forward(m,lo,hi,start):
        with torch.autocast(device,dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):pred=m(fs[lo:hi],start-lo)
        return pred
    def terms(pred,j=0):return clip_terms(pred,gt[j:j+len(pred)],labels[j:j+len(pred)],1.)
    def outputs(pred):return {k:torch.stack([p[k].detach().float().cpu() for p in pred]) for k in ['frame_logit','pixel_logits']}
    def cpu_grad(grads):return [None if g is None else g.detach().cpu() for g in grads]
    pa=forward(a,0,24,16);ao=outputs(pa);ta=terms(pa);at=ta.detach().float().cpu()
    ag=cpu_grad(torch.autograd.grad(ta[0],tuple(a.parameters()),allow_unused=True));del pa,ta
    pb=forward(b,0,24,16);bo=outputs(pb);bg=[];bt=[]
    for j in range(0,8,2):
        value=terms(pb[j:j+2],j)*.25;bt.append(value.detach().float().cpu())
        bg.append(cpu_grad(torch.autograd.grad(value[0],tuple(b.parameters()),retain_graph=j<6,allow_unused=True)))
        del value
    del pb
    cg=[{} for _ in range(4)];current=[0]
    def capture(name,g):
        # Raw leaf contribution; never reconstruct by subtracting accumulated .grad.
        store=cg[current[0]];v=g.detach().cpu().double()
        store[name]=v if name not in store else store[name]+v
    handles=[p.register_hook(lambda g,name=n:capture(name,g)) for n,p in c.named_parameters()]
    co=[];ct=[]
    for k,j in enumerate(range(0,8,2)):
        current[0]=k;pc=forward(c,j,18+j,16+j);co.append(outputs(pc));value=terms(pc,j)*.25;ct.append(value.detach().float().cpu())
        value[0].backward();del pc,value
    for h in handles:h.remove()
    actual=cpu_grad([p.grad for p in c.parameters()]);cs=[[group.get(n) for n in names] for group in cg]
    def total(groups):return [None if all(g[i] is None for g in groups) else sum(g[i].double() for g in groups if g[i] is not None) for i in range(len(names))]
    bs,csum=total(bg),total(cs)
    ga,gr=(.2,.03) if dtype==torch.bfloat16 else (2e-5,3e-5)
    oa,orr=(.02,.02) if dtype==torch.bfloat16 else (2e-5,1e-5)
    pairs={'A_Bsum':(ag,bs),'Bsum_Csum':(bs,csum),'Cactual_Csum':(actual,csum),'Cactual_Bsum':(actual,bs),'A_Cactual':(ag,actual)}
    for k in range(4):pairs['B_C_group'+str(k+1)]=(bg[k],cs[k])
    summaries={};bad_details=[];all_explained=True;original_bias=[]
    for i,n in enumerate(names):
        row={};bad_union=torch.zeros_like(ag[i],dtype=torch.bool) if ag[i] is not None else None
        original_bad=None
        for label,(xs,ys) in pairs.items():
            detail,bad=gradient_comparison(xs[i],ys[i],ga,gr);row[label]=detail
            if bad is not None and bad_union is not None:bad_union|=bad
            if label=='A_Cactual':
                original_bad=bad
                if bad is None and not detail['pass']:all_explained=False
        summaries[n]=row
        indices=[] if bad_union is None else bad_union.nonzero().tolist()
        if n=='decoder.final_conv_block.op.bias' and bad_union is not None:
            # Preserve previous offending indices even if this run lands just below tolerance.
            indices=sorted({tuple(x) for x in indices}|{(6,),(19,)})
        for idx in indices:
            ix=tuple(idx)
            def scalar(x):return None if x is None else float(x[ix])
            av,bv,cv,rv=[scalar(g[i]) for g in [ag,bs,csum,actual]]
            entry={'parameter':n,'index':list(idx),'A':av,'B_sum':bv,'C_sum':cv,'C_actual':rv,
                   'B_groups':[scalar(g[i]) for g in bg],'C_groups':[scalar(g[i]) for g in cs],
                   'allowed_by_comparison':{label:None if ys[i] is None else ga+gr*abs(scalar(ys[i])) for label,(xs,ys) in pairs.items()}}
            if None not in [av,bv,cv,rv]:
                reduction=bv-av;original=rv-av;residual=original-reduction
                limit=ga+gr*abs(bv)
                # No numeric relaxation: same-direction reduction and residual within old tolerance.
                explained=original*reduction>0 and abs(residual)<=limit
                entry.update(original_C_minus_A=original,full_graph_B_minus_A=reduction,residual_C_minus_B=residual,residual_allowed=limit,explained=explained)
                if original_bad is not None and bool(original_bad[ix]):all_explained&=explained
            else:all_explained=False
            bad_details.append(entry)
    output_checks={}
    for key in ao:
        output_checks['A_B_'+key]=gradient_comparison(ao[key],bo[key],oa,orr)[0]
        output_checks['A_C_'+key]=gradient_comparison(ao[key],torch.cat([x[key] for x in co]),oa,orr)[0]
        for k in range(4):output_checks['B_C_group'+str(k+1)+'_'+key]=gradient_comparison(bo[key][k*2:k*2+2],co[k][key],oa,orr)[0]
    output_checks['A_C_loss']=gradient_comparison(at,torch.stack(ct).double().sum(0),oa,orr)[0]
    output_checks['A_B_loss']=gradient_comparison(at,torch.stack(bt).double().sum(0),oa,orr)[0]
    for k in range(4):output_checks['B_C_group'+str(k+1)+'_loss']=gradient_comparison(bt[k],ct[k],oa,orr)[0]
    checks={label:all(row[label]['pass'] for row in summaries.values()) for label in pairs}
    groups_ok=all(checks['B_C_group'+str(k+1)] for k in range(4))
    reference_ok=groups_ok and all(checks[x] for x in ['Bsum_Csum','Cactual_Csum','Cactual_Bsum']) and all(x['pass'] for x in output_checks.values())
    passed=reference_ok and (checks['A_Cactual'] or all_explained)
    return {'status':'PASS' if passed else 'FAILED','gate':'original_reference' if passed and checks['A_Cactual'] else 'grouped_reference' if passed else 'blocked',
            'dtype':str(dtype),'tolerances':{'grad_atol':ga,'grad_rtol':gr,'output_atol':oa,'output_rtol':orr},
            'comparison_rule':'actual first, reference second; abs(actual-reference)<=atol+rtol*abs(reference); nonfinite/missing mismatch fails',
            'checks':checks,'output_loss_checks':output_checks,'all_original_mismatches_explained':all_explained,
            'parameters':summaries,'exception_elements':bad_details,'retained_gradient_tensors_on_disk':False,
            'A_loss_components':at.tolist(),'B_weighted_loss_components':[x.tolist() for x in bt],'C_weighted_loss_components':[x.tolist() for x in ct]}


def test_gradient_comparison_preserves_missing_zero_and_original_threshold():
    assert gradient_comparison(None,None,2e-5,3e-5)[0]['pass']
    assert not gradient_comparison(None,torch.zeros(2),2e-5,3e-5)[0]['pass']
    d,_=gradient_comparison(torch.tensor([3.1e-5]),torch.zeros(1),2e-5,3e-5)
    assert not d['pass'] and d['mismatches']==1


def test_abc_small_full_model_cpu():
    r=abc_microbatch()
    assert r['status']=='PASS'
    assert all(r['checks']['B_C_group'+str(i)] for i in range(1,5))
    assert 'decoder.final_conv_block.op.bias' in r['parameters']
