"""Fixed-weight BN-statistics intervention, ADR-0052.

Immediate caller this module's run; reuse original model/evaluate/preview.
Only five head BN forwards are temporarily replaced, never model.train().
"""
from contextlib import contextmanager
import copy
import json
import subprocess
import time
import traceback
from types import MethodType
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from .ainpaint_spatial import build,load_adaptation,WEIGHT,DINO,AINPAINT
from .ainpaint_spatial_experiment import OUT as SOURCE,NAME,RGBOnlyCache
from .data import ROOT,REPO,read
from .insource_probe import evaluate
from .experiment import csvsave
from .joint_experiment import save,digest,preview
from .readout_adaptation import add_area,METRICS
from research_tools.rgbd_slice.experiment import state_hash

OUT=ROOT/'pvst_anakin_ainpaint_bn_diagnosis_v1'
CONDITIONS=['RUNNING_BN','IMAGE_BN']
BN_NAMES=['proj_norm','up1.norm','up2.norm','up3.norm','final_conv_block.1']
FIXED=['7e2e57_fake','95q8g9_fake','4xd3r9_fake','7ogeg4_fake']


def head_bn(model):
    result={n:m for n,m in model.head.named_modules() if isinstance(m,nn.BatchNorm2d)}
    assert list(result)==BN_NAMES,list(result)
    return result


def image_forward(self,x):
    assert not self.training and not torch.is_grad_enabled()
    return F.batch_norm(x,None,None,self.weight,self.bias,True,0.,self.eps)


@contextmanager
def image_bn(model):
    assert not any(m.training for m in model.modules())
    modules=head_bn(model);old={n:m.forward for n,m in modules.items()}
    try:
        for m in modules.values():m.forward=MethodType(image_forward,m)
        yield
    finally:
        for n,m in modules.items():m.forward=old[n]


def mode_check(model):
    assert not any(m.training for m in model.modules())
    drops=[n for n,m in model.named_modules() if isinstance(m,nn.Dropout)]
    assert drops and all(not m.training for m in model.modules() if isinstance(m,nn.Dropout))
    return dict(all_modules_eval=True,dropout_modules=len(drops),dropout_all_disabled=True)


@torch.no_grad()
def fixed_checks(model,cache,rows,old):
    rows={r['video_id']:r for r in rows};prior={(r['video_id'],r['frame_index']):r for r in old['frames']};records=[]
    for vid in FIXED:
        row=rows[vid];i=row['preview_targets'][1]
        short=copy.deepcopy(row);short['evaluation_targets']=[i]
        real=next(r for r in rows.values() if r['pair_id']==row['pair_id'] and r['role']=='real')
        real=copy.deepcopy(real);real['evaluation_targets']=[real['preview_targets'][1]]
        replay,_=evaluate(model,cache,[short,real],NAME)
        current=replay['frames'][0];ref=prior[(vid,i)]
        delta={k:current[k]-ref[k] for k in ['pixel_AP','Dice','IoU','GT_recall','fake_background_FPR']}
        assert max(abs(v) for v in delta.values())<=1e-6,delta
        records.append(dict(video_id=vid,frame_index=i,pts=row['pts'][i],saved_frame_metric_differences=delta))
    row=rows[FIXED[0]];i=row['preview_targets'][1];frame=cache.get(row,i,'evaluation');state=state_hash(model.state_dict())
    with torch.autocast('cuda',dtype=torch.bfloat16):a=model([frame])['pixel'];b=model([frame])['pixel']
    same=(a.float()-b.float()).abs();repeat=dict(RUNNING_BN=dict(max_abs=float(same.max()),mean_abs=float(same.mean())))
    assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)
    with image_bn(model),torch.autocast('cuda',dtype=torch.bfloat16):a=model([frame])['pixel'];b=model([frame])['pixel']
    same=(a.float()-b.float()).abs();repeat['IMAGE_BN']=dict(max_abs=float(same.max()),mean_abs=float(same.mean()));assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)
    assert state==state_hash(model.state_dict());mode_check(model)
    return dict(status='PASS',fixed_mid_targets=records,repeat_target=dict(video_id=row['video_id'],frame_index=i),same_condition_work_logit_repeat=repeat,repeat_tolerance=dict(atol=1e-5,rtol=1e-5),saved_metric_limit=1e-6,weights_buffers_unchanged=True,old_native_FP32_logits_saved=False,replay_scope='Saved native per-frame metrics; old full FP32 logits were not saved, so no claim of old logit bitwise replay')


@contextmanager
def running_statistics(model):
    """Passive BN-input channel moments, scalar summaries only; no extra forward."""
    records={n:[] for n in head_bn(model)};handles=[]
    def make(name):
        def capture(m,args):
            x=args[0].detach().float();mu=x.mean((0,2,3));var=x.var((0,2,3),unbiased=False)
            records[name].append(dict(mean_abs_gap=float((mu-m.running_mean.float()).abs().mean()),standardized_mean_abs_gap=float(((mu-m.running_mean.float()).abs()/torch.sqrt(m.running_var.float()+m.eps)).mean()),mean_variance=float(var.mean()),running_mean_variance=float(m.running_var.float().mean()),variance_mean_ratio=float(var.mean()/m.running_var.float().mean().clamp_min(1e-12))))
        return capture
    try:
        for n,m in head_bn(model).items():handles.append(m.register_forward_pre_hook(make(n)))
        yield records
    finally:
        for h in handles:h.remove()


def save_stats(stats,frames):
    per=[];summary=[]
    for name,records in stats.items():
        assert len(records)==len(frames)==824
        for f,r in zip(frames,records):per.append(dict(video_id=f['video_id'],frame_index=f['frame_index'],pts=f['pts'],BN=name,**r))
        for vid in dict.fromkeys(f['video_id'] for f in frames):
            rs=[r for f,r in zip(frames,records) if f['video_id']==vid]
            summary.append(dict(video_id=vid,BN=name,frames=len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in records[0]}))
    csvsave(OUT/'running_BN_input_statistics.csv',summary)
    # Scalar per-frame summaries are small; no channel maps or features stored.
    csvsave(OUT/'running_BN_input_statistics_frames.csv',per)


def persist(name,e):
    add_area(e);save(OUT/f'{name}.json',e);csvsave(OUT/f'{name}_frames.csv',e['frames']);csvsave(OUT/f'{name}_videos.csv',e['videos'])


def previews(rows,preds):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    html=['<!doctype html><meta charset="utf-8"><h1>固定24目标：RUNNING_BN / IMAGE_BN</h1><p>每幅RGB / GT / 概率，概率统一0–1，二值固定0.5。差分IMAGE−RUNNING统一[-1,1]：红更高、蓝更低、白为0。仅预览缩放。</p>']
    for name in CONDITIONS:preview(rows,preds[name],name,'evaluation',output_root=OUT)
    for row in rows:
        for i in row['preview_targets']:
            vid=row['video_id'];html.append(f'<h3>{row["pair_id"]}/{row["role"]} 帧{i} PTS {row["pts"][i]:.6f}s</h3><div style="display:flex">')
            for name in CONDITIONS:
                a=preds[name][(vid,i)];h,w=a.shape;fac=min(1.,480/w);stem=f'{name}_{vid}_{i:05d}'
                Image.fromarray(((a>=0)*255).astype(np.uint8)).resize((round(w*fac),round(h*fac)),Image.Resampling.NEAREST).save(folder/'evaluation'/(stem+'_binary.png'))
                html.append(f'<div style="width:50%">{name}<img width="100%" src="evaluation/{stem}.png"><img width="33%" src="evaluation/{stem}_binary.png"></div>')
            sig=lambda x:1/(1+np.exp(-np.clip(x,-80,80)))
            d=sig(preds['IMAGE_BN'][(vid,i)])-sig(preds['RUNNING_BN'][(vid,i)])
            rgb=np.stack([1+np.minimum(d,0),1-np.abs(d),1-np.maximum(d,0)],-1)
            diff=f'diff_{vid}_{i:05d}.png';Image.fromarray((rgb*255).round().astype(np.uint8)).resize((round(w*fac),round(h*fac))).save(folder/'evaluation'/diff)
            html.append(f'</div><p>概率差分[-1,1]</p><img width="40%" src="evaluation/{diff}">')
    (folder/'index.html').write_text('\n'.join(html))


def report(s):
    a,b=[s['results'][n] for n in CONDITIONS]
    lines=['# 固定权重BN统计使用方式诊断','计算COMPLETE、退出码0；同一权重，仅干预5个定位BN。不是新训练模型成绩，不替换原报告。','|指标|RUNNING_BN|IMAGE_BN|IMAGE−RUNNING|','|---|---:|---:|---:|']
    for k in METRICS:lines.append(f'|{k}|{a["macro"][k]:.8f}|{b["macro"][k]:.8f}|{b["macro"][k]-a["macro"][k]:+.8f}|')
    lines+=['\n|来源|恒定AP/GT面积|RUNNING AP|IMAGE AP|AP差值|RUNNING/IMAGE Dice|RUNNING/IMAGE召回|','|---|---:|---:|---:|---:|---|---|'];table=[]
    for row in s['dev_rows']:
        if row['role']!='fake':continue
        va,vb=[next(v for v in e['videos'] if v['video_id']==row['video_id']) for e in [a,b]]
        assert va['constant_AP']==vb['constant_AP']
        lines.append(f'|{row["pair_id"]}|{va["constant_AP"]:.8f}|{va["pixel_AP"]:.8f}|{vb["pixel_AP"]:.8f}|{vb["pixel_AP"]-va["pixel_AP"]:+.8f}|{va["Dice"]:.8f}/{vb["Dice"]:.8f}|{va["GT_recall"]:.8f}/{vb["GT_recall"]:.8f}|')
        for name,e,v in zip(CONDITIONS,[a,b],[va,vb]):
            r=next(q for q in e['videos'] if q['pair_id']==row['pair_id'] and q['role']=='real')
            table.append(dict(pair_id=row['pair_id'],condition=name,constant_AP=v['constant_AP'],**{k:r[k] if k=='real_FPR' else v[k] for k in METRICS}))
    csvsave(OUT/'source_comparison.csv',table)
    lines+=['\n## 精确干预','整体eval/no_grad，LoRA与head dropout关闭；仅head内五个BatchNorm2d临时调用F.batch_norm(x,None,None,weight,bias,training=True,momentum=0,eps=原eps)。不使用或更新running统计、计数，不改DINO LayerNorm，不跨帧累积，不用GT选择统计。forward在finally恢复。',
        '实际BN：'+json.dumps(s['BN_scope'],ensure_ascii=False),
        '回放与重复：'+json.dumps(s['preflight'],ensure_ascii=False),
        '完整824帧RUNNING与原报告差：'+json.dumps(s['replay'],ensure_ascii=False),
        '原native FP32预测未保存，旧回放比较逐帧指标，不宣称旧logit逐值相同；本轮固定同输入重复logit差另列。',
        '工作RGB、pad14/crop、BF16前向/FP32原生恢复和计分完全复用。原4对完整824唯一帧，412正GT fake/412real；AP仅fake正GT帧→视频→来源宏平均。无训练GT校准、优化器/反向、训练、阈值搜索或新前端。',
        'RUNNING完整前向被动记录BN输入的通道均值及biased空间方差与running统计差异，只有标量摘要；不单凭均值差宣布故障。',
        '## 身份与成本',json.dumps(s['identity'],ensure_ascii=False),json.dumps(s['cost'],ensure_ascii=False),
        f'产物 {OUT}；固定预览 {OUT}/preview/index.html；执行HEAD {s["identity"]["execution_head"]}。计算与Git发布分开记录。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


@torch.no_grad()
def run():
    start=time.monotonic();torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'identity.json').exists():
        assert (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE','Existing incomplete result preserved; no automatic retry'
        print('EXISTING_COMPLETE_REUSED');return
    source_id=read(SOURCE/'identity.json');old=read(SOURCE/'final_dev.json');protocol=read(SOURCE/'protocol.json');rows=protocol['dev'];path=SOURCE/'final.pt'
    assert digest(WEIGHT)==source_id['generic_weight_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['condition']==NAME and cp['updates']==12845 and cp['epoch']==5 and cp['identity']==source_id
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_checkpoint=str(path),checkpoint_sha256=digest(path),source_identity_sha256=digest(SOURCE/'identity.json'),protocol_sha256=digest(SOURCE/'protocol.json'),old_evaluation_sha256=digest(SOURCE/'final_dev.json'),generic_weight=str(WEIGHT),generic_weight_sha256=source_id['generic_weight_sha256'],official_AINPAINT_revision=source_id['official_AINPAINT_revision'],official_DINO_revision=source_id['official_DINO_revision'])
    for p,key in [(AINPAINT,'official_AINPAINT_revision'),(DINO,'official_DINO_revision')]:assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==identity[key]
    save(OUT/'identity.json',identity);save(OUT/'dev_rows.json',rows);save(OUT/'config.json',dict(conditions=CONDITIONS,BN_names=BN_NAMES,T=1,batch=1,precision='original BF16/FP32',threshold_logit=0,preflight_fixed='Four pairs midpoint preview targets; repeat first fake target',repeat_tolerance=dict(atol=1e-5,rtol=1e-5),saved_metric_limit=1e-6))
    m=build();load_adaptation(m,cp['adaptation_state']);del cp;m=m.cuda().eval();cache=RGBOnlyCache();before=state_hash(m.state_dict());mode=mode_check(m)
    scope=[dict(name=n,channels=x.num_features,eps=x.eps,affine=x.affine,num_batches_tracked=int(x.num_batches_tracked)) for n,x in head_bn(m).items()]
    print('PREFLIGHT',flush=True);save(OUT/'status.json',dict(status='RUNNING',stage='preflight'));check=fixed_checks(m,cache,rows,old);check['mode']=mode;save(OUT/'preflight.json',check)
    results={};preds={};snapshots={};torch.cuda.reset_peak_memory_stats();calls=m.calls
    for name in CONDITIONS:
        print('EVALUATE',name,flush=True);save(OUT/'status.json',dict(status='RUNNING',stage='evaluate',condition=name));mode_check(m)
        if name=='RUNNING_BN':
            with running_statistics(m) as stats:e,p=evaluate(m,cache,rows,NAME,capture=True,capture_dtype=np.float32)
            save_stats(stats,e['frames'])
        else:
            with image_bn(m):e,p=evaluate(m,cache,rows,NAME,capture=True,capture_dtype=np.float32)
        assert e['scored_frames']==824 and e['frame_identity_sha256']==old['frame_identity_sha256'];assert all(p.grad is None for p in m.parameters())
        snapshots[name]=state_hash(m.state_dict());assert snapshots[name]==before
        persist(name,e);results[name]=e;preds[name]=p
    ref={(r['video_id'],r['frame_index']):r for r in old['frames']};replay={k:max(abs(f[k]-ref[(f['video_id'],f['frame_index'])][k]) for f in results['RUNNING_BN']['frames'] if f[k] is not None) for k in METRICS if k!='prediction_area'}
    replay['macro_differences']={k:results['RUNNING_BN']['macro'][k]-old['macro'][k] for k in METRICS}
    assert max(v for k,v in replay.items() if k!='macro_differences')<=1e-6,replay
    assert digest(path)==identity['checkpoint_sha256'] and digest(SOURCE/'final_dev.json')==identity['old_evaluation_sha256']
    cost=dict(conditions_seconds={n:results[n]['seconds'] for n in CONDITIONS},evaluation_forwards=m.calls-calls,preflight_forwards=calls,optimizer_updates=0,backward_calls=0,BN_calibration_forwards=0,new_frontend_frames=0,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),weights_buffers_sha256_before=before,weights_buffers_sha256_after=snapshots,all_weights_buffers_unchanged=True,dropout_disabled=mode_check(m)['dropout_all_disabled'])
    assert cost['evaluation_forwards']==1648
    del m;torch.cuda.empty_cache();previews(rows,preds)
    s=dict(status='COMPLETE',exit_code=0,identity=identity,dev_rows=rows,BN_scope=scope,preflight=check,replay=replay,results=results,cost=cost,wall_seconds=time.monotonic()-start)
    save(OUT/'summary.json',s);report(s);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


if __name__=='__main__':
    try:run()
    except Exception as error:
        OUT.mkdir(exist_ok=True,parents=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error)));raise
