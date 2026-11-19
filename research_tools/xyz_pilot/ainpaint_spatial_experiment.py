"""ADR-0051 single spatial condition; online LoRA, original full-frame metrics.

Immediate output: one12845-update result and fixed24 preview. A small RGB-only
adapter reuses existing observation files; no geometry/visual frontend is run.
"""
import copy
import json
import random
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .ainpaint_spatial import build,adaptation_state,load_adaptation,AINPAINT,DINO,WEIGHT
from .data import ROOT,read,REPO
from .experiment import seed,csvsave
from .insource_model import losses
from .insource_probe import evaluate
from .joint_experiment import save,digest,preview
from .readout_adaptation import add_area,METRICS
from research_tools.rgbd_slice.experiment import state_hash

SOURCE=ROOT/'pvst_xyz_source_expansion_v1'
OUT=ROOT/'pvst_anakin_ainpaint_spatial_baseline_v1'
NAME='AINPAINT_SPATIAL';STEPS=12845


class RGBOnlyCache:
    def get(self,row,i,partition):
        lo,hi=row[partition+'_range'];assert lo<=i<hi
        e=torch.load(SOURCE/'cache'/row['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
        assert e['frame_index']==i and e['pts']==row['pts'][i] and e['native_hw']==row['native_hw']
        f={k:e[k] for k in ['rgb','gt_work','frame_index','pts','native_hw','native_GT_pixels','supervised']}
        assert list(f['rgb'].shape[:2])==row['work_hw'] and f['gt_work'].shape==f['rgb'].shape[:2]
        for k in ['rgb','gt_work']:f[k]=f[k].cuda()
        return f
    def window(self,row,i,partition,condition):
        assert condition==NAME
        return [self.get(row,i,partition)]


def frozen_hash(m):return state_hash({n:p for n,p in m.backbone.named_parameters() if not p.requires_grad})
def bn_hash(m):return state_hash(dict(m.head.named_buffers()))
def optimizer(m):return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)

def update(m,opt,cache,row,i):
    start=time.monotonic();m.train();opt.zero_grad(set_to_none=True);f=cache.get(row,i,'train');assert f['supervised']
    with torch.autocast('cuda',dtype=torch.bfloat16):out=m([f])
    parts=losses(out,f);assert torch.isfinite(parts).all();parts.sum().backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1,error_if_nonfinite=True)
    opt.step();torch.cuda.synchronize()
    return dict(loss=float(parts.sum()),pixel_BCE=float(parts[0]),soft_Dice=float(parts[1]),gradient_norm=float(norm),seconds=time.monotonic()-start)


def progress(stage,**kw):
    x=dict(status='RUNNING',stage=stage,**kw);save(OUT/'status.json',x);print(json.dumps(x),flush=True)

def persist(stem,e):
    add_area(e);save(OUT/f'{stem}.json',e);csvsave(OUT/f'{stem}_frames.csv',e['frames']);csvsave(OUT/f'{stem}_videos.csv',e['videos'])


def preflight(m,cache,rows):
    byid={r['video_id']:r for r in rows};row=byid['9mygy9_fake'];i=100;f=cache.get(row,i,'train')
    frozen=frozen_hash(m);before={k:v.detach().cpu().clone() for k,v in adaptation_state(m).items()};opt=optimizer(m);records=[]
    seen={};h=m.head.register_forward_pre_hook(lambda mod,args,kw:seen.update(grid=list(args[0].shape),target_size=list(kw['target_size'])),with_kwargs=True)
    for step in range(3):records.append(update(m,opt,cache,row,i))
    h.remove()
    changed=[n for n,p in m.named_parameters() if p.requires_grad and not torch.equal(before[n],p.detach().cpu())]
    grads={n:dict(finite=bool(torch.isfinite(p.grad).all()),norm=float(p.grad.float().norm())) for n,p in m.named_parameters() if p.requires_grad}
    assert all(v['finite'] for v in grads.values()) and all(p.grad is None for p in m.backbone.parameters() if not p.requires_grad)
    assert any('lora_A' in n for n in changed) and any('lora_B' in n for n in changed) and any(n.startswith('head.') for n in changed)
    assert frozen==frozen_hash(m)
    m.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=m([f])
    assert list(out['pixel'].shape)==row['work_hw']
    r=dict(status='PASS',video_id=row['video_id'],target=i,work_hw=row['work_hw'],head_grid=seen['grid'],head_target_hw=seen['target_size'],updates=records,finite_trainable_gradients=True,changed_LoRA_A_tensors=sum('lora_A' in n for n in changed),changed_LoRA_B_tensors=sum('lora_B' in n for n in changed),changed_head_tensors=sum(n.startswith('head.') for n in changed),frozen_DINO_unchanged=True,DINO_forwards=m.calls,discarded=True)
    save(OUT/'preflight.json',r);return r


def save_checkpoint(m,opt,identity,initial_hash):
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    r=dict(condition=NAME,updates=STEPS,epoch=5,adaptation_state={k:v.cpu() for k,v in adaptation_state(m).items()},optimizer=opt.state_dict(),random_state=rng,identity=identity,initial_adaptation_sha256=initial_hash)
    temp=OUT/'final.pt.tmp';torch.save(r,temp);temp.replace(OUT/'final.pt')


def previews(rows,preds):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    # Reuse already validated reference images, no extra reference forward.
    ref=ROOT/'pvst_xyz_dformer_stage4_adaptation_v1/preview/evaluation'
    name='FROZEN_REFERENCE_12845';preview(rows,preds,NAME,'evaluation',output_root=OUT)
    html=['<!doctype html><meta charset="utf-8"><h1>固定24目标：冻结PVST参照 / AINPAINT空间基线</h1><p>每幅RGB/GT/概率0–1，下图0.5二值；仅预览缩放，未按结果选图。</p>']
    for row in rows:
        for i in row['preview_targets']:
            html.append(f'<h3>{row["pair_id"]}/{row["role"]} 帧{i} PTS {row["pts"][i]:.6f}s</h3><div style="display:flex">')
            for label in [name,NAME]:
                stem=f'{label}_{row["video_id"]}_{i:05d}'
                if label==name:
                    for suffix in ['.png','_binary.png']:
                        source=ref/(stem+suffix);assert source.is_file();(folder/'evaluation'/(stem+suffix)).symlink_to(source)
                else:
                    a=preds[(row['video_id'],i)];H,W=a.shape;fac=min(1.,480/W)
                    Image.fromarray(((a>=0)*255).astype(np.uint8)).resize((round(W*fac),round(H*fac)),Image.Resampling.NEAREST).save(folder/'evaluation'/(stem+'_binary.png'))
                html.append(f'<div style="width:50%">{label}<img width="100%" src="evaluation/{stem}.png"><img width="33%" src="evaluation/{stem}_binary.png"></div>')
            html.append('</div>')
    (folder/'index.html').write_text('\n'.join(html))


def report(s):
    a,b=s['reference'],s['evaluation'];lines=['# ANAKIN通用DINOv2＋官方AINPAINT空间定位基线','计算COMPLETE，退出码0。数据/监督曝光/画面范围/评价口径一致；骨干预训练与容量不同，不是XYZ单因素消融。','|指标|冻结PVST12845|空间基线12845|差值|','|---|---:|---:|---:|']
    for k in METRICS:lines.append(f'|{k}|{a["macro"][k]:.8f}|{b["macro"][k]:.8f}|{b["macro"][k]-a["macro"][k]:+.8f}|')
    comparison=[];lines+=['\n|来源|冻结AP|空间AP|差值|冻结/空间Dice|','|---|---:|---:|---:|---|']
    for row in s['protocol']['dev']:
        if row['role']!='fake':continue
        vs=[next(v for v in e['videos'] if v['video_id']==row['video_id']) for e in [a,b]]
        lines.append(f'|{row["pair_id"]}|{vs[0]["pixel_AP"]:.8f}|{vs[1]["pixel_AP"]:.8f}|{vs[1]["pixel_AP"]-vs[0]["pixel_AP"]:+.8f}|{vs[0]["Dice"]:.8f}/{vs[1]["Dice"]:.8f}|')
        for name,e,v in zip(['FROZEN_REFERENCE_12845',NAME],[a,b],vs):
            real=next(r for r in e['videos'] if r['pair_id']==row['pair_id'] and r['role']=='real')
            comparison.append(dict(pair_id=row['pair_id'],condition=name,**{k:real[k] if k=='real_FPR' else v[k] for k in METRICS}))
    csvsave(OUT/'source_comparison.csv',comparison)
    lines+=['\n## 实际协议','仅通用DINOv2 ViT-L/14 registers初始化，官方HighResContextHead新初始化；不加载AINPAINT成品、RAFT或PVST训练权重。r4/alpha8/dropout.25，qkv/proj/fc1/fc2实际匹配见model_scope.json。普通DINO权重冻结，LoRA/head训练模式且正常求梯度。',
    '读取原工作RGB/GT，T1逐帧；ImageNet归一化后右下pad14，x_norm_patchtokens已排除CLS/register且严格核对网格；官方头输出padding尺寸后裁回工作画面，再按原FP32 bilinear恢复原生。无mask-aware crop、增强或padding背景loss。',
    '原11训练来源2569合法目标，原订单前12845逐项一致，每目标5次；40y1x9 fake6–9不监督。原4开发824/412正GT来源隔离。AdamW1e-4/wd.01/clip1，BF16/FP32 BCE+Dice各1/epsilon1，无帧头。',
    '每轮前100既定训练目标以train/no_grad仅更新BN统计，保持官方dropout，不reset BN；500额外前向单列。每轮共同28/新增16观察不代表完整训练集；仅末次开发，不选best/阈值。',
    'AP仅正GT fake帧全原生排序→视频→来源宏平均；real不混入AP。Dice/IoU/召回与FPR固定logit≥0。',
    '训练观察：'+json.dumps({k:{n:v['macro'] for n,v in x.items()} for k,x in s['monitors'].items()},ensure_ascii=False),
    '## 成本与身份',json.dumps(s['cost'],ensure_ascii=False),json.dumps(s['identity'],ensure_ascii=False),
    '与作者原裁剪/损失/数据设置不同，仅保留官方空间架构。单seed反复查看开发集，不称最终测试或三维独立收益。',
    f'结果 {OUT}；预览 {OUT}/preview/index.html；执行HEAD {s["identity"]["execution_head"]}。计算与Git发布单列，完成后不追加其他条件。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def run():
    start=time.monotonic();torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'identity.json').exists():
        assert (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE','Incomplete existing run preserved; no automatic retry'
        print('EXISTING_COMPLETE_REUSED');return
    protocol=read(SOURCE/'protocol.json');orders=read(SOURCE/'TRAIN_EXPANDED_FULL_orders.json')[:STEPS];assert len(orders)==STEPS==5*protocol['Nexp']
    revision=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    identity=dict(execution_head=revision(REPO),official_AINPAINT_revision=revision(AINPAINT),official_DINO_revision=revision(DINO),generic_weight=str(WEIGHT),generic_weight_sha256=digest(WEIGHT),generic_weight_bytes=WEIGHT.stat().st_size,source_protocol_sha256=digest(SOURCE/'protocol.json'),source_config_sha256=digest(SOURCE/'config.json'),source_orders_sha256=digest(SOURCE/'TRAIN_EXPANDED_FULL_orders.json'),reference_metrics_sha256=digest(SOURCE/'TRAIN_EXPANDED_FULL_step12845_dev.json'))
    save(OUT/'identity.json',identity);save(OUT/'protocol.json',protocol);save(OUT/'orders.json',orders)
    save(OUT/'config.json',dict(condition=NAME,seed=20260909,epochs=5,updates=STEPS,batch=1,T=1,lora=dict(rank=4,alpha=8,dropout=.25,target_modules=['qkv','proj','fc1','fc2']),lr=1e-4,weight_decay=.01,clip_grad=1,precision='BF16 forward/FP32 BCE+softDice eps1',BN_train_only_per_epoch=100,padding='normalize RGB then right/bottom zero to14; head target padded; crop then native bilinear',new_MoGe=0,new_DFormer=0))
    cache=RGBOnlyCache();seed();m=build().cuda();frozen=frozen_hash(m);initial_hash=state_hash(adaptation_state(m));initial_bn=bn_hash(m)
    scope=dict(base_frozen_parameters=sum(p.numel() for p in m.backbone.parameters() if not p.requires_grad),LoRA_parameters=sum(p.numel() for p in m.backbone.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in m.head.parameters()),matched_modules=[n for n,mod in m.backbone.named_modules() if hasattr(mod,'lora_A')],trainable_names=[n for n,p in m.named_parameters() if p.requires_grad])
    save(OUT/'model_scope.json',scope);save(OUT/'initialization.json',dict(frozen_DINO_sha256=frozen,adaptation_sha256=initial_hash,head_BN_sha256=initial_bn))
    torch.save({k:v.cpu() for k,v in adaptation_state(m).items()},OUT/'initial_adaptation.pt')
    progress('preflight');check=preflight(m,cache,protocol['expanded']);del m;torch.cuda.empty_cache()
    seed();m=build().cuda();assert frozen_hash(m)==frozen and state_hash(adaptation_state(m))==initial_hash
    opt=optimizer(m);assert not opt.state;byid={r['video_id']:r for r in protocol['expanded']};history=[];monitors={};training=bnseconds=monitorseconds=0.;bnforwards=0
    torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    with (OUT/'updates.jsonl').open('x') as log:
        for step,item in enumerate(orders,1):
            rec=update(m,opt,cache,byid[item['video_id']],item['target']);training+=rec['seconds'];rec.update(updates=step,**item);log.write(json.dumps(rec)+'\n');log.flush()
            if step%500==0:progress('train',updates=step,U=STEPS,remaining_training_seconds=training/step*(STEPS-step))
            if step%2569==0:
                epoch=step//2569;ep=orders[step-2569:step];t=time.monotonic();m.train()
                with torch.no_grad():
                    for x in ep[:100]:
                        f=cache.get(byid[x['video_id']],x['target'],'train')
                        with torch.autocast('cuda',dtype=torch.bfloat16):m([f])
                        bnforwards+=1
                torch.cuda.synchronize();bnseconds+=time.monotonic()-t
                common,_=evaluate(m,cache,protocol['original'],NAME,monitor=True);extra,_=evaluate(m,cache,protocol['added'],NAME,monitor=True)
                persist(f'epoch{epoch}_common_monitor',common);persist(f'epoch{epoch}_new_monitor',extra);monitors[str(epoch)]=dict(common=common,new=extra);save(OUT/'monitors.json',monitors);monitorseconds+=common['seconds']+extra['seconds']
                history.append(dict(epoch=epoch,updates=step,training_seconds=training,BN_statistics_forwards=bnforwards,common_AP=common['macro']['pixel_AP'],new_AP=extra['macro']['pixel_AP']));csvsave(OUT/'history.csv',history)
                progress('epoch_complete',epoch=epoch,updates=step,remaining_training_seconds=training/step*(STEPS-step))
    assert step==STEPS and bnforwards==500 and frozen_hash(m)==frozen and bn_hash(m)!=initial_bn
    save_checkpoint(m,opt,identity,initial_hash);progress('final_dev',updates=step);ev,preds=evaluate(m,cache,protocol['dev'],NAME,capture=True,capture_dtype=np.float32);persist('final_dev',ev)
    ref=read(SOURCE/'TRAIN_EXPANDED_FULL_step12845_dev.json');assert ev['scored_frames']==824 and ev['frame_identity_sha256']==ref['frame_identity_sha256']
    cost=dict(training_seconds=training,BN_statistics_seconds=bnseconds,monitor_seconds=monitorseconds,dev_seconds=ev['seconds'],train_eval_wall_seconds=time.monotonic()-begin,optimizer_updates=step,supervised_presentations=step,BN_statistics_forwards=bnforwards,monitor_forwards=220,dev_forwards=824,formal_DINO_forwards=m.calls,preflight_DINO_forwards=check['DINO_forwards'],new_MoGe=0,new_DFormer=0,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),frozen_DINO_unchanged=frozen_hash(m)==frozen,head_BN_changed=bn_hash(m)!=initial_bn)
    assert m.calls==STEPS+500+220+824
    del m,opt;torch.cuda.empty_cache();previews(protocol['dev'],preds)
    s=dict(status='COMPLETE',exit_code=0,identity=identity,protocol=protocol,scope=scope,preflight=check,reference=ref,evaluation=ev,monitors=monitors,cost=cost,wall_seconds=time.monotonic()-start)
    save(OUT/'summary.json',s);report(s);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


if __name__=='__main__':
    try:run()
    except Exception as e:
        OUT.mkdir(exist_ok=True,parents=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(e)));raise
