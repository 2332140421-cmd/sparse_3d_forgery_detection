"""ADR-0050: one real DFormer stage4 fine-tune, 12845 frozen-order updates."""
import copy
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch
from .data import ROOT,REPO,read
from .insource_model import XYZLocalization,losses
from .insource_probe import PartitionCache,update,evaluate
from .joint_experiment import save,digest,checkpoint,preview
from .experiment import seed,csvsave
from .readout_adaptation import add_area,METRICS
from .dformer_stage4 import Stage4Localization,OfficialStage4,backbone_input,stage4_parameter
from research_tools.rgbd_slice.model import backbone
from research_tools.rgbd_slice.experiment import state_hash

SOURCE=ROOT/'pvst_xyz_source_expansion_v1'
OUT=ROOT/'pvst_xyz_dformer_stage4_adaptation_v1'
NAME='DFORMER_STAGE4_TRAINABLE';MODE='XYZ_TEMPORAL';STEPS=12845
FIXED=[('9mygy9_fake',100),('4xeqd7_fake',182),('7e2e57_fake',60),('95q8g9_fake',36)]
TOL=dict(split_full=dict(atol=2e-5,rtol=3e-5),original_cache=dict(atol=.002,rtol=.002),pixel=dict(atol=.002,rtol=.002))


def frozen_hash(stage):
    return state_hash({k:v for k,v in stage.encoder.state_dict().items() if k not in {n for n,p in stage.encoder.named_parameters() if p.requires_grad}})


def diff(a,b,tolerance):
    dtype_a=str(a.dtype);dtype_b=str(b.dtype);a=a.detach().float();b=b.detach().float();assert a.shape==b.shape
    error=(a-b).abs();allowed=tolerance['atol']+tolerance['rtol']*b.abs()
    record=dict(shape=list(a.shape),dtype_a=dtype_a,dtype_b=dtype_b,max_abs=float(error.max()),mean_abs=float(error.mean()),outside=int((error>allowed).sum()),**tolerance)
    assert record['outside']==0,record
    return record


class PrefixCache(PartitionCache):
    def __init__(self,c):super().__init__(c,cache_root=SOURCE/'cache')
    def get(self,row,i,partition,image_only=False,device='cuda'):
        assert not image_only
        f=super().get(row,i,partition,False,device)
        p=torch.load(OUT/'prefix'/row['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
        assert p['frame_index']==i and p['pts']==row['pts'][i] and p['video_sha256']==row['video_sha256'] and p['padded_hw']==row['padded_hw']
        f['prefix_rgb']=p['rgb'].to(device);f['prefix_depth']=p['depth'].to(device)
        return f


def build(c,provider,initial):
    seed();enc=backbone(provider);m=Stage4Localization(c['h'],enc);m.pvst.load_state_dict(initial,strict=True);m=m.cuda()
    opt=torch.optim.AdamW([{'params':m.pvst.parameters(),'lr':1e-4},{'params':[p for p in m.stage4.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=.01)
    assert not opt.state and state_hash(m.pvst.state_dict())==state_hash(initial)
    return m,opt


def persist(stem,e):
    add_area(e);save(OUT/f'{stem}.json',e);csvsave(OUT/f'{stem}_frames.csv',e['frames']);csvsave(OUT/f'{stem}_videos.csv',e['videos'])


def progress(stage,**fields):
    r=dict(status='RUNNING',stage=stage,**fields);save(OUT/'status.json',r);print(json.dumps(r),flush=True)


@torch.no_grad()
def prepare_prefix(c,provider,rows):
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();enc=backbone(provider,OUT/'official_encoder_load.json').cuda();stage=OfficialStage4(enc);before=state_hash(enc.state_dict())
    cache=PartitionCache(c,cache_root=SOURCE/'cache');count=0
    for row in rows:
        partition='train' if row['split']=='train' else 'evaluation';folder=OUT/'prefix'/row['video_id'];folder.mkdir(parents=True)
        for i in range(row['actual_frames']):
            f=cache.get(row,i,partition);x,depth=backbone_input(f)
            with torch.autocast('cuda',dtype=torch.bfloat16):_,rgb,e=stage.prefix(x,depth)
            assert rgb.dtype==e.dtype==torch.float32,'Keep real raw stage3 state precision'
            packet=dict(frame_index=i,pts=row['pts'][i],video_sha256=row['video_sha256'],padded_hw=row['padded_hw'],rgb=rgb.cpu(),depth=e.cpu())
            torch.save(packet,folder/f'{i:05d}.pt');count+=1
        progress('prefix',video=row['video_id'],unique_prefix_frames=count)
    assert state_hash(enc.state_dict())==before
    cost=dict(unique_prefix_frames=count,frozen_prefix_extractions=count,new_MoGe_frames=0,new_full_frozen_visual_cache=0,seconds=time.monotonic()-started,bytes=sum(p.stat().st_size for p in (OUT/'prefix').glob('*/*.pt')),raw_dtype='FP32',low_features='Original observation visual[0:3], half storage/FP32 load; not duplicated',encoder_unchanged=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    save(OUT/'prefix_cost.json',cost);del stage,enc;torch.cuda.empty_cache();return cost


def preflight(c,provider,initial,rows):
    torch.cuda.reset_peak_memory_stats();byid={r['video_id']:r for r in rows};cache=PrefixCache(c);oldcache=PartitionCache(c,cache_root=SOURCE/'cache')
    m,opt=build(c,provider,initial);m.eval();plain=XYZLocalization(c['h']).cuda().eval();plain.load_state_dict(initial,strict=True)
    original_frozen=frozen_hash(m.stage4);records=[]
    with torch.no_grad():
        for vid,i in FIXED:
            progress('parity',video_id=vid,target=i)
            row=byid[vid];part='train' if row['split']=='train' else 'evaluation';f=cache.get(row,i,part);x,d=backbone_input(f);seen={}
            handle=m.stage4.encoder.stages[2][-1].register_forward_hook(lambda mod,args,result:seen.update(rgb=result[0].permute(0,3,1,2),depth=result[1].permute(0,3,1,2)))
            with torch.autocast('cuda',dtype=torch.bfloat16):full,_=m.stage4.encoder(x,d)
            handle.remove()
            entry={key:diff(f['prefix_'+key],seen[key],TOL['split_full']) for key in ['rgb','depth']}
            with torch.autocast('cuda',dtype=torch.bfloat16):top=m.stage4(f['prefix_rgb'],f['prefix_depth'])
            feature=[diff(full[j][0].half().float(),f['visual'][j],TOL['original_cache']) for j in range(4)]
            topcheck=diff(top,full[3],TOL['split_full'])
            fs=cache.window(row,i,part,MODE);original=oldcache.window(row,i,part,MODE)
            with torch.autocast('cuda',dtype=torch.bfloat16):a=m(fs,True)['pixel'];b=plain(original,True)['pixel']
            records.append(dict(video_id=vid,target=i,pts=row['pts'][i],entry=entry,split_top=topcheck,features_vs_original=feature,pixel=diff(a,b,TOL['pixel']),depth_nonzero=int(f['prefix_depth'].count_nonzero()),depth_abs_mean=float(f['prefix_depth'].abs().mean())))
    del plain;torch.cuda.empty_cache()
    # Passive retention on real per-observation stage4 output, no changed attention path.
    row=byid['9mygy9_fake'];target=100;outputs=[]
    def retain(mod,args,result):result.retain_grad();outputs.append(result)
    handle=m.stage4.register_forward_hook(retain);before={n:p.detach().clone() for n,p in m.stage4.encoder.named_parameters() if p.requires_grad}
    result=update(m,opt,cache,row,target,MODE);handle.remove()
    changed=[n for n,p in m.stage4.encoder.named_parameters() if p.requires_grad and not torch.equal(before[n],p)]
    grads={n:dict(present=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else False,norm=float(p.grad.float().norm()) if p.grad is not None else None) for n,p in m.stage4.encoder.named_parameters() if p.requires_grad}
    assert changed and all(g['present'] and g['finite'] for g in grads.values())
    history=[dict(slot=i,norm=float(v.grad.float().norm()),finite=bool(torch.isfinite(v.grad).all())) for i,v in enumerate(outputs)]
    assert len(history)>1 and all(x['finite'] and x['norm']>0 for x in history[:-1]),'History stage4 gradient detached'
    assert original_frozen==frozen_hash(m.stage4) and not any(mod.training for mod in m.stage4.encoder.modules())
    boundary=dict(trainable_names=[n for n,p in m.stage4.encoder.named_parameters() if p.requires_grad],stage4_parameters=sum(p.numel() for p in m.stage4.parameters() if p.requires_grad),PVST_parameters=sum(p.numel() for p in m.pvst.parameters()),official_total_parameters=sum(p.numel() for p in m.stage4.parameters()),no_separate_output_norm=True,classification_head_absent=True)
    save(OUT/'boundary.json',boundary)
    check=dict(status='PASS',fixed_parity=records,real_update=result,stage4_gradients=grads,changed_stage4_tensors=changed,history_output_gradients=history,frozen_parameters_buffers_unchanged=True,discarded=True,tolerances=TOL,validation_full_official_forwards=len(FIXED),validation_split_stage4_forwards=m.stage4.calls,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    save(OUT/'preflight.json',check);del m,opt,outputs,before;torch.cuda.empty_cache();return check


def previews(rows,reference,new):
    folder=OUT/'preview';folder.mkdir(exist_ok=True);names=['FROZEN_REFERENCE_12845',NAME+'_12845'];preds=[reference,new]
    h=['<!doctype html><meta charset="utf-8"><h1>固定24目标：冻结参照 / DFormer真实末级微调</h1><p>各图RGB/GT/概率（灰阶0–1），下图为logit≥0二值；仅预览缩放，不按结果选择。</p>']
    for name,pred in zip(names,preds):
        assert len(pred)==24;preview(rows,pred,name,'evaluation',output_root=OUT)
        for (vid,i),a in pred.items():
            H,W=a.shape;f=min(1.,480/W);Image.fromarray(((a>=0)*255).astype(np.uint8)).resize((round(W*f),round(H*f)),Image.Resampling.NEAREST).save(folder/'evaluation'/f'{name}_{vid}_{i:05d}_binary.png')
    for r in rows:
        for i in r['preview_targets']:
            h.append(f'<h3>{r["pair_id"]}/{r["role"]} 帧{i} PTS {r["pts"][i]:.6f}s</h3><div style="display:flex">')
            for name in names:
                path=f'evaluation/{name}_{r["video_id"]}_{i:05d}';h.append(f'<div style="width:50%">{name}<a href="{path}.png"><img width="100%" src="{path}.png"></a><img width="33%" src="{path}_binary.png"></div>')
            h.append('</div>')
    (folder/'index.html').write_text('\n'.join(h))


def report(s):
    old,new=s['reference'],s['evaluation'];lines=['# DFormer真实末级微调开发对照','计算COMPLETE，退出码0。唯一新增一组12845更新，冻结参照直接读取原同预算结果。','|指标|冻结12845|末级可训练12845|差值|','|---|---:|---:|---:|']
    for k in METRICS:lines.append(f'|{k}|{old["macro"][k]:.8f}|{new["macro"][k]:.8f}|{new["macro"][k]-old["macro"][k]:+.8f}|')
    lines+=['\n|来源|冻结AP|微调AP|差值|冻结/微调Dice|','|---|---:|---:|---:|---|'];table=[]
    for r in s['protocol']['dev']:
        if r['role']!='fake':continue
        a,b=[next(v for v in e['videos'] if v['video_id']==r['video_id']) for e in [old,new]]
        lines.append(f'|{r["pair_id"]}|{a["pixel_AP"]:.8f}|{b["pixel_AP"]:.8f}|{b["pixel_AP"]-a["pixel_AP"]:+.8f}|{a["Dice"]:.8f}/{b["Dice"]:.8f}|')
        for n,e,v in [('FROZEN_REFERENCE_12845',old,a),(NAME,new,b)]:
            real=next(q for q in e['videos'] if q['pair_id']==r['pair_id'] and q['role']=='real')
            table.append(dict(pair_id=r['pair_id'],condition=n,**{k:real[k] if k=='real_FPR' else v[k] for k in METRICS}))
    csvsave(OUT/'source_comparison.csv',table)
    lines+=['\n## 实际边界与协议','官方forward依次执行四组RGB/depth下采样和block。缓存前三组结束后NCHW的原始RGB256和depth128状态（FP32），不是归一化输出猜测或零depth中间状态。训练实际调用downsample_layers[3]、downsample_layers_e[3]、stages[3]两个block；官方没有独立输出norm，相关norm位于这些模块内部。最后block drop_depth=True，不加分类尾部。',
        '末级输出按原缓存half→FP32路径做可微cast；前三层visual复用原缓存，XYZ/MoGe完全复用。每更新重算当前及所有唯一历史观测末级，不跨更新缓存C/R或末级输出。官方backbone始终eval，末级requires_grad=True，全部BN buffers/DropPath策略保持。',
        '同PVST随机初值严格加载原initial.pt/XYZ，官方通用末级初值，全新AdamW两组lr1e-4/1e-5、wd.01/clip1、BF16前向/FP32两项空间loss。只使用原扩充订单前12845项、11来源全帧，5完整遍历；40y1x9 fake6–9仍不监督计分。完整824开发和原指标规则不变，无帧分类或阈值搜索。',
        '实际参数边界：'+json.dumps(s['boundary'],ensure_ascii=False),
        '冻结参照12845步完整逐帧/视频结果只读引用；只补24固定预览前向，不重训、不以25690步作匹配主表。',
        '\n## 成本和核验',json.dumps(s['prefix_cost'],ensure_ascii=False),json.dumps(s['cost'],ensure_ascii=False),
        '完整前后向检查更新丢弃，正式重置；split/full、缓存特征与同PVST像素一致性、历史输出梯度及冻结权重/buffer检验证据见preflight.json。',
        f'执行HEAD {s["identity"]["execution_head"]}；总流程{s["wall_seconds"]:.2f}秒。完整产物 {OUT}；固定预览 {OUT}/preview/index.html。',
        '本轮是单seed开发筛选，不证明骨干错误、三维独立贡献或最终测试性能。计算与Git发布分开；不追加预算/seed/结构。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def run():
    started=time.monotonic();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    protocol=read(SOURCE/'protocol.json');c=read(SOURCE/'config.json');provider=read(SOURCE/'provider_config.json');rows=protocol['expanded']+protocol['dev']
    orders=read(SOURCE/'TRAIN_EXPANDED_FULL_orders.json')[:STEPS];assert len(orders)==STEPS==5*protocol['Nexp']
    assert c['scale']==12.880994319915771 and c['h']==.5
    initial=torch.load(SOURCE/'initial.pt',weights_only=True,map_location='cpu')['XYZ']
    refcp=SOURCE/'TRAIN_EXPANDED_FULL_step012845.pt';reference=read(SOURCE/'TRAIN_EXPANDED_FULL_step12845_dev.json')
    c=copy.deepcopy(c);c.update(conditions=[NAME],epochs=5,updates_per_condition=STEPS,evaluation_steps=[STEPS],targets_per_epoch=2569,stage4_lr=1e-5)
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_protocol_sha256=digest(SOURCE/'protocol.json'),source_config_sha256=digest(SOURCE/'config.json'),source_orders_sha256=digest(SOURCE/'TRAIN_EXPANDED_FULL_orders.json'),initial_file_sha256=digest(SOURCE/'initial.pt'),initial_PVST_sha256=state_hash(initial),reference_checkpoint=str(refcp),reference_checkpoint_sha256=digest(refcp),reference_metrics_sha256=digest(SOURCE/'TRAIN_EXPANDED_FULL_step12845_dev.json'),provider=read(SOURCE/'identity.json')['provider'])
    save(OUT/'identity.json',identity);save(OUT/'config.json',c);save(OUT/'protocol.json',protocol);save(OUT/'orders.json',orders)
    prefix_cost=prepare_prefix(c,provider,rows);check=preflight(c,provider,initial,rows);cache=PrefixCache(c)
    # Frozen reference is replayed only on the fixed 24 preview targets.
    cp=torch.load(refcp,map_location='cpu',weights_only=False);assert cp['updates']==STEPS and cp['condition']=='TRAIN_EXPANDED_FULL' and cp['initial_sha256']==state_hash(initial)
    plain=XYZLocalization(c['h']).cuda().eval();plain.load_state_dict(cp['state_dict'],strict=True)
    short=copy.deepcopy(protocol['dev'])
    for r in short:r['evaluation_targets']=r['preview_targets']
    ref24,refpred=evaluate(plain,PartitionCache(c,cache_root=SOURCE/'cache'),short,MODE,capture=True,capture_dtype=np.float32);assert ref24['scored_frames']==24
    oldframes={(f['video_id'],f['frame_index']):f for f in reference['frames']};replay=[]
    for f in ref24['frames']:
        old=oldframes[(f['video_id'],f['frame_index'])];delta={k:f[k]-old[k] for k in ['pixel_AP','Dice','IoU','GT_recall','real_FPR'] if f[k] is not None}
        assert all(abs(v)<.001 for v in delta.values());replay.append(dict(video_id=f['video_id'],frame_index=f['frame_index'],delta=delta))
    save(OUT/'reference_preview_replay.json',replay);del plain,cp;torch.cuda.empty_cache()
    m,opt=build(c,provider,initial);frozen_before=frozen_hash(m.stage4);official_initial=state_hash(m.stage4.encoder.state_dict());stage_initial=state_hash({n:p for n,p in m.stage4.encoder.named_parameters() if p.requires_grad})
    save(OUT/'initialization.json',dict(PVST_sha256=state_hash(m.pvst.state_dict()),official_encoder_sha256=official_initial,trainable_stage4_sha256=stage_initial,frozen_sha256=frozen_before,optimizer_fresh=True))
    byid={r['video_id']:r for r in protocol['expanded']};history=[];online=[];monitors={};train_calls=0;train_seconds=0.;monitor_seconds=0.;torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    with (OUT/'updates.jsonl').open('x') as log:
        for step,item in enumerate(orders,1):
            calls=m.stage4.calls;record=update(m,opt,cache,byid[item['video_id']],item['target'],MODE);train_seconds+=record['seconds'];train_calls+=m.stage4.calls-calls
            record.update(updates=step,video_id=item['video_id'],target=item['target'],epoch=item['epoch'],stage4_forwards=m.stage4.calls-calls);log.write(json.dumps(record)+'\n');log.flush();online.append(record)
            if step%500==0 or step%2569==0:
                h=dict(updates=step,**{k:float(np.mean([x[k] for x in online])) for k in ['loss','pixel_BCE','soft_Dice']},stage4_forwards=sum(x['stage4_forwards'] for x in online),seconds=sum(x['seconds'] for x in online));history.append(h);online=[];csvsave(OUT/'history.csv',history);progress('train',updates=step,U=STEPS,remaining_train_seconds=train_seconds/step*(STEPS-step))
            if step%2569==0:
                epoch=step//2569;common,_=evaluate(m,cache,protocol['original'],MODE,monitor=True);extra,_=evaluate(m,cache,protocol['added'],MODE,monitor=True)
                persist(f'epoch{epoch}_common_monitor',common);persist(f'epoch{epoch}_new_monitor',extra);monitors[str(epoch)]=dict(common=common,new=extra);monitor_seconds+=common['seconds']+extra['seconds'];save(OUT/'monitors.json',monitors)
    assert step==STEPS and frozen_before==frozen_hash(m.stage4)
    assert stage_initial!=state_hash({n:p for n,p in m.stage4.encoder.named_parameters() if p.requires_grad})
    checkpoint(OUT/'DFORMER_STAGE4_TRAINABLE_step012845.pt',m,opt,STEPS,NAME,identity,state_hash(initial))
    progress('final_dev',updates=step);calls=m.stage4.calls
    ev,pred=evaluate(m,cache,protocol['dev'],MODE,capture=True,capture_dtype=np.float32);persist('final_dev',ev)
    assert ev['scored_frames']==824 and ev['frame_identity_sha256']==reference['frame_identity_sha256']
    assert frozen_before==frozen_hash(m.stage4)
    cost=dict(training_seconds=train_seconds,monitor_seconds=monitor_seconds,final_dev_seconds=ev['seconds'],train_and_eval_wall_seconds=time.monotonic()-begin,training_stage4_forwards=train_calls,all_formal_stage4_forwards=m.stage4.calls,dev_stage4_forwards=m.stage4.calls-calls,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),frozen_parameters_buffers_unchanged=True,stage4_parameters_changed=True)
    del m,opt;torch.cuda.empty_cache();previews(protocol['dev'],refpred,pred)
    summary=dict(status='COMPLETE',exit_code=0,identity=identity,config=c,protocol=protocol,boundary=read(OUT/'boundary.json'),prefix_cost=prefix_cost,preflight=check,reference=reference,evaluation=ev,monitors=monitors,cost=cost,history=history,updates=STEPS,wall_seconds=time.monotonic()-started,new_MoGe_frames=0)
    save(OUT/'summary.json',summary);report(summary);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


def main():
    if (OUT/'identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':print('EXISTING_COMPLETE_REUSED');return
        raise SystemExit('Existing incomplete run preserved, no automatic retry')
    try:run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error)));raise


if __name__=='__main__':main()
