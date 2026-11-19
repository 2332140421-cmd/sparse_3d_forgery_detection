"""ADR-0053: matched native versus full-frame-degraded 288-patch experiment."""
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .native_patch_data import SOURCE,OUT,SIZE,STRIDE,STEPS,CONDITIONS,NativeData,EvalData,TiledModel,prepare,patch_logits,masked_losses,tiled_logits,tiles
from .ainpaint_spatial import build,adaptation_state,load_adaptation,WEIGHT,DINO,AINPAINT
from .ainpaint_spatial_experiment import frozen_hash,optimizer
from .insource_probe import evaluate
from .data import ROOT,REPO,read
from .experiment import seed,csvsave
from .joint_experiment import save,digest
from .readout_adaptation import add_area,METRICS
from research_tools.rgbd_slice.experiment import state_hash

BATCH=8


def progress(stage,**kw):
    x=dict(status='RUNNING',stage=stage,**kw);save(OUT/'status.json',x);print(json.dumps(x),flush=True)


def update(m,opt,data,row,item,condition):
    begin=time.monotonic();m.train();opt.zero_grad(set_to_none=True)
    rgb,gt,valid=data.training_crop(row,item,condition)
    gt=torch.as_tensor(gt.copy(),device='cuda');valid=torch.as_tensor(valid,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):z=patch_logits(m,[rgb])[0]
    parts=masked_losses(z,gt,valid);assert torch.isfinite(parts).all();parts.sum().backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1,error_if_nonfinite=True)
    opt.step();torch.cuda.synchronize()
    return dict(loss=float(parts.sum()),pixel_BCE=float(parts[0]),soft_Dice=float(parts[1]),gradient_norm=float(norm),seconds=time.monotonic()-begin,GT_pixels=int(gt.sum()),valid_pixels=int(valid.sum()))


@torch.no_grad()
def BN_statistics(m,data,byid,plan,condition):
    start=time.monotonic();m.train()
    for item in plan:
        rgb,_,_=data.training_crop(byid[item['video_id']],item,condition)
        with torch.autocast('cuda',dtype=torch.bfloat16):patch_logits(m,[rgb])
    torch.cuda.synchronize();return time.monotonic()-start


def preflight(initial,initial_hash,protocol,plan,data):
    byid={r['video_id']:r for r in protocol['expanded']}
    wanted=[next(x for x in plan if x['foreground_guided']),next(x for x in plan if byid[x['video_id']]['role']=='real')]
    evidence=[]
    for item in wanted:
        row=byid[item['video_id']];a,g,v=data.training_crop(row,item,CONDITIONS[0]);b,h,w=data.training_crop(row,item,CONDITIONS[1])
        assert np.array_equal(g,h) and np.array_equal(v,w)
        evidence.append(dict(video_id=row['video_id'],frame_index=item['target'],PTS=row['pts'][item['target']],native_hw=row['native_hw'],crop=[item['y'],item['x'],SIZE,SIZE],same_GT_and_valid=True,RGB_different_values=int((a!=b).sum()),RGB_mean_abs_difference=float(np.abs(a.astype(float)-b).mean()),GT_pixels=int(g.sum())))
    checks={};row=next(r for r in protocol['dev'] if r['video_id']=='7e2e57_fake')
    for condition in CONDITIONS:
        seed();m=build().cuda();load_adaptation(m,initial);assert state_hash(adaptation_state(m))==initial_hash
        fixed=frozen_hash(m);opt=optimizer(m);torch.cuda.reset_peak_memory_stats()
        before={k:v.detach().cpu().clone() for k,v in adaptation_state(m).items()}
        rec=[update(m,opt,data,byid[x['video_id']],x,condition) for x in wanted]
        changed=[k for k,v in adaptation_state(m).items() if not torch.equal(v.detach().cpu(),before[k])]
        assert any('lora_' in k for k in changed) and any(k.startswith('head.') and k.endswith('weight') for k in changed)
        m.eval();native,coverage=tiled_logits(m,data.rgb(row,0),row['work_hw'],condition,BATCH)
        assert native.shape==tuple(row['native_hw']) and np.isfinite(native).all() and frozen_hash(m)==fixed
        checks[condition]=dict(status='PASS',updates=rec,changed_trainable_tensors=len(changed),tiled_video=row['video_id'],tiled_frame=0,coverage=coverage,patch_forwards=m.calls,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),frozen_DINO_unchanged=True,discarded=True)
        del m,opt,before;torch.cuda.empty_cache()
    save(OUT/'preflight.json',dict(status='PASS',inputs=evidence,conditions=checks,inference_batch=BATCH,discarded=True))
    return checks


def checkpoint(m,opt,condition,identity,initial_hash):
    import random
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    p=OUT/f'{condition}_final.pt';tmp=p.with_suffix('.tmp')
    torch.save(dict(condition=condition,updates=STEPS,epoch=5,adaptation_state={k:v.cpu() for k,v in adaptation_state(m).items()},optimizer=opt.state_dict(),random_state=rng,identity=identity,initial_adaptation_sha256=initial_hash),tmp);tmp.replace(p)


def previews(protocol,data,pred):
    folder=OUT/'preview';folder.mkdir(exist_ok=True);html=['<!doctype html><meta charset="utf-8"><h1>固定24目标：NATIVE / LOWDETAIL</h1><p>原生RGB、官方GT；两条件原生概率[0,1]及0.5二值。点击图像看原生尺寸；未按结果选图。tile288/stride144，全图均匀logit融合。</p>']
    contact=[]
    for row in protocol['dev']:
        for i in row['preview_targets']:
            vid=row['video_id'];rgb=data.rgb(row,i);gt=data.gt(row,i);stem=f'{vid}_{i:05d}'
            Image.fromarray(rgb).save(folder/(stem+'_rgb.png'));Image.fromarray(gt.astype(np.uint8)*255).save(folder/(stem+'_gt.png'))
            html.append(f'<h3>{row["pair_id"]}/{row["role"]} 帧{i} PTS {row["pts"][i]:.6f}s</h3><div style="display:flex">')
            for suffix,label in [('_rgb','原生RGB'),('_gt','官方GT')]:html.append(f'<div>{label}<a href="{stem}{suffix}.png"><img width="300" src="{stem}{suffix}.png"></a></div>')
            html.append('</div><div style="display:flex">');thumbs=[rgb,np.repeat(gt[...,None]*255,3,axis=-1).astype(np.uint8)]
            for name in CONDITIONS:
                z=pred[name][(vid,i)];p=1/(1+np.exp(-np.clip(z,-80,80)));prob=(p*255).round().astype(np.uint8)
                path=f'{name}_{stem}';Image.fromarray(prob).save(folder/(path+'_prob.png'));Image.fromarray((z>=0).astype(np.uint8)*255).save(folder/(path+'_binary.png'))
                html.append(f'<div>{name}<a href="{path}_prob.png"><img width="480" src="{path}_prob.png"></a><br><a href="{path}_binary.png"><img width="240" src="{path}_binary.png"></a></div>')
                thumbs.append(np.repeat(prob[...,None],3,axis=-1))
            html.append('</div>')
            if row['role']=='fake' and i==row['preview_targets'][1]:
                height=round(rgb.shape[0]*320/rgb.shape[1]);contact.append(np.concatenate([np.asarray(Image.fromarray(x).resize((320,height))) for x in thumbs],axis=1))
    if contact:Image.fromarray(np.concatenate(contact,axis=0)).save(folder/'fixed_fake_mid_contact.png')
    (folder/'index.html').write_text('\n'.join(html))


def report(s):
    a,b=[s['results'][n]['evaluation'] for n in CONDITIONS];lines=['# ANAKIN原生局部细节匹配对照','两条件各12845更新，原824开发帧完整原生滑窗评价；计算COMPLETE，退出码0。','|指标|NATIVE_PATCH|LOWDETAIL_PATCH|NATIVE−LOWDETAIL|','|---|---:|---:|---:|']
    for k in METRICS:lines.append(f'|{k}|{a["macro"][k]:.8f}|{b["macro"][k]:.8f}|{a["macro"][k]-b["macro"][k]:+.8f}|')
    table=[];diffs=[];lines+=['\n|来源|恒定AP|NATIVE AP|LOW AP|差值|NATIVE/LOW Dice|NATIVE/LOW召回|','|---|---:|---:|---:|---:|---|---|']
    for row in s['protocol']['dev']:
        if row['role']!='fake':continue
        va,vb=[next(v for v in e['videos'] if v['video_id']==row['video_id']) for e in [a,b]]
        ra,rb=[next(v for v in e['videos'] if v['pair_id']==row['pair_id'] and v['role']=='real') for e in [a,b]]
        assert va['constant_AP']==vb['constant_AP']
        lines.append(f'|{row["pair_id"]}|{va["constant_AP"]:.8f}|{va["pixel_AP"]:.8f}|{vb["pixel_AP"]:.8f}|{va["pixel_AP"]-vb["pixel_AP"]:+.8f}|{va["Dice"]:.8f}/{vb["Dice"]:.8f}|{va["GT_recall"]:.8f}/{vb["GT_recall"]:.8f}|')
        for name,v,r in zip(CONDITIONS,[va,vb],[ra,rb]):table.append(dict(pair_id=row['pair_id'],condition=name,constant_AP=v['constant_AP'],**{k:r[k] if k=='real_FPR' else v[k] for k in METRICS}))
        diffs.append(dict(pair_id=row['pair_id'],**{k:ra[k]-rb[k] if k=='real_FPR' else va[k]-vb[k] for k in METRICS}))
    csvsave(OUT/'source_comparison.csv',table);csvsave(OUT/'source_differences.csv',diffs)
    lines+=['\n## 冻结协议','NATIVE从原视频原生RGB直接裁288；LOW先整帧INTER_AREA缩至原work_hw，再整帧INTER_LINEAR恢复原生，再裁相同位置。GT不退化；小边原生padding及模型pad14排除监督。模型为同一官方DINOv2通用初始化、LoRA4/8/.25和HighResContextHead，无XYZ/时间；只改变输入策略。',
        '原11来源2569目标×5轮，原订单前12845；共享crop_plan与BN_plan。real均匀，fake正GT以.5概率先均匀选正像素再均匀选合法包含它的crop，其余均匀，不拒绝小目标。原40y1x9 fake6–9不监督。AdamW1e-4/wd.01/clip1，BF16/FP32 BCE+Dice各1/eps1、单crop更新，无其他增强。每轮同100训练crop仅BN统计，两条件各500，使用running BN评价。',
        '盲滑窗tile288/stride144/末端贴边，无GT坐标，FP32 logit均匀平均后全原生计分。AP仅正GT fake帧→视频→来源宏平均，real FPR与fake背景FPR独立、logit≥0。旧全图384仅历史，不是本轮严格因果对照。无训练观察专项，不用在线loss代替训练集定位成绩。',
        '## 实际资源',json.dumps(s['preparation'],ensure_ascii=False),json.dumps({n:s['results'][n]['cost'] for n in CONDITIONS},ensure_ascii=False),json.dumps(s['identity'],ensure_ascii=False),
        f'原生数据与两条件产物 {OUT}；固定预览 {OUT}/preview/index.html；执行HEAD {s["identity"]["execution_head"]}。单seed开发探针，无80%或三维/时序有效性声明，完成后停止。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def run():
    start=time.monotonic();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():
        assert (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE','Existing incomplete run preserved; no automatic retry'
        print('EXISTING_COMPLETE_REUSED');return
    protocol=read(SOURCE/'protocol.json');orders=read(SOURCE/'TRAIN_EXPANDED_FULL_orders.json')[:STEPS]
    assert len(orders)==STEPS==5*protocol['Nexp'] and sum(len(r['evaluation_targets']) for r in protocol['dev'])==824
    assert len(protocol['expanded'])==22 and len(protocol['dev'])==8
    assert {r['pair_id'][3:] for r in protocol['expanded']}==set('40y1x9 95qvg9 9l8w54 4qwgw4 4ym5n4 76rjm7 9mygy9 4ymrn4 4wge37 71gz67 4xeqd7'.split())
    assert {r['pair_id'][3:] for r in protocol['dev']}==set('7e2e57 95q8g9 4xd3r9 7ogeg4'.split())
    assert {r['source_id'] for r in protocol['expanded']}.isdisjoint({r['source_id'] for r in protocol['dev']})
    source_id=read(ROOT/'pvst_anakin_ainpaint_spatial_baseline_v1/identity.json');assert digest(WEIGHT)==source_id['generic_weight_sha256']
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_protocol_sha256=digest(SOURCE/'protocol.json'),source_orders_sha256=digest(SOURCE/'TRAIN_EXPANDED_FULL_orders.json'),generic_weight=str(WEIGHT),generic_weight_sha256=source_id['generic_weight_sha256'],official_DINO_revision=source_id['official_DINO_revision'],official_AINPAINT_revision=source_id['official_AINPAINT_revision'])
    for p,k in [(DINO,'official_DINO_revision'),(AINPAINT,'official_AINPAINT_revision')]:assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==identity[k]
    save(OUT/'identity.json',identity);save(OUT/'protocol.json',protocol);save(OUT/'orders.json',orders)
    save(OUT/'config.json',dict(conditions=CONDITIONS,seed=20260909,training_crop=SIZE,tile=SIZE,stride=STRIDE,inference_batch=BATCH,epochs=5,updates=STEPS,sampling='real uniform; positive fake .5 positive-pixel containing/.5 uniform',RGB_degradation='whole frame native uint8 -> old work_hw cv2.INTER_AREA -> native cv2.INTER_LINEAR uint8 -> crop; GT unchanged',padding='black native image padding to288 excluded from loss; normalized zero model pad14 then crop logits',lr=1e-4,weight_decay=.01,clip_grad=1,loss='pixel BCE + soft Dice eps1; valid only',BN_train_only_extra_per_epoch=100,extra_augmentations=[],initialization='fresh generic DINO and same new LoRA/head, never old ANAKIN terminal',evaluation='running BN eval; full native uniform-logit tile fusion, no GT routing'))
    progress('native_prepare');plan=prepare(protocol,orders);identity['crop_plan_sha256']=digest(OUT/'crop_plan.json');identity['tile_plan_sha256']=digest(OUT/'tile_plan.json');save(OUT/'identity.json',identity)
    data=NativeData();byid={r['video_id']:r for r in protocol['expanded']};seed();m=build();initial={k:v.clone() for k,v in adaptation_state(m).items()};initial_hash=state_hash(initial);fixed_hash=frozen_hash(m)
    save(OUT/'initialization.json',dict(adaptation_sha256=initial_hash,frozen_DINO_sha256=fixed_hash,LoRA_parameters=sum(p.numel() for p in m.backbone.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in m.head.parameters())))
    torch.save(initial,OUT/'initial_adaptation.pt');del m
    progress('preflight');checks=preflight(initial,initial_hash,protocol,plan,data)
    results={};history=[];predictions={}
    for name in CONDITIONS:
        seed();m=build().cuda();load_adaptation(m,initial);assert state_hash(adaptation_state(m))==initial_hash and frozen_hash(m)==fixed_hash
        opt=optimizer(m);assert not opt.state;torch.cuda.reset_peak_memory_stats();begin=time.monotonic();training=bnseconds=0.;bncalls=0;epoch=[]
        with (OUT/f'{name}_updates.jsonl').open('x') as log:
            for step,item in enumerate(plan,1):
                r=update(m,opt,data,byid[item['video_id']],item,name);training+=r['seconds'];r.update(updates=step,video_id=item['video_id'],target=item['target'],epoch=item['epoch'],y=item['y'],x=item['x']);log.write(json.dumps(r)+'\n');log.flush();epoch.append(r)
                if step%500==0:progress('train',condition=name,updates=step,U=STEPS,remaining_training_seconds=training/step*(STEPS-step))
                if step%2569==0:
                    ep=step//2569;bnseconds+=BN_statistics(m,data,byid,plan[(ep-1)*2569:(ep-1)*2569+100],name);bncalls+=100
                    history.append(dict(condition=name,epoch=ep,updates=step,**{k:float(np.mean([r[k] for r in epoch])) for k in ['loss','pixel_BCE','soft_Dice']},GT_pixels=sum(r['GT_pixels'] for r in epoch),valid_pixels=sum(r['valid_pixels'] for r in epoch),positive_crops=sum(r['GT_pixels']>0 for r in epoch),seconds=sum(r['seconds'] for r in epoch)));epoch=[];csvsave(OUT/'history.csv',history)
                    progress('epoch_complete',condition=name,epoch=ep,updates=step,remaining_training_seconds=training/step*(STEPS-step))
        assert step==STEPS and bncalls==500 and frozen_hash(m)==fixed_hash
        checkpoint(m,opt,name,identity,initial_hash);progress('final_dev',condition=name,updates=step)
        m.eval();calls=m.calls;wrapper=TiledModel(m,name,BATCH);cache=EvalData(data,wrapper)
        e,pred=evaluate(wrapper,cache,protocol['dev'],name,capture=True,capture_dtype=np.float32);add_area(e)
        reference=read(ROOT/'pvst_anakin_ainpaint_spatial_baseline_v1/final_dev.json')
        assert e['scored_frames']==824 and e['frame_identity_sha256']==reference['frame_identity_sha256']
        tiling=[dict(video_id=f['video_id'],frame_index=f['frame_index'],pts=f['pts'],**r) for f,r in zip(e['frames'],wrapper.records)]
        assert len(tiling)==824 and all(r['coverage_min']>=1 for r in tiling)
        expected=sum(len(tiles(r['native_hw']))*len(r['evaluation_targets']) for r in protocol['dev']);assert m.calls-calls==expected==sum(r['tiles'] for r in tiling)
        save(OUT/f'{name}_dev.json',e);csvsave(OUT/f'{name}_dev_frames.csv',e['frames']);csvsave(OUT/f'{name}_dev_videos.csv',e['videos']);csvsave(OUT/f'{name}_tiling.csv',tiling)
        cost=dict(training_seconds=training,BN_seconds=bnseconds,evaluation_seconds=e['seconds'],wall_seconds=time.monotonic()-begin,optimizer_updates=step,supervised_crops=step,BN_extra_forwards=bncalls,evaluation_tiles=expected,formal_DINO_patch_forwards=m.calls,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),frozen_DINO_unchanged=frozen_hash(m)==fixed_hash,initial_adaptation_sha256=initial_hash)
        assert m.calls==STEPS+500+expected
        results[name]=dict(status='COMPLETE',exit_code=0,cost=cost,evaluation=e);predictions[name]=pred;save(OUT/f'{name}_result.json',results[name])
        # EvalData owns the wrapper/model; release it between conditions too.
        del cache,wrapper,m,opt;torch.cuda.empty_cache()
    assert results[CONDITIONS[0]]['evaluation']['frame_identity_sha256']==results[CONDITIONS[1]]['evaluation']['frame_identity_sha256']
    previews(protocol,data,predictions)
    summary=dict(status='COMPLETE',exit_code=0,identity=identity,protocol=protocol,config=read(OUT/'config.json'),preparation=read(OUT/'native_manifest.json'),preflight=checks,results=results,history=history,wall_seconds=time.monotonic()-start)
    save(OUT/'summary.json',summary);report(summary);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


if __name__=='__main__':
    try:run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error)));raise
