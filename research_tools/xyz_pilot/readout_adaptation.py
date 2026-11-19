"""ADR-0047: matched readout-only and full continuation from original B.

Reuses the original losses/update/evaluation. Arm labels never control history:
all calls explicitly use XYZ_TEMPORAL. Source experiments remain read-only.
"""
import collections
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import traceback
import zipfile
import numpy as np
from PIL import Image
import torch
from .data import REPO, read
from .insource_model import XYZLocalization
from .insource_probe import OUT as SOURCE, PartitionCache, epoch_orders, update, evaluate
from .joint_experiment import save, digest, checkpoint
from .experiment import csvsave
from research_tools.rgbd_slice.experiment import state_hash
from scripts.review_anakin import decode, numeric_members, mask_array

OUT=SOURCE.parent/'pvst_xyz_readout_adaptation_v1'
START=SOURCE/'XYZ_TEMPORAL_epoch10.pt'
HEADS={'pixel_context','local_out','detail','pixel'}
ARMS=['B_READOUT_ONLY','B_FULL_CONTINUE']
MODE='XYZ_TEMPORAL'
METRICS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area']


def tree_hash(value):
    """Numeric optimizer/RNG identity, ignoring tensor device only."""
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous();h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(str(type(v)).encode())
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def freeze(model,readout_only):
    for name,p in model.named_parameters():
        p.requires_grad_(not readout_only or name.split('.')[0] in HEADS);p.grad=None


def frozen_hash(model):
    values={n:p for n,p in model.named_parameters() if not p.requires_grad}
    values.update({'BUFFER:'+n:b for n,b in model.named_buffers()});return state_hash(values)


def frozen_optimizer_hash(model,opt):
    return tree_hash({n:opt.state[p] for n,p in model.named_parameters() if not p.requires_grad})


def restore(c,readout_only):
    # Fresh deserialization prevents mutable optimizer aliases between arms.
    cp=torch.load(START,map_location='cpu',weights_only=False)
    assert cp['condition']==MODE and cp['updates']==10260
    assert cp['identity']==read(SOURCE/'run_identity.json')
    assert len(cp['optimizer']['param_groups'])==1
    assert cp['optimizer']['param_groups'][0]['lr']==c['lr'] and cp['optimizer']['param_groups'][0]['weight_decay']==c['weight_decay']
    model=XYZLocalization(c['h']).cuda();model.load_state_dict(cp['state_dict'],strict=True)
    assert state_hash(model.state_dict())==state_hash(cp['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    opt.load_state_dict(cp['optimizer']);assert tree_hash(opt.state_dict())==tree_hash(cp['optimizer'])
    assert all(float(s['step'])==10260 for s in opt.state.values())
    freeze(model,readout_only)
    rs=cp['random_state'];random.setstate(rs['python']);np.random.set_state(rs['numpy'])
    torch.set_rng_state(rs['torch']);torch.cuda.set_rng_state_all(rs['cuda'])
    actual=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    assert tree_hash(actual)==tree_hash(rs)
    info=dict(model_sha256=state_hash(model.state_dict()),optimizer_sha256=tree_hash(opt.state_dict()),random_state_sha256=tree_hash(actual),
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],total_parameters=sum(p.numel() for p in model.parameters()),start_updates=10260)
    assert info['trainable_parameters']==(31873 if readout_only else 631170)
    return model,opt,info


def add_area(ev):
    # Exact identity from TP/GT and FP/background; no extra model evaluation.
    for f in ev['frames']:
        if f['role']=='real':a=f['real_FPR']
        elif f['GT_pixels']:
            r=f['GT_pixels']/f['native_pixels'];a=f['GT_recall']*r+(f['fake_background_FPR'] or 0)*(1-r)
        else:a=f['fake_empty_FPR']
        f['prediction_area']=a
    for v in ev['videos']:
        v['prediction_area']=float(np.mean([f['prediction_area'] for f in ev['frames'] if f['video_id']==v['video_id']]))
    ev['macro']['prediction_area']=float(np.mean([v['prediction_area'] for v in ev['videos'] if v['role']=='fake']))
    return ev


def persist_eval(name,ev):
    save(OUT/f'{name}.json',ev);csvsave(OUT/f'{name}_frames.csv',ev['frames']);csvsave(OUT/f'{name}_videos.csv',ev['videos'])


def progress(stage,**kw):
    x=dict(status='RUNNING',stage=stage,**kw);save(OUT/'status.json',x);print(json.dumps(x),flush=True)


def preview(rows,predictions):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    html=['<!doctype html><meta charset="utf-8"><title>PVST定位读出适配</title><h1>固定42目标：共同B / READOUT_ONLY / FULL_CONTINUE</h1><p>每行依次RGB、官方GT、起点B概率/二值、读出适配概率/二值、完整继续概率/二值。概率统一灰阶0–1；二值logit≥0；仅预览缩放，不按效果选帧。</p>']
    assert all(len(p)==42 and all(a.dtype==np.float32 for a in p.values()) for p in predictions.values())
    for row in rows:
        ids=row['preview_targets'];_,rgb=decode(Path(row['video_path']),set(ids))
        z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None;members=numeric_members(z.namelist()) if z else None
        for i in ids:
            gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool)
            arrays=[np.asarray(rgb[i]),np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1)]
            for name in ['B_START',*ARMS]:
                a=predictions[name][(row['video_id'],i)];p=1/(1+np.exp(-np.clip(a,-40,40)))
                arrays += [np.repeat((p*255).astype(np.uint8)[...,None],3,-1),np.repeat(((a>=0)*255).astype(np.uint8)[...,None],3,-1)]
            panels=[]
            for a in arrays:
                f=min(1.,320/a.shape[1]);panels.append(np.asarray(Image.fromarray(a).resize((round(a.shape[1]*f),round(a.shape[0]*f)))))
            fn=f'{row["video_id"]}_{i:05d}.png';Image.fromarray(np.concatenate(panels,axis=1)).save(folder/fn)
            html.append(f'<h3>{row["pair_id"]}/{row["role"]} 帧{i} PTS {row["pts"][i]:.6f}s</h3><a href="{fn}"><img width="100%" src="{fn}"></a>')
        if z:z.close()
    (folder/'index.html').write_text('\n'.join(html))


def report(s):
    evs=s['evaluations'];names=['A_REFERENCE','B_START',*ARMS]
    lines=['# PVST定位读出适配（同来源后段）','计算 COMPLETE，退出码0；每组追加3完整轮/3078更新，累计13338。第13轮固定末次，不选择best。',
        '\n|条件|AP|Dice|IoU|召回|real FPR|fake背景FPR|fake前景面积|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in names:lines.append('|'+n+'|'+'|'.join(f'{evs[n]["macro"][k]:.8f}' for k in METRICS)+'|')
    lines+=['\n|来源|起点AP|读出AP|完整AP|读出−起点|完整−起点|读出−完整|','|---|---:|---:|---:|---:|---:|---:|']
    comparisons=[]
    for row in s['protocol']['rows']:
        if row['role']!='fake':continue
        vv={n:next(v for v in evs[n]['videos'] if v['video_id']==row['video_id']) for n in names}
        vs=[vv[n]['pixel_AP'] for n in ['B_START',*ARMS]]
        lines.append('|'+row['pair_id']+'|'+'|'.join(f'{v:.8f}' for v in [*vs,vs[1]-vs[0],vs[2]-vs[0],vs[1]-vs[2]])+'|')
        for metric in METRICS:
            def val(n):
                if metric=='real_FPR':return next(v for v in evs[n]['videos'] if v['pair_id']==row['pair_id'] and v['role']=='real')[metric]
                return vv[n][metric]
            comparisons.append(dict(pair_id=row['pair_id'],metric=metric,B_START=val('B_START'),B_READOUT_ONLY=val(ARMS[0]),B_FULL_CONTINUE=val(ARMS[1]),readout_minus_start=val(ARMS[0])-val('B_START'),full_minus_start=val(ARMS[1])-val('B_START'),readout_minus_full=val(ARMS[0])-val(ARMS[1])))
    csvsave(OUT/'source_differences.csv',comparisons)
    macro=[]
    for k in METRICS:
        v=[evs[n]['macro'][k] for n in ['B_START',*ARMS]]
        macro.append(dict(metric=k,readout_minus_start=v[1]-v[0],full_minus_start=v[2]-v[0],readout_minus_full=v[1]-v[2]))
    csvsave(OUT/'macro_differences.csv',macro)
    lines+=['\n## 恢复、冻结与覆盖',
        '共同B完整模型严格加载；按原全参数顺序创建AdamW并恢复状态后再冻结。各条件独立反序列化并恢复Python/NumPy/CPU/CUDA随机状态，模型/优化器/随机状态hash一致，读出动量、方差和step不重置。所有调用显式使用XYZ_TEMPORAL和temporal=True。',
        'READOUT_ONLY仅pixel_context/local_out/detail/pixel共31873参数，完整组631170。冻结参数grad=None，冻结权重、全部buffer和冻结优化器状态保持；四读出模块均实际更新。副本预检丢弃，两组正式重新恢复。',
        '原70/30分区、scale=12.880994319915771、h=.5、8槽/.1秒、BF16前向/FP32 loss和原生恢复均不变。新MoGe/冻结骨干提取均0。AdamW1e-4/wd.01、clip1，两项空间loss各1，无帧头。',
        '每轮1026目标real515/fake511；第11–13轮seed+epoch无放回同序列，角色次数及hash见coverage.json。40y1x9 fake6–9不监督计分，历史不跨分区。后段444唯一目标222正fake/222 real；AP按正fake帧→视频→7来源宏平均。原A只引用已存指标。面积按同一TP/GT、FP/背景计数恒等式复原。',
        '固定28目标观察不代表整个训练集，逐轮观察和两项在线loss见history.csv；固定42目标概率0–1、二值0.5预览保留。',
        '\n## 回放与成本',json.dumps(s['baseline_replay'],ensure_ascii=False),
        json.dumps({n:{k:v for k,v in x.items() if k!='restoration'} for n,x in s['results'].items()},ensure_ascii=False),
        f'主流程{s["wall_seconds"]:.2f}秒；执行HEAD {s["identity"]["execution_head"]}；预览 {OUT}/preview/index.html。',
        '单seed已知来源后段实验，冻结策略收益不证明实现错误，不定位哪一层导致旧差距；不宣称跨来源、三维独立贡献或时间边界。计算与Git发布分别记录。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for n in ARMS:
        hs=[h for h in s['history'] if h['condition']==n]
        for ax,k in zip(axs,['pixel_BCE','soft_Dice']):ax.plot([h['epoch'] for h in hs],[h[k] for h in hs],label=n);ax.set_title(k);ax.set_xlabel('Original epoch');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=140);plt.close(fig)


def run():
    started=time.monotonic();torch.set_num_threads(4)
    c=read(SOURCE/'config.json');protocol=read(SOURCE/'protocol.json');rows=protocol['rows'];byid={r['video_id']:r for r in rows}
    assert c['scale']==12.880994319915771 and c['h']==.5 and len(rows)==14
    assert sum(len(r['evaluation_targets']) for r in rows)==444
    orders=epoch_orders(rows,13,c['seed']);assert orders[:10]==read(SOURCE/'orders.json');orders=orders[10:]
    assert [x[0]['epoch'] for x in orders]==[11,12,13] and all(len(x)==1026 for x in orders)
    OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),checkpoint=str(START),checkpoint_sha256=digest(START),source_protocol_sha256=digest(SOURCE/'protocol.json'),source_config_sha256=digest(SOURCE/'config.json'),source_orders_sha256=digest(SOURCE/'orders.json'),source_run_identity=read(SOURCE/'run_identity.json'),input_mode=MODE,temporal=True)
    save(OUT/'identity.json',identity);save(OUT/'protocol.json',protocol);save(OUT/'config.json',dict(original=c,append_epochs=[11,12,13],arms=ARMS,readout_modules=sorted(HEADS),updates=3078,final_updates=13338));save(OUT/'orders.json',orders)
    counts=collections.Counter(byid[x['video_id']]['role'] for seq in orders for x in seq)
    save(OUT/'coverage.json',dict(updates_per_arm=3078,per_epoch=1026,unique_targets=1026,roles=dict(counts),orders_sha256=digest(OUT/'orders.json'),evaluation_targets=444,monitor_targets=28,preview_targets=42))
    cache=PartitionCache(c);checks={};r=byid['9mygy9_fake'];target=r['train_targets'][-1]
    for name in ARMS:
        progress('preflight',condition=name);m,opt,info=restore(c,name==ARMS[0])
        before=frozen_hash(m);osig=frozen_optimizer_hash(m,opt)
        weights={h:state_hash({n:p for n,p in m.named_parameters() if n.split('.')[0]==h}) for h in HEADS}
        stat=update(m,opt,cache,r,target,MODE)
        changed={h:weights[h]!=state_hash({n:p for n,p in m.named_parameters() if n.split('.')[0]==h}) for h in HEADS}
        assert all(changed.values()) and before==frozen_hash(m) and osig==frozen_optimizer_hash(m,opt)
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
        grads={h:all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for n,p in m.named_parameters() if n.split('.')[0]==h) for h in HEADS};assert all(grads.values())
        checks[name]=dict(restoration=info,update=stat,readout_changed=changed,finite_gradients=grads,frozen_unchanged=True,discarded=True)
        del m,opt;torch.cuda.empty_cache()
    for k in ['model_sha256','optimizer_sha256','random_state_sha256']:assert len({x['restoration'][k] for x in checks.values()})==1
    save(OUT/'preflight.json',checks);progress('baseline_replay');m,opt,_=restore(c,False)
    ev,preds=evaluate(m,cache,rows,MODE,capture=True,capture_dtype=np.float32);ev=add_area(ev);persist_eval('B_START',ev)
    evaluations={'B_START':ev};predictions={'B_START':preds};del m,opt;torch.cuda.empty_cache()
    original=read(SOURCE/'XYZ_TEMPORAL_epoch10_evaluation.json');assert ev['frame_identity_sha256']==original['frame_identity_sha256']
    for new,old in zip(ev['videos'],original['videos']):
        assert new['video_id']==old['video_id']
        assert all(abs(new[k]-old[k])<1e-3 for k in ['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR'] if new[k] is not None and old[k] is not None),'Per-video baseline mismatch'
    replay={k:ev['macro'][k]-original['macro'][k] for k in original['macro']};save(OUT/'baseline_replay.json',replay)
    assert all(abs(replay[k])<1e-3 for k in ['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR']),'Baseline mismatch'
    evaluations['A_REFERENCE']=add_area(read(SOURCE/'XYZ_CURRENT_epoch10_evaluation.json'));results={};history=[]
    for name in ARMS:
        m,opt,info=restore(c,name==ARMS[0]);begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
        frozen_before=frozen_hash(m);opt_before=frozen_optimizer_hash(m,opt)
        head_before={h:state_hash({n:p for n,p in m.named_parameters() if n.split('.')[0]==h}) for h in HEADS}
        count=0;train_seconds=0.;eval_seconds=0.
        with (OUT/f'{name}_updates.jsonl').open('x') as log:
            for seq in orders:
                epoch=seq[0]['epoch'];progress('train',condition=name,epoch=epoch,additional_updates=count);records=[]
                for item in seq:
                    rec=update(m,opt,cache,byid[item['video_id']],item['target'],MODE);count+=1;train_seconds+=rec['seconds']
                    rec.update(condition=name,epoch=epoch,additional_updates=count,cumulative_updates=10260+count,video_id=item['video_id'],target=item['target'])
                    log.write(json.dumps(rec)+'\n');log.flush();records.append(rec)
                progress('monitor',condition=name,epoch=epoch,additional_updates=count)
                mon,_=evaluate(m,cache,rows,MODE,monitor=True);mon=add_area(mon);assert mon['scored_frames']==28
                persist_eval(f'{name}_epoch{epoch}_monitor',mon);eval_seconds+=mon['seconds']
                h=dict(condition=name,epoch=epoch,additional_updates=count,cumulative_updates=10260+count,**{k:float(np.mean([x[k] for x in records])) for k in ['loss','pixel_BCE','soft_Dice']},training_seconds=sum(x['seconds'] for x in records),monitor_AP=mon['macro']['pixel_AP'],monitor_Dice=mon['macro']['Dice'])
                history.append(h);save(OUT/'history.json',history);csvsave(OUT/'history.csv',history)
                print('EPOCH',json.dumps(h),'remaining_training_estimate_seconds',train_seconds/(epoch-10)*(13-epoch),flush=True)
        assert count==3078 and frozen_before==frozen_hash(m) and opt_before==frozen_optimizer_hash(m,opt)
        changed={h:head_before[h]!=state_hash({n:p for n,p in m.named_parameters() if n.split('.')[0]==h}) for h in HEADS};assert all(changed.values())
        checkpoint(OUT/f'{name}_epoch13.pt',m,opt,13338,name,identity,info['model_sha256']);progress('final_evaluation',condition=name,additional_updates=count)
        ev,preds=evaluate(m,cache,rows,MODE,capture=True,capture_dtype=np.float32);ev=add_area(ev)
        assert ev['scored_frames']==444 and ev['frame_identity_sha256']==original['frame_identity_sha256']
        evaluations[name]=ev;predictions[name]=preds;persist_eval(name,ev);eval_seconds+=ev['seconds']
        results[name]=dict(additional_updates=count,cumulative_updates=13338,restoration=info,readout_changed=changed,frozen_weights_buffers_unchanged=True,frozen_optimizer_unchanged=True,training_seconds=train_seconds,evaluation_seconds=eval_seconds,wall_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        save(OUT/f'{name}_result.json',results[name]);del m,opt;torch.cuda.empty_cache()
    assert digest(START)==identity['checkpoint_sha256'];preview(rows,predictions)
    s=dict(status='COMPLETE',exit_code=0,identity=identity,protocol=protocol,config=c,preflight=checks,baseline_replay=replay,results=results,evaluations=evaluations,history=history,wall_seconds=time.monotonic()-started,new_geometry_frames=0,new_visual_frames=0)
    save(OUT/'summary.json',s);report(s);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


def main():
    if (OUT/'identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':
            assert read(OUT/'identity.json')['checkpoint_sha256']==digest(START);print('EXISTING_COMPLETE_REUSED');return
        raise SystemExit('Existing execution preserved; no automatic retry')
    try:run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error)));raise


if __name__=='__main__':main()
