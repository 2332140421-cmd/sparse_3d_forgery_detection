"""ADR-0045: fixed 70/30 time-partition localization probe, three conditions.

prepare freezes CPU-only protocol/scale; run consumes it and completes ten full
without-replacement epochs. No frontend, old checkpoint initialization, or old
cross-source dev-media read is provided by this entry.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from .data import OUT as SOURCE, REPO, Cache, read
from .model import history_indices
from .insource_model import XYZLocalization, ImageOnly, COMMON, losses
from .experiment import eligible, seed, csvsave
from .joint_experiment import save, digest, checkpoint, preview
from research_tools.rgbd_slice.experiment import state_hash
from research_tools.v8.spatial_localization.metrics import spatial_frame
from scripts.review_anakin import numeric_members, mask_array

OUT=SOURCE.parent/'pvst_xyz_insource_probe_v1'
CONFIG=REPO/'docs/pvst/xyz_insource_config_v1.json'
CONDITIONS=['IMAGE_ONLY','XYZ_CURRENT','XYZ_TEMPORAL']


def progress(stage,**kw):
    state=dict(status='RUNNING',stage=stage,**kw);save(OUT/'status.json',state);print(json.dumps(state),flush=True)


def boundary_rows(rows):
    result=[]
    for pair in dict.fromkeys(r['pair_id'] for r in rows):
        paired=[r for r in rows if r['pair_id']==pair]
        assert len(paired)==2 and {r['role'] for r in paired}=={'real','fake'}
        start=min(r['pts'][0] for r in paired);end=max(r['pts'][-1] for r in paired)
        boundary=start+.7*(end-start)
        for source in paired:
            row=dict(source);cut=next(i for i,p in enumerate(row['pts']) if p>=boundary)
            assert 0<cut<row['actual_frames']
            row.update(original_split=row['split'],boundary_pts=boundary,
                       train_range=[0,cut],evaluation_range=[cut,row['actual_frames']])
            row['train_targets']=[i for i in range(cut) if eligible(row,i)]
            row['evaluation_targets']=[i for i in range(cut,row['actual_frames']) if eligible(row,i)]
            legal=row['train_targets'];row['monitor_targets']=[legal[int(f*(len(legal)-1))] for f in [1/3,2/3]]
            ev=row['evaluation_targets'];row['preview_targets']=[ev[0],ev[len(ev)//2],ev[-1]]
            result.append(row)
    return result


def partition_history(row,target,partition):
    start,stop=row[partition+'_range']
    assert start<=target<stop
    return [i if i is not None and start<=i<stop else None for i in history_indices(row['pts'],target,8,.1)]


def epoch_orders(rows,epochs=10,seed_value=20260909):
    targets=[dict(video_id=r['video_id'],target=i) for r in rows for i in r['train_targets']]
    result=[]
    for epoch in range(1,epochs+1):
        order=targets.copy();random.Random(seed_value+epoch).shuffle(order)
        result.append([dict(x,epoch=epoch) for x in order])
    return result


class PartitionCache:
    def __init__(self,c,cache_root=None):self.c=c;self.xyz=Cache(c,cache_root)
    def get(self,row,i,partition,image_only=False,device='cuda'):
        lo,hi=row[partition+'_range'];assert lo<=i<hi,'WINDOW_CROSSES_PARTITION'
        if not image_only:return self.xyz.get(row,i,device=device)
        # Existing .pt bundles all observations; numeric model input deliberately
        # selects only visual/RGB and supervision. XYZ is not normalized or sent
        # to GPU, and valid is not given to IMAGE_ONLY.
        e=torch.load(self.xyz.root/row['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
        assert e['frame_index']==i and e['pts']==row['pts'][i]
        assert e['native_hw']==row['native_hw'] and e['padded_hw']==row['padded_hw']
        f={k:e[k] for k in ['rgb','gt_work','pts','frame_index','padded_hw','native_hw','native_GT_pixels','supervised']}
        assert tuple(f['rgb'].shape[:2])==tuple(row['work_hw'])
        for k in ['rgb','gt_work']:f[k]=f[k].to(device)
        f['visual']=[v.to(device=device,dtype=torch.float32) for v in e['visual']]
        return f
    def window(self,row,target,partition,condition):
        ids=partition_history(row,target,partition) if condition=='XYZ_TEMPORAL' else [target]
        memo={i:self.get(row,i,partition,condition=='IMAGE_ONLY') for i in set(ids) if i is not None}
        return [memo[i] if i is not None else None for i in ids]


def prepare():
    started=time.monotonic();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Prepared protocol already exists; preserve it')
    c=read(CONFIG);original=read(SOURCE/'manifest.json');rows=boundary_rows([r for r in original['rows'] if r['split']=='train'])
    assert len(rows)==14 and list(dict.fromkeys(r['pair_id'] for r in rows))==c['sources']
    missing=[str(SOURCE/'cache'/r['video_id']/f'{i:05d}.pt') for r in rows for i in range(r['actual_frames']) if not(SOURCE/'cache'/r['video_id']/f'{i:05d}.pt').is_file()]
    if missing:
        save(OUT/'asset_gap.json',dict(status='ASSET_BLOCKED',missing=missing,required_new_frontend_frames=len(missing),cost='Must estimate before any recomputation; no extraction authorized'))
        raise FileNotFoundError('Missing frozen cache: '+str(missing[:5]))
    samples=[];scale_frames=0
    for r in rows:
        for i in range(*r['train_range']):
            e=torch.load(SOURCE/'cache'/r['video_id']/f'{i:05d}.pt',weights_only=True,map_location='cpu')
            assert e['pts']==r['pts'][i] and e['frame_index']==i
            z=e['xyz_provider'][...,2][e['valid']].numpy();z=z[np.isfinite(z)&(z>0)]
            if len(z):samples.extend(z[np.linspace(0,len(z)-1,min(256,len(z)),dtype=int)].tolist())
            scale_frames+=1
    c['scale']=float(np.median(samples));assert c['scale']>0 and c['h']==.5
    orders=epoch_orders(rows,c['epochs'],c['seed']);hist=[]
    for r in rows:
        counts=collections.Counter();unique=collections.Counter();spans=[]
        for i in r['evaluation_targets']:
            ids=[j for j in partition_history(r,i,'evaluation')[:-1] if j is not None]
            counts[len(ids)]+=1;unique[len(set(ids))]+=1
            spans.append(r['pts'][i]-r['pts'][min(ids)] if ids else 0.)
        hist.append(dict(video_id=r['video_id'],pair_id=r['pair_id'],role=r['role'],boundary_pts=r['boundary_pts'],train_range=r['train_range'],evaluation_range=r['evaluation_range'],train_targets=len(r['train_targets']),evaluation_targets=len(r['evaluation_targets']),history_slots=dict(sorted(counts.items())),unique_history_observations=dict(sorted(unique.items())),history_span_seconds_min=min(spans),history_span_seconds_max=max(spans)))
    protocol=dict(rows=rows,boundary_rule='Shared pair boundary=min(first PTS)+0.7*(max(last PTS)-min(first PTS)); train PTS<boundary, eval PTS>=boundary',source_manifest_sha256=digest(SOURCE/'manifest.json'),scale=dict(value=c['scale'],frames=scale_frames,sampled_points=len(samples),rule='Same deterministic <=256 positive valid Z/frame; train partition only, including unlabeled historical observations'),new_frontend_frames=0)
    save(OUT/'protocol.json',protocol);save(OUT/'config.json',c);save(OUT/'orders.json',orders);save(OUT/'history_coverage.json',hist)
    csvsave(OUT/'partitions.csv',[{k:v for k,v in r.items() if k!='unique_history_observations'} for r in hist])
    save(OUT/'preparation.json',dict(status='PREPARED',exit_code=0,seconds=time.monotonic()-started,training_targets=len(orders[0]),total_updates=sum(map(len,orders)),evaluation_targets=sum(len(r['evaluation_targets']) for r in rows),scale=c['scale']))
    save(OUT/'status.json',dict(status='PREPARED',exit_code=0))
    print(json.dumps(read(OUT/'preparation.json')),flush=True)


def update(model,opt,cache,row,target,condition,diagnose=False):
    model.train();opt.zero_grad(set_to_none=True);start=time.monotonic()
    fs=cache.window(row,target,'train',condition);assert fs[-1]['supervised']
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(fs,condition=='XYZ_TEMPORAL')
    parts=losses(out,fs[-1]);assert torch.isfinite(parts).all()
    parts.sum().backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    before=model.pixel[-1].weight.detach().clone() if diagnose else None
    opt.step();torch.cuda.synchronize()
    if diagnose:assert not torch.equal(before,model.pixel[-1].weight) and opt.state
    return dict(loss=float(parts.sum()),pixel_BCE=float(parts[0]),soft_Dice=float(parts[1]),gradient_norm=float(norm),seconds=time.monotonic()-start,blocks=out['blocks'],window_blocks=out['window_blocks'])


@torch.no_grad()
def evaluate(model,cache,rows,condition,monitor=False,capture=False,capture_dtype=np.float16):
    model.eval();started=time.monotonic();records=[];videos=[];pred={}
    partition='train' if monitor else 'evaluation'
    for row in rows:
        wanted=row['monitor_targets'] if monitor else row['evaluation_targets']
        z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None
        members=numeric_members(z.namelist()) if z else None;per=[]
        for i in wanted:
            fs=cache.window(row,i,partition,condition)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(fs,condition=='XYZ_TEMPORAL')
            gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool)
            assert int(gt.sum())==fs[-1]['native_GT_pixels'] and list(gt.shape)==row['native_hw']
            native=F.interpolate(out['pixel'].float()[None,None],size=row['native_hw'],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            assert np.isfinite(native).all()
            overlap=spatial_frame(native>=0,gt);neg=gt.size-int(gt.sum());fp=int(((native>=0)&~gt).sum())
            item=dict(video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],frame_index=i,pts=row['pts'][i],native_pixels=gt.size,GT_pixels=int(gt.sum()),
                pixel_AP=float(average_precision_score(gt.ravel(),native.ravel())) if gt.any() else None,
                constant_AP=float(gt.mean()) if gt.any() else None,Dice=overlap['dice'],IoU=overlap['iou'],
                GT_recall=float(((native>=0)&gt).sum()/gt.sum()) if gt.any() else None,
                real_FPR=overlap['empty_fpr'] if row['role']=='real' else None,
                fake_background_FPR=fp/neg if row['role']=='fake' and gt.any() and neg else None,
                fake_empty_FPR=overlap['empty_fpr'] if row['role']=='fake' and not gt.any() else None)
            per.append(item);records.append(item)
            if capture and i in row['preview_targets']:pred[(row['video_id'],i)]=native.astype(capture_dtype)
        if z:z.close()
        avg=lambda k:float(np.mean([x[k] for x in per if x[k] is not None])) if any(x[k] is not None for x in per) else None
        videos.append(dict(video_id=row['video_id'],pair_id=row['pair_id'],role=row['role'],frames=len(per),positive_GT_frames=sum(x['GT_pixels']>0 for x in per),empty_GT_frames=sum(x['GT_pixels']==0 for x in per),**{k:avg(k) for k in ['pixel_AP','constant_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','fake_empty_FPR']}))
    keys=[(x['video_id'],x['frame_index'],x['pts'],x['native_pixels'],x['GT_pixels']) for x in records]
    assert len(keys)==len(set((x[0],x[1]) for x in keys))
    macro={k:float(np.mean([v[k] for v in videos if v['role']=='fake' and v[k] is not None])) for k in ['pixel_AP','constant_AP','Dice','IoU','GT_recall','fake_background_FPR']}
    macro['real_FPR']=float(np.mean([v['real_FPR'] for v in videos if v['role']=='real']))
    return dict(macro=macro,videos=videos,frames=records,scored_frames=len(records),frame_identity_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),seconds=time.monotonic()-started),pred


def report(summary):
    results=summary['results'];metrics=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR']
    lines=['# PVST同源片段定位探针','计算COMPLETE，退出码0；第10轮固定主结果，无best选择。',
        '|评价段指标|I图像|A同帧XYZ|B时空XYZ|A−I|B−A|','|---|---:|---:|---:|---:|---:|']
    for k in metrics:
        vs=[results[n]['evaluations']['10']['evaluation']['macro'][k] for n in CONDITIONS]
        lines.append('|'+k+'|'+'|'.join(f'{v:.8f}' for v in [*vs,vs[1]-vs[0],vs[2]-vs[1]])+'|')
    lines+=['\n|来源|I AP|A AP|B AP|A−I|B−A|','|---|---:|---:|---:|---:|---:|']
    comparison=[]
    for row in summary['protocol']['rows']:
        if row['role']!='fake':continue
        values=[next(v for v in results[n]['evaluations']['10']['evaluation']['videos'] if v['video_id']==row['video_id']) for n in CONDITIONS]
        vs=[v['pixel_AP'] for v in values]
        lines.append('|'+row['pair_id']+'|'+'|'.join(f'{v:.8f}' for v in [*vs,vs[1]-vs[0],vs[2]-vs[1]])+'|')
        comparison.append(dict(pair_id=row['pair_id'],I_AP=vs[0],A_AP=vs[1],B_AP=vs[2],A_minus_I=vs[1]-vs[0],B_minus_A=vs[2]-vs[1]))
    csvsave(OUT/'comparison.csv',comparison)
    lines+=['\n## 训练观察与固定评价','|条件|轮次|观察AP|观察Dice|评价AP|评价Dice|','|---|---:|---:|---:|---:|---:|']
    for n in CONDITIONS:
        for epoch,ev in results[n]['evaluations'].items():
            tr,te=ev['monitor']['macro'],ev['evaluation']['macro'];lines.append(f"|{n}|{epoch}|{tr['pixel_AP']:.8f}|{tr['Dice']:.8f}|{te['pixel_AP']:.8f}|{te['Dice']:.8f}|")
    lines+=['\n## 协议和实现',
        '仅原train7来源按共同实际PTS跨度70%切分，同源后30%全部合法目标评价。原dev4不训练、不评价。边界定义及逐帧身份见protocol.json；前后窗口严格分区，评价起始历史缺失不借用训练帧或复制。40y1x9_fake 6–9不监督计分，可以作为训练区内无标签历史。',
        '训练段valid正Z按每帧最多256点确定单一尺度'+str(summary['config']['scale'])+'，h=.5固定不搜索。工作长边384及原生恢复与之前一致，几何无效像素保留视觉路径；不称世界配准/米制重建。新MoGe/冻结视觉提取均为0。',
        'I只保留共同多尺度融合、local32、RGB16及同一80通道像素解码器，context32恒零，无XYZ/valid模型输入，无块/关系/分类模块；既有打包缓存反序列化后仅选择视觉/RGB/监督字段。A只当前，B分区内8槽/.1秒窗口联合注意力，D128/4头/2层、位置与完整梯度路径不变。没有帧BCE或帧检测成绩。',
        '三组仅pixel BCE+soft Dice各1；seed20260909、AdamW1e-4/wd.01、clip1、BF16/FP32 loss不变。每轮所有合法训练目标无放回各一次，三组完整相同顺序，共10轮；第3、10轮保存权重和评价，不选best。',
        'AP仅fake正GT帧全原生画面→视频均值→来源宏平均；空GT不补AP，real FPR单列；fake背景FPR在正GT帧的GT之外区域计算。固定logit≥0，AP@IoU未定义，不声称时间起止定位。',
        '共同初始化与参数量：'+json.dumps(summary['initialization'],ensure_ascii=False),
        '\n## 成本和终态',json.dumps({k:{x:v[x] for x in ['updates','training_seconds','evaluation_seconds','wall_seconds','peak_allocated','peak_reserved']} for k,v in results.items()},ensure_ascii=False),
        'CPU准备：'+json.dumps(summary['preparation'],ensure_ascii=False),
        '总主流程秒：'+str(summary['wall_seconds'])+'；退出码0。执行身份：'+json.dumps(summary['identity'],ensure_ascii=False),
        '完整产物：'+str(OUT)+'；固定评价首/中/末预览：preview/index.html。逐来源全部空间指标与FPR见epoch10_evaluation_videos.csv，历史长度见history_coverage.json。',
        '本对照只说明同来源后段的表现，不衡量跨来源泛化。I→A是整个XYZ块/空间支路增量，不能证明几何独立贡献；A→B是本结构历史连接增量。模块原因仍属假设，不以一次结果认定实现必错。完成后停止。']
    text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    history=read(OUT/'history.json');fig,ax=plt.subplots(figsize=(7,4))
    for n in CONDITIONS:
        hs=[x for x in history if x['condition']==n];ax.plot([x['epoch'] for x in hs],[x['loss'] for x in hs],label=n)
    ax.set_xlabel('Full epochs');ax.set_ylabel('Mean online pixel BCE + Dice');ax.legend();fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)
    html=['<!doctype html><meta charset="utf-8"><h1>同源评价段固定首/中/末：I / A / B</h1><p>各面板RGB / GT / 预测概率；固定0–1色阶，仅预览缩放，real为合法零GT。按pair ID，不按成绩排序。</p>']
    for row in sorted(summary['protocol']['rows'],key=lambda r:r['video_id']):
        for i in row['preview_targets']:
            html.append(f'<h3>{row["pair_id"]}/{row["role"]} 帧{i}, PTS {row["pts"][i]:.6f}</h3>')
            for n in CONDITIONS:html.append(f'<img width="32%" src="evaluation/{n}_{row["video_id"]}_{i:05d}.png">')
    (OUT/'preview'/'index.html').write_text('\n'.join(html))


def run():
    started=time.monotonic();torch.set_num_threads(4);c=read(OUT/'config.json');protocol=read(OUT/'protocol.json');rows=protocol['rows'];orders=read(OUT/'orders.json')
    assert c==read(CONFIG) and len(orders)==10
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config_sha256=digest(CONFIG),protocol_sha256=digest(OUT/'protocol.json'),orders_sha256=digest(OUT/'orders.json'),source_manifest_sha256=digest(SOURCE/'manifest.json'),source_sha256={f:digest(REPO/'research_tools/xyz_pilot'/f) for f in ['insource_probe.py','insource_model.py','model.py','joint_model.py']})
    assert identity['source_manifest_sha256']==protocol['source_manifest_sha256'];save(OUT/'run_identity.json',identity)
    seed();reference=XYZLocalization(c['h']);initial={k:v.clone() for k,v in reference.state_dict().items()};image=ImageOnly(reference)
    common=lambda model:{k:v for k,v in model.state_dict().items() if k.split('.')[0] in COMMON}
    common_hash=state_hash(common(reference));assert state_hash(common(image))==common_hash
    initial_image={k:v.clone() for k,v in image.state_dict().items()};torch.save(dict(XYZ=initial,IMAGE_ONLY=initial_image),OUT/'initial.pt')
    initial_info=dict(common_sha256=common_hash,XYZ_sha256=state_hash(initial),IMAGE_ONLY_sha256=state_hash(initial_image),parameters=dict(IMAGE_ONLY=sum(p.numel() for p in image.parameters()),XYZ_CURRENT=sum(p.numel() for p in reference.parameters()),XYZ_TEMPORAL=sum(p.numel() for p in reference.parameters())),forensic_initialization=False)
    initial_info['IMAGE_ONLY_zero_context_coefficients']=image.pixel[0].weight[:,48:].numel()
    initial_info['IMAGE_ONLY_output_effective_parameters']=initial_info['parameters']['IMAGE_ONLY']-initial_info['IMAGE_ONLY_zero_context_coefficients']
    save(OUT/'initialization.json',initial_info);del reference,image
    cache=PartitionCache(c);byid={r['video_id']:r for r in rows};checks={};row=byid['9mygy9_fake'];target=row['train_targets'][-1]
    for name in CONDITIONS:
        seed();reference=XYZLocalization(c['h']);reference.load_state_dict(initial,strict=True)
        model=ImageOnly(reference) if name=='IMAGE_ONLY' else reference;model=model.cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);torch.cuda.reset_peak_memory_stats()
        progress('preflight',condition=name,video=row['video_id'],target=target)
        checks[name]=update(model,opt,cache,row,target,name,True)
        checks[name].update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        del model,opt,reference;torch.cuda.empty_cache()
    save(OUT/'preflight.json',checks);results={};history=[]
    for name in CONDITIONS:
        seed();reference=XYZLocalization(c['h']);reference.load_state_dict(initial,strict=True)
        model=ImageOnly(reference) if name=='IMAGE_ONLY' else reference
        init_hash=state_hash(model.state_dict());assert init_hash==(initial_info['IMAGE_ONLY_sha256'] if name=='IMAGE_ONLY' else initial_info['XYZ_sha256'])
        assert state_hash(common(model))==common_hash
        model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        torch.cuda.reset_peak_memory_stats();begin=time.monotonic();step=0;training_seconds=0.;evaluation_seconds=0.;evs={}
        with (OUT/f'{name}_updates.jsonl').open('w') as log:
            for epoch,sequence in enumerate(orders,1):
                progress('train',condition=name,epoch=epoch,updates=step);epoch_records=[]
                for item in sequence:
                    record=update(model,opt,cache,byid[item['video_id']],item['target'],name);step+=1;training_seconds+=record['seconds']
                    record.update(epoch=epoch,step=step,condition=name,**{k:item[k] for k in ['video_id','target']});log.write(json.dumps(record)+'\n');log.flush();epoch_records.append(record)
                history.append(dict(condition=name,epoch=epoch,updates=step,**{k:float(np.mean([x[k] for x in epoch_records])) for k in ['loss','pixel_BCE','soft_Dice']},seconds=sum(x['seconds'] for x in epoch_records)))
                save(OUT/'history.json',history);csvsave(OUT/'history.csv',history)
                print('EPOCH',name,epoch,'seconds',history[-1]['seconds'],'remaining_train_estimate',training_seconds/epoch*(10-epoch),flush=True)
                if epoch in [3,10]:
                    checkpoint(OUT/f'{name}_epoch{epoch:02d}.pt',model,opt,step,name,identity,init_hash)
                    progress('evaluate',condition=name,epoch=epoch,updates=step)
                    mon,_=evaluate(model,cache,rows,name,True);ev,preds=evaluate(model,cache,rows,name,capture=epoch==10)
                    evs[str(epoch)]=dict(monitor=mon,evaluation=ev);evaluation_seconds+=mon['seconds']+ev['seconds']
                    for label,value in [('monitor',mon),('evaluation',ev)]:
                        save(OUT/f'{name}_epoch{epoch:02d}_{label}.json',value);csvsave(OUT/f'{name}_epoch{epoch:02d}_{label}_videos.csv',value['videos']);csvsave(OUT/f'{name}_epoch{epoch:02d}_{label}_frames.csv',value['frames'])
                    if epoch==10:preview(rows,preds,name,'evaluation',output_root=OUT)
        results[name]=dict(updates=step,initial_sha256=init_hash,training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,wall_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),evaluations=evs)
        save(OUT/f'{name}_result.json',results[name]);del model,opt,reference;torch.cuda.empty_cache()
    for epoch in ['3','10']:
        for label in ['monitor','evaluation']:assert len({results[n]['evaluations'][epoch][label]['frame_identity_sha256'] for n in CONDITIONS})==1
    summary=dict(status='COMPLETE',exit_code=0,config=c,protocol=protocol,preparation=read(OUT/'preparation.json'),identity=identity,initialization=initial_info,preflight=checks,results=results,wall_seconds=time.monotonic()-started,peak_host_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_geometry_frames=0,new_visual_frames=0)
    save(OUT/'summary.json',summary);report(summary);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    if args.stage=='run' and (OUT/'run_identity.json').exists():
        if (OUT/'summary.json').exists() and read(OUT/'summary.json')['status']=='COMPLETE':
            assert read(OUT/'config.json')==read(CONFIG);print('EXISTING_COMPLETE_REUSED');return
        raise SystemExit('Existing run preserved; no duplicate/retry')
    try:
        prepare() if args.stage=='prepare' else run()
    except Exception as error:
        OUT.mkdir(parents=True,exist_ok=True);previous=read(OUT/'status.json') if (OUT/'status.json').exists() else None
        (OUT/f'{args.stage}_error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(error),previous=previous));raise


if __name__=='__main__':main()
