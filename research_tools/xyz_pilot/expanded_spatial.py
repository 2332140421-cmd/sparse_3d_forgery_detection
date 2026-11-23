"""ADR-0054: one expanded, source-isolated full-frame spatial experiment.

CLI -> frozen source pool plus RGB/GT-only work caches -> online existing
SpatialBaseline training/evaluation. Reuses the model, optimizer update and
native metrics; the old fixed-11-source entry cannot express this frozen pool.
"""
import argparse, collections, copy, html, json, math, random, subprocess, time, traceback, zipfile
from pathlib import Path
import av, cv2, numpy as np, torch
from PIL import Image
from . import ainpaint_spatial_experiment as base
from .ainpaint_spatial import build, adaptation_state, load_adaptation, WEIGHT, DINO, AINPAINT
from .data import ROOT, DATA, REPO, read, work_shape
from .experiment import seed, csvsave
from .insource_probe import evaluate, epoch_orders
from .joint_experiment import save, digest
from .readout_adaptation import add_area, METRICS
from research_tools.rgbd_slice.experiment import state_hash
from scripts.review_anakin import numeric_members, mask_array, decode
from scripts.anakin_source_scaleup import source, tier, content_priority

OUT=ROOT/'pvst_anakin_expanded_spatial_training_v1'
POOL=DATA/'secondary_datasets/pvst_geometry_screen_v1/anakin_pvst_source_pool_v1'
OLD=ROOT/'pvst_anakin_ainpaint_spatial_baseline_v1'
HIST={'7e2e57','95q8g9','4xd3r9','7ogeg4'}
EVAL_EPOCHS=[1,3,5,10]
NAME='EXPANDED_SPATIAL'

def progress(stage,**kw):
    value=dict(status='RUNNING',stage=stage,**kw);save(OUT/'status.json',value);print(json.dumps(value),flush=True)

def split_for(r,proposal):
    assert r['source_group_id'] and not r['role_conflict']
    assert r['role']!='test_reserved' and r['human_mask_status']!='not_corresponding'
    assert r['technical_status']=='PASS'
    role={'train_draft':'train','dev_draft':'dev'}.get(proposal,proposal)
    assert role in ('train','dev')
    if role=='train':assert not any(x in ('dev','validation','val') or 'test' in x for x in r['historical_roles'])
    return role

def freeze():
    inventory=read(POOL/'metadata_inventory.json')['rows'];meta={r['id']:r for r in inventory}
    draft={r['id']:r['proposal'] for r in read(POOL/'partition_draft.json')}
    current=read(base.SOURCE/'protocol.json');oldrows=current['expanded']+current['dev']
    rows={};excluded=[]
    for old in oldrows:
        r=copy.deepcopy(old);id=r['pair_id'][3:];m=meta[id]
        assert m['role']==r['split'] and not m['role_conflict'] and m['human_mask_status']!='not_corresponding'
        r.update(source_group_id=m['source_group_id'],cache_kind='existing',cache_root=str(base.SOURCE/'cache'),official_task=m['task'],edit_tier=tier(m['task']),content_priority=content_priority(m['task']))
        rows[(id,r['role'])]=r
    for record in read(POOL/'inventory.json'):
        id=record['pair_id'][3:];m=meta[id]
        if record['mask']['status']!='PASS' or m['human_mask_status']=='not_corresponding' or m['role_conflict'] or m['role']=='test_reserved':
            excluded.append(dict(id=id,technical=record['mask']['status'],human=m['human_mask_status'],role=m['role']));continue
        role=split_for(m,draft[id])
        for video_role in ['real','fake']:
            if (id,video_role) in rows:
                assert rows[(id,video_role)]['split']==role;continue
            media=record['media'][video_role];hw=list(media['sizes_hw'][0]);n=media['frame_count'];wh=list(work_shape(hw,384))
            rows[(id,video_role)]=dict(pair_id='an:'+id,source_id=record['source_id'],source_group_id=m['source_group_id'],split=role,role=video_role,video_id=id+'_'+video_role,video_path=media['path'],video_sha256=media['sha256'],mask_zip=record['mask']['path'],mask_sha256=record['mask']['mask_sha256'],index_base=record['mask']['index_base'],actual_frames=n,pts=[x['seconds'] for x in media['pts']],native_hw=hw,work_hw=wh,cache_kind='work_rgb',cache_root=str(OUT/'cache'),official_task=m['task'],edit_tier=record['edit_tier'],content_priority=record['content_priority'])
    rows=sorted(rows.values(),key=lambda r:r['video_id']);groups=collections.defaultdict(set)
    protected={r['source_group_id'] for r in inventory if r['role']=='test_reserved'}
    for r in rows:
        id=r['pair_id'][3:];assert r['source_group_id'] not in protected
        groups[r['source_group_id']].add(r['split']);assert len(groups[r['source_group_id']])==1
        n=r['actual_frames'];valid=[i for i in range(n) if not(id=='40y1x9' and r['role']=='fake' and 6<=i<10)]
        r.update(train_range=[0,n],evaluation_range=[0,n],train_targets=valid if r['split']=='train' else [],evaluation_targets=valid if r['split']=='dev' else [])
        r['monitor_targets']=[valid[int(f*(len(valid)-1))] for f in [1/3,2/3]] if r['split']=='train' else []
        r['preview_targets']=([valid[0],valid[len(valid)//2],valid[-1]] if id in HIST else [valid[len(valid)//2]]) if r['split']=='dev' else []
        old_camera=meta[id].get('camera_evidence','')
        if old_camera and '（自动' not in old_camera and '明确人工' in old_camera:r['camera_group']='human_record_not_automatic'
        elif '较稳定' in old_camera:r['camera_group']='automatic_stable'
        elif '可见' in old_camera:r['camera_group']='automatic_motion'
        else:r['camera_group']='automatic_uncertain'
        r['human_status']=meta[id]['human_mask_status'];r['human_note']='official labels + technical checks; individual semantic review not required/claimed'
    protocol=dict(rows=rows,excluded=excluded,protected_test_sources=len(protected),seed=20260909,epochs=10,supervision_exclusion={'40y1x9_fake':[6,10]},user_authorization='unconfirmed human semantics/camera do not block; explicit negatives and technical/identity/test conflicts remain excluded')
    protocol['train_sources']=len({r['source_group_id'] for r in rows if r['split']=='train'});protocol['dev_sources']=len({r['source_group_id'] for r in rows if r['split']=='dev'})
    protocol['training_frames']=sum(len(r['train_targets']) for r in rows);protocol['dev_frames']=sum(len(r['evaluation_targets']) for r in rows);protocol['updates']=10*protocol['training_frames']
    return protocol

def prepare():
    start=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Prepared protocol preserved; do not repeat preparation'
    p=freeze();save(OUT/'protocol.json',p)
    train=[r for r in p['rows'] if r['split']=='train'];orders=epoch_orders(train,10);save(OUT/'orders.json',orders);save(OUT/'BN_orders.json',[x[:100] for x in orders])
    save(OUT/'config.json',dict(condition=NAME,seed=20260909,epochs=10,updates=p['updates'],batch=1,lr=1e-4,weight_decay=.01,clip_grad=1,loss='pixel BCE + soft Dice eps1, coefficients1/1',BF16=True,work_long_edge=384,pad_multiple=14,evaluate_epochs=EVAL_EPOCHS,best_selection='ADDITIONAL_DEV macro AP; earliest evaluated epoch on tie',BN_training_forwards_per_epoch=100,BN_rule='frozen first100 targets of each epoch order',human_review_required=False,new_MoGe=0,new_DFormer=0,new_neural_precompute=0))
    save(OUT/'data_identity.json',{name:digest(POOL/name) for name in ['metadata_inventory.json','partition_draft.json','inventory.json','human_evidence.json']})
    new_frames=reused=0;cost=[]
    for r in p['rows']:
        if r['cache_kind']=='existing':
            assert all((Path(r['cache_root'])/r['video_id']/f'{i:05d}.pt').is_file() for i in range(r['actual_frames']))
            reused+=r['actual_frames'];continue
        progress('prepare_work_RGB_GT',video=r['video_id'],new_frames=new_frames)
        folder=OUT/'cache'/r['video_id'];folder.mkdir(parents=True)
        n=r['actual_frames'];h,w=r['work_hw'];rgb=np.lib.format.open_memmap(folder/'rgb.npy',mode='w+',dtype=np.uint8,shape=(n,h,w,3));gt_work=np.lib.format.open_memmap(folder/'gt.npy',mode='w+',dtype=np.bool_,shape=(n,h,w));counts=[]
        z=zipfile.ZipFile(r['mask_zip']) if r['role']=='fake' else None;members=numeric_members(z.namelist()) if z else None
        if z:assert len(members)==n and [x[0] for x in members]==list(range(r['index_base'],r['index_base']+n))
        with av.open(r['video_path']) as c:
            count=0
            for i,f in enumerate(c.decode(video=0)):
                assert i<n and abs(float(f.pts*f.time_base)-r['pts'][i])<1e-8 and [f.height,f.width]==r['native_hw']
                image=f.to_ndarray(format='rgb24');rgb[i]=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA) if image.shape[:2]!=(h,w) else image
                gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(r['native_hw'],bool);assert list(gt.shape)==r['native_hw']
                gt_work[i]=cv2.resize(gt.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool);counts.append(int(gt.sum()));count+=1
        if z:z.close()
        assert count==n;rgb.flush();gt_work.flush();del rgb,gt_work
        save(folder/'identity.json',dict(video_sha256=r['video_sha256'],mask_sha256=r['mask_sha256'],pts=r['pts'],native_hw=r['native_hw'],work_hw=r['work_hw'],native_GT_pixels=counts))
        new_frames+=n;cost.append(dict(video_id=r['video_id'],frames=n))
    save(OUT/'preparation.json',dict(status='COMPLETE',new_work_RGB_GT_frames=new_frames,reused_existing_frames=reused,cache_bytes=sum(f.stat().st_size for f in (OUT/'cache').rglob('*') if f.is_file()),seconds=time.monotonic()-start,new_MoGe=0,new_DFormer=0,new_neural_precompute=0))
    csvsave(OUT/'sources.csv',[{k:r[k] for k in ['pair_id','source_group_id','split','role','video_id','actual_frames','edit_tier','content_priority','camera_group','human_status']} for r in p['rows']])
    progress('PREPARED',train_sources=p['train_sources'],dev_sources=p['dev_sources'],training_frames=p['training_frames'],dev_frames=p['dev_frames'],updates=p['updates'])

class Cache:
    def __init__(self):self.arrays={}
    def get(self,row,i,partition):
        assert i in row['train_targets'] if partition=='train' else i in row['evaluation_targets']
        if row['cache_kind']=='existing':
            e=torch.load(Path(row['cache_root'])/row['video_id']/f'{i:05d}.pt',map_location='cpu',weights_only=True)
            assert e['frame_index']==i and e['pts']==row['pts'][i] and e['native_hw']==row['native_hw']
            f={k:e[k] for k in ['rgb','gt_work','native_GT_pixels','supervised','frame_index','pts','native_hw']}
        else:
            vid=row['video_id']
            if vid not in self.arrays:
                root=Path(row['cache_root'])/vid;m=read(root/'identity.json');assert m['video_sha256']==row['video_sha256'] and m['pts']==row['pts']
                self.arrays[vid]=(np.load(root/'rgb.npy',mmap_mode='r'),np.load(root/'gt.npy',mmap_mode='r'),m['native_GT_pixels'])
            rgb,gt,counts=self.arrays[vid];f=dict(rgb=torch.from_numpy(rgb[i].copy()),gt_work=torch.from_numpy(gt[i].copy()),native_GT_pixels=counts[i],supervised=True,frame_index=i,pts=row['pts'][i],native_hw=row['native_hw'])
        assert list(f['rgb'].shape[:2])==row['work_hw'] and f['gt_work'].shape==f['rgb'].shape[:2]
        for key in ['rgb','gt_work']:f[key]=f[key].cuda()
        return f
    def window(self,row,i,partition,condition):assert condition==NAME;return [self.get(row,i,partition)]

def persist(stem,ev):
    add_area(ev);save(OUT/f'{stem}.json',ev);csvsave(OUT/f'{stem}_frames.csv',ev['frames']);csvsave(OUT/f'{stem}_videos.csv',ev['videos']);return ev

def aggregate(ev,ids):
    videos=[v for v in ev['videos'] if v['pair_id'][3:] in ids];result={}
    for key in METRICS+['constant_AP']:
        vs=[v[key] for v in videos if v['role']==('real' if key=='real_FPR' else 'fake') and v.get(key) is not None]
        result[key]=float(np.mean(vs)) if vs else None
    result['sources']=len({v['pair_id'] for v in videos});result['frames']=sum(v['frames'] for v in videos)
    return result

def grouped(ev,rows):
    pairs={r['pair_id'][3:]:r for r in rows if r['role']=='fake'}
    groups={'HISTORICAL_DEV4':HIST,'ADDITIONAL_DEV':set(pairs)-HIST,'EXPANDED_DEV':set(pairs)}
    for field in ['edit_tier','content_priority','camera_group']:
        for value in sorted({r[field] for r in pairs.values()}):groups[field+':'+value]={id for id,r in pairs.items() if r[field]==value}
    return {name:aggregate(ev,ids) for name,ids in groups.items()}

def checkpoint(m,opt,epoch,step,identity,initial):
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    value=dict(condition=NAME,epoch=epoch,updates=step,adaptation_state={k:v.detach().cpu() for k,v in adaptation_state(m).items()},optimizer=opt.state_dict(),random_state=rng,identity=identity,initial_adaptation_sha256=initial)
    path=OUT/f'epoch{epoch:02d}.pt';torch.save(value,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
    return path

def fixed_images(rows,pred,name):
    folder=OUT/'preview';folder.mkdir(exist_ok=True)
    for row in rows:
        ids=row['preview_targets'];_,images=decode(Path(row['video_path']),set(ids));z=zipfile.ZipFile(row['mask_zip']) if row['role']=='fake' else None;members=numeric_members(z.namelist()) if z else None
        for i in ids:
            gt=mask_array(z.read(members[i][1]))[1] if z else np.zeros(row['native_hw'],bool);a=pred[(row['video_id'],i)];h,w=gt.shape;shape=(min(480,w),max(1,round(h*min(1,480/w))))
            stem=row['video_id']+f'_{i:05d}'
            images[i].resize(shape).save(folder/(stem+'_rgb.png'));Image.fromarray((gt*255).astype(np.uint8)).resize(shape,Image.Resampling.NEAREST).save(folder/(stem+'_gt.png'))
            p=1/(1+np.exp(-np.clip(a,-80,80)));Image.fromarray(np.round(p*255).astype(np.uint8)).resize(shape).save(folder/(stem+'_'+name+'_prob.png'));Image.fromarray(((a>=0)*255).astype(np.uint8)).resize(shape,Image.Resampling.NEAREST).save(folder/(stem+'_'+name+'_binary.png'))
        if z:z.close()

def preview(rows):
    text=['<!doctype html><meta charset="utf-8"><h1>固定目标：旧11来源空间基线／扩充末次</h1><p>RGB、GT、旧/新概率与0.5二值；概率统一0–1，仅展示缩放。历史4各首中末，新增每视频中帧，不按效果选图。</p><select id="filter"><option value="">全部官方描述类别</option>']
    text += ['<option>'+html.escape(x)+'</option>' for x in sorted({r['edit_tier'] for r in rows})];text+=['</select>']
    for r in rows:
        for i in r['preview_targets']:
            stem=r['video_id']+f'_{i:05d}';text.append(f'<section data-type="{r["edit_tier"]}"><h3>{r["pair_id"]}/{r["role"]} 帧{i} PTS {r["pts"][i]:.6f}</h3><p>{html.escape(r["official_task"])}</p>')
            for suffix in ['rgb','gt','old_prob','old_binary','new_prob','new_binary']:text.append(f'<label>{suffix}<img width="16%" src="{stem}_{suffix}.png"></label>')
            text.append('</section>')
    text.append('<script>document.getElementById("filter").onchange=e=>{for(const s of document.querySelectorAll("section"))s.hidden=!!e.target.value&&s.dataset.type!==e.target.value}</script>');(OUT/'preview/index.html').write_text('\n'.join(text))



def completed_checks(s):
    expected={(r['video_id'],i) for r in s['protocol']['rows'] if r['split']=='train' for i in r['train_targets']};observed=collections.Counter()
    with (OUT/'updates.jsonl').open() as handle:
        for line in handle:
            record=json.loads(line);observed[(record['video_id'],record['target'])]+=1
    assert set(observed)==expected and set(observed.values())=={10} and sum(observed.values())==s['updates']
    initial=torch.load(OUT/'initial_adaptation.pt',map_location='cpu',weights_only=True);checks={}
    for epoch in EVAL_EPOCHS:
        cp=torch.load(OUT/f'epoch{epoch:02d}.pt',map_location='cpu',weights_only=False);assert cp['condition']==NAME and cp['epoch']==epoch and cp['updates']==epoch*len(expected) and cp['identity']==s['identity']
        assert set(cp['adaptation_state'])==set(initial) and all(v.shape==initial[k].shape and torch.isfinite(v).all() for k,v in cp['adaptation_state'].items())
        steps={float(v['step']) for v in cp['optimizer']['state'].values()};assert steps=={cp['updates']}
        counts={k:int(v) for k,v in cp['adaptation_state'].items() if k.endswith('num_batches_tracked')};assert len(counts)==5 and set(counts.values())=={cp['updates']+100*epoch}
        assert set(cp['random_state'])=={'python','numpy','torch','cuda'}
        checks[str(epoch)]=dict(updates=cp['updates'],optimizer_state_count=len(cp['optimizer']['state']),BN_counts=counts,changed_adaptation_tensors=sum(not torch.equal(initial[k],v) for k,v in cp['adaptation_state'].items()))
    evaluations=read(OUT/'evaluations.json');identities={s['reference']['frame_identity_sha256']}|{v['evaluation']['frame_identity_sha256'] for v in evaluations.values()};assert len(identities)==1
    expected_dev=s['protocol']['dev_frames'];assert all(v['evaluation']['scored_frames']==expected_dev for v in evaluations.values())
    coverage=[]
    for r in s['protocol']['rows']:
        if r['split']=='train':coverage.append(dict(video_id=r['video_id'],pair_id=r['pair_id'],role=r['role'],unique_targets=len(r['train_targets']),presentations=sum(observed[(r['video_id'],i)] for i in r['train_targets']),per_target_presentations=10))
    csvsave(OUT/'actual_exposure.csv',coverage)
    save(OUT/'verification.json',dict(status='PASS',checkpoints=checks,actual_updates=sum(observed.values()),each_legal_training_target_exposure=10,unique_training_targets=len(observed),development_frame_identity_consistent=True,development_frames=expected_dev,frozen_DINO_unchanged='asserted at preflight and end of original runner',BN_statistics_forwards=1000,retained_test_media_access=False,new_MoGe=0,new_DFormer=0))

def official_action(task):
    """Action first: white shirt/hat describe the removed object, not edit type."""
    words=task.lower().replace(':',' ').split()
    if any(x in words for x in ['remove','delete','erase']):return 'removal'
    if any(x in words for x in ['add','insert','copy','clone','duplicate']):return 'addition_copy'
    return 'replacement_local'

def reporting_tables(s):
    # Derived editorial fields only. Frozen protocol/orders/GT remain unchanged.
    rows=[dict(r,edit_tier=official_action(r['official_task'])) for r in s['protocol']['rows']]
    dev=[r for r in rows if r['split']=='dev'];metadata=[dict(pair_id=r['pair_id'],role=r['role'],split=r['split'],inherited_keyword_tier=old['edit_tier'],official_action=r['edit_tier'],content_priority=r['content_priority'],camera_group=r['camera_group'],task=r['official_task']) for r,old in zip(rows,s['protocol']['rows'])]
    csvsave(OUT/'reporting_categories.csv',metadata)
    s['reference_groups']=grouped(s['reference'],dev);s['final_groups']=grouped(s['final'],dev)
    evaluations=read(OUT/'evaluations.json')
    for h in s['history']:
        if str(h['epoch']) in evaluations:
            h['groups']=grouped(evaluations[str(h['epoch'])]['evaluation'],dev)
    s['best_groups']=grouped(evaluations[str(s['best_epoch'])]['evaluation'],dev)
    groups=dict(reference=s['reference_groups'],**{k:grouped(v['evaluation'],dev) for k,v in evaluations.items()});save(OUT/'groups.json',groups)
    comparison=[]
    for name,evaluation in [('REFERENCE_11',s['reference']),('EPOCH10',s['final']),('SELECTED_ADDITIONAL_DEV',evaluations[str(s['best_epoch'])]['evaluation'])]:
        for row in dev:
            if row['role']!='fake':continue
            fake=next(v for v in evaluation['videos'] if v['video_id']==row['video_id']);real=next(v for v in evaluation['videos'] if v['pair_id']==row['pair_id'] and v['role']=='real')
            comparison.append(dict(condition=name,pair_id=row['pair_id'],historical=row['pair_id'][3:] in HIST,official_action=row['edit_tier'],content_priority=row['content_priority'],camera_group=row['camera_group'],positive_GT_frames=fake['positive_GT_frames'],empty_GT_frames=fake['empty_GT_frames'],**{k:real[k] if k=='real_FPR' else fake[k] for k in METRICS+['constant_AP']}))
    csvsave(OUT/'source_comparison.csv',comparison)
    flat=[]
    for condition,gs in groups.items():
        for group,values in gs.items():flat.append(dict(condition=condition,group=group,**values))
    csvsave(OUT/'group_metrics.csv',flat)
    coverage={split:dict(sources=len({r['source_group_id'] for r in rows if r['split']==split}),videos=sum(r['split']==split for r in rows),legal_frames=sum(len(r['train_targets'] if split=='train' else r['evaluation_targets']) for r in rows if r['split']==split),official_action=dict(collections.Counter(r['edit_tier'] for r in rows if r['split']==split and r['role']=='fake')),content_priority=dict(collections.Counter(r['content_priority'] for r in rows if r['split']==split and r['role']=='fake')),camera_group=dict(collections.Counter(r['camera_group'] for r in rows if r['split']==split and r['role']=='fake'))) for split in ['train','dev']}
    save(OUT/'coverage.json',coverage);s['coverage']=coverage
    s['cost']['expected_formal_new_model_DINO_forwards']=s['updates']+1000+4*(s['protocol']['dev_frames']+sum(len(r['monitor_targets']) for r in rows if r['split']=='train'))
    save(OUT/'summary.json',s);preview(dev)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    measured=[h for h in s['history'] if 'groups' in h]
    for group in ['HISTORICAL_DEV4','ADDITIONAL_DEV','EXPANDED_DEV']:axes[0].plot([h['epoch'] for h in measured],[h['groups'][group]['pixel_AP'] for h in measured],marker='o',label=group)
    axes[0].plot([h['epoch'] for h in measured],[h['monitor_AP'] for h in measured],marker='x',label='fixed train monitor')
    for key in ['pixel_BCE','soft_Dice']:axes[1].plot([h['epoch'] for h in s['history']],[h[key] for h in s['history']],label=key)
    axes[0].set_ylabel('Pixel AP');axes[1].set_ylabel('Mean online loss component')
    for ax in axes:ax.set_xlabel('Complete epoch');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)
    return rows

def report(s):
    completed_checks(s)
    reporting_tables(s)
    lines=['# ANAKIN扩充来源整帧空间训练','计算COMPLETE；单组10完整epoch。官方标签＋技术映射＋来源隔离使用，未逐条人工语义确认。旧人工原话保留，使用授权不改写为人工验收。',f"训练{s['protocol']['train_sources']}来源/{2*s['protocol']['train_sources']}视频、{s['protocol']['training_frames']}目标/轮；开发{s['protocol']['dev_sources']}来源、{s['protocol']['dev_frames']}目标。总{s['updates']}更新，额外BN前向{s['BN_forwards']}。",'|范围/模型|AP|Dice|IoU|召回|real FPR|fake背景FPR|面积|常数AP|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['HISTORICAL_DEV4','ADDITIONAL_DEV','EXPANDED_DEV']:
        for name,g in [('旧11来源',s['reference_groups']),('第10轮',s['final_groups']),('开发选择epoch'+str(s['best_epoch']),s['best_groups'])]:
            v=g[group];lines.append('|'+group+'/'+name+'|'+'|'.join(f'{v[k]:.8f}' if v[k] is not None else '未定义' for k in METRICS+['constant_AP'])+'|')
    lines+=['\n## 全部开发来源','|来源/官方类别|旧AP|末次AP|差值|旧Dice|末次Dice|末次召回|real FPR|常数AP|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['protocol']['rows']:
        if r['split']!='dev' or r['role']!='fake':continue
        a=next(v for v in s['reference']['videos'] if v['video_id']==r['video_id']);b=next(v for v in s['final']['videos'] if v['video_id']==r['video_id']);real=next(v for v in s['final']['videos'] if v['pair_id']==r['pair_id'] and v['role']=='real')
        vals=[a['pixel_AP'],b['pixel_AP'],b['pixel_AP']-a['pixel_AP'] if b['pixel_AP'] is not None and a['pixel_AP'] is not None else None,a['Dice'],b['Dice'],b['GT_recall'],real['real_FPR'],b['constant_AP']];lines.append('|'+r['pair_id']+'/'+official_action(r['official_task'])+'|'+'|'.join(f'{v:.8f}' if v is not None else '未定义' for v in vals)+'|')
    lines+=['\n## 分项与曲线','分项按官方描述的编辑动作优先整理；旧粗关键词把白衣/白帽当外观的3条元数据标签在报告中更正，不改变冻结清单、GT、训练或选择标准，见reporting_categories.csv。','类别来自官方文字或传统背景匹配，并非人工确认；对象编辑和外观/背景参照分开，不把大面积高常数AP的编辑当作运动对象定位已解决。完整分组在groups.json。','|epoch|更新|在线BCE|在线Dice loss|观察AP|历史4 AP|新增开发AP|全部开发AP|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for h in s['history']:
        g=h.get('groups');lines.append(f"|{h['epoch']}|{h['updates']}|{h['pixel_BCE']:.6f}|{h['soft_Dice']:.6f}|"+('|'.join(f'{x:.6f}' for x in [h['monitor_AP'],g['HISTORICAL_DEV4']['pixel_AP'],g['ADDITIONAL_DEV']['pixel_AP'],g['EXPANDED_DEV']['pixel_AP']]) if g else '未评价|未评价|未评价|未评价')+'|')
    lines+=['固定观察每训练视频1/3、2/3合法帧，仅用于观察，不等于完整训练集。AP仅fake非空GT帧连续FP32 logit排序→视频平均→来源等权；空GT AP不填0或1。操作点logit≥0，running BN且eval。','\n## 成本、冻结与边界',json.dumps(s['cost'],ensure_ascii=False),json.dumps(s['coverage'],ensure_ascii=False),f"协议hash {s['identity']['protocol_sha256']}；执行HEAD {s['identity']['execution_head']}。",'旧模型和新模型训练规模/更新不同，只是扩充训练方案比较，非等预算纯样本量因果对照。旧11权重只推理，不改BN；新模型通用DINO和seed20260909的新LoRA/head初始化，不续训。全画面长边384、ImageNet/pad14/去pad/原生恢复，未变更结构、loss、BN或裁剪。','原40y1x9 fake[6,10)不监督不计分；技术待核27及明确反证、冲突/未知身份、94测试保护来源均排除。未扩入其余旧来源。新增MoGe、DFormer、神经特征预计算均0；只新增工作RGB/GT缓存。',f'产物 {OUT}；预览 {OUT}/preview/index.html。计算和Git发布状态分别记录，完成后不追加实验。']
    text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text);(REPO/'docs/pvst/anakin_expanded_spatial_training_v1.md').write_text(text)

def run():
    start=time.monotonic();torch.set_num_threads(4);p=read(OUT/'protocol.json');rows=p['rows'];train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];orders=read(OUT/'orders.json');cache=Cache()
    if (OUT/'summary.json').exists():assert read(OUT/'summary.json')['status']=='COMPLETE';print('EXISTING_COMPLETE');return
    assert not (OUT/'identity.json').exists(),'Incomplete run preserved; explicit recovery required'
    identity=dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),protocol_sha256=digest(OUT/'protocol.json'),orders_sha256=digest(OUT/'orders.json'),generic_weight_sha256=digest(WEIGHT),reference_checkpoint_sha256=digest(OLD/'final.pt'),official_DINO_revision=subprocess.check_output(['git','-C',str(DINO),'rev-parse','HEAD'],text=True).strip(),official_AINPAINT_revision=subprocess.check_output(['git','-C',str(AINPAINT),'rev-parse','HEAD'],text=True).strip())
    save(OUT/'identity.json',identity);base.OUT=OUT
    seed();model=build().cuda();scope=dict(trainable=sum(x.numel() for x in model.parameters() if x.requires_grad),names=[n for n,x in model.named_parameters() if x.requires_grad]);save(OUT/'model_scope.json',scope)
    initial=state_hash(adaptation_state(model));frozen=base.frozen_hash(model);torch.save({k:v.detach().cpu() for k,v in adaptation_state(model).items()},OUT/'initial_adaptation.pt')
    # Existing real preflight performs three discarded updates and validates LoRA/head/padding.
    progress('preflight');check=base.preflight(model,cache,train);step_seconds=np.mean([r['seconds'] for r in check['updates']]);save(OUT/'estimate.json',dict(sample_updates=3,seconds_per_update=step_seconds,updates=p['updates'],estimated_training_seconds=step_seconds*p['updates'],note='preflight estimate only; first full epoch estimate recorded separately'))
    del model;torch.cuda.empty_cache()
    cp=torch.load(OLD/'final.pt',map_location='cpu',weights_only=False);assert cp['condition']=='AINPAINT_SPATIAL' and cp['updates']==12845 and cp['identity']['generic_weight_sha256']==identity['generic_weight_sha256']
    seed();model=build().cuda();load_adaptation(model,cp['adaptation_state']);del cp;before=state_hash(adaptation_state(model));progress('reference_dev');ev,pred=evaluate(model,cache,dev,NAME,capture=True,capture_dtype=np.float32);assert state_hash(adaptation_state(model))==before
    reference=persist('reference_dev',ev);reference_groups=grouped(reference,dev)
    original=read(OLD/'final_dev.json');delta={k:reference_groups['HISTORICAL_DEV4'][k]-original['macro'][k] for k in METRICS};save(OUT/'reference_replay.json',dict(deltas=delta,original_identity_frames=original['scored_frames'],same_adaptation_state_before_after=True));assert max(abs(x) for x in delta.values())<.005,delta
    fixed_images(dev,pred,'old');del model,pred;torch.cuda.empty_cache()
    seed();model=build().cuda();assert state_hash(adaptation_state(model))==initial and base.frozen_hash(model)==frozen;opt=base.optimizer(model);history=[];evs={};step=bn=0;training_seconds=bn_seconds=eval_seconds=0.;begin=time.monotonic();byid={r['video_id']:r for r in train};torch.cuda.reset_peak_memory_stats()
    with (OUT/'updates.jsonl').open('x') as log:
        for epoch,sequence in enumerate(orders,1):
            loss=[]
            for item in sequence:
                rec=base.update(model,opt,cache,byid[item['video_id']],item['target']);step+=1;training_seconds+=rec['seconds'];loss.append([rec['pixel_BCE'],rec['soft_Dice']]);rec.update(updates=step,**item);log.write(json.dumps(rec)+'\n')
                if step%500==0:log.flush();progress('train',epoch=epoch,epochs=len(orders),updates=step,U=p['updates'],loss=rec['loss'],lr=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-begin,remaining_training_seconds=training_seconds/step*(p['updates']-step))
            log.flush();t=time.monotonic();model.train()
            with torch.no_grad():
                for item in sequence[:100]:
                    f=cache.get(byid[item['video_id']],item['target'],'train')
                    with torch.autocast('cuda',dtype=torch.bfloat16):model([f])
                    bn+=1
            torch.cuda.synchronize();bn_seconds+=time.monotonic()-t
            h=dict(epoch=epoch,updates=step,pixel_BCE=float(np.mean(loss,0)[0]),soft_Dice=float(np.mean(loss,0)[1]),elapsed_seconds=time.monotonic()-begin,BN_forwards=bn)
            if epoch in EVAL_EPOCHS:
                progress('evaluate',epoch=epoch,updates=step);mon,_=evaluate(model,cache,train,NAME,monitor=True);persist(f'epoch{epoch:02d}_monitor',mon);h['monitor_AP']=mon['macro']['pixel_AP'];ev,pred=evaluate(model,cache,dev,NAME,capture=epoch==10,capture_dtype=np.float32);persist(f'epoch{epoch:02d}_dev',ev);g=grouped(ev,dev);evs[str(epoch)]=dict(evaluation=ev,groups=g);h['groups']=g;eval_seconds+=mon['seconds']+ev['seconds'];checkpoint(model,opt,epoch,step,identity,initial)
                if epoch==10:fixed_images(dev,pred,'new')
            history.append(h);save(OUT/'history.json',history);csvsave(OUT/'history.csv',[{k:v for k,v in x.items() if k!='groups'} for x in history]);save(OUT/'evaluations.json',evs);progress('epoch_complete',epochs=len(orders),U=p['updates'],loss=h['pixel_BCE']+h['soft_Dice'],lr=opt.param_groups[0]['lr'],remaining_training_seconds=training_seconds/step*(p['updates']-step),**dict(h,elapsed_seconds=time.monotonic()-begin))
    assert step==p['updates'] and bn==1000 and base.frozen_hash(model)==frozen
    best=min(EVAL_EPOCHS,key=lambda e:(-evs[str(e)]['groups']['ADDITIONAL_DEV']['pixel_AP'],e))
    for name,epoch in [('final.pt',10),('best_additional_dev.pt',best)]: (OUT/name).symlink_to(OUT/f'epoch{epoch:02d}.pt')
    costs=dict(training_seconds=training_seconds,BN_seconds=bn_seconds,evaluation_seconds=eval_seconds,train_eval_wall_seconds=time.monotonic()-begin,wall_seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),BN_forwards=bn,reference_dev_frames=reference['scored_frames'],dev_frames=evs['10']['evaluation']['scored_frames'],new_MoGe=0,new_DFormer=0,preparation=read(OUT/'preparation.json'))
    s=dict(status='COMPLETE',exit_code=0,protocol=p,identity=identity,updates=step,BN_forwards=bn,history=history,reference=reference,reference_groups=reference_groups,final=evs['10']['evaluation'],final_groups=evs['10']['groups'],best_epoch=best,best_groups=evs[str(best)]['groups'],cost=costs)
    save(OUT/'groups.json',dict(reference=reference_groups,**{k:v['groups'] for k,v in evs.items()}));save(OUT/'summary.json',s);preview(dev);report(s);save(OUT/'status.json',dict(status='COMPLETE',exit_code=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','report']);args=parser.parse_args()
    try:
        if args.action=='prepare':prepare()
        elif args.action=='run':run()
        else:report(read(OUT/'summary.json'))
    except Exception as e:
        OUT.mkdir(exist_ok=True,parents=True);(OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',dict(status='FAILED',exit_code=1,error=str(e)));raise
