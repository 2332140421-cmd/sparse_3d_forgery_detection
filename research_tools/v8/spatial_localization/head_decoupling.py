"""This two-condition experiment's read-only preparation, smoke and final report.

Called by the existing run.py; training and native evaluation remain shared.
"""
import copy
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.prepare_v8_assets import digest
from research_tools.v8.local_geometry_pilot.runner import seed_all
from .joint_data import Video, save, SEED
from .model import LocalizationModel
from .metrics import evaluate, to_source_rgb
from .data import to_source

CONDITIONS=('COUPLED','SEPARATE_HEADS')


def parameter_sha(model):
    h=hashlib.sha256()
    for k,v in model.named_parameters():
        if k.startswith('detection_head.'):continue
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def initial_model(condition,fitted,source):
    if condition not in CONDITIONS:raise ValueError(condition)
    seed_all(SEED)
    model=LocalizationModel('COMPONENT_LOC',fitted['statistics'],output_mode=condition)
    initial=torch.load(source/'paired_initialization.pt',map_location='cpu',weights_only=False)
    if initial['statistics_sha256']!=digest(source/'statistics.json'):raise ValueError('INITIAL_STATISTICS_CHANGED')
    # Recreating the exact seed/module draw proves this file is the untrained initialization.
    if parameter_sha(model)!=initial['sha256']:raise ValueError('SOURCE_NOT_PAIRED_RANDOM_INITIALIZATION')
    missing=model.load_state_dict(initial['model'],strict=False)
    if missing.unexpected_keys or any(not k.startswith('detection_head.') for k in missing.missing_keys):
        raise ValueError('INCOMPATIBLE_COMMON_MODULES')
    if parameter_sha(model)!=initial['sha256']:raise ValueError('INITIAL_PAYLOAD_MISMATCH')
    if condition=='SEPARATE_HEADS':model.detection_head.load_state_dict(model.structure.cell_head.state_dict())
    return model


def source_files(source,rows):
    files=[source/n for n in ['joint_manifest.json','statistics.json','paired_initialization.pt','progress.json']]
    for c in ['CELL_LOC','COMPONENT_LOC']:
        files.extend(source/c/n for n in ['summary.json','history.json','history.csv','best.pt','last.pt'])
    files.extend(source/'cache'/r['video_id']/'complete.json' for r in rows)
    return files


def prepare(config,out):
    if config['epochs']!=30 or config['seed']!=SEED:raise ValueError('FROZEN_30_EPOCH_SINGLE_SEED_PROTOCOL')
    source=Path(config['source_run']).resolve();out=out.resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('OUTPUT_SOURCE_OVERLAP')
    if Path(config['cache_root']).resolve()!=source/'cache':raise ValueError('EXPLICIT_SOURCE_CACHE_REQUIRED')
    manifest=json.loads((source/'joint_manifest.json').read_text());fitted=json.loads((source/'statistics.json').read_text())
    rows=[copy.deepcopy(r) for r in manifest['rows'] if r['split']=='dev' or (r['split']=='train' and r.get('train_monitor'))]
    for dataset in ['ActivityForensics','ANAKIN']:
        for split,count in [('train',8),('dev',16)]:
            rr=[r for r in rows if r['dataset']==dataset and r['split']==split]
            if len(rr)!=count or len({r['pair_id'] for r in rr})!=count//2:raise ValueError('FROZEN_SUBSET_COUNTS')
    if set(r['source_id'] for r in rows if r['split']=='train')&set(r['source_id'] for r in rows if r['split']=='dev'):
        raise ValueError('SOURCE_LEAKAGE')
    if fitted['train_video_ids']!=[r['video_id'] for r in manifest['rows'] if r['split']=='train']:
        raise ValueError('ORIGINAL_TRAIN_STATISTICS_IDENTITY')
    if len(fitted['train_video_ids'])!=96:raise ValueError('ORIGINAL_STATISTICS_EXPECTED_96_TRAIN')
    refs_path=out/'asset_references.json'
    if refs_path.exists():verify_source(out)
    checks=[]
    for r in rows:
        video=Video(out,r,fitted['statistics'],fitted['grouping'],cache_root=config['cache_root'],read_only=True)
        try:
            required={'x','mu','valid','content','area','predecessor','association_weight','history_available',
                      'delta_t','rgb512','xyz','gt','pixel_known','frame_label','content512'}
            if not required<=set(video.arrays):raise ValueError('MISSING_CANONICAL_CACHE_FIELDS:'+r['video_id'])
            for name,array in video.arrays.items():
                if len(array)!=r['actual_frames']:raise ValueError('CACHE_FRAME_COUNT:'+r['video_id']+':'+name)
            if video.arrays['rgb512'].shape[1:]!=(512,512,3) or video.arrays['gt'].shape[1:]!=(512,512):
                raise ValueError('CACHE_512_SHAPE')
            if video.fitted_components.shape!=(r['actual_frames'],256):raise ValueError('COMPONENT_FRAME_COUNT')
            meta=json.loads((video.directory/'complete.json').read_text())
            if len(meta['frames'])!=len(r['pts']):raise ValueError('CACHE_PTS_LENGTH')
            if any(abs(a['pts_s']-b['pts_s'])>1e-8 for a,b in zip(meta['frames'],r['pts'])):raise ValueError('CACHE_NATIVE_PTS')
            checks.append({'video_id':r['video_id'],'frames':r['actual_frames'],'status':'VERIFIED_READ_ONLY',
                           'cache_metadata_sha256':digest(video.directory/'complete.json')})
        finally:video.close()
    frozen={'schema':config['protocol'],'selection':'exact original train_monitor plus all dev; no resampling',
            'source_manifest_sha256':digest(source/'joint_manifest.json'),'rows':rows}
    path=out/'joint_manifest.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('FROZEN_MANIFEST_CHANGED')
    save(path,frozen)
    stat_path=out/'statistics.json'
    if stat_path.exists() and digest(stat_path)!=digest(source/'statistics.json'):raise ValueError('FROZEN_STATISTICS_CHANGED')
    if not stat_path.exists():stat_path.write_bytes((source/'statistics.json').read_bytes())
    if not refs_path.exists():
        files=source_files(source,rows)
        save(refs_path,{'source_run':str(source),'cache_root':config['cache_root'],
            'normalization_origin':'96 original training videos; no dev fit; only16 updated in this probe',
            'small_file_hashes':{str(p):digest(p) for p in files if p.suffix!='.pt'},
            'checkpoint_size_mtime':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in files if p.suffix=='.pt'},
            'initialization_sha256':digest(source/'paired_initialization.pt'),'cache_checks':checks})
    return rows,fitted


def verify_source(out):
    refs=json.loads((out/'asset_references.json').read_text())
    for filename,sha in refs['small_file_hashes'].items():
        if digest(Path(filename))!=sha:raise ValueError('READ_ONLY_SOURCE_CHANGED:'+filename)
    for filename,expected in refs['checkpoint_size_mtime'].items():
        p=Path(filename)
        if [p.stat().st_size,p.stat().st_mtime_ns]!=expected:raise ValueError('SOURCE_CHECKPOINT_CHANGED:'+filename)
    if digest(Path(refs['source_run'])/'paired_initialization.pt')!=refs['initialization_sha256']:
        raise ValueError('SOURCE_INITIALIZATION_CHANGED')


def smoke(config,out,rows,fitted,update_pair,progress,parameters_sha):
    path=out/'resource_smoke.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():raise ValueError('SMOKE_CONFIG_CHANGED')
        return old
    records=[];common=None
    for c in CONDITIONS:
        model=initial_model(c,fitted,Path(config['source_run'])).cuda();sha=parameter_sha(model)
        if common is not None and sha!=common:raise ValueError('UNPAIRED_COMMON_PARAMETERS')
        common=sha;opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        for dataset in ['ActivityForensics','ANAKIN']:
            first=next(r['pair_id'] for r in rows if r['dataset']==dataset and r['split']=='train')
            pair=[r for r in rows if r['pair_id']==first];model.train();before=parameters_sha(model)
            progress(out,'smoke_update',condition=c,dataset=dataset);torch.cuda.reset_peak_memory_stats()
            updated=update_pair(model,opt,pair,out,fitted,config,scaler)
            if before==parameters_sha(model):raise ValueError('NO_SMOKE_UPDATE')
            options={'cache_root':config['cache_root'],'read_only':True}
            scored=evaluate(model,out,pair,fitted,config['chunk_frames'],config['amp'],**options)
            ap_eval=evaluate(model,out,pair,fitted,config['chunk_frames'],config['amp'],pixel_ap=True,**options) if dataset=='ANAKIN' else None
            records.append({'condition':c,'dataset':dataset,'pair_id':first,**updated,'evaluation_s':scored['elapsed_s'],
                'exact_ap_evaluation_s':ap_eval['elapsed_s'] if ap_eval else None,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in model.parameters()),
                'additional_parameters':sum(p.numel() for p in model.detection_head.parameters()) if hasattr(model,'detection_head') else 0})
        del model,opt;torch.cuda.empty_cache()
    estimates={}
    for c in CONDITIONS:
        train_s=eval_s=ap_s=0.
        for record in [r for r in records if r['condition']==c]:
            train_frames=sum(r['actual_frames'] for r in rows if r['dataset']==record['dataset'] and r['split']=='train')
            all_frames=sum(r['actual_frames'] for r in rows if r['dataset']==record['dataset'])
            train_s+=record['elapsed_s']/record['frames']*train_frames*30
            eval_s+=record['evaluation_s']/record['frames']*all_frames*30
            if record['exact_ap_evaluation_s'] is not None:
                ap_s+=max(0,record['exact_ap_evaluation_s']-record['evaluation_s'])/record['frames']*all_frames*6
        estimates[c]={'training_s':train_s,'routine_eval_s':eval_s,'extra_AP_extrapolated_s':ap_s,'total_s':train_s+eval_s+ap_s}
    result={'status':'COMPLETE','config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'common_initial_parameters_sha256':common,'records':records,'projection_30_epochs':estimates,
        'projection_limits':'one full pair/dataset; excludes final best/last extra eval and exports; includes measured ANAKIN AP extrapolation',
        'smoke_weights':'discarded; formal conditions recreate the paired random initialization'}
    save(path,result);return result


def curves(out,condition,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in history];fig,axes=plt.subplots(3,4,figsize=(18,11))
    for ax,key in zip(axes.flat[:5],['loss','video','frame','pixel_bce','pixel_dice']):
        ax.plot(epochs,[r['online_train'][key] for r in history],label='online',linestyle=':')
        k='loss' if key=='loss' else 'loss_'+key
        for split in ['train','dev']:
            for d in ['ANAKIN','ActivityForensics']:
                ax.plot(epochs,[r[split][d][k] for r in history],label=split+'/'+d)
        ax.set_title(key+' loss; frame before x0.5')
    for ax,key in zip(axes.flat[5:7],['spatial_dice','spatial_iou']):
        for split in ['train','dev']:ax.plot(epochs,[r[split]['ANAKIN'][key] for r in history],label=split)
        ax.set_title('ANAKIN '+key)
    for ax,task,key in [(axes.flat[7],'video','auroc'),(axes.flat[8],'frame','auroc'),(axes.flat[9],'frame','ap')]:
        for split in ['train','dev']:
            for d in ['ANAKIN','ActivityForensics']:ax.plot(epochs,[r[split][d][task][key] for r in history],label=split+'/'+d)
        ax.set_title(task+' '+key)
    for split in ['train','dev']:
        for d in ['ANAKIN','ActivityForensics']:axes.flat[10].plot(epochs,[r[split][d]['real_spatial_fpr'] for r in history],label=split+'/'+d)
    axes.flat[10].set_title('Real spatial FP');axes.flat[11].plot(epochs,[r['gradient_norm_mean'] for r in history],label='pre-clip norm (clip1)')
    axes.flat[11].set_title('Gradient norm; lr=1e-4 (CSV)')
    for ax in axes.flat:ax.legend(fontsize=5);ax.grid(alpha=.2);ax.set_xlabel('epoch')
    fig.tight_layout();fig.savefig(out/condition/'curves.png');plt.close(fig)


def sample_index(out,rows,conditions=CONDITIONS,title='第30轮主比较；best补充'):
    from PIL import Image, ImageDraw
    import matplotlib
    directory=out/'comparison';directory.mkdir(exist_ok=True);entries=[];details=[]
    for checkpoint in ['last','best']:
        for r in rows:
            if r['dataset']!='ANAKIN':continue
            for fi in [int(r['actual_frames']*f) for f in [.25,.5,.75]]:
                name=f'{r["video_id"]}_{fi}';data=[]
                for c in conditions:
                    with np.load(out/c/'samples'/checkpoint/r['split']/(name+'.npz'),allow_pickle=False) as z:
                        data.append({k:z[k].copy() for k in z.files})
                if not np.array_equal(data[0]['native_gt'],data[1]['native_gt']):raise ValueError('SAMPLE_GT_MISMATCH')
                gt=data[0]['native_gt'].astype(bool);zs=[to_source(d['logits'],r['mapping512']) for d in data]
                rgb=to_source_rgb(data[0]['rgb512'],r['mapping512'])
                binary=[np.repeat((a>=0)[...,None]*np.uint8(255),3,-1) for a in zs]
                gt_rgb=np.repeat(gt[...,None]*np.uint8(255),3,-1)
                heat=[(matplotlib.colormaps['coolwarm'](np.clip((a+6)/12,0,1))[...,:3]*255).astype(np.uint8) for a in zs]
                w=320;h=max(1,round(rgb.shape[0]*w/rgb.shape[1]));panel=Image.new('RGB',(w*4,2*(h+28)),'white');draw=ImageDraw.Draw(panel)
                for row,images in enumerate([[rgb,gt_rgb,*binary],[rgb,gt_rgb,*heat]]):
                    titles=['RGB','GT edit region',*[c+' binary>=0' for c in conditions]] if row==0 else ['RGB','GT',*[c+' logit [-6,+6]' for c in conditions]]
                    for j,a in enumerate(images):
                        draw.text((j*w+4,row*(h+28)+5),titles[j],fill='black')
                        interpolation=Image.Resampling.NEAREST if j==1 or (row==0 and j>=2) else Image.Resampling.BILINEAR
                        panel.paste(Image.fromarray(a).resize((w,h),resample=interpolation),(j*w,row*(h+28)+28))
                png=f'{checkpoint}_{name}.png';panel.save(directory/png)
                detail={'checkpoint':checkpoint,'video_id':r['video_id'],'pair_id':r['pair_id'],'split':r['split'],'role':r['role'],
                        'frame_index':fi,'pts_s':float(data[0]['pts_s']),'conditions':{}}
                for c,a in zip(conditions,zs):
                    pred=a>=0;tp=int((pred&gt).sum());fg=int(gt.sum());pp=int(pred.sum())
                    detail['conditions'][c]={'dice':2*tp/(fg+pp) if fg else None,'iou':tp/(fg+pp-tp) if fg else None,
                        'gt_recall':tp/fg if fg else None,'background_fpr':float((pred&~gt).sum()/max(1,(~gt).sum())),
                        'predicted_area':float(pred.mean())}
                details.append(detail)
                entries.append('<h3>'+html.escape(f'{checkpoint} {r["split"]} {r["pair_id"]} {r["role"]} frame{fi} PTS{float(data[0]["pts_s"]):.4f}')+'</h3><p>'+html.escape(r['task'])+'</p><img loading="lazy" style="max-width:100%" src="comparison/'+png+'">')
    save(out/'fixed_sample_statistics.json',details)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V8空间定位对照</title><h1>'+html.escape(title)+'</h1><p>'+html.escape('RGB | GT | '+' | '.join(conditions))+'</p><p>固定25/50/75%；二值阈值0，连续logit固定绝对色阶[-6,+6]、超界仅显示裁剪，原始float32 NPZ不裁剪。无视频阈值门控。</p>'+''.join(entries))
    return details


def report(out,rows):
    summaries={c:json.loads((out/c/'summary.json').read_text()) for c in CONDITIONS}
    if any(s['last_epoch']!=30 for s in summaries.values()):raise ValueError('FIXED_30_EPOCHS_REQUIRED')
    for c in CONDITIONS:curves(out,c,json.loads((out/c/'history.json').read_text()))
    samples=sample_index(out,rows);lines=['# V8检测／空间定位输出分离对照','',
        '主比较：两条件相同第30轮；best按各自开发ANAKIN Dice选择，仅作补充。共享完整特征未detach，任务仍共同影响共享编码器。','',
        '|条件|checkpoint/epoch|集合|数据集|Dice/IoU/pixel AP|real空间FP|video AUROC/AP|frame AUROC/AP|',
        '|---|---|---|---|---|---|---|---|']
    fmt=lambda v:'未计算' if v is None else f'{v:.6f}'
    for checkpoint in ['last','best']:
        for c,s in summaries.items():
            for split,key in [('train',checkpoint+'_train'),('dev',checkpoint+'_metrics')]:
                for dataset in ['ANAKIN','ActivityForensics']:
                    m=s[key]['datasets'][dataset]
                    lines.append(f'|{c}|{checkpoint}/{s[checkpoint+"_epoch"]}|{split}|{dataset}|'+
                        '/'.join(fmt(m[k]) for k in ['spatial_dice','spatial_iou','pixel_ap'])+f'|{fmt(m["real_spatial_fpr"])}|'+
                        '/'.join(fmt(m['video'][k]) for k in ['auroc','ap'])+'|'+ '/'.join(fmt(m['frame'][k]) for k in ['auroc','ap'])+'|')
    a,b=[summaries[c] for c in CONDITIONS]
    tr=b['last_train']['datasets']['ANAKIN']['spatial_dice']-a['last_train']['datasets']['ANAKIN']['spatial_dice']
    dv=b['last_metrics']['datasets']['ANAKIN']['spatial_dice']-a['last_metrics']['datasets']['ANAKIN']['spatial_dice']
    # Quantified descriptive differences, not significance thresholds or model selection.
    lines.extend(['',f'第30轮SEPARATE_HEADS−COUPLED：训练Dice {tr:+.6f}，开发Dice {dv:+.6f}。',
        '完整训练集16视频、开发32视频；不把旧96视频pilot或历史COMPONENT_ADAPTED当匹配对照。'])
    if tr>0 and dv<=0:
        decision='下一步优先增加空间标注source及编辑形态覆盖：训练改善未转化为开发改善。'
    elif tr>0 and dv>0:
        decision='下一步优先增加独立空间标注source及编辑形态覆盖，验证本次训练/开发改善能否稳定复现；单seed不证明普遍收益。'
    else:
        da=b['last_metrics']['datasets']['dataset_macro']['auroc']-a['last_metrics']['datasets']['dataset_macro']['auroc']
        decision='下一步优先核查像素特征或空间监督的可拟合性；检测排序改善未伴随训练定位改善。' if da>0 else '下一步不扩大该修改，先重新判断完整定位路径的可拟合性限制。'
    lines.extend([decision,'这按连续差值描述方向，没有预注册的显著性或实际重要性阈值；微小差异不等于明确改善。'])
    for split in ['train','dev']:
        rr=[r for r in samples if r['checkpoint']=='last' and r['split']==split]
        for metric in ['background_fpr','gt_recall','dice']:
            values={c:[r['conditions'][c][metric] for r in rr if r['conditions'][c][metric] is not None] for c in CONDITIONS}
            means={c:float(np.mean(values[c])) for c in CONDITIONS}
            lines.append(f'固定三时刻{split} {metric}均值：'+', '.join(f'{c}={means[c]:.6f}' for c in CONDITIONS)+
                f'；差值{means[CONDITIONS[1]]-means[CONDITIONS[0]]:+.6f}，只描述固定样例，不替代全视频主指标。')
    lines.extend(['','第30轮全部ANAKIN开发fake（不删低分样本）：','|video|条件|Dice|IoU|pixel AP|video logit|','|---|---|---|---|---|---|'])
    for c,s in summaries.items():
        for r in s['last_metrics']['records']:
            if r['dataset']=='ANAKIN' and r['role']=='fake':
                lines.append(f'|{r["pair_id"]}|{c}|'+ '|'.join(fmt(r[k]) for k in ['spatial_dice','spatial_iou','pixel_ap','logit'])+'|')
    lines.append('逐视频/帧检测分数与空间指标见各轮epoch_metrics；固定样例位置和形状见index.html。没有按视频预测屏蔽mask。')
    costs={c:{'active_training_evaluation_s':s['elapsed_s'],'parameters':s['parameters'],'added_parameters':s['additional_detection_parameters']} for c,s in summaries.items()}
    costs['new_output_logical_bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    costs['original_cache_copied_bytes']=0
    save(out/'costs.json',costs);save(out/'result_summary.json',summaries)
    lines.extend(['', '成本（实际墙钟含训练及评价；不含Codex实现时间）：',json.dumps(costs,ensure_ascii=False),
        '固定30轮轨迹、loss分项、梯度范数和实际峰值显存见history.csv/curves.png。预算内仍低的训练Dice不证明架构绝对无效。',
        '仅完成两个条件、seed20260909。本轮结束，不自动扩大数据、追加seed或调整阈值。'])
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')
