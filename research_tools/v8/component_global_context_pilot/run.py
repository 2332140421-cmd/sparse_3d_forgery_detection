"""One cache-only matched experiment; reuses V8 weighting and metric functions."""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import io
import os
import random
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file
from research_tools.v8.full_coverage_observation_field.pipeline import _metric, _source_macro, _bootstrap_diff, _source_class_weights
from research_tools.v8.local_geometry_pilot.data import Dataset, digest
from research_tools.v8.local_geometry_pilot.runner import atomic_torch, seed_all, rng_state, restore_rng, validate_checkpoint, csv_rows
from .model import GlobalContextModel, CONDITIONS, area_lse

SEEDS = (20260909, 20260910, 20260911)
STOP = False


def save_json(path, value):
    """Strict finite JSON; undefined metrics are already None at their source."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(payload + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def prepare(reference, base, out, progress):
    saved = read_json(reference / "inputs/manifest.json")
    original = read_json(base / "data_manifest.json")["rows"]
    if saved["identity"]["baseline_manifest_sha256"] != sha256_file(base / "data_manifest.json"):
        raise ValueError("BASELINE_MANIFEST_CHANGED")
    rows = saved["rows"]
    if len(rows) != 801 or len({r['window_id'] for r in rows}) != 801:
        raise ValueError("WINDOW_IDENTITIES")
    original_map = {r['window_id']: r for r in original}
    for i, r in enumerate(rows):
        if any(r[k] != v for k, v in original_map[r['window_id']].items()):
            raise ValueError(f"FROZEN_ROW_CHANGED:{r['window_id']}")
        meta = read_json(base / 'frontend' / (Path(r['shared_path']).name.split('.')[0] + '.json'))
        for key in ('window_id', 'source_id', 'role', 'video_sha256'):
            if meta[key] != r[key]: raise ValueError(f"FRONTEND_IDENTITY:{r['window_id']}:{key}")
        if meta['frame_indices'] != r['v8_frame_indices']: raise ValueError('FRAME_IDENTITY')
        np.testing.assert_allclose(meta['pts_s'], r['v8_timestamps_s'], atol=1e-8, rtol=0)
        for kind in ('shared', 'valid', 'content'):
            if sha256_file(r[kind + '_path']) != r[kind + '_sha256']:
                raise ValueError(f"INPUT_HASH:{r['window_id']}:{kind}")
        if (r['role'] == 'fake') != bool(r['label']): raise ValueError('LABEL_ROLE')
        if i % 100 == 0: progress('input_audit', i + 1, 801)
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'validation']
    if (len(train), len(val), len({r['source_id'] for r in train}), len({r['source_id'] for r in val})) != (718, 83, 122, 14):
        raise ValueError('FROZEN_SPLIT_COUNTS')
    if {r['source_id'] for r in train} & {r['source_id'] for r in val}: raise ValueError('SOURCE_LEAKAGE')
    statistics = read_json(reference / 'standardizer.json')
    if sha256_file(reference / 'standardizer.json') != saved['standardizer_sha256']: raise ValueError('STATISTICS_IDENTITY')
    if sha256_file(reference / 'grouping_config.json') != saved['grouping_sha256']: raise ValueError('GROUPING_IDENTITY')
    input_id = digest({'reference_manifest_sha256': sha256_file(reference / 'inputs/manifest.json'),
                       'baseline_manifest_sha256': sha256_file(base / 'data_manifest.json'),
                       'consumed': [{k: r[k] for k in ('window_id','label','split','shared_sha256','valid_sha256','content_sha256')} for r in rows],
                       'statistics_sha256': saved['standardizer_sha256'], 'grouping_sha256': saved['grouping_sha256']})
    save_json(out / 'input_manifest.json', {'input_id': input_id, 'rows': rows, 'reference': str(reference),
               'note': 'Read-only V8 mean/mask/shared caches; dense XYZ is not needed by MEAN_REPLICATED.'})
    save_json(out / 'standardizer.json', statistics)
    save_json(out / 'grouping_config.json', read_json(reference / 'grouping_config.json'))
    return rows, statistics, input_id


def load_inputs(rows, statistics):
    """Preload the small exact MEAN inputs, not dense XYZ or decoded RGB copies."""
    started = time.monotonic()
    old = Dataset(rows, statistics)
    arrays = {k: torch.from_numpy(np.stack([s[k] for s in old.shared])) for k in old.shared[0]}
    for key in ('valid', 'content'):
        arrays[key] = torch.from_numpy(np.stack([np.load(r[key + '_path'], allow_pickle=False) for r in rows]))
    arrays['label'] = torch.tensor([float(r['label']) for r in rows])
    nbytes = sum(v.numel() * v.element_size() for v in arrays.values())
    if not all(torch.isfinite(v).all() for k, v in arrays.items() if v.is_floating_point()):
        raise ValueError('NONFINITE_VALID_INPUT')
    area_counts = arrays['content'].sum((-1, -2)).to(arrays['area'])
    torch.testing.assert_close(area_counts, arrays['area'], atol=0, rtol=0)
    if (arrays['valid'] & ~arrays['content']).any(): raise ValueError('GEOMETRY_OUTSIDE_CONTENT')
    if arrays['component_ids'].min() < 0 or arrays['component_ids'].max() >= 256: raise ValueError('COMPONENT_INDEX')
    arrays = {k: v.to('cuda') for k,v in arrays.items()}
    torch.cuda.synchronize()
    return arrays, {'preload_s': time.monotonic() - started, 'resident_input_bytes': nbytes}


def select(arrays, indices):
    ix = torch.as_tensor(indices, device='cuda', dtype=torch.long)
    return {k: v.index_select(0, ix) for k,v in arrays.items()}


def scores(model, arrays, indices, microbatch):
    model.eval(); result = []
    with torch.inference_mode():
        for start in range(0, len(indices), microbatch):
            result.append(model(select(arrays, indices[start:start + microbatch])).cpu().numpy())
    z = np.concatenate(result)
    if not np.isfinite(z).all(): raise FloatingPointError('NONFINITE_MODEL_SCORES')
    return z


def epoch(model, arrays, order, weights, optimizer, microbatch):
    model.train(); losses = []; optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(order), 8):
        group = order[start:start + 8]
        for j in range(0, len(group), microbatch):
            ids = group[j:j + microbatch]; batch = select(arrays, ids)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch['label'],
                      weight=weights[torch.as_tensor(ids, device='cuda')], reduction='sum')
            (loss / len(group)).backward(); losses.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    value = float((torch.stack(losses).sum() / len(order)).item())
    if not np.isfinite(value): raise FloatingPointError('NONFINITE_TRAINING_LOSS')
    return value


def smoke(out, rows, arrays, stats, identity, microbatch):
    ids = [next(i for i,r in enumerate(rows) if r['split']=='train' and r['role']==role) for role in ('real','fake')]
    seed_all(SEEDS[0]); a = GlobalContextModel(CONDITIONS[0], stats).cuda()
    seed_all(SEEDS[0]); b = GlobalContextModel(CONDITIONS[1], stats).cuda()
    for x,y in zip(a.parameters(), b.parameters()): torch.testing.assert_close(x,y,atol=0,rtol=0)
    batch = select(arrays, ids)
    opt = torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.monotonic()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(a(batch),batch['label']);loss.backward()
    for module in (a.region_projection,a.region_attention,a.component_residual,a.cell_head,a.gru,a.patch_encoder):
        gg=[p.grad for p in module.parameters()]
        if not all(g is not None and torch.isfinite(g).all() for g in gg): raise ValueError('SMOKE_GRADIENT')
        if not any(torch.count_nonzero(g)>0 for g in gg): raise ValueError('SMOKE_ZERO_GRADIENT')
    torch.nn.utils.clip_grad_norm_(a.parameters(),1.);opt.step();torch.cuda.synchronize()
    step_s=time.monotonic()-started
    a.eval()
    with torch.inference_mode(): expected=a(batch,True)
    path=out/'smoke';path.mkdir(exist_ok=True)
    atomic_torch(path/'checkpoint.pt',{'model':a.state_dict(),'optimizer':opt.state_dict(),'identity':identity,'rng':rng_state()})
    restored=GlobalContextModel(CONDITIONS[0],stats).cuda()
    cp=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    restored.load_state_dict(cp['model']);restored.eval()
    ropt=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001);ropt.load_state_dict(cp['optimizer'])
    with torch.inference_mode(): actual=restored(batch,True)
    torch.testing.assert_close(expected['window_logit'],actual['window_logit'],atol=1e-6,rtol=1e-6)
    two=area_lse(actual['frame_logits'],batch['area'].sum(2),(1,))
    torch.testing.assert_close(two,actual['window_logit'],atol=1e-6,rtol=1e-6)
    # Optimizer continuation, not just weight loading.
    for net,optimizer in ((a,opt),(restored,ropt)):
        optimizer.zero_grad(set_to_none=True);net.train()
        torch.nn.functional.binary_cross_entropy_with_logits(net(batch),batch['label']).backward();optimizer.step()
    for x,y in zip(a.parameters(),restored.parameters()): torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
    # Representative full physical batch, same forward/backward and clipping.
    del a,b,restored,opt,ropt,batch
    model=GlobalContextModel(CONDITIONS[0],stats).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train_ids=[i for i,r in enumerate(rows) if r['split']=='train'][:8]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();ts=time.monotonic()
    epoch(model,arrays,train_ids,torch.ones(len(rows),device='cuda'),optimizer,microbatch)
    torch.cuda.synchronize();full_s=time.monotonic()-ts
    record={'status':'PASS','identity':identity,'windows':[rows[i]['window_id'] for i in ids],
            'two_window_step_s':step_s,'effective_batch_step_s':full_s,'windows_per_second':8/full_s,
            'epoch_estimate_s':full_s*90,'six_model_training_estimate_s':full_s*90*600,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'parameter_count':sum(p.numel() for p in model.parameters()),'save_reload_error':float((expected['window_logit']-actual['window_logit']).abs().max()),
            'metric_format':_metric(np.array([rows[i]['label'] for i in ids]),actual['window_logit'].cpu().numpy()),
            'note':'First-step estimate includes initialization; no smoke model enters formal results.'}
    save_json(path/'summary.json',record);return record


def train(out, rows, arrays, stats, identity, microbatch, progress):
    model_dir=out/'models';model_dir.mkdir(exist_ok=True);records=[]
    train_ids=[i for i,r in enumerate(rows) if r['split']=='train'];val_ids=[i for i,r in enumerate(rows) if r['split']=='validation']
    wm=_source_class_weights([rows[i] for i in train_ids]);save_json(out/'training_weights.json',wm)
    weights=torch.tensor([wm.get(r['window_id'],0.) for r in rows],device='cuda')
    for seed in SEEDS:
        for condition in CONDITIONS:
            seed_all(seed);model=GlobalContextModel(condition,stats).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            order_rng=torch.Generator().manual_seed(seed)
            initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
            key=f'{condition}__seed{seed}';final=model_dir/f'{key}.pt';last=model_dir/f'{key}.last.pt'
            mid={**identity,'condition':condition,'seed':seed};ep=0;losses=[];elapsed=0.;peak=0.;order_hashes=[]
            if final.exists() or last.exists():
                cp=torch.load(final if final.exists() else last,map_location='cpu',weights_only=False)
                validate_checkpoint(cp,mid)
                if cp['microbatch']!=microbatch or cp['initial_parameter_sha256']!=initial_hash:raise ValueError('RESUME_CONFIG')
                model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
                ep,losses,elapsed,peak=cp['epoch'],cp['loss_history'],cp['training_elapsed_s'],cp['peak_allocated_gib']
                restore_rng(cp['rng']);order_rng.set_state(cp['order_rng']);order_hashes=cp['order_hashes']
            print(f'MODEL {key} epoch={ep} CUDA={next(model.parameters()).device}',flush=True)
            while ep<100:
                torch.cuda.reset_peak_memory_stats();ts=time.monotonic()
                order=[train_ids[i] for i in torch.randperm(len(train_ids),generator=order_rng).tolist()]
                loss=epoch(model,arrays,order,weights,optimizer,microbatch);torch.cuda.synchronize()
                elapsed+=time.monotonic()-ts;ep+=1;losses.append(loss);order_hashes.append(digest(order))
                peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
                cp={'identity':mid,'condition':condition,'seed':seed,'epoch':ep,'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'rng':rng_state(),'order_rng':order_rng.get_state(),
                    'loss_history':losses,'training_elapsed_s':elapsed,'peak_allocated_gib':peak,'microbatch':microbatch,
                    'initial_parameter_sha256':initial_hash,'parameter_count':sum(p.numel() for p in model.parameters()),'order_hashes':order_hashes}
                atomic_torch(last,cp)
                progress('train',len(records),6,condition=condition,seed=seed,epoch=ep,loss=loss,model_elapsed_s=elapsed)
                print(f'EPOCH {key} {ep}/100 loss={loss:.8g} elapsed_s={elapsed:.2f}',flush=True)
                if STOP: raise InterruptedError('SAFE_STOP_AFTER_SAVED_EPOCH')
            if not final.exists():atomic_torch(final,cp)
            score_path=model_dir/f'{key}.scores.npz';ts=time.monotonic()
            if score_path.exists():
                with np.load(score_path,allow_pickle=False) as z:
                    if str(z['identity'])!=digest(mid) or str(z['checkpoint_sha256'])!=sha256_file(final):raise ValueError('SCORES_IDENTITY')
            else:
                tz,vz=scores(model,arrays,train_ids,microbatch),scores(model,arrays,val_ids,microbatch)
                tmp=score_path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez(f,train_scores=tz,val_scores=vz,identity=digest(mid),checkpoint_sha256=sha256_file(final))
                os.replace(tmp,score_path)
            rec={'condition':condition,'seed':seed,'epoch':ep,'status':'COMPLETE','checkpoint':str(final),
                 'checkpoint_sha256':sha256_file(final),'parameter_count':cp['parameter_count'],'loss_first':losses[0],
                 'loss_last':losses[-1],'loss_epochs':len(losses),'training_elapsed_s':elapsed,'peak_allocated_gib':peak,
                 'score_s':time.monotonic()-ts,'initial_parameter_sha256':initial_hash,'order_hashes':order_hashes,'identity':mid}
            records.append(rec);save_json(model_dir/'fold_models.json',{'planned':6,'completed':len(records),'failed':0,'records':records})
            csv_rows(model_dir/f'{key}.loss.csv',[{'epoch':i+1,'loss':v} for i,v in enumerate(losses)])
            del model,optimizer
    for seed in SEEDS:
        pair=[r for r in records if r['seed']==seed]
        if pair[0]['initial_parameter_sha256']!=pair[1]['initial_parameter_sha256'] or pair[0]['order_hashes']!=pair[1]['order_hashes']:
            raise ValueError('PAIRED_INIT_OR_ORDER_CHANGED')
    return records


def evaluate(out, rows, records):
    ev=out/'evaluation';ev.mkdir(exist_ok=True);metrics=[];per_source=[];window_rows=[];macro={};seed_macro={}
    for condition in CONDITIONS:
        for split,field in (('train','train_scores'),('validation','val_scores')):
            rr=[r for r in rows if r['split']==split];y=np.array([r['label'] for r in rr]);zz=[]
            for seed in SEEDS:
                with np.load(out/'models'/f'{condition}__seed{seed}.scores.npz',allow_pickle=False) as z:zz.append(z[field].copy())
            for seed,z in list(zip(SEEDS,zz))+[('MEAN_LOGIT',np.mean(zz,axis=0))]:
                if len(z)!=len(rr) or not np.isfinite(z).all():raise ValueError('SCORE_COVERAGE')
                sm=_source_macro(rr,z,y,[r['window_id'] for r in rr])
                metrics.append({'condition':condition,'seed':seed,'split':split,**_metric(y,z),
                                'source_macro_auroc':sm['mean'],'source_count':sm['source_count'],
                                'n_real':int((y==0).sum()),'n_fake':int((y==1).sum())})
                if split=='validation':
                    if seed=='MEAN_LOGIT':macro[condition]=sm['per_source']
                    else:seed_macro[(condition,seed)]=sm['mean']
                for s,auc in sm['per_source'].items():
                    subset=[r for r in rr if r['source_id']==s]
                    per_source.append({'condition':condition,'seed':seed,'split':split,'source':s,'auroc':auc,
                                       'n_real':sum(r['label']==0 for r in subset),'n_fake':sum(r['label']==1 for r in subset)})
                window_rows.extend({'condition':condition,'seed':seed,'split':split,'window_id':r['window_id'],
                                    'source':r['source_id'],'label':r['label'],'logit':float(v)} for r,v in zip(rr,z))
    comparison=_bootstrap_diff(macro[CONDITIONS[0]],macro[CONDITIONS[1]],reps=10000,seed=20260909)
    differences=[{'source':s,'component_auroc':macro[CONDITIONS[0]][s],'cell_auroc':macro[CONDITIONS[1]][s],'difference':v}
                 for s,v in comparison['per_source'].items()]
    for r in differences:
        rr=[z for z in rows if z['split']=='validation' and z['source_id']==r['source']]
        r.update(n_real=sum(z['label']==0 for z in rr),n_fake=sum(z['label']==1 for z in rr))
        for seed in SEEDS:
            a={x['condition']:x['auroc'] for x in per_source if x['split']=='validation' and x['source']==r['source'] and x['seed']==seed}
            r[f'difference_seed_{seed}']=a[CONDITIONS[0]]-a[CONDITIONS[1]]
    comparison['directions']={'positive':sum(r['difference']>0 for r in differences),'tied':sum(r['difference']==0 for r in differences),'negative':sum(r['difference']<0 for r in differences)}
    comparison['per_seed']={str(s):seed_macro[(CONDITIONS[0],s)]-seed_macro[(CONDITIONS[1],s)] for s in SEEDS}
    summary={'metrics':metrics,'comparison':comparison,'models':records,'ensemble':'mean three window logits before metrics','threshold':'logit >= 0'}
    save_json(ev/'summary.json',summary);csv_rows(ev/'metrics.csv',metrics);csv_rows(ev/'per_source_metrics.csv',per_source)
    csv_rows(ev/'per_source_differences.csv',differences)
    (out/'scores').mkdir(exist_ok=True);csv_rows(out/'scores/window_scores.csv',window_rows)
    return summary


def region_statistics(rows, arrays):
    result={}
    for split in ('train','validation'):
        ix=[i for i,r in enumerate(rows) if r['split']==split]
        comp=arrays['component_ids'][ix].cpu().numpy();active=(arrays['area'][ix]>0).cpu().numpy()
        cc=[];cells=[];sizes=[]
        for ids,mask in zip(comp.reshape(-1,256),active.reshape(-1,256)):
            _,counts=np.unique(ids[mask],return_counts=True);cc.append(len(counts));sizes.extend(counts.tolist());cells.append(int(mask.sum()))
        if max(sizes)>4:raise ValueError('COMPONENT_SIZE_LIMIT')
        result[split]={'frame_count':len(cc),'GLOBAL_COMPONENT':np.percentile(cc,[0,25,50,75,100]).tolist(),
                       'GLOBAL_CELL':np.percentile(cells,[0,25,50,75,100]).tolist(),
                       'component_size_histogram':{str(k):sizes.count(k) for k in range(1,5)}}
    return result


def visualizations(out, rows, arrays, stats, microbatch):
    """Two label-stratified ID-first windows, not selected by response or accuracy."""
    from PIL import Image
    chosen=[min((i for i,r in enumerate(rows) if r['split']=='validation' and r['role']==role),key=lambda i:rows[i]['window_id'])
            for role in ('real','fake')]
    batch=select(arrays,chosen);evidence={}
    for condition in CONDITIONS:
        values=[]
        for seed in SEEDS:
            cp=torch.load(out/'models'/f'{condition}__seed{seed}.pt',map_location='cpu',weights_only=False)
            model=GlobalContextModel(condition,stats).cuda();model.load_state_dict(cp['model']);model.eval()
            with torch.inference_mode():values.append(model(batch,True)['cell_logits'].cpu().numpy())
            del model
        evidence[condition]=np.mean(values,axis=0)
    folder=out/'evaluation/examples';folder.mkdir(parents=True,exist_ok=True);index=[]
    for j,i in enumerate(chosen):
        r=rows[i]
        for ti in (0,8):
            raw=r['sources'][ti]
            if sha256_file(raw['path'])!=raw['sha256']:raise ValueError('EXAMPLE_RAW_IDENTITY')
            with np.load(raw['path'],allow_pickle=False) as z:rgb=z['rgb'].copy()
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG');uri='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
            svg=['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="340">','<rect width="850" height="340" fill="white"/>',
                 f'<text x="8" y="20">{r["window_id"]}; frame {raw["frame_index"]}; PTS {raw["pts_s"]:.6f}s</text>']
            ids=batch['component_ids'][j,ti].cpu().numpy();active=(batch['area'][j,ti]>0).cpu().numpy()
            content=batch['content'][j,ti].cpu().numpy()
            for col,title in enumerate(('RGB','GLOBAL_COMPONENT','GLOBAL_CELL')):
                ox=8+col*280;oy=48
                svg.extend([f'<text x="{ox}" y="42">{title}</text>',f'<image x="{ox}" y="{oy}" width="256" height="256" href="{uri}"/>'])
                if col:
                    logits=evidence[title][j,ti]
                    for cell in np.flatnonzero(active):
                        yy,xx=np.where(content[cell]);x=ox+(cell%16)*16+int(xx.min());y=oy+(cell//16)*16+int(yy.min())
                        v=float(logits[cell]);s=max(-1.,min(1.,v/10))
                        color=f'rgb({int(255*max(s,0))},{int(100*(1-abs(s)))},{int(255*max(-s,0))})'
                        svg.append(f'<rect x="{x}" y="{y}" width="{int(xx.max()-xx.min()+1)}" height="{int(yy.max()-yy.min()+1)}" fill="{color}" opacity="0.55"><title>cell {cell}, logit {v:.6f}</title></rect>')
                        for neighbor,vertical in ((cell+1,True),(cell+16,False)):
                            if neighbor>=256 or (vertical and cell%16==15) or not active[neighbor]:continue
                            if title=='GLOBAL_COMPONENT' and ids[cell]==ids[neighbor]:continue
                            x0=ox+(cell%16)*16;y0=oy+(cell//16)*16
                            coords=f'{x0+16},{y0} {x0+16},{y0+16}' if vertical else f'{x0},{y0+16} {x0+16},{y0+16}'
                            svg.append(f'<polyline points="{coords}" stroke="white" stroke-width="1"/>')
            svg.extend(['<text x="8" y="328" font-size="12">Fixed logit color range [-10,10]: blue negative / red positive, clipped; white = region boundary. Weakly supervised response, not localization truth.</text>','</svg>'])
            name=f'{j}_{ti}.svg';(folder/name).write_text('\n'.join(svg))
            index.append({'window_id':r['window_id'],'source':r['source_id'],'role':r['role'],'source_frame':raw['frame_index'],'pts_s':raw['pts_s'],'file':name,'selection':'first window ID per role; target indices 0 and 8','color_range':[-10,10]})
    save_json(folder/'index.json',index)
    (folder/'index.html').write_text('<meta charset="utf-8"><h1>固定规则案例，统一logit色标；不是定位真值</h1>'+''.join(f'<p>{a["window_id"]} {a["pts_s"]}s</p><img src="{a["file"]}">' for a in index))


def report(out, summary, costs, region_stats):
    # Reuse the loss SVG generation below, with six complete saved histories.
    ev=out/'evaluation';colors=('#1565c0','#d32f2f','#388e3c','#7b1fa2','#ef6c00','#00838f')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="480">','<rect width="1100" height="480" fill="white"/>',
         '<text x="50" y="25">Weighted window BCE; final epoch 100</text>','<path d="M50 40V390H720" fill="none" stroke="black"/>']
    for i,r in enumerate(summary['models']):
        with (out/'models'/f"{r['condition']}__seed{r['seed']}.loss.csv").open() as f:ll=[float(a['loss']) for a in csv.DictReader(f)]
        points=' '.join(f'{50+j/99*670:.2f},{390-min(v,1)*330:.2f}' for j,v in enumerate(ll))
        svg.extend([f'<polyline points="{points}" fill="none" stroke="{colors[i]}"/>',f'<text x="740" y="{60+i*40}" font-size="13" fill="{colors[i]}">{r["condition"]} {r["seed"]}</text>'])
    for e in (1,25,50,75,100):svg.append(f'<text x="{50+(e-1)/99*670}" y="415">{e}</text>')
    for v in (0,.2,.4,.6,.8,1):svg.append(f'<text x="10" y="{390-v*330}">{v}</text>')
    svg.extend(['<text x="350" y="450">Epoch</text>','</svg>']);(ev/'train_loss.svg').write_text('\n'.join(svg))
    lines=['# V8 动态component与单格区域的全帧关系匹配对照','',
           '状态：COMPLETE，6/6正式模型，各100 epochs，使用最后epoch。两条件同源隔离、同718训练窗口/122 source、83验证窗口/14 source（42 real/41 fake），每窗16时刻。','',
           '研究问题：同样的三维、RGB、观测状态与时间路径下，动态成员分区是否优于singleton格区域？不是三维对二维或attention对无attention的比较。','',
           '## 实际计算路径','',
           '77维格输入＋既有MEAN_REPLICATED patch编码64维→141维cell encoder→content-aware固定格邻接消息→区域面积加权hidden/有效点加权XYZ/面积比例/成员数，共133维→Linear133→128+ReLU→单层4头同帧attention→residual+LayerNorm→原component_residual广播回cell→原多前驱关联、真实delta_t和cell GRU→cell head→面积LSE。旧component residual没有额外调用。','',
           'GLOBAL_COMPONENT只用缓存成员分区；GLOBAL_CELL只用active格singleton。两者参数、初始化、顺序、统计、分类损失和聚合一致；区域ID仅用于索引。真实内容缺失区域不删除，纯padding不参加。区域没有有效几何时XYZ为有限占位，状态仍在shared输入。','',
           f"参数量：每条件{summary['models'][0]['parameter_count']}；区域数[min,P25,median,P75,max]：{region_stats}。",'',
           '## 集成结果（先平均三seed logit）','',
           '| split | condition | macro AUROC | pooled AUROC | AP | P | R | F1 | ACC | TN/FP/FN/TP |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for m in summary['metrics']:
        if m['seed']!='MEAN_LOGIT':continue
        lines.append(f"| {m['split']} | {m['condition']} | "+' | '.join(f"{m[k]:.6f}" if m[k] is not None else 'NA' for k in ('source_macro_auroc','auroc','ap','precision','recall','f1','accuracy'))+f" | {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']} |")
    c=summary['comparison'];lines+=['',f"GLOBAL_COMPONENT−GLOBAL_CELL：{c['mean']}；source配对bootstrap 10000次/seed20260909，95% CI {c['ci95']}。",
           f"source方向{c['directions']}；逐seed差值{c['per_seed']}。逐seed AUROC均值不等于平均logit后的AUROC。",'',
           '## 成本与验证','',f'阶段成本：{costs}。',
           f"六模型实际训练累计{sum(r['training_elapsed_s'] for r in summary['models']):.3f}秒；最大allocated峰值{max(r['peak_allocated_gib'] for r in summary['models']):.3f}GiB。",
           '恢复checkpoint含模型/optimizer/epoch/完整loss/RNG/数据顺序生成器及输入配置身份；已完成合法模型与分数不重训。输入预载GPU，训练不运行前端，不读取V7数值资产。','',
           '## 判断与边界','',
           ('主点估计为正，但是否稳定需结合CI与seed/source方向；不仅凭点估计宣布动态分组有效。' if c['mean']>0 else '本轮主点估计未支持动态component优于singleton格；不据此宣称两者等价。'),
           '这是反复开发使用的窗口验证集，没有独立sealed test。训练拟合不能证明跨source泛化。区域属性与全帧attention相对旧模型同时变化，历史分数差不能归因于attention。',
           '本轮frame logit为明确面积LSE响应，监督和性能评价仍是窗口级；cell热图不是像素真值定位。没有新component永久轨迹，没有未来帧attention。',
           '已有全视频接口保留；本轮不将801窗口改成全视频训练，不处理272全视频，也不追加模型。',
           '历史干净参考：identityfix FULL/RGB_2D macro=0.785714/0.761905，局部几何MEAN_REPLICATED/DENSE_LOCAL=0.817460/0.738095；不是本轮匹配对照。','',
           '少量固定规则案例见evaluation/examples/index.html：按role取最早window ID，目标索引0与8；统一[-10,10]logit色标，白线表示区域边界，非逐图拉伸。图像来自已有V8 raw RGB，不重提特征。',
           '全部逐seed/逐source/窗口分数见evaluation/与scores/；正式checkpoint保留数据目录，不提交Git。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stage',choices=('smoke','all'),default='all');p.add_argument('--microbatch',type=int,default=8)
    args=p.parse_args();base,reference,out=map(lambda s:Path(s).resolve(),(args.baseline,args.reference,args.output))
    if base.name!='v8_full_coverage_observation_field_identityfix_v1' or reference.name!='v8_local_geometry_encoding_pilot_v1':raise ValueError('ONLY_CLEAN_V8_INPUTS')
    if out in (base,reference) or base in out.parents or reference in out.parents:raise ValueError('OUTPUT_NOT_INDEPENDENT')
    if args.microbatch not in (1,2,4,8):raise ValueError('MICROBATCH_MUST_DIVIDE_EFFECTIVE_BATCH')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'protocol.json').exists() and not args.resume:raise ValueError('RESUME_REQUIRED')
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    torch.set_num_threads(2)
    # Same native float32 / TF32 settings as the frozen local-geometry training.
    def term(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,term)
    def progress(stage,completed,total,**extra):
        save_json(out/'progress.json',{'status':'RUNNING','stage':stage,'completed':completed,'total':total,'pid':os.getpid(),'updated_unix':time.time(),**extra})
    start=time.monotonic();phase='prepare';costs=read_json(out/'stage_costs.json') if (out/'stage_costs.json').exists() else {}
    try:
        save_json(out/'final_status.json',{'status':'RUNNING','pid':os.getpid(),'started_unix':time.time()})
        ts=time.monotonic();rows,stats,input_id=prepare(reference,base,out,progress)
        costs['input_audit_cumulative_s']=costs.get('input_audit_cumulative_s',0)+time.monotonic()-ts
        config={'conditions':list(CONDITIONS),'seeds':list(SEEDS),'epochs':100,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'effective_batch':8,
                'microbatch':args.microbatch,'clip':1.,'patch_chunk':2048,'hidden':128,'region_node_dim':133,'heads':4,'attention_layers':1,'dropout':0.,
                'condition_difference':'membership partition only','geometry':'MEAN_REPLICATED; identical masks and pixel-weighted region XYZ',
                'standardization':'reuse frozen train-only local-geometry statistics','loss':'same source/class weighted window BCE',
                'threshold':0.,'bootstrap':{'replicates':10000,'seed':20260909,'unit':'source','paired':True},
                'code_sha256':{f:sha256_file(Path(__file__).parent/f) for f in ('model.py','run.py')},
                'reused_code_sha256':{str(f):sha256_file(f) for f in (Path(__file__).parents[1]/'local_geometry_pilot/model.py',Path(__file__).parents[1]/'local_geometry_pilot/data.py',Path(__file__).parents[1]/'full_coverage_observation_field/pipeline.py')},
                'torch':str(torch.__version__),'device':torch.cuda.get_device_name(),'randomness':{'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}}
        identity={'input_id':input_id,'config_id':digest(config),'statistics_sha256':sha256_file(reference/'standardizer.json')}
        protocol={**config,'identity':identity,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'base':str(base),'reference':str(reference)}
        if (out/'protocol.json').exists():
            if read_json(out/'protocol.json')['identity']!=identity:raise ValueError('PROTOCOL_RESUME_MISMATCH')
        else:save_json(out/'protocol.json',protocol)
        arrays,loading=load_inputs(rows,stats);costs['input_loading']=loading
        rs=region_statistics(rows,arrays);save_json(out/'region_statistics.json',rs)
        print('INPUTS',len(rows),'GPU',torch.cuda.get_device_name(),'resident_bytes',loading['resident_input_bytes'],flush=True)
        phase='smoke';sp=out/'smoke/summary.json';ts=time.monotonic()
        sr=read_json(sp) if sp.exists() else smoke(out,rows,arrays,stats,identity,args.microbatch)
        if sr['identity']!=identity:raise ValueError('SMOKE_IDENTITY')
        costs['smoke_cumulative_s']=costs.get('smoke_cumulative_s',0)+time.monotonic()-ts;save_json(out/'stage_costs.json',costs)
        print('SMOKE',sr,flush=True)
        if args.stage=='smoke':save_json(out/'final_status.json',{'status':'SMOKE_PASS','models':0});return
        phase='train';records=train(out,rows,arrays,stats,identity,args.microbatch,progress)
        phase='evaluate';ts=time.monotonic();summary=evaluate(out,rows,records);costs['evaluation_s']=time.monotonic()-ts
        phase='visualization';ts=time.monotonic();visualizations(out,rows,arrays,stats,args.microbatch);costs['visualization_s']=time.monotonic()-ts
        phase='report';costs['this_invocation_s']=time.monotonic()-start
        costs['free_disk_bytes']=os.statvfs(out).f_bavail*os.statvfs(out).f_frsize
        costs['new_storage_bytes']=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        report(out,summary,costs,rs);save_json(out/'stage_costs.json',costs)
        save_json(out/'final_status.json',{'status':'COMPLETE','models':6,'epoch_each':100,'report_complete':True,'input_id':input_id,'finished_unix':time.time()})
        save_json(out/'progress.json',{'status':'COMPLETE','stage':'report','completed':6,'total':6,'updated_unix':time.time()})
        print('COMPLETE',summary['comparison'],flush=True)
    except BaseException as exc:
        costs['last_failed_invocation_s']=time.monotonic()-start;save_json(out/'stage_costs.json',costs)
        save_json(out/'final_status.json',{'status':'STOPPED_SAFE' if isinstance(exc,InterruptedError) else 'FAILED','stage':phase,
                  'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'updated_unix':time.time()})
        raise


if __name__=='__main__':main()
