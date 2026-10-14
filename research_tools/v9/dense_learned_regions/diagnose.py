"""Finite frozen diagnostics: training clips, reset memory, zero decoder context."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import ROOT, read_json, read_frames, gpu_batch, save, sha256_file, digest
from .model import CONDITIONS, normalized_lse
from .train import initial_model
from .evaluate import metric

BASE=ROOT/'derived/v9_dense_learned_regions_pilot_v1'
OUT=ROOT/'derived/v9_minimal_mechanism_diagnosis_v1'


def csv_save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_model(condition,stats,records):
    record=next(r for r in records if r['condition']==condition)
    cp=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['epoch']==20 and cp['identity']==record['identity'] and cp['identity']['condition']==condition
    # Original json_safe serializes Python True as integer 1. Restore only this
    # documented metadata field for the in-memory training hash; numeric stats
    # must match the checkpoint's actual float32 buffers exactly.
    assert stats['unique_frames_only'] in (True,1)
    assert cp['identity']['standardizer_hash']==digest({**stats,'unique_frames_only':True})
    if condition!='RGB_QUERY':
        for k in ('mean','scale'):assert torch.equal(cp['model'][k].flatten(),torch.tensor(stats[k],dtype=torch.float32))
    assert [h['epoch'] for h in cp['history']]==list(range(1,21))
    model=initial_model(condition,stats);model.load_state_dict(cp['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,cp


def frame_metrics(rows,predictions):
    y=np.concatenate([predictions[r['video_id']]['labels'] for r in rows]);z=np.concatenate([predictions[r['video_id']]['frame_logits'] for r in rows])
    fake=[r for r in rows if r['role']=='fake'];eligible=[r for r in fake if len(np.unique(predictions[r['video_id']]['labels']))==2]
    real=[r for r in rows if r['role']=='real']
    return {'pooled':metric(y,z),'real_parent_macro_fpr':float(np.mean([np.mean(predictions[r['video_id']]['frame_logits']>=0) for r in real])),
        'fake_parent_macro_ap':float(np.mean([average_precision_score(predictions[r['video_id']]['labels'],predictions[r['video_id']]['frame_logits']) for r in eligible])) if eligible else None,
        'fake_parent_macro_auroc':float(np.mean([roc_auc_score(predictions[r['video_id']]['labels'],predictions[r['video_id']]['frame_logits']) for r in eligible])) if eligible else None,
        'eligible_two_class_fake_parents':len(eligible),'excluded_single_class_fake_parents':len(fake)-len(eligible)}


def loss_summary(cp):
    history=cp['history'];rows=[]
    for h in history:
        clip=.5*h['clip_bce_parent_sum'];frame=.5*h['frame_bce_parent_sum']
        assert abs(h['loss']-(clip+.5*frame))<1e-6
        rows.append({'epoch':h['epoch'],'total':h['loss'],'clip_bce_parent_mean':clip,'frame_bce_parent_mean':frame})
    return {'first':rows[0],'last':rows[-1],'last5_total_range':[min(h['total'] for h in rows[-5:]),max(h['total'] for h in rows[-5:])],
            'epoch16_to20_total_change':rows[-1]['total']-rows[-5]['total'],'history':rows}


def training_clips(out,rows,stats,records,progress):
    train=[r for r in rows if r['split']=='train'];results={};losses={};per_parent=[]
    for condition in CONDITIONS:
        started=time.monotonic();model,cp=load_model(condition,stats,records);losses[condition]=loss_summary(cp);predictions={};cy=[];cz=[]
        folder=out/'training_clips'/condition;folder.mkdir(parents=True,exist_ok=True)
        for n,row in enumerate(train):
            path=folder/f'{row["video_id"]}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
            else:
                zs=[];ys=[];pts=[];indices=[];clip_index=[];clipz=[];clipy=[]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for j,c in enumerate(row['clips']):
                        batch=gpu_batch(read_frames(BASE,row,c['indices'],condition!='RGB_QUERY'))
                        assert float(batch['dt'][0])==0
                        frame,_=model.sequence(batch)
                        area=batch['content'].sum((1,2,3)).float();clip=normalized_lse(frame,area,0)
                        labels=[row['labels'][i] for i in c['indices']];assert c['clip_label']==int(any(labels))
                        zs.extend(frame.cpu().numpy().tolist());ys.extend(labels);pts.extend(row['pts_s'][i] for i in c['indices'])
                        indices.extend(c['indices']);clip_index.extend([j]*len(labels));clipz.append(float(clip));clipy.append(c['clip_label'])
                pred={'frame_logits':np.array(zs),'labels':np.array(ys),'pts_s':np.array(pts),'frame_indices':np.array(indices),
                      'clip_indices':np.array(clip_index),'clip_logits':np.array(clipz),'clip_labels':np.array(clipy)}
                np.savez(path,**pred)
            expected=[i for c in row['clips'] for i in c['indices']]
            assert np.array_equal(pred['frame_indices'],expected)
            assert np.array_equal(pred['pts_s'],[row['pts_s'][i] for i in expected])
            assert np.array_equal(pred['labels'],[row['labels'][i] for i in expected])
            assert np.isfinite(pred['frame_logits']).all() and np.isfinite(pred['clip_logits']).all()
            predictions[row['video_id']]=pred;cy.extend(pred['clip_labels']);cz.extend(pred['clip_logits'])
            eligible=len(np.unique(pred['labels']))==2
            per_parent.append({'condition':condition,'video_id':row['video_id'],'source':row['source_id'],'role':row['role'],
                'frame_uses':len(expected),'positive_frame_uses':int(pred['labels'].sum()),'two_class':eligible,
                'frame_ap':float(average_precision_score(pred['labels'],pred['frame_logits'])) if eligible else None,
                'frame_auroc':float(roc_auc_score(pred['labels'],pred['frame_logits'])) if eligible else None})
            progress('training_clip_inference',n+1,len(train),condition=condition)
        assert len(cy)==976 and sum(len(v['labels']) for v in predictions.values())==31232
        results[condition]={'clip':metric(cy,cz),'frame':frame_metrics(train,predictions),'inference_s':time.monotonic()-started}
        save(out/'training_metrics.json',results);del model;torch.cuda.empty_cache()
    save(out/'loss_summary.json',losses);csv_save(out/'training_parent_temporal_metrics.csv',per_parent)
    return results,losses


def intervene_block(model,batch):
    """One pixel encoding; reset-state decoder versus exact zero context at concat."""
    f=model.encode(batch['rgb'],batch['visual'],batch['content'],batch['xyz'],batch['valid'])
    h,_=model.pool(f,batch['content']);hidden=batch['rgb'].new_zeros(64,256);states=[]
    assert float(batch['dt'][0])==0
    for j in range(len(f)):
        hidden=model.gru(torch.cat((h[j],batch['dt'][j].expand(64,1)),-1),hidden);states.append(hidden)
    reset=model.decode(f,torch.stack(states),batch['content'])
    # Do not zero hidden and then apply a biased projection: zero the final context.
    head_input=torch.cat((f,torch.zeros_like(f)),-1)
    pixels=model.head(head_input)[:,:,0].float()
    zero=normalized_lse(pixels,batch['content'].flatten(1).float(),1)
    return reset,zero,f


def ranking_changes(rows,official,other):
    a=np.array([float(official[r['video_id']]['video_logit']) for r in rows]);b=np.array([float(other[r['video_id']]['video_logit']) for r in rows])
    af=np.concatenate([official[r['video_id']]['frame_logits'] for r in rows]);bf=np.concatenate([other[r['video_id']]['frame_logits'] for r in rows])
    changed=0;pairs=0
    for i,r in enumerate(rows):
        if r['role']!='fake':continue
        for j,s in enumerate(rows):
            if s['role']!='real':continue
            pairs+=1;changed+=int(np.sign(a[i]-a[j])!=np.sign(b[i]-b[j]))
    return {'video_logit_mean_abs_change':float(np.mean(abs(b-a))),'video_logit_max_abs_change':float(np.max(abs(b-a))),
        'video_spearman':float(spearmanr(a,b).statistic),'cross_label_pair_order_changes':changed,'cross_label_pairs':pairs,
        'frame_mean_abs_change':float(np.mean(abs(bf-af))),'frame_max_abs_change':float(np.max(abs(bf-af))),
        'frame_spearman':float(spearmanr(af,bf).statistic),'video_threshold_prediction_changes':int(np.sum((a>=0)!=(b>=0)))}


def validation(out,rows,stats,records,progress):
    val=[r for r in rows if r['split']=='validation'];model,_=load_model('XYZ_PARTITION',stats,records)
    predictions={m:{} for m in ('CONTINUOUS_SAVED','RESET32','ZERO_CONTEXT')};checks=[];started=time.monotonic()
    for n,row in enumerate(val):
        old=BASE/'responses/XYZ_PARTITION'/f'{row["video_id"]}.npz'
        with np.load(old,allow_pickle=False) as z:predictions['CONTINUOUS_SAVED'][row['video_id']]={k:z[k].copy() for k in z.files}
        paths={m:out/'validation'/m/f'{row["video_id"]}.npz' for m in ('RESET32','ZERO_CONTEXT')}
        if all(p.exists() for p in paths.values()):
            for m,p in paths.items():
                with np.load(p,allow_pickle=False) as z:predictions[m][row['video_id']]={k:z[k].copy() for k in z.files}
        else:
            results={m:[] for m in paths};areas=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,row['actual_frames'],32):
                    indices=list(range(start,min(start+32,row['actual_frames'])))
                    batch=gpu_batch(read_frames(BASE,row,indices,True))
                    # Reader dt=0 only at this block's first frame; all other deltas native.
                    expected=np.r_[0,np.diff([row['pts_s'][i] for i in indices])]
                    assert np.allclose(batch['dt'].cpu().numpy(),expected,rtol=1e-6,atol=1e-8)
                    reset,zero,f=intervene_block(model,batch)
                    if n==0 and start==0:
                        saved=predictions['CONTINUOUS_SAVED'][row['video_id']]['frame_logits'][:3]
                        error=float(np.max(abs(reset[:3].cpu().numpy()-saved)))
                        assert error<.002,('UNINTERVENED_PREFIX_REPRODUCTION',error)
                        captured=[]
                        handle=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
                        p=model.head(torch.cat((f[:1],torch.zeros_like(f[:1])),-1));handle.remove()
                        assert torch.equal(captured[0][...,:256],f[:1]) and not captured[0][...,256:].any()
                        checks.append({'prefix_saved_frame_max_error':error,'tolerance':.002,'head_context_exact_zero':True,'head_pixel_feature_unchanged':True})
                    results['RESET32'].extend(reset.cpu().numpy().tolist());results['ZERO_CONTEXT'].extend(zero.cpu().numpy().tolist())
                    areas.extend(batch['content'].sum((1,2,3)).cpu().numpy().tolist())
            for m,path in paths.items():
                frames=np.array(results[m]);assert len(frames)==row['actual_frames'] and np.isfinite(frames).all()
                video=float(np.logaddexp.reduce(frames+np.log(areas))-np.log(np.sum(areas)))
                pred={'frame_logits':frames,'video_logit':np.array(video),'pts_s':np.array(row['pts_s']),'labels':np.array(row['labels'])}
                path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**pred);predictions[m][row['video_id']]=pred
        for m in predictions:
            pred=predictions[m][row['video_id']]
            assert np.array_equal(pred['pts_s'],row['pts_s']) and np.array_equal(pred['labels'],row['labels'])
        progress('validation_interventions',n+1,len(val),condition='XYZ_PARTITION')
    results={}
    for mode,pred in predictions.items():
        assert sum(len(v['frame_logits']) for v in pred.values())==27505
        results[mode]={'video':metric([r['role']=='fake' for r in val],[float(pred[r['video_id']]['video_logit']) for r in val]),'frame':frame_metrics(val,pred)}
        if mode!='CONTINUOUS_SAVED':results[mode]['changes_from_official']=ranking_changes(val,predictions['CONTINUOUS_SAVED'],pred)
    save(out/'validation_metrics.json',results);save(out/'checks.json',checks)
    save(out/'validation_cost.json',{'shared_two_intervention_inference_s':time.monotonic()-started,'official_continuous':'saved responses, no full continuous rerun'})
    return results


def report(out,training,losses,validation_results,cost):
    text='# V9 最小机制诊断\n\n冻结推理完成，无训练、前端重提、阈值变化或正式结果改写。\n\n'
    text+='## 训练片段拟合\n\n976训练clip（不是完整训练视频），31232 clip×frame使用，31144去重帧，重复88次；fake上下文122片段仍为负。\n\n'
    text+='|条件|clip AUROC|clip AP|TN/FP/FN/TP|frame AP|frame AUROC|frame P|frame Recall|real parent FPR|fake parent macro AP/AUROC|\n|---|---:|---:|---|---:|---:|---:|---:|---:|---|\n'
    for c,r in training.items():
        v=r['clip'];f=r['frame'];p=f['pooled']
        text+=f'|{c}|{v["auroc"]:.6f}|{v["ap"]:.6f}|{v["tn"]}/{v["fp"]}/{v["fn"]}/{v["tp"]}|{p["ap"]:.6f}|{p["auroc"]:.6f}|{p["precision"]:.6f}|{p["recall"]:.6f}|{f["real_parent_macro_fpr"]:.6f}|{f["fake_parent_macro_ap"]:.6f}/{f["fake_parent_macro_auroc"]:.6f}|\n'
    text+='\n## loss记录（统一为parent平均BCE）\n\n|条件|首/末总loss|首/末clip BCE|首/末frame BCE|后5轮总loss范围|epoch16→20变化|\n|---|---|---|---|---|---|\n'
    for c,r in losses.items():
        a,b=r['first'],r['last'];text+=f'|{c}|{a["total"]:.6f}/{b["total"]:.6f}|{a["clip_bce_parent_mean"]:.6f}/{b["clip_bce_parent_mean"]:.6f}|{a["frame_bce_parent_mean"]:.6f}/{b["frame_bce_parent_mean"]:.6f}|{r["last5_total_range"]}|{r["epoch16_to20_total_change"]:+.6f}|\n'
    text+='\n原记录clip/frame字段是两parent之和，此处除2；总loss=clip BCE+0.5 frame BCE。fake parent两类齐全数量见training_metrics.json，不将重复clip帧当独立原生帧。训练clip与完整验证video不能直接相减称泛化差距。\n\n'
    text+='## 完整验证诊断（28视频/27505帧，每项相同集合）\n\n|模式|video AUROC|video AP|TN/FP/FN/TP|frame AP|frame AUROC|real-video FPR|fake-video macro frame AP|\n|---|---:|---:|---|---:|---:|---:|---:|\n'
    for m,r in validation_results.items():
        v=r['video'];f=r['frame'];p=f['pooled'];text+=f'|{m}|{v["auroc"]:.6f}|{v["ap"]:.6f}|{v["tn"]}/{v["fp"]}/{v["fn"]}/{v["tp"]}|{p["ap"]:.6f}|{p["auroc"]:.6f}|{f["real_parent_macro_fpr"]:.6f}|{f["fake_parent_macro_ap"]:.6f}|\n'
    text+='\n## 分数与排序变化\n\n'
    for m in ('RESET32','ZERO_CONTEXT'):text+=m+': `'+str(validation_results[m]['changes_from_official'])+'`\n\n'
    text+='## 判断与边界\n\n待按实际指标补充判断。置零是在最终concat处，不是置零hidden后经过bias投影；这是推理干预，不是独立训练的像素基线或正式贡献消融。重置改善不独立证明长记忆有害。没有新增bootstrap、可视化或外部测试。\n\n'
    text+='## 成本\n\n'+str(cost)+'\n'
    (out/'report.md').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();records=read_json(BASE/'models/fold_models.json')['records'];stats=read_json(BASE/'standardizer.json');rows=read_json(BASE/'video_manifest.json')['rows']
    files=[BASE/n for n in ('protocol.json','standardizer.json','video_manifest.json','models/fold_models.json','evaluation/summary.json')]
    files.extend(Path(r['checkpoint']) for r in records);files.extend((BASE/'responses').rglob('*.*'))
    hashes={str(p):sha256_file(p) for p in files if p.is_file()}
    for rec in records:
        for row in [r for r in rows if r['split']=='validation']:
            meta=read_json(BASE/'responses'/rec['condition']/f'{row["video_id"]}.json')
            assert meta=={'checkpoint_sha256':hashes[rec['checkpoint']],'video_sha256':row['video_sha256']}
    identity={'original_hashes':hashes,'diagnostic_code_sha256':sha256_file(__file__),'threshold':0,'no_training':True}
    if (out/'identity.json').exists():assert read_json(out/'identity.json')==identity,'DIAGNOSTIC_RESUME_IDENTITY'
    save(out/'identity.json',identity);torch.cuda.reset_peak_memory_stats()
    def progress(stage,completed,total,**kw):
        s={'status':'RUNNING','stage':stage,'completed':completed,'total':total,**kw};save(out/'progress.json',s)
        if completed%20==0 or completed==total:print(s,flush=True)
    try:
        training,losses=training_clips(out,rows,stats,records,progress);val=validation(out,rows,stats,records,progress)
        assert all(sha256_file(p)==h for p,h in hashes.items()),'ORIGINAL_ASSET_MODIFIED'
        cost={'wall_s':time.monotonic()-started,'training_clip_inference_s':{k:v['inference_s'] for k,v in training.items()},
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
              'storage_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
        save(out/'costs.json',cost);report(out,training,losses,val,cost)
        save(out/'final_status.json',{'status':'COMPUTE_COMPLETE_PENDING_REVIEW','training_clips_per_condition':976,'validation_frames_per_mode':27505,'original_hashes_unchanged':True})
    except BaseException as e:
        save(out/'final_status.json',{'status':'FAILED','error':repr(e),'elapsed_s':time.monotonic()-started});raise


if __name__=='__main__':main()
