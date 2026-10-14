"""Full causal validation; saved scalar responses and preselected diagnostics."""
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from .data import read_frames,save,read_json,sha256_file,gpu_batch
from .model import CONDITIONS,normalized_lse
from .train import initial_model


def metric(y,z):
    y=np.asarray(y,bool);z=np.asarray(z);p=z>=0
    tn=int((~p&~y).sum());fp=int((p&~y).sum());fn=int((~p&y).sum());tp=int((p&y).sum())
    return {'auroc':float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None,'ap':float(average_precision_score(y,z)) if y.any() else None,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'accuracy':float((p==y).mean()),'tn':tn,'fp':fp,'fn':fn,'tp':tp}


def picture(out,condition,row,fi,batch,pixels,assignment,z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb=batch['rgb'][0].cpu().permute(1,2,0).numpy();content=batch['content'][0,0].cpu().numpy()
    fig,axs=plt.subplots(1,3,figsize=(12,4));axs[0].imshow(rgb)
    regions=assignment.argmax(-1).cpu().reshape(256,256).numpy();axs[1].imshow(np.ma.masked_where(~content,regions),vmin=0,vmax=63,cmap='tab20',interpolation='nearest')
    im=axs[2].imshow(np.ma.masked_where(~content,pixels.cpu().numpy()),vmin=-8,vmax=8,cmap='coolwarm',interpolation='nearest');fig.colorbar(im,ax=axs[2],ticks=[-8,0,8],label='Absolute pixel logit')
    for a in axs:a.axis('off')
    axs[0].set_title('RGB');axs[1].set_title('Region identity, not authenticity');axs[2].set_title(f'Frame logit {float(z):.3f}')
    fig.suptitle(f'{condition} {row["source_id"]}/{row["role"]} frame{fi} PTS{row["pts_s"][fi]:.6f}');fig.tight_layout()
    p=out/'diagnostics'/condition;p.mkdir(parents=True,exist_ok=True);fig.savefig(p/f'{row["video_id"]}_{fi}.png',dpi=110);plt.close(fig)


def evaluate(out,rows,stats,records,progress):
    val=[r for r in rows if r['split']=='validation'];results={};predictions={};cost={}
    for condition in CONDITIONS:
        started=time.monotonic();model=initial_model(condition,stats)
        record=next(r for r in records if r['condition']==condition);cp=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        if cp['epoch']!=20 or cp['identity']!=record['identity']:raise ValueError('FINAL_MODEL_IDENTITY')
        model.load_state_dict(cp['model'],strict=True);model.eval();prediction={}
        region_path=out/'evaluation'/f'{condition}_regions.json'
        region_stats=read_json(region_path) if region_path.exists() else []
        folder=out/'responses'/condition;folder.mkdir(parents=True,exist_ok=True)
        for n,row in enumerate(val):
            path=folder/f'{row["video_id"]}.npz';identity={'video_sha256':row['video_sha256'],'checkpoint_sha256':sha256_file(record['checkpoint'])}
            if path.exists() and path.with_suffix('.json').exists():
                if read_json(path.with_suffix('.json'))!=identity:raise ValueError('RESPONSE_IDENTITY')
                with np.load(path) as z:prediction[row['video_id']]={k:z[k].copy() for k in z.files}
                continue
            hidden=None;frames=[];areas=[];previous_regions=None;changes=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,row['actual_frames'],32):
                    indices=list(range(start,min(start+32,row['actual_frames'])));b=gpu_batch(read_frames(out,row,indices,condition!='RGB_QUERY'))
                    b['dt'][0]=0. if start==0 else row['pts_s'][start]-row['pts_s'][start-1]
                    if hidden is None:hidden=b['rgb'].new_zeros(64,256)
                    for j,fi in enumerate(indices):
                        z,hidden,pixels,assignment=model.frame(b['rgb'][j:j+1],b['visual'][j:j+1],b['content'][j:j+1],b.get('xyz',None)[j:j+1] if 'xyz' in b else None,b.get('valid',None)[j:j+1] if 'valid' in b else None,b['dt'][j],hidden,True)
                        frames.append(float(z));areas.append(int(b['content'][j].sum()))
                        current_regions=assignment.argmax(-1)
                        active=b['content'][j].flatten()
                        if previous_regions is not None:changes.append(float((current_regions[active]!=previous_regions[active]).float().mean()))
                        previous_regions=current_regions
                        if fi in row.get('diagnostic_indices',[]):
                            picture(out,condition,row,fi,{k:v[j:j+1] for k,v in b.items()},pixels,assignment,z)
                            # QUERY columns sum to 1 by construction: their column
                            # masses cannot diagnose overlap/collapse. Normalize
                            # across queries ONLY for this descriptive statistic.
                            descriptive=assignment.float()/assignment.float().sum(-1,keepdim=True).clamp_min(1e-20)
                            masses=descriptive.sum(0);masses=masses/masses.sum()
                            occupancy=torch.bincount(current_regions[active],minlength=64).float()/active.sum()
                            columns=torch.nn.functional.normalize(assignment.float().T,dim=-1)
                            cos=columns@columns.T
                            region_stats.append({'video_id':row['video_id'],'frame':fi,'mass_min':float(masses.min()),'mass_max':float(masses.max()),
                                'effective_regions':float(torch.exp(-(masses*masses.clamp_min(1e-20).log()).sum())),
                                'largest_argmax_region_fraction':float(occupancy.max()),'mean_query_overlap_cosine':float((cos.sum()-cos.diagonal().sum())/(64*63)),
                                'mass_definition':'pixel-row-normalized weights for description only; not new model normalization'})
                    del b
            frame=np.array(frames);area=np.array(areas);video=float(np.logaddexp.reduce(frame+np.log(area))-np.log(area.sum()))
            region_stats.append({'video_id':row['video_id'],'kind':'continuous_query_index_changes_not_physical_tracking',
                'mean_adjacent_argmax_change_fraction':float(np.mean(changes)) if changes else None,'frames':len(frames)})
            prediction[row['video_id']]={'video_logit':np.array(video),'frame_logits':frame,'pts_s':np.array(row['pts_s']),'labels':np.array(row['labels'])}
            save(region_path,region_stats)
            np.savez(path,**prediction[row['video_id']]);save(path.with_suffix('.json'),identity);progress('evaluate',n+1,len(val),condition=condition)
        y=[r['role']=='fake' for r in val];z=[float(prediction[r['video_id']]['video_logit']) for r in val]
        yy=np.concatenate([prediction[r['video_id']]['labels'] for r in val]);zz=np.concatenate([prediction[r['video_id']]['frame_logits'] for r in val])
        eligible=[r for r in val if r['role']=='fake' and len(set(r['labels']))==2]
        results[condition]={'video':metric(y,z),'frame':metric(yy,zz),
            'fake_video_macro_frame_ap':float(np.mean([average_precision_score(r['labels'],prediction[r['video_id']]['frame_logits']) for r in eligible])),
            'eligible_fake_videos':len(eligible),'real_video_macro_frame_fpr':float(np.mean([np.mean(prediction[r['video_id']]['frame_logits']>=0) for r in val if r['role']=='real']))}
        save(region_path,region_stats);predictions[condition]=prediction;cost[condition]=time.monotonic()-started
        del model;torch.cuda.empty_cache()
    sources=sorted({r['source_id'] for r in val});rng=np.random.default_rng(20260909);sampling=rng.integers(0,len(sources),(10000,len(sources)));paired=[]
    for a,b in [('XYZ_QUERY','RGB_QUERY'),('XYZ_PARTITION','XYZ_QUERY')]:
        boot=[]
        for sample in sampling:
            chosen=[r for i in sample for r in val if r['source_id']==sources[i]];y=[r['role']=='fake' for r in chosen]
            boot.append(roc_auc_score(y,[float(predictions[a][r['video_id']]['video_logit']) for r in chosen])-roc_auc_score(y,[float(predictions[b][r['video_id']]['video_logit']) for r in chosen]))
        paired.append({'comparison':a+' - '+b,'difference':results[a]['video']['auroc']-results[b]['video']['auroc'],'ci95':np.quantile(boot,[.025,.975]).tolist(),'replicates':10000,'seed':20260909})
    save(out/'evaluation/summary.json',results);save(out/'evaluation/paired.json',paired);save(out/'evaluation/costs.json',cost)
    return results,paired
