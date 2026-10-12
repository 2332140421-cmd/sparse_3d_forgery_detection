"""Video metrics and paired source bootstrap; only saved scores are consumed."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.component_global_context_pilot.model import CONDITIONS
from research_tools.v8.full_coverage_observation_field.pipeline import _metric
from research_tools.v8.local_geometry_pilot.runner import csv_rows

SEEDS = (20260909, 20260910, 20260911)


def paired_pooled(rows, a, b):
    sources = sorted({r['source_id'] for r in rows})
    y = np.array([r['label'] for r in rows])
    indices = [np.array([i for i,r in enumerate(rows) if r['source_id']==s]) for s in sources]
    def measures(ii, z):
        yy,zz=y[ii],z[ii]
        return np.array([roc_auc_score(yy,zz),average_precision_score(yy,zz),
                         f1_score(yy,zz>=0,zero_division=0),accuracy_score(yy,zz>=0)])
    rng=np.random.default_rng(20260909); draws=[]
    for _ in range(10000):
        ii=np.concatenate([indices[i] for i in rng.integers(0,len(sources),len(sources))])
        draws.append(measures(ii,a)-measures(ii,b))
    point=measures(np.arange(len(rows)),a)-measures(np.arange(len(rows)),b)
    ci=np.quantile(draws,[.025,.975],axis=0)
    return {k:{'difference':float(point[i]),'ci95':[float(ci[0,i]),float(ci[1,i])]} for i,k in enumerate(('pooled_auroc','ap','f1','accuracy'))}


def window_addendum(reference,out):
    path=out/'window_auxiliary_bootstrap.json'
    if path.exists(): return
    with (reference/'scores/window_scores.csv').open() as f:
        saved=[r for r in csv.DictReader(f) if r['split']=='validation' and r['seed']=='MEAN_LOGIT']
    ids=sorted({r['window_id'] for r in saved}); mapping={(r['condition'],r['window_id']):r for r in saved}
    rows=[{'source_id':mapping[(CONDITIONS[0],w)]['source'],'label':int(mapping[(CONDITIONS[0],w)]['label'])} for w in ids]
    if len(ids)!=83 or len({r['source_id'] for r in rows})!=14: raise ValueError('WINDOW_ADDENDUM_POPULATION')
    z=[np.array([float(mapping[(c,w)]['logit']) for w in ids]) for c in CONDITIONS]
    for w in ids:
        if any(mapping[(c,w)]['label']!=mapping[(CONDITIONS[0],w)]['label'] or mapping[(c,w)]['source']!=mapping[(CONDITIONS[0],w)]['source'] for c in CONDITIONS):
            raise ValueError('WINDOW_CONDITION_IDENTITIES')
    save_json(path,{'scope':'previous window experiment auxiliary supplement; original macro primary unchanged',
              'replicates':10000,'seed':20260909,'unit':'paired source; pooled metrics recomputed on all sampled windows',
              'windows':len(ids),'sources':14,'comparison':'GLOBAL_COMPONENT-GLOBAL_CELL',**paired_pooled(rows,*z)})


def evaluate(out,manifest,summaries,conditions=CONDITIONS):
    directory=out/'evaluation';directory.mkdir(exist_ok=True)
    metrics=[];scores=[];per_source=[];ensemble={}
    y=np.array([r['video_label'] for r in manifest]); rows=[{**r,'label':r['video_label']} for r in manifest]
    val=np.array([r['split']=='validation' for r in rows])
    for condition in conditions:
        zz=np.array([[s['video_logits'][f'{condition}__seed{seed}'] for s in summaries] for seed in SEEDS])
        if not np.isfinite(zz).all(): raise ValueError('NONFINITE_VIDEO_SCORE')
        ensemble[condition]=zz.mean(0)
        for seed,z in list(zip(SEEDS,zz))+[('MEAN_VIDEO_LOGIT',zz.mean(0))]:
            scores.extend({'condition':condition,'seed':seed,'video_id':r['video_id'],'source':r['source_id'],
                           'split':r['split'],'role':r['role'],'label':r['video_label'],'logit':float(v)} for r,v in zip(rows,z))
            for split in ('train','validation'):
                ii=np.flatnonzero([r['split']==split for r in rows]); aucs=[]
                for source in sorted({rows[i]['source_id'] for i in ii}):
                    jj=np.array([i for i in ii if rows[i]['source_id']==source])
                    if len(np.unique(y[jj]))!=2: raise ValueError('VIDEO_SINGLE_CLASS_SOURCE')
                    auc=float(roc_auc_score(y[jj],z[jj]));aucs.append(auc)
                    per_source.append({'split':split,'condition':condition,'seed':seed,'source':source,'auroc':auc,
                                       'n_real':int((y[jj]==0).sum()),'n_fake':int((y[jj]==1).sum())})
                metrics.append({'split':split,'condition':condition,'seed':seed,'videos':len(ii),'sources':len(aucs),
                                'source_pair_ordering_mean':float(np.mean(aucs)),**_metric(y[ii],z[ii])})
    comparison=(paired_pooled([r for r in rows if r['split']=='validation'],ensemble[conditions[0]][val],ensemble[conditions[1]][val])
                if len(conditions)==2 else None)
    directions=[]
    for source in sorted({r['source_id'] for r in rows if r['split']=='validation'}) if len(conditions)==2 else []:
        ss=[r for r in per_source if r['split']=='validation' and r['source']==source and r['seed']=='MEAN_VIDEO_LOGIT']
        a={r['condition']:r['auroc'] for r in ss}
        directions.append({'source':source,'difference':a[conditions[0]]-a[conditions[1]],**a})
    result={'status':'COMPLETE','primary':'pooled video AUROC; frozen window models, not full-video training',
            'ensemble':'each seed full-video area LSE then mean final video logits','threshold':0,
            'bootstrap':{'replicates':10000,'seed':20260909,'paired_unit':'source'},
            'metrics':metrics,'paired_comparison':comparison,'source_directions':{
                name:sum((r['difference']>0 if name=='positive' else r['difference']<0 if name=='negative' else r['difference']==0) for r in directions)
                for name in ('positive','tie','negative')}}
    csv_rows(directory/'metrics.csv',metrics);csv_rows(directory/'video_scores.csv',scores)
    csv_rows(directory/'per_source_metrics.csv',per_source)
    if directions:csv_rows(directory/'per_source_differences.csv',directions)
    save_json(directory/'summary.json',result)
    return result
