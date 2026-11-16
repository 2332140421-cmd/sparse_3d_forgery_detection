"""Seven shared frozen grid-C frames and two fixed temporal-order readouts.

CLI caller only. Reuses Local.encode/grouping/frozen_pool/CurrentReadout; creates
no trainable modules beyond loading the existing frozen readout, and no labels
enter feature extraction. Geometry follows the original per-frame MoGe settings.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch
from .data import geometry_shape, save
from .experiment import git, state_hash
from .set_fit_check import build, SOURCE
from .grid_localization import grouping, frozen_pool, CurrentReadout
from scripts.prepare_v8_assets import digest
from scripts.synthetic_temporal_pair import OUT as INPUT, rgb_hash

ROOT=SOURCE.parent
BASE=ROOT/'pvst_rgb_partition_anakin_dev_v1'
GRID=ROOT/'pvst_slab_grid_localization_v1'
OUT=ROOT/'pvst_synthetic_temporal_readout_v1'


def read(p):return json.loads(p.read_text())


def aligned_history(entries, sequence):
    """Exact integer keys, no tracked-object claim; diagnostic CPU float64 math."""
    maps=[{tuple(k):i for i,k in enumerate(entries[s]['keys'].tolist())} for s in sequence]
    shared=sorted(set.intersection(*(set(m) for m in maps)))
    xs=[entries[s]['features'][torch.tensor([m[k] for k in shared],dtype=torch.long)].double().numpy() for s,m in zip(sequence,maps)]
    norms=[np.linalg.norm(x,axis=1) for x in xs]
    keep=np.logical_and.reduce([np.isfinite(n)&(n>0) for n in norms])
    x=[a[keep]/n[keep,None] for a,n in zip(xs,norms)]
    distances=[float(np.mean(1-np.sum(a*b,axis=1))) for a,b in zip(x[:-1],x[1:])] if keep.any() else None
    coverage=[]
    for s,m in zip(sequence,maps):
        selected=[m[k] for k,ok in zip(shared,keep) if ok]
        coverage.append({'source_frame':s,'all_keys':len(m),'shared_key_fraction':len(selected)/len(m) if m else None,'pixel_coverage':int(entries[s]['counts'][selected].sum())/int(entries[s]['counts'].sum()) if int(entries[s]['counts'].sum()) else None})
    return {'sequence':sequence,'common_before_zero_filter':len(shared),'excluded_zero_norm_keys':int((~keep).sum()),'common_keys':int(keep.sum()),'coverage':coverage,'distances':distances,'total':sum(distances) if distances else None}


def progress(stage,**kw):
    save(OUT/'status.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),**kw});print(stage,kw,flush=True)


@torch.no_grad()
def run():
    start=time.monotonic();torch.set_num_threads(4)
    if OUT.exists():
        old=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if old.get('status')=='COMPLETE':print('Existing COMPLETE; verify identity before reuse');return
        raise RuntimeError('Existing partial preserved; no automatic retry')
    OUT.mkdir();progress('identity')
    m=read(INPUT/'manifest.json');assert read(INPUT/'validation.json')['status']=='PASS'
    c=read(BASE/'config.json');provider=read(SOURCE/'config.json')
    assert c['scale']==1.9888148307800293 and c['h']==.125 and c['d0']==0
    assert m['main']=={'t':6,'H':3} and m['control']=={'t':3,'H':3}
    original_identity=read(BASE/'identity.json');grid_identity=read(GRID/'identity.json')
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt';rw=GRID/'SLAB_GRID_CURRENT/final.pt'
    assert digest(weight)==grid_identity['weight_sha256']
    for f,k in [('config.json','config_sha256'),('manifest.json','manifest_sha256')]:assert digest(BASE/f)==grid_identity[k]
    cp=torch.load(weight,map_location='cpu',weights_only=True)
    assert (cp['condition'],cp['epoch'],cp['updates'])==('RGB_DEPTH_PARTITION',12,576) and cp['identity']==original_identity['identity']
    rcp=torch.load(rw,map_location='cpu',weights_only=True)
    assert (rcp['condition'],rcp['epoch'],rcp['updates'])==('SLAB_GRID_CURRENT',12,576)
    assert rcp['identity']==read(GRID/'formal_identity.json')
    identity={'execution_head':git('rev-parse','HEAD'),'input_manifest_sha256':digest(INPUT/'manifest.json'),'encoder_weight':str(weight),'encoder_sha256':digest(weight),'readout_weight':str(rw),'readout_sha256':digest(rw),'manifest_sha256':digest(BASE/'manifest.json'),'scale':c['scale'],'h':c['h'],'provider':{k:provider[k] for k in ['moge_revision','moge_weights','moge_code','utils3d_code']},'moge_weight_sha256':digest(Path(provider['moge_weights'])),'unique_frames':list(range(7)),'feature_dtype':'BF16 encode/pool arithmetic, stored C FP32; readout FP32','distance_dtype':'CPU float64 cosine reductions only'}
    save(OUT/'identity.json',identity)
    rgbs=[]
    for i in range(7):
        a=np.array(Image.open(INPUT/f'ordered/rgb/{i:03d}.png').convert('RGB'))
        assert rgb_hash(a)==m['mapping']['ordered'][i]['rgb_sha256'];rgbs.append(a)
    # These synthetic PNGs have no previous compatible MoGe artifact. Never
    # borrow geometry from another source or version, nor use synthetic alpha.
    assert all(a.shape==(360,640,3) for a in rgbs)
    sys.path[:0]=[provider['moge_code'],provider['utils3d_code']]
    from moge.model.v2 import MoGeModel
    moge=MoGeModel.from_pretrained(provider['moge_weights']).cuda().eval()
    hw,tokens,internal=geometry_shape(rgbs[0].shape[:2]);assert hw==(360,640)
    geom=[];geometry_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i,a in enumerate(rgbs):
        progress('geometry',frame=i)
        res=moge.infer(torch.from_numpy(a).permute(2,0,1).cuda().float()/255,num_tokens=tokens,force_projection=True,apply_mask=False,use_fp16=True)
        q=res['points'].float().cpu();v=res['mask'].cpu().bool()&torch.isfinite(q).all(-1)
        assert q.shape==(360,640,3)
        q=torch.where(v[...,None],q,0)
        torch.save({'xyz_provider':q,'valid':v,'source_frame':i,'rgb_sha256':rgb_hash(a)},OUT/f'geometry_{i:02d}.pt')
        geom.append((q/c['scale'],v));del res
    torch.cuda.synchronize();geometry_seconds=time.monotonic()-geometry_start
    del moge;torch.cuda.empty_cache()
    progress('frozen_model')
    teacher=build(c,OUT,rgb_only=True).eval();teacher.load_state_dict(cp['state_dict'],strict=True);teacher.requires_grad_(False);del cp
    before=state_hash(teacher.state_dict());entries={};extract_start=time.monotonic()
    for i,(a,(xyz,valid)) in enumerate(zip(rgbs,geom)):
        progress('feature',frame=i)
        xyz=xyz.cuda();valid=valid.cuda();points,groups,L=grouping(xyz,valid)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            local,_=teacher.local.encode(torch.from_numpy(a),torch.zeros(valid.shape))
            C=frozen_pool(teacher.pool,local,points,[groups[1]],valid.shape)[0]
        ids,index=groups[1];ny=(360+L-1)//L;nx=(640+L-1)//L
        _,slab=torch.unique(torch.floor(xyz[...,2][valid]/.125).long(),sorted=True,return_inverse=True)
        unique,check=torch.unique(slab*(ny*nx)+(points//640//L)*nx+(points%640//L),sorted=True,return_inverse=True)
        assert torch.equal(index,check)
        keys=torch.stack((ids,unique.remainder(ny*nx)//nx,unique.remainder(nx)),-1)
        counts=torch.bincount(index,minlength=len(ids));assert int(counts.sum())==int(valid.sum())
        assert C.shape==(len(ids),512) and torch.isfinite(C).all()
        entries[i]={'features':C.cpu(),'ids':ids.cpu(),'keys':keys.cpu(),'counts':counts.cpu(),'source_frame':i,'rgb_sha256':rgb_hash(a),'L':L,'valid_pixels':int(valid.sum()),'total_pixels':valid.numel()}
        torch.save(entries[i],OUT/f'features_{i:02d}.pt');del local,C,xyz,valid
    torch.cuda.synchronize();extract_seconds=time.monotonic()-extract_start
    assert state_hash(teacher.state_dict())==before;del teacher;torch.cuda.empty_cache()
    readout=CurrentReadout().cuda().eval();readout.load_state_dict(rcp['state_dict'],strict=True);readout.requires_grad_(False);del rcp
    rows=[]
    for t in [3,6]:
        sources=[[m['mapping'][role][i]['source_layer'] for i in range(t-3,t+1)] for role in ['ordered','perturbed']]
        current=[entries[s[-1]] for s in sources];assert sources[0][-1]==sources[1][-1]
        logits=[readout(e['features'].cuda(),e['ids'].cuda()).cpu() for e in current]
        assert all(torch.isfinite(z).all() for z in logits)
        histories=[aligned_history(entries,s) for s in sources]
        row={'t':t,'H':3,'current_C_max_abs':float((current[0]['features']-current[1]['features']).abs().max()),'current_logit_max_abs':float((logits[0]-logits[1]).abs().max()),'current_units':len(logits[0]),'ordered_history':histories[0],'perturbed_history':histories[1],'history_delta':histories[1]['total']-histories[0]['total'] if histories[0]['total'] is not None and histories[1]['total'] is not None else None}
        assert row['current_C_max_abs']==0 and row['current_logit_max_abs']==0
        if t==3:assert row['history_delta']==0
        rows.append(row)
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'rows':rows,'unique_feature_frames':7,'new_moge_frames':7,'geometry_seconds':geometry_seconds,'extract_seconds':extract_seconds,'wall_seconds':time.monotonic()-start,'features':[{k:v for k,v in e.items() if k not in ['features','ids','keys','counts']}|{'units':len(e['ids'])} for e in entries.values()],'frozen_teacher_state_unchanged':True,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    save(OUT/'summary.json',summary);save(OUT/'status.json',{'status':'COMPLETE','exit_code':0});print(json.dumps(rows),flush=True)


if __name__=='__main__':
    try:run()
    except Exception:
        if OUT.exists():
            (OUT/'error.log').write_text(traceback.format_exc());save(OUT/'status.json',{'status':'FAILED','exit_code':1})
        raise
