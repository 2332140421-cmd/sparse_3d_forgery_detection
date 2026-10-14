"""One optimizer step per source pair, exact parent-level normalization."""
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import read_frames, save, digest, gpu_batch
from .model import DenseRegions, CONDITIONS, normalized_lse

SEED=20260909


def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)


def parent_weights(row):
    ys=np.array([row['labels'][i] for c in row['clips'] for i in c['indices']]);counts=np.bincount(ys,minlength=2)
    present=int((counts>0).sum())
    return np.array([1/(present*n) if n else 0 for n in counts])


def pair_step(model,optimizer,out,pair):
    optimizer.zero_grad(set_to_none=True);details=[]
    for row in pair:
        weights=parent_weights(row)
        for c in row['clips']:
            cpu=read_frames(out,row,c['indices'],model.condition!='RGB_QUERY')
            batch=gpu_batch(cpu)
            # Clip starts deliberately reset memory and delta_t.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                frames,_=model.sequence(batch,checkpoint_frames=True)
                areas=batch['content'].sum((1,2,3)).float();clip=normalized_lse(frames,areas,0)
                lc=F.binary_cross_entropy_with_logits(clip,clip.new_tensor(float(c['clip_label'])))/len(row['clips'])
                y=torch.tensor([row['labels'][i] for i in c['indices']],device='cuda',dtype=torch.float32)
                w=torch.tensor(weights,device='cuda',dtype=torch.float32)[y.long()]
                lf=(F.binary_cross_entropy_with_logits(frames,y,reduction='none')*w).sum()
                loss=.5*(lc+.5*lf)
            loss.backward()
            details.append([float(loss.detach()),float(lc.detach()),float(lf.detach())])
            del batch,cpu,frames,clip,loss
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('NONFINITE_GRADIENT')
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    return np.sum(details,axis=0).tolist()


def initial_model(condition,stats):
    seed()
    if condition=='RGB_QUERY':model=DenseRegions(condition)
    else:model=DenseRegions(condition,stats['mean'],stats['scale'])
    return model.cuda().to(memory_format=torch.channels_last)


def train(out,rows,stats,protocol,progress):
    sources=sorted({r['source_id'] for r in rows if r['split']=='train'});records=[]
    (out/'models').mkdir(exist_ok=True)
    for condition in CONDITIONS:
        model=initial_model(condition,stats);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        from .data import read_json
        identity={'condition':condition,'protocol_hash':digest(protocol),'standardizer_hash':digest(stats),'implementation':read_json(out/'implementation.json'), 'parameters':sum(p.numel() for p in model.parameters())}
        path=out/'models'/f'{condition}.pt';history=[];epoch=0;elapsed=0.
        if path.exists():
            cp=torch.load(path,map_location='cpu',weights_only=False)
            if cp['identity']!=identity:raise ValueError('MODEL_RESUME_IDENTITY')
            model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);epoch=cp['epoch'];history=cp['history'];elapsed=cp['training_s']
            torch.set_rng_state(cp['rng_cpu']);torch.cuda.set_rng_state_all(cp['rng_cuda']);np.random.set_state(cp['rng_numpy']);random.setstate(cp['rng_python'])
        model.train();torch.cuda.reset_peak_memory_stats()
        for e in range(epoch,20):
            started=time.monotonic();order=np.random.default_rng(SEED+e).permutation(sources).tolist();losses=[]
            for j,source in enumerate(order):
                pair=sorted([r for r in rows if r['split']=='train' and r['source_id']==source],key=lambda r:r['role'])
                if [r['role'] for r in pair]!=['fake','real']:raise ValueError('SOURCE_PAIR')
                pair_start=time.monotonic();losses.append(pair_step(model,optimizer,out,pair))
                cost_path=out/'training_costs.json';cost=read_json(cost_path) if cost_path.exists() else {'actual_s':0.,'pair_steps_executed':0}
                cost['actual_s']+=time.monotonic()-pair_start;cost['pair_steps_executed']+=1;save(cost_path,cost)
                progress('train',e*len(sources)+j+1,20*len(sources),condition=condition,epoch=e+1,source=source,loss=losses[-1][0])
            duration=time.monotonic()-started;elapsed+=duration
            history.append({'epoch':e+1,'loss':float(np.mean(np.array(losses)[:,0])),'clip_bce_parent_sum':float(np.mean(np.array(losses)[:,1])),'frame_bce_parent_sum':float(np.mean(np.array(losses)[:,2])),'elapsed_s':duration})
            cp={'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':e+1,'history':history,'training_s':elapsed,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
                'source_order':order,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
            torch.save(cp,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path);save(out/'models'/f'{condition}_loss.json',history)
        records.append({'condition':condition,'seed':SEED,'epoch':20,'status':'COMPLETE','checkpoint':str(path),'identity':identity,'training_s':elapsed,
                        'peak_allocated_gib':cp['peak_allocated_gib'],'peak_reserved_gib':cp['peak_reserved_gib']})
        save(out/'models/fold_models.json',{'records':records});del model,optimizer;torch.cuda.empty_cache()
    return records


def smoke(out,rows,stats,progress):
    source=sorted({r['source_id'] for r in rows if r['split']=='train'})[0];pair=sorted([r for r in rows if r['source_id']==source],key=lambda r:r['role'])
    model=initial_model(CONDITIONS[0],stats);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();loss=pair_step(model,optimizer,out,pair);torch.cuda.synchronize();duration=time.monotonic()-started
    obj={'status':'PASS','source':source,'source_pair_s':duration,'loss':loss,'estimated_three_condition_training_s':duration*122*20*3,
         'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
         'formal_initialization_reset':True}
    from .data import read_json
    history=read_json(out/'smoke_history.json') if (out/'smoke_history.json').exists() else []
    history.append(obj);save(out/'smoke_history.json',history)
    save(out/'resource_smoke.json',obj);del model,optimizer;torch.cuda.empty_cache();return obj
