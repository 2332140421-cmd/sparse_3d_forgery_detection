"""One matched RGB-D pilot, independent prepare/check/fit/train/evaluate/report CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .data import Video,frames,geometry,save
from .model import RGBDSlices,backbone
from research_tools.depth_slice.data import prepare as verify_registered
from research_tools.depth_slice.model import terms
from research_tools.depth_slice.experiment import seed_all,aggregate,checkpoint_state
from research_tools.v8.spatial_localization.metrics import classification,spatial_frame,history_csv
from scripts.prepare_v8_assets import digest

REPO=Path(__file__).resolve().parents[2]
CONDITIONS=['RGBD_LOCAL','RGBD_SLICE_RELATION']
NORM_CONDITION='RGBD_SLICE_RELATION_NORM'


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make(config,out,condition):
    seed_all(config['seed']);m=RGBDSlices(backbone(config),condition in [CONDITIONS[1],NORM_CONDITION],context_norm=condition==NORM_CONDITION)
    initialization_root=Path(config.get('initialization_root',out))
    common=torch.load(initialization_root/'common_initial.pt',map_location='cpu',weights_only=True);m.local.load_state_dict(common,strict=True)
    if state_hash(m.local.state_dict())!=json.loads((initialization_root/'initialization.json').read_text())['common_sha256']:raise ValueError('COMMON_INITIALIZATION_CHANGED')
    return m.cuda()


def optimizer(model,c):
    enc=list(model.local.encoder.parameters());enc_ids={id(p) for p in enc}
    return torch.optim.AdamW([{'params':enc,'lr':c['encoder_lr']},{'params':[p for p in model.parameters() if id(p) not in enc_ids],'lr':c['new_lr']}],weight_decay=.01)


def forward(model,fs):
    with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):return model(fs)


def grad_info(model):
    groups={'rgb_stem':'local.encoder.downsample_layers.','depth_stem':'local.encoder.downsample_layers_e.','rgbd_interaction':'local.encoder.stages.','decoder':'local.refine','pixel':'local.final','frame':'local.frame'}
    if model.relation:groups.update(pool='pool.',time='time.',depth='depth.',pixel_context='pixel_context.',frame_context='frame_context.')
    if getattr(model,'context_norm',None) is not None:groups['context_norm']='context_norm.'
    info={}
    for name,prefix in groups.items():
        gs=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        info[name]={'present':bool(gs),'finite':all(bool(torch.isfinite(g).all()) for g in gs),'norm':float(torch.sqrt(sum((g.float().square().sum() for g in gs),torch.zeros((),device='cuda'))))}
    return info


def update(model,opt,fs,diagnose=False,context_diagnostics=False):
    model.train();opt.zero_grad(set_to_none=True);started=time.monotonic();parts=[]
    # Same one-clip mean and one AdamW step. A has no temporal graph, so native
    # frame-wise backward is exactly its clip loss gradient (not extra updates).
    if not model.relation:
        for f in fs:
            pred=forward(model,[f])[0];t=terms(pred,f['gt']);loss=t.sum()/len(fs)
            if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
            loss.backward();parts.append(t.detach().cpu().numpy());del pred,t,loss
    else:
        pred=forward(model,fs);t=torch.stack([terms(p,f['gt']) for p,f in zip(pred,fs)]).mean(0);loss=t.sum()
        if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
        loss.backward();parts=[t.detach().cpu().numpy()];del pred,t,loss
    info=grad_info(model) if diagnose else None
    if context_diagnostics:
        names=['frame_context','pixel_context','context_norm']
        norms=torch.stack([torch.sqrt(sum((p.grad.detach().float().square().sum() for p in getattr(model,name).parameters() if p.grad is not None),torch.zeros((),device='cuda'))) for name in names])
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    before=model.local.final[-1].weight.detach().clone() if diagnose else None
    opt.step();torch.cuda.synchronize()
    if diagnose and torch.equal(before,model.local.final[-1].weight):raise ValueError('NO_PIXEL_PARAMETER_UPDATE')
    extra={'preclip_context_gradients':dict(zip(names,norms.cpu().tolist())),'clip_coefficient':float((1./(norm+1e-6)).clamp(max=1.))} if context_diagnostics else {}
    return {**extra,'terms':np.mean(parts,axis=0).tolist(),'seconds':time.monotonic()-started,'gradient_norm':float(norm),'gradients':info,'optimizer_states':len(opt.state),
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


class ContextScaleRecorder:
    """Detached scalar reductions, one host transfer at the end of each update.

    Hooks use the existing forward tensors. No RNG calls, per-slice CPU sync,
    retained activations, extra forward or change to the backward graph.
    """
    def __init__(self,model):
        self.active=False;self.rows=[];self.handles=[]
        def norm_hook(module,args,output):
            if self.active:
                a=args[0].detach().float();b=output.detach().float()
                self.rows.append(torch.stack((a.square().sum(),a.new_tensor(a.numel()),b.square().sum(),b.new_tensor(b.numel()),a.new_zeros(()),a.new_zeros(()),a.new_zeros(()))))
        def pixel_hook(module,args,output):
            if self.active:
                x=output.detach().float();z=x.new_zeros(())
                self.rows.append(torch.stack((z,z,z,z,x.square().sum(),x.new_tensor(x.numel()),z)))
        def frame_hook(module,args,output):
            if self.active:
                x=output.detach().float();z=x.new_zeros(())
                self.rows.append(torch.stack((z,z,z,z,z,z,x.abs().sum())))
        for module,hook in [(model.context_norm,norm_hook),(model.pixel_context,pixel_hook),(model.frame_context,frame_hook)]:self.handles.append(module.register_forward_hook(hook))
    def begin(self,frames):self.rows=[];self.frames=frames;self.active=True
    def finish(self):
        self.active=False
        totals=torch.stack(self.rows).sum(0).cpu().tolist() if self.rows else [0.]*7
        self.rows=[]
        rms=lambda a,n:(a/n)**.5 if n else None
        return {'raw_token_rms':rms(*totals[:2]),'normalized_token_rms':rms(*totals[2:4]),'pixel_projection_rms':rms(*totals[4:6]),'frame_residual_mean_abs':totals[6]/self.frames}
    def close(self):
        for handle in self.handles:handle.remove()


def load_clip(row,clip,c):
    v=Video(row,c)
    try:return frames(v,clip['start'],clip['end'])
    finally:v.close()


def checks(c,out,manifest,progress):
    path=out/'checks.json'
    if path.exists():
        if json.loads(path.read_text())['status']!='PASS':raise ValueError('EXISTING_FAILED_CHECK')
        return
    fake=next(r for r in manifest['rows'] if r['split']=='train' and r['role']=='fake');clip=next(x for x in manifest['clips'] if x['video_id']==fake['video_id']);fs=load_clip(fake,clip,c)
    if not any(f['gt'].any() for f in fs):raise ValueError('SMOKE_NEEDS_NONEMPTY_GT')
    a=make(c,out,CONDITIONS[0]).eval()
    sizes={CONDITIONS[0]:{'parameters':sum(p.numel() for p in a.parameters()),'trainable':sum(p.numel() for p in a.parameters() if p.requires_grad)}}
    with torch.no_grad():pa=forward(a,fs)
    reference=[{'pixel':p['pixel'].float().cpu(),'frame':p['frame'].float().cpu()} for p in pa];del a,pa;torch.cuda.empty_cache()
    b=make(c,out,CONDITIONS[1]).eval()
    sizes[CONDITIONS[1]]={'parameters':sum(p.numel() for p in b.parameters()),'trainable':sum(p.numel() for p in b.parameters() if p.requires_grad)};save(out/'model_sizes.json',sizes)
    with torch.no_grad():pb=forward(b,fs)
    errors={'pixel':0.,'frame':0.}
    for p,r in zip(pb,reference):
        for key in errors:
            v=p[key].float().cpu();torch.testing.assert_close(v,r[key],atol=.002,rtol=.002);errors[key]=max(errors[key],float((v-r[key]).abs().max()))
    del pb,reference
    bn={k:v.clone() for k,v in b.local.encoder.named_buffers()};opt=optimizer(b,c);records=[];torch.cuda.reset_peak_memory_stats()
    for i in range(2):
        progress('B_two_update_check',step=i+1);r=update(b,opt,fs,diagnose=True);records.append(r)
        if not all(g['present'] and g['finite'] for g in r['gradients'].values()):raise ValueError('GRADIENT_MISSING_OR_NONFINITE')
    for name in records[0]['gradients']:
        if not any(s['gradients'][name]['norm']>0 for s in records):raise ValueError('NO_NONZERO_BRANCH_GRADIENT:'+name)
    for k,v in b.local.encoder.named_buffers():
        if not torch.equal(v,bn[k]):raise ValueError('PRETRAINED_BN_CHANGED')
    save(path,{'status':'PASS','zero_residual_max_error':errors,'zero_tolerance_BF16':{'atol':.002,'rtol':.002},'video':fake['video_id'],'range':[clip['start'],clip['end']],'steps':records})
    del b,opt,fs;torch.cuda.empty_cache()


def preview(path,image,gt,z):
    prob=1/(1+np.exp(-np.clip(z,-60,60)));heat=cv2.applyColorMap((prob*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
    gt_rgb=np.repeat((gt.astype(np.uint8)*255)[...,None],3,-1);binary=np.repeat(((z>=0)*255).astype(np.uint8)[...,None],3,-1)
    path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),np.concatenate([image,gt_rgb,heat,binary],1)[:,:,::-1])


@torch.no_grad()
def evaluate(model,rows,c,progress,*,clip_list=None,samples=None,pixel_ap=True):
    model.eval();records=[];started=time.monotonic()
    for r in rows:
        if clip_list is not None and not any(x['video_id']==r['video_id'] for x in clip_list):continue
        progress('evaluate',video=r['video_id'],scope='training_monitor_clips' if clip_list else 'complete_dev')
        n=r['actual_frames'];v=Video(r,c);a={k:[] for k in ['dice','iou','pixel_ap','empty_fpr','gt_recall','pred_area','loss','scores','labels','slices']}
        ranges=[(s,min(s+24,n),max(0,n-24) if s+24>=n else s) for s in range(0,n,24)] if clip_list is None else [(x['start'],x['end'],x['start']) for x in clip_list if x['video_id']==r['video_id']]
        pts=np.array([p['pts_s'] for p in r['pts']]);chosen={int(np.argmin(abs(pts-(pts[0]+x*(pts[-1]-pts[0]))))) for x in [.25,.5,.75]} if clip_list is None else {s+(e-s)//2 for s,e,_ in ranges}
        for start,end,lo in ranges:
            fs=frames(v,lo,end);pred=forward(model,fs)
            for i in range(start,end):
                p=pred[i-lo];gt=v.gt(i);z=p['pixel'].float().cpu().numpy();m=spatial_frame(z>=0,gt)
                a['scores'].append(float(p['frame']));a['labels'].append(bool(gt.any()));a['slices'].append(p['slices']);a['pred_area'].append(float((z>=0).mean()));a['loss'].append(float(terms(p,torch.from_numpy(gt)).sum()))
                if gt.any():
                    for k in ['dice','iou']:a[k].append(m[k])
                    a['gt_recall'].append(float(((z>=0)&gt).sum()/gt.sum()))
                    if pixel_ap:a['pixel_ap'].append(float(average_precision_score(gt.reshape(-1),z.reshape(-1))))
                else:a['empty_fpr'].append(m['empty_fpr'])
                if samples and i in chosen and r['video_id'] in c['sample_video_ids']:preview(samples/f'{r["video_id"]}_{i}.jpg',np.array(v.rgb[i]),gt,z)
            del fs,pred
        v.close();record={k:r[k] for k in ['video_id','source_id','pair_id','split','role','generator']}
        for k in ['dice','iou','pixel_ap','empty_fpr','gt_recall','pred_area','loss']:record[k]=float(np.mean(a[k])) if a[k] else None
        for k in ['scores','labels','slices']:record[k]=a[k]
        record.update(logit=float(np.logaddexp.reduce(a['scores'])-np.log(len(a['scores']))),frames=len(a['scores']));records.append(record)
    if samples:
        samples.mkdir(parents=True,exist_ok=True);(samples/'index.html').write_text('<meta charset="utf-8"><p>固定身份和PTS：RGB | GT编辑区 | 概率0–1 | logit≥0</p>'+''.join(f'<p>{p.name}</p><img width="100%" src="{p.name}">' for p in sorted(samples.glob('*.jpg'))))
    return {'metrics':aggregate(records),'records':records,'seconds':time.monotonic()-started,'scope':'exact backpropagated fixed monitor clips' if clip_list else 'complete dev; nonoverlap24 windows, tail left context to24, each native frame once; causal window resets',
            'pixel_ap_scope':'positive-GT-frame AP mean, then fake-video macro; not AP@IoU'}


def fitting(c,out,m,progress):
    folder=out/'fit';folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    pair=next(r['pair_id'] for r in m['rows'] if r['split']=='train');rr=[r for r in m['rows'] if r['split']=='train' and r['pair_id']==pair]
    cc=[next(x for x in m['clips'] if x['video_id']==r['video_id']) for r in rr]
    model=make(c,out,CONDITIONS[0]);opt=optimizer(model,c);fs=[load_clip(r,clip,c) for r,clip in zip(rr,cc)]
    history=[];started=time.monotonic();passed=False
    initial=evaluate(model,rr,c,progress,clip_list=cc,samples=folder/'initial_samples',pixel_ap=False);save(folder/'initial.json',initial)
    for i in range(1,151):
        if time.monotonic()-started>=2700:break
        progress('local_fit_check',update=i,pair=pair);record=update(model,opt,fs[(i-1)%2]);record['update']=i;history.append(record)
        if i%10==0 or i==150 or time.monotonic()-started>=2700:
            result=evaluate(model,rr,c,progress,clip_list=cc,samples=folder/'samples' if i%25==0 or i==150 else None,pixel_ap=False)
            record['monitor']=result['metrics'];save(folder/'history.json',history)
            fake=result['metrics'];passed=fake['dice']>=.8 and fake['real_fpr']<=.05
            print('FIT_CHECK '+json.dumps(record),flush=True)
            if passed:break
    final=evaluate(model,rr,c,progress,clip_list=cc,samples=folder/'samples',pixel_ap=False)
    # A clear but sub-goal fit is reported separately; threshold frozen before run.
    clear=final['metrics']['dice']>=c['fit_clear_dice'] and final['metrics']['GT_recall']>=c['fit_clear_recall'] and final['metrics']['real_fpr']<=.05
    result={'status':'PASS' if passed or clear else 'FIT_BLOCKED','goal_reached':passed,'clear_fit':clear,'updates':len(history),'seconds':time.monotonic()-started,'pair':pair,'clips':cc,'final':final,'history':history,'weights_discarded':True,'initial':initial['metrics']}
    save(folder/'result.json',result);history_csv(folder/'history.csv',[{**{k:v for k,v in row.items() if k not in ['monitor','gradients']},**{k:row.get('monitor',{}).get(k) for k in ['dice','GT_recall','real_fpr']}} for row in history])
    del model,opt,fs;torch.cuda.empty_cache();return result


def train(condition,c,out,m,identity,progress):
    folder=out/condition;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    model=make(c,out,condition);opt=optimizer(model,c);rows=m['rows'];dev=[r for r in rows if r['split']=='dev'];trainrows=[r for r in rows if r['split']=='train'];monitor_pair=trainrows[0]['pair_id'];monitor_rows=[r for r in trainrows if r['pair_id']==monitor_pair];monitor_clips=[next(x for x in m['clips'] if x['video_id']==r['video_id']) for r in monitor_rows]
    bn_hash=state_hash(dict(model.local.encoder.named_buffers()));history=[];best=-1.;best_epoch=0;bad=0;begin=1;updates=0
    if (folder/'last.pt').exists():
        s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if s['identity']!=identity:raise ValueError('RESUME_IDENTITY_CHANGED')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);history=s['history'];best=s['best'];best_epoch=s['best_epoch'];bad=s['bad'];begin=s['epoch']+1;updates=history[-1]['updates']
        random.setstate(s['rng']['python']);np.random.set_state(s['rng']['numpy']);torch.set_rng_state(s['rng']['torch']);torch.cuda.set_rng_state_all(s['rng']['cuda'])
    started=time.monotonic();prior_seconds=history[-1]['cumulative_s'] if history else 0.
    scales=ContextScaleRecorder(model) if condition==NORM_CONDITION else None
    if not (folder/'epoch0_dev.json').exists():save(folder/'epoch0_dev.json',evaluate(model,dev,c,progress))
    for epoch in range(begin,31):
        ep=time.monotonic();order=list(m['clips']);random.Random(c['seed']+epoch).shuffle(order);online=[];execution_records=[];torch.cuda.reset_peak_memory_stats()
        for ci,clip in enumerate(order):
            progress('train',condition=condition,epoch=epoch,clip=ci,total=len(order),updates=updates)
            row=next(r for r in trainrows if r['video_id']==clip['video_id']);fs=load_clip(row,clip,c)
            if scales:scales.begin(len(fs))
            record=update(model,opt,fs,context_diagnostics=scales is not None);online.append(record['terms']);updates+=1
            if scales:
                record.update(scales.finish());execution_records.append(record)
                with (folder/'update_history.jsonl').open('a') as log:log.write(json.dumps({'epoch':epoch,'update':updates,'video_id':clip['video_id'],'range':[clip['start'],clip['end']],**record})+'\n')
            del fs
        dm=evaluate(model,dev,c,progress);mm=evaluate(model,monitor_rows,c,progress,clip_list=monitor_clips,pixel_ap=False)
        improved=dm['metrics']['dice']>best
        if improved:best=dm['metrics']['dice'];best_epoch=epoch;bad=0
        else:bad+=1
        if state_hash(dict(model.local.encoder.named_buffers()))!=bn_hash:raise ValueError('ENCODER_BN_BUFFERS_CHANGED')
        parts=np.mean(online,axis=0);r={'epoch':epoch,'updates':updates,'loss':float(parts.sum()),'frame_BCE':float(parts[0]),'pixel_BCE':float(parts[1]),'dice_loss':float(parts[2]),'dev':dm['metrics'],'train_monitor_clips':mm['metrics'],'best_epoch':best_epoch,'bad':bad,'epoch_s':time.monotonic()-ep,'cumulative_s':prior_seconds+time.monotonic()-started,'lr':[g['lr'] for g in opt.param_groups],'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        if scales:
            keys=['raw_token_rms','normalized_token_rms','pixel_projection_rms','frame_residual_mean_abs','gradient_norm','clip_coefficient']
            r['context_diagnostics']={k:float(np.mean([x[k] for x in execution_records if x[k] is not None])) if any(x[k] is not None for x in execution_records) else None for k in keys}
            r['context_diagnostics'].update({name+'_preclip_gradient':float(np.mean([x['preclip_context_gradients'][name] for x in execution_records])) for name in ['frame_context','pixel_context','context_norm']})
            r['update_seconds']=sum(x['seconds'] for x in execution_records);r['epoch_evaluation_seconds']=dm['seconds']+mm['seconds']
        history.append(r);save(folder/'history.json',history);history_csv(folder/'history.csv',history);save(folder/f'epoch{epoch:02d}_dev.json',dm)
        s=checkpoint_state(model,opt,epoch,best_epoch,best,bad,history,identity);s['config']=c;torch.save(s,folder/'last.pt')
        if improved:torch.save(s,folder/'best.pt')
        print('EPOCH_COMPLETE '+condition+' '+json.dumps(r),flush=True)
        if epoch>=10 and bad>=8:break
    final_eval_started=time.monotonic()
    s=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model'])
    bd=evaluate(model,dev,c,progress,samples=folder/'samples');bm=evaluate(model,monitor_rows,c,progress,clip_list=monitor_clips,samples=folder/'train_monitor_samples',pixel_ap=False)
    save(folder/'best_dev.json',bd);save(folder/'best_train_monitor.json',bm)
    summary={'status':'COMPLETE','condition':condition,'best_epoch':best_epoch,'last_epoch':history[-1]['epoch'],'updates':updates,'best_dev':bd['metrics'],'best_train_monitor_clips':bm['metrics'],'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'seconds':prior_seconds+time.monotonic()-started,'peak_allocated':max(r['peak_allocated'] for r in history),'peak_reserved':max(r['peak_reserved'] for r in history),'peak_rss':max(r['peak_rss'] for r in history)}
    if scales:
        s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model'])
        ld=evaluate(model,dev,c,progress,samples=folder/'last_samples');lm=evaluate(model,monitor_rows,c,progress,clip_list=monitor_clips,pixel_ap=False)
        save(folder/'last_dev.json',ld);save(folder/'last_train_monitor.json',lm)
        summary.update(last_dev=ld['metrics'],last_train_monitor_clips=lm['metrics'],context_norm_eps=model.context_norm.eps,final_evaluation_seconds=time.monotonic()-final_eval_started,training_update_seconds=sum(x['update_seconds'] for x in history),epoch_evaluation_seconds=sum(x['epoch_evaluation_seconds'] for x in history),seconds=prior_seconds+time.monotonic()-started)
        summary['peak_allocated']=max(summary['peak_allocated'],torch.cuda.max_memory_allocated());summary['peak_reserved']=max(summary['peak_reserved'],torch.cuda.max_memory_reserved());summary['peak_rss']=max(summary['peak_rss'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        scales.close()
    save(folder/'summary.json',summary);del model,opt;torch.cuda.empty_cache();return summary


def report(c,out,m,status,summaries,error=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for condition in CONDITIONS:
        folder=out/condition
        if not (folder/'history.json').exists():continue
        history=json.loads((folder/'history.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,3));epochs=[r['epoch'] for r in history]
        ax[0].plot(epochs,[r['loss'] for r in history]);ax[0].set_title('Online clip loss')
        for split in ['dev','train_monitor_clips']:ax[1].plot(epochs,[r[split]['dice'] for r in history],label=split)
        ax[1].legend();fig.tight_layout();fig.savefig(folder/'curves.png');plt.close(fig)
    data={}
    for split in ['train','dev']:
        rr=[r for r in m['rows'] if r['split']==split];ss=[m['label_statistics'][r['video_id']] for r in rr]
        data[split]={'pairs':len(rr)//2,'videos':len(rr),'frames':sum(s['frames'] for s in ss),'positive_frames':sum(s['positive_frames'] for s in ss),'all_nonempty_fake':sum(s['all_nonempty'] for s in ss),'mixed_fake':sum(s['mixed'] for s in ss)}
    diff={};paired=[]
    if len(summaries)==2:
        a,b=summaries
        diff={k:b['best_dev'][k]-a['best_dev'][k] for k in ['dice','iou','pixel_ap','real_fpr','GT_recall']}
        aa=json.loads((out/CONDITIONS[0]/'best_dev.json').read_text())['records'];bb=json.loads((out/CONDITIONS[1]/'best_dev.json').read_text())['records']
        for x in aa:
            if x['role']!='fake':continue
            y=next(v for v in bb if v['video_id']==x['video_id']);paired.append({'video_id':x['video_id'],'pair_id':x['pair_id'],'generator':x['generator'],'A_Dice':x['dice'],'B_Dice':y['dice'],'difference':y['dice']-x['dice']})
    result={'status':status,'data':data,'summaries':summaries,'B_minus_A':diff,'paired_fake':paired,'error':error,'storage_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())};save(out/'result_summary.json',result)
    lines=['# 预训练RGB-D编码与深度切片关系匹配实验',f'状态：{status}。两个条件均使用深度，非纯二维/三维对照。',
           '数据：'+json.dumps(data,ensure_ascii=False),'官方原版DFormer-S RGB-D ImageNet预训练，保留多尺度64/128/256/512、stride4/8/16/32，编码器1e-5微调，BN统计固定且仿射可训练；多尺度局部融合256，逐级上采样/跳连、原生RGB细节卷积及独立帧头。共同初始化精确复制，B额外4×128池化和两组512维8head显式时间→深度关系，像素/帧增量投影零初始化。',
           '官方加载记录：'+str(out/'official_load.json')+'；全部保留编码器strict加载；忽略分类pred及官方分割本就删除的末端仅分类深度更新，已对照证明RGB多尺度输出一致。',
           '新MoGe-2：完整宽高比输入长边≤768、不放大小视频；num_tokens=round(input_h*input_w/14²)，实际内部网格与输出记录在每视频complete.json。旧256场不进入模型。原生插值回映不是独立测量。encoder_depth采用官方Greys_r灰度处理及(.48,.28)标准化；slice_depth始终Z/固定s，h.125,d0=0，非米。',
           '片段清单严格复用上轮首/中/尾24帧，每片段全帧等权损失并只更新一次；A逐帧梯度累积，B完整因果关系反向。frame BCE+pixel BCE+softDice等权，空GT沿用epsilon1。最多30轮、最少10轮、开发Dice耐心8。阈值logit≥0，视频frame log-mean-exp温度1。',
           '训练监测仅固定第一个训练pair的真实反向首段，不将未训练时段冒充拟合。开发完整视频24帧因果窗口，尾段补左上下文、每帧一次、边界重置；不声称无限历史/精确时间边界。pixel AP为正GT帧AP再fake视频宏平均，不是AP@IoU。',
           '必要检查：'+str(out/'checks.json')+'；独立拟合诊断：'+str(out/'fit/result.json')+'（检查权重丢弃）。',
           '|条件|best/结束epoch|Dice|IoU|pixel AP|GT召回|real像素FPR|frame AUROC/AP|video AUROC/AP|','|---|---|---|---|---|---|---|---|---|']
    for s in summaries:
        v=s['best_dev'];lines.append('|'+ '|'.join(map(str,[s['condition'],str(s['best_epoch'])+'/'+str(s['last_epoch']),v['dice'],v['iou'],v['pixel_ap'],v['GT_recall'],v['real_fpr'],str(v['frame']['auroc'])+'/'+str(v['frame']['ap']),str(v['video']['auroc'])+'/'+str(v['video']['ap'])]))+'|')
        lines.append('该条件全部同best指标及成本：'+json.dumps(s,ensure_ascii=False))
        epoch0=json.loads((out/s['condition']/'epoch0_dev.json').read_text())['metrics'];lines.append('相对epoch0的开发Dice变化：'+str(v['dice']-epoch0['dice'])+'；训练监测Dice：'+str(s['best_train_monitor_clips']['dice'])+'。')
    lines += ['B−A：'+json.dumps(diff,ensure_ascii=False),'逐fake视频：'+json.dumps(paired,ensure_ascii=False),
              '定位错误证据：分别检查编辑区召回（漏覆盖）、real像素FPR（背景误报）及固定样例。低误报不能单独证明定位成功；本报告不凭数值虚构肉眼边界结论。共同样例身份/PTS/概率色阶固定，不按预测挑选。',
              '限制：相机运动与逐帧几何可比性未解决；单seed小队列不能证明三维增益，或把切片关系收益外推到其他数据。开发fake全程编辑，无法检验复杂时间切换。',
              '完整产物：`'+str(out)+'`；计算与Git发布状态分开。'+('错误：'+str(error) if error else '')]
    text='\n\n'.join(lines)+'\n';(out/'report.md').write_text(text);(REPO/'docs/pvst/rgbd_slice_relations_v1.md').write_text(text)
    page=REPO/'docs/pvst/current_status.md';old=page.read_text().split('<!-- rgbd-slice-pilot -->')[0];page.write_text(old+'<!-- rgbd-slice-pilot -->\n\n## RGB-D切片匹配实验\n\n'+status+'；见[报告](rgbd_slice_relations_v1.md)。两个条件均使用深度。旧结果保留。\n')


def publish(out):
    try:
        git('add','docs/pvst/rgbd_slice_relations_v1.md','docs/pvst/current_status.md');git('commit','-m','Report matched pretrained RGB-D slice experiment')
        head=git('rev-parse','HEAD');subprocess.run(['git','push','origin','pvst-rgbd-slice-relations'],cwd=REPO,check=True,timeout=120)
        remote=git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/pvst-rgbd-slice-relations').split()[0];save(out/'publication.json',{'status':'COMPLETE' if remote==head else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:save(out/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'head':git('rev-parse','HEAD')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();c=json.loads(Path(args.config).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    if git('branch','--show-current')!='pvst-rgbd-slice-relations':raise ValueError('WRONG_BRANCH')
    if git('status','--porcelain','--','research_tools/rgbd_slice','docs/pvst/rgbd_slice_config_v1.json'):raise ValueError('UNCOMMITTED_EXECUTION_SOURCE')
    code={str(f.relative_to(REPO)):digest(f) for f in (REPO/'research_tools/rgbd_slice').glob('*.py')};code['pair_relation']=digest(REPO/'research_tools/depth_slice/model.py')
    identity=hashlib.sha256(json.dumps({'code':code,'config':c,'source_manifest':digest(Path(c['source_manifest']))},sort_keys=True).encode()).hexdigest()
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())['identity']!=identity:raise ValueError('PROTOCOL_CHANGED')
    save(out/'identity.json',{'identity':identity,'head':git('rev-parse','HEAD'),'code':code});save(out/'config.json',c)
    if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text())['status']=='COMPLETE':return
    started=time.monotonic();summaries=[];manifest=None
    def progress(stage,**kw):
        record={'status':'RUNNING','stage':stage,'pid':os.getpid(),'wall_s':time.monotonic()-started,**kw};save(out/'progress.json',record);print(json.dumps(record),flush=True)
    try:
        torch.set_num_threads(4);seed_all(c['seed']);progress('prepare');save(out/'launch.json',{'pid':os.getpid(),'head':git('rev-parse','HEAD'),'gpu':torch.cuda.get_device_name(),'free_total':torch.cuda.mem_get_info(),'started_unix':time.time()})
        manifest=verify_registered(c,out,progress);prior=json.loads(Path(c['source_manifest']).read_text())
        if manifest['clips']!=prior['clips']:raise ValueError('FROZEN_CLIP_LIST_CHANGED')
        geometry(c,out,manifest['rows'],progress)
        if not (out/'common_initial.pt').exists():
            seed_all(c['seed']);encoder=backbone(c,out/'official_load.json');model=RGBDSlices(encoder,False);torch.save(model.local.state_dict(),out/'common_initial.pt');save(out/'initialization.json',{'common_sha256':state_hash(model.local.state_dict()),'local_parameters':sum(p.numel() for p in model.parameters()),'all_common_tensors_exactly_copied':True});del model,encoder
        checks(c,out,manifest,progress);fit=fitting(c,out,manifest,progress)
        if fit['status']!='PASS':status='FIT_BLOCKED'
        else:
            for condition in CONDITIONS:summaries.append(train(condition,c,out,manifest,identity,progress))
            status='COMPLETE'
        report(c,out,manifest,status,summaries);save(out/'progress.json',{'status':status,'stage':'report_complete','pid':os.getpid(),'exit_code':0 if status=='COMPLETE' else 2})
        publish(out);(out/'exit_code.txt').write_text('0\n' if status=='COMPLETE' else '2\n')
        if status!='COMPLETE':raise SystemExit(2)
    except SystemExit:raise
    except BaseException as error:
        detail=traceback.format_exc();print(detail,flush=True);(out/'error.txt').write_text(detail);status='OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else 'FAILED';save(out/'progress.json',{'status':status,'stage':'execution_failed','pid':os.getpid(),'error':str(error),'exit_code':1});(out/'exit_code.txt').write_text('1\n')
        if manifest:report(c,out,manifest,status,summaries,detail);publish(out)
        raise


if __name__=='__main__':main()
