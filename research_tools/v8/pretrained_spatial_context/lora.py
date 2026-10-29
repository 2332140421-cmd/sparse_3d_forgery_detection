"""One spatial LoRA adaptation condition; reuse native metrics and fixed previews."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from research_tools.v8.local_geometry_pilot.runner import seed_all,atomic_torch,restore_rng,rng_state
from research_tools.v8.spatial_localization.joint_data import Video,save,SEED
from research_tools.v8.spatial_localization.metrics import history_csv
from research_tools.v8.spatial_localization.head_decoupling import sample_index
from scripts.prepare_v8_assets import digest
from . import run
from .assets import official,prepare,patch_features,verify_source
from .model import SpatialModel,pixel_sums
from .metrics import evaluate,curves,boundary_samples

CONDITION='SPATIAL_LORA_ADAPT'


class RGBVideo:
    """Read existing RGB/GT only. Never open patch.npy for adaptation or scoring."""
    def __init__(self,config,out,row,fit,context=False):
        self.row=row
        self.base=Video(Path(config['source_run']),row,fit['statistics'],fit['grouping'],read_only=True)
    def chunk(self,start,end):
        return {k:torch.from_numpy(np.array(self.base.arrays[k][start:end],copy=True)) for k in ['rgb512','gt','content512']}
    def native_gt(self,i):return self.base.native_gt(i)
    def close(self):self.base.close()


def checkpoint_blocks(backbone):
    """Checkpoint existing ViT blocks without changing names or upstream files."""
    blocks=[m for name,m in backbone.named_modules() if name.endswith('blocks') and isinstance(m,nn.ModuleList)]
    if len(blocks)!=1:raise ValueError('EXPECTED_ONE_DINO_BLOCK_LIST')
    for block in blocks[0]:
        original=block.forward
        def forward(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        block.forward=forward
    return len(blocks[0])


class LoRASpatial(SpatialModel):
    def __init__(self,network,statistics,activation_checkpoint=True):
        super().__init__('SPATIAL_TRANSFER',network.encoder.localization_head,statistics,freeze_decoder_bn=True)
        # These branches are unused by forward_backbone; don't keep a second decoder.
        network.encoder.localization_head=nn.Identity()
        network.encoder.head_features=nn.Identity();network.encoder.classifier=nn.Identity()
        self.network=network.requires_grad_(False).eval()
        self.lora_names=[k for k,p in self.network.backbone.named_parameters() if 'lora_' in k]
        if not self.lora_names:raise ValueError('OFFICIAL_TRAINED_LORA_REQUIRED')
        for k,p in self.network.backbone.named_parameters():p.requires_grad_(k in self.lora_names)
        self.checkpointed_blocks=checkpoint_blocks(self.network.backbone) if activation_checkpoint else 0
        self.train()
    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'network'):self.network.eval()
        return self
    def chunk(self,cpu,previous,active,first,m256,m512):
        device=next(self.parameters()).device
        with torch.autocast(device.type,enabled=False):
            features=patch_features(self.network,cpu['rgb512'].to(device)).half().float()
        logits=self.decoder(features,target_size=(518,518))[:,0,:512,:512]
        return previous,active,logits
    def compact_state(self):
        return {'lora':{k:p.detach().cpu().clone() for k,p in self.network.backbone.named_parameters() if k in self.lora_names},
                'decoder':{k:v.detach().cpu().clone() for k,v in self.decoder.state_dict().items()}}
    def restore(self,state):
        if set(state)!= {'lora','decoder'} or set(state['lora'])!=set(self.lora_names):raise ValueError('COMPACT_CHECKPOINT_KEYS')
        with torch.no_grad():
            for k,p in self.network.backbone.named_parameters():
                if k in self.lora_names:p.copy_(state['lora'][k])
        self.decoder.load_state_dict(state['decoder'],strict=True);self.bn_status()
    def parameter_groups(self,config):
        return [{'params':[p for p in self.network.backbone.parameters() if p.requires_grad],'lr':config['lora_lr'],'name':'lora'},
                {'params':list(self.decoder.parameters()),'lr':config['decoder_lr'],'name':'decoder'}]


def backward_video(model,video,chunk_frames,amp=True):
    """Immediate backward, exact .5/video-frame normalization, including short tail."""
    n=video.row['actual_frames'];parts=np.zeros(3);device=next(model.parameters()).device
    for start in range(0,n,chunk_frames):
        cpu=video.chunk(start,min(start+chunk_frames,n))
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
            z=model.chunk(cpu,None,None,start==0,video.row['mapping256'],video.row['mapping512'])[2]
            values=pixel_sums(z,cpu['gt'].to(device),cpu['content512'].to(device))
        loss=.5*values[0]/n
        if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAIN_LOSS')
        loss.backward();parts+=np.array([float(v.detach()) for v in values])*.5/n
    return dict(zip(['loss','bce','dice'],parts.tolist()))


def pair_update(model,opt,pair,config,out,fit):
    started=time.monotonic();model.train();opt.zero_grad(set_to_none=True);losses={k:0. for k in ['loss','bce','dice']}
    for row in sorted(pair,key=lambda r:r['role']):
        video=RGBVideo(config,out,row,fit)
        try:
            parts=backward_video(model,video,config['chunk_frames'],config['amp'])
            for key in losses:losses[key]+=parts[key]
        finally:video.close()
    grads={group['name']:sum(float(p.grad.abs().sum()) for p in group['params'] if p.grad is not None) for group in opt.param_groups}
    norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('FROZEN_GRADIENT')
    opt.step();torch.cuda.synchronize();model.bn_status()
    if model.network.training or any(m.training for m in model.network.modules()):raise ValueError('BACKBONE_MUST_STAY_EVAL')
    return {**losses,'gradient_norm':norm,'gradient_abs_sum':grads,'seconds':time.monotonic()-started,
            'frames':sum(r['actual_frames'] for r in pair)}


def evaluation(model,config,out,rows,fit,**kwargs):
    result=evaluate(model,config,out,rows,fit,video_factory=RGBVideo,**kwargs)
    for record in result['records']:
        tn,fp,fn,tp=np.asarray(record['pixel_confusion_matrix']).reshape(-1)
        record['gt_recall']=float(tp/(tp+fn)) if tp+fn else None
    recalls=[r['gt_recall'] for r in result['records'] if r['gt_recall'] is not None]
    result['metrics']['gt_recall']=float(np.mean(recalls)) if recalls else None
    return result


@torch.no_grad()
def initialization_check(model,config,out,rows,fit):
    model.eval();checks=[]
    for portrait in [False,True]:
        row=next(r for r in rows if r['split']=='train' and r['role']=='fake' and (r['source_hw'][0]>r['source_hw'][1])==portrait)
        v=RGBVideo(config,out,row,fit);fi=row['actual_frames']//2
        try:
            cpu=v.chunk(fi,fi+1)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.chunk(cpu,None,None,True,row['mapping256'],row['mapping512'])[2]
            cache=Path(config['original_run'])/'features'/row['video_id']/'patch.npy'
            meta=json.loads(cache.with_name('complete.json').read_text())
            if meta['video_sha256']!=row['video_sha256'] or meta['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('PARITY_CACHE_IDENTITY')
            f=torch.from_numpy(np.array(np.load(cache,mmap_mode='r')[fi:fi+1],copy=True)).float().cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):old=model.decoder(f,target_size=(518,518))[:,0,:512,:512]
            mask=cpu['content512'].cuda();delta=(z.float()-old.float()).abs()[mask]
            mean,maxdiff=float(delta.mean()),float(delta.max());flips=float(((z>=0)!=(old>=0))[mask].float().mean())
            if mean>.01 or maxdiff>.1 or flips>.005:raise ValueError('INITIALIZATION_CACHE_PARITY')
            checks.append({'pair_id':row['pair_id'],'frame_index':fi,'portrait':portrait,'mean_abs':mean,'max_abs':maxdiff,'threshold_flip':flips})
        finally:v.close()
    save(out/'initialization_parity.json',checks)


def smoke(model,config,out,rows,fit):
    model.eval();initial=model.compact_state();frozen={k:p.detach().cpu().clone() for k,p in list(model.network.backbone.named_parameters())[:2] if not p.requires_grad}
    records=[];train=[r for r in rows if r['split']=='train']
    for portrait in [False,True]:
        pid=next(r['pair_id'] for r in train if (r['source_hw'][0]>r['source_hw'][1])==portrait)
        pair=[r for r in train if r['pair_id']==pid];model.restore(initial)
        opt=torch.optim.AdamW(model.parameter_groups(config),weight_decay=1e-4)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        update=pair_update(model,opt,pair,config,out,fit)
        state=model.compact_state()
        for name in ['lora','decoder']:
            if not any(not torch.equal(initial[name][k],v) for k,v in state[name].items()):raise ValueError('SMOKE_DID_NOT_UPDATE:'+name)
        if not all(update['gradient_abs_sum'][k]>0 for k in ['lora','decoder']):raise ValueError('SMOKE_MISSING_GRADIENT')
        if any(not torch.equal(p.detach().cpu(),frozen[k]) for k,p in model.network.backbone.named_parameters() if k in frozen):raise ValueError('FROZEN_WEIGHT_CHANGED')
        peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        ev=evaluation(model,config,out,pair,fit);ap=evaluation(model,config,out,pair,fit,pixel_ap=True)
        row=pair[0];v=RGBVideo(config,out,row,fit);cpu=v.chunk(0,1);model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.chunk(cpu,None,None,True,row['mapping256'],row['mapping512'])[2].clone()
        atomic_torch(out/'smoke_reload.pt',state);model.restore(initial);model.restore(torch.load(out/'smoke_reload.pt',weights_only=False,map_location='cpu'))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):b=model.chunk(cpu,None,None,True,row['mapping256'],row['mapping512'])[2]
        torch.testing.assert_close(a,b,atol=0,rtol=0);v.close();(out/'smoke_reload.pt').unlink()
        records.append({'pair_id':pid,'portrait':portrait,**update,'peak_bytes':peak,'evaluation_s':ev['seconds'],
                        'exact_AP_evaluation_s':ap['seconds'],'bn_status':model.bn_status(),'reload_max_difference':float((a-b).abs().max())})
        del opt,a,b;torch.cuda.empty_cache()
    model.restore(initial)
    counts={'lora':sum(p.numel() for p in model.network.backbone.parameters() if p.requires_grad),'decoder':sum(p.numel() for p in model.decoder.parameters()),
            'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad),'checkpointed_blocks':model.checkpointed_blocks}
    rate=sum(r['seconds'] for r in records)/sum(r['frames'] for r in records)
    erate=sum(r['evaluation_s'] for r in records)/sum(r['frames'] for r in records)
    aprate=max(0,sum(r['exact_AP_evaluation_s']-r['evaluation_s'] for r in records))/sum(r['frames'] for r in records)
    monitor=[r for r in rows if r.get('train_monitor')];dev=[r for r in rows if r['split']=='dev']
    budget={'train30_s':rate*sum(r['actual_frames'] for r in train)*30,
            'monitor_dev30_s':erate*sum(r['actual_frames'] for r in monitor+dev)*30,
            'periodic_AP_extra_s':aprate*sum(r['actual_frames'] for r in dev)*6,
            'final_best_last_train_dev_s':2*(erate+aprate)*sum(r['actual_frames'] for r in rows)}
    result={'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'records':records,'parameters':counts,'projection_max30':budget,'budget_limits':'two complete pairs, extrapolation only; excludes IO; early stopping may shorten',
            'smoke_weights':'discarded; reload official initialization and reset seed for formal training','chunk_frames':config['chunk_frames']}
    save(out/'resource_smoke.json',result);return result


def payload(model,opt,identity,epoch,position,history,online,updates,best,best_epoch,anchor,stale,elapsed):
    return {'identity':identity,'model':model.compact_state(),'optimizer':opt.state_dict(),'rng':rng_state(),
            'amp':{},'amp_dtype':'FP32 backbone; differentiable FP16 roundtrip; BF16 decoder; no scaler',
            'epoch':epoch,'pair_index':position,'history':history,'online':online,'updates':updates,
            'best':best,'best_epoch':best_epoch,'patience_ref':anchor,'stale':stale,'elapsed_s':elapsed}


def train(model,config,out,rows,fit,resume=False):
    folder=out/CONDITION;folder.mkdir(exist_ok=True)
    identity={'config':config,'manifest_sha256':digest(out/'manifest.json'),'official_load_sha256':digest(out/'official_load.json'),
              'code':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'references':json.loads((out/'baseline_references.json').read_text())}
    if (folder/'summary.json').exists():
        if json.loads((folder/'summary.json').read_text())['identity']!=identity:raise ValueError('COMPLETED_IDENTITY_CHANGED')
        return
    trainrows=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];monitor=[r for r in trainrows if r.get('train_monitor')]
    if len(monitor)!=8:raise ValueError('EXPECTED_ORIGINAL_FOUR_MONITOR_PAIRS')
    pairs={p:[r for r in trainrows if r['pair_id']==p] for p in sorted({r['pair_id'] for r in trainrows})}
    seed_all(SEED);opt=torch.optim.AdamW(model.parameter_groups(config),weight_decay=1e-4)
    initial=json.loads((Path(config['original_run'])/'epoch0.json').read_text());best=initial['dev']['metrics']['spatial_dice']
    epoch=position=updates=best_epoch=stale=0;history=[];online=[];anchor=best;elapsed=0.
    last=folder/'last.pt'
    if last.exists():
        if not resume:raise ValueError('RESUME_REQUIRED')
        cp=torch.load(last,map_location='cpu',weights_only=False)
        if cp['identity']!=identity:raise ValueError('RESUME_IDENTITY')
        model.restore(cp['model']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
        epoch,position,history,online,updates,best,best_epoch,anchor,stale,elapsed=(cp[k] for k in ['epoch','pair_index','history','online','updates','best','best_epoch','patience_ref','stale','elapsed_s'])
        if epoch!=len(history) or position!=len(online):raise ValueError('RESUME_POSITION')
        del cp
    else:atomic_torch(folder/'best.pt',payload(model,opt,identity,0,0,[],[],0,best,0,anchor,0,0.))
    begin=time.monotonic();model.train();model.bn_status()
    while epoch<30 and stale<8:
        keys=sorted(pairs);random.Random(SEED+epoch).shuffle(keys);torch.cuda.reset_peak_memory_stats()
        for j in range(position,len(keys)):
            run.progress(out,'train',condition=CONDITION,epoch=epoch+1,pair=j+1,total_pairs=len(keys))
            online.append(pair_update(model,opt,pairs[keys[j]],config,out,fit));updates+=1
            atomic_torch(last,payload(model,opt,identity,epoch,j+1,history,online,updates,best,best_epoch,anchor,stale,elapsed+time.monotonic()-begin))
            if run.STOP:raise InterruptedError('SAFE_PAIR_BOUNDARY_STOP')
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();bn=model.bn_status()
        run.progress(out,'evaluate',condition=CONDITION,epoch=epoch+1)
        tm=evaluation(model,config,out,monitor,fit);dv=evaluation(model,config,out,dev,fit,pixel_ap=(epoch+1)%5==0)
        dice=dv['metrics']['spatial_dice'];improved=dice>best
        if improved:best=dice;best_epoch=epoch+1
        if dice>anchor+.002:anchor=dice;stale=0
        else:stale+=1
        record={'epoch':epoch+1,'online':{k:float(np.mean([r[k] for r in online])) for k in ['loss','bce','dice']},
                'monitor':tm['metrics'],'dev':dv['metrics'],'monitor_scope':'original ANAKIN4 train pairs, not full train',
                'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in online])),
                'gradient_abs_sum_mean':{k:float(np.mean([r['gradient_abs_sum'][k] for r in online])) for k in ['lora','decoder']},
                'lr':{'lora':config['lora_lr'],'decoder':config['decoder_lr']},'updates':updates,
                'train_s':sum(r['seconds'] for r in online),'eval_s':tm['seconds']+dv['seconds'],
                'frames':sum(r['frames'] for r in online),'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'bn_status':bn}
        save(folder/'epoch_metrics'/f'epoch_{epoch+1:02d}.json',{'monitor':tm,'dev':dv,'pair_order':keys,'updates':online})
        history.append(record);save(folder/'history.json',history);history_csv(folder/'history.csv',history);curves(out,CONDITION,history,train_split='monitor')
        epoch+=1;position=0;online=[]
        cp=payload(model,opt,identity,epoch,0,history,online,updates,best,best_epoch,anchor,stale,elapsed+time.monotonic()-begin)
        atomic_torch(last,cp)
        if improved:atomic_torch(folder/'best.pt',cp)
        del cp
        print(f'{CONDITION} epoch={epoch:02d} online={record["online"]["loss"]:.6f} monitor_dice={tm["metrics"]["spatial_dice"]:.6f} dev_dice={dice:.6f} best={best:.6f}@{best_epoch} stale={stale}/8 BN=FIXED',flush=True)
    training_elapsed=elapsed+time.monotonic()-begin;metrics={};started=time.monotonic()
    for ck in ['last','best']:
        if ck=='best':model.restore(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
        run.progress(out,'final_metrics',condition=CONDITION,checkpoint=ck)
        metrics[ck]={split:evaluation(model,config,out,rr,fit,pixel_ap=True,samples=folder/'samples'/ck)
                     for split,rr in [('train',trainrows),('dev',dev)]}
    costs={'training_loop_s':training_elapsed,'train_updates_s':sum(r['train_s'] for r in history),'routine_evaluation_s':sum(r['eval_s'] for r in history),
           'final_evaluation_s':time.monotonic()-started,'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in history),
           'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in history),'parameters':json.loads((out/'resource_smoke.json').read_text())['parameters']}
    save(folder/'summary.json',{'status':'COMPLETE','best_epoch':best_epoch,'last_epoch':epoch,'updates':updates,'identity':identity,
                               'stop_reason':'patience8_delta0.002' if stale>=8 else 'max30','costs':costs,**metrics})


def recall_metrics(result):
    records=[r for r in result['records'] if r['role']=='fake']
    values=[];cm=np.zeros((2,2),np.int64)
    for r in records:
        a=np.asarray(r['pixel_confusion_matrix']);tn,fp,fn,tp=a.reshape(-1);values.append(tp/(tp+fn));cm+=a
    tn,fp,fn,tp=cm.reshape(-1)
    return {'video_mean_GT_recall':float(np.mean(values)),'pooled_GT_recall':float(tp/(tp+fn)),'pooled_BG_FPR':float(fp/(fp+tn))}


def report(config,out,rows):
    baseline=Path(config['baseline_run']);original=Path(config['original_run'])
    old=json.loads((baseline/'SPATIAL_TRANSFER/summary.json').read_text());historical=json.loads((original/'SPATIAL_TRANSFER/summary.json').read_text())
    new=json.loads((out/CONDITION/'summary.json').read_text());summaries={'固定BN冻结视觉':old,'LoRA适配固定BN':new,'原BN历史参考':historical}
    lines=['# V8空间定位LoRA适配对照','',
           '仅新训练SPATIAL_LORA_ADAPT；全部条件从同一官方已训练LoRA/decoder初始化。主要对照固定BN冻结视觉，原BN仅历史参考。',
           '完整输入/全部帧/空间BCE+Dice不变；无三维增强，无检测门控。单seed和8条开发fake仅作描述性比较。','',
           '|条件|best/结束轮次|完整训练Dice|开发Dice|IoU|pixel AP|GT召回(视频均值)|real空间FP|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    keys=['spatial_dice','spatial_iou','pixel_ap','real_spatial_fpr']
    for label,s in summaries.items():
        m=s['best']['dev']['metrics'];recall=recall_metrics(s['best']['dev'])['video_mean_GT_recall']
        vals=[s['best']['train']['metrics']['spatial_dice']]+[m[k] for k in keys[:3]]+[recall,m['real_spatial_fpr']]
        lines.append('|'+ '|'.join([label,f'{s["best_epoch"]}/{s["last_epoch"]}']+[f'{v:.6f}' for v in vals])+'|')
    delta={k:new['best']['dev']['metrics'][k]-old['best']['dev']['metrics'][k] for k in keys}
    delta['gt_recall']=recall_metrics(new['best']['dev'])['video_mean_GT_recall']-recall_metrics(old['best']['dev'])['video_mean_GT_recall']
    lines.append('LoRA−固定BN冻结视觉：'+json.dumps(delta,ensure_ascii=False))
    a={r['video_id']:r for r in old['best']['dev']['records'] if r['role']=='fake'};b={r['video_id']:r for r in new['best']['dev']['records'] if r['role']=='fake'}
    if set(a)!=set(b) or len(a)!=8:raise ValueError('MATCHED_EIGHT_DEV_FAKE')
    paired=[{'video_id':v,'pair_id':a[v]['pair_id'],'generator':a[v]['generator'],'baseline':a[v]['spatial_dice'],'lora':b[v]['spatial_dice'],
             'delta':b[v]['spatial_dice']-a[v]['spatial_dice']} for v in a]
    lines.extend(['','|开发fake|类型|冻结视觉Dice|LoRA Dice|差|','|---|---|---:|---:|---:|'])
    for r in paired:lines.append('|'+ '|'.join([r['pair_id'],r['generator']]+[f'{r[k]:.6f}' for k in ['baseline','lora','delta']])+'|')
    groups={}
    subsets={'全部8条':paired,'原较弱6条':[r for r in paired if r['pair_id'] not in {'an:4rvro9','an:4wgx37'}]}
    for gen in sorted({r['generator'] for r in paired}):subsets[gen]=[r for r in paired if r['generator']==gen]
    for label,rs in subsets.items():
        ds=[r['delta'] for r in rs];groups[label]={'n':len(ds),'mean':float(np.mean(ds)),'up':sum(d>0 for d in ds),'tie':sum(d==0 for d in ds),'down':sum(d<0 for d in ds)}
        lines.append(label+'：'+json.dumps(groups[label],ensure_ascii=False))
    train_delta=new['best']['train']['metrics']['spatial_dice']-old['best']['train']['metrics']['spatial_dice']
    gap=new['best']['train']['metrics']['spatial_dice']-new['best']['dev']['metrics']['spatial_dice']
    lines.append(f'完整训练Dice变化={train_delta:+.6f}，开发变化={delta["spatial_dice"]:+.6f}，best训练−开发差={gap:+.6f}；曲线中的monitor仅为固定4pair，不冒充完整训练。')
    lines.extend(['','|LoRA checkpoint|集合|Dice|IoU|pixel AP|real FP|','|---|---|---:|---:|---:|---:|'])
    for ck in ['best','last']:
        for split in ['train','dev']:
            m=new[ck][split]['metrics'];lines.append('|'+ '|'.join([ck+'/'+str(new[ck+'_epoch']),split]+[f'{m[k]:.6f}' for k in keys])+'|')
    # Read-only link: existing baseline predictions, never rerun the frozen model.
    link=out/'FROZEN_VISUAL'
    if not link.exists():link.symlink_to(baseline/'SPATIAL_TRANSFER',target_is_directory=True)
    chosen=[r for r in rows if r['split']=='dev' or r.get('train_monitor')]
    sample_index(out,chosen,conditions=('FROZEN_VISUAL',CONDITION),title='固定BN冻结视觉 vs 在线LoRA适配；各自best/last')
    boundaries=boundary_samples(out,rows,conditions=('FROZEN_VISUAL',CONDITION))
    lines.append('固定25/50/75%原生2px边界统计：'+json.dumps(boundaries,ensure_ascii=False))
    for label,s in [('冻结视觉',old),('LoRA',new)]:lines.append(label+'完整fake覆盖/背景：'+json.dumps(recall_metrics(s['best']['dev']),ensure_ascii=False))
    lines.append('边界只代表固定时刻；完整GT覆盖与背景统计单列，背景误报下降本身不等于定位改善。样例索引见index.html，原始F32响应未裁剪。')
    lines.append(f'实际{new["last_epoch"]}轮/{new["updates"]}次更新，best第{new["best_epoch"]}轮，停止原因{new["stop_reason"]}；成本：'+json.dumps(new['costs'],ensure_ascii=False))
    lines.append('smoke与满30轮预算外推见resource_smoke.json；正式成本单列，不将估计当实测。旧特征仅用于初始化核对；训练/评分全用当前LoRA在线FP32视觉、可微FP16量化、BF16 decoder。')
    storage=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
    lines.append(f'报告前新增输出存储={storage}字节（checkpoint/必要固定样例/曲线/日志，无逐轮特征，无主干副本）。')
    if delta['spatial_dice']>0 and groups['原较弱6条']['up']>=4:decision='唯一下一步建议：保留视觉适配基础，先扩充空间监督覆盖以检验开发收益的稳定性。'
    elif delta['spatial_dice']>0:decision='唯一下一步建议：保留本轮结果，优先增加弱场景的空间监督覆盖，核实收益是否能泛化。'
    else:decision='唯一下一步建议：停止增加当前模型适配能力，优先检查并扩充空间监督覆盖。'
    lines.extend([decision,'本轮收益只能归于视觉适配条件，不归于三维或component；不自动追加实验。'])
    result={'status':'COMPLETE','baseline_best_epoch':old['best_epoch'],'new_best_epoch':new['best_epoch'],'new_last_epoch':new['last_epoch'],
            'baseline':old['best'],'lora':new['best'],'historical':historical['best'],'delta':delta,'paired_dev_fake':paired,'groups':groups,
            'boundary':boundaries,'train_dev_gap':gap,'costs':new['costs'],'output_bytes_before_report':storage,'recommendation':decision}
    save(out/'result_summary.json',result);(out/'report_zh.md').write_text('\n'.join(lines)+'\n')


def reference_snapshot(config,out):
    paths=[]
    for root in [Path(config['baseline_run']),Path(config['original_run'])]:
        paths.extend([root/'manifest.json',root/'epoch0.json',root/'official_load.json',root/'SPATIAL_TRANSFER/summary.json',root/'SPATIAL_TRANSFER/history.json'])
    value={str(p):digest(p) for p in paths};path=out/'baseline_references.json'
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('READ_ONLY_BASELINE_CHANGED')
    if not path.exists():save(path,value)
    return value



def publish_results(config,out,repo=None):
    """Publish only this completed experiment's small results, on a clean task branch."""
    repo=Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    if git('branch','--show-current')!='v8-spatial-lora-adaptation':raise ValueError('PUBLICATION_BRANCH_CHANGED')
    if git('status','--porcelain'):raise ValueError('PUBLICATION_WORKTREE_CHANGED; preserve user changes')
    result=json.loads((out/'result_summary.json').read_text())
    if result['status']!='COMPLETE':raise ValueError('RESULTS_NOT_COMPLETE')
    document=repo/'docs/experiments/v8-spatial-lora-adaptation.md'
    text=document.read_text().replace('RUNNING；','COMPLETE；').replace('PREFLIGHT_COMPLETE；','COMPLETE；')
    m=result['lora']['dev']['metrics'];train=result['lora']['train']['metrics']
    row='|SPATIAL_LORA_ADAPT|'+str(result['new_best_epoch'])+'/'+str(result['new_last_epoch'])+'|'+ '|'.join(f'{v:.6f}' for v in [train['spatial_dice'],m['spatial_dice'],m['spatial_iou'],m['pixel_ap'],m['real_spatial_fpr']])+'|'
    text=text.replace('|SPATIAL_LORA_ADAPT|尚未训练|—|—|—|—|—|',row).replace('最终结果尚未产生，不预填收益。','最终指标见下文完成结果和小型结果JSON。').replace('完成后集中审阅并提交小型结果和必要曲线；不追加实验。','已集中汇总并提交小型结果和必要曲线；不追加实验。')
    text+='\n## 完成结果（自动集中汇总）\n\n'+(out/'report_zh.md').read_text()
    document.write_text(text)
    small={k:result[k] for k in ['status','baseline_best_epoch','new_best_epoch','new_last_epoch','delta','paired_dev_fake','groups','boundary','train_dev_gap','costs','output_bytes_before_report','recommendation']}
    small['best_metrics']={name:{split:result[name][split]['metrics'] for split in ['train','dev']} for name in ['baseline','lora','historical']}
    result_path=repo/'docs/experiments/v8-spatial-lora-adaptation-results.json';save(result_path,small)
    curve=repo/'docs/experiments/v8-spatial-lora-adaptation-curves.png';shutil.copyfile(out/CONDITION/'curves.png',curve)
    status=repo/'docs/v8/current_status.md';text=status.read_text()
    start=text.index('## 当前执行：空间定位LoRA适配');end=text.index('## 已完成：固定decoder BN',start)
    m=result['lora']['dev']['metrics'];d=result['delta'];summary=json.loads((out/CONDITION/'summary.json').read_text())
    text=text[:start]+f"## 已完成：空间定位LoRA适配\n\n单条件SPATIAL_LORA_ADAPT已完成训练、最终评价和报告。实际{summary['last_epoch']}轮/{summary['updates']}更新，best第{summary['best_epoch']}轮。\n开发Dice/IoU/AP/real FP={m['spatial_dice']:.6f}/{m['spatial_iou']:.6f}/{m['pixel_ap']:.6f}/{m['real_spatial_fpr']:.6f}；\n相对固定BN冻结视觉Dice差{d['spatial_dice']:+.6f}。详见[完成报告](../experiments/v8-spatial-lora-adaptation.md)。\n仅视觉适配，不归因于三维；旧资产只读核对通过，不自动追加实验。\n\n"+text[end:]
    status.write_text(text)
    paths=[str(p.relative_to(repo)) for p in [document,result_path,curve,status]]
    subprocess.run(['git','add','--',*paths],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Record completed V8 spatial LoRA adaptation results'],cwd=repo,check=True)
    subprocess.run(['git','push','origin','v8-spatial-lora-adaptation'],cwd=repo,check=True)
    head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/v8-spatial-lora-adaptation').split()[0]
    if head!=remote:raise ValueError('PUBLICATION_REMOTE_HEAD_MISMATCH')
    save(out/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote,'files':paths})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--phase',choices=['smoke','all'],default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--publish-results',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);signal.signal(signal.SIGTERM,run.stop);signal.signal(signal.SIGINT,run.stop)
    try:
        if config['seed']!=SEED or config['epochs']!=30 or not config['freeze_decoder_bn']:raise ValueError('FROZEN_PROTOCOL')
        if any(Path(config[k]).resolve()==out.resolve() or Path(config[k]).resolve() in out.resolve().parents for k in ['source_run','baseline_run','original_run']):raise ValueError('SOURCE_OUTPUT_OVERLAP')
        run.progress(out,'assets');refs=reference_snapshot(config,out);rows,fit=prepare(config,out)
        if rows!=json.loads((Path(config['baseline_run'])/'manifest.json').read_text())['rows']:raise ValueError('MATCHED_BASELINE_MANIFEST')
        seed_all(SEED);network=official(config,out);model=LoRASpatial(network,fit['statistics'],config['activation_checkpoint']).cuda()
        run.progress(out,'initialization_parity');initialization_check(model,config,out,rows,fit)
        initial=model.compact_state();smoke_path=out/'resource_smoke.json'
        if not smoke_path.exists():
            run.progress(out,'smoke');smoke(model,config,out,rows,fit)
        elif json.loads(smoke_path.read_text())['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():
            raise ValueError('SMOKE_CONFIG_CHANGED')
        if args.phase=='smoke':save(out/'progress.json',{'status':'SMOKE_COMPLETE','pid':os.getpid()});return
        model.restore(initial);seed_all(SEED)
        train(model,config,out,rows,fit,args.resume);run.progress(out,'report');report(config,out,rows)
        verify_source(config,out)
        if reference_snapshot(config,out)!=refs:raise ValueError('BASELINE_CHANGED')
        if args.publish_results:publish_results(config,out)
        save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'unix_time':time.time()})
    except BaseException as e:
        save(out/'progress.json',{'status':'INTERRUPTED' if isinstance(e,InterruptedError) else 'FAILED','pid':os.getpid(),
                                'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
