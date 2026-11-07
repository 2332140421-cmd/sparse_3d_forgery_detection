"""One pilot CLI: prepare -> checks -> train -> best evaluation -> report/publish.

This is an experiment-specific runner, not a job scheduler. Resume restores the
checkpoint, optimizer, RNG, history and early stopping under a pinned identity.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import av
import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchvision.models import resnet18

from .data import Video, prepare, save
from .model import DepthSliceRelation, terms
from research_tools.v8.spatial_localization.metrics import classification, spatial_frame
from scripts.prepare_v8_assets import digest

REPO=Path(__file__).resolve().parents[2]


def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def observe(config,out,rows,progress):
    """Only frozen ImageNet RGB maps are new. Existing MoGe/RGB remain read-only."""
    state=torch.load(config['resnet_weights'],map_location='cpu',weights_only=True)
    backbone=resnet18(weights=None);backbone.load_state_dict(state)
    backbone=torch.nn.Sequential(*list(backbone.children())[:-2]).cuda().eval().requires_grad_(False)
    weight_sha=digest(Path(config['resnet_weights']));cost=time.monotonic()
    for r in rows:
        folder=out/'visual'/r['video_id'];done=folder/'complete.json'
        identity={'video_sha256':r['video_sha256'],'weights_sha256':weight_sha,'native_hw':r['source_hw'],'frames':r['actual_frames'],'stride':32,'dtype':'float16','backbone':'official ImageNet ResNet18 all stages, frozen eval, native RGB'}
        if done.exists():
            if json.loads(done.read_text())!=identity:raise ValueError('VISUAL_CACHE_IDENTITY')
            continue
        folder.mkdir(parents=True,exist_ok=True);v=Video(r,config);n=r['actual_frames'];features=None;decoded=0
        with av.open(r['video_path']) as container:
            stream=container.streams.video[0];stream.thread_count=2
            for i,frame in enumerate(container.decode(stream)):
                if i>=n or frame.pts!=r['pts'][i]['pts_ticks'] or str(frame.time_base)!=r['pts'][i]['time_base']:raise ValueError('NATIVE_PTS_DECODE')
                if [frame.height,frame.width]!=r['source_hw']:raise ValueError('NATIVE_HW')
                if i in {0,n//2,n-1} and not np.array_equal(frame.to_ndarray(format='rgb24'),v.rgb[i]):raise ValueError('RGB_DECODE_CACHE_MISMATCH')
                x=torch.from_numpy(np.array(v.rgb[i],copy=True)).permute(2,0,1)[None].cuda().float()/255
                x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):f=backbone(x)
                if features is None:features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float16,shape=(n,*f.shape[1:]))
                features[i]=f[0].float().cpu().numpy().astype(np.float16);decoded+=1
        v.close()
        if decoded!=n:raise ValueError('NATIVE_DECODE_FRAME_COUNT')
        features.flush();save(done,identity);progress('frozen_visual',video=r['video_id'],frames=n)
    del backbone;torch.cuda.empty_cache()
    save(out/'prepare_cost.json',{'wall_s':time.monotonic()-cost,'moge_calls':0,'rgb_source':'read-only registered native cache','new_feature':'frozen ImageNet ResNet18 native stride32'})


def frames(v,out,start,end):
    visual=np.load(out/'visual'/v.row['video_id']/'features.npy',mmap_mode='r')
    result=[]
    for i in range(start,end):
        f=v.frame(i);f['visual']=torch.from_numpy(np.array(visual[i],copy=True));result.append(f)
    return result


def amp():return torch.autocast('cuda',dtype=torch.bfloat16)


def forward(model,fs):
    # Checkpoint inputs remain CPU observations. Saved tensors within each frame
    # and relation chunk use the standard, autograd-preserving CPU offload.
    with torch.autograd.graph.save_on_cpu(pin_memory=True),amp():return model(fs)


def gradients(model):
    result={}
    for prefix in ['geo','fusion','token','time','depth','context','pixel','frame_head','frame_visual']:
        ps=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result[prefix]={'present':bool(ps),'finite':all(bool(torch.isfinite(g).all()) for g in ps),
                        'norm':float(torch.sqrt(sum((g.float().square().sum() for g in ps),torch.zeros((),device='cuda'))))}
    return result


def smoke(config,out,rows,clips,progress):
    r=next(r for r in rows if r['split']=='train' and r['role']=='fake')
    clip=next(c for c in clips if c['video_id']==r['video_id']);v=Video(r,config)
    fs=frames(v,out,clip['start'],clip['end']);v.close()
    if not any(f['gt'].any() for f in fs):raise ValueError('SMOKE_REQUIRES_POSITIVE_GT')
    seed_all(config['seed']);model=DepthSliceRelation().cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01);records=[]
    for step in range(2):
        progress('smoke',step=step);opt.zero_grad(set_to_none=True);pred=forward(model,fs)
        loss=torch.stack([terms(p,f['gt']) for p,f in zip(pred,fs)]).mean(0).sum()
        if not torch.isfinite(loss):raise ValueError('SMOKE_LOSS')
        loss.backward();g=gradients(model)
        if not all(x['present'] and x['finite'] for x in g.values()):raise ValueError('SMOKE_GRADIENT_PATH:'+str(g))
        before=model.pixel[-1].weight.detach().clone();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
        if torch.equal(before,model.pixel[-1].weight):raise ValueError('SMOKE_NO_UPDATE')
        records.append({'step':step+1,'loss':float(loss),'gradients':g,'optimizer_states':len(opt.state),'frame_logits':[float(p['frame']) for p in pred],'loss_terms':torch.stack([terms(p,f['gt']) for p,f in zip(pred,fs)]).mean(0).detach().cpu().tolist(),'slices':[p['slices'] for p in pred]});del pred,loss
    if not all(any(step['gradients'][name]['norm']>0 for step in records) for name in records[0]['gradients']):raise ValueError('SMOKE_BRANCH_NEVER_RECEIVED_NONZERO_GRADIENT')
    save(out/'smoke.json',{'status':'PASS','video_id':r['video_id'],'range':[clip['start'],clip['end']],'updates':records,'peak_allocated':torch.cuda.max_memory_allocated()})
    del opt,model,fs;torch.cuda.empty_cache()


def aggregate(records):
    mean=lambda k,rr:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None
    fake=[r for r in records if r['role']=='fake'];real=[r for r in records if r['role']=='real']
    return {'dice':mean('dice',fake),'iou':mean('iou',fake),'pixel_ap':mean('pixel_ap',fake),'real_fpr':mean('empty_fpr',real),'GT_recall':mean('gt_recall',fake),'pred_area':mean('pred_area',fake),
            'frame':classification([x for r in records for x in r['labels']],[x for r in records for x in r['scores']]),
            'video':classification([int(r['role']=='fake') for r in records],[r['logit'] for r in records]),'loss':mean('loss',records)}


@torch.no_grad()
def evaluate(model,rows,config,out,progress,samples=None):
    started=time.monotonic();model.eval();records=[]
    for r in rows:
        progress('evaluate',video=r['video_id']);v=Video(r,config);values={k:[] for k in ['dice','iou','pixel_ap','empty_fpr','loss','scores','labels','slices','gt_recall','pred_area']}
        n=r['actual_frames'];chosen={int(np.argmin(abs(np.array([p['pts_s'] for p in r['pts']])-(r['pts'][0]['pts_s']+f*(r['pts'][-1]['pts_s']-r['pts'][0]['pts_s']))))) for f in [.25,.5,.75]}
        for start in range(0,n,24):
            end=min(start+24,n);lo=max(0,end-24) if end==n else start
            fs=frames(v,out,lo,end)
            with amp():pred=model(fs)
            for i in range(start,end):
                p=pred[i-lo];gt=v.gt(i);z=p['pixel'].float().cpu().numpy();m=spatial_frame(z>=0,gt)
                values['scores'].append(float(p['frame']));values['labels'].append(bool(gt.any()));values['slices'].append(p['slices']);values['loss'].append(float(terms(p,torch.from_numpy(gt)).sum()))
                values['pred_area'].append(float((z>=0).mean()))
                if gt.any():
                    values['gt_recall'].append(float(((z>=0)&gt).sum()/gt.sum()))
                    for k in ['dice','iou']:values[k].append(m[k])
                    values['pixel_ap'].append(float(average_precision_score(gt.reshape(-1),z.reshape(-1))))
                else:values['empty_fpr'].append(m['empty_fpr'])
                if samples and r['video_id'] in config['sample_video_ids'] and i in chosen:
                    samples.mkdir(parents=True,exist_ok=True);im=np.array(v.rgb[i]);gtim=np.repeat((gt*255).astype(np.uint8)[...,None],3,-1)
                    prob=1/(1+np.exp(-np.clip(z,-60,60)));heat=cv2.applyColorMap((prob*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1];binary=np.repeat(((z>=0)*255).astype(np.uint8)[...,None],3,-1)
                    cv2.imwrite(str(samples/f'{r["video_id"]}_{i}.jpg'),np.concatenate([im,gtim,heat,binary],1)[:,:,::-1])
            del fs,pred
        v.close();rec={k:r[k] for k in ['video_id','pair_id','source_id','role','split','generator']}
        for k in ['dice','iou','pixel_ap','empty_fpr','loss','gt_recall','pred_area']:rec[k]=float(np.mean(values[k])) if values[k] else None
        rec.update({k:values[k] for k in ['scores','labels','slices']});rec['logit']=float(np.logaddexp.reduce(values['scores'])-np.log(n));rec['frames']=n;records.append(rec)
    if samples:
        samples.mkdir(parents=True,exist_ok=True);(samples/'index.html').write_text('<meta charset="utf-8"><p>固定身份与PTS 25/50/75%；RGB | GT编辑区 | 概率0–1 | logit≥0</p>'+''.join(f'<p>{p.name}</p><img width="100%" src="{p.name}">' for p in sorted(samples.glob('*.jpg'))))
    return {'metrics':aggregate(records),'records':records,'seconds':time.monotonic()-started,
            'scope':'all legal native frames; nonoverlap24 windows, tail uses left context to24; each frame once; causal attention; history resets at regular window boundaries',
            'spatial':'positive-GT frame mean then fake-video macro; pixel AP positive-frame macro then fake-video macro; real FPR separate'}


def checkpoint_state(model,opt,epoch,best_epoch,best,bad,history,identity):
    return {'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best':best,'bad':bad,'history':history,'identity':identity,
            'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}


def report(out,config,manifest,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    if history:
        fig,ax=plt.subplots(1,2,figsize=(10,3));epochs=[r['epoch'] for r in history]
        ax[0].plot(epochs,[r['loss'] for r in history],label='online train');ax[0].legend()
        for key in ['dev','monitor']:ax[1].plot(epochs,[r[key]['dice'] for r in history],label=key)
        ax[1].legend();fig.tight_layout();fig.savefig(out/'curves.png');plt.close(fig)
    dataset={}
    for split in ['train','dev']:
        rr=[r for r in manifest['rows'] if r['split']==split];ss=[manifest['label_statistics'][r['video_id']] for r in rr]
        dataset[split]={'pairs':len(rr)//2,'videos':len(rr),'frames':sum(s['frames'] for s in ss),'positive_frames':sum(s['positive_frames'] for s in ss),
                        'all_nonempty_fake':sum(s['all_nonempty'] for s in ss),'mixed_fake':sum(s['mixed'] for s in ss)}
    summary.update(data=dataset,storage_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    save(out/'result_summary.json',summary)
    text=['# 深度切片关系学习首轮实验','',f'状态：{summary["status"]}。独立方法DEPTH_SLICE_RELATION，单seed20260909；不是旧PVST体素实验。',
          '数据：'+json.dumps(dataset,ensure_ascii=False),
          '工作流：冻结官方ImageNet ResNet18原生stride32局部视觉→回映每个原生像素；XYZ/UV逐点128→256编码→融合128→同深度片mean/max→512维C→两组时间/深度显式8head成对关系→像素局部旁路+片上下文定位/独立帧头。无效几何保留视觉路径。d=Z/1.9888148307800293，h=1/8，d0=0；单位非米，无新增配准。',
          '训练片段：每视频预先固定首、中、尾最多3段24连续原生帧。开发完整原生帧，每帧一次；24帧因果窗口，最后尾段补左历史，常规边界重置历史。frame BCE+pixel BCE+softDice等权，逐帧/片段均值；空GT Dice使用epsilon1仍参与。',
          '全部指标固定logit≥0；视频log-mean-exp温度1；空间Dice/IoU：正GT帧均值后fake视频宏平均；pixel AP：正GT帧AP均值后fake视频宏平均，并非AP@IoU。',
          '监测：固定前两个训练pair的完整视频，称训练监测子集，不代表全训练集。best仅按开发Dice，min10/max30/patience8。',
          '完成摘要：'+json.dumps(summary,ensure_ascii=False),
          '','## 结论边界','相机运动本轮允许，不证明深度可比性已解决；此前CPU审计显示漂移/跨片问题。没有匹配二维对照，不主张三维优于二维。全程编辑样本不能检验复杂时间边界。检测监督只来自逐帧GT是否含编辑区，ANAKIN裁剪边界未作为伪造区间。']
    if summary.get('status')=='COMPLETE':
        best=json.loads((out/'best_dev.json').read_text());monitor=json.loads((out/'best_monitor.json').read_text());text+=['', 'best开发：'+json.dumps(best['metrics'],ensure_ascii=False),'best训练监测子集：'+json.dumps(monitor['metrics'],ensure_ascii=False),
            'epoch0开发：'+json.dumps(json.loads((out/'epoch0_dev.json').read_text())['metrics'],ensure_ascii=False),
            '学习判断：best监测Dice='+str(monitor['metrics']['dice'])+'，开发Dice='+str(best['metrics']['dice'])+'；相对epoch0开发Dice变化='+str(best['metrics']['dice']-json.loads((out/'epoch0_dev.json').read_text())['metrics']['dice'])+'。这些只证明此队列的拟合与开发差异，不证明泛化。',
            '定位错误证据：best开发编辑区召回='+str(best['metrics']['GT_recall'])+'，fake预测面积比例='+str(best['metrics']['pred_area'])+'，real误报面积='+str(best['metrics']['real_fpr'])+'。1−召回衡量漏覆盖，real面积衡量背景误报，二者分别解释；固定样例需人工查看，不将数字当成精确边界证据。',
            '训练监测/开发差距及固定样例见best_monitor.json、best_dev.json和samples/index.html。降低real误报不自动意味着GT覆盖提升。固定样例只展示预先选择身份及PTS，不按预测挑帧。']
    (out/'report.md').write_text('\n\n'.join(text)+'\n')
    doc=REPO/'docs/pvst/depth_slice_relation_pilot_v1.md';doc.write_text('\n\n'.join(text)+f'\n\n完整资产与固定样例：`{out}`。\n')


def publish(out):
    try:
        git('add','docs/pvst/depth_slice_relation_pilot_v1.md','docs/pvst/current_status.md')
        git('commit','-m','Report depth-slice relation pilot completion')
        head=git('rev-parse','HEAD');subprocess.run(['git','push','origin',git('branch','--show-current')],cwd=REPO,check=True,timeout=120)
        remote=git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/'+git('branch','--show-current')).split()[0]
        save(out/'publication.json',{'status':'COMPLETE' if remote==head else 'VERIFY_MISMATCH','head':head,'remote':remote})
    except Exception:
        save(out/'publication.json',{'status':'FAILED','error':traceback.format_exc(),'local_head':git('rev-parse','HEAD')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text())['status']=='COMPLETE':raise SystemExit('Already complete')
    started=time.monotonic();head=git('rev-parse','HEAD')
    relevant=git('status','--porcelain','--','research_tools/depth_slice','docs/pvst/depth_slice_pilot_config.json','docs/decisions/0033-depth-slice-relation-pilot.md')
    if relevant:raise ValueError('EXECUTION_SOURCE_UNCOMMITTED')
    identity=hashlib.sha256(json.dumps({'config':config,'source_manifest_sha256':digest(Path(config['source_manifest'])),'weights_sha256':digest(Path(config['resnet_weights'])),'code':{str(f.relative_to(REPO)):digest(f) for f in (REPO/'research_tools/depth_slice').glob('*.py')}},sort_keys=True).encode()).hexdigest()
    def progress(stage,**kw):
        record={'status':'RUNNING','stage':stage,'pid':os.getpid(),'head':head,'elapsed_s':time.monotonic()-started,**kw};save(out/'progress.json',record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    try:
        torch.set_num_threads(4);seed_all(config['seed']);progress('prepare');save(out/'config.json',config);save(out/'launch.json',{'head':head,'identity':identity,'pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'gpu_free_total':torch.cuda.mem_get_info(),'started_unix':time.time()})
        manifest=prepare(config,out,progress);rows=manifest['rows'];clips=manifest['clips'];observe(config,out,rows,progress)
        if args.prepare_only:return
        if not (out/'smoke.json').exists():smoke(config,out,rows,clips,progress)
        seed_all(config['seed']);model=DepthSliceRelation().cuda();opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
        dev=[r for r in rows if r['split']=='dev'];monitor=[r for r in rows if r['pair_id'] in config['monitor_pairs'] and r['split']=='train']
        history=[];best=-1.;best_epoch=0;bad=0;begin=1;updates=0
        if args.resume and (out/'last.pt').exists():
            state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            if state['identity']!=identity:raise ValueError('RESUME_IDENTITY_CHANGED')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);history=state['history'];best=state['best'];best_epoch=state['best_epoch'];bad=state['bad'];begin=state['epoch']+1;updates=history[-1]['updates']
            random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        elif (out/'last.pt').exists():raise ValueError('CHECKPOINT_EXISTS_USE_RESUME')
        if not (out/'epoch0_dev.json').exists():save(out/'epoch0_dev.json',evaluate(model,dev,config,out,progress))
        for epoch in range(begin,31):
            ep=time.monotonic();model.train();order=list(clips);random.Random(config['seed']+epoch).shuffle(order);losses=[];torch.cuda.reset_peak_memory_stats()
            for ci,c in enumerate(order):
                progress('train',epoch=epoch,clip=ci,total=len(order),updates=updates)
                r=next(r for r in rows if r['video_id']==c['video_id']);v=Video(r,config);fs=frames(v,out,c['start'],c['end']);v.close();opt.zero_grad(set_to_none=True)
                pred=forward(model,fs);parts=torch.stack([terms(x,f['gt']) for x,f in zip(pred,fs)]).mean(0);loss=parts.sum()
                if not torch.isfinite(loss):raise ValueError('NONFINITE_TRAIN_LOSS')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();updates+=1;losses.append(parts.detach().float().cpu().numpy());del fs,pred,parts,loss
            dm=evaluate(model,dev,config,out,progress,samples=out/f'samples_epoch{epoch:02d}' if epoch%5==0 else None);mm=evaluate(model,monitor,config,out,progress)
            score=dm['metrics']['dice'];improved=score>best
            if improved:best=score;best_epoch=epoch;bad=0
            else:bad+=1
            row={'epoch':epoch,'updates':updates,'loss':float(np.mean(np.sum(losses,axis=1))),'frame_BCE':float(np.mean(losses,0)[0]),'pixel_BCE':float(np.mean(losses,0)[1]),'dice_loss':float(np.mean(losses,0)[2]),'dev':dm['metrics'],'monitor':mm['metrics'],'epoch_s':time.monotonic()-ep,'cumulative_s':time.monotonic()-started,'best_epoch':best_epoch,'bad':bad,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'lr':config['lr'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            history.append(row);save(out/'history.json',history)
            from research_tools.v8.spatial_localization.metrics import history_csv
            history_csv(out/'history.csv',history);save(out/f'epoch{epoch:02d}_dev.json',dm)
            state=checkpoint_state(model,opt,epoch,best_epoch,best,bad,history,identity);state['config']=config
            torch.save(state,out/'last.pt')
            if improved:torch.save(state,out/'best.pt')
            print('EPOCH_COMPLETE '+json.dumps(row),flush=True)
            if epoch>=10 and bad>=8:break
        state=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        bd=evaluate(model,dev,config,out,progress,samples=out/'samples');bm=evaluate(model,monitor,config,out,progress);save(out/'best_dev.json',bd);save(out/'best_monitor.json',bm)
        summary={'status':'COMPLETE','best_epoch':best_epoch,'last_epoch':history[-1]['epoch'],'updates':updates,'parameters':sum(p.numel() for p in model.parameters()),'D':512,'best_dev':bd['metrics'],'best_monitor':bm['metrics'],'wall_s':time.monotonic()-started,'peak_allocated':max(r['peak_allocated'] for r in history),'peak_reserved':max(r['peak_reserved'] for r in history),'execution_head':head,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'C_max_occupied_per_frame':max(max(r['slices']) for r in bd['records']),'C_max_window_units_upper_bound':24*max(max(r['slices']) for r in bd['records'])}
        report(out,config,manifest,summary);save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'exit_code':0})
        status=REPO/'docs/pvst/current_status.md';old=status.read_text();marker='<!-- depth-slice-pilot -->';old=old.split(marker)[0];status.write_text(old+marker+'\n\n## 深度切片关系首轮训练\n\n已完成；单条件独立新方法，见[结果](depth_slice_relation_pilot_v1.md)。旧PVST资源/模型完成数不变。\n')
        publish(out);(out/'exit_code.txt').write_text('0\n')
    except BaseException as exc:
        error=traceback.format_exc();print(error,flush=True);(out/'error.txt').write_text(error);save(out/'progress.json',{'status':'OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED','stage':'execution_failed','pid':os.getpid(),'error':str(exc),'exit_code':1});(out/'exit_code.txt').write_text('1\n')
        if 'manifest' in locals():
            report(out,config,manifest,{'status':'OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'FAILED','error':str(exc),'execution_head':head,'wall_s':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        raise


if __name__=='__main__':main()
