"""Two matched visual conditions on the frozen expanded ANAKIN research subset."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from scripts.prepare_v8_assets import digest
from research_tools.v8.spatial_localization.joint_data import save,SEED
from research_tools.v8.local_geometry_pilot.runner import seed_all
from research_tools.v8.spatial_localization.head_decoupling import sample_index
from . import lora,run
from .assets import official,patch_features,AINPAINT_REV,DINO_REV
from .metrics import boundary_samples
from .expansion_data import VisualVideo

CONDITIONS=('FROZEN_VISUAL','LORA_ADAPT')


class MatchedSpatial(lora.LoRASpatial):
    def __init__(self,network,condition,strategy='none'):
        if condition not in CONDITIONS:raise ValueError('MATCHED_CONDITION')
        super().__init__(network,{},False);self.adapt=condition=='LORA_ADAPT';self.condition=condition
        for k,p in self.network.backbone.named_parameters():p.requires_grad_(self.adapt and k in self.lora_names)
        if self.adapt and strategy!='none':self.checkpointed_blocks=lora.checkpoint_blocks(self.network.backbone,every=1 if strategy=='all' else 2)
    def chunk(self,cpu,previous,active,first,m256,m512):
        if self.adapt:return super().chunk(cpu,previous,active,first,m256,m512)
        z=self.decoder(cpu['features'].to(next(self.parameters()).device),target_size=(518,518))[:,0,:512,:512]
        return previous,active,z


def parameter_counts(model):
    return {'lora':sum(p.numel() for p in model.network.backbone.parameters() if p.requires_grad),
            'decoder':sum(p.numel() for p in model.decoder.parameters()),'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad),
            'checkpointed_blocks':model.checkpointed_blocks}


def benchmark(config,out,rows,network):
    path=out/'resource_smoke.json'
    identity={'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'manifest_sha256':digest(out/'manifest.json')}
    if path.exists():
        result=json.loads(path.read_text())
        if result['identity']['manifest_sha256']!=identity['manifest_sha256'] or result.get('frozen_execution_config_sha256',result['identity']['config_sha256'])!=identity['config_sha256']:raise ValueError('BENCHMARK_IDENTITY_CHANGED')
        if 'checkpoint_strategy' in config and result['execution']!={'chunk_frames':config['chunk_frames'],'checkpoint_strategy':config['checkpoint_strategy']}:raise ValueError('FROZEN_EXECUTION_CHANGED')
        return result
    pair=[r for r in rows if r['pair_id']=='an:95qlg9'];records=[];parameters={};queue=[('all',2),('none',2)]
    free,total=torch.cuda.mem_get_info();trial=0
    while queue and trial<4:
        strategy,chunk=queue.pop(0);trial+=1;model=None;opt=None
        try:
            seed_all(SEED);model=MatchedSpatial(copy.deepcopy(network),'LORA_ADAPT',strategy).cuda();seed_all(SEED)
            opt=torch.optim.AdamW(model.parameter_groups(config),weight_decay=1e-4);initial=model.compact_state()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();cfg={**config,'condition':'LORA_ADAPT','chunk_frames':chunk}
            update=lora.pair_update(model,opt,pair,cfg,out,{},video_factory=VisualVideo)
            state=model.compact_state()
            if any(not update['gradient_abs_sum'][k]>0 for k in ['lora','decoder']):raise ValueError('SMOKE_MISSING_GRADIENT')
            if any(not any(not torch.equal(initial[k][n],v) for n,v in state[k].items()) for k in ['lora','decoder']):raise ValueError('SMOKE_MISSING_PARAMETER_UPDATE')
            record={'status':'ok','strategy':strategy,'chunk_frames':chunk,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**update,'bn_status':model.bn_status()}
            record['frames_per_second']=record['frames']/record['seconds'];record['headroom_ok']=record['peak_reserved_bytes']<free*.8
            records.append(record)
            if trial==2:queue.append(('none',4))
            elif trial==3:queue.append((strategy,chunk*2))
        except torch.cuda.OutOfMemoryError as e:
            records.append({'status':'OOM','strategy':strategy,'chunk_frames':chunk,'error':str(e),'pair_frames':sum(r['actual_frames'] for r in pair),'complete_pair_update':False})
            if trial==2:queue.append(('half',2))
            elif trial==3:queue.append(('all',4))
        finally:
            del model,opt;torch.cuda.empty_cache()
    good=[r for r in records if r['status']=='ok' and r['headroom_ok']]
    if not good:raise RuntimeError('NO_FULL_SCALE_RESOURCE_CONFIGURATION_WITH_HEADROOM')
    chosen=min(good,key=lambda r:r['seconds']);execution={'chunk_frames':chosen['chunk_frames'],'checkpoint_strategy':chosen['strategy']}
    if 'checkpoint_strategy' in config and execution!={'chunk_frames':config['chunk_frames'],'checkpoint_strategy':config['checkpoint_strategy']}:raise ValueError('PROBE_DIFFERS_FROM_FROZEN_EXECUTION')
    evaluations={}
    for c in CONDITIONS:
        seed_all(SEED);model=MatchedSpatial(copy.deepcopy(network),c,execution['checkpoint_strategy']).cuda();seed_all(SEED)
        parameters[c]=parameter_counts(model);cfg={**config,**execution,'condition':c}
        if c=='FROZEN_VISUAL':
            opt=torch.optim.AdamW(model.parameter_groups(cfg),weight_decay=1e-4);before=model.compact_state();torch.cuda.reset_peak_memory_stats()
            update=lora.pair_update(model,opt,pair,cfg,out,{},video_factory=VisualVideo);after=model.compact_state()
            if any(not torch.equal(v,after['lora'][k]) for k,v in before['lora'].items()):raise ValueError('FROZEN_LORA_UPDATED')
            if not any(not torch.equal(v,after['decoder'][k]) for k,v in before['decoder'].items()):raise ValueError('FROZEN_DECODER_DID_NOT_UPDATE')
            records.append({'condition':c,'strategy':execution['checkpoint_strategy'],'chunk_frames':execution['chunk_frames'],
                            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**update})
            del opt
        normal=lora.evaluation(model,cfg,out,pair,{},video_factory=VisualVideo);ap=lora.evaluation(model,cfg,out,pair,{},video_factory=VisualVideo,pixel_ap=True)
        evaluations[c]={'seconds_per_frame':normal['seconds']/sum(r['actual_frames'] for r in pair),
                        'exact_AP_seconds_per_frame':ap['seconds']/sum(r['actual_frames'] for r in pair)}
        # Strict compact reload parity, without saving a frozen backbone.
        row=pair[0];v=VisualVideo(cfg,out,row);cpu=v.chunk(0,1);model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.chunk(cpu,None,None,True,row['mapping256'],row['mapping512'])[2].clone()
        state=model.compact_state();model.restore(state)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):b=model.chunk(cpu,None,None,True,row['mapping256'],row['mapping512'])[2]
        torch.testing.assert_close(a,b,atol=0,rtol=0);v.close();del model,a,b;torch.cuda.empty_cache()
    n=sum(r['actual_frames'] for r in pair);train=sum(r['actual_frames'] for r in rows if r['split']=='train')
    dev=sum(r['actual_frames'] for r in rows if r['split']=='dev');monitor=sum(r['actual_frames'] for r in rows if r.get('train_monitor'))
    budgets={}
    for c in CONDITIONS:
        measured=chosen if c=='LORA_ADAPT' else records[-1];e=evaluations[c]
        budgets[c]={'train30_s':measured['seconds']/n*train*30,'monitor_dev30_s':e['seconds_per_frame']*(monitor+dev)*30,
                    'periodic_AP_extra_s':max(0,e['exact_AP_seconds_per_frame']-e['seconds_per_frame'])*dev*6,
                    'final_best_last_s':e['exact_AP_seconds_per_frame']*(train+dev)*2}
    result={'identity':identity,'records':records,'execution':execution,'parameters':parameters,'evaluation_rates':evaluations,
            'projection_max30':budgets,'available_bytes_before_probe':free,'limits':'one complete fixed training pair per compute combination; at most4; all smoke weights discarded; budgets extrapolate, exclude IO and shared-machine variation'}
    save(path,result);save(out/'execution_config.json',execution);return result


@torch.no_grad()
def features(config,out,rows,network):
    network.cuda().eval();load=json.loads((out/'official_load.json').read_text())
    for r in rows:
        if r['split']=='test_reserved':continue
        path=Path(r['feature_path']);meta=path.with_name('complete.json')
        expected={'checkpoint_sha256':config['checkpoint_sha256'],'video_sha256':r['video_sha256'],'official_revision':AINPAINT_REV,'dino_revision':DINO_REV,
                  'pts_sha256':hashlib.sha256(json.dumps(r['pts'],sort_keys=True).encode()).hexdigest(),
                  'shape':[r['actual_frames'],1024,37,37],'dtype':'float16','preprocess':load['preprocess']}
        if meta.exists():
            info=json.loads(meta.read_text())
            if any(info[k]!=v for k,v in expected.items()):raise ValueError('FROZEN_FEATURE_IDENTITY')
            a=np.load(path,mmap_mode='r')
            if list(a.shape)!=expected['shape'] or a.dtype!=np.float16:raise ValueError('FROZEN_FEATURE_HEADER')
            continue
        if Path(config['original_run']).resolve() in path.resolve().parents:raise ValueError('MISSING_REFERENCE_FEATURE_NO_OVERWRITE')
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('patch.part.npy')
        a=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=tuple(expected['shape']));cfg={**config,'condition':'LORA_ADAPT'}
        v=VisualVideo(cfg,out,r);started=time.monotonic()
        try:
            for start in range(0,r['actual_frames'],config['feature_chunk']):
                end=min(start+config['feature_chunk'],r['actual_frames']);rgb=v.chunk(start,end)['rgb512'].cuda()
                z=patch_features(network,rgb)
                if not torch.isfinite(z).all():raise ValueError('NONFINITE_FEATURES')
                a[start:end]=z.float().cpu().numpy().astype(np.float16)
        finally:v.close()
        a.flush();del a;tmp.replace(path);save(meta,{**expected,'sha256':digest(path),'seconds':time.monotonic()-started})
        run.progress(out,'features',video=r['video_id'],frames=r['actual_frames'],total_frames=r['actual_frames'])
    network.cpu();torch.cuda.empty_cache()


def distribution(records,rows):
    fake=[r for r in records if r['role']=='fake'];lookup={r['video_id']:r for r in rows};values=[r['spatial_dice'] for r in fake]
    groups={}
    for name,fn in [('type',lambda r:r['generator']),('area',lambda r:lookup[r['video_id']]['area_group'])]:
        groups[name]={k:{'n':sum(fn(r)==k for r in fake),'mean':float(np.mean([r['spatial_dice'] for r in fake if fn(r)==k])),
                         'median':float(np.median([r['spatial_dice'] for r in fake if fn(r)==k]))} for k in sorted({fn(r) for r in fake})}
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),'dice_below_0_05':sum(v<.05 for v in values),'groups':groups,
            'videos':[{k:r[k] for k in ['video_id','pair_id','generator','spatial_dice','spatial_iou','pixel_ap']} for r in fake]}


def report(config,out,rows):
    summaries={c:json.loads((out/c/'summary.json').read_text()) for c in CONDITIONS};reference=json.loads((out/'LORA_24_REFERENCE/summary.json').read_text())
    assets=json.loads((out/'asset_report.json').read_text());probe=json.loads((out/'resource_smoke.json').read_text())
    lines=['# V8空间监督扩量：冻结视觉与LoRA匹配对照','',
           '研究子集划分，不冒充完整官方benchmark；test_reserved仅技术验收，从未评分或提取测试DINO。',
           '仅检验视觉适配和当前扩量设置，不归因于三维或component。全部帧/完整分辨率/空间BCE+Dice不变。',
           '实际队列：'+json.dumps(assets,ensure_ascii=False),'',
           '|条件|checkpoint/轮次|完整训练Dice|开发Dice|IoU|pixel AP|GT召回|real空间FP|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for c,s in summaries.items():
        for ck in ['best','last']:
            m=s[ck]['dev']['metrics'];vals=[s[ck]['train']['metrics']['spatial_dice']]+[m[k] for k in ['spatial_dice','spatial_iou','pixel_ap','gt_recall','real_spatial_fpr']]
            lines.append('|'+ '|'.join([c,ck+'/'+str(s[ck+'_epoch'])]+[f'{v:.6f}' for v in vals])+'|')
    m=reference['metrics'];lines.append('|LORA_24_REFERENCE|旧best/12|不重新评分旧train|'+ '|'.join(f'{m[k]:.6f}' for k in ['spatial_dice','spatial_iou','pixel_ap','gt_recall','real_spatial_fpr'])+'|')
    delta={k:summaries['LORA_ADAPT']['best']['dev']['metrics'][k]-summaries['FROZEN_VISUAL']['best']['dev']['metrics'][k] for k in ['spatial_dice','spatial_iou','pixel_ap','gt_recall','real_spatial_fpr']}
    lines.append('同队列LoRA−冻结视觉：'+json.dumps(delta,ensure_ascii=False));distributions={}
    for c,s in summaries.items():
        distributions[c]=distribution(s['best']['dev']['records'],rows);lines.append(c+'逐视频/分组：'+json.dumps(distributions[c],ensure_ascii=False))
        gap=s['best']['train']['metrics']['spatial_dice']-s['best']['dev']['metrics']['spatial_dice']
        lines.append(f'{c} best完整train−dev差={gap:+.6f}，曲线monitor仅原4训练pair，不是完整train。实际{s["last_epoch"]}轮/{s["updates"]}更新，成本：'+json.dumps(s['costs'],ensure_ascii=False))
        lines.append(c+'完整fake覆盖/背景：'+json.dumps(lora.recall_metrics(s['best']['dev']),ensure_ascii=False))
    old_ids={r['video_id'] for r in json.loads((Path(config['original_run'])/'manifest.json').read_text())['rows'] if r['split']=='dev'}
    original8={c:distribution([r for r in s['best']['dev']['records'] if r['video_id'] in old_ids],rows) for c,s in summaries.items()}
    lines.append('原8pair开发子集仅历史接续、不用于选模：'+json.dumps(original8,ensure_ascii=False))
    chosen=[r for r in rows if r['split']=='dev' or r.get('train_monitor')];sample_index(out,chosen,conditions=CONDITIONS,title='扩量同队列冻结视觉 vs LoRA：各自best/last')
    boundaries=boundary_samples(out,rows,conditions=CONDITIONS);lines.append('固定25/50/75%边界（原生2px，仅固定样例）和完整背景统计须分开解释：'+json.dumps(boundaries,ensure_ascii=False))
    lines.append('吞吐实测及外推（非正式耗时）：'+json.dumps(probe,ensure_ascii=False))
    storage=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink());lines.append(f'报告前新增独立输出存储={storage}字节；旧缓存直接引用，无新MoGe/CoTracker，无逐轮DINO特征。')
    decision='下一步建议：先在更多独立source上验证当前视觉适配收益的稳定性。' if delta['spatial_dice']>0 else '下一步建议：保留冻结视觉基线，优先复核空间监督覆盖与弱目标，不继续增加当前适配能力。'
    lines.extend([decision,'只作单seed描述，不能把背景误报降低本身写成定位改善；不自动执行后续训练或测试评分。'])
    save(out/'result_summary.json',{'status':'COMPLETE','actual_pairs':assets['actual_pairs'],'conditions':{c:{k:s[k] for k in ['best_epoch','last_epoch','updates','best','last','costs']} for c,s in summaries.items()},
        'reference':reference,'delta':delta,'distributions':distributions,'original8':original8,'boundary':boundaries,'output_bytes':storage,'recommendation':decision})
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')


def references(config,out,rows):
    path=out/'baseline_references.json'
    if path.exists():
        saved=json.loads(path.read_text())
        for name,sha in saved['small_sha256'].items():
            if digest(Path(name))!=sha:raise ValueError('READ_ONLY_BASELINE_CHANGED:'+name)
        for name,value in saved['large_size_mtime'].items():
            st=Path(name).stat()
            if [st.st_size,st.st_mtime_ns]!=value:raise ValueError('READ_ONLY_BASELINE_CHANGED:'+name)
        return
    small=[Path(config['original_run'])/'manifest.json',Path(config['old_lora_run'])/'SPATIAL_LORA_ADAPT/summary.json',Path(config['old_lora_run'])/'report_zh.md']
    large=[Path(config['old_lora_run'])/'SPATIAL_LORA_ADAPT/best.pt']
    for r in rows:
        if Path(config['source_run']).resolve() in Path(r['cache_dir']).resolve().parents:
            small.append(Path(r['cache_dir'])/'complete.json')
            large.extend(Path(r['cache_dir'])/(k+'.npy') for k in ['rgb512','gt','content512'])
            large.append(Path(r['feature_path']))
    save(path,{'small_sha256':{str(p):digest(p) for p in small},'large_size_mtime':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in large},
              'verification':'existing old full acceptance plus current metadata/header and size/mtime; no repeated large-file scans'})


def publish(config,out):
    """Only this experiment's documents; publication status cannot erase experiment completion."""
    repo=Path(__file__).resolve().parents[3]
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    if git('branch','--show-current')!='v8-spatial-supervision-expansion' or git('status','--porcelain'):raise ValueError('PUBLICATION_REQUIRES_CLEAN_TASK_BRANCH')
    r=json.loads((out/'result_summary.json').read_text());doc=repo/'docs/experiments/v8-spatial-supervision-expansion.md'
    text=doc.read_text().replace('RUNNING；','COMPLETE；').replace('PREFLIGHT_COMPLETE；','COMPLETE；')
    marker='\n## 完成结果\n';text=text.split(marker)[0]+marker+'\n'+(out/'report_zh.md').read_text();doc.write_text(text)
    summary=repo/'docs/experiments/v8-spatial-supervision-expansion-results.json'
    small={k:r[k] for k in ['status','actual_pairs','delta','distributions','original8','boundary','output_bytes','recommendation']}
    small['conditions']={c:{'best_epoch':s['best_epoch'],'last_epoch':s['last_epoch'],'updates':s['updates'],'costs':s['costs'],
                           **{ck:{split:s[ck][split]['metrics'] for split in ['train','dev']} for ck in ['best','last']}} for c,s in r['conditions'].items()}
    small['reference']=r['reference'];save(summary,small)
    paths=[doc,summary]
    for c in CONDITIONS:
        curve=repo/f'docs/experiments/v8-spatial-supervision-expansion-{c.lower()}.png';shutil.copyfile(out/c/'curves.png',curve);paths.append(curve)
    status=repo/'docs/v8/current_status.md';text=status.read_text();start=text.index('## 当前执行：空间监督扩量');end=text.index('## 已完成：空间定位LoRA适配',start)
    text=text[:start]+f"## 已完成：空间监督扩量匹配对照\n\n实际pair数{r['actual_pairs']}，两条件仅比较同一队列上的视觉适配。\n开发Dice增强差{r['delta']['spatial_dice']:+.6f}；详见[完成报告](../experiments/v8-spatial-supervision-expansion.md)。\n实验COMPLETE先写入数据目录，Git交付/远端核验单独记录；test_reserved未评分，不追加实验。\n\n"+text[end:];status.write_text(text);paths.append(status)
    subprocess.run(['git','add','--',*[str(p.relative_to(repo)) for p in paths]],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Record completed spatial supervision expansion results'],cwd=repo,check=True)
    head=git('rev-parse','HEAD');save(out/'publication.json',{'status':'COMMITTED','commit':head})
    subprocess.run(['git','push','origin','v8-spatial-supervision-expansion'],cwd=repo,check=True)
    save(out/'publication.json',{'status':'PUSHED','commit':head})
    remote=git('-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/v8-spatial-supervision-expansion').split()[0]
    if remote!=head:raise ValueError('REMOTE_HEAD_MISMATCH')
    save(out/'publication.json',{'status':'COMPLETE','commit':head,'remote_head':remote})


def publish_after_completion(config,out):
    try:publish(config,out)
    except BaseException as e:
        previous=json.loads((out/'publication.json').read_text()) if (out/'publication.json').exists() else {}
        save(out/'publication.json',{**previous,'status':'FAILED','last_completed_publication_stage':previous.get('status'),'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        raise  # Real nonzero exit and error log remain; experiment progress stays COMPLETE.
    else:
        state=json.loads((out/'progress.json').read_text());state['publication_status']='COMPLETE';save(out/'progress.json',state)



def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=['smoke','all'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--publish-results',action='store_true');a=p.parse_args()
    config=json.loads(a.config.read_text());out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);signal.signal(signal.SIGTERM,run.stop);signal.signal(signal.SIGINT,run.stop)
    try:
        if config['seed']!=SEED or config['epochs']!=30:raise ValueError('FROZEN_SEED_AND_BUDGET')
        manifest=json.loads((out/'manifest.json').read_text());rows=manifest['rows']
        if not manifest['training_allowed']:raise ValueError('TRAIN_AND_DEV_MUST_BOTH_EXPAND; ASSETS_ONLY')
        original=json.loads((Path(config['original_run'])/'manifest.json').read_text())['rows'];new={r['video_id']:r for r in rows}
        if any(any(new[r['video_id']][k]!=r[k] for k in ['split','role','video_sha256','source_id','pts','mapping512']) for r in original):raise ValueError('ORIGINAL_ROLES_CHANGED')
        groups={}
        for r in rows:
            if groups.setdefault(r['source_group'],r['split'])!=r['split']:raise ValueError('SOURCE_LEAKAGE')
        references(config,out,rows);run.progress(out,'official_initialization');seed_all(SEED);network=official(config,out).cpu()
        model=MatchedSpatial(copy.deepcopy(network),'LORA_ADAPT').cuda()
        lora.initialization_check(model,{**config,'condition':'LORA_ADAPT'},out,rows,{},video_factory=VisualVideo);del model;torch.cuda.empty_cache()
        run.progress(out,'throughput_probe');probe=benchmark(config,out,rows,network)
        if a.phase=='smoke':save(out/'progress.json',{'status':'SMOKE_COMPLETE','pid':os.getpid()});return
        run.progress(out,'features');features(config,out,rows,network)
        execution=probe['execution'];dev=[r for r in rows if r['split']=='dev'];monitor=[r for r in rows if r.get('train_monitor')]
        if not (out/'epoch0.json').exists():
            model=MatchedSpatial(copy.deepcopy(network),'FROZEN_VISUAL').cuda();cfg={**config,**execution,'condition':'FROZEN_VISUAL'}
            save(out/'epoch0.json',{'monitor':lora.evaluation(model,cfg,out,monitor,{},video_factory=VisualVideo),
                'dev':lora.evaluation(model,cfg,out,dev,{},pixel_ap=True,video_factory=VisualVideo)});del model;torch.cuda.empty_cache()
        for c in CONDITIONS:
            seed_all(SEED);model=MatchedSpatial(copy.deepcopy(network),c,execution['checkpoint_strategy']).cuda();seed_all(SEED)
            lora.train(model,{**config,**execution,'condition':c},out,rows,{},a.resume,condition=c,video_factory=VisualVideo)
            del model;torch.cuda.empty_cache()
        reference=out/'LORA_24_REFERENCE/summary.json';old=Path(config['old_lora_run'])/'SPATIAL_LORA_ADAPT/best.pt'
        if not reference.exists():
            run.progress(out,'old_model_reference');model=MatchedSpatial(copy.deepcopy(network),'LORA_ADAPT').cuda()
            model.restore(torch.load(old,map_location='cpu',weights_only=False)['model'])
            result=lora.evaluation(model,{**config,**execution,'condition':'LORA_ADAPT'},out,dev,{},video_factory=VisualVideo,pixel_ap=True)
            save(reference,{**result,'checkpoint':str(old),'checkpoint_sha256':digest(old),'source_best_epoch':12,'scope':'one frozen inference on new dev; no adaptation'})
            del model;torch.cuda.empty_cache()
        if digest(out/'manifest.json')!=probe['identity']['manifest_sha256']:raise ValueError('FROZEN_MANIFEST_CHANGED')
        run.progress(out,'report');report(config,out,rows);references(config,out,rows)
        save(out/'progress.json',{'status':'COMPLETE','stage':'report_complete','pid':os.getpid(),'unix_time':time.time(),'publication_status':'not_requested' if not a.publish_results else 'pending'})
    except BaseException as e:
        save(out/'progress.json',{'status':'INTERRUPTED' if isinstance(e,InterruptedError) else 'FAILED','pid':os.getpid(),'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()});raise
    if a.publish_results:publish_after_completion(config,out)


if __name__=='__main__':main()
