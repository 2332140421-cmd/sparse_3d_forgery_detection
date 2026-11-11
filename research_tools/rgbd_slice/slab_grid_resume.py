"""Resume only original extraction, then ONE resource update; never train arms.

Caller: CLI. Reuses original extract/load_cache/make/update, with separate state.
No model, pool, precision, tolerance or training-default changes.
"""
import json
import os
from pathlib import Path
import resource
import time
import traceback

import torch
from . import slab_grid_experiment as original
from .data import save
from .experiment import git,state_hash,REPO
from .set_fit_check import build
from .grid_localization import CurrentReadout,CONDITIONS
from scripts.prepare_v8_assets import digest

OUT=original.ROOT/'pvst_slab_grid_cache_resume_check_v1'
DOC=REPO/'docs/pvst/slab_grid_cache_resume_check_v1.md'


def read(p):return json.loads(Path(p).read_text())
def progress(stage,**extra):
    save(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'stage':stage,**extra});print(stage,json.dumps(extra),flush=True)
def preserved():
    paths=[p for p in original.OUT.iterdir() if p.is_file()]
    paths += [p for p in (original.ROOT/'pvst_slab_grid_pooling_parity_check_v1').rglob('*') if p.is_file()]
    return {str(p):digest(p) for p in paths}


def report(s):
    lines=['# 整片／方格原入口缓存续跑与单次资源验收',f"本次状态：{s['status']}，退出码{s['exit_code']}；正式训练更新0，模型完成0/2。",f"缓存完整原核验：{s['cache_status']}；资源验收：{s['resource_status']}。",f"复用已完整核验缓存{s.get('reused_frames',0)}帧；新增完整缓存{s.get('added_frames',0)}帧。"]
    if (OUT/'same_call_ABC.json').exists():
        e=read(OUT/'same_call_ABC.json');f=e['identity'];rows=[[k,x['mismatches'],x['max_absolute'],x['max_absolute_element']] for k,x in e['comparisons'].items()]
        lines += [f"再次超差：{f['pair_id']}/{f['video_id']}，原生帧{f['frame_index']}，PTS={f['pts']:.9f}s，形状{f['B_shape']}，B/C={f['B_dtype']}/{f['C_dtype']}。",original.table(['比较','超差元素','最大绝对差','最坏片号/通道/允许误差'],rows),'A由当次尚在内存的同一local计算；额外编码0次。B/C先保存，随后保存A/B/C和三方误差；原断言未改，不自动重试。']
    elif (OUT/'original_mismatch_error.txt').exists():lines += ['原超差已先保存，补充取证状态见evidence_recording_error.json；不能用记录错误覆盖原始超差。']
    if s.get('resource_check'):
        c=s['resource_check'];lines += ['资源验收：'+json.dumps(c,ensure_ascii=False),'验收权重丢弃，不写正式checkpoint。']
    if s.get('error'):lines+=['错误见本次failure.json及execution.log。']
    lines += [f"总墙钟{s['seconds']:.3f}秒；GPU峰值allocated/reserved {s['peak_allocated']/1024**3:.3f}/{s['peak_reserved']/1024**3:.3f}GiB；进程峰值RSS {s['peak_RSS']/1024**3:.3f}GiB。",'首次失败日志、退出码1及三方诊断保持原样，旧1008帧缓存内容SHA前后相同。新增完整视频缓存写原cache目录，旧文件不覆盖。诊断的单独72帧没有作为正式缓存。',f"执行HEAD {s['identity']['execution_head']}；产物 {OUT}。计算/发布分开。本轮不启动576步训练。"]
    text='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(text);DOC.write_text(text)


def main():
    started=time.monotonic();torch.set_num_threads(4);assert git('branch','--show-current')=='pvst-slab-grid-localization'
    if OUT.exists():
        old=read(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
        if old.get('status')=='COMPLETE' and old['identity']['source_sha256']==digest(REPO/'research_tools/rgbd_slice/slab_grid_resume.py') and old['identity']['extract_source_sha256']==digest(REPO/'research_tools/rgbd_slice/slab_grid_experiment.py'):print('REUSE COMPLETE');return
        raise ValueError('EXISTING_RUN_INSPECT_NO_RETRY')
    OUT.mkdir();identity={'execution_head':git('rev-parse','HEAD'),'baseline_head':'726dabccb832fa77c50d88408b0ab2ba51b8a842','scope':'one cache continuation then one resource update; no formal training','source_sha256':digest(REPO/'research_tools/rgbd_slice/slab_grid_resume.py'),'extract_source_sha256':digest(REPO/'research_tools/rgbd_slice/slab_grid_experiment.py')};save(OUT/'identity.json',identity);progress('verify_assets')
    keep=preserved();save(OUT/'original_preservation.json',keep);cache_before={str(p):digest(p) for p in (original.OUT/'cache').glob('*.pt')};save(OUT/'original_cache_sha256.json',cache_before)
    summary={'status':'FAILED','exit_code':1,'identity':identity,'cache_status':'NOT_COMPLETE','resource_status':'NOT_STARTED','formal_updates':0};teacher=None;before=None;resource_updates=0
    try:
        config=read(original.BASE/'config.json');m=read(original.BASE/'manifest.json');base=read(original.BASE/'identity.json');oldid=read(original.OUT/'identity.json');rows=m['rows'];train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];clips=m['clips']
        assert {Path(p).stem for p in cache_before}=={r['video_id'] for r in train[:14]}
        assert len(cache_before)==14 and len(train)==16 and len(dev)==8 and len(clips)==48 and sum(r['actual_frames'] for r in dev)==924
        for file,key in [('manifest.json','frozen_manifest_sha256'),('config.json','config_sha256')]:assert digest(original.BASE/file)==base[key]
        assert digest(original.BASE/'manifest.json')==oldid['manifest_sha256'] and digest(original.BASE/'config.json')==oldid['config_sha256']
        weight=original.BASE/'RGB_DEPTH_PARTITION/last.pt';assert digest(weight)==oldid['weight_sha256']
        identity.update(weight_sha256=digest(weight),manifest_sha256=digest(original.BASE/'manifest.json'),config_sha256=digest(original.BASE/'config.json'));save(OUT/'identity.json',identity)
        assert digest(REPO/'research_tools/rgbd_slice/grid_localization.py')==oldid['source_code']['research_tools/rgbd_slice/grid_localization.py']
        for path,sha in base['code'].items():assert digest(REPO/path)==sha
        for r in rows:
            assert digest(Path(r['video_path']))==r['video_sha256']
            if r['role']=='fake':assert digest(Path(r['mask_zip']))==r['mask_sha256']
            for path,sha in next(a for a in base['assets'] if a['video_id']==r['video_id'])['cache_sha256'].items():assert digest(Path(path))==sha
            ref=original.REF/'cache'/f"{r['video_id']}.pt";assert digest(ref)==read(original.REF/'cache_identity.json')['files'][ref.name]
        teacher=build(config,OUT,rgb_only=True).eval();cp=torch.load(weight,map_location='cpu',weights_only=True)
        assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==base['identity'];assert read(OUT/'initialization.json')['model_sha256']==cp['initial_model_sha256'];teacher.load_state_dict(cp['state_dict'],strict=True);del cp
        teacher.requires_grad_(False);before=state_hash(teacher.state_dict());torch.cuda.reset_peak_memory_stats()
        stats,extraction=original.extract(teacher,train+dev,clips,config,resume_run=OUT)
        assert extraction['train_frames']==1152 and extraction['dev_frames']==924 and extraction['frames']==2076
        assert sum(x['frames'] for x in extraction['reused'])==1008
        assert state_hash(teacher.state_dict())==before and all(p.grad is None for p in teacher.parameters());del teacher;teacher=None;torch.cuda.empty_cache();summary['cache_status']='PASS'
        cache=original.load_cache(rows);tr=[r for r in stats if r['split']=='train'];P=sum(r['valid_GT_pixels'] for r in tr);N=sum(r['valid_pixels'] for r in tr);assert 0<P<N
        pw=torch.tensor((N-P)/P,device='cuda');save(OUT/'supervision.json',{'train_valid_pixels':N,'train_valid_GT_pixels':P,'w_pos':float(pw)})
        maximum=max(tr,key=lambda r:r['SLAB_GRID_CURRENT_units']);clip=next(c for c in clips if c['video_id']==maximum['video_id'] and c['start']<=maximum['frame_index']<c['end']);devmax=max((r for r in stats if r['split']=='dev'),key=lambda r:r['SLAB_GRID_CURRENT_units'])
        torch.manual_seed(original.SEED);a=CurrentReadout();initial={k:v.clone() for k,v in a.state_dict().items()};initial_hash=state_hash(initial);del a
        free,total=torch.cuda.mem_get_info();progress('resource_check',clip=clip,train_max_frame=maximum,dev_max_frame=devmax,starting_free=free)
        a=original.make(initial);assert state_hash(a.state_dict())==initial_hash;opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01);torch.cuda.reset_peak_memory_stats()
        check=original.update(a,opt,cache,clip,CONDITIONS[1],pw);resource_updates=1
        check.update(clip=clip,train_max_frame=maximum,dev_max_frame=devmax,starting_free=free,GPU_total=total,GPU_name=torch.cuda.get_device_name(),initial_sha256=initial_hash,query_chunk=512,passed=True,discarded_update=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        save(OUT/'resource_update.json',check)
        if devmax['SLAB_GRID_CURRENT_units']>maximum['SLAB_GRID_CURRENT_units']:
            q=cache[devmax['video_id']][devmax['frame_index']][CONDITIONS[1]];a.eval();t=time.monotonic()
            with torch.no_grad():z=a(q['features'].cuda(),q['ids'].cuda());assert torch.isfinite(z).all()
            torch.cuda.synchronize();check['larger_dev_forward_seconds']=time.monotonic()-t;del z
        check['peak_allocated']=torch.cuda.max_memory_allocated();check['peak_reserved']=torch.cuda.max_memory_reserved();save(OUT/'resource_check.json',check);del a,opt;torch.cuda.empty_cache()
        summary.update(status='COMPLETE',exit_code=0,resource_status='PASS',resource_check=check,extraction=extraction)
    except BaseException:
        summary['error']=traceback.format_exc();summary['last_stage']=read(OUT/'status.json');save(OUT/'failure.json',summary);print(summary['error'],flush=True)
    if teacher is not None and before is not None:summary['frozen_state_unchanged']=state_hash(teacher.state_dict())==before
    summary.update(resource_updates=resource_updates,seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    for key,file in [('reused_frames','reused_cache.json'),('added_frames','added_cache.json')]:summary[key]=sum(x['frames'] for x in read(OUT/file)) if (OUT/file).exists() else 0
    assert preserved()==keep and all(digest(Path(p))==sha for p,sha in cache_before.items())
    summary['original_logs_diagnosis_and_cache_unchanged']=True;save(OUT/'summary.json',summary);report(summary);save(OUT/'status.json',{'status':summary['status'],'stage':'report_complete','exit_code':summary['exit_code']})
    if summary['exit_code']:raise SystemExit(summary['exit_code'])


if __name__=='__main__':main()
