"""Formal-only runner after the frozen cache/resource PASS; no extraction/probe.

Uses existing make/update/evaluate. Adds only asset gate, history preservation,
matched order, per-epoch timing and finalization for this two-condition experiment.
"""
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import torch
from . import slab_grid_experiment as experiment
from .grid_localization import CurrentReadout,CONDITIONS
from .data import save
from .experiment import REPO,git,state_hash
from scripts.prepare_v8_assets import digest

OUT=experiment.OUT
PASS=experiment.ROOT/'pvst_slab_grid_cache_resume_check_v1'
DOC=REPO/'docs/pvst/slab_grid_formal_results_v1.md'
BASE_HEAD='12cde417b51a9e777b9979a222a1d48743e173f2'


def read(p):return json.loads(Path(p).read_text())
def progress(stage,**extra):
    save(OUT/'formal_status.json',{'status':'RUNNING','pid':os.getpid(),'stage':stage,**extra});print(stage,json.dumps(extra),flush=True)


def assert_gate(passed):
    assert passed['status']=='COMPLETE' and passed['cache_status']=='PASS' and passed['resource_status']=='PASS'
    check=passed['resource_check'];assert check['passed'] and check['discarded_update'] and check['parameter_changed'] and check['optimizer_states']>0
    assert passed['extraction']['frames']==2076 and passed['extraction']['train_frames']==1152 and passed['extraction']['dev_frames']==924
    assert check['dev_max_frame']['SLAB_GRID_CURRENT_units']<=check['train_max_frame']['SLAB_GRID_CURRENT_units'] or 'larger_dev_forward_seconds' in check


def archive_first(root):
    """Preserve exact old top-level products before current report/summary evolve."""
    archive=root/'history'/'first_failed_execution';archive.mkdir(parents=True,exist_ok=False)
    hashes={}
    for src in root.iterdir():
        if src.is_file():shutil.copy2(src,archive/src.name);hashes[src.name]=digest(src);assert digest(archive/src.name)==hashes[src.name]
    save(archive/'preservation_manifest.json',hashes)
    return {'path':str(archive),'sha256':hashes}


def assert_same_frames(a,b,expected):
    names=['video_id','frame_index','pts','valid_pixels','valid_GT_pixels'];aa=[[r[k] for k in names] for r in a['frame_records']];bb=[[r[k] for k in names] for r in b['frame_records']]
    assert a['frames']==b['frames']==expected and aa==bb and len({(r[0],r[1]) for r in aa})==expected
    assert [(r['video_id'],r['GT_pixels'],r['valid_GT_pixels'],r['GT_coverage']) for r in a['records']]==[(r['video_id'],r['GT_pixels'],r['valid_GT_pixels'],r['GT_coverage']) for r in b['records']]


def render(summary):
    results=summary.get('results',{});lines=['# 整片与深度片内固定方格：正式匹配训练',f"本次正式计算{summary['status']}，退出码{summary['exit_code']}。缓存提取和资源验收复用上轮PASS，没有重复运行。"]
    if len(results)==2:
        for split,label in [('train','原48段训练回测'),('dev','原4对完整开发视频')]:
            rows=[[k]+[results[k][split]['macro'][m] for m in ['pixel_AP','Dice','IoU','GT_recall','real_FPR']] for k in CONDITIONS]
            rows.append(['细分−整片']+[rows[1][i]-rows[0][i] for i in range(1,6)])
            lines += [label,experiment.table(['条件','pixel AP','Dice','IoU','GT召回','real FPR'],rows)]
        lines += [experiment.table(['pair','整片AP','细分AP','ΔAP','整片Dice','细分Dice','ΔDice'],[[r[k] for k in ['pair_id','SLAB_AP','GRID_AP','AP_delta','SLAB_Dice','GRID_Dice','Dice_delta']] for r in summary['comparison']])]
        count=sum(r['AP_delta']>0 for r in summary['comparison']);delta=results[CONDITIONS[1]]['dev']['macro']['pixel_AP']-results[CONDITIONS[0]]['dev']['macro']['pixel_AP']
        lines += [f'开发AP改善{count}/4，宏平均变化{delta:+.6f}。'+('本次支持相同深度归片下的空间细分贡献。' if count>=2 and delta>0 else '本次没有多视频一致的开发定位收益。')]
    lines += ['原GT oracle（独立上限，非模型成绩）：整片Dice 0.446189，细分0.942438；不得把上限增量当作训练增量。','两组同512维当前帧读出、初始化、原12轮顺序、像素计数BCE和预算；唯一变量是分组粒度。不是历史或三维优于二维的证据。',f"执行HEAD {summary['identity']['execution_head']}；缓存来源/资源凭据/旧失败归档见formal_identity.json和history/first_failed_execution/。",'状态formal_status.json、真实退出码formal_exit_code.txt、日志formal_execution.log；首次status.json/exit_code.txt=1/failure.json仍保留其原历史语义。']
    report='\n\n'.join(lines)+'\n';(OUT/'report.md').write_text(report);DOC.write_text(report)


def main():
    started=time.monotonic();torch.set_num_threads(4);assert git('branch','--show-current')=='pvst-slab-grid-localization'
    passed=read(PASS/'summary.json');assert_gate(passed)
    if (OUT/'formal_status.json').exists():
        s=read(OUT/'summary.json')
        if s.get('status')=='COMPLETE' and s.get('formal_training') and s['identity']['runner_sha256']==digest(REPO/'research_tools/rgbd_slice/slab_grid_train.py'):print('REUSE FORMAL COMPLETE');return
        raise ValueError('FORMAL_RUNNING_OR_PARTIAL_NO_DUPLICATE_OR_RETRY')
    assert not any((OUT/k).exists() for k in CONDITIONS)
    assert git('merge-base',BASE_HEAD,'HEAD')==BASE_HEAD
    assert not git('status','--porcelain','--','research_tools/rgbd_slice')
    archive=archive_first(OUT);identity={'execution_head':git('rev-parse','HEAD'),'baseline_head':BASE_HEAD,'runner_sha256':digest(REPO/'research_tools/rgbd_slice/slab_grid_train.py'),'pass_summary_sha256':digest(PASS/'summary.json'),'pass_identity':passed['identity'],'first_failure_archive':archive,'seed':20260909,'epochs':12,'updates_per_condition':576,'optimizer':{'AdamW_lr':1e-4,'weight_decay':.01},'query_chunk':512,'precision':'unchanged FP32 efficient SDPA; frozen C from original BF16 source','model_code_sha256':digest(REPO/'research_tools/rgbd_slice/grid_localization.py'),'update_eval_source_sha256':digest(REPO/'research_tools/rgbd_slice/slab_grid_experiment.py')}
    save(OUT/'formal_identity.json',identity);progress('formal_precheck')
    summary={'formal_training':True,'status':'FAILED','exit_code':1,'identity':identity,'results':{}};results=summary['results'];history=[]
    try:
        assert identity['update_eval_source_sha256']==passed['identity']['extract_source_sha256']
        m=read(experiment.BASE/'manifest.json');config=read(experiment.BASE/'config.json');orders=read(experiment.BASE/'orders.json');old=read(OUT/'identity.json');baseline=read(experiment.BASE/'identity.json')
        assert config['seed']==20260909 and config['epochs']==12
        for name,key in [('manifest.json','manifest_sha256'),('config.json','config_sha256'),('orders.json','orders_sha256')]:assert digest(experiment.BASE/name)==old[key]
        assert digest(experiment.BASE/'manifest.json')==passed['identity']['manifest_sha256'] and digest(experiment.BASE/'config.json')==passed['identity']['config_sha256']
        weight=experiment.BASE/'RGB_DEPTH_PARTITION/last.pt';assert digest(weight)==passed['identity']['weight_sha256']==old['weight_sha256']
        cp=torch.load(weight,map_location='cpu',weights_only=True);assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==baseline['identity'];del cp
        assert identity['model_code_sha256']==old['source_code']['research_tools/rgbd_slice/grid_localization.py']
        files=passed['extraction']['files'];assert {p.name for p in (OUT/'cache').glob('*.pt')}==set(files)
        for name,sha in files.items():assert digest(OUT/'cache'/name)==sha
        rows=m['rows'];train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];clips=m['clips'];assert len(train)==16 and len(dev)==8 and len(clips)==48
        canonical=lambda x:sorted((r['video_id'],r['start'],r['end']) for r in x)
        for epoch in range(1,13):assert canonical(orders[str(epoch)])==canonical(clips) and all(r['end']-r['start']==24 for r in orders[str(epoch)])
        cache=experiment.load_cache(rows)
        for row in rows:
            expected=sorted({i for lo,hi,_ in experiment.windows(row,clips if row['split']=='train' else None) for i in range(lo,hi)});assert sorted(cache[row['video_id']])==expected
            for i in expected:
                f=cache[row['video_id']][i];assert f['pts']==row['pts'][i]['pts_s']
                for condition in CONDITIONS:
                    q=f[condition];assert q['features'].shape==(len(q['ids']),512) and q['features'].dtype==torch.float32 and int(q['counts'].sum())==f['valid_pixels'] and int(q['edit_counts'].sum())==f['valid_GT_pixels']
        stat=list(csv.DictReader((PASS/'cache_frames.csv').open()));tr=[r for r in stat if r['split']=='train'];assert len(tr)==1152 and len(stat)==2076
        P=sum(int(r['valid_GT_pixels']) for r in tr);N=sum(int(r['valid_pixels']) for r in tr);pw=torch.tensor((N-P)/P,device='cuda');assert float(pw)==read(PASS/'supervision.json')['w_pos']
        torch.manual_seed(20260909);a=CurrentReadout();initial={k:v.clone() for k,v in a.state_dict().items()};initial_hash=state_hash(initial);params=sum(p.numel() for p in a.parameters());del a
        assert initial_hash==passed['resource_check']['initial_sha256']
        save(OUT/'formal_initialization.json',{'state_dict_sha256':initial_hash,'parameters':params,'newly_seeded':True,'old_readout_or_resource_weights_loaded':False})
        save(OUT/'formal_config.json',{'seed':20260909,'epochs':12,'updates_per_condition':576,'orders_sha256':digest(experiment.BASE/'orders.json'),'manifest_sha256':digest(experiment.BASE/'manifest.json'),'cache_sha256':files,'w_pos':float(pw),'lr':1e-4,'weight_decay':.01,'initial_sha256':initial_hash,'whole_current_frame_candidates':True,'GT_only_for_supervision':True})
        for condition in CONDITIONS:
            a=experiment.make(initial);assert state_hash(a.state_dict())==initial_hash;opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01);updates=0;start=time.monotonic();train_s=eval_s=0.;torch.cuda.reset_peak_memory_stats();folder=OUT/condition;folder.mkdir();first_epoch_seconds=None
            with (folder/'updates.jsonl').open('w') as log:
                for epoch in range(1,13):
                    losses=[];epoch_start=time.monotonic();epoch_train=0.
                    for clip in orders[str(epoch)]:
                        record=experiment.update(a,opt,cache,clip,condition,pw);updates+=1;losses.append(record['loss']);train_s+=record['seconds'];epoch_train+=record['seconds'];record.update(epoch=epoch,update=updates,clip=clip);log.write(json.dumps(record)+'\n');log.flush()
                    evaluated=experiment.evaluate(a,cache,dev,condition);assert evaluated['frames']==924;eval_s+=evaluated['seconds'];save(folder/f'epoch{epoch:02d}_dev.json',evaluated)
                    seconds=time.monotonic()-epoch_start
                    if first_epoch_seconds is None:first_epoch_seconds=seconds
                    h={'condition':condition,'epoch':epoch,'updates':updates,'train_loss':experiment.mean(losses),'dev_pixel_AP':evaluated['macro']['pixel_AP'],'dev_Dice':evaluated['macro']['Dice'],'epoch_train_seconds':epoch_train,'epoch_dev_seconds':evaluated['seconds'],'epoch_seconds':seconds,'elapsed_seconds':time.monotonic()-start,'remaining_estimate_from_first_epoch_seconds':(12-epoch)*first_epoch_seconds,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()};history.append(h);experiment.csvsave(OUT/'history.csv',history);progress('formal_train',**h)
            assert updates==576
            torch.save({'condition':condition,'updates':updates,'epoch':12,'initial_sha256':initial_hash,'identity':identity,'state_dict':{k:v.cpu() for k,v in a.state_dict().items()}},folder/'final.pt')
            training=experiment.evaluate(a,cache,train,condition,clips);assert training['frames']==1152;save(folder/'final_train.json',training);save(folder/'final_dev.json',evaluated)
            for split,data in [('train',training),('dev',evaluated)]:experiment.csvsave(folder/f'{split}_videos.csv',data['records']);experiment.csvsave(folder/f'{split}_frames.csv',data['frame_records'])
            results[condition]={'status':'COMPLETE','updates':updates,'initial_sha256':initial_hash,'train':training,'dev':evaluated,'seconds':time.monotonic()-start,'train_update_seconds':train_s,'epoch_dev_eval_seconds':eval_s,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'first_epoch_seconds':first_epoch_seconds};save(OUT/'formal_partial_results.json',results);del a,opt;torch.cuda.empty_cache()
        for split,frames in [('train',1152),('dev',924)]:assert_same_frames(results[CONDITIONS[0]][split],results[CONDITIONS[1]][split],frames)
        comparison=[]
        for x,y in zip(results[CONDITIONS[0]]['dev']['records'],results[CONDITIONS[1]]['dev']['records']):
            if x['role']=='fake':comparison.append({'pair_id':x['pair_id'],'video_id':x['video_id'],'SLAB_AP':x['pixel_AP'],'GRID_AP':y['pixel_AP'],'AP_delta':y['pixel_AP']-x['pixel_AP'],'SLAB_Dice':x['Dice'],'GRID_Dice':y['Dice'],'Dice_delta':y['Dice']-x['Dice'],'GT_coverage':x['GT_coverage']})
        experiment.csvsave(OUT/'dev_video_comparison.csv',comparison);summary.update(status='COMPLETE',exit_code=0,comparison=comparison,GT_coverage_and_frame_identity_same=True,resource_check_reused=True,additional_resource_updates=0,additional_frontend_or_cache_extraction=0)
    except BaseException:
        summary['error']=traceback.format_exc();summary['last_stage']=read(OUT/'formal_status.json');save(OUT/'formal_failure.json',summary);print(summary['error'],flush=True)
    summary['wall_seconds']=time.monotonic()-started;save(OUT/'summary.json',summary);render(summary)
    if summary['status']=='COMPLETE':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for k in CONDITIONS:
            rr=[r for r in history if r['condition']==k];ax.plot([r['updates'] for r in rr],[r['dev_pixel_AP'] for r in rr],label=k)
        ax.set(xlabel='Updates',ylabel='Dev valid-pixel AP');ax.legend();fig.tight_layout();fig.savefig(OUT/'dev_pixel_AP.png');plt.close(fig)
    save(OUT/'formal_status.json',{'status':summary['status'],'stage':'report_complete','exit_code':summary['exit_code']})
    if summary['exit_code']:raise SystemExit(summary['exit_code'])


if __name__=='__main__':main()
