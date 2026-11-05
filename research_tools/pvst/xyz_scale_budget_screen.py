"""CPU-only three-scale screen; native_dense and original stride8 windows.

Caller this bounded CLI. Outputs counts/report, never routing/model assets.
Existing scale selector also freezes config and reads dev on success; this thin
entry instead stops after training-only evidence and retains raw point weights.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .max_window_check import BASE,ROOT,VIDEO,read,save,sha,git,bounded_process,host
OUT=BASE/'pvst_xyz_scale_budget_screen_v1'
HS=[.25,.5,1.]
BUDGET=(200000,50000)


def occupied(coords,members=None):
    """Exact integer keys, same as probe counter; members count raw valid points."""
    if not len(coords):return coords,np.zeros(0,np.int64)
    lo=coords.min(0);span=coords.max(0)-lo+1
    if int(span[0])*int(span[1])*int(span[2])>=2**62:raise ValueError('COUNT_KEY_OVERFLOW')
    a=coords-lo;key=(a[:,0]*span[1]+a[:,1])*span[2]+a[:,2]
    _,index,inverse,count=np.unique(key,return_index=True,return_inverse=True,return_counts=True)
    if members is not None:
        count=np.zeros(len(index),np.int64);np.add.at(count,inverse,members)
    return coords[index],count


def frame_levels(points,h):
    if not np.isfinite(points).all():raise ValueError('NONFINITE_VALID_POINTS')
    coords,members=occupied(np.floor(points/np.float32(h)).astype(np.int64));result={h:(coords,members)}
    while h<2.:
        h*=2;coords,members=occupied(coords//2,members);result[h]=(coords,members)
    return result


def windows(row,counts):
    prefix=np.vstack((np.zeros((1,2),np.int64),np.cumsum(counts,axis=0)));result=[]
    for start in range(0,row['actual_frames'],8):
        end=min(start+8,row['actual_frames']);left=max(0,start-16);n=prefix[end]-prefix[left]
        result.append({'video_id':row['video_id'],'source_id':row['source_id'],'target_range':[start,end],'history_range':[left,start],'input_range':[left,end],'fine_nodes':int(n[0]),'coarse_nodes':int(n[1])})
    return result


def member_summary(hist):
    n=sum(hist.values());keys=np.array(sorted(hist),np.int64);cum=np.cumsum([hist[int(k)] for k in keys])
    def q(p):
        pos=(n-1)*p;lo=int(np.floor(pos));hi=int(np.ceil(pos));a=keys[np.searchsorted(cum,lo,side='right')];b=keys[np.searchsorted(cum,hi,side='right')]
        return float(a+(b-a)*(pos-lo))
    return {'blocks':n,'median':q(.5) if n else None,'P90':q(.9) if n else None,'max':int(keys[-1]) if n else None,'singleton_fraction':hist.get(1,0)/n if n else None,'valid_point_members':sum(k*v for k,v in hist.items())}


def worker():
    from .data import native_dense
    started=time.monotonic();spec=read(OUT/'identity.json');rows=spec['train_rows'];scale=spec['global_scale'];saved={};hist={h:[Counter(),Counter()] for h in HS};r={'status':'RUNNING','stage':'stage1','stage1':[],'stage2':[],'selected_h':None}
    def progress(**kw):save(OUT/'progress.json',{'status':'RUNNING','stage':r['stage'],'elapsed_s':time.monotonic()-started,'host':host(),**kw})
    def count(row,i,base,validate=False,stats=False):
        q,valid=native_dense(row,i);points=(q/scale)[valid]
        levels=frame_levels(points,base)
        if any(int(m.sum())!=len(points) for _,m in levels.values()):raise ValueError('RAW_POINT_MEMBERS_LOST')
        if validate:
            for h,(coords,members) in levels.items():
                direct,weights=occupied(np.floor(points/np.float32(h)).astype(np.int64))
                if not np.array_equal(coords,direct) or not np.array_equal(members,weights):raise ValueError('DIRECT_NATIVE_MERGE_MISMATCH:'+str((i,h)))
        record={'video_id':row['video_id'],'frame':i,'pts_s':row['pts'][i]['pts_s'],'valid_points':len(points),'levels':{str(h):len(x[0]) for h,x in levels.items()},'direct_merge_checked':validate}
        if stats:
            for h in HS:
                for j,m in enumerate([levels[h][1],levels[h*2][1]]):
                    values,freq=np.unique(m,return_counts=True);hist[h][j].update(dict(zip(values.tolist(),freq.tolist())))
        with (OUT/'frames.jsonl').open('a') as f:f.write(__import__('json').dumps(record)+'\n')
        saved[(row['video_id'],i)]=record;progress(video=row['video_id'],frame=i,completed_unique_frames=len(saved));return record
    try:
        for path,h in spec['assets_sha256'].items():
            if sha(Path(path))!=h:raise ValueError('ASSET_CHANGED:'+path)
        if os.environ['CUDA_VISIBLE_DEVICES']!='':raise ValueError('CUDA_NOT_DISABLED')
        row=next(x for x in rows if x['video_id']==VIDEO)
        for i in range(104,128):count(row,i,.25,validate=i in [104,116,127],stats=True)
        for h in HS:
            values=np.array([[saved[(VIDEO,i)]['levels'][str(h)],saved[(VIDEO,i)]['levels'][str(h*2)]] for i in range(104,128)],np.int64);n=values.sum(0)
            item={'h':h,'coarse_h':h*2,'fine_nodes':int(n[0]),'coarse_nodes':int(n[1]),'valid_points':sum(saved[(VIDEO,i)]['valid_points'] for i in range(104,128)),
                  'per_frame':{k:{'min':int(values[:,j].min()),'median':float(np.median(values[:,j])),'max':int(values[:,j].max())} for j,k in enumerate(['fine','coarse'])},
                  'members':{k:member_summary(hist[h][j]) for j,k in enumerate(['fine','coarse'])},'reduction_vs_1_8':{'fine':1-int(n[0])/2970629,'coarse':1-int(n[1])/1488042},
                  'fine_budget_pass':bool(n[0]<=BUDGET[0]),'coarse_budget_pass':bool(n[1]<=BUDGET[1]),'BF16_single_feature_payload_bytes':2*(256*int(n[0])+512*int(n[1]))}
            item['within_budget']=item['fine_budget_pass'] and item['coarse_budget_pass'];r['stage1'].append(item)
        if r['stage1'][0]['fine_nodes']!=1488042:raise ValueError('NEW_1_4_FINE_DOES_NOT_MATCH_OLD_1_8_COARSE')
        save(OUT/'stage1.json',r['stage1']);passing=[x['h'] for x in r['stage1'] if x['within_budget']]
        if not passing:r.update(status='NO_CANDIDATE',stage='stage1_stop')
        else:
            r['stage']='stage2';all_windows={h:[] for h in passing};complete_videos=0
            for row in rows:
                # Reuse stage1 per-frame counts; every physical frame counted once.
                for i in range(row['actual_frames']):
                    if (row['video_id'],i) not in saved:count(row,i,min(passing))
                for h in passing:
                    counts=[[saved[(row['video_id'],i)]['levels'][str(h)],saved[(row['video_id'],i)]['levels'][str(h*2)]] for i in range(row['actual_frames'])]
                    all_windows[h]+=windows(row,counts)
                complete_videos+=1
                save(OUT/'stage2_partial.json',{'complete_videos':complete_videos,'complete_frames':sum(x['actual_frames'] for x in rows[:complete_videos]),'complete_windows':len(all_windows[passing[0]])})
            for h,ww in all_windows.items():
                values=np.array([[w['fine_nodes'],w['coarse_nodes']] for w in ww]);over=values>np.array(BUDGET)
                item={'h':h,'videos':len(rows),'frames':sum(row['actual_frames'] for row in rows),'windows':len(ww),'fine_over':int(over[:,0].sum()),'coarse_over':int(over[:,1].sum()),'either_over':int(over.any(1).sum()),'within_budget':not bool(over.any())}
                for j,k in enumerate(['fine','coarse']):item[k]={'median':float(np.median(values[:,j])),'P95':float(np.quantile(values[:,j],.95)),'max':int(values[:,j].max()),'max_window':ww[int(values[:,j].argmax())]}
                r['stage2'].append(item);save(OUT/f'windows_h{h}.json',ww)
            good=[x['h'] for x in r['stage2'] if x['within_budget']];r.update(selected_h=min(good) if good else None,status='CPU_BUDGET_CANDIDATE' if good else 'NO_CANDIDATE',stage='count_complete')
    except Exception as e:r.update(status='FAILED',error=str(e),traceback=traceback.format_exc())
    r.update(elapsed_s=time.monotonic()-started,completed_unique_frames=len(saved),host=host());save(OUT/'worker_result.json',r)
    return 1 if r['status']=='FAILED' else 0


def make_identity():
    c=read(ROOT/'docs/pvst/config_v03.json');source=Path(c['asset_root']);manifest=source/'manifest.json';scale_path=source/'scale.json';train=[r for r in read(manifest)['rows'] if r['split']=='train'];scale=read(scale_path)['s']
    matches=[r for r in train if r['video_id']==VIDEO]
    if len(matches)!=1:raise ValueError('FIXED_VIDEO_NOT_UNIQUE_TRAIN')
    row=matches[0]
    if row['source_hw']!=[720,1280] or row['pair_id']!='an:4wdo37' or len(row['pts'])!=row['actual_frames'] or not np.all(np.diff([p['pts_s'] for p in row['pts'][104:128]])>0):raise ValueError('FIXED_IDENTITY_PTS')
    old=BASE/'pvst_point_voxel_v03/coordinate_counts'/f'{VIDEO}.json';anchor=read(old);sig=anchor['identity']
    if sig['manifest']!=sha(manifest) or sig['scale']!=sha(scale_path) or sig['video_sha256']!=row['video_sha256'] or sig['hs']!=[.03125,.0625,.125] or sig['counter']!='v03_exact_native_float32_floor_v1':raise ValueError('REFERENCE_IDENTITY')
    n=np.array(anchor['counts'],np.int64)[104:128,2:4].sum(0)
    if tuple(n)!=(2970629,1488042):raise ValueError('REFERENCE_NODE_MISMATCH')
    paths=[manifest,scale_path,old,ROOT/'docs/pvst/config_v03.json',*Path(__file__).parent.glob('*.py')]
    for row in matches:
        p=Path(row['dense_source']);meta=read(p/'complete.json')
        if meta['video_sha256']!=row['video_sha256']:raise ValueError('DENSE_IDENTITY:'+row['video_id'])
        for name in ['xyz','valid']:
            a=np.load(p/(name+'.npy'),mmap_mode='r')
            if len(a)!=row['actual_frames']:raise ValueError('DENSE_FRAME_COUNT')
        paths.append(p/'complete.json')
    if c['voxel_sizes']['XYZ_PVST'] is not None:raise ValueError('FORMAL_SCALE_CHANGED')
    return {'source_commit':git('rev-parse','HEAD'),'candidates':HS,'budget':BUDGET,'global_scale':scale,'anchor_1_8':n.tolist(),'anchor_source':str(old),'train_rows':train,'videos':len(train),'frames':sum(r['actual_frames'] for r in train),'windows':sum((r['actual_frames']+7)//8 for r in train),'assets_sha256':{str(p):sha(p) for p in paths},'window':'stride8; left=max(0,start-16); end=min(start+8,n); prefix sums; time nodes not spatial union','CUDA_VISIBLE_DEVICES':''}


def report(result,spec,process):
    summary={**result,'execution_HEAD':spec['source_commit'],'global_scale':spec['global_scale'],'reference_1_8':spec['anchor_1_8'],'expected_train':{k:spec[k] for k in ['videos','frames','windows']},'process':process,'CUDA_VISIBLE_DEVICES':'','formal_XYZ':None,'formal_models_completed':0}
    save(OUT/'summary.json',summary);save(ROOT/'docs/pvst/xyz_scale_budget_screen_v1.json',summary)
    lines=['# PVST v0.3 XYZ统一尺度CPU节点预算筛选','',f"**状态：{result['status']}；仅CPU计数，正式XYZ仍null，正式模型0/2。**",'',f"执行HEAD：`{spec['source_commit']}`；全局scale={spec['global_scale']}，单位为归一化坐标单位，不解释为米。",'复用native_dense原生回映、有效性及全局scale；逐帧占据求和，不跨帧空间去重。原生点/像素覆盖不变，没有模型构建、CUDA调用或前端提取。',
        '固定帧104/116/127逐层直接floor与整数归并的坐标及原始点成员数完全比较；负数用floor。成员分位数为24帧全部占据块的精确线性分位数（每块等权），不把一个细块视为一个点。',
        '1/8锚点复用原计数并核对manifest/scale/video SHA和区间：2,970,629/1,488,042。','',
        '## 固定高负载窗口[104,128)','|fine/coarse h|有效点|fine/coarse节点|预算fine/coarse|节点减少fine/coarse|单份BF16特征载荷GiB|','|---|---:|---:|---|---|---:|']
    for x in result.get('stage1',[]):lines.append(f"|{x['h']}/{x['coarse_h']}|{x['valid_points']}|{x['fine_nodes']}/{x['coarse_nodes']}|{x['fine_budget_pass']}/{x['coarse_budget_pass']}|{x['reduction_vs_1_8']['fine']:.2%}/{x['reduction_vs_1_8']['coarse']:.2%}|{x['BF16_single_feature_payload_bytes']/2**30:.4f}|")
    lines+=['','载荷仅为2×(256×Nfine+512×Ncoarse)字节，两尺度各一份BF16特征，不是训练峰值；本轮不测边数或推测完整显存。','']
    for x in result.get('stage1',[]):lines+=['h='+str(x['h'])+'逐帧节点min/median/max：'+str(x['per_frame'])+'；块内原始点median/P90/max/singleton：'+str(x['members'])]
    lines+=['','## 训练队列确认',f"原队列核对：{spec['videos']}训练视频/{spec['frames']}帧/{spec['windows']}候选窗口；窗口含原历史边界与尾段，不只取选中训练片段。"]
    if result.get('stage2'):
        for x in result['stage2']:lines+=['```json',__import__('json').dumps(x,ensure_ascii=False,indent=2),'```']
    elif result['status'] in ['TIMEOUT','FAILED']:lines+=['计数未完成/错误，不可判定全队列预算通过或失败；已完成范围：'+str(result.get('last_progress',{}))+'；完整视频范围：'+str(result.get('stage2_partial',{}))]
    else:lines+=['阶段一无合格候选，按规则停止，阶段二未执行。']
    if result.get('stage1') and not any(x['within_budget'] for x in result['stage1']):
        close=min(result['stage1'],key=lambda x:max(x['fine_nodes']/BUDGET[0],x['coarse_nodes']/BUDGET[1]));lines+=['最接近预算h='+str(close['h'])+'，fine/coarse分别为预算的'+str(close['fine_nodes']/BUDGET[0])+'/'+str(close['coarse_nodes']/BUDGET[1])+'倍；不得因均值较低而放行。']
    lines+=[f"推荐CPU节点预算合格候选：{result.get('selected_h')}。这不是冻结正式尺度或4090训练通过。",'统一更粗尺度会降低结构空间分辨率；边长相对1/8分别扩大2/4/8倍，全部点旁路和128/256/512宽度保持。',f"计数worker时间{result.get('elapsed_s')}秒；父进程墙时{process['wall_s']}秒；真实退出码{process['returncode']}，终止信号{process['termination_signal']}。",'产物：`'+str(OUT)+'`；逐帧/逐窗口JSON留derived，不提交大数组。未读开发/测试坐标或标签，不改旧结果/config/ADR/节点预算。',str(result.get('error',''))]
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text);(ROOT/'docs/pvst/xyz_scale_budget_screen_v1.md').write_text(text)
    p=ROOT/'docs/pvst/current_status.md';p.write_text(p.read_text()+f"\nXYZ扩展尺度CPU筛选：{result['status']}，候选{result.get('selected_h')}；正式XYZ仍null、模型0/2。见[报告](xyz_scale_budget_screen_v1.md)。\n")


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:raise SystemExit(worker())
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'identity.json').exists():raise RuntimeError('EXISTING_RUN: verify and reuse, no retry')
    spec=make_identity();save(OUT/'identity.json',spec)
    process=bounded_process([sys.executable,'-u','-m',__spec__.name,'--worker'],1800,OUT/'worker.log');save(OUT/'process.json',process)
    result=read(OUT/'worker_result.json') if (OUT/'worker_result.json').exists() else {'status':'TIMEOUT' if process['timed_out'] else 'FAILED','stage1':read(OUT/'stage1.json') if (OUT/'stage1.json').exists() else [],'last_progress':read(OUT/'progress.json') if (OUT/'progress.json').exists() else {},'stage2_partial':read(OUT/'stage2_partial.json') if (OUT/'stage2_partial.json').exists() else {}}
    if process['timed_out']:result['status']='TIMEOUT'
    elif process['returncode']!=0 and result['status']!='FAILED':result['status']='FAILED'
    report(result,spec,process);save(OUT/'progress_final.json',{'status':result['status'],'stage':'report_complete'})
    import subprocess
    try:
        subprocess.run(['git','add','--','docs/pvst/xyz_scale_budget_screen_v1.md','docs/pvst/xyz_scale_budget_screen_v1.json','docs/pvst/current_status.md'],check=True);subprocess.run(['git','commit','-m','Record CPU XYZ scale budget screening'],check=True)
        h=git('rev-parse','HEAD');save(OUT/'publication.json',{'status':'COMMITTED','commit':h,'calculation_status':result['status']});subprocess.run(['git','push','origin','pvst-point-voxel-v03'],check=True)
        remote=git('ls-remote',git('remote','get-url','--push','origin'),'refs/heads/pvst-point-voxel-v03').split()[0]
        if remote!=h:raise ValueError('REMOTE_HEAD_MISMATCH')
        save(OUT/'publication.json',{'status':'COMPLETE','commit':h,'remote_head':remote,'calculation_status':result['status']})
    except Exception:save(OUT/'publication.json',{'status':'FAILED','calculation_status':result['status'],'error':traceback.format_exc()})

if __name__=='__main__':main()
