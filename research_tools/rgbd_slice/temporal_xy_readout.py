"""CPU-only fixed image-grid alignment of seven existing frozen C caches.

CLI outputs one two-time table. Count-weighted collapse across depth is required
because a native XY tile can contain several depth slabs; no new feature model.
"""
import csv
import json
import os
from pathlib import Path
import time
import torch
from .synthetic_temporal_readout import aligned_history, OUT as SOURCE
from .experiment import git
from .data import save
from scripts.prepare_v8_assets import digest

OUT=SOURCE.parent/'pvst_synthetic_temporal_xy_readout_v1'


def collapse_xy(e):
    keys=e['keys'];x=e['features'].double();n=e['counts'].double()
    assert e['L']==23 and keys.shape==(len(x),3) and x.shape[1]==512
    assert torch.isfinite(x).all() and (n>0).all()
    xy,index=torch.unique(keys[:,1:],dim=0,sorted=True,return_inverse=True)
    counts=n.new_zeros(len(xy)).index_add_(0,index,n)
    sums=x.new_zeros(len(xy),512).index_add_(0,index,x*n[:,None])
    assert int(counts.sum())==int(n.sum())==e['valid_pixels']
    return {'features':sums/counts[:,None],'keys':xy,'counts':counts,'L':23,'source_frame':e['source_frame']}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4);start=time.monotonic()
    paths=[SOURCE/f'features_{i:02d}.pt' for i in range(7)]
    hashes={p.name:digest(p) for p in paths}
    identity={'execution_head':git('rev-parse','HEAD'),'source':str(SOURCE),'features_sha256':hashes,'source_summary_sha256':digest(SOURCE/'summary.json'),'collapse':'sum(valid_pixel_count * C) / sum(valid_pixel_count) within exact cached (row,col); CPU FP64','distance':'original mean cosine distance per adjacent frame, summed over three transitions','L':23}
    if OUT.exists():
        old=json.loads((OUT/'summary.json').read_text())
        assert old['status']=='COMPLETE' and old['identity']['features_sha256']==hashes and old['identity']['collapse']==identity['collapse']
        print('REUSE COMPLETE');return
    OUT.mkdir();save(OUT/'identity.json',identity)
    entries={};frames=[]
    for i,p in enumerate(paths):
        e=torch.load(p,map_location='cpu',weights_only=True)
        assert e['source_frame']==i and len(torch.unique(e['keys'],dim=0))==len(e['keys'])
        entries[i]=collapse_xy(e)
        frames.append({'frame':i,'original_depth_grid_units':len(e['keys']),'xy_units':len(entries[i]['keys']),'valid_pixels':e['valid_pixels']})
    rows=[]
    for t,a,b in [(3,[0,1,2,3],[0,1,2,3]),(6,[3,4,5,6],[3,5,4,6])]:
        ordered=aligned_history(entries,a);perturbed=aligned_history(entries,b)
        delta=perturbed['total']-ordered['total'] if ordered['total'] is not None and perturbed['total'] is not None else None
        if t==3:assert delta==0
        rows.append({'t':t,'H':3,'ordered':ordered,'perturbed':perturbed,'delta':delta})
    assert hashes=={p.name:digest(p) for p in paths}
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'frames':frames,'rows':rows,'seconds':time.monotonic()-start,'new_depth_frames':0,'new_features':0,'model_calls':0,'source_unchanged':True}
    save(OUT/'summary.json',summary)
    with (OUT/'results.csv').open('w') as f:
        fields=['t','H','common_keys','ordered_distances','ordered_total','perturbed_distances','perturbed_total','delta'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({'t':r['t'],'H':3,'common_keys':r['ordered']['common_keys'],'ordered_distances':r['ordered']['distances'],'ordered_total':r['ordered']['total'],'perturbed_distances':r['perturbed']['distances'],'perturbed_total':r['perturbed']['total'],'delta':r['delta']})
    report=['# 固定二维格对齐的冻结时序读出','仅CPU读取features_00.pt至features_06.pt。原L=23及缓存格行/格列不变。同XY格多个深度片按有效像素数加权平均C，不重新池化像素；该口径是本轮固定的诊断合并，不是纯二维编码器。FP64仅用于诊断归约。','四帧共有XY格上剔除任一帧零范数，逐相邻帧平均余弦距离，三项相加。相同窗口两版共用同一键集合。','|t|H|共有格|顺序三项|顺序总分|扰动三项|扰动总分|扰动−顺序|','|---|---|---|---|---|---|---|---|']
    for r in rows:report.append(f"|{r['t']}|3|{r['ordered']['common_keys']}|{r['ordered']['distances']}|{r['ordered']['total']}|{r['perturbed']['distances']}|{r['perturbed']['total']}|{r['delta']}|")
    report+=['\n## 覆盖及排除（逐窗口、逐源帧）']
    for r in rows:report.append(f"t={r['t']}：零范数剔除{r['ordered']['excluded_zero_norm_keys']}格；覆盖="+json.dumps(r['ordered']['coverage'],ensure_ascii=False))
    report+=['\n## 判断','t=3差值为0，固定无扰动对照通过。t=6按实测正负方向解读；该距离读出只检验同一组C在固定图像格下是否呈现顺序敏感性，不代表学习式时序模型有效，也不证明自然伪造收益或深度独立贡献。','上轮绝对深度号＋格行列共有键为0仍有效，本轮没有改写原报告或缓存。当前帧C/逐格读出最大差0仅引用上轮结果，没有重新加载权重。','源哈希前后不变；新深度0、新特征0、模型调用0；耗时'+str(summary['seconds'])+'秒，退出0。执行HEAD '+identity['execution_head'],'结果绝对目录：'+str(OUT)]
    (OUT/'report.md').write_text('\n\n'.join(report)+'\n');save(OUT/'status.json',{'status':'COMPLETE','exit_code':0})
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
