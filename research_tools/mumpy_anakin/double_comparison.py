"""Saved-metric CPU comparison; no model/checkpoint imports."""
import csv,json
from pathlib import Path
KEYS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area']
def read(p):return json.loads(p.read_text())
def writecsv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compare(out):
    config=read(out/'config.json');old=Path(config['single_reference_root'])/'formal'
    dest=out/'comparison';dest.mkdir(exist_ok=True);series={}
    for name,folder in [('SingleAugment',old),('DoubleAugment',out)]:
        series[name]=[(e,read(folder/f'epoch{e:02d}_dev.json')) for e in range(1,11) if (folder/f'epoch{e:02d}_dev.json').exists()]
    rows=[];details=[];curves=[]
    for name,evs in series.items():
        if not evs:continue
        chosen=max(evs,key=lambda x:(x[1]['groups']['新增15开发来源']['pixel_AP'],-x[0]))
        for e,ev in evs:
            for g,m in ev['groups'].items():curves.append(dict(condition=name,epoch=e,group=g,**{k:m.get(k) for k in KEYS}))
        for label,(epoch,ev) in [('first',evs[0]),('last_available',evs[-1]),('selected',chosen)]:
            for g,m in ev['groups'].items():rows.append(dict(condition=name,comparison=label,epoch=epoch,group=g,**{k:m.get(k) for k in KEYS}))
            for s in ev.get('sources',[]):details.append(dict(condition=name,comparison=label,epoch=epoch,pair_id=s['pair_id'],**{k:s.get(k) for k in KEYS}))
    writecsv(dest/'groups.csv',rows);writecsv(dest/'sources.csv',details);writecsv(dest/'epochs.csv',curves)
    deltas=[]
    for label in ['first','last_available','selected']:
        a={r['pair_id']:r for r in details if r['condition']=='SingleAugment' and r['comparison']==label}
        b={r['pair_id']:r for r in details if r['condition']=='DoubleAugment' and r['comparison']==label}
        for p in sorted(set(a)&set(b)):
            deltas.append(dict(comparison=label,pair_id=p,single_epoch=a[p]['epoch'],double_epoch=b[p]['epoch'],**{k:b[p][k]-a[p][k] if a[p][k] is not None and b[p][k] is not None else None for k in KEYS}))
    writecsv(dest/'double_minus_single_sources.csv',deltas)
    group_deltas=[]
    for label in ['first','last_available','selected']:
        a={r['group']:r for r in rows if r['condition']=='SingleAugment' and r['comparison']==label}
        b={r['group']:r for r in rows if r['condition']=='DoubleAugment' and r['comparison']==label}
        for g in sorted(set(a)&set(b)):
            group_deltas.append(dict(comparison=label,group=g,single_epoch=a[g]['epoch'],double_epoch=b[g]['epoch'],**{k:b[g][k]-a[g][k] if a[g][k] is not None and b[g][k] is not None else None for k in KEYS}))
    writecsv(dest/'double_minus_single_groups.csv',group_deltas)
    lines=['# 官方MumPy架构在ANAKIN上的增广策略对照','仅训练增广策略改变；相同完整开发计分，无开发GT裁剪。各自开发选择不是独立测试。末次主结果为epoch10；未到epoch10时last_available不称末次。',
           '|条件|比较|epoch|范围|'+'|'.join(KEYS)+'|','|'+'---|'*(4+len(KEYS))]
    for r in rows:
        if r['group'] in ['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照']:
            lines.append('|'+ '|'.join(str(r[k]) for k in ['condition','comparison','epoch','group']+KEYS)+'|')
    lines+=['','## Double−Single（各自末轮与开发选择不混用）','|比较|分组|Single轮|Double轮|'+'|'.join(KEYS)+'|','|'+'---|'*(4+len(KEYS))]
    for r in group_deltas:lines.append('|'+ '|'.join(str(r[k]) for k in ['comparison','group','single_epoch','double_epoch']+KEYS)+'|')
    stats=[]
    for p in sorted(out.glob('epoch*_augmentation.json')):stats.append(dict(epoch=int(p.name[5:7]),**read(p)))
    (dest/'augmentation_by_epoch.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    lines+=['','逐来源差值：double_minus_single_sources.csv；完整曲线epochs.csv；逐轮增广计数和前景占比：augmentation_by_epoch.json。',
            '不重新推理SingleAugment或二维模型；本结果不证明XYZ或全视频C有效。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
