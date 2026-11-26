"""Small CPU report/export companion, no model imports or checkpoint access."""
import argparse,csv,json
from pathlib import Path
CONDITIONS=('A_CURRENT_XY','B_FULL_XY','C_FULL_XYZ')
KEYS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area']
def read(p):return json.loads(p.read_text())
def csvsave(p,rows):
    if not rows:return
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def smoke_report(root):
    d=read(root/'result.json');lines=['# 全片变化关系短预检（不是效果实验）','预检全部更新丢弃；不报告AP收益。','|病例|条件|T|M|合法帧|步|loss|前向秒|反向秒|更新秒|IO秒|allocated|reserved|','|'+ '---|'*13]
    for r in d['cases']:
        for i,s in enumerate(r['steps'],1):lines.append('|'+ '|'.join(str(v) for v in [r['video'],r['condition'],r['T'],r['M'],r['supervised_frames'],i,s['loss'],s['forward_seconds'],s['backward_seconds'],s['update_seconds'],r['IO_seconds'],s['peak_allocated_bytes'],s['peak_reserved_bytes']])+'|')
    lines+=['BN未改、实际梯度/官方头对齐/缓存z0误差及恢复证据见result.json。SDPA后端由PyTorch分派，无单独内核profiling，不冒称已测Flash。正式训练未启动。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    paths=sorted((root/'preview').glob('*png'))
    (root/'preview/index.html').write_text('<meta charset="utf-8"><h1>预检固定frame0，非开发效果</h1>'+''.join(f'<p>{p.name}<br><img width="300" src="{p.name}"></p>' for p in paths))
def report(out):
    rows=[];history=[];complete={};source_tables=[]
    if (out/'frozen_reference.json').exists():
        ref=read(out/'frozen_reference.json')
        rows.extend(dict(condition='R0_FROZEN_EPOCH01',epoch=0,group=g,**m) for g,m in ref['groups'].items())
    for c in CONDITIONS:
        folder=out/c
        if (folder/'complete.json').exists():complete[c]=read(folder/'complete.json')
        for e in range(1,11):
            p=folder/f'epoch{e:02d}_dev.json'
            if not p.exists():continue
            ev=read(p);rows.extend(dict(condition=c,epoch=e,group=g,**m) for g,m in ev['groups'].items())
            m=read(folder/f'epoch{e:02d}_monitor.json') if (folder/f'epoch{e:02d}_monitor.json').exists() else None
            rs=list(csv.DictReader((folder/'updates.csv').open())) if (folder/'updates.csv').exists() else []
            current=[r for r in rs if int(r['epoch'])==e]
            loss={k:sum(float(r[k]) for r in current)/len(current) for k in ['loss','pixel_BCE','soft_Dice','block_BCE']} if current else {}
            history.append(dict(condition=c,epoch=e,updates=e*178,**loss,monitor_AP=m['macro']['pixel_AP'] if m else None,**ev['macro']))
            source_tables.extend(dict(condition=c,epoch=e,**s) for s in ev['sources'])
    csvsave(out/'group_metrics.csv',rows);csvsave(out/'history.csv',history);csvsave(out/'source_metrics.csv',source_tables)
    status='COMPLETE' if len(complete)==3 else 'PARTIAL'
    (out/'summary.json').write_text(json.dumps(dict(status=status,complete=complete,new_backbone=0,new_MoGe=0),ensure_ascii=False,indent=2)+'\n')
    lines=['# AINPAINT视觉底座上的全片变化关系定位实验v1','计算状态：'+status+'；本研究新增设计，不是原论文复现。','R0为已有指标引用，epoch0不是新增模块训练收益。固定末次epoch10主表，新增15开发AP选择仅补充。所有FPR是比例，阈值0.5，AP正GT fake帧→视频→来源等权。',
           '|条件|轮|范围|'+'|'.join(KEYS)+'|','|'+'---|'*(3+len(KEYS))]
    for r in rows:
        if r['group'] in ['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照'] and (r['epoch'] in [0,10] or r['condition'] in complete and r['epoch']==complete[r['condition']]['selected_epoch']):
            lines.append('|'+ '|'.join(str(r[k]) for k in ['condition','epoch','group']+KEYS)+'|')
    deltas=[]
    for left,right in [('A_CURRENT_XY','B_FULL_XY'),('B_FULL_XY','C_FULL_XYZ')]:
        a={s['source_group_id']:s for s in source_tables if s['condition']==left and s['epoch']==10};b={s['source_group_id']:s for s in source_tables if s['condition']==right and s['epoch']==10}
        if a and b:
            assert a.keys()==b.keys()
            for p in a:deltas.append(dict(comparison=right+'-'+left,pair_id=a[p]['pair_id'],**{k:b[p][k]-a[p][k] for k in KEYS}))
    csvsave(out/'final_differences.csv',deltas)
    lines+=['B−A是时空变化关系整套增量；C−B是XYZ组织与几何编码方案增量，不称纯深度或三维必要性证明。观察32目标不是全训练集。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    page=['<meta charset="utf-8"><h1>固定54目标，概率统一0–1，二值阈值0.5</h1>']
    if (out/'protocol.json').exists():
        for r in read(out/'protocol.json')['rows']:
            if r['split']!='dev':continue
            for i in r['preview_targets']:
                stem=r['video_id']+f'_{i:05d}';page.append(f'<h2>{stem} PTS {r["pts"][i]:.6f}</h2>')
                for label in ['rgb','gt','FROZEN_EPOCH01_prob','FROZEN_EPOCH01_mask']+[c+'_'+s+'_'+kind for c in CONDITIONS for s in ['last','selected'] for kind in ['prob','mask']]:
                    path=out/'preview'/(stem+'_'+label+'.png')
                    if path.exists():page.append(f'<label>{label}<img width="210" src="{path.name}"></label>')
    (out/'preview').mkdir(exist_ok=True);(out/'preview/index.html').write_text('\n'.join(page))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.output)
