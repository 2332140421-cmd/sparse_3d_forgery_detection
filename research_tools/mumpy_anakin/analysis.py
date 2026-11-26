"""CPU closeout: saved metrics only; no checkpoint/model imports or inference."""
import csv
import json
import statistics
import time
from pathlib import Path

ROOT=Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
REPO=Path(__file__).resolve().parents[2]
KEYS=['pixel_AP','Dice','IoU','GT_recall','real_FPR','fake_background_FPR','prediction_area','constant_AP']
GROUPS=['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照']
def read(p):return json.loads(p.read_text())
def mean(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None
def sources(ev):
    if 'sources' in ev:return {s['pair_id']:s for s in ev['sources']}
    return {p:{k:mean(v.get(k) for v in ev['videos'] if v['pair_id']==p and v['role']==('real' if k=='real_FPR' else 'fake')) for k in KEYS} for p in {v['pair_id'] for v in ev['videos']}}
def grouped(ss,cats):
    hist={'an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4'}
    ids=[set(ss),hist,set(ss)-hist,{p for p,c in cats.items() if c['content_priority']=='object_edit_pending_human'},{p for p,c in cats.items() if c['content_priority']=='appearance_background_reference'}]
    return {g:{k:mean(ss[p].get(k) for p in ps) for k in KEYS} for g,ps in zip(GROUPS,ids)}
def table(head,rows):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in r)+'|' for r in rows])
def writecsv(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def signature(ev):
    fs=ev.get('frames')
    if fs is None:return None
    result={(f['video_id'],f['frame_index']):(f['pts'],f['GT_pixels'],f['native_pixels']) for f in fs}
    assert len(result)==len(fs)==5152
    return result
def main():
    start=time.monotonic();out=ROOT/'pvst_anakin_mumpy_baseline_v1/formal';dest=out/'analysis';dest.mkdir(exist_ok=True)
    old=ROOT/'pvst_anakin_expanded_spatial_training_v1'
    cats=read(ROOT/'pvst_fullvideo_matched_localization_v1/categories.json')
    evs=[read(out/f'epoch{e:02d}_dev.json') for e in range(1,11)]
    ss=[sources(e) for e in evs];gs=[grouped(s,cats) for s in ss]
    sig=signature(evs[0]);assert all(signature(e)==sig for e in evs)
    # Old spatial JSON stores per-frame records separately in CSV.
    for epoch in [1,10]:
        with (old/f'epoch{epoch:02d}_dev_frames.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        oldsig={(r['video_id'],int(r['frame_index'])):(float(r['pts']),int(r['GT_pixels']),int(r['native_pixels'])) for r in rows}
        assert oldsig==sig and len(rows)==5152
    models={'二维epoch01':sources(read(old/'epoch01_dev.json')),'二维epoch10':sources(read(old/'epoch10_dev.json')),'MumPy epoch01':ss[0],'MumPy epoch10':ss[-1]}
    grouped_models={n:grouped(s,cats) for n,s in models.items()}
    core=[dict(model=n,group=g,**m) for n,groups in grouped_models.items() for g,m in groups.items()]
    writecsv(dest/'group_comparison.csv',core)
    history=[]
    for e in range(1,11):
        loss=read(out/f'epoch{e:02d}_loss.json');mon=read(out/f'epoch{e:02d}_monitor.json')
        assert len(mon['frames'])==32 and len(sources(mon))==8
        history.append(dict(epoch=e,updates=loss['updates'],weighted_focal=loss['weighted_focal'],soft_IoU=loss['soft_IoU'],monitor_AP=mon['macro']['pixel_AP'],historical4_AP=gs[e-1][GROUPS[1]]['pixel_AP'],additional15_AP=gs[e-1][GROUPS[2]]['pixel_AP'],**gs[e-1][GROUPS[0]]))
    writecsv(dest/'epoch_history.csv',history)
    comparisons=[]
    for p in sorted(ss[0]):
        row=dict(pair_id=p,**cats[p])
        for n,s in models.items():
            for k in KEYS:row[n+'_'+k]=s[p].get(k)
        for k in KEYS:
            row['M1_minus_D1_'+k]=ss[0][p][k]-models['二维epoch01'][p][k]
            row['M10_minus_M1_'+k]=ss[-1][p][k]-ss[0][p][k]
        comparisons.append(row)
    writecsv(dest/'source_comparison.csv',comparisons)
    def direction(field):
        ds=[(r['pair_id'],r[field]) for r in comparisons];top=max(ds,key=lambda x:x[1]);return dict(up=sum(v>1e-8 for _,v in ds),down=sum(v< -1e-8 for _,v in ds),near=sum(abs(v)<=1e-8 for _,v in ds),largest=top,without_largest_mean=mean(v for p,v in ds if p!=top[0]))
    state=read(out/'status.json');assert state['status']=='COMPLETE' and state['updates']==19070
    result=dict(status='CPU_ANALYSIS_COMPLETE',formal=state,frame_identity_GT_match=True,groups=grouped_models,directions={k:direction(k) for k in ['M1_minus_D1_pixel_AP','M10_minus_M1_pixel_AP','M10_minus_M1_Dice']},new_training=0,new_model_instantiation=0,new_checkpoint_load=0,new_forward=0,new_depth=0,new_features=0)
    lines=['## 正式运行后的CPU结果分析','',
      '正式运行COMPLETE、退出码0：10轮、19070次联合优化更新、228780次监督目标呈现；三个SGD各执行19070次step。运行秒数'+str(state['seconds'])+'。短预检及其失败日志保留，不属于本次正式失败。',
      '89训练来源/178视频，每轮22878合法目标；19开发来源/38视频/5152唯一目标（2575 fake，其中2570正GT、5空GT；2577 real）。94测试保护来源未参与。官方标注技术核验后直接使用，不声称逐条人工语义验收。',
      'CPU核对10轮开发及二维epoch01/10的全部(video_id, frame_index, PTS, GT像素数, 原生总像素数)一致。AP仅非空GT fake帧→视频→来源等权；空GT不补AP。Dice/IoU/召回与两类FPR固定logit≥0（概率0.5）；FPR和面积均为比例，非百分数。',
      'MumPy仅三帧输入及中帧输出，不含XYZ C或全片关系。与二维参照的架构、训练范围、优化器、损失及输入处理不同，差值不代表纯时间模块贡献。MumPy读取已有长边384工作RGB再缩放224×224，并非原生直接缩放224。',
      '', '### 四份权重及开发分组',table(['模型','分组']+KEYS,[[r['model'],r['group']]+[r[k] for k in KEYS] for r in core]),
      '', '### 十轮训练与开发变化','训练观察固定32目标/8训练来源，不代表完整训练集，也不与另一实验不同观察清单直接比较。开发选择依据新增15来源AP，在全部10个候选轮次中选第1轮；末次主结果仍为第10轮，均不称独立测试成绩。',
      table(['epoch','累计更新','在线Focal','在线softIoU','观察AP','原4 AP','新增15 AP','全部19 AP','Dice','召回','real FPR','fake背景FPR','预测面积'],[[r[k] for k in ['epoch','updates','weighted_focal','soft_IoU','monitor_AP','historical4_AP','additional15_AP','pixel_AP','Dice','GT_recall','real_FPR','fake_background_FPR','prediction_area']] for r in history]),
      '', '### 全部19来源：二维与MumPy首轮/末轮',table(['来源','常数AP','二维1 AP/Dice/召回','二维10 AP/Dice/召回','M1 AP/Dice/召回','M10 AP/Dice/召回','M1−二维1 AP','M10−M1 AP'],[[r['pair_id'],r['MumPy epoch01_constant_AP']]+[' / '.join(f'{r[n+"_"+k]:.6f}' for k in ['pixel_AP','Dice','GT_recall']) for n in models]+[r['M1_minus_D1_pixel_AP'],r['M10_minus_M1_pixel_AP']] for r in comparisons]),
      '', '方向统计（1e-8内记近零；方向数不等于实质改善数）：'+json.dumps(result['directions'],ensure_ascii=False),
      '完整分组指标见group_comparison.csv，逐来源指标及差值见source_comparison.csv；对象/外观分类来自既有官方描述预筛，不是人工异常真值。常数AP即相同层级GT占比，不因高分而剔除大面积来源。',
      '', '### 解释与唯一下一步',
      '观察事实：相对二维首轮，MumPy首轮AP为4升15降，上升为4ny267、4rv8o9、7e20j7和4xd3r9；过街仅+0.002661且Dice/召回均0，不算实质检出。对象组AP0.326195→0.187798，外观/背景组AP0.666052→0.790999，但后者Dice0.513163→0.120828、召回0.425557→0.190418，不能称整体定位改善。4ny267归外观/背景参照，其他两条实质AP增益属于对象组；增益并非完全由背景造成，但未形成多来源总体优势。去掉最大AP增量7e20j7后，其余18来源平均增量仍−0.138103。分类仅沿官方文字预筛，不自行重标。\n\n大面积48owl9、72yzw9常数AP分别0.711899、0.753499，MumPy首轮AP0.945178、0.922230却召回仅0.003100、0.009771：排序高不等于0.5下覆盖完整。不能用这两条高AP宣布目标场景已解决。对象组常数AP0.020269，MumPy首轮0.187798有高于常数的排序证据，但弱于二维0.326195。\n\n原四来源：猫首轮AP/Dice/召回0.033071/0/0，末轮0.010108/0.000940/0.001010，未恢复实质定位；骑马首轮0.240444/0.026924/0.014921，末轮0.141651/0.000252/0.000127，覆盖极少；过街首末AP0.015467/0.017725、Dice/召回均0，仍未检出；7ogeg4从0.049235/0.000145/0.000078到0.237534/0.120408/0.073080，有恢复但远低于二维首轮0.656180/0.485468/0.414713。它们均保留在主表。\n\n仅查看已有固定中帧：猫概率图很弱、二值几乎空；骑马首轮仅零星点，未覆盖GT主体；过街首轮响应落在GT之外、末轮几乎空；7ogeg4末轮有局部恢复但不完整。图像仅辅助说明，与各自逐帧统计相符，不把一个中帧推广为全片。概率图与二值图分开理解，不逐图拉伸或用PNG重算AP。\n\n首轮→末轮：10/19 AP上升、9/19下降；Dice6升7降6近零。宏AP仍0.283041→0.207931、Dice0.107967→0.045842、召回0.109043→0.035154，fake预测面积0.002889→0.001329，real FPR0.001094→0.000997、fake背景FPR0.001357→0.000769。4ny267、4rv8o9、7e20j7、9jng87大幅下降，7e80j7等改善，属于来源依赖及输出更保守，并非全部来源退化。AP排序也变差，不能用纯置信度平移解释。\n\n训练观察AP0.703973→0.933398，在线Focal0.035298→0.009481、softIoU loss0.895649→0.649192；观察AP中间有波动。开发较首轮下降，支持训练观察与开发分化；只有32目标/8来源观察，不能代替全训练结果或据此确定过拟合的具体原因。学习率、分辨率、结构与迁移适配均未独立归因；没有证据证明某层实现错误。\n\n唯一下一步建议（本轮未执行）：优先设计一个预先固定的训练后期稳定性单因素对照，例如仅改变MumPy更新计划的学习率衰减时点，保留数据、初始化、输入、架构、损失与总更新，并保留本次计划为参照；不做多方案搜索。依据是当前已能拟合观察目标、部分对象有定位能力，但后期开发排序和覆盖下降；先检验训练适配比直接叠加XYZ/C更有依据。该建议不表示已证实学习率是根因，也不将本基线负结果外推为数据无用或PVST假设无效。',
      '', '### 计算边界与交付','本轮只读JSON/CSV及既有PNG，不加载检查点、不实例化模型；新增训练/前向/深度/特征提取均0。原程序报告、日志、预测和权重均不修改。派生结果仅写formal/analysis。',
      '已有固定预览：'+str(out/'preview/index.html'), 'CPU复算入口：python -m research_tools.mumpy_anakin.analysis。']
    (dest/'report.md').write_text('\n\n'.join(lines)+'\n')
    result['CPU_analysis_seconds']=time.monotonic()-start
    lines.append('CPU汇总脚本墙钟：'+str(result['CPU_analysis_seconds'])+'秒；人工读取与报告整理不计入训练耗时。聚合/身份回归2项通过。')
    (dest/'report.md').write_text('\n\n'.join(lines)+'\n')
    (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    report=REPO/'docs/pvst/anakin_mumpy_baseline_v1.md';text=report.read_text();marker='\n## 正式运行后的CPU结果分析'
    text=text.split(marker)[0];text=text.replace('状态：**实现、CPU检查和真实短预检通过；正式训练尚未启动，等待用户在终端启动**。2026-11-26。','状态：**正式10轮运行、评价及固定预览已完成（退出码0）；CPU结果分析已完成**。2026-11-26。');text=text.replace('**正式训练尚未启动，等待用户在终端启动。**','**历史交付时正式训练尚未启动；现已由用户运行完成，正式结果见下节。**')
    report.write_text(text+'\n\n'+'\n\n'.join(lines)+'\n')
    print(json.dumps(result['directions'],ensure_ascii=False));print(table(['模型']+KEYS,[[n]+[g[GROUPS[0]][k] for k in KEYS] for n,g in grouped_models.items()]));print('CPU seconds',result['CPU_analysis_seconds'])
if __name__=='__main__':main()
