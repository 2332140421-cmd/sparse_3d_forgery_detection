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


def augmentation_totals(records):
    """Weight by actual targets, not an unweighted mean of subgroup means."""
    from collections import Counter
    ops=Counter();groups={};seconds=0.;targets=0
    for r in records:
        targets+=r['targets'];seconds+=r['augmentation_seconds'];ops.update(r['operation_counts'])
        assert sum(r['operation_counts'].values())==2*r['targets']
        assert sum(r.get(g,{}).get('targets',0) for g in ['fake_nonempty','fake_empty','real_empty','real_nonempty'])==r['targets']
        for g in ['fake_nonempty','fake_empty','real_empty','real_nonempty']:
            if g not in r:continue
            s=groups.setdefault(g,Counter())
            for k in ['targets','before_fraction_sum','after_fraction_sum','became_empty']:s[k]+=r[g][k]
    result={}
    for g,s in groups.items():
        result[g]=dict(s,before_fraction_mean=s['before_fraction_sum']/s['targets'],after_fraction_mean=s['after_fraction_sum']/s['targets'],became_empty_ratio=s['became_empty']/s['targets'])
    return dict(targets=targets,operation_counts=dict(ops),operation_total=sum(ops.values()),groups=result,augmentation_seconds=seconds)


def directions(left,right,ids,metric='pixel_AP'):
    import statistics
    ds=sorted([(p,right[p][metric]-left[p][metric]) for p in ids],key=lambda x:x[1],reverse=True)
    pos=[(p,v) for p,v in ds if v>1e-8];negative=[(p,v) for p,v in ds if v< -1e-8]
    return dict(up=len(pos),down=len(negative),near=len(ds)-len(pos)-len(negative),largest_gains=pos[:3],largest_losses=negative[-3:],
                mean_delta=statistics.mean(v for _,v in ds),without_largest_gain_mean=statistics.mean(v for _,v in ds[1:]) if len(ds)>1 else None,
                top3_positive_gain_share=sum(v for _,v in pos[:3])/sum(v for _,v in pos) if pos else None)


INTERPRETATION = """### 观察事实：没有获得整体、多来源一致的定位优势

全部19来源：首轮D−S为AP−0.066387、Dice−0.012186、召回+0.062770；末轮AP−0.032125、Dice−0.011806、召回+0.005128。
各自开发选择S1对D3：AP−0.018214、Dice−0.035607、召回+0.026700，real FPR从0.001094到0.012103、fake背景FPR从0.001357到0.016387，面积从0.002889到0.018814。
更大的覆盖和较高召回伴随明显误报代价，并未换来AP/Dice整体优势；不把高召回单独宣布为成功。

逐来源AP首轮10升9降、末轮7升12降、各自开发选择9升10降；方向数含极小变化，不等于实质恢复。
选择结果对象16来源AP从0.187798到0.208358（9升7降），但Dice从0.105556到0.083451，召回0.093785到0.159925；real FPR到0.014326、fake背景FPR到0.019439。
所以不能说对象编辑完全无增益，也不能说总体实质改善。最大对象AP增量7e80j7为+0.443272，去掉它后的其余15对象来源平均AP增量−0.007621；此项只是敏感性描述，不更改全部来源主表。
全部19选择结果的正向AP增量前三为7e80j7、7ogeg4、9kgy54，占全部正向增量约85.4%；同时4ny267、7e20j7、4rv8o9等大幅下降，存在明显来源依赖。

背景3来源选择AP从0.790999降至0.565989，Dice0.120828降至0.013205。48owl9、72yzw9常数AP约0.711899、0.753499；D3 AP约0.827705、0.867848，但召回仅0.001233、0.019082，不能把高排序数字写成主体覆盖充分。
对象16常数AP约0.020269，D3的0.208358高于该参照，表示存在可读的排序证据，但不足以证明定位已经有效解决。没有因高GT占比剔除任何来源。

### 历史四来源及既有固定中帧

- 猫7e2e57：首轮D的AP/Dice/召回0.174044/0.136780/0.278087，相比S1有部分恢复；但D3为0.029030/0/0，D10为0.020588/0.003185/0.003100。提升未保留到开发选择或末轮。D3配对real FPR0.071663、fake背景FPR0.031322；不能用有响应代替正确覆盖。
- 骑马95q8g9：D3 AP0.216308（S1为0.240444，排序略低），Dice0.215957、召回0.492175明显高于S1的0.026924/0.014921；但real FPR0.051044、fake背景FPR0.117560，输出较宽。D10 AP0.106048、Dice0.143183、召回0.361192：保留部分阈值覆盖，但AP明显下降，不能只解释为操作点平移。
- 过街4xd3r9：D1/D3/D10 AP约0.017740/0.023492/0.016838，Dice和召回始终0。D3 AP虽高于常数约0.015999，并不表示0.5下检出了编辑主体。
- 7ogeg4：D3 AP0.375283较S1的0.049235提高，但Dice/召回仅0.001019/0.000519；D10 AP0.338266、Dice0.056507、召回0.032672，AP仍高于S10，但覆盖反而弱于S10的Dice0.120408/召回0.073080。

只查看既定四条fake中帧，不新增选帧或预测：D选择结果的猫响应主要在GT外左侧背景，骑马图较宽并蔓延到邻近背景，过街二值图为空，7ogeg4概率图有响应却几乎未过0.5；D末轮骑马仍宽、7ogeg4只剩局部覆盖。
对应S/D预览均原统一0–1概率色阶、0.5二值图。以上是少量预定图的辅助观察，主体结论以完整逐帧统计为准，不从PNG重算AP或挑好图。

### 后期退化没有被充分缓解

S首轮→末轮AP0.283041→0.207931（−0.075110）、Dice0.107967→0.045842（−0.062125）；D首轮→末轮AP0.216654→0.175806（−0.040848）、Dice0.095781→0.034035（−0.061746）。
D的AP绝对降幅看似较小，但首轮起点已更低；从各自开发选择至末轮，D3→D10 AP下降0.089020，反而大于S1→S10的0.075110，不能据首末差值宣称稳定性已改善。
D首末来源AP为8升11降、Dice5升8降6近零，召回0.171812→0.040282、面积0.026404→0.008719；同时AP下降，因此不是只改变0.5下覆盖。
D2的real FPR达0.036473，后期下降但仍高于S末轮；过程有明显波动，不是每一来源、每一轮都单调退化。

共同未增强训练观察AP：S 0.703973→0.933398，D 0.244948→0.791914；D后期仍在提高，但开发从D3以后总体走弱。
在线Focal/softIoU在两组各自均下降；D的增强输入更难且前景占比不同，不能把较高在线loss直接当成拟合差的证据。
共同观察支持训练观察与开发分化，但仅32目标/8来源，不代表完整训练集；开发最佳轮不同不能证明过拟合或学习率是唯一根因。

### 增广作用与成本的解释

109100次增强前非空fake目标，平均前景占比从0.060931升至0.100617；其中10668次变为空，比例0.097782。提高平均前景占比和产生空裁剪可以同时发生，不矛盾。
原空fake5170次、real空114510次分别登记；它们没有被补成前景，也没有因裁剪空而丢弃或反复抽样。
总计228780目标、457560项操作，非形变组及形变组各228780，实际计数符合每目标两项执行。
这些是整个策略的实际结果，不能把性能变化归因到单一RandomCrop、RandomRotate或某种前景比例；本轮没有事后调整采样。

D主流程19434.38秒（5小时23分54秒），S为18057.17秒（5小时0分57秒），增加约1377.21秒。
D累计增广时间887.54秒，包含于训练/主流程，不重复计入；其训练阶段记录16315.49秒，S14909.36秒。
峰值allocated/reserved均约5.694/5.881 GiB，未见显存规模变化。实际耗时不同不属于训练预算改变；两组监督次数及联合更新匹配。

### 唯一下一步建议（未执行）

本次证据不足以把Double选作后续PVST改造底座；它没有稳定保留首轮局部恢复，整体AP/Dice较低、误报代价更大。也不因此否定MumPy、ANAKIN或PVST思想。
仅推荐下一轮做一个预先登记的训练后期稳定性对照：以当前SingleAugment为固定参照，只改变一个预先登记的PolynomialLR衰减速度（当前调度已从首步开始衰减，不能误写成更早开始），其余数据、初始化、目标序列、输入、架构、损失和总预算不变，不做多计划搜索，也不同时叠加XYZ/C。
依据是共同观察持续提高而开发排序/覆盖下降，恢复默认增广未解决这一分化；学习率仍只是待检验因素，不是已定位根因。
已有AINPAINT二维参照epoch01全部19 AP约0.379856，仅作不同模型背景；架构/输入/损失等并未匹配，不能解释成纯时间或三维贡献。
"""


def analyze(out):
    """Independent CPU closeout; never calls experiment/report/model code."""
    import statistics
    import time
    started=time.monotonic();out=Path(out);old=Path(read(out/'config.json')['single_reference_root'])/'formal'
    dest=out/'analysis';dest.mkdir(exist_ok=True)
    from .analysis import table
    def frame_identity(ev):
        fs=ev['frames'];s={(f['video_id'],f['frame_index']):(f['pts'],f['GT_pixels'],f['native_pixels']) for f in fs}
        assert len(s)==len(fs)==5152,'FRAME_COVERAGE_OR_DUPLICATE'
        return s
    state={n:read(p/'status.json') for n,p in [('Single',old),('Double',out)]}
    for s in state.values():assert s['status']=='COMPLETE' and s['exit_code']==0 and s['updates']==19070 and s['target_presentations']==228780
    configs=[read(p/'config.json') for p in [old,out]]
    config_diffs={k:[configs[0].get(k),configs[1].get(k)] for k in configs[0].keys()|configs[1].keys() if configs[0].get(k)!=configs[1].get(k)}
    assert set(config_diffs)<={'augmentation','augmentation_policy','condition','read_only_cache','single_reference_root'},'OTHER_CONFIG_DIFFERENCE'
    for name in ['protocol.json','orders.json','monitor.json','initialization.json']:assert read(old/name)==read(out/name),('MATCHING_FAILED',name)
    for key in ['official_revision','weight_sha256','source_protocol_sha256','source_categories_sha256','torch_version','shared_helpers_sha256']:
        assert read(old/'identity.json')[key]==read(out/'identity.json')[key],('IDENTITY_FIELD_CHANGED',key)
    selected={n:read(p/'selection.json')['epoch'] for n,p in [('Single',old),('Double',out)]}
    metrics={};epochs=[];signature=None;previews={};loss_seconds={};peaks={};eval_seconds={};summary={}
    for n,p in [('Single',old),('Double',out)]:
        metrics[n]={};loss_seconds[n]=0.;eval_seconds[n]=0.;peaks[n]=[0,0]
        for e in range(1,11):
            ev=read(p/f'epoch{e:02d}_dev.json');sig=frame_identity(ev)
            if signature is None:signature=sig
            assert sig==signature,('DIFFERENT_FRAMES_PTS_GT',n,e)
            metrics[n][e]=dict(macro=ev['macro'],groups=ev['groups'],sources={s['pair_id']:s for s in ev['sources']})
            l=read(p/f'epoch{e:02d}_loss.json');m=read(p/f'epoch{e:02d}_monitor.json')
            assert l['updates']==1907*e and l['target_presentations']==22878*e
            assert len(m['frames'])==32 and len(m['sources'])==8
            loss_seconds[n]+=l['seconds'];eval_seconds[n]+=ev['seconds']+m['seconds']
            peaks[n]=[max(peaks[n][i],l[k]) for i,k in enumerate(['peak_allocated_bytes','peak_reserved_bytes'])]
            epochs.append(dict(condition=n,epoch=e,updates=l['updates'],weighted_focal=l['weighted_focal'],soft_IoU=l['soft_IoU'],monitor_AP=m['macro']['pixel_AP'],additional15_AP=ev['groups']['新增15开发来源']['pixel_AP'],**ev['macro']))
        actual=max(metrics[n],key=lambda e:(metrics[n][e]['groups']['新增15开发来源']['pixel_AP'],-e));assert actual==selected[n]
        sm=read(p/'summary.json');assert sm['status']=='COMPLETE' and sm['epochs_evaluated']==list(range(1,11)) and sm['selected_epoch']==actual
        assert sm['selected']==metrics[n][actual]['macro'] and sm['last']==metrics[n][10]['macro']
        summary[n]=sm
        fixed=[(r['video_id'],i) for r in read(p/'protocol.json')['rows'] if r['split']=='dev' for i in r['preview_targets']]
        assert len(fixed)==54
        assert all((p/'preview'/f'{v}_{i:05d}_{label}_{kind}.png').is_file() for v,i in fixed for label in ['selected','last'] for kind in ['prob','mask'])
        previews[n]=fixed
    assert previews['Single']==previews['Double']
    aug_records=[dict(epoch=e,**read(out/f'epoch{e:02d}_augmentation.json')) for e in range(1,11)]
    aug=augmentation_totals(aug_records);assert aug['targets']==228780 and aug['operation_total']==457560
    hist={'an:7e2e57','an:95q8g9','an:4xd3r9','an:7ogeg4'}
    ids=set(metrics['Single'][1]['sources'])
    original_categories=read(out.parent.parent/'pvst_fullvideo_matched_localization_v1/categories.json')
    objects={p for p,c in original_categories.items() if c['content_priority']=='object_edit_pending_human'}
    directions_table=[];summary_directions={}
    source_tables={};group_tables={}
    for comparison,es in [('首轮',{'Single':1,'Double':1}),('末轮',{'Single':10,'Double':10}),('各自开发选择',selected)]:
        a=metrics['Single'][es['Single']];b=metrics['Double'][es['Double']]
        source_tables[comparison]=[];group_tables[comparison]=[]
        for p in sorted(ids):
            row=dict(comparison=comparison,pair_id=p,single_epoch=es['Single'],double_epoch=es['Double'],constant_AP=a['sources'][p]['constant_AP'])
            for k in KEYS:
                row['Single_'+k]=a['sources'][p][k];row['Double_'+k]=b['sources'][p][k];row['delta_'+k]=b['sources'][p][k]-a['sources'][p][k]
            source_tables[comparison].append(row)
        for g in ['全部19开发来源','原4开发来源','新增15开发来源','对象编辑16来源','外观背景参照']:
            for n,e in es.items():group_tables[comparison].append(dict(condition=n,epoch=e,group=g,**metrics[n][e]['groups'][g]))
        summary_directions[comparison]={}
        for name,ps in [('全部19',ids),('对象16',objects)]:
            summary_directions[comparison][name]={k:directions(a['sources'],b['sources'],ps,k) for k in ['pixel_AP','Dice','GT_recall']}
            for k,d in summary_directions[comparison][name].items():directions_table.append(dict(comparison=comparison,group=name,metric=k,up=d['up'],down=d['down'],near=d['near'],mean_delta=d['mean_delta']))
    late={n:{k:directions(metrics[n][1]['sources'],metrics[n][10]['sources'],ids,k) for k in ['pixel_AP','Dice','GT_recall']} for n in metrics}
    writecsv(dest/'epoch_history.csv',epochs);writecsv(dest/'direction_counts.csv',directions_table)
    # Existing first/last/selected source/group tables remain in comparison/.
    writecsv(dest/'augmentation_by_epoch.csv',[dict(epoch=r['epoch'],targets=r['targets'],seconds=r['augmentation_seconds'],fake_positive_before=r['fake_nonempty']['before_fraction_mean'],fake_positive_after=r['fake_nonempty']['after_fraction_mean'],became_empty=r['fake_nonempty']['became_empty'],became_empty_ratio=r['fake_nonempty']['became_empty']/r['fake_nonempty']['targets'],**r['operation_counts']) for r in aug_records])
    checks=dict(config_differences=config_diffs,protocol_orders_monitor_initialization_equal=True,all20_development_frames_PTS_GT_equal=True,fixed54_targets_equal=True,initial_model_sha256=read(out/'initialization.json')['model_sha256'],actual_state=state,selected_epochs=selected)
    result=dict(status='CPU_ANALYSIS_COMPLETE',checks=checks,directions=summary_directions,first_to_last=late,augmentation=aug,training_seconds=loss_seconds,dev_and_monitor_seconds=eval_seconds,peak_allocated_reserved_bytes=peaks,new_training=0,new_checkpoint_load=0,new_model_instantiation=0,new_forward=0,new_depth=0,new_features=0)
    f=lambda x:f'{x:.6f}' if isinstance(x,float) else str(x)
    def tbl(head,rs):return table(head,[[f(x) for x in row] for row in rs])
    lines=['## 正式运行后的CPU匹配分析','',
      'Double正式状态COMPLETE，退出码0；实际完成10轮、19070次联合更新、228780次目标呈现，10轮均有完整开发评价。开发选择为第3轮（不是预设首轮），Single选择第1轮；按新增15来源宏AP、同分取早已用实际10点评价复核。两者均为开发选择，不称独立测试。',
      f"Double保存的主流程耗时{state['Double']['seconds']:.2f}秒（约{state['Double']['seconds']/3600:.3f}小时），Single {state['Single']['seconds']:.2f}秒；旧smoke失败不属于本次正式失败。",
      '### 匹配核对与计分','保存的protocol、orders、monitor、initialization完全一致；初始化hash '+checks['initial_model_sha256']+'。配置差异仅增广策略及condition/read_only_cache/single_reference_root元数据；官方版本、通用权重SHA、公共评价helper摘要及Torch版本一致。源码摘要不同对应已登记的策略、类别插值修正及统计/独立输出，不是结构或训练预算变化。未加载权重做复验。',
      '89训练来源/178视频，每轮22878合法目标；19开发来源/38视频/5152唯一目标，含原4和新增15，类别分组沿既有对象16/背景3。94保护测试来源未参与；40y1x9_fake[6,10)仍不监督。实际每轮loss记录的累计更新和目标呈现与共同预算一致。',
      '20次开发评价的(video_id,frame_index,PTS,GT像素数,原生总像素数)逐项一致，54个预览目标也相同。正GT fake帧AP→视频→来源等权；空GT不补AP，误报仍保留。Dice/IoU/召回、real FPR及fake背景FPR均固定0.5；全篇FPR和预测面积是比例（0.01=1%）。开发完整画面，无GT驱动裁剪/TTA。',
      '同一完整官方MumPy、三帧中帧定位、224输入、FP32、共同初值和目标序列、三组SGD/多项式调度、WeightedFocal＋softIoU。唯一方法差异为Single→默认Double及其mask最近邻修正。图像仍是旧长边384工作RGB再缩224，不是原生直接224；没有XYZ C或全片关系。',
      '### 全部19开发来源主表',tbl(['条件','比较','epoch']+KEYS,[[n,label,e]+[metrics[n][e]['macro'][k] for k in KEYS] for label,es in [('首轮',{'Single':1,'Double':1}),('末轮',{'Single':10,'Double':10}),('各自开发选择',selected)] for n,e in es.items()]),
      '### 分组结果',tbl(['比较','条件','epoch','分组']+KEYS,[[label,r['condition'],r['epoch'],r['group']]+[r[k] for k in KEYS] for label,rs in group_tables.items() for r in rs]),
      '### 全部19来源明细','S/D分别是Single/Double；以下分别比较同预算首轮、末轮和各自开发选择，后者是S1对D3，不能混作相同更新步数对照。',]
    for label,rows in source_tables.items():
        lines +=['#### '+label,tbl(['来源','常数AP','S AP','D AP','ΔAP','S Dice','D Dice','ΔDice','S召回','D召回','Δ召回'],[[r['pair_id'],r['constant_AP']]+[r[p+k] for k in ['pixel_AP','Dice','GT_recall'] for p in ['Single_','Double_','delta_']] for r in rows])]
    lines+=['方向数使用1e-8近零容差，仅数值方向不代表实质改善。',tbl(['比较','分组','指标','升','降','近零','均值增量'],[[r[k] for k in ['comparison','group','metric','up','down','near','mean_delta']] for r in directions_table]),
      '### 十轮训练观察与开发','曲线：'+str(dest/'learning_curves.svg'),'观察为共同未增强32目标/8来源，不是完整训练集。在线loss来自不同增强输入、前景占比及难度，不用两者loss高低判定谁拟合更好。',
      tbl(['条件','轮','累计更新','在线Focal','在线softIoU','未增强观察AP','新增15 AP','全部19 AP','Dice','召回','real FPR','fake背景FPR','面积'],[[r[k] for k in ['condition','epoch','updates','weighted_focal','soft_IoU','monitor_AP','additional15_AP','pixel_AP','Dice','GT_recall','real_FPR','fake_background_FPR','prediction_area']] for r in epochs]),
      '### 实际增广与成本','每个目标两项操作；总目标'+str(aug['targets'])+'、总操作'+str(aug['operation_total'])+'。',tbl(['操作','次数'],list(aug['operation_counts'].items())),
      tbl(['增强前类别（224网格）','次数','增强前平均前景比','增强后平均前景比','变空数','变空比例'],[[g,s['targets'],s['before_fraction_mean'],s['after_fraction_mean'],s['became_empty'],s['became_empty_ratio']] for g,s in aug['groups'].items()]),
      '这里原非空/原空均指增强前224工作mask，不把原空fake自动解释为原生官方GT也为空。前景占比是逐目标平均，不是全体像素合池；操作计数由真实执行记录累计，不是计划分布。',
      f"增强累计耗时{aug['augmentation_seconds']:.2f}秒（是训练耗时的子集，不能重复加到总耗时）。Single/Double保存的训练阶段耗时{loss_seconds['Single']:.2f}/{loss_seconds['Double']:.2f}秒；开发加观察{eval_seconds['Single']:.2f}/{eval_seconds['Double']:.2f}秒。训练阶段记录含checkpoint等组织开销。两组峰值allocated/reserved相同：{peaks['Double'][0]}/{peaks['Double'][1]}字节。",
      '### 结果判断与固定预览',INTERPRETATION,
      '### 产物与边界','原始程序报告、summary、配置、日志、预测和旧Single报告不修改；复用comparison/的groups.csv、sources.csv、各项delta及epochs.csv，本次派生统计仅在analysis/。','CPU复算：python -m research_tools.mumpy_anakin.double_comparison --analyze --output '+str(out),
      'Double既有预览：'+str(out/'preview/index.html'),'Single既有预览：'+str(old/'preview/index.html'),
      '本轮新增训练、检查点加载、模型实例化/前向、深度和特征提取全部0。']
    result['CPU_analysis_seconds']=time.monotonic()-started
    (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines.append('CPU汇总脚本耗时：'+str(result['CPU_analysis_seconds'])+'秒（另有人工阅读与报告整理），全部新增模型计算为0。')
    (dest/'report.md').write_text('\n\n'.join(lines)+'\n')
    report=Path(__file__).resolve().parents[2]/'docs/pvst/anakin_mumpy_doubleaugment_v1.md'
    previous=report.read_text().split('\n## 正式运行后的CPU匹配分析')[0]
    previous=previous.replace('状态：实现、CPU检查和真实短预检完成，退出码0；**正式10轮尚未启动，由用户终端/tmux启动**。2026-11-26。','状态：**正式10轮、评价和固定预览已完成，退出码0；CPU匹配分析已完成**。2026-11-26。以下实现、短预检与启动说明为历史交付记录，正式结果见末节。')
    previous=previous.replace('本次formal目录尚不存在。','预检交付时formal目录尚不存在；现正式运行已完成。')
    report.write_text(previous+'\n\n'+'\n\n'.join(lines)+'\n')
    print(json.dumps(dict(selected=selected,directions=summary_directions,late=late,augmentation=aug,seconds=result['CPU_analysis_seconds']),ensure_ascii=False,indent=2))
    return lines,result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Pure CPU saved-metric analysis, no models or checkpoints')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.analyze:analyze(args.output)
    else:compare(args.output)
