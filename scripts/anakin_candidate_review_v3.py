"""Next local training-source review batch; reuse v2 bounded pool and previews.

No model scores, new downloads, frontend extraction or automatic split freezing.
Only unreviewed train/unassigned non-obvious-moving sources are shown.
"""
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback
import cv2
from scripts import anakin_candidate_review as prior
from scripts import anakin_candidate_review_v2 as v2
from scripts.prepare_geometry_screen import save,csvsave
from scripts.prepare_v8_assets import digest

PREVIOUS=v2.OUT
OUT=PREVIOUS.with_name('anakin_pvst_candidate_review_v3')


def choose(selection,human_ids):
    order=sorted(p['metadata']['video-id'] for p in selection['prescreen']);random.Random(20260909).shuffle(order);tie={s:i for i,s in enumerate(order)}
    candidates=[];excluded=[]
    for p in selection['prescreen']:
        id=p['metadata']['video-id'];reason=None
        if id in selection['selected'] or 'an:'+id in human_ids:reason='previously previewed or reviewed; no repeat request'
        elif p['role'] not in ['train','candidate_unassigned']:reason='existing dev/test/conflict role, never move into training'
        elif p['camera_auto']['state'].startswith('可见'):reason='existing prefilter indicates global motion; not used to fill quota'
        if reason:excluded.append(dict(id=id,source_id=p['source_id'],reason=reason));continue
        candidates.append(copy.deepcopy(p))
    candidates.sort(key=lambda p:(not p['camera_auto']['stable_candidate'],p['role']!='train',tie[p['metadata']['video-id']]))
    selected=candidates[:12]
    return selected,excluded,[p['metadata']['video-id'] for p in candidates[12:]]


def render(records):
    prior.render(records,records)
    page=OUT/'index.html';s=page.read_text()
    s=s.replace('ANAKIN候选：人工确认前，不冻结新训练划分','ANAKIN第三批：新增训练来源候选，等待人工确认')
    s=s.replace('首批最多12对','本批最多12对').replace('人工三项：相机是否固定或近固定；具体对象及运动/遮挡/形态；mask与时间轴是否对应画面。','人工四项：机位；实际编辑内容；对象运动情况；mask与时间轴对应性。')
    s=s.replace('2. 实际修改哪个对象，涉及什么运动、遮挡或形态','2. 实际修改了什么对象/内容')
    s=s.replace('<label>3. mask与编辑时间轴是否对应画面','<label>3. 对象运动情况（走动、遮挡、相对运动或不确定）<textarea data-field="motion"></textarea></label><label>4. mask与编辑时间轴是否对应画面')
    s=s.replace("'pvst-candidate-'","'pvst-candidate-v3-'")
    s=s.replace('<p>人工四项：','<p>自动机位只作预筛；“不确定”不能解释成固定或移动。已有train角色保留，未分配仍未分配；不自动纳入训练。</p><p>人工四项：')
    page.write_text(s);v2.decorate_pending_page(page,records)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';started=time.monotonic();cv2.setNumThreads(2)
    if OUT.exists():
        if (OUT/'summary.json').exists() and v2.read(OUT/'summary.json')['status']=='COMPLETE':print('EXISTING_COMPLETE_REUSED');return
        raise RuntimeError('Existing incomplete review preserved')
    OUT.mkdir(parents=True)
    selection=v2.read(PREVIOUS/'selection.json');inventory=v2.read(PREVIOUS/'metadata_inventory.json')
    # Original annotations are read/copied, never modified or mapped by row order.
    human_paths=[v2.HUMAN,PREVIOUS/'human_review_original.json']
    human_records=[]
    for p in human_paths:
        if p.exists():human_records.append(dict(path=str(p),sha256=digest(p),raw=v2.read(p)))
    human_ids={x['pair'] for f in human_records for x in f['raw'] if any(x.get(k) for k in ['camera','edit','mask'])}
    save(OUT/'previous_human_records.json',human_records)
    selected,excluded,remaining=choose(selection,human_ids)
    assert len({p['source_id'] for p in selected})==len(selected)
    old=v2.read(prior.OUT/'inventory.json')+v2.read(PREVIOUS/'inventory.json')
    assert not {p['source_id'] for p in selected}&{r['source_id'] for r in old}
    chosen=dict(seed=20260909,previous_pool=len(selection['prescreen']),previous_previewed=len(selection['selected']),unpreviewed=len(selection['prescreen'])-len(selection['selected']),
        selected=[p['metadata']['video-id'] for p in selected],prescreen=selected,excluded=excluded,remaining=remaining,
        rule='Only existing v2 pool; exclude previous previews/human IDs, dev/test roles, observed global-motion tier. Stable tier then train role then sorted-ID seed20260909 shuffle. Uncertain camera retained for human; no model score/GT-area/visible-artifact selection.')
    save(OUT/'selection.json',chosen)
    save(OUT/'input_identity.json',dict(execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),previous_selection_sha256=digest(PREVIOUS/'selection.json'),previous_inventory_sha256=digest(PREVIOUS/'metadata_inventory.json'),official_revision=inventory['official_revision'],download_bytes=0))
    info=v2.read(v2.A/'references/dataset_info.json');assert info['sha']==inventory['official_revision']
    entries={e['rfilename']:{'size':e['size'],'lfs':e.get('lfs'),'blob_id':e['blobId']} for e in info['siblings']}
    # Repoint only the reused preview helpers in this independent CPU process.
    prior.OUT=OUT;v2.OUT=OUT;records=[];errors=[]
    for p in selected:
        try:r=v2.prepare(p,entries,info['sha'])
        except (OSError,ValueError,AssertionError) as error:
            errors.append(dict(id=p['metadata']['video-id'],error=str(error)));save(OUT/'asset_errors.json',errors);continue
        r['review_group']='training_source_candidate_not_frozen';r['human_semantic_alignment']='PENDING'
        records.append(r);save(OUT/'inventory.json',records);print('PREVIEW',r['pair_id'],r['mask']['status'],flush=True)
    assert records,'No previewable local sources'
    render(records)
    csvsave(OUT/'inventory.csv',[dict(pair_id=r['pair_id'],source_id=r['source_id'],role=r['split'],official_task=r['official_task'],camera_auto=r['camera'],mask_technical=r['mask']['status'],mask_semantic='PENDING_HUMAN',time_profile=r['mask']['profile'],real_path=r['media']['real']['path'],fake_path=r['media']['fake']['path'],mask_path=r['mask']['path']) for r in records])
    csvsave(OUT/'human_review_blank.csv',[dict(pair_id=r['pair_id'],camera='',edit='',motion='',mask='') for r in records])
    summary=dict(status='COMPLETE',exit_code=0,previous_pool=chosen['previous_pool'],unpreviewed=chosen['unpreviewed'],selected=chosen['selected'],prepared=len(records),technical_pass=sum(r['mask']['status']=='PASS' for r in records),technical_pending=[r['pair_id'] for r in records if r['mask']['status']!='PASS'],asset_errors=errors,
        stable_prefilter=sum(r['camera_auto']['stable_candidate'] for r in records),uncertain_camera=sum(not r['camera_auto']['stable_candidate'] for r in records),roles={role:sum(r['split']==role for r in records) for role in ['train','candidate_unassigned']},new_human_approved=0,download_bytes=0,seconds=time.monotonic()-started,models=0,depth=0,features=0)
    save(OUT/'summary.json',summary)
    text=['# ANAKIN第三批训练来源人工审阅','素材预览COMPLETE，退出码0；未自动纳入训练，人工语义确认仍待完成。',json.dumps(summary,ensure_ascii=False),
        '仅复用第二批34来源预筛与元数据，扣除已展示12来源后剩22；已有dev/test角色不回流训练，明显全局运动者不用于补足数量。稳定候选与机位不确定分别保留；未重跑背景估计、未下载、未读取模型成绩。官方LFS/SHA、完整解码PTS和逐帧mask数量/尺寸重新按本批指定资产核对，技术PASS不代表人工语义PASS。',
        '|pair|既有角色|官方编辑描述|机位预筛|技术mask|','|---|---|---|---|---|']
    for r in records:text.append('|'+r['pair_id']+'|'+r['split']+'|'+r['official_task']+'|'+r['camera_auto']['state']+'|'+r['mask']['status']+'|')
    text += ['已有人工原文保存在previous_human_records.json，保留原路径与SHA；未将未映射的“四条自然编辑”批次强行绑定ID。人工只需填写机位、实际编辑、对象运动、mask对应四项，不要求判断三维异常。自然协调编辑仍有价值。',
        '如有技术待核，页面仅独立显示源mask，不补帧、不伪造GT或叠加。官方mask无独立PTS，只核实数字帧号/尺寸与fake解码时间轴；real/fake时标不同明确标示，不能当像素配准。',
        '本批不足12是可用候选范围限制；不把已dev或明显移动样本凑为训练来源。原训练/开发/测试角色、旧成绩及原文件均不改。',str(OUT/'index.html')]
    (OUT/'report.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUT.exists():(OUT/'error.log').write_text(traceback.format_exc())
        raise
