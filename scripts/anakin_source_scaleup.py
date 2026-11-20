"""CPU source-group inventory and bounded review preparation; no model calls.

The CLI serves the source-coverage question, emits metadata and a human page,
and reuses the existing strict decode/mask/preview functions.
"""
import csv, json, random, re, os, time, shutil, subprocess, html
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
import cv2
from huggingface_hub import hf_hub_download
from scripts import anakin_candidate_review_v2 as old
from scripts import anakin_candidate_review as prior
from scripts.prepare_geometry_screen import save, csvsave
from scripts.prepare_v8_assets import digest, verify
from scripts.review_anakin import decode

OUT=old.OUT.with_name('anakin_pvst_source_pool_v1'); A=old.A; D=old.DERIVED

def source(url):
    u=urlparse(url.removeprefix('anakin:')); host=(u.hostname or '').lower().removeprefix('www.')
    if host in ('vimeo.com','player.vimeo.com'):
        ids=re.findall(r'\d+',u.path)
        return 'vimeo:'+ids[-1] if ids else None
    if host in ('youtube.com','youtu.be','m.youtube.com'):
        vid=u.path.strip('/') if host=='youtu.be' else parse_qs(u.query).get('v',[''])[0]
        return 'youtube:'+vid if vid else None
    return host+u.path.rstrip('/') if host and u.path.strip('/') else None

def protect(roles,official):
    if 'test' in official or any('test' in x for x in roles):return 'test_reserved'
    if any(x in roles for x in ('dev','validation','val')):return 'dev'
    if 'train' in roles:return 'train'
    return 'candidate_unassigned'

def tier(task):
    s=task.lower()
    if any(x in s for x in ('tattoo','text','logo','color','necklace','earring','badge','pattern','shirt','hat','mask color')):return 'appearance_reference'
    if any(x in s for x in ('remove','delete','erase')):return 'removal'
    if any(x in s for x in ('add','clone','copy','duplicate','insert')):return 'addition_copy'
    return 'replacement_local'

def content_priority(task):
    text=task.lower()
    reference=('picture','poster','print','emblem','sticker','chart','graph','screen','monitor','decorations','beads','pendant','chain','large letters','background to','instead of a mirror','back image','footage','eiffel tower','operator shadow')
    return 'appearance_background_reference' if any(word in text for word in reference) else 'object_edit_pending_human'

def validate_selection(rows, selected):
    protected={r['source_group_id'] for r in rows if r['role']=='test_reserved'}
    assert not protected.intersection(r['source_group_id'] for r in selected)
    assert len({r['source_group_id'] for r in selected})==len(selected)
    assert all(not r['role_conflict'] for r in selected)

def inventory():
    info=old.read(OUT/'references/dataset_info.json'); entries={e['rfilename']:{'size':e['size'],'lfs':e.get('lfs'),'blob_id':e['blobId']} for e in info['siblings']}
    assert info['sha']==old.read(A/'references/dataset_info.json')['sha'],'metadata revision changed; inspect first'
    checks={name:verify(A/'references'/name,entries[name]) for name in ('metadata.csv','README.md','ANAKIN.py')}
    assert all(v['status']=='VERIFIED' for v in checks.values()),checks
    save(OUT/'references/verification.json',checks)
    for name in ('metadata.csv','README.md','ANAKIN.py'):shutil.copyfile(A/'references'/name,OUT/'references'/name)
    allmeta=list(csv.DictReader((OUT/'references/metadata.csv').open())); byid={r['video-id']:r for r in allmeta}
    masked=[r for r in allmeta if r['has-masks']=='TRUE']; ids=[r['video-id'] for r in masked];random.Random(47).shuffle(ids); n=len(ids)
    splits={id:('train' if i<int(n*.7) else 'validation' if i<int(n*.8) else 'test') for i,id in enumerate(ids)}
    roles=defaultdict(set); provenance=defaultdict(list)
    def walk(x,path):
        if isinstance(x,dict):
            p=x.get('pair_id','').removeprefix('an:'); role=x.get('split')
            if p in byid and role:
                sid=source(byid[p]['url']);roles[sid].add(role);provenance[sid].append({'file':str(path),'id':p,'role':role})
            for v in x.values():
                if isinstance(v,(list,dict)):walk(v,path)
        elif isinstance(x,list):
            for v in x:walk(v,path)
    for pattern in ('*/manifest.json','*/protocol.json'):
        for p in sorted(D.glob(pattern)):
            try:walk(old.read(p),p)
            except (ValueError,OSError):pass
    historical=[]; reviewed=set(); prior_records={}
    for version in ('v1','v2','v3'):
        folder=OUT.with_name('anakin_pvst_candidate_review_'+version)
        p=folder/'inventory.json'
        if p.exists():
            for r in old.read(p):prior_records[r['pair_id'].removeprefix('an:')]=r;reviewed.add(r['pair_id'].removeprefix('an:'))
        for name in ('human_review_registered.json','human_review_original.json','previous_human_records.json'):
            p=folder/name
            if p.exists():historical.append({'path':str(p),'sha256':digest(p),'raw':old.read(p)})
    for p in [old.HUMAN,D/'pvst_xyz_source_expansion_v1/human_review.json']:
        if p.exists():historical.append({'path':str(p),'sha256':digest(p),'raw':old.read(p)})
    save(OUT/'human_evidence.json',{'verbatim_sources':historical,'unmapped_batch':'4条自然编辑：3固定、1轻微移动，均协调难辨；ID对应未确认，不扩展逐ID结论'})
    official=defaultdict(set)
    for r in masked:official[source(r['url'])].add(splits[r['video-id']])
    rows=[]
    for r in masked:
        id=r['video-id']; sid=source(r['url']); role=protect(roles[sid],official[sid]); assets={}
        for folder,ext in [('trimmed','mp4'),('edited','mp4'),('masks','zip')]:
            f=f'{folder}/{id}.{ext}';p=A/f
            assets[folder]={'official_path':f,'path':str(p if p.exists() else OUT/'assets'/f),'exists':p.is_file(),'bytes':entries.get(f,{}).get('size'),'published':f in entries}
        rows.append({'id':id,'source_group_id':sid,'url':r['url'],'official_split':splits[id],'historical_roles':sorted(roles[sid]),'role':role,'role_conflict':bool('train' in roles[sid] and any(v in roles[sid] for v in ['dev','validation','val'])),'task':r['task'],'type':r['manipulation-type'],'edit_tier':tier(r['task']),'previously_previewed':id in reviewed,'assets':assets,'role_evidence':provenance[sid]})
    eligible=[r for r in rows if r['source_group_id'] and r['role']!='test_reserved' and not r['previously_previewed'] and not r['role_conflict'] and r['edit_tier']!='appearance_reference' and all(a['published'] for a in r['assets'].values())]
    random.Random(20260909).shuffle(eligible);eligible.sort(key=lambda r:({'addition_copy':0,'removal':1,'replacement_local':2}[r['edit_tier']],not all(a['exists'] for a in r['assets'].values())))
    selected=[];seen={source(byid[i]['url']) for i in reviewed}
    for r in eligible:
        if r['source_group_id'] not in seen:selected.append(r);seen.add(r['source_group_id'])
        if len(selected)==120:break
    save(OUT/'metadata_inventory.json',{'revision':info['sha'],'all_metadata_records':len(allmeta),'masked_records':len(rows),'source_groups':len({r['source_group_id'] for r in rows if r['source_group_id']}),'rows':rows})
    validate_selection(rows,selected)
    save(OUT/'selection.json',{'seed':20260909,'rule':'previous preview source exclusion; all historical/official test source quarantine; metadata object edits first; stable seeded tie; one edit per original; no scores/area selection','selected':selected,'eligible_records':len(eligible)})
    return rows,selected,entries,info

def render(results):
    prior.OUT=OUT;prior.render(results,results);old.OUT=OUT;old.decorate_pending_page(OUT/'index.html',results)
    page=OUT/'index.html';s=page.read_text();
    clocks={}
    for r in results:
        pair=r['pair_id'];folder=OUT/pair.replace(':','_');m=r['media']['fake'];clocks[pair]=[x['seconds'] for x in m['pts']]
        controls=f'<p>fake零基帧 <input class="frame-index" type="number" min="0" max="{m["frame_count"]-1}" value="0"><button class="jump-frame">跳到原生帧</button><span class="frame-pts"></span>（real仅相对播放时间联动，不假定逐帧配准）</p>'
        extra=''
        if r['mask']['status']!='PASS':
            keep={0,m['frame_count']//2,m['frame_count']-1}
            images=decode(Path(m['path']),keep)[1] if any(not (folder/f'unpaired_rgb_{i:04d}.png').exists() for i in keep) else {i:None for i in keep}
            for i,image in sorted(images.items()):
                name=f'unpaired_rgb_{i:04d}.png'
                if image is not None:image.thumbnail((640,640));image.save(folder/name)
                extra+=f'<p>独立fake RGB帧{i}，PTS {m["pts"][i]["seconds"]:.6f}s；不与旁边未对齐mask配对</p><img loading="lazy" src="{folder.name}/{name}">'
        details=html.escape(json.dumps({'source_group_id':r.get('source_group_id'),'source_url':r.get('source_url'),'revision':r.get('official_revision'),'real_path':r['media']['real'].get('path'),'fake_path':m.get('path'),'mask_path':r['mask'].get('path'),'real_frames':r['media']['real']['frame_count'],'fake_frames':m['frame_count'],'native_hw':m.get('sizes_hw'),'fps':m.get('average_rate')},ensure_ascii=False,indent=2))
        marker='<section data-pair="'+pair+'">';s=s.replace(marker,marker+controls+'<details><summary>来源与本地素材身份</summary><pre>'+details+'</pre></details>'+extra,1)
    s=s.replace('首批最多12对','本批来源扩展').replace('人工三项：','人工四项（上方批量表为正式填写入口）：')
    rows=[{'pair':r['pair_id'],'source_group_id':r['source_group_id'],'role':r['split'],'type':r.get('content_priority','')+'/'+r.get('edit_tier',r['generator']),'technical':r['mask']['status']} for r in results]
    ui='''<div id="batch"><h2>批量审阅</h2><p>默认未填；只应用非空字段。自然协调不是排除条件。导出包含精确ID、批次覆盖和旧人工原文。</p><label>角色<select id="roleFilter"><option value="">全部</option></select></label><label>类型<select id="typeFilter"><option value="">全部</option></select></label><label>技术<select id="techFilter"><option value="">全部</option></select></label><button id="selectVisible">选择可见</button><button id="clearSelected">取消选择</button><div id="common"></div><button id="applyBatch">应用到已选</button><button id="downloadReview">导出human_review.json</button><div id="reviewRows"></div></div>'''
    script='''<script>
const clocks=CLOCKS;
for(const sec of document.querySelectorAll('section')){const i=sec.querySelector('.frame-index'),b=sec.querySelector('.jump-frame');b.onclick=()=>{const n=Math.max(0,Math.min(clocks[sec.dataset.pair].length-1,Math.round(Number(i.value))));i.value=n;const pts=clocks[sec.dataset.pair][n];sec.querySelector('.frame-pts').textContent='实际fake PTS '+pts.toFixed(6)+'s';for(const v of sec.querySelectorAll('video')){v.pause();v.currentTime=pts-clocks[sec.dataset.pair][0]}}}
const records=RECORDS, evidence=EVIDENCE, fields={camera:['固定','近固定','移动','不确定'],edit_type:['添加','移除','替换或局部修改','其他'],motion:['是','否','不确定'],mask:['对应','部分对应','不对应','未确认']};let events=[];
const box=document.getElementById('reviewRows'),state={};
function editor(parent,key,data){for(const [name,opts] of Object.entries(fields)){const el=document.createElement('select');el.dataset.name=name;el.innerHTML='<option value="">'+({camera:'机位',edit_type:'实际编辑',motion:'对象是否运动',mask:'mask语义'}[name])+'：未填</option>'+opts.map(x=>'<option>'+x+'</option>').join('');el.value=data[name]||'';el.onchange=()=>{data[name]=el.value;persist()};parent.append(el)}const t=document.createElement('input');t.placeholder='实际编辑描述 / 原文';t.dataset.name='description';t.value=data.description||'';t.oninput=()=>{data.description=t.value;persist()};parent.append(t)}
function persist(){try{localStorage.setItem('pvst-source-pool-v1',JSON.stringify({state,batch_events:events}))}catch(e){}}
let cached={};try{const saved=JSON.parse(localStorage.getItem('pvst-source-pool-v1')||'{}');cached=saved.state||saved;events=saved.batch_events||[]}catch(e){}
for(const r of records){state[r.pair]=cached[r.pair]||{};const d=document.createElement('div');d.dataset.pair=r.pair;d.innerHTML='<input type="checkbox"><a href="#'+r.pair+'">'+r.pair+'</a> '+r.role+' / '+r.type+' / '+r.technical+' ';editor(d,r.pair,state[r.pair]);box.append(d);const section=document.querySelector('section[data-pair="'+r.pair+'"]');section.id=r.pair;}
const common={};editor(document.getElementById('common'),'batch',common);
for(const [id,key] of [['roleFilter','role'],['typeFilter','type'],['techFilter','technical']]){const el=document.getElementById(id);for(const v of [...new Set(records.map(r=>r[key]))])el.add(new Option(v,v));el.onchange=filter}
function filter(){for(const r of records){const show=[['roleFilter','role'],['typeFilter','type'],['techFilter','technical']].every(([id,key])=>!document.getElementById(id).value||document.getElementById(id).value===r[key]);box.querySelector('[data-pair="'+r.pair+'"]').hidden=!show;document.getElementById(r.pair).hidden=!show}}
document.getElementById('selectVisible').onclick=()=>{for(const d of box.children)if(!d.hidden)d.querySelector('input').checked=true};document.getElementById('clearSelected').onclick=()=>{for(const d of box.children)d.querySelector('input').checked=false};
document.getElementById('applyBatch').onclick=()=>{const values=Object.fromEntries(Object.entries(common).filter(([k,v])=>v!==''));const ids=[];for(const d of box.children)if(d.querySelector('input').checked){ids.push(d.dataset.pair);Object.assign(state[d.dataset.pair],values);for(const e of d.querySelectorAll('[data-name]'))e.value=state[d.dataset.pair][e.dataset.name]||''}events.push({ids,values,at:new Date().toISOString()});persist()};
document.getElementById('downloadReview').onclick=()=>{const value={schema:'pvst-source-pool-human-v1',records:records.map(r=>({...r,...state[r.pair]})),batch_events:events,previous_verbatim:evidence};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.download='human_review.json';a.click()};
</script>'''.replace('CLOCKS',json.dumps(clocks)).replace('RECORDS',json.dumps(rows,ensure_ascii=False)).replace('EVIDENCE',json.dumps(old.read(OUT/'human_evidence.json'),ensure_ascii=False).replace('</','<\\/'))
    s=re.sub(r'<label>[^<]*<textarea data-field="[^"]+"></textarea></label>','',s)
    s=s.replace('<button id="export">导出人工JSON</button>','').replace('<script>','<script>\n',1);s=s.replace('document.getElementById("export").onclick=','const legacyExport=')
    s=s.replace('href="inventory.csv"','href="candidate_manifest.csv"')
    s=re.sub(r'const legacyExport=[^\n]*\n','',s)
    links='<p><a href="source_pool_manifest.csv">全量来源清单</a> / <a href="partition_draft.json">划分草案（未冻结）</a> / <a href="human_evidence.json">原人工证据</a> / <a href="../anakin_pvst_candidate_review_v1/index.html">旧批次1</a> / <a href="../anakin_pvst_candidate_review_v2/index.html">旧批次2</a> / <a href="../anakin_pvst_candidate_review_v3/index.html">旧批次3</a></p>'
    page.write_text(s.replace('<h1>',links+ui+'<h1>',1)+script)

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cv2.setNumThreads(2);start=time.time()
    if (OUT/'selection.json').exists():
        raise RuntimeError('Existing frozen selection/results preserved; use --finalize only after completed preparation. Do not overwrite or reselect.')
    (OUT/'references').mkdir(parents=True,exist_ok=True)
    if not (OUT/'references/dataset_info.json').exists():
        from huggingface_hub import HfApi
        remote=HfApi(token=False).dataset_info('AlexBlck/ANAKIN',files_metadata=True)
        save(OUT/'references/dataset_info.json',{'sha':remote.sha,'siblings':[{'rfilename':e.rfilename,'size':e.size,'blobId':e.blob_id,'lfs':dict(e.lfs) if e.lfs else None} for e in remote.siblings]})
    rows,selected,entries,info=inventory(); downloads=[]
    jobs=[a for r in selected for a in r['assets'].values() if not a['exists']]
    save(OUT/'download_plan.json',{'files':jobs,'bytes':sum(a['bytes'] for a in jobs)})
    def fetch(a):
        p=Path(a['path']);t=time.time();hf_hub_download('AlexBlck/ANAKIN',a['official_path'],repo_type='dataset',revision=info['sha'],local_dir=OUT/'assets',token=False)
        return {'file':a['official_path'],'bytes':p.stat().st_size,'seconds':time.time()-t}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch,jobs):downloads.append(result);save(OUT/'downloads.json',downloads);print('download',result,flush=True)
    # Existing helper expects canonical folder layout. Symlinks do not alter originals.
    root=OUT/'assets'
    for r in selected:
        for a in r['assets'].values():
            target=root/a['official_path'];target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.symlink_to(a['path'])
    old.A=root;old.OUT=OUT;prior.OUT=OUT
    results=[];failures=[]
    for r in selected:
        try:
            id=r['id'];meta=next(x for x in csv.DictReader((A/'references/metadata.csv').open()) if x['video-id']==id)
            path=root/'trimmed'/f'{id}.mp4'
            import av, numpy as np
            with av.open(str(path)) as c:n=sum(1 for _ in c.decode(video=0))
            m,ims=decode(path,set(np.linspace(0,n-1,5,dtype=int).tolist()))
            p={'metadata':meta,'source_id':'anakin:'+r['url'],'role':r['role'],'role_provenance':r['role_evidence'],'real_meta':m,'camera_auto':old.camera_check(sorted(ims.items()))}
            result=old.prepare(p,entries,info['sha']);result['source_group_id']=r['source_group_id'];result['edit_tier']=r['edit_tier'];results.append(result)
            save(OUT/'inventory.json',results);print('prepared',id,result['mask']['status'],flush=True)
        except Exception as e:
            failures.append({'id':r['id'],'error':repr(e)});save(OUT/'failures.json',failures);print('FAILED',r['id'],repr(e),flush=True)
    render(results)
    csvsave(OUT/'candidate_manifest.csv',[{'pair_id':r['pair_id'],'source_group_id':r['source_group_id'],'role':r['split'],'task':r['official_task'],'tier':r['edit_tier'],'technical':r['mask']['status'],'camera_auto':r['camera'],'human_semantics':'未确认'} for r in results])
    draft=[]
    for r in rows:
        proposal=r['role']
        if proposal=='candidate_unassigned':proposal='dev_draft' if r['official_split']=='validation' else 'train_draft'
        draft.append({'id':r['id'],'source_group_id':r['source_group_id'],'proposal':proposal,'basis':'historical role and official source-group quarantine; not frozen','media_allowed':r['role']!='test_reserved','semantic_quality':'not certified'})
    save(OUT/'partition_draft.json',draft)
    summary={'status':'COMPLETE','exit_code':0,'records':len(rows),'source_groups':len({r['source_group_id'] for r in rows}),'selected_sources':len(selected),'prepared':len(results),'technical_pass':sum(r['mask']['status']=='PASS' for r in results),'technical_unresolved':sum(r['mask']['status']!='PASS' for r in results),'failures':failures,'roles':dict(Counter(r['split'] for r in results)),'tiers':dict(Counter(r['edit_tier'] for r in results)),'stable_auto':sum(r['camera_auto']['stable_candidate'] for r in results),'download_bytes':sum(x['bytes'] for x in downloads),'seconds':time.time()-start,'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'neural_forwards':0,'human_confirmed_new':0}
    save(OUT/'summary.json',summary);print(summary,flush=True)


def finalize():
    """Attach prior evidence and produce the small reproducible handover tables."""
    command="CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/rgbd-packages:. /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u -m scripts.anakin_source_scaleup"
    (OUT/'commands.txt').write_text('cd '+str(Path.cwd())+'\n'+command+'\n# --finalize only regenerates small handover artifacts\n')
    save(OUT/'code_identity.json',{'base_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'entry':str(Path(__file__).resolve()),'entry_sha256':digest(Path(__file__)),'method_changed':False,'cpu_only':True})
    inv=old.read(OUT/'metadata_inventory.json');validate_selection(inv['rows'],old.read(OUT/'selection.json')['selected']); results=old.read(OUT/'inventory.json')
    for result in results:
        result['content_priority']=content_priority(result['official_task'])
        save(OUT/result['pair_id'].replace(':','_')/'identity.json',result)
    save(OUT/'inventory.json',results)
    byid={r['pair_id'].removeprefix('an:'):r for r in results}
    prior_byid={}; historical=[]
    for version in ('v1','v2','v3'):
        p=OUT.with_name('anakin_pvst_candidate_review_'+version)/'inventory.json'
        if p.exists():
            for r in old.read(p):prior_byid[r['pair_id'].removeprefix('an:')]=(r,str(p))
    evidence=old.read(OUT/'human_evidence.json');human=defaultdict(list)
    def humanwalk(x,p):
        if isinstance(x,dict):
            pair=x.get('pair',x.get('pair_id',''))
            if pair.startswith('an:'):human[pair.removeprefix('an:')].append({'source':p,'raw':x})
            else:
                for k,v in x.items():
                    if k.startswith('an:'):human[k.removeprefix('an:')].append({'source':p,'raw':v})
            for v in x.values():
                if isinstance(v,(dict,list)):humanwalk(v,p)
        elif isinstance(x,list):
            for v in x:humanwalk(v,p)
    for e in evidence['verbatim_sources']:humanwalk(e['raw'],e['path'])
    for r in inv['rows']:
        id=r['id'];r['human_evidence']=human[id]
        for asset in r['assets'].values():
            asset['exists_before']=asset['exists'];asset['exists']=Path(asset['path']).is_file()
        mask_values=[]
        for e in human[id]:
            raw=e['raw']
            if isinstance(raw,dict):
                for key in ('mask','raw_mask','human_semantic_mask'):
                    if isinstance(raw.get(key),str):mask_values.append(raw[key])
        r['content_priority']=content_priority(r['task'])
        r['human_mask_status']='not_corresponding' if any(v in ('不对应','does_not_correspond') for v in mask_values) else 'partial' if '部分对应' in mask_values else 'corresponding' if any(v in ('对应','corresponds') for v in mask_values) else 'unconfirmed'
        r['technical_status']='NOT_CHECKED'
        if id in prior_byid:
            rr,p=prior_byid[id];r['technical_status']=rr['mask']['status'];r['technical_evidence']=p;r['camera_evidence']=rr.get('camera')
        if id in byid:
            rr=byid[id];r['technical_status']=rr['mask']['status'];r['technical_evidence']=str(OUT/'inventory.json');r['camera_evidence']=rr['camera'];r['mask_profile']=rr['mask']['profile']
        r['candidate_use']='metadata_only_test' if r['role']=='test_reserved' else 'pending_human'
        if id in byid and r['technical_status']=='PASS':
            camera=byid[id]['camera_auto']
            r['candidate_use']='PVST_camera_prescreen_only' if camera['stable_candidate'] else 'spatial_moving_prescreen' if camera['state'].startswith('可见') else 'camera_uncertain'
        if r['technical_status'] not in ('PASS','NOT_CHECKED'):r['candidate_use']='technical_pending'
        if id in byid and r['technical_status']=='PASS' and r['content_priority']=='appearance_background_reference':r['candidate_use']='appearance_background_spatial_reference'
        if r['human_mask_status']=='not_corresponding':r['candidate_use']='human_mask_rejected_pending_review'
        r['preview_path']=str(OUT/'index.html')+'#an:'+id if id in byid else None
    remaining=[r for r in inv['rows'] if not r['previously_previewed'] and r['id'] not in byid and r['role']!='test_reserved' and not r['role_conflict'] and r['edit_tier']!='appearance_reference']
    csvsave(OUT/'remaining_metadata_candidates.csv',[{k:r[k] for k in ['id','source_group_id','role','task','edit_tier']} for r in remaining])
    fresh=sorted([r for r in inv['rows'] if r['role']=='test_reserved' and not r['historical_roles']],key=lambda r:r['id'])
    random.Random(20260909).shuffle(fresh)
    save(OUT/'fresh_test_draft.json',{'seed':20260909,'available_metadata_only_sources':len(fresh),'proposed_ids':[r['id'] for r in fresh[:20]],'status':'identity metadata only; semantics unconfirmed; no media access; not frozen'})
    save(OUT/'metadata_inventory.json',inv)
    csvsave(OUT/'source_pool_manifest.csv',[{k:r.get(k) for k in ('id','source_group_id','url','official_split','historical_roles','role','role_conflict','task','type','edit_tier','technical_status','content_priority','human_mask_status','candidate_use','preview_path')} for r in inv['rows']])
    summary=old.read(OUT/'summary.json')
    initial_selection=old.read(OUT/'selection.json')['selected']
    summary['content_priority']=dict(Counter(r['content_priority'] for r in results));summary['technical_pass_content_priority']=dict(Counter(r['content_priority'] for r in results if r['mask']['status']=='PASS'))
    summary['reused_files']=sum(a['exists'] for r in initial_selection for a in r['assets'].values())
    summary['complete_reused_triplets']=sum(all(a['exists'] for a in r['assets'].values()) for r in initial_selection)
    proposals={r['id']:r['proposal'] for r in old.read(OUT/'partition_draft.json')}
    summary['object_pass_draft_roles']=dict(Counter(proposals[r['id']] for r in inv['rows'] if r['id'] in byid and r['technical_status']=='PASS' and r['content_priority']=='object_edit_pending_human'))
    summary['new_technical_pass_draft_roles']=dict(Counter(proposals[r['id']] for r in inv['rows'] if r['id'] in byid and r['technical_status']=='PASS'))
    recommendations=[{'id':r['id'],'source_group_id':r['source_group_id'],'role_draft':proposals[r['id']],'use':r['candidate_use'],'tier':r['edit_tier'],'human_mask':r['human_mask_status'],'task':r['task']} for r in inv['rows'] if r['id'] in byid and r['technical_status']=='PASS']
    csvsave(OUT/'recommendations.csv',recommendations)
    summary['registered_old_technical']=dict(Counter(r['technical_status'] for r in inv['rows'] if r['id'] in prior_byid));summary['new_use']=dict(Counter(r['candidate_use'] for r in inv['rows'] if r['id'] in byid));summary['all_role_groups']=dict(Counter(r['role'] for r in inv['rows']));summary['individual_human_evidence_ids']=sum(bool(v) for v in human.values());summary['human_mask_status']=dict(Counter(r['human_mask_status'] for r in inv['rows']));summary['available_masked_assets']={folder:sum(r['assets'][folder]['exists'] for r in inv['rows']) for folder in ['trimmed','edited','masks']};summary['download_files']=len(old.read(OUT/'downloads.json'));summary['old_asset_files']={folder:len(list((A/folder).glob('*.'+ext))) for folder,ext in [('trimmed','mp4'),('edited','mp4'),('masks','zip')]};save(OUT/'summary.json',summary)
    # The historical problems remain unresolved; no fabricated mapping fixes.
    pending=[{'id':r['id'],'technical_status':r['technical_status'],'evidence':r.get('technical_evidence'),'note':'没有新官方映射依据；原判定保留，不裁帧/重编号/缩放'} for r in inv['rows'] if r['technical_status'] not in ('PASS','NOT_CHECKED')];save(OUT/'technical_pending.json',pending)
    report=f'''# ANAKIN独立来源扩充：等待人工审阅

本轮只做CPU媒体/官方mask核对与页面生成，神经模型、深度、特征提取、训练均为0。未改变旧数据角色与成绩。

官方版本 `{inv['revision']}`；metadata总记录{inv['all_metadata_records']}，带mask记录{inv['masked_records']}；规范化原视频URL得到{inv['source_groups']}个独立来源。只计可核验URL，不以不同剪辑文件当新增来源。带mask官方记录split按官方loader seed47复现，297/43/85。

历史manifest/protocol及官方test按整个原视频隔离，测试保护{summary['all_role_groups'].get('test_reserved',0)}来源：仅身份元数据，未下载/解码/展示。历史角色证据逐条保存在metadata_inventory.json，train/dev冲突保留并阻止进入新训练队列。当前4开发继续是历史开发，不称最终测试。

## 实际准备

|项目|数量|
|---|---:|
|本轮新候选来源|{summary['selected_sources']}|
|可打开的预览来源|{summary['prepared']}|
|新候选技术PASS|{summary['technical_pass']}|
|新候选技术待核|{summary['technical_unresolved']}|
|预备失败|{len(summary['failures'])}|
|自动固定倾向（非人工确认）|{summary['stable_auto']}|
|新增逐ID人工确认|0|
|新获取必要资产字节|{summary['download_bytes']}|
|下载与CPU准备秒数|{summary['seconds']:.1f}|

本批复用{summary['reused_files']}个既有文件（{summary['complete_reused_triplets']}套完整triplet，其余为部分已有资产），其余仅获取缺失指定资产。旧资产文件数{summary['old_asset_files']}，带mask记录中实际现有文件{summary['available_masked_assets']}（新取得{summary['download_files']}个必要文件）。逐ID人工mask原话整理{summary['human_mask_status']}；不把有相机/编辑描述就算mask已确认。旧技术记录分布{summary['registered_old_technical']}。新来源角色{summary['roles']}；官方描述预筛覆盖{summary['tiers']}。类别仅是元数据提示，不把“splicing”全当运动对象添加，不虚构人工描述。进一步按描述单列{summary['content_priority']}；技术PASS为{summary['technical_pass_content_priority']}。图表、屏幕、配饰和背景替换保留为较低优先级参照，不能拿它们凑运动对象主组。

固定机位主组尚不能最终计数：新人工确认0。技术PASS且传统背景匹配稳定的待审组与其他空间补充/机位待核组分别为{summary['new_use']}。移动机位不自动纳入PVST固定机位主组；匹配不足记不确定，不当作移动结论。

## 推荐审阅顺序与限制

先看技术PASS的添加/复制来源，弥补旧训练以移除为主的覆盖；其次看不同场景的移除；其余局部修改作参照。页面按ID展示、不按旧模型成绩排序，可过滤角色/编辑类型/技术状态。协调自然的编辑同样保留。

有编号和尺寸明确的mask才生成叠加/时间轴，未明确者只显示官方原掩膜。mask的PTS由所对应fake帧承接，没有独立标注时钟。真伪联动是片内时间联动，不承诺像素配准。mask非空变化不等于真实编辑起止。原40y1x9_fake[6,10)监督排除保持原样。

新候选由原120来源清单冻结后准备，无GT面积/预测分数筛选。选定120是批量人工候选规模，技术问题和真实机位仍会减少最终数量；不能写成已获得120个合格训练来源。未复审的旧问题没有官方映射新证据，继续待核，见technical_pending.json。

人工证据保留在human_evidence.json，逐ID只引用明确ID记录；未映射四条自然编辑批次及第三批共同意见均保留原范围，不扩展成每ID mask通过。新增四字段默认未填写，支持选中ID批量写入及逐项例外修正，导出具体覆盖ID与原人工证据。

## 划分与下一轮草案（未冻结、未执行）

partition_draft.json保留已有train/dev/test，未分配来源参考官方train/validation提出train_draft/dev_draft；94个测试保护来源仅作元数据预留，语义质量未确认，不是已验收测试集。其中未有历史角色的{len(fresh)}个官方测试来源中，固定seed提出20个fresh_test_draft.json，只有身份元数据，不作合格测试集宣称。人工确认前不自动纳入训练。全部真假/剪辑/编辑版本依原视频分组，现有开发不回流训练。

50–100独立训练来源是后续组织目标，当前可扩充规模上界为本批技术PASS、人工尚待确认的来源，加既有可用train；本批新增技术PASS按草案角色为{summary['new_technical_pass_draft_roles']}，其中对象编辑待审且技术PASS的草案角色为{summary['object_pass_draft_roles']}。以上仅是人工确认前上界；真实可达规模取决于机位/对象/掩膜语义。添加/复制官方描述覆盖增加，但尚不能承诺获得多少运动对象添加实例。素材获取之外的主要限制是映射与人工语义/机位确认。

唯一下一步：人工确认后冻结来源隔离清单，先固定现有空间基线，按无放回完整遍历（候选10轮，需结合实际帧量与资源事先定稿）验证多来源定位；不机械套12845步。随后图像/XYZ同帧/XYZ时空共用数据和视觉底座；独立测试不调参。本轮不选择新解码器、BN组合或时序结构。

## 位置与执行

审阅页：`{OUT/'index.html'}`

全量清单：`{OUT/'source_pool_manifest.csv'}`

下载逐文件记录：downloads.json；完整decode PTS及官方mask检查：各an_ID/identity.json。原始素材优先引用既有ANAKIN目录；新增必要trimmed/edited/masks在本目录assets/。新增字节为取得文件的逻辑大小，不含协议开销或重传。

入口：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=.../rgbd-packages:. python -m scripts.anakin_source_scaleup`，一次准备后可用`--finalize`复用已有预览重建小型汇总；缺少待核RGB固定图时才补解码。完整命令见同目录commands.txt。计算状态{summary['status']}，退出码{summary['exit_code']}；Git发布状态单独登记于publication.json。用户人工审阅为本轮终点。
'''
    (OUT/'report.md').write_text(report);Path('docs/pvst/anakin_source_scaleup_v1.md').write_text(report)
    render(results)

if __name__=='__main__':
    import sys
    if '--finalize' in sys.argv:finalize()
    else:run();finalize()

