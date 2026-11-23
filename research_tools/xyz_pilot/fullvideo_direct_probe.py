"""Bounded two-update full-video resource probe; never formal training.

Calls FullVideoDirect and original SpatialBaseline. Emits independent per-case
results; no dev scoring, MoGe, long-task polling or automatic retry/fallback.
"""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from .ainpaint_spatial import build, load_adaptation, WEIGHT, DINO, AINPAINT
from .fullvideo_direct_model import FullVideoDirect, prepare_frame, video_loss, MODES, SCALE

DATA = Path('/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived')
OUT = DATA / 'pvst_fullvideo_direct_prototype_v1'
OLD = DATA / 'pvst_xyz_temporal_pilot_v1'
SPATIAL = DATA / 'pvst_anakin_expanded_spatial_training_v1'
CASES = ('4ym5n4_fake', '40y1x9_fake', '9mygy9_fake')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, x in sorted(state.items()):
        h.update(name.encode())
        h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def progress(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def seed():
    random.seed(20260909)
    np.random.seed(20260909)
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)


def sync():
    torch.cuda.synchronize()


def memory():
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def identity(row, cp):
    return dict(schema=1, case=row['video_id'], video_sha256=row['video_sha256'],
                mask_sha256=row['mask_sha256'], source_manifest_sha256=digest(OLD/'manifest.json'),
                adaptation_sha256=digest(SPATIAL/'epoch01.pt'),
                spatial_identity=cp['identity'], epoch=cp['epoch'], updates=cp['updates'],
                preprocessing='original SpatialBaseline RGB ImageNet / pad14 / BF16 eval',
                scale=SCALE, h=.5, origin=[0,0,0], T=row['actual_frames'],
                provider_config_sha256=digest(OLD/'provider_config.json'),
                pts=row['pts'], work_hw=row['work_hw'], native_hw=row['native_hw'])


def prepare(row, ident, root):
    folder = root/'observations'/row['video_id']
    meta = folder/'identity.json'
    folder.mkdir(parents=True, exist_ok=True)
    if meta.exists():
        assert json.loads(meta.read_text()) == ident, 'Frozen cache identity mismatch'
    else:
        assert not list(folder.iterdir()), 'Unidentified cache directory preserved'
        save(meta, ident)
    missing = [i for i in range(row['actual_frames']) if not (folder/f'{i:05d}.pt').exists()]
    if not missing:
        progress('reuse_frozen_cache', case=row['video_id'], frames=row['actual_frames'])
        return folder, dict(new_frozen_forwards=0, reused_frames=row['actual_frames'], seconds=0.)
    start = time.monotonic()
    cp = torch.load(SPATIAL/'epoch01.pt', map_location='cpu', weights_only=False)
    assert cp['epoch'] == 1 and cp['condition'] == 'EXPANDED_SPATIAL'
    assert digest(WEIGHT) == cp['identity']['generic_weight_sha256']
    for repo, key in [(DINO, 'official_DINO_revision'), (AINPAINT, 'official_AINPAINT_revision')]:
        assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == cp['identity'][key]
    model = build()
    load_adaptation(model, cp['adaptation_state'])
    del cp
    model.requires_grad_(False).eval().cuda()
    assert all(not m.training for m in model.modules())
    frozen_before = state_hash(model.state_dict())
    captured = {}
    def detail_hook(_m, args):
        captured['detail'] = args[0].detach()
    handles = []
    # SpatialBaseline calls forward_features directly, so module forward hooks
    # do not fire. Hook the existing token_grid conversion locally instead.
    from . import ainpaint_spatial as spatial
    original_grid = spatial.token_grid
    def capture_grid(result, hw):
        grid = original_grid(result, hw)
        captured['patch'] = grid.detach()
        return grid
    handles.append(model.head.predictor.register_forward_pre_hook(detail_hook))
    spatial.token_grid = capture_grid
    try:
        with torch.no_grad():
            for i in missing:
                e = torch.load(OLD/'cache'/row['video_id']/f'{i:05d}.pt', map_location='cpu', weights_only=True)
                assert e['frame_index'] == i and e['pts'] == row['pts'][i]
                assert e['native_hw'] == row['native_hw'] and list(e['rgb'].shape[:2]) == row['work_hw']
                h, w = row['work_hw']; padded = [((h+13)//14)*14, ((w+13)//14)*14]
                captured.clear()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model([dict(rgb=e['rgb'].cuda())])['pixel']
                assert captured['patch'].shape == (1,1024,padded[0]//14,padded[1]//14)
                assert captured['detail'].shape[1] == 32 and z.shape == (h,w)
                f = {k:e[k] for k in ['xyz_provider','valid','gt_work','frame_index','pts','supervised']}
                f.update(patch=captured['patch'][0].cpu(), detail=captured['detail'][0].cpu(),
                         z0=z.float().cpu(), dino_padded_hw=padded, work_hw=[h,w],
                         native_hw=row['native_hw'], video_id=row['video_id'])
                assert all(torch.isfinite(f[k]).all() for k in ['patch','detail','z0'])
                tmp = folder/f'{i:05d}.tmp'; torch.save(f, tmp);tmp.replace(folder/f'{i:05d}.pt')
                progress('frozen_observation', case=row['video_id'], frame=i+1, T=row['actual_frames'])
        sync()
        assert model.calls == len(missing)
        assert frozen_before == state_hash(model.state_dict()), 'Frozen weights/buffers changed'
    finally:
        spatial.token_grid = original_grid
        for handle in handles:handle.remove()
    count = model.calls
    del model, captured, handles, e, f, z
    gc.collect();torch.cuda.empty_cache()
    cost = dict(new_frozen_forwards=count, reused_frames=row['actual_frames']-count,
                seconds=time.monotonic()-start, frozen_parameters_and_buffers_unchanged=True,
                cache_bytes=sum((folder/f'{i:05d}.pt').stat().st_size for i in range(row['actual_frames'])))
    save(folder/'preparation.json',cost)
    return folder, cost


def load_frames(row, folder):
    frames=[]
    for i in range(row['actual_frames']):
        e=torch.load(folder/f'{i:05d}.pt', map_location='cpu', weights_only=True)
        assert e['video_id']==row['video_id'] and e['frame_index']==i and e['pts']==row['pts'][i]
        assert e['native_hw']==row['native_hw'] and e['work_hw']==row['work_hw']
        h,w=e['work_hw'];assert e['dino_padded_hw']==[((h+13)//14)*14,((w+13)//14)*14]
        assert e['z0'].shape==(h,w) and e['patch'].shape==(1024,e['dino_padded_hw'][0]//14,e['dino_padded_hw'][1]//14)
        assert e['gt_work'].shape==e['valid'].shape==(h,w)
        assert e['supervised']==(not(row['video_id']=='40y1x9_fake' and 6<=i<10))
        frames.append(prepare_frame(e))
    return frames


def gradients(model):
    groups={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            group=name.split('.')[0]
            groups[group]=groups.get(group,0.)+float(p.grad.float().square().sum())
    return {k:v**.5 for k,v in groups.items()}


def measure(frames, mode, initial, chunk, publish):
    seed();model=FullVideoDirect(chunk).cuda();model.load_state_dict(initial,strict=True)
    assert state_hash(model.state_dict()) == state_hash(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    counts=[len(f['keys']) for f in frames]
    result=dict(mode=mode,T=len(frames),supervised_frames=sum(f['supervised'] for f in frames),
                frame_counts=counts,N_min=min(counts),N_max=max(counts),M=sum(counts),
                effective_points=sum(len(f['points']) for f in frames),
                initial_sha256=state_hash(initial),query_chunk=chunk,activation_checkpoint='non-reentrant',
                free_gpu_bytes_at_start=torch.cuda.mem_get_info()[0],steps=[],status='RUNNING')
    publish(result)
    torch.cuda.reset_peak_memory_stats()
    for step in (1,2):
        progress('main_forward',mode=mode,step=step,T=len(frames),M=sum(counts))
        before={k:state_hash({n:p for n,p in model.named_parameters() if n.split('.')[0]==k}) for k in ['visual','geometry','point','score','value','layers','correction']}
        opt.zero_grad(set_to_none=True);sync();start=time.monotonic()
        # CPU profiler records the actual dispatched aten SDPA operator; this
        # reports dispatch rather than inferring Flash from enabled switches.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs,info=model(frames,mode,progress=progress,diagnostics=True)
                loss,parts=video_loss(outputs,frames)
            sync();forward_seconds=time.monotonic()-start
            assert torch.isfinite(loss)
            if step==1:
                result['initial_z0_max_abs']=max(float((z-f['z0'].cuda()).abs().max()) for z,f in zip(outputs,frames))
                assert result['initial_z0_max_abs']==0
                result['output_work_shapes']=[list(z.shape) for z in outputs]
                result['native_restoration']='FP32 bilinear align_corners=False to each frame native_hw'
                restored=F.interpolate(outputs[0].float()[None,None],size=frames[0]['native_hw'],mode='bilinear',align_corners=False)
                assert list(restored.shape[-2:])==frames[0]['native_hw']
                del restored
                # Direct dependence on later original blocks, using this same
                # full-video graph; extra relation VJP is timed separately.
                start_check=time.monotonic()
                g=torch.autograd.grad(info['relations'][:counts[0]].float().square().sum(),
                                      info['tokens'],retain_graph=True)[0]
                sync();result['relation_dependency_check_seconds']=time.monotonic()-start_check
                cross=float(g[counts[0]:].abs().max()) if sum(counts[1:]) else 0.
                result['first_frame_R_future_C_gradient_max_abs']=cross
                assert cross>0 if mode=='FULL_VIDEO' else cross==0
                del g
            sync();b=time.monotonic();loss.backward();sync();backward_seconds=time.monotonic()-b
        grads=gradients(model)
        if step==2:
            for group in ['visual','geometry','point','score','value','layers','pixel_context','correction']:
                assert grads.get(group,0)>0, ('second-step missing gradient',group)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm)
        sync();u=time.monotonic();opt.step();sync();update_seconds=time.monotonic()-u
        changed={k:state_hash({n:p for n,p in model.named_parameters() if n.split('.')[0]==k})!=v for k,v in before.items()}
        if step==2:assert all(changed.values()), changed
        backends=sorted({x.key for x in prof.key_averages() if 'scaled_dot_product' in x.key or 'flash_attention' in x.key})
        rec=dict(step=step,loss=float(loss),pixel_BCE=float(parts[0]),soft_Dice=float(parts[1]),
                 forward_seconds=forward_seconds,backward_seconds=backward_seconds,update_seconds=update_seconds,
                 gradient_norm=float(norm),gradient_groups=grads,changed_groups=changed,actual_dispatch=backends,**memory())
        result['steps'].append(rec);publish(result);progress('step_complete',mode=mode,lr=1e-4,**rec)
        del outputs,info,loss,parts,prof
    result.update(status='PASS',exit_code=0,optimizer_updates=2,main_forward_calls=2,
                  loss_backward_calls=2,relation_dependency_vjp_calls=1,**memory())
    del model,opt;gc.collect();torch.cuda.empty_cache()
    return result


def summarize(root):
    cases=[]
    for path in sorted((root/'cases').glob('*/result.json')):
        cases.append(json.loads(path.read_text()))
    summary=dict(cases=cases,scope='resource preflight only; no AP/dev evaluation or formal training',
                 pending_cases=[c for c in CASES if not any(x['case']==c and x['status']=='PASS' for x in cases)])
    save(root/'summary.json',summary)
    rows=[]
    for case in cases:
        for mode in case.get('modes',[]):
            for s in mode['steps']:
                rows.append(dict(case=case['case'],mode=mode['mode'],T=mode['T'],M=mode['M'],
                                 **{k:s[k] for k in ['step','loss','forward_seconds','backward_seconds','update_seconds','peak_allocated_bytes','peak_reserved_bytes']}))
    if rows:
        with (root/'measurements.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# PVST完整视频原始块资源预检','不是正式训练或定位效果比较。预检更新丢弃，不保存训练权重。',
           '|病例|状态|模式|T|M|更新|','|---|---|---|---:|---:|---:|']
    for c in cases:
        for m in c.get('modes',[]):lines.append(f"|{c['case']}|{m['status']}|{m['mode']}|{m['T']}|{m['M']}|{m.get('optimizer_updates',0)}|")
        if c['status']!='PASS':lines.append(f"\n失败阶段：{c.get('stage')}；{c.get('error')}。")
    lines+=['\n待完成病例：'+', '.join(summary['pending_cases']),
            '完整逐步耗时、梯度、SDPA派发、显存与缓存身份见cases/*/result.json和measurements.csv。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=CASES,default=[CASES[0]])
    p.add_argument('--query-chunk',type=int,choices=[0,1024],default=0)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    phase='environment';case=None;record={};exit_code=0
    try:
        assert torch.cuda.is_available(),'CUDA unavailable; no CPU downgrade of the real probe'
        torch.set_num_threads(4)
        manifest=json.loads((OLD/'manifest.json').read_text())
        cp=torch.load(SPATIAL/'epoch01.pt',map_location='cpu',weights_only=False)
        assert cp['epoch']==1 and cp['updates']==22878 and cp['condition']=='EXPANDED_SPATIAL'
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        for case in args.cases:
            row=next(x for x in manifest['rows'] if x['video_id']==case)
            assert row['split']=='train' and row['role']=='fake'
            ident=identity(row,cp);path=root/'cases'/case/'result.json'
            if path.exists():
                old=json.loads(path.read_text());assert old['identity']==ident
                assert old['status']=='PASS','Failed run preserved; choose an explicit new --output for a new attempt'
                progress('reuse_PASS_case',case=case);continue
            record=dict(case=case,status='RUNNING',identity=ident,execution_head=head,modes=[],new_MoGe=0)
            save(path,record);phase='frozen_observation_preparation'
            folder,cost=prepare(row,ident,root);record['preparation']=cost;save(path,record)
            phase='load_route';frames=load_frames(row,folder)
            record['frozen_cache_bytes']=sum((folder/f'{i:05d}.pt').stat().st_size for i in range(len(frames)))
            seed();initial=FullVideoDirect(args.query_chunk).state_dict()
            record['new_parameter_count']=sum(x.numel() for x in initial.values())
            for mode in MODES:
                phase=mode;progress('case_mode',case=case,mode=mode,T=len(frames),M=sum(len(f['keys']) for f in frames))
                def publish(value):
                    if not any(m['mode']==value['mode'] for m in record['modes']):
                        record['modes'].append(value)
                    save(path,record)
                measured=measure(frames,mode,initial,args.query_chunk,publish)
                save(path,record)
            record.update(status='PASS',exit_code=0,preflight_updates_discarded=4)
            save(path,record);summarize(root)
            del frames,initial
    except Exception as exc:
        exit_code=1
        record.update(status='FAILED',stage=phase,error=repr(exc),traceback=traceback.format_exc(),exit_code=1)
        if torch.cuda.is_available():record.update(memory())
        save(root/'cases'/(case or 'environment')/'result.json',record)
        progress('FAILED',case=case,stage=phase,error=repr(exc));traceback.print_exc()
    finally:
        summarize(root)
    raise SystemExit(exit_code)


if __name__=='__main__':main()
