"""User-launched three-arm full-video experiment; bounded --smoke only for Codex."""
import argparse,collections,gc,json,math,random,subprocess,time,traceback
from pathlib import Path
import torch
from . import fullvideo_matched_data as olddata
from .fullvideo_matched import (exclusive_lock,csvsave,save_checkpoint,restore_checkpoint,score_video,
    aggregate_records,groups,persist_eval,preview_writer)
from .fullvideo_direct_probe import save,state_hash,seed,sync,memory
from .change_relation_model import CONDITIONS,ChangeRelation,make_head,organize,video_loss

REPO=Path(__file__).resolve().parents[2]
ROOT=olddata.ROOT/'pvst_ainpaint_change_relation_v1'
SOURCE=olddata.OUT
ADAPTATION=olddata.SOURCE/'epoch01.pt'
CONFIG=REPO/'docs/pvst/ainpaint_change_relation_config_v1.json'


def plan(out):
    config=olddata.read(CONFIG)
    for key,value in dict(seed=20260909,epochs=10,updates_per_condition=1780,supervised_presentations_per_condition=228780,D=128,heads=4,rounds=2,h=.5,query_chunk=0).items():assert config[key]==value,('FIXED_CONFIG',key)
    protocol=olddata.read(SOURCE/'protocol.json');olddata.validate_protocol(protocol)
    sourceid=olddata.read(SOURCE/'identity.json')
    assert (SOURCE/'prepared.json').is_file(),'OBSERVATIONS_MISSING_NO_EXTRACTION_AUTHORIZED'
    monitor=olddata.read(olddata.ROOT/'pvst_anakin_mumpy_baseline_v1/formal/monitor.json')
    assert len(monitor)==32
    train=sorted(r['video_id'] for r in protocol['rows'] if r['split']=='train');orders=[]
    for epoch in range(1,11):
        order=train.copy();random.Random(20260909+epoch).shuffle(order);orders.append(order)
    identity=dict(config=config,source_identity=sourceid,code={n:olddata.digest(REPO/'research_tools/xyz_pilot'/n) for n in ['change_relation.py','change_relation_model.py','change_relation_report.py']},
                  head_source=sourceid['adaptation_sha256'],official_head_source=olddata.digest(olddata.AINPAINT/'ainpaint/modules/context_decoder.py'))
    if (out/'identity.json').exists():assert olddata.read(out/'identity.json')==identity,'PROTOCOL_OR_CODE_CHANGED'
    else:
        save(out/'identity.json',identity);save(out/'config.json',config);save(out/'protocol.json',protocol)
        save(out/'orders.json',orders);save(out/'monitor.json',monitor);save(out/'categories.json',olddata.read(SOURCE/'categories.json'))
    assert olddata.read(out/'orders.json')==orders and olddata.read(out/'monitor.json')==monitor
    if not (out/'frozen_reference.json').exists():save(out/'frozen_reference.json',olddata.read(SOURCE/'frozen_reference.json'))
    return protocol,identity,config


def initial(out,config):
    seed();model=ChangeRelation(make_head(ADAPTATION),query_chunk=config['query_chunk'])
    if (out/'initial.pt').exists():
        state=torch.load(out/'initial.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        assert state_hash(state)==olddata.read(out/'initialization.json')['model_sha256']
    else:
        olddata.atomic_torch(out/'initial.pt',model.state_dict())
        save(out/'initialization.json',dict(model_sha256=state_hash(model.state_dict()),head_sha256=state_hash(model.head.state_dict()),
             parameters=sum(p.numel() for p in model.parameters()),head_parameters=sum(p.numel() for p in model.head.parameters()),
             source=str(ADAPTATION),initial_visual_observations='frozen DINO/LoRA epoch01; no backbone instantiated'))
    return model


def optimizer(model):
    return torch.optim.AdamW([dict(params=model.head.parameters(),lr=1e-5),dict(params=[p for n,p in model.named_parameters() if not n.startswith('head.')],lr=1e-4)],weight_decay=.01)


class Cache:
    def __init__(self):self.cache=olddata.VideoCache(SOURCE)
    def load(self,row,condition):
        start=time.monotonic();frames=organize(self.cache.load(row),condition)
        return frames,time.monotonic()-start


def update(model,opt,frames,condition,emit):
    model.train();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();sync();start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pixels,aux=model(frames,condition,progress=emit);loss,parts=video_loss(pixels,aux,frames)
    sync();forward=time.monotonic()-start;assert torch.isfinite(loss)
    emit('backward',T=len(frames),M=sum(len(f['keys']) for f in frames),loss=float(loss))
    tick=time.monotonic();loss.backward();sync();backward=time.monotonic()-tick
    grads=collections.defaultdict(float)
    for name,p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            root='encoder' if name.split('.')[0] in ['visual','geometry','point','score','value'] else name.split('.')[0]
            grads[root]+=float(p.grad.float().square().sum())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    tick=time.monotonic();opt.step();sync();upd=time.monotonic()-tick
    return dict(loss=float(loss.detach()),pixel_BCE=float(parts[0].detach()),soft_Dice=float(parts[1].detach()),block_BCE=float(parts[2].detach()),
                seconds=time.monotonic()-start,forward_seconds=forward,backward_seconds=backward,update_seconds=upd,
                gradient_norm=float(norm),gradient_L2={k:math.sqrt(v) for k,v in grads.items()},T=len(frames),M=sum(len(f['keys']) for f in frames),
                N_min=min(len(f['keys']) for f in frames),N_max=max(len(f['keys']) for f in frames),
                supervised_frames=sum(f['supervised'] for f in frames),lr=[g['lr'] for g in opt.param_groups],**memory())


@torch.no_grad()
def evaluate(model,cache,rows,condition,emit,categories=None,monitor=None,capture=None):
    model.eval();records=[];start=time.monotonic()
    wanted=collections.defaultdict(list)
    if monitor:
        for v,i in monitor:wanted[v].append(i)
    for vi,row in enumerate(rows):
        if monitor and row['video_id'] not in wanted:continue
        frames,io=cache.load(row,condition);emit('evaluate',condition=condition,video=row['video_id'],video_position=vi+1,T=len(frames),M=sum(len(f['keys']) for f in frames),io_seconds=io)
        with torch.autocast('cuda',dtype=torch.bfloat16):pixels,_=model(frames,condition)
        records.extend(score_video(row,frames,pixels,wanted=wanted[row['video_id']] if monitor else None,capture=capture))
        del frames,pixels
    ev=aggregate_records(records);ev['seconds']=time.monotonic()-start
    if categories:ev['groups']=groups(ev,categories)
    return ev


def buffers(model):return state_hash(dict(model.named_buffers()))


def smoke(out,protocol,ident,config,emit):
    root=out/'smoke'/str(time.time_ns());root.mkdir(parents=True);cache=Cache();byid={r['video_id']:r for r in protocol['rows']}
    extrema=olddata.read(SOURCE/'queue_extrema.json');maxid=extrema['train']['max_M']['video_id'];assert maxid=='4ro5o4_fake' and extrema['train']['max_M']['M']==518262
    train=[r for r in protocol['rows'] if r['split']=='train']
    max_t=max(train,key=lambda r:r['actual_frames'])['video_id'];max_pixels=max(train,key=lambda r:r['actual_frames']*math.prod(r['work_hw']))['video_id']
    cases=[('4ym5n4_fake',c) for c in CONDITIONS]+[(v,'C_FULL_XYZ') for v in dict.fromkeys([maxid,max_t,max_pixels])]
    results=[]
    save(root/'planned_cases.json',cases)
    for vid,condition in cases:
        stage='IO';row=byid[vid];start=time.monotonic();emit('smoke_case',video=vid,condition=condition)
        try:
            frames,io=cache.load(row,condition)
            M=sum(len(f['keys']) for f in frames)
            if vid==maxid:assert M==518262 and len(frames)==125
            model=initial(out,config).cuda();opt=optimizer(model);bn=buffers(model)
            record=dict(video=vid,condition=condition,T=len(frames),M=M,IO_seconds=io,supervised_frames=sum(f['supervised'] for f in frames),steps=[])
            stage='zero_fusion_parity'
            if vid=='4ym5n4_fake':
                model.eval()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    f=frames[0];zeros=torch.zeros((len(f['keys']),128),device='cuda');z=model.decode(f,zeros)
                    pad=model.head(f['patch'].cuda()[None],target_size=f['dino_padded_hw']);h,w=f['work_hw']
                    err=(z.float()-pad[0,0,:h,:w].float()).abs()
                    record['official_head_parity_max']=float(err.max());record['cache_z0_max']=float((z.float().cpu()-f['z0']).abs().max())
                    assert float(err.max())==0.,'ADAPTER_HEAD_PARITY_FAILED'
            for step in [1,2]:
                stage=f'update{step}';rec=update(model,opt,frames,condition,emit);record['steps'].append(rec)
                assert rec['gradient_L2'].get('aux',0)>0 and rec['gradient_L2'].get('encoder',0)>0 and rec['gradient_L2'].get('rounds',0)>0
                assert buffers(model)==bn,'BN_BUFFERS_CHANGED'
                emit('smoke_step',video=vid,condition=condition,step=step,**rec)
            stage='checkpoint_restore'
            checkpoint_path=root/(vid+'_'+condition+'_discarded.pt');identity=dict(scope='discarded smoke',identity=ident)
            before=state_hash(model.state_dict());save_checkpoint(checkpoint_path,model,opt,identity,condition,1,2,[dict(r,video_id=vid) for r in record['steps']])
            cp=restore_checkpoint(checkpoint_path,model,opt,identity,condition);assert state_hash(model.state_dict())==before and cp['updates']==2
            # Actual scoring/CSV/preview interface only on fixed frame0 of small case.
            if vid=='4ym5n4_fake':
                stage='evaluation_interface'
                model.eval()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pixels,_=model(frames,condition)
                capture=preview_writer(root,condition)
                ev=aggregate_records(score_video(dict(row,preview_targets=[0]),frames,pixels,wanted=[0],capture=capture))
                persist_eval(root/(condition+'_interface.json'),ev);record['interface_scored_frames']=1
                del pixels
            record.update(status='PASS',BN_buffers_unchanged=True,checkpoint_restore='PASS',seconds=time.monotonic()-start,exit_code=0)
            results.append(record);save(root/'results.json',results);checkpoint_path.unlink()
            del model,opt,cp,frames;gc.collect();torch.cuda.empty_cache()
        except Exception:
            save(root/'failure.json',dict(video=vid,condition=condition,stage=stage,error=traceback.format_exc(),completed=results,exit_code=1,formal_training_started=False))
            save(out/'smoke_latest.json',dict(path=str(root),status='FAILED'));raise
    save(root/'result.json',dict(status='PASS',cases=results,formal_training_started=False,new_backbone=0,new_MoGe=0,query_chunk=config['query_chunk'],attention='PyTorch SDPA; no attention weights; kernel not separately profiled',exit_code=0))
    save(out/'smoke_latest.json',dict(path=str(root),status='PASS'));emit('SMOKE_COMPLETE',output=str(root),formal_training_started=False)
    from .change_relation_report import smoke_report
    smoke_report(root)


def run_condition(out,protocol,ident,config,condition,resume,emit):
    folder=out/condition;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        completed=olddata.read(folder/'complete.json');assert completed['updates']==1780 and completed['supervised_presentations']==228780 and completed['exit_code']==0
        return
    if (folder/'last.pt').exists():assert resume,'EXPLICIT_RESUME_REQUIRED'
    model=initial(out,config).cuda();opt=optimizer(model);bn=buffers(model);records=[];epoch=1;position=0
    if (folder/'last.pt').exists():
        cp=restore_checkpoint(folder/'last.pt',model,opt,ident,condition);records=cp['records'];epoch=cp['epoch'];position=cp['position'];del cp
    rows={r['video_id']:r for r in protocol['rows']};orders=olddata.read(out/'orders.json');monitor=olddata.read(out/'monitor.json');cache=Cache();timings=[]
    for e in range(epoch,11):
        order=orders[e-1]
        for j in range(position if e==epoch else 0,len(order)):
            row=rows[order[j]];frames,io=cache.load(row,condition);done=len(records)
            factor=.1+.9*(1+math.cos(math.pi*done/1779))/2
            for g,lr in zip(opt.param_groups,[1e-5,1e-4]):g['lr']=lr*factor
            emit('train_video',condition=condition,epoch=e,epochs=10,video=row['video_id'],position=j+1,videos=178,step=done+1,total_steps=1780,T=len(frames),M=sum(len(f['keys']) for f in frames),IO_seconds=io)
            rec=update(model,opt,frames,condition,emit);assert buffers(model)==bn
            rec.update(video_id=row['video_id'],epoch=e,IO_seconds=io);records.append(rec);timings.append(rec['seconds']+io)
            save_checkpoint(folder/'last.pt',model,opt,ident,condition,e,j+1,records)
            csvsave(folder/'updates.csv',[{k:v for k,v in r.items() if k!='gradient_L2'} for r in records])
            emit('train_progress',condition=condition,step=len(records),total_steps=1780,estimated_remaining_train_seconds=sum(timings[-20:])/len(timings[-20:])*(1780-len(records)),**rec)
            del frames
        olddata.atomic_torch(folder/f'epoch{e:02d}.pt',torch.load(folder/'last.pt',map_location='cpu',weights_only=False))
        for label,selected_rows in [('monitor',[r for r in protocol['rows'] if r['split']=='train']),('dev',[r for r in protocol['rows'] if r['split']=='dev'])]:
            p=folder/f'epoch{e:02d}_{label}.json'
            if not p.exists():
                ev=evaluate(model,cache,selected_rows,condition,emit,categories=olddata.read(out/'categories.json') if label=='dev' else None,monitor=monitor if label=='monitor' else None,
                            capture=preview_writer(out,condition+'_last') if label=='dev' and e==10 else None)
                assert len(ev['frames'])==(5152 if label=='dev' else 32);persist_eval(p,ev)
            emit('epoch_metrics',condition=condition,epoch=e,scope=label,macro=olddata.read(p)['macro'])
        save_checkpoint(folder/'last.pt',model,opt,ident,condition,e+1,0,records)
        from .change_relation_report import report
        report(out)
    assert len(records)==1780 and sum(r['supervised_frames'] for r in records)==228780
    selected=max(range(1,11),key=lambda e:(olddata.read(folder/f'epoch{e:02d}_dev.json')['groups']['新增15开发来源']['pixel_AP'],-e))
    # Only fixed targets are scored, but each video still supplies its whole context.
    restore_checkpoint(folder/f'epoch{selected:02d}.pt',model,opt,ident,condition)
    devrows=[r for r in protocol['rows'] if r['split']=='dev']
    fixed=[(r['video_id'],i) for r in devrows for i in r['preview_targets']]
    ev=evaluate(model,cache,devrows,condition,emit,monitor=fixed,capture=preview_writer(out,condition+'_selected'))
    persist_eval(folder/'selected_fixed_preview.json',ev)
    save(folder/'complete.json',dict(status='COMPLETE',updates=len(records),supervised_presentations=228780,selected_epoch=selected,final_epoch=10,exit_code=0))
    del model,opt;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--condition',choices=CONDITIONS);parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output;lock=exclusive_lock(out);start=time.monotonic()
    def emit(stage,**kwargs):print(json.dumps(dict(stage=stage,invocation_seconds=time.monotonic()-start,**kwargs),ensure_ascii=False),flush=True)
    try:
        if args.report_only:
            from .change_relation_report import report
            report(out);return
        torch.set_num_threads(4);p,ident,config=plan(out)
        save(out/f'invocation_{time.time_ns()}.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arguments={k:str(v) for k,v in vars(args).items()}))
        if args.plan_only:emit('PLAN_READY',train_videos=178,updates_per_condition=1780,new_frontend=0);return
        assert torch.cuda.is_available(),'CUDA_UNAVAILABLE'
        if args.smoke:smoke(out,p,ident,config,emit);return
        assert olddata.read(out/'smoke_latest.json')['status']=='PASS','RESOURCE_PREFLIGHT_REQUIRED'
        delivered=olddata.read(out/'preflight_delivery.json')
        assert delivered['model_sha256']==ident['code']['change_relation_model.py'] and delivered['config']==config,'PREFLIGHT_MODEL_OR_CONFIG_CHANGED'
        for condition in [args.condition] if args.condition else CONDITIONS:
            save(out/'status.json',dict(status='RUNNING',condition=condition,formal_training_started=True))
            run_condition(out,p,ident,config,condition,args.resume,emit)
        from .change_relation_report import report
        report(out)
        save(out/'status.json',dict(status='COMPLETE' if all((out/c/'complete.json').exists() for c in CONDITIONS) else 'PARTIAL',exit_code=0,seconds=time.monotonic()-start))
    except BaseException as e:
        failure=dict(status='FAILED',traceback=traceback.format_exc(),seconds=time.monotonic()-start,exit_code=130 if isinstance(e,KeyboardInterrupt) else 1)
        save(out/f'failure_{time.time_ns()}.json',failure)
        if not (out/'status.json').exists() or olddata.read(out/'status.json').get('status')!='COMPLETE':save(out/'status.json',failure)
        raise
    finally:lock.close()
if __name__=='__main__':main()
