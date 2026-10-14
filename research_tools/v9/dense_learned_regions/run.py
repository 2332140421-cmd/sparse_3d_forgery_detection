"""Finite V9 pilot; resumable video blocks and epoch checkpoints."""
import argparse
import fcntl
import os
import time
import traceback
from pathlib import Path
import torch
from .data import OUT,prepare,frontend,standardize,save,read_json,sha256_file
from .train import smoke,train
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--stage',choices=['prepare','smoke','all'],default='all');args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();phase='prepare'
    def progress(stage,completed,total,**extra):
        nonlocal phase
        phase=stage;state={'status':'RUNNING','stage':stage,'completed':completed,'total':total,'pid':os.getpid(),'updated_unix':time.time(),**extra}
        save(out/'progress.json',state);print(state,flush=True)
    try:
        rows,protocol=prepare(out)
        if args.stage=='prepare':return
        if args.stage=='smoke':
            source=sorted({r['source_id'] for r in rows if r['split']=='train'})[0];pair=[r for r in rows if r['source_id']==source]
            frontend(out,pair,progress)
            # Resource-only normalization; formal full-train statistics fitted later.
            stats=standardize(out,pair);smoke(out,rows,stats,progress);save(out/'smoke_status.json',{'status':'COMPLETE','elapsed_s':time.monotonic()-started});return
        assets=read_json('/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_video_global_context_v1/protocol.json')['frozen_files']
        for name,expected in assets.items():
            if sha256_file(name)!=expected:raise ValueError('OLD_FROZEN_ASSET_CHANGED:'+name)
        save(out/'old_asset_hashes.json',assets)
        front_started=time.monotonic();frontend(out,rows,progress)
        save(out/'frontend_invocation.json',{'elapsed_s':time.monotonic()-front_started,'scope':'this full frontend invocation; prior smoke block processing recorded separately; interrupted-call full wallclock unavailable'})
        progress('standardizer',0,1)
        stats_started=time.monotonic();stats=standardize(out,rows)
        save(out/'standardizer_cost.json',{'elapsed_s':time.monotonic()-stats_started})
        progress('standardizer',1,1)
        if not (out/'resource_smoke.json').exists():smoke(out,rows,stats,progress)
        implementation={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
        if (out/'implementation.json').exists() and read_json(out/'implementation.json')!=implementation:raise ValueError('IMPLEMENTATION_CHANGED')
        save(out/'implementation.json',implementation)
        records=train(out,rows,stats,protocol,progress)
        (out/'evaluation').mkdir(exist_ok=True)
        results,paired=evaluate(out,rows,stats,records,progress)
        for name,expected in assets.items():
            if sha256_file(name)!=expected:raise ValueError('OLD_ASSET_MODIFIED_DURING_RUN:'+name)
        disk=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        costs={'frontend':read_json(out/'frontend_costs.json'),'training':read_json(out/'training_costs.json'),
            'evaluation':read_json(out/'evaluation/costs.json'),'storage_bytes':disk,'models':records,
            'last_invocation_s':time.monotonic()-started,'early_smoke_total_s':None,'early_smoke_note':'早期性能测试总墙钟未完整记录；不以最后一次恢复冒充总成本'}
        save(out/'costs.json',costs)
        text='# V9 稠密局部表示与学习区域 pilot\n\n计算完成：3/3模型各20轮；28完整验证视频。单seed开发实验。\n\n'
        text+='## 数据与方法\n\n训练覆盖：'+str(protocol['coverage'])+'。训练122source/244视频，不是全部206779训练帧；验证14source/28视频/27505连续原生帧。\n\n'
        text+='从头共享初始化1373057参数，冻结ResNet18 layer1完整图和MoGe2。RGB局部/XYZ局部12通道先编码到像素256维，64query；竞争条件只改分配归一化轴。区域attention及因果GRU、像素旁路解码相同。没有CoTracker。\n\n'
        text+='每parent clip均值与跨所有片段统一帧类别平衡，真假parent各半，source pair后step。AdamW lr.001/wd.0001，20轮最后epoch，FP32参数与bf16计算。验证仅视频首帧重置。\n\n'
        text+='## 视频指标（阈值0）\n\n|条件|AUROC|AP|P|Recall|F1|ACC|TN/FP/FN/TP|\n|---|---:|---:|---:|---:|---:|---:|---|\n'
        for c,r in results.items():
            v=r['video'];text+=f'|{c}|'+''.join(f'{v[k]:.6f}|' for k in ('auroc','ap','precision','recall','f1','accuracy'))+f'{v["tn"]}/{v["fp"]}/{v["fn"]}/{v["tp"]}|\n'
        text+='\n## 帧指标\n\n|条件|AP|AUROC|fake-video-macro AP|real-video-macro FPR|P|Recall|有效fake数|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for c,r in results.items():
            v=r['frame'];text+=f'|{c}|{v["ap"]:.6f}|{v["auroc"]:.6f}|{r["fake_video_macro_frame_ap"]:.6f}|{r["real_video_macro_frame_fpr"]:.6f}|{v["precision"]:.6f}|{v["recall"]:.6f}|{r["eligible_fake_videos"]}|\n'
        text+='\n## 配对及判断\n\n'+str(paired)+'\n\n'
        for p in paired:
            text+=f'{p["comparison"]}点估计{p["difference"]:+.6f}，95%CI{p["ci95"]}；'+('方向正向，但单seed开发结果仍需确认。' if p['difference']>0 else '本轮未获得正向视频排序增益，不追加调参。')+'\n\n'
        text+='## 区域与成本\n\n区域质量/重叠及连续query索引变化见evaluation/*_regions.json。QUERY列质量按定义恒定，不能据此称没有塌缩；另报告像素行归一化描述质量、argmax占比和query余弦重叠，均不改变模型。\n\n'
        text+=f'前端实测{costs["frontend"]["elapsed_s"]:.2f}s，正式训练实测{costs["training"]["actual_s"]:.2f}s，评价{costs["evaluation"]}；存储{disk/2**30:.3f}GiB。显存分模型见costs.json；早期smoke总耗时未完整记录。\n\n'
        text+='## 限制\n\nXYZ增量包括有效性信息；相机坐标不是统一世界运动。学习区域不是物理对象。完整帧视频排序和帧排序/召回/误报须分别看，不以某项替代主指标。固定阈值0，无空间真值，像素响应不是经验证定位。三条件不使用V8检测器初始化；旧资产hash前后相同。验证集反复开发使用，结果不代表独立泛化。\n'
        (out/'report.md').write_text(text,encoding='utf-8')
        save(out/'final_status.json',{'status':'COMPUTE_COMPLETE_PENDING_GIT','models_completed':3,'epochs':20,'validation_videos':28,'validation_frames':27505,'last_invocation_s':time.monotonic()-started})
    except BaseException as e:
        save(out/'final_status.json',{'status':'FAILED','stage':phase,'error':repr(e),'traceback':traceback.format_exc(),'last_invocation_s':time.monotonic()-started});raise


if __name__=='__main__':main()
