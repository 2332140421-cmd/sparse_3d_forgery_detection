"""CPU-only visualization of nine fixed cases and already-saved responses."""
import argparse
import base64
import csv
import html
import io
import json
import math
import re
import textwrap
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .manual_review import DERIVED, sha
from .manual_diagnosis import GROUPS, SEEDS

CASES = ('V002','V007','V012','V019','V021','V024','V028','V030','V035')
COLORS = ('#bc2525','#2371b7','#299345')
plt.rcParams['svg.fonttype'] = 'none'


def model_axes(models):
    if len(models)!=9 or len(set(models))!=9: raise ValueError('MODEL_AXIS_IDENTITY')
    return [[models.index(f'{g}__seed{s}') for s in SEEDS] for g in GROUPS]


def choose_times(pts, annotation):
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?|\.\d+)\s*[-–]\s*(\d+(?:\.\d+)?|\.\d+)\s*', annotation)
    interval = tuple(map(float, match.groups())) if match else None
    lo,hi = (max(pts[0],interval[0]),min(pts[-1],interval[1])) if interval else (pts[0],pts[-1])
    reason = '人工区间与保存PTS交集'
    if interval is None or lo>hi:
        lo,hi=pts[0],pts[-1];reason='全视频（时间为空/不可解析/无交集）'
    indices=[int(np.argmin(abs(pts-t))) for t in np.linspace(lo,hi,3)]
    return indices,interval,reason


def mask_canvas(patches):
    return patches.reshape(16,16,16,16).transpose(0,2,1,3).reshape(256,256)


def mapped_heat(values, content, area, mapping):
    if not np.array_equal(content.sum((1,2)),area): raise ValueError('CONTENT_AREA_MISMATCH')
    left,top=mapping['pad_xy'];h,w=mapping['content_hw']
    expected=np.zeros((256,256),bool);expected[top:top+h,left:left+w]=True
    if not np.array_equal(mask_canvas(content),expected): raise ValueError('PADDING_MAPPING_MISMATCH')
    heat=np.repeat(np.repeat(np.asarray(values,dtype=float).reshape(16,16),16,axis=0),16,axis=1)
    heat[~expected]=np.nan
    return heat[top:top+h,left:left+w]


def svg_comparison(path, rgb, heats, limit, record, frame_id, pts, frame_z):
    norm=Normalize(-limit,limit);cmap=matplotlib.colormaps['coolwarm']
    W=480;H=round(rgb.height*W/rgb.width);thumb=rgb.resize((W,H),Image.Resampling.LANCZOS)
    panels=[thumb]
    for heat in heats:
        # Nearest-neighbor expansion of raw cell logits; no heat smoothing.
        resized=np.asarray(Image.fromarray(heat.astype(np.float32)).resize(rgb.size,Image.Resampling.NEAREST))
        color=(cmap(norm(resized))[...,:3]*255).astype(np.uint8)
        over=(.52*np.asarray(rgb)+.48*color).astype(np.uint8)
        panels.append(Image.fromarray(over).resize((W,H),Image.Resampling.NEAREST))
    notes=textwrap.wrap(record['notes'],width=56) or ['（无备注）'];header=100+len(notes)*24
    height=header+2*(H+45)+85
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
           '<rect width="100%" height="100%" fill="white"/>']
    def txt(x,y,text,size=17):parts.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="#111">{html.escape(text)}</text>')
    txt(15,25,f'{record["review_id"]} | source frame {frame_id} | PTS {pts:.6f}s | saved video logits, threshold 0')
    txt(15,50,'video: '+', '.join(f'{g}={float(record[g+"_ensemble_logit"]):.4f}' for g in GROUPS),14)
    for i,line in enumerate(notes):txt(15,78+i*24,'人工备注：'+line if i==0 else line)
    for i,panel in enumerate(panels):
        x=15+(i%2)*490;y=header+(i//2)*(H+45)
        label='Original RGB' if i==0 else f'{GROUPS[i-1]} | frame logit {frame_z[i-1]:.4f}'
        txt(x,y,label,15);buf=io.BytesIO();panel.save(buf,format='JPEG',quality=94)
        parts.append(f'<image x="{x}" y="{y+10}" width="{W}" height="{H}" href="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"/>')
    y=header+2*(H+45)+5
    # Same normalization for all 81 heat panels; white means exactly zero.
    stops=''.join(f'<stop offset="{i}%" stop-color="rgb({",".join(str(int(c*255)) for c in cmap(i/100)[:3])})"/>' for i in range(101))
    parts.append(f'<defs><linearGradient id="scale">{stops}</linearGradient></defs><rect x="150" y="{y}" width="700" height="15" fill="url(#scale)"/>')
    for x,v in [(150,-limit),(500,0),(850,limit)]:txt(x,y+35,str(v),14)
    txt(15,y+65,'共同绝对logit色阶：蓝色为负，红色为正；仅粗格显示，无空间真值。',16)
    parts.append('</svg>');path.write_text('\n'.join(parts))


def run(old, diagnosis, out):
    wall=time.monotonic();cpu=time.process_time();out.mkdir(parents=True,exist_ok=True)
    frozen={r['video_id']:r for r in json.loads((old/'video_manifest.json').read_text())['rows']}
    records={r['review_id']:r for r in csv.DictReader((diagnosis/'review_predictions.csv').open(encoding='utf-8-sig'))}
    preserve=[old/'video_manifest.json',old/'evaluation/video_scores.csv',old/'evaluation/summary.json',old/'report.md']
    before={str(p):sha(p) for p in preserve};loaded={};missing=[];maximum=0
    for rid in CASES:
        record=records[rid];row=frozen[record['video_id']]
        if record['video_sha256']!=row['video_sha256'] or sha(row['video_path'])!=row['video_sha256']:raise ValueError('VIDEO_SHA_MISMATCH')
        try:
            summary=json.loads((old/'responses'/row['video_id']/'summary.json').read_text())
            if summary['video_id']!=row['video_id'] or summary['video_sha256']!=row['video_sha256']:raise ValueError('RESPONSE_IDENTITY')
            axes=model_axes(summary['models']);cc=[];ff=[];ii=[];tt=[]
            for b in summary['blocks']:
                if sha(b['response_path'])!=b['response_sha256']:raise ValueError('RESPONSE_HASH')
                with np.load(b['response_path'],allow_pickle=False) as z:
                    if z['cell_logits'].shape!=(len(z['pts_s']),9,256) or z['frame_logits'].shape!=(len(z['pts_s']),9):raise ValueError('RESPONSE_SHAPE')
                    cc.append(z['cell_logits']);ff.append(z['frame_logits']);ii.append(z['frame_indices']);tt.append(z['pts_s'])
            cells=np.concatenate(cc);frames=np.concatenate(ff);indices=np.concatenate(ii);pts=np.concatenate(tt)
            if not np.isfinite(cells).all() or not np.isfinite(frames).all() or not np.all(np.diff(indices)==1) or not np.all(np.diff(pts)>=0):raise ValueError('RESPONSE_FINITE_TIME')
            # Display-only common range from all saved cell values in these nine cases.
            maximum=max(maximum,float(np.abs(cells).max()))
            loaded[rid]=(record,row,summary,axes,cells,frames,indices,pts)
        except FileNotFoundError as e:missing.append(f'{rid}: {e.filename}')
    limit=max(1,math.ceil(maximum));page=['<!doctype html><meta charset="utf-8"><title>既有格响应诊断</title>',
        '<style>body{max-width:1100px;margin:auto;font:17px sans-serif}img{width:100%;height:auto}section{border-top:1px solid #aaa}pre{white-space:pre-wrap}</style>',
        '<h1>9条事后诊断视频：已保存格响应</h1>',
        f'<p>全图共同绝对logit色阶[-{limit},+{limit}]，0为中点；蓝负红正。无逐图min-max、无相对排名图。</p>',
        '<p>人工描述不是空间bbox或可靠帧级真值；格图不能证明定位。frame logit无帧级监督，视频正式分数未重新聚合。</p>']
    report=['# 已保存格/帧响应可视化','',f'固定9条：{", ".join(CASES)}；未增删，属于事后诊断集合，不是正式新测试。',
        f'三组各三seed按summary.models映射；显示取逐格/帧均值，统一色阶[-{limit},{limit}]。不改正式视频LSE/分数。',
        '原文件/响应SHA、input身份、帧/PTS、content/area/letterbox映射核对；不调用模型或前端。','']
    selections=[]
    for rid,(record,row,summary,axes,cells,frames,indices,pts) in loaded.items():
        chosen,interval,reason=choose_times(pts,record['anomaly_times']);wanted={int(indices[j]):j for j in chosen};pictures={};metas={};masks={}
        for b in summary['blocks']:
            if not any(b['start']<=fi<=b['end'] for fi in wanted):continue
            ip=Path(b['input_path']);meta=json.loads(ip.with_suffix('.json').read_text())
            if meta['video_sha256']!=row['video_sha256'] or meta['identity']!=summary['identity'] or sha(ip)!=meta['sha256']:raise ValueError('INPUT_IDENTITY_HASH')
            with np.load(ip,allow_pickle=False) as z:
                for fi in wanted:
                    if not b['start']<=fi<=b['end']:continue
                    k=fi-b['start'];m=meta['frames'][k];j=wanted[fi]
                    if m['frame_index']!=fi or abs(m['pts_s']-pts[j])>1e-9:raise ValueError('INPUT_RESPONSE_FRAME_PTS')
                    content=np.unpackbits(z['content'][k],count=65536,bitorder='little').reshape(256,16,16).astype(bool)
                    metas[fi]=m;masks[fi]=(content,z['area'][k].copy())
        with av.open(row['video_path']) as c:
            c.streams.video[0].codec_context.thread_count=2
            for fi,frame in enumerate(c.decode(video=0)):
                if fi in wanted:
                    m=metas[fi];actual=float(frame.pts*frame.time_base)-m['time_origin_s']
                    if abs(actual-m['pts_s'])>1e-8 or [frame.height,frame.width]!=m['mapping']['source_hw']:raise ValueError('DECODE_NATIVE_FRAME_PTS_MAPPING')
                    pictures[fi]=frame.to_image()
                if fi>=max(wanted):break
        if set(pictures)!=set(wanted):raise ValueError('DECODE_SELECTED_FRAME_MISSING')
        page.append(f'<section id="{rid}"><h2>{rid}</h2><p>{html.escape(record["notes"])}</p>')
        means=np.stack([frames[:,a].mean(1) for a in axes],axis=1)
        fig,ax=plt.subplots(figsize=(10,3.8))
        for g,a,color in zip(GROUPS,axes,COLORS):
            for s in a:ax.plot(pts,frames[:,s],color=color,alpha=.2,lw=.6)
            ax.plot(pts,frames[:,a].mean(1),color=color,label=g,lw=1.8)
        ax.axhline(0,color='black',ls='--',lw=.9,label='fixed threshold 0')
        if interval:ax.axvspan(max(pts[0],interval[0]),min(pts[-1],interval[1]),color='gold',alpha=.2,label='human description interval (not GT)')
        ax.set(xlabel='Native video PTS (seconds)',ylabel='Saved frame logit',title=rid+' | thick: mean seeds; thin: individual seeds');ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(out/f'{rid}_time.svg');plt.close(fig)
        no_outside=bool(interval and interval[0]<=pts[0] and interval[1]>=pts[-1])
        page.append(f'<p>选时规则：{reason}。'+('人工区间覆盖全部保存时刻，无区间外对照。' if no_outside else '人工时间记录仅作描述。')+'</p>')
        page.append(f'<img src="{rid}_time.svg" alt="{rid}时间曲线">')
        for n,j in enumerate(chosen,1):
            fi=int(indices[j]);content,area=masks[fi];m=metas[fi];heat=[mapped_heat(cells[j,a].mean(0),content,area,m['mapping']) for a in axes]
            filename=f'{rid}_{n}.svg';svg_comparison(out/filename,pictures[fi],heat,limit,record,fi,float(pts[j]),means[j])
            page.append(f'<p>source frame {fi}, PTS {pts[j]:.6f}s</p><img src="{filename}" alt="{rid}固定时刻{n}对照">')
            selections.append({'review_id':rid,'sample':n,'source_frame':fi,'pts_s':float(pts[j]),'selection_rule':reason,
                               **{g+'_min':float(np.nanmin(h)) for g,h in zip(GROUPS,heat)},
                               **{g+'_max':float(np.nanmax(h)) for g,h in zip(GROUPS,heat)}})
        page.append('</section>')
        report.append(f'- {rid}：帧{[int(indices[j]) for j in chosen]}，PTS{[round(float(pts[j]),6) for j in chosen]}；'+('无区间外对照。' if no_outside else '有区间外时刻，可查看但不作显著性或因果结论。'))
        print(f'VISUALIZED {rid}',flush=True)
    if selections:
        with (out/'selected_frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,list(selections[0]));w.writeheader();w.writerows(selections)
    report+=['','## 解释范围','绝对高响应指较高logit，正响应需大于等于0；负值中相对较高不能称为正证据。',
        '人工备注无空间bbox，是否与手/物体等位置可能重合需逐图定性查看；不能计算命中率、IoU或定位AP。',
        '当前自动生成过程不判断哪个区域是伪造；图与曲线不能证明因果依据或准确定位。无区间外对照时不宣称异常发生引起上升。',
        '正式video logit直接取既有CSV；逐格/帧均值仅显示，不替代正式每seed全视频LSE后均值。']
    for p in preserve:
        if sha(p)!=before[str(p)]:raise ValueError('ORIGINAL_FILE_CHANGED')
    costs={'status':'COMPLETE' if len(loaded)==9 else 'PARTIAL_MISSING_RESPONSES','videos':len(loaded),'comparison_figures':len(selections),
           'time_curves':len(loaded),'missing':missing,'common_logit_range':[-limit,limit],
           'cpu_process_s':time.process_time()-cpu,'wall_s':time.monotonic()-wall,'gpu_used':False,'original_hashes':before}
    report+=['',f'成本与验收：{json.dumps(costs,ensure_ascii=False)}']
    (out/'report.md').write_text('\n'.join(report)+'\n');(out/'index.html').write_text('\n'.join(page))
    (out/'visualization_status.json').write_text(json.dumps(costs,ensure_ascii=False,indent=2));print(json.dumps(costs,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,default=DERIVED/'v8_deepaction_cross_dataset_v1')
    p.add_argument('--diagnosis',type=Path,default=DERIVED/'v8_deepaction_manual_diagnosis_v1')
    p.add_argument('--output',type=Path,default=DERIVED/'v8_deepaction_response_visualization_v1')
    a=p.parse_args();run(a.experiment,a.diagnosis,a.output)
