"""Five fixed-frame RGB/GT, cached normalized depth and frozen linear evidence.

CLI caller only. Existing Video/visual_scores supply the unchanged observations
and successful prepool readout. Adds display-only crops and region summaries;
no pooling, relation inference, fitting or new frontend is needed.
"""
import html
import json
import os
from pathlib import Path
import resource
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps,colors,pyplot as plt

from .data import Video,save
from .experiment import REPO,git,state_hash
from .set_fit_check import build
from .grid_feature_readout_probe import visual_scores,read,BASE,ROOT
from .local_pixel_readout_probe import csvsave
from scripts.prepare_v8_assets import digest
from scripts.review_anakin import decode

PRIOR=ROOT/'pvst_grid_feature_readout_probe_v1'
OUT=ROOT/'pvst_cat_boundary_evidence_v1'
INDICES=[0,30,60,90,119]


def regions(gt,valid,width):
    """Euclidean pixel-centre distance to opposite class, no change to raw GT."""
    gt=np.asarray(gt,bool);valid=np.asarray(valid,bool)
    if not gt.any():return {k:np.zeros_like(gt) for k in ['GT_inner_band','GT_interior','outside_adjacent']}
    inner=cv2.distanceTransform(gt.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    outer=cv2.distanceTransform((~gt).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return {'GT_inner_band':valid&gt&(inner<=width),'GT_interior':valid&gt&(inner>width),'outside_adjacent':valid&~gt&(outer<=width)}


def crop_box(gt,margin):
    yy,xx=np.where(gt)
    if not len(xx):return None
    h,w=gt.shape
    return [max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(w,int(xx.max())+1+margin),min(h,int(yy.max())+1+margin)]


def paint(values,valid,norm,cmap):
    image=(colormaps[cmap](norm(np.where(valid,values,norm.vmin)))[:,:,:3]*255).round().astype('uint8')
    image[~valid]=[255,0,255]
    return image


def outline(image,gt):
    result=image.copy();cs,_=cv2.findContours(gt.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(result,cs,-1,(255,255,0),1)


def legend(norm,cmap,title,path):
    fig,ax=plt.subplots(figsize=(7,1.2));fig.subplots_adjust(bottom=.5,top=.75)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ax,orientation='horizontal',label=title)
    fig.savefig(path,dpi=120,bbox_inches='tight');plt.close(fig)


@torch.no_grad()
def main():
    start=time.monotonic();torch.set_num_threads(4)
    assert git('branch','--show-current')=='pvst-grid-feature-readout-probe'
    assert not OUT.exists(), 'EXISTING_OUTPUT_INSPECT_BEFORE_REUSE'
    OUT.mkdir();save(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'stage':'verify'})
    prior=read(PRIOR/'summary.json');assert prior['status']=='COMPLETE' and prior['exit_code']==0
    weight=BASE/'RGB_DEPTH_PARTITION/last.pt';coefficient=PRIOR/'PREPOOL_LINEAR_coefficients.json'
    c=read(BASE/'config.json');m=read(BASE/'manifest.json');baseline=read(BASE/'identity.json')
    row=next(r for r in m['rows'] if r['pair_id']=='an:7e2e57' and r['role']=='fake' and r['split']=='dev')
    assert row['actual_frames']==120 and row['source_hw']==[720,1280]
    assert digest(weight)==prior['identity']['weight_sha256'] and digest(BASE/'manifest.json')==prior['identity']['manifest_sha256'] and digest(BASE/'config.json')==prior['identity']['config_sha256']
    params=read(coefficient) if coefficient.exists() else None
    if params:assert len(params['coef'])==len(params['scaler_mean'])==len(params['scaler_scale'])==256 and not params['warnings']
    assert digest(Path(row['video_path']))==row['video_sha256'] and digest(Path(row['mask_zip']))==row['mask_sha256']
    for p,sha in next(a for a in baseline['assets'] if a['video_id']==row['video_id'])['cache_sha256'].items():assert digest(Path(p))==sha
    for p,sha in baseline['code'].items():assert digest(REPO/p)==sha
    raw,images=decode(Path(row['video_path']),set(INDICES));assert raw['frame_count']==120
    for i in INDICES:assert abs(raw['pts'][i]['seconds']-row['pts'][i]['pts_s'])<1e-8
    protected=[weight,BASE/'manifest.json',coefficient,PRIOR/'summary.json',Path(row['mask_zip'])]
    geometry=Path(c.get('geometry_root',Path(c['output'])/'geometry'))/row['video_id']
    protected += [geometry/'complete.json',geometry/'xyz.npy',geometry/'valid.npy']
    hashes={str(p):digest(p) for p in protected if p.exists()}
    identity={'execution_head':git('rev-parse','HEAD'),'base_head':'ba8b1cbde5b7e9487ab4cec539f652157d440be6','prior_successful_execution_head':prior['identity']['execution_head'],'video_id':row['video_id'],'pair_id':row['pair_id'],'role':'fake','frame_indices':INDICES,'video_sha256':row['video_sha256'],'mask_sha256':row['mask_sha256'],'protected_sha256':hashes,'geometry_root':str(geometry),'scale':c['scale'],'source_hw':row['source_hw'],'human_record':'user review: fixed camera; added cat crosses front; unnatural cat/background compositing. Human visible cues, not 3D temporal anomaly ground truth.','response_available':params is not None,'prepool_feature_dim':256,'new_encoder_frames':5 if params else 0,'new_frontend_frames':0,'new_fits':0,'pool_and_relation_calls':0}
    save(OUT/'identity.json',identity)
    model=None;before=None
    if params:
        model=build(c,OUT,rgb_only=True).eval();cp=torch.load(weight,map_location='cpu',weights_only=True)
        assert cp['condition']=='RGB_DEPTH_PARTITION' and cp['epoch']==12 and cp['updates']==576 and cp['identity']==baseline['identity']
        assert cp['initial_model_sha256']==read(OUT/'initialization.json')['model_sha256'];model.load_state_dict(cp['state_dict'],strict=True);del cp
        model.requires_grad_(False);before=state_hash(model.state_dict());torch.cuda.reset_peak_memory_stats()
        # Any accidental execution of these modules is a diagnostic error.
        def forbidden(*args):raise AssertionError('POOL_OR_RELATION_MUST_NOT_RUN')
        handles=[model.pool.register_forward_pre_hook(forbidden),model.sets.register_forward_pre_hook(forbidden)]
    v=Video(row,c);frames=[];stats=[];width=max(1,int(np.floor(min(row['source_hw'])*.01+.5)));margin=max(1,int(np.floor(min(row['source_hw'])*.05+.5)))
    try:
        for i in INDICES:
            f=v.frame(i);rgb=f['rgb'].numpy();gt=f['gt'].numpy();valid=f['valid'].numpy();depth=f['xyz'][...,2].numpy()
            assert np.array_equal(rgb,np.array(images[i])) and f['pts']==row['pts'][i]['pts_s'] and rgb.shape==(720,1280,3)
            assert np.isfinite(depth[valid]).all() and (depth[valid]>0).all()
            z=np.full(gt.shape,np.nan,dtype=np.float32)
            if params:
                with torch.autocast('cuda',dtype=torch.bfloat16):local,_=model.local.encode(f['rgb'],torch.zeros_like(f['depth']))
                assert local.shape[0]==256
                points=torch.where(f['valid'].reshape(-1))[0].cuda();z.reshape(-1)[points.cpu().numpy()]=visual_scores(local,points,gt.shape,params)
                assert np.isfinite(z[valid]).all();del local,points
            box=crop_box(gt,margin);parts=regions(gt,valid,width)
            assert np.array_equal(parts['GT_inner_band']|parts['GT_interior'],gt&valid)
            record={'frame_index':i,'pts':f['pts'],'PTS_ticks':row['pts'][i]['pts_ticks'],'time_base':row['pts'][i]['time_base'],'native_H':720,'native_W':1280,'GT_pixels':int(gt.sum()),'valid_GT_pixels':int((gt&valid).sum()),'GT_valid_coverage':float((gt&valid).sum()/gt.sum()) if gt.any() else None,'valid_pixels':int(valid.sum()),'bbox_plus_margin_xyxy':box,'margin_pixels':margin,'band_width_pixels':width,'official_mask_member':v.members[i][1],'RGB_matches_original_decode':True}
            for name,mask in parts.items():
                n=int(mask.sum());stats.append({'frame_index':i,'pts':f['pts'],'region':name,'valid_pixels':n,'logit_median':float(np.median(z[mask])) if n and params else None,'normalized_depth_median':float(np.median(depth[mask])) if n else None})
            frames.append((rgb,gt,valid,depth,z,record));print('frame',i,f['pts'],'GT',record['GT_pixels'],flush=True)
    finally:
        v.close()
        if model:
            for h in handles:h.remove()
    depth_values=np.concatenate([a[3][a[2]] for a in frames]);dlo,dhi=np.quantile(depth_values,[.01,.99]);dnorm=colors.LogNorm(vmin=dlo,vmax=dhi,clip=True)
    zlim=float(np.quantile(np.concatenate([abs(a[4][a[2]]) for a in frames]),.99)) if params else 1.;znorm=colors.Normalize(-zlim,zlim,clip=True)
    legend(dnorm,'viridis','Z / s (log display; normalized units, not metres)',OUT/'depth_legend.png')
    if params:legend(znorm,'coolwarm','PREPOOL_LINEAR logit (not calibrated threshold)',OUT/'logit_legend.png')
    page=['<!doctype html><meta charset="utf-8"><title>7e2e57 fixed cat boundary review</title><style>body{font:16px sans-serif;margin:24px;max-width:1700px} .row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%;height:auto}figure{margin:0}figcaption{margin:6px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}.legend{max-width:750px}</style><h1>猫与背景接缝：五个固定帧</h1><p>仅an:7e2e57/fake的0、30、60、90、119帧。人工观察：固定机位，新增猫从前方经过，猫与背景融合不自然；不是三维异常真值。黄色轮廓=官方编辑GT，不是接缝标签。紫色=几何无效，不填成零深度或零响应。</p>',f'<p>深度五帧统一范围[{dlo:.6g}, {dhi:.6g}]，对数显示；响应统一范围[{-zlim:.6g}, {zlim:.6g}]。两者显示范围分别由五帧有效值1%/99%分位数、绝对logit99%分位数固定，超出仅显示截断，不改变输入。响应红色只表示相对高logit，不是校准伪造判定。</p><img class="legend" src="depth_legend.png">']
    if params:page.append('<img class="legend" src="logit_legend.png">')
    for rgb,gt,valid,depth,z,rec in frames:
        i=rec['frame_index'];prefix=f'frame_{i:03d}';rgbview=outline(rgb,gt);dv=outline(paint(depth,valid,dnorm,'viridis'),gt);zv=outline(paint(z,valid,znorm,'coolwarm'),gt) if params else np.full_like(rgb,128)
        Image.fromarray(rgb).save(OUT/f'{prefix}_rgb_raw.png');Image.fromarray(gt.astype('uint8')*255).save(OUT/f'{prefix}_GT_binary.png')
        page.append(f'<h2>帧{i} · PTS {rec["pts"]:.9f}s · 1280×720 · GT有效覆盖{rec["GT_valid_coverage"]:.2%}</h2><p>官方mask成员 {html.escape(rec["official_mask_member"])}；<a href="{prefix}_rgb_raw.png">无轮廓原图</a> / <a href="{prefix}_GT_binary.png">官方GT（原协议二值化）</a></p>')
        for crop,label in [(False,'全帧（点击查看原尺寸）'),(True,f'同框放大：GT外接框＋固定{margin}px，原生坐标{rec["bbox_plus_margin_xyxy"]}；只裁显示')]:
            page.append(f'<p>{label}</p><div class="row">')
            for name,image,caption in [('rgb',rgbview,'RGB＋官方GT轮廓'),('depth',dv,'回映后实际 Z/s＋GT轮廓'),('logit',zv,'PREPOOL_LINEAR logit＋GT轮廓' if params else '响应不可复现：成功系数缺失')]:
                im=Image.fromarray(image)
                if crop and rec['bbox_plus_margin_xyxy']:im=im.crop(rec['bbox_plus_margin_xyxy'])
                file=f'{prefix}_{name}'+('_crop' if crop else '')+'.png';im.save(OUT/file)
                page.append(f'<figure><a href="{file}"><img loading="lazy" src="{file}"></a><figcaption>{caption}</figcaption></figure>')
            page.append('</div>')
        rec.update(depth_display_below=int((depth[valid]<dlo).sum()),depth_display_above=int((depth[valid]>dhi).sum()),logit_display_clipped=int((abs(z[valid])>zlim).sum()) if params else None)
        rs=[r for r in stats if r['frame_index']==i];page.append('<p>以下仅辅助读图：边缘为GT内侧距背景≤7px；内部为剩余GT；外邻背景为GT外距离≤7px；全部再取几何有效交集。空区不可比较。</p><table><tr><th>区域</th><th>有效像素</th><th>logit中位数</th><th>Z/s中位数</th></tr>')
        for r in rs:
            fmt=lambda x:'不可比较' if x is None else f'{x:.6f}'
            page.append(f'<tr><td>{r["region"]}</td><td>{r["valid_pixels"]}</td><td>{fmt(r["logit_median"])}</td><td>{fmt(r["normalized_depth_median"])}</td></tr>')
        page.append('</table>')
    page.append('<p>正常猫／草地深度差不等于异常；五张静帧不能证明连续时序异常。<a href="report.md">中文报告</a> / <a href="frames.csv">帧核对</a> / <a href="regions.csv">区域统计</a></p>')
    (OUT/'index.html').write_text('\n'.join(page))
    assert all(digest(Path(p))==sha for p,sha in hashes.items())
    if model:assert state_hash(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    csvsave(OUT/'frames.csv',[a[5] for a in frames]);csvsave(OUT/'regions.csv',stats)
    summary={'status':'COMPLETE','exit_code':0,'identity':identity,'frames':[a[5] for a in frames],'regions':stats,'display':{'depth_low':float(dlo),'depth_high':float(dhi),'depth_scale':'log','logit_low':-zlim,'logit_high':zlim,'invalid_color':'magenta','contour_color':'yellow','range_selection':'five-frame valid depth 1%-99%; five-frame abs logit99%; display only'},'seconds':time.monotonic()-start,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_allocated':torch.cuda.max_memory_allocated() if model else None,'model_and_source_unchanged':True,'new_encoder_frames':5 if params else 0,'new_fits':0,'new_frontend_frames':0}
    save(OUT/'summary.json',summary);save(OUT/'status.json',{'status':'COMPLETE','stage':'preview_ready_for_visual_review','exit_code':0});(OUT/'exit_code.txt').write_text('0\n')
    print('COMPLETE',summary['seconds'],str(OUT/'index.html'),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():save(OUT/'status.json',{'status':'FAILED','exit_code':1,'error':traceback.format_exc()});(OUT/'exit_code.txt').write_text('1\n')
        raise
