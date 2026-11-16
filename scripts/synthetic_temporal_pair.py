"""One procedural adjacent-layer-swap pair; CLI produces PNG inputs and audit.

Supports a controlled temporal-order test. No model imports. Independent PNG
validation is needed because lossy playback cannot establish pixel equality.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import av
import numpy as np
from PIL import Image
from scripts.prepare_geometry_screen import ROOT, save

OUT = ROOT / 'synthetic_temporal_pairs_v2'
N, FPS, HEIGHT, WIDTH = 12, 12, 360, 640
MAPPING = list(range(N))
MAPPING[4], MAPPING[5] = 5, 4


def rgb_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def composite(background, layer):
    a = layer[..., 3:4].astype(np.uint32)
    return ((layer[..., :3].astype(np.uint32)*a + background.astype(np.uint32)*(255-a) + 127)//255).astype(np.uint8)


def assets():
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    bg = np.stack([90+x//9, 125+y//7, 150+(x+y)//14], -1).clip(0,255).astype(np.uint8)
    bg[280:] = (75, 92, 103)
    bg[::40, :, :] = np.maximum(bg[::40, :, :], 170)
    layers = []
    for i in range(N):
        cx, cy = 120+30*i, 210
        dx, dy = x-cx, y-cy
        radius = np.sqrt((dx/40)**2+(dy/45)**2)
        alpha = np.clip((1-radius)*255*3, 0,255).astype(np.uint8)
        stripe = ((dx+2*dy)//12 % 2).astype(bool)
        rgb = np.empty((HEIGHT, WIDTH,3),np.uint8)
        rgb[:] = (235,145,35)
        rgb[stripe] = (55,90,170)
        # Fixed object-local dots and bright centre distinguish texture from background.
        rgb[(dx+12)**2+(dy+15)**2 < 8**2] = (250,245,210)
        layers.append(np.dstack([rgb,alpha]))
    return bg, layers


def check_arrays(normal, disturbed, masks_a, masks_b, expected):
    failures = []
    changed = []
    for i,(a,b,ma,mb,e) in enumerate(zip(normal,disturbed,masks_a,masks_b,expected)):
        d = np.any(a != b, axis=-1)
        if d.any(): changed.append(i)
        if not np.array_equal(b,e): failures.append({'frame':i,'check':'source_permutation'})
        if np.any(d & ~(ma|mb)): failures.append({'frame':i,'check':'outside_alpha_union'})
    if changed != [4,5]: failures.append({'check':'changed_frames','actual':changed})
    if not np.array_equal(normal[6],disturbed[6]): failures.append({'check':'t6_current'})
    a,b = [[rgb_hash(f) for f in seq[3:6]] for seq in [normal,disturbed]]
    if Counter(a)!=Counter(b) or a==b: failures.append({'check':'history_multiset_or_order'})
    if not all(np.array_equal(a,b) for a,b in zip(normal[:4],disturbed[:4])): failures.append({'check':'unperturbed_t3'})
    return {'status':'PASS' if not failures else 'FAILED','failures':failures,'changed_frames':changed,'history_hashes':{'顺序':a,'扰动':b},'t':6,'H':3,'control_t':3}


def validate(out):
    m=json.loads((out/'manifest.json').read_text());assert m['frames']==N and m['fps']==FPS
    arrays=[]; masks=[]
    bg=np.array(Image.open(out/'background.png').convert('RGB'))
    layers=[np.array(Image.open(out/f'layers/{i:03d}.png').convert('RGBA')) for i in range(N)]
    for role,mapping in [('ordered',list(range(N))),('perturbed',MAPPING)]:
        frames=[np.array(Image.open(out/f'{role}/rgb/{i:03d}.png').convert('RGB')) for i in range(N)]
        ms=[np.array(Image.open(out/f'{role}/masks/{i:03d}.png'))>0 for i in range(N)]
        assert len(list((out/role/'rgb').glob('*.png')))==N
        for i,s in enumerate(mapping):
            assert frames[i].shape==(HEIGHT,WIDTH,3)
            assert np.array_equal(frames[i],composite(bg,layers[s])), (role,i,'composition')
            assert np.array_equal(ms[i],layers[s][...,3]>0), (role,i,'mask')
            assert m['mapping'][role][i]['source_layer']==s
        arrays.append(frames);masks.append(ms)
    result=check_arrays(*arrays,*masks,[composite(bg,layers[s]) for s in MAPPING])
    media=[]
    for role,frames in zip(['ordered','perturbed'],arrays):
        with av.open(str(out/f'{role}/lossless.mkv')) as c:
            stream=c.streams.video[0]; assert stream.average_rate==FPS
            decoded=list(c.decode(stream));assert len(decoded)==N
            for i,f in enumerate(decoded):
                assert np.array_equal(f.to_ndarray(format='rgb24'),frames[i]), (role,i,'lossless_decode')
                assert abs(float(f.pts*f.time_base)-i/FPS)<.0011
            media.append({'role':role,'codec':stream.codec_context.name,'decoded_frames':len(decoded),'fps':str(stream.average_rate),'pts_tolerance_s':.0011,'RGB_exact':True})
    result['media']=media
    save(out/'validation.json',result)
    assert result['status']=='PASS',result
    return result


def render(out):
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>受控时序配对</title>
<style>body{font:17px sans-serif;max-width:1400px;margin:24px}.row{display:flex;gap:12px}.row>div{width:49%}img{width:100%}button{margin:6px}#aux img{width:32%}</style>
<h1>程序生成：顺序／扰动（待人工确认）</h1><p>静态背景＋同一对象RGBA序列；12帧，12fps。只交换对象层4/5，主时刻t=6，H=3；无扰动对照t=3。用于受控时序能力检查，不是真实伪造检测收益或三维贡献证据。</p>
<button id="play">同步播放/暂停</button><button onclick="go(3)">对照3</button><button onclick="go(4)">交换4</button><button onclick="go(5)">交换5</button><button onclick="go(6)">主时刻6</button>
<label>帧<input id="slider" type="range" min="0" max="11" value="0"></label><span id="label"></span>
<p>同步播放器直接显示验收PNG；帧加载完成后一起更新。默认按12fps目标节拍，磁盘加载慢时播放会变慢，不丢帧。</p>
<div class="row"><div>顺序<img id="a"></div><div>扰动<img id="b"></div></div>
<div id="aux"><p>顺序目标mask／扰动目标mask／RGB差分（仅验收；白=有差异）</p><img id="ma"><img id="mb"><img id="diff"></div>
<p id="note"></p><p>对象每帧向右30像素，纹理随对象平移；4→5交换后影响过渡3→4、4→5、5→6。t=6有对象mask，但两版当前RGB差分为零。</p>
<p>人工请确认：运动是否连续可辨；交换是否只影响局部运动衔接；3/6的对应时刻是否清楚。尚未人工确认。</p>
<a href="manifest.json">来源及完整帧映射</a> / <a href="validation.json">验收结果</a> / <a href="report.md">报告</a> / <a href="ordered/lossless.mkv">顺序无损FFV1</a> / <a href="perturbed/lossless.mkv">扰动无损FFV1</a>
<p>无损PNG为后续统一输入，文件内无文字/水印/叠加。FFV1不能在浏览器播放时可用VLC；网页无需视频解码器。</p>
<p><a href="../temporal_controls_v1/index.html">旧GRIP三组：仅定性观察</a>；四条自然编辑仍保留3固定/1轻移、均协调难辨的批次记录，未猜逐条ID。</p>
<script>
let frame=0,playing=false,busy=false,timer;
async function go(i){if(busy)return;busy=true;let f=String(i).padStart(3,'0');let urls=[`ordered/rgb/${f}.png`,`perturbed/rgb/${f}.png`,`ordered/masks/${f}.png`,`perturbed/masks/${f}.png`,`diff/${f}.png`];try{let images=await Promise.all(urls.map(src=>new Promise((res,rej)=>{let im=new Image();im.onload=()=>res(im);im.onerror=rej;im.src=src})));['a','b','ma','mb','diff'].forEach((id,k)=>document.getElementById(id).src=images[k].src);frame=i;slider.value=i;label.textContent=` 帧${i} / ${(i/12).toFixed(3)}秒`;note.textContent=i===6?'主时刻：当前完全相同；历史层顺序3,4,5 vs 3,5,4':i===3?'无扰动对照：当前及过去0,1,2逐帧一致':''}finally{busy=false}}
async function tick(){if(!playing)return;await go((frame+1)%12);timer=setTimeout(tick,1000/12)}
play.onclick=()=>{playing=!playing;if(playing)tick();else clearTimeout(timer)};slider.oninput=()=>{playing=false;clearTimeout(timer);go(Number(slider.value))};go(0);
</script>''')


def main():
    p=argparse.ArgumentParser();p.add_argument('--validate-only',action='store_true');p.add_argument('--out',type=Path,default=OUT);args=p.parse_args();out=args.out
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if args.validate_only: print(validate(out));return
    if out.exists():raise RuntimeError('Output already exists; use --validate-only, never overwrite')
    start=time.monotonic();out.mkdir(parents=True)
    bg,layers=assets();Image.fromarray(bg).save(out/'background.png');(out/'layers').mkdir()
    for i,layer in enumerate(layers):Image.fromarray(layer).save(out/f'layers/{i:03d}.png')
    mapping={}
    for role,perm in [('ordered',list(range(N))),('perturbed',MAPPING)]:
        for folder in ['rgb','masks']:(out/role/folder).mkdir(parents=True)
        mapping[role]=[]
        for i,s in enumerate(perm):
            frame=composite(bg,layers[s]);mask=(layers[s][...,3]>0).astype(np.uint8)*255
            Image.fromarray(frame).save(out/f'{role}/rgb/{i:03d}.png');Image.fromarray(mask).save(out/f'{role}/masks/{i:03d}.png')
            mapping[role].append({'frame':i,'pts_num':i,'pts_den':FPS,'source_layer':s,'rgb_sha256':rgb_hash(frame),'mask_pixels':int((mask>0).sum())})
        subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate',str(FPS),'-i',str(out/role/'rgb/%03d.png'),'-c:v','ffv1','-level','3','-pix_fmt','bgr0',str(out/role/'lossless.mkv')],check=True)
    (out/'diff').mkdir()
    for i,s in enumerate(MAPPING):Image.fromarray(np.any(composite(bg,layers[i])!=composite(bg,layers[s]),axis=-1).astype(np.uint8)*255).save(out/f'diff/{i:03d}.png')
    save(out/'manifest.json',{'source':'Fully procedural NumPy RGB/alpha; no external dataset object or model','generator_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'frames':N,'fps':FPS,'size_hw':[HEIGHT,WIDTH],'object':'textured ellipse; centre(120+30*i,210), radii(40,45), soft alpha clip((1-radius)*765); integer alpha composite with rounding','background':'fixed RGB gradient, horizontal grid and floor; generated clean with no object','labels':['顺序','扰动'],'mapping':mapping,'swap':[4,5],'affected_transitions':[[3,4],[4,5],[5,6]],'main':{'t':6,'H':3},'control':{'t':3,'H':3},'mask_semantics':'actual composition alpha > 0; not an anomaly truth label','human_confirmed':False})
    result=validate(out);render(out)
    save(out/'status.json',{'status':result['status'],'passed_pairs':1,'exit_code':0,'seconds':time.monotonic()-start,'awaiting_human':True})
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
