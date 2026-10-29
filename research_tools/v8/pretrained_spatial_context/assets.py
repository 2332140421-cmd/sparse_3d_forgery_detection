"""Pinned official localization weights, exact ANAKIN queue and shared features."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from scripts.prepare_v8_assets import digest
from research_tools.v8.spatial_localization.joint_data import Video,save

AINPAINT_REV='753ae1b700b054c97e15569718690211e36cdf53'
DINO_REV='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
UNUSED=('encoder.head_features.','encoder.classifier.')


def normalize518(rgb):
    x=rgb.permute(0,3,1,2).float()/255
    x=F.pad(x,(0,6,0,6),value=0.)
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x-mean)/std


def validate_state(expected,state):
    needed={k for k in expected if not k.startswith(UNUSED)}
    received={k for k in state if not k.startswith(UNUSED)}
    if needed!=received:raise ValueError('LOCALIZATION_KEYS:'+str({'missing':sorted(needed-received),'unexpected':sorted(received-needed)}))
    wrong=[k for k in needed if expected[k].shape!=state[k].shape]
    if wrong:raise ValueError('LOCALIZATION_SHAPES:'+str(wrong))
    return {k:v for k,v in state.items() if k in needed}


def official(config,out):
    root=Path(config['official_code']);dino=Path(config['dino_code'])
    for path,revision in [(root,AINPAINT_REV),(dino,DINO_REV)]:
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=revision:
            raise ValueError('OFFICIAL_CODE_REVISION:'+str(path))
    checkpoint=Path(config['checkpoint'])
    if not checkpoint.is_file():raise FileNotFoundError('TRAINED_SPATIAL_CHECKPOINT_REQUIRED:'+str(checkpoint))
    sha=digest(checkpoint)
    if sha!=config.get('checkpoint_sha256'):raise ValueError('CHECKPOINT_SHA256_MUST_BE_PINNED')
    sys.path.insert(0,str(root))
    from ainpaint.model_builder import get_network
    original=torch.hub.load
    def local_hub(repo,name,pretrained=True,**kwargs):
        if repo!='facebookresearch/dinov2' or name!='dinov2_vitl14_reg' or pretrained:
            raise ValueError('NO_UNPINNED_BACKBONE_DOWNLOAD')
        return original(str(dino),name,source='local',pretrained=False)
    with patch('torch.hub.load',local_hub):
        model=get_network('dinov2_vitl14_reg',0.,'test',True,enable_localization=True,
            num_classes=2,lora=True,lora_rank=4,lora_alpha=8,lora_dropout=.05,video_mode='spatial')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=cp.get('model_state_dict',cp);filtered=validate_state(model.state_dict(),state)
    missing=model.load_state_dict(filtered,strict=False)
    if missing.unexpected_keys or any(not k.startswith(UNUSED) for k in missing.missing_keys):raise ValueError('UNEXPLAINED_LOAD_MISMATCH')
    model.eval().requires_grad_(False)
    report={'official_revision':AINPAINT_REV,'dino_revision':DINO_REV,'checkpoint':str(checkpoint),
        'checkpoint_sha256':sha,'checkpoint_bytes':checkpoint.stat().st_size,'source_epoch':cp.get('epoch'),
        'localization_keys':len(filtered),'excluded_classification_keys':[k for k in state if k.startswith(UNUSED)],
        'backbone_parameters':sum(p.numel() for p in model.backbone.parameters()),
        'trained_lora_parameters':sum(p.numel() for k,p in model.backbone.named_parameters() if 'lora_' in k),
        'decoder_parameters':sum(p.numel() for p in model.encoder.localization_head.parameters()),
        'preprocess':'RGB/255; black pad right6 bottom6 to518; official ImageNet mean/std; no crop',
        'loaded_backbone_lora_decoder':'all required keys and shapes matched; only unused classification excluded'}
    save(out/'official_load.json',report)
    return model


def prepare(config,out):
    source=Path(config['source_run']);root=source.resolve();dest=out.resolve()
    if root==dest or root in dest.parents or dest in root.parents:raise ValueError('SOURCE_OUTPUT_OVERLAP')
    original=json.loads((source/'joint_manifest.json').read_text());fit=json.loads((source/'statistics.json').read_text())
    rows=[r for r in original['rows'] if r['dataset']=='ANAKIN']
    for split,count in [('train',48),('dev',16)]:
        rr=[r for r in rows if r['split']==split]
        if len(rr)!=count:raise ValueError('ANAKIN_FIXED_COUNTS')
        for pair in {r['pair_id'] for r in rr}:
            if sorted(r['role'] for r in rr if r['pair_id']==pair)!=['fake','real']:raise ValueError('PAIR_ROLES')
    if {r['source_id'] for r in rows if r['split']=='train'}&{r['source_id'] for r in rows if r['split']=='dev'}:raise ValueError('SOURCE_LEAKAGE')
    if fit['train_video_ids']!=[r['video_id'] for r in original['rows'] if r['split']=='train'] or len(fit['train_video_ids'])!=96:
        raise ValueError('ORIGINAL_TRAIN96_STATISTICS')
    refs={'source_run':str(source),'manifest_sha256':digest(source/'joint_manifest.json'),
        'statistics_sha256':digest(source/'statistics.json'),'statistics_origin':'original96 train; no refitting; ANAKIN48 updated here',
        'cache_metadata':{}}
    for r in rows:
        if r['spatial_status']!=('edit_mask' if r['role']=='fake' else 'known_background'):raise ValueError('ANAKIN_SPATIAL_LABEL_REQUIRED')
        v=Video(source,r,fit['statistics'],fit['grouping'],read_only=True)
        try:
            if any(len(a)!=r['actual_frames'] for a in v.arrays.values()):raise ValueError('CACHE_FRAME_COUNT')
            if v.arrays['rgb512'].shape[1:]!=(512,512,3):raise ValueError('RGB512_SHAPE')
            if not np.asarray(v.arrays['pixel_known']).all():raise ValueError('MISSING_GT_IS_NOT_BACKGROUND')
            refs['cache_metadata'][r['video_id']]=digest(v.directory/'complete.json')
        finally:v.close()
    for name,value in [('manifest.json',{'rows':rows,'selection':'all original ANAKIN train24/dev8 pairs'}),('source_references.json',refs)]:
        p=out/name
        if p.exists() and json.loads(p.read_text())!=value:raise ValueError('FROZEN_INPUT_CHANGED:'+name)
        if not p.exists():save(p,value)
    return rows,fit


def verify_source(config,out):
    source=Path(config['source_run']);r=json.loads((out/'source_references.json').read_text())
    if digest(source/'joint_manifest.json')!=r['manifest_sha256'] or digest(source/'statistics.json')!=r['statistics_sha256']:
        raise ValueError('ORIGINAL_ASSETS_CHANGED')
    for video,sha in r['cache_metadata'].items():
        if digest(source/'cache'/video/'complete.json')!=sha:raise ValueError('ORIGINAL_CACHE_CHANGED')
    if config.get('reference_run'):verify_reference(out)


def referenced_decoder(config,out,rows):
    """Reuse verified assets by metadata; mmap only official decoder tensors, no DINO."""
    reference=Path(config['reference_run']);features=Path(config['features_root']);source=Path(config['source_run'])
    if reference.resolve()==out.resolve() or reference.resolve() in out.resolve().parents or out.resolve() in reference.resolve().parents:
        raise ValueError('REFERENCE_OUTPUT_OVERLAP')
    if rows!=json.loads((reference/'manifest.json').read_text())['rows']:raise ValueError('EXACT_REFERENCE_QUEUE')
    original=json.loads((reference/'official_load.json').read_text())
    if original['checkpoint_sha256']!=config['checkpoint_sha256'] or original['official_revision']!=AINPAINT_REV:
        raise ValueError('OFFICIAL_REFERENCE_IDENTITY')
    oldrefs=json.loads((reference/'source_references.json').read_text())
    if oldrefs['statistics_sha256']!=digest(source/'statistics.json'):raise ValueError('REFERENCE_STATISTICS')
    small={reference/name for name in ['manifest.json','epoch0.json','official_load.json','source_references.json','result_summary.json','fixed_boundary_statistics.json']}
    large={Path(config['checkpoint'])}
    for c in ['SPATIAL_TRANSFER','SPATIAL_3D_CONTEXT']:
        small.update(reference/c/name for name in ['summary.json','history.json'])
        large.update(reference/c/name for name in ['best.pt','last.pt'])
    for r in rows:
        directory=features/r['video_id'];meta=directory/'complete.json';m=json.loads(meta.read_text())
        expected={'checkpoint_sha256':config['checkpoint_sha256'],'official_revision':AINPAINT_REV,'dino_revision':DINO_REV,
            'video_sha256':r['video_sha256'],'pts_sha256':hashlib.sha256(json.dumps(r['pts'],sort_keys=True).encode()).hexdigest(),
            'shape':[r['actual_frames'],1024,37,37],'dtype':'float16','preprocess':original['preprocess']}
        if any(m[k]!=v for k,v in expected.items()):raise ValueError('REFERENCE_FEATURE_METADATA:'+r['video_id'])
        a=np.load(directory/'patch.npy',mmap_mode='r')
        if list(a.shape)!=expected['shape'] or a.dtype!=np.float16:raise ValueError('REFERENCE_FEATURE_HEADER')
        del a
        small.add(meta);large.add(directory/'patch.npy')
        cache=source/'cache'/r['video_id'];small.add(cache/'complete.json');large.update(cache.glob('*.npy'))
    refs={'reference_run':str(reference),'features_root':str(features),'epoch0_sha256':digest(reference/'epoch0.json'),
        'small_file_sha256':{str(p):digest(p) for p in sorted(small)},
        'large_file_size_mtime':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(large)},
        'verification':'existing full SHA acceptance reused; metadata/headers and size/mtime checked; no repeated large-file scan',
        'new_visual_feature_extraction':False,'new_visual_feature_storage_bytes':0}
    path=out/'reference_reuse.json'
    if path.exists() and json.loads(path.read_text())!=refs:raise ValueError('REFERENCE_ASSETS_CHANGED')
    if not path.exists():save(path,refs)
    if Path(config['checkpoint']).stat().st_size!=original['checkpoint_bytes']:raise ValueError('OFFICIAL_CHECKPOINT_SIZE')
    root=Path(config['official_code'])
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=AINPAINT_REV:raise ValueError('OFFICIAL_CODE_REVISION')
    sys.path.insert(0,str(root))
    from ainpaint.modules.context_decoder import HighResContextHead
    decoder=HighResContextHead(1024,1,norm='BN')
    cp=torch.load(config['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    prefix='encoder.localization_head.'
    decoder.load_state_dict({k[len(prefix):]:v for k,v in cp['model_state_dict'].items() if k.startswith(prefix)},strict=True)
    decoder.eval();del cp
    save(out/'official_load.json',{**original,'initialization':'same official decoder; backbone/LoRA represented by shared frozen features; no backbone constructed'})
    target=out/'epoch0.json'
    if target.exists() and digest(target)!=refs['epoch0_sha256']:raise ValueError('EPOCH0_REFERENCE_CHANGED')
    if not target.exists():target.write_bytes((reference/'epoch0.json').read_bytes())
    verify_reference(out)
    return decoder,original['backbone_parameters']


def verify_reference(out):
    refs=json.loads((out/'reference_reuse.json').read_text())
    for name,sha in refs['small_file_sha256'].items():
        if digest(Path(name))!=sha:raise ValueError('READ_ONLY_REFERENCE_CHANGED:'+name)
    for name,expected in refs['large_file_size_mtime'].items():
        p=Path(name)
        if [p.stat().st_size,p.stat().st_mtime_ns]!=expected:raise ValueError('READ_ONLY_REFERENCE_CHANGED:'+name)


def patch_features(model,rgb):
    x=normalize518(rgb)
    _,features=model.encoder.forward_backbone(x)
    if features.shape[1:]!=(1024,37,37):raise ValueError('OFFICIAL_PATCH_FEATURE_SHAPE')
    return features


@torch.no_grad()
def extract(config,out,rows,fit,model,progress):
    """One frozen visual pass, shared FP16 patch maps, never rerun V8 providers."""
    device=next(model.parameters()).device;model.eval();costs=[];identity=json.loads((out/'official_load.json').read_text())
    for i,r in enumerate(rows):
        folder=out/'features'/r['video_id'];meta=folder/'complete.json'
        expected={'checkpoint_sha256':identity['checkpoint_sha256'],'video_sha256':r['video_sha256'],
            'official_revision':AINPAINT_REV,'dino_revision':DINO_REV,
            'pts_sha256':hashlib.sha256(json.dumps(r['pts'],sort_keys=True).encode()).hexdigest(),
            'shape':[r['actual_frames'],1024,37,37],'dtype':'float16','preprocess':identity['preprocess']}
        if meta.exists():
            saved=json.loads(meta.read_text())
            if any(saved[k]!=v for k,v in expected.items()):raise ValueError('SHARED_FEATURE_IDENTITY')
            if digest(folder/'patch.npy')!=saved['sha256']:raise ValueError('SHARED_FEATURE_SHA')
            costs.append(saved['seconds']);continue
        folder.mkdir(parents=True,exist_ok=True)
        rgb=np.load(Path(config['source_run'])/'cache'/r['video_id']/'rgb512.npy',mmap_mode='r')
        tmp=folder/'patch.part.npy';a=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=tuple(expected['shape']))
        started=time.monotonic()
        for start in range(0,len(rgb),config['feature_chunk']):
            part=torch.from_numpy(np.array(rgb[start:start+config['feature_chunk']],copy=True)).to(device)
            f=patch_features(model,part)
            if not torch.isfinite(f).all():raise ValueError('NONFINITE_PATCH_FEATURES')
            a[start:start+len(f)]=f.float().cpu().numpy().astype(np.float16)
            progress('features',video=r['video_id'],video_index=i,frames=min(start+len(f),len(rgb)),total_frames=len(rgb))
        torch.cuda.synchronize();a.flush();del a;tmp.replace(folder/'patch.npy')
        seconds=time.monotonic()-started;costs.append(seconds)
        save(meta,{**expected,'sha256':digest(folder/'patch.npy'),'seconds':seconds})
    save(out/'feature_costs.json',{'seconds':sum(costs),'frames':sum(r['actual_frames'] for r in rows),
        'seconds_per_frame':sum(costs)/sum(r['actual_frames'] for r in rows),
        'bytes':sum((out/'features'/r['video_id']/'patch.npy').stat().st_size for r in rows),
        'backbone_training':'frozen including trained LoRA; always eval; no per-epoch encoder calls'})


class CachedVideo:
    def __init__(self,config,out,row,fit,context):
        self.row=row;self.context=context
        self.base=Video(Path(config['source_run']),row,fit['statistics'],fit['grouping'],read_only=True)
        self.features=np.load(Path(config.get('features_root',out/'features'))/row['video_id']/'patch.npy',mmap_mode='r')
    def chunk(self,start,end):
        if self.context:a=self.base.chunk(start,end)
        else:a={k:torch.from_numpy(np.array(self.base.arrays[k][start:end],copy=True)) for k in ['content512','gt']}
        a['features']=torch.from_numpy(np.array(self.features[start:end],copy=True)).float()
        return a
    def native_gt(self,i):return self.base.native_gt(i)
    def close(self):self.base.close()
