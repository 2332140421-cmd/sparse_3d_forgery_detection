"""AINPAINT feature fusion + two explicit change/global relation rounds.

Caller change_relation.py; inputs are read-only frozen patch/geometry packets,
outputs all-frame pixel logits and soft block supervision. No backbone or new
frontend. Independent from the old z0+correction model.
"""
import bisect
import math
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scipy.spatial import cKDTree
from .ainpaint_spatial import AINPAINT
from .fullvideo_direct_model import DirectLayer

CONDITIONS=('A_CURRENT_XY','B_FULL_XY','C_FULL_XYZ')


def slots(pts,i):
    result=[]
    for offset in [-.8,-.2,.2,.8]:
        t=pts[i]+offset
        if t<pts[0]-1e-9 or t>pts[-1]+1e-9:continue
        p=bisect.bisect_left(pts,t);candidates=[j for j in [p-1,p] if 0<=j<len(pts)]
        j=min(candidates,key=lambda j:(abs(pts[j]-t),j))
        if j!=i and j not in result:result.append(j)
    return result or [i]


def neighbors(keys):
    n=len(keys)
    if n<2:return torch.zeros((n,1),dtype=torch.long)
    _,ids=cKDTree(keys.numpy()).query(keys.numpy(),k=min(9,n))
    return torch.tensor(np.stack([row[row!=i][:min(8,n-1)] for i,row in enumerate(ids)]),dtype=torch.long)


def organize(frames,condition):
    """GT-free grouping; common relation support is the old valid point list."""
    result=[]
    for old in frames:
        f=dict(old);h,w=f['work_hw'];points=f['points'];y=points//w;x=points%w
        if condition!='C_FULL_XYZ':
            # Exact lexicographic (x,y,0) order, using integer packing instead of slow tuple sorting.
            gh=(h+13)//14;codes=(x//14)*gh+y//14
            unique,index=torch.unique(codes,sorted=True,return_inverse=True)
            keys=torch.stack((unique//gh,unique%gh,torch.zeros_like(unique)),1)
            absolute=torch.stack(((x.float()+.5)/w,(y.float()+.5)/h,torch.zeros_like(x,dtype=torch.float)),1)
            relative=torch.stack(((x.float()+.5)/14-keys[index,0]-.5,(y.float()+.5)/14-keys[index,1]-.5,torch.zeros_like(x,dtype=torch.float)),1)
            centers=(keys.float()+.5)*14;centers[:,0]/=w;centers[:,1]/=h;centers[:,2]=0
            f.update(keys=keys,index=index)
        else:
            absolute=f['xyz_normalized'].reshape(-1,3)[points]
            relative=absolute/.5-torch.floor(absolute/.5)-.5
            centers=(f['keys'].float()+.5)*.5
        f.update(point_coordinates=torch.cat((absolute,relative),1),centers=centers,
                 neighbors=neighbors(f['keys']),counts=torch.bincount(f['index'],minlength=len(f['keys'])))
        assert int(f['counts'].sum())==len(points)
        result.append(f)
    return result


def block_targets(f,device):
    counts=f['counts'].to(device).float();g=f['gt_work'].to(device).flatten()[f['points'].to(device)].float()
    mass=torch.zeros_like(counts).index_add(0,f['index'].to(device),g)
    return mass/counts.clamp_min(1),counts


def block_loss(logits,target,counts):
    return (F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none')*counts).sum()/counts.sum().clamp_min(1)


class ChangeRound(nn.Module):
    def __init__(self,query_chunk=0):
        super().__init__();self.norm=nn.LayerNorm(128)
        self.q=nn.Linear(128,128);self.k=nn.Linear(128,128);self.v=nn.Linear(128,128)
        self.dt=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,32))
        self.change=nn.Sequential(nn.Linear(416,256),nn.GELU(),nn.Linear(256,128))
        self.dnorm=nn.LayerNorm(128);self.nq=nn.Linear(128,128);self.nk=nn.Linear(128,128)
        self.compare=nn.Sequential(nn.Linear(384,256),nn.GELU(),nn.Linear(256,128))
        self.slot_weight=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,1))
        self.global_layer=DirectLayer(query_chunk)

    def local(self,current,position,candidates,candidate_positions,dts,neighbor_ids,self_only=False):
        if not len(current):return current
        norm=self.norm(current);own=self.v(norm);messages=[]
        q=self.q(norm+position).view(1,-1,4,32).transpose(1,2)
        for other,pos,dt in zip(candidates,candidate_positions,dts):
            if self_only:
                m=own
            else:
                other_norm=self.norm(other)
                k=self.k(other_norm+pos).view(1,-1,4,32).transpose(1,2)
                v=self.v(other_norm).view(1,-1,4,32).transpose(1,2)
                m=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(-1,128)
            time=self.dt(current.new_full((len(current),1),dt))
            # own and m are both in this exact shared V projection space.
            d=self.change(torch.cat((own,m,m-own,time),1))
            ids=neighbor_ids.to(current.device)
            dn=self.dnorm(d);weight=(self.nq(dn)[:,None]*self.nk(dn[ids])).sum(-1)/math.sqrt(128)
            beta=weight.float().softmax(-1).to(d.dtype)
            surrounding=(beta[...,None]*d[ids]).sum(1)
            messages.append(self.compare(torch.cat((norm,d,d-surrounding),1)))
        rs=torch.stack(messages,1)
        a=self.slot_weight(rs).float().softmax(1).to(rs.dtype)
        return current+(a*rs).sum(1)


def make_head(adaptation_path):
    sys.path.insert(0,str(AINPAINT)) if str(AINPAINT) not in sys.path else None
    from ainpaint.modules.context_decoder import HighResContextHead
    head=HighResContextHead(1024,1,norm='BN')
    cp=torch.load(adaptation_path,map_location='cpu',weights_only=False)
    assert cp['epoch']==1
    head.load_state_dict({k[5:]:v for k,v in cp['adaptation_state'].items() if k.startswith('head.')},strict=True)
    return head


class ChangeRelation(nn.Module):
    def __init__(self,head,activation_checkpoint=True,query_chunk=0):
        super().__init__();self.head=head;self.activation_checkpoint=activation_checkpoint
        self.visual=nn.Sequential(nn.Conv2d(1024,128,1),nn.GELU())
        self.geometry=nn.Sequential(nn.Linear(6,64),nn.GELU(),nn.Linear(64,256))
        self.point=nn.Sequential(nn.Linear(128,128),nn.GELU());self.score=nn.Linear(128,1);self.value=nn.Linear(128,128)
        self.position=nn.Sequential(nn.Linear(3,64),nn.GELU(),nn.Linear(64,128))
        self.time=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,128))
        self.rounds=nn.ModuleList([ChangeRound(query_chunk) for _ in range(2)])
        self.relation_norm=nn.LayerNorm(128);self.aux=nn.Linear(128,1)
        self.fusion512=nn.Conv2d(128,512,1,bias=False);self.fusion128=nn.Conv2d(128,128,1,bias=False)
        nn.init.zeros_(self.fusion512.weight);nn.init.zeros_(self.fusion128.weight)
        self.train()

    def train(self,mode=True):
        super().train(mode)
        for m in self.head.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self

    def checked(self,fn,*args):
        return checkpoint(fn,*args,use_reentrant=False) if self.activation_checkpoint and torch.is_grad_enabled() else fn(*args)

    def encode(self,f):
        device=next(self.parameters()).device;h,w=f['work_hw']
        local=self.visual(f['patch'].to(device)[None]);local=F.interpolate(local,size=f['dino_padded_hw'],mode='bilinear',align_corners=False)[0,:,:h,:w]
        if not len(f['points']):return local.new_empty((0,128),dtype=torch.float32)
        gamma,beta=self.geometry(f['point_coordinates'].to(device)).chunk(2,-1)
        v=local.permute(1,2,0).reshape(-1,128)[f['points'].to(device)]
        x=self.point(v*(1+.1*gamma.tanh())+.1*beta);index=f['index'].to(device);n=len(f['keys']);z=self.score(x).float()[:,0]
        mx=z.new_full((n,),-torch.inf).scatter_reduce(0,index,z,reduce='amax',include_self=True)
        a=(z-mx[index]).exp();den=a.new_zeros(n).index_add(0,index,a)
        return x.new_zeros((n,128),dtype=torch.float32).index_add(0,index,self.value(x).float()*(a/den[index])[:,None])

    def relation_map(self,f,r):
        h,w=f['work_hw'];canvas=r.new_zeros((h*w,128));idx=f['index'].to(r.device)
        if len(r):canvas=canvas.index_copy(0,f['points'].to(r.device),r[idx])
        # Do not stretch content over padding. Pad right/bottom first.
        ph,pw=f['dino_padded_hw']
        return F.pad(canvas.T.reshape(1,128,h,w),(0,pw-w,0,ph-h))

    def decode(self,f,r):
        device=next(self.parameters()).device;relation=self.relation_map(f,self.relation_norm(r))
        x=self.head.proj_relu(self.head.proj_norm(self.head.proj(f['patch'].to(device)[None])))
        x=x+self.fusion512(F.interpolate(relation,size=x.shape[-2:],mode='bilinear',align_corners=False))
        x=self.head.up2(self.head.up1(x))
        x=x+self.fusion128(F.interpolate(relation,size=x.shape[-2:],mode='bilinear',align_corners=False))
        x=self.head.predictor(self.head.dropout(self.head.final_conv_block(self.head.up3(x))))
        x=F.interpolate(x,size=f['dino_padded_hw'],mode='bilinear',align_corners=False)
        h,w=f['work_hw'];return x[0,0,:h,:w]

    def forward(self,frames,condition,progress=None):
        assert condition in CONDITIONS and frames
        device=next(self.parameters()).device;counts=[len(f['keys']) for f in frames];pts=[f['pts'] for f in frames]
        cs=[];positions=[];globalpos=[]
        for i,f in enumerate(frames):
            def enc(_patch,frame=f):return self.encode(frame)
            cs.append(self.checked(enc,f['patch']))
            if progress and (i%20==0 or i+1==len(frames)):progress('encode',frame=i+1,T=len(frames),M=sum(counts))
            pos=self.position(f['centers'].to(device));positions.append(pos)
            globalpos.append(pos+(self.time(pos.new_full((len(pos),1),pts[i]-pts[0])) if condition!='A_CURRENT_XY' else torch.zeros_like(pos)))
        for ri,layer in enumerate(self.rounds):
            updated=[]
            for i,f in enumerate(frames):
                candidates=[i] if condition=='A_CURRENT_XY' else [j for j in slots(pts,i) if counts[j]]
                if not candidates:candidates=[i]
                self_only=candidates==[i]
                def change_bound(current,*others,layer=layer,indices=candidates,i=i,f=f,self_only=self_only):
                    return layer.local(current,positions[i],others,[positions[j] for j in indices],[pts[j]-pts[i] for j in indices],f['neighbors'],self_only)
                updated.append(self.checked(change_bound,cs[i],*[cs[j] for j in candidates]))
            allc=torch.cat(updated);pos=torch.cat(globalpos)
            if condition=='A_CURRENT_XY':cs=[self.checked(layer.global_layer.attend,c,p) for c,p in zip(updated,positions)]
            else:cs=list(self.checked(layer.global_layer.attend,allc,pos).split(counts))
            if progress:progress('relation_round',round=ri+1,M=sum(counts),T=len(frames))
        outputs=[];aux=[]
        for i,(f,r) in enumerate(zip(frames,cs)):
            def dec(x,frame=f):return self.decode(frame,x)
            outputs.append(self.checked(dec,r))
            if progress and (i%20==0 or i+1==len(frames)):progress('decode',frame=i+1,T=len(frames),M=sum(counts))
            aux.append(self.aux(self.relation_norm(r)).float()[:,0])
        return outputs,aux


def video_loss(outputs,aux,frames):
    terms=[]
    for z,logit,f in zip(outputs,aux,frames):
        if not f['supervised']:continue
        g=f['gt_work'].to(z.device).float();p=z.float().sigmoid()
        bce=F.binary_cross_entropy_with_logits(z.float(),g)
        dice=1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)
        y,n=block_targets(f,z.device)
        block=block_loss(logit,y,n) if len(logit) else z.sum()*0
        terms.append(torch.stack((bce,dice,block)))
    assert terms,'NO_SUPERVISION'
    components=torch.stack(terms).mean(0)
    return components[0]+components[1]+.1*components[2],components
