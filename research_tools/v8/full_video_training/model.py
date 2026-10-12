"""Differentiable causal chunks; no detach, no block labels, no new parameters."""
import torch
from torch.utils.checkpoint import checkpoint

from research_tools.v8.local_geometry_pilot.model import geometry_input


class VideoTraining:
    def __init__(self,model):
        self.model=model

    def chunk(self,cpu,previous,previous_active,first):
        # A pure function: all context is explicit and input storage is never mutated.
        m=self.model;device=next(m.parameters()).device
        a={k:v.to(device,non_blocking=False) for k,v in cpu.items()}
        x,area,mu,valid,content=(a[k] for k in ('x','area','mu','valid','content'))
        t,n,_=x.shape;active=area>0
        xyz=mu[None,:,:,None,None,:].expand(1,t,n,16,16,3)
        patches=geometry_input(xyz,valid[None],content[None],mu[None],m.absolute_mean,
                               m.absolute_scale,m.relative_scale,'MEAN_REPLICATED')
        cm=content.reshape(-1,16,16);selected=cm.any((1,2))
        px,masks=patches[selected],cm[selected]
        geom=torch.zeros((t*n,64),device=device,dtype=x.dtype)
        if len(px):
            geom[selected]=torch.cat([m.patch_encoder(px[i:i+m.patch_chunk],masks[i:i+m.patch_chunk])
                                      for i in range(0,len(px),m.patch_chunk)])
        h=m.cell_encoder(torch.cat((torch.where(active[...,None],x,0.),geom.reshape(t,n,64)),-1))
        h=m.spatial(h,a['component_ids'],area,mu,(valid&content).sum((-1,-2)))
        logits=[]
        for ti in range(t):
            if first and ti==0:
                prev=previous;aux=torch.zeros((n,4),device=device,dtype=x.dtype)
            else:
                pi=a['predecessor'][ti].long()
                aw=torch.where(previous_active[pi]&active[ti,:,None],a['association_weight'][ti],0.)
                coverage=aw.sum(-1);aw=aw/coverage[:,None].clamp_min(1e-8)
                prev=(previous[pi]*aw[...,None]).sum(1)
                prev=torch.where((coverage>0)[:,None],prev,m.no_history)
                dt=a['delta_t'][ti].expand(n,1)
                aux=torch.cat((dt,coverage[:,None],a['history_available'][ti,:,None].float(),1.-coverage[:,None]),-1)
            current=m.gru(m.temporal_in(torch.cat((h[ti],aux),-1)),prev)
            previous=torch.where(active[ti,:,None],current,0.)
            previous_active=active[ti]
            logits.append(m.cell_head(previous).squeeze(-1))
        logits=torch.stack(logits)
        weights=torch.where(area>0,area.double().clamp_min(1).log(),-torch.inf)
        numerator=torch.logsumexp(logits.double()+weights,(0,1))
        return previous,previous_active,numerator,area.double().sum(),logits

    def video(self,cpu,chunk_size,use_checkpoint=True,return_cells=False):
        device=next(self.model.parameters()).device
        h=torch.zeros((256,128),device=device);active=torch.zeros(256,dtype=torch.bool,device=device)
        numerator=torch.tensor(-torch.inf,device=device,dtype=torch.float64)
        total=torch.tensor(0.,device=device,dtype=torch.float64);cells=[]
        for start in range(0,len(cpu['x']),chunk_size):
            part={k:v[start:start+chunk_size] for k,v in cpu.items()}
            args=(part,h,active,start==0)
            outputs=checkpoint(self.chunk,*args,use_reentrant=False) if use_checkpoint else self.chunk(*args)
            h,active,block,area,cell=outputs
            numerator=torch.logaddexp(numerator,block);total=total+area
            if return_cells:cells.append(cell)
        logit=numerator-total.log()
        return (logit,torch.cat(cells)) if return_cells else logit
