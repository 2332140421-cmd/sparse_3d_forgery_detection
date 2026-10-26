"""Complete paired 256-structure/512-edit-region model for the joint pilot."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from research_tools.v8.component_global_context_pilot.model import GlobalContextModel
from research_tools.v8.full_video_training.model import VideoTraining

CONDITIONS=('CELL_LOC','COMPONENT_LOC')
STRUCTURE_KEYS=('x','mu','valid','content','area','component_ids','predecessor','association_weight','history_available','delta_t')


def dense_image(a):
    t=len(a)
    if a.ndim==5:return a.reshape(t,16,16,16,16,3).permute(0,5,1,3,2,4).reshape(t,3,256,256)
    return a.reshape(t,16,16,16,16).permute(0,1,3,2,4).reshape(t,1,256,256)


class LocalizationModel(nn.Module):
    def __init__(self,condition,statistics):
        super().__init__()
        if condition not in CONDITIONS:raise ValueError(condition)
        self.condition=condition
        self.structure=GlobalContextModel('GLOBAL_CELL' if condition=='CELL_LOC' else 'GLOBAL_COMPONENT',statistics,patch_chunk=512)
        self.rgb1=nn.Conv2d(3,32,3,stride=2,padding=1);self.rgb2=nn.Conv2d(32,32,3,padding=1)
        self.skip=nn.Conv2d(3,8,1)
        self.geo1=nn.Conv2d(7,32,3,padding=1);self.geo2=nn.Conv2d(32,32,3,padding=1)
        self.fuse=nn.Conv2d(192,32,1);self.refine=nn.Conv2d(32,32,3,padding=1)
        self.decode=nn.Conv2d(40,16,3,padding=1);self.residual=nn.Conv2d(16,1,1)

    def pixel(self,a,hidden,cells):
        content=dense_image(a['content']).bool();mask=a['content512'][:,None].bool()
        rgb=a['rgb512'].permute(0,3,1,2).float()/255
        rgb=torch.where(mask,rgb,0.)
        h=torch.where(content,F.relu(self.rgb1(rgb)),0.)
        h=torch.where(content,F.relu(self.rgb2(h)),0.)
        skip=torch.where(mask,F.relu(self.skip(rgb)),0.)
        valid=dense_image(a['valid']).bool()&content
        xyz=dense_image(a['xyz']);safe=torch.where(valid,xyz,0.)
        mu=F.interpolate(a['mu'].transpose(1,2).reshape(-1,3,16,16),size=(256,256),mode='nearest')
        m=self.structure
        absolute=torch.where(valid,(safe-m.absolute_mean[None,:,None,None])/m.absolute_scale[None,:,None,None],0.)
        relative=torch.where(valid,(safe-mu)/m.relative_scale[None,:,None,None],0.)
        inputs=torch.where(content,torch.cat((absolute,relative,valid.float()),1),0.)
        geom=torch.where(content,F.relu(self.geo1(inputs)),0.)
        geom=torch.where(content,F.relu(self.geo2(geom)),0.)
        context=F.interpolate(hidden.transpose(1,2).reshape(-1,128,16,16),size=(256,256),mode='nearest')
        merged=torch.where(content,F.relu(self.fuse(torch.cat((h,geom,context),1))),0.)
        merged=torch.where(content,F.relu(self.refine(merged)),0.)
        up=F.interpolate(merged,size=(512,512),mode='bilinear',align_corners=False)
        y=torch.where(mask,F.relu(self.decode(torch.cat((up,skip),1))),0.)
        active=(a['area']>0).reshape(-1,1,16,16)
        coarse=F.interpolate(torch.where(active,cells.reshape(-1,1,16,16),0.),size=(512,512),mode='bilinear',align_corners=False)
        coverage=F.interpolate(active.float(),size=(512,512),mode='bilinear',align_corners=False)
        logits=coarse/coverage.clamp_min(1e-8)+self.residual(y)
        return torch.where(mask,logits,0.)[:,0]

    def chunk(self,cpu,previous,previous_active,first):
        device=next(self.parameters()).device
        # Preserve the existing scatter/attention path in FP32; AMP is local to
        # the new convolutional branch, where mixed source scatter dtypes do not occur.
        with torch.autocast(device_type=device.type,enabled=False):
            h,active,_,_,cells,history=VideoTraining(self.structure).chunk(
                {k:cpu[k] for k in STRUCTURE_KEYS},previous.float(),previous_active,first,True)
        a={k:cpu[k].to(device) for k in ['rgb512','xyz','valid','content','content512','mu','area']}
        return h,active,self.pixel(a,history,cells)


def loss_sums(logits,a):
    mask=a['content512'].bool();areas=mask.sum((-1,-2)).double()
    weighted=torch.where(mask,logits.double(),-torch.inf)
    frame=torch.logsumexp(weighted,(-1,-2))-areas.log()
    numerator=torch.logsumexp(frame+areas.log(),0)
    labels=a['frame_label'].to(logits.device)
    frame_loss=F.binary_cross_entropy_with_logits(frame,labels.double(),reduction='sum')
    known=a['pixel_known'].bool();gt=a['gt'].float()
    pixel_bce=F.binary_cross_entropy_with_logits(logits.float(),gt,reduction='none')
    bce=torch.where(mask,pixel_bce,0.).sum((-1,-2))/areas.float()
    p=torch.sigmoid(logits.float())*mask;target=gt*mask
    positive=target.sum((-1,-2))>0
    dice=torch.where(positive,1-(2*(p*target).sum((-1,-2))+1)/(p.sum((-1,-2))+target.sum((-1,-2))+1),0.)
    pixel=((bce+dice)*known).sum()
    return numerator,areas.sum(),frame_loss,pixel,known.sum(),frame


def video_loss(model,video,chunk_size=2,use_checkpoint=True):
    device=next(model.parameters()).device
    previous=torch.zeros(256,128,device=device);active=torch.zeros(256,dtype=torch.bool,device=device)
    numerator=torch.tensor(-torch.inf,device=device,dtype=torch.float64);area=torch.tensor(0.,device=device,dtype=torch.float64)
    frames=torch.tensor(0.,device=device);pixels=frames.clone();known=frames.clone();count=video.row['actual_frames']
    def step(cpu,h,pa,first):
        h,pa,logits=model.chunk(cpu,h,pa,first)
        labels={k:cpu[k].to(device) for k in ['content512','gt','pixel_known','frame_label']}
        n,a,f,p,k,_=loss_sums(logits,labels)
        return h,pa,n,a,f,p,k
    for start in range(0,count,chunk_size):
        cpu=video.chunk(start,min(start+chunk_size,count))
        args=(cpu,previous,active,start==0)
        result=checkpoint(step,*args,use_reentrant=False) if use_checkpoint else step(*args)
        previous,active,n,a,f,p,k=result
        numerator=torch.logaddexp(numerator,n);area=area+a;frames=frames+f;pixels=pixels+p;known=known+k
    z=numerator-area.log()
    vl=F.binary_cross_entropy_with_logits(z,torch.tensor(float(video.row['label']),device=device,dtype=z.dtype))
    fl=frames/count;pl=pixels/known.clamp_min(1)
    return vl+.5*fl+pl,{'video':vl,'frame':fl,'pixel':pl,'logit':z}
