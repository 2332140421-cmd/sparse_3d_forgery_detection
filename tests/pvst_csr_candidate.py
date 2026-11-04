"""CSR-only candidate reproduction; test caller only, blocked by torch2.5 constructor backward."""
import torch
import torch.nn.functional as F
from research_tools.pvst.sparse_attention import CSRSoftmax,intersection_padding

def csr_candidate_head(q,k,v,bias,crow,col,relation,head,dropout,training,trace=None):
    def emit(stage,tensor):
        if trace is not None:trace(stage,tensor,head)
    emit('head_start',q)
    # Public differentiable values(); no _values(), detach or dense conversion.
    with torch.autocast(device_type=q.device.type,enabled=False):
        qf,kf,vf=q.float().contiguous(),k.float().contiguous(),v.float().contiguous()
        pad=intersection_padding(len(qf),len(kf))
        if pad:
            kf=torch.cat((kf,kf.new_zeros((pad,kf.shape[1]))),0)
            vf=torch.cat((vf,vf.new_zeros((pad,vf.shape[1]))),0)
        emit('FP32_contiguous',qf)
        mask=torch.sparse_csr_tensor(crow,col,qf.new_ones(len(col)),size=(len(q),len(kf)))
        emit('CSR_created',mask)
        scores=torch.sparse.sampled_addmm(mask,qf,kf.T,beta=0,alpha=q.shape[-1]**-.5)
        emit('sampled_addmm_complete',scores)
        del mask
        values=scores.values()+bias.reshape(-1,bias.shape[-1])[relation,head].float()
        del scores
        emit('bias_added',values);emit('softmax_start',values)
        probabilities=CSRSoftmax.apply(values,crow);del values
        emit('softmax_complete',probabilities)
        dropped=F.dropout(probabilities,dropout,training);del probabilities
        emit('dropout_complete',dropped)
        matrix=torch.sparse_csr_tensor(crow,col,dropped,size=(len(q),len(kf)))
        del dropped
        emit('CSR_weights_complete',matrix)
        result=torch.sparse.mm(matrix,vf);emit('sparse_mm_complete',result)
        return result.to(q.dtype)
