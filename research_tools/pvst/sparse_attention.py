"""Exact valid-edge attention for SparseLayer; no new learned parameters.

SparseLayer is the only model caller. The checked CUDA SDDMM/softmax/SpMM
requires FP32, so cast one head at a time, never expand full E x head_dim.
The old query gather remains the historical default, not a silent fallback.
"""
import torch
import torch.nn.functional as F


def edge_topology(graph,dilation,chunk):
    """CSR columns sorted within rows; relation IDs follow the same permutation."""
    degrees=[];columns=[];relations=[];n=len(graph.coords)
    for start in range(0,n,chunk):
        index,frame=graph.query(start,min(start+chunk,n),dilation)
        ordered,slot=index.masked_fill(index<0,n).sort(dim=1)
        valid=ordered<n
        degrees.append(valid.sum(1));columns.append(ordered[valid])
        relations.append((frame[:,None]*81+slot)[valid])
    degree=torch.cat(degrees);crow=torch.cat((degree.new_zeros(1),degree.cumsum(0)))
    col=torch.cat(columns);relation=torch.cat(relations)
    return crow,col,relation


def sparse_head(q,k,v,bias,crow,col,relation,head,dropout,training,trace=None):
    def emit(stage,tensor):
        if trace is not None:trace(stage,tensor,head)
    emit('head_start',q)
    # Public differentiable values(); no _values(), detach or dense conversion.
    with torch.autocast(device_type=q.device.type,enabled=False):
        qf,kf,vf=q.float().contiguous(),k.float().contiguous(),v.float().contiguous()
        emit('FP32_contiguous',qf)
        mask=torch.sparse_csr_tensor(crow,col,qf.new_ones(len(col)),size=(len(q),len(k)))
        emit('CSR_created',mask)
        scores=torch.sparse.sampled_addmm(mask,qf,kf.T,beta=0,alpha=q.shape[-1]**-.5)
        emit('sampled_addmm_complete',scores)
        coo=scores.to_sparse_coo()
        emit('CSR_to_COO_complete',coo)
        values=coo.values()+bias.reshape(-1,bias.shape[-1])[relation,head].float()
        emit('bias_added',values)
        # Builder guarantees unique sorted columns/rows. Avoid a redundant E-sort.
        scores=torch.sparse_coo_tensor(coo.indices(),values,coo.shape,is_coalesced=True)
        emit('softmax_start',scores)
        weights=torch.sparse.softmax(scores,dim=1)
        emit('softmax_complete',weights)
        dropped=F.dropout(weights.values(),dropout,training)
        emit('dropout_complete',dropped)
        matrix=torch.sparse_coo_tensor(weights.indices(),dropped,weights.shape,is_coalesced=True)
        # COO SpMM backward in torch2.5 creates dense N x N. CSR uses SDDMM.
        matrix=matrix.to_sparse_csr();emit('COO_to_CSR_complete',matrix)
        result=torch.sparse.mm(matrix,vf);emit('sparse_mm_complete',result)
        return result.to(q.dtype)
