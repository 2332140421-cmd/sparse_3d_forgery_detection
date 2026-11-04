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


def sparse_head(q,k,v,bias,crow,col,relation,head,dropout,training):
    # Public differentiable values(); no _values(), detach or dense conversion.
    with torch.autocast(device_type=q.device.type,enabled=False):
        qf,kf,vf=q.float().contiguous(),k.float().contiguous(),v.float().contiguous()
        mask=torch.sparse_csr_tensor(crow,col,qf.new_ones(len(col)),size=(len(q),len(k)))
        scores=torch.sparse.sampled_addmm(mask,qf,kf.T,beta=0,alpha=q.shape[-1]**-.5)
        coo=scores.to_sparse_coo()
        values=coo.values()+bias.reshape(-1,bias.shape[-1])[relation,head].float()
        # Builder guarantees unique sorted columns/rows. Avoid a redundant E-sort.
        scores=torch.sparse_coo_tensor(coo.indices(),values,coo.shape,is_coalesced=True)
        weights=torch.sparse.softmax(scores,dim=1)
        dropped=F.dropout(weights.values(),dropout,training)
        matrix=torch.sparse_coo_tensor(weights.indices(),dropped,weights.shape,is_coalesced=True)
        # COO SpMM backward in torch2.5 creates dense N x N. CSR uses SDDMM.
        return torch.sparse.mm(matrix.to_sparse_csr(),vf).to(q.dtype)
