"""Matched dense encoders, query/partition pooling and causal region memory."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

CONDITIONS = ('RGB_QUERY', 'XYZ_QUERY', 'XYZ_PARTITION')


def relations(values, valid):
    """Raw differences before standardization; no wrapped or invalid edges."""
    right = torch.zeros_like(valid); right[..., :-1] = valid[..., :-1] & valid[..., 1:]
    down = torch.zeros_like(valid); down[..., :-1, :] = valid[..., :-1, :] & valid[..., 1:, :]
    dx = torch.zeros_like(values); dx[..., :-1] = values[..., 1:] - values[..., :-1]
    dy = torch.zeros_like(values); dy[..., :-1, :] = values[..., 1:, :] - values[..., :-1, :]
    masks = torch.cat((valid, right, down), 1)
    nums = torch.cat((values * valid, dx * right, dy * down), 1)
    return nums, masks


def normalized_lse(z, area, dim):
    z = z.float(); area = area.float()
    return torch.logsumexp(z + torch.where(area > 0, area.clamp_min(1e-30).log(), -torch.inf), dim) - area.sum(dim).log()


class DenseRegions(nn.Module):
    def __init__(self, condition, mean=None, scale=None):
        super().__init__()
        if condition not in CONDITIONS: raise ValueError(condition)
        self.condition = condition
        self.register_buffer('mean', torch.tensor(mean if mean is not None else [0.] * 9).view(1, 9, 1, 1))
        self.register_buffer('scale', torch.tensor(scale if scale is not None else [1.] * 9).view(1, 9, 1, 1))
        self.visual = nn.Conv2d(64, 128, 1)
        self.local = nn.Sequential(nn.Conv2d(12, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.fuse = nn.Conv2d(261, 256, 1)
        self.queries = nn.Parameter(torch.randn(64, 256) / 16)
        self.q = nn.Linear(256, 256); self.k = nn.Linear(256, 256); self.v = nn.Linear(256, 256)
        self.norm = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.gru = nn.GRUCell(257, 256)
        self.dq = nn.Linear(256, 256); self.dk = nn.Linear(256, 256); self.dv = nn.Linear(256, 256)
        self.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode(self, rgb, visual, content, xyz=None, valid=None):
        # RGB condition never inspects geometry fields, including validity.
        if self.condition == 'RGB_QUERY': values, mask = rgb, content
        else:
            if xyz is None or valid is None: raise ValueError('GEOMETRY_REQUIRED')
            values, mask = xyz, valid & content
        nums, masks = relations(values, mask)
        if self.condition != 'RGB_QUERY':
            nums = ((nums - self.mean) / self.scale) * masks.repeat_interleave(3, 1)
        nums = nums * content
        loc = self.local(torch.cat((nums, masks.float()), 1).contiguous(memory_format=torch.channels_last)) * content
        vis = F.interpolate(self.visual(visual), size=rgb.shape[-2:], mode='bilinear', align_corners=False) * content
        h, w = rgb.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=rgb.device), torch.linspace(-1, 1, w, device=rgb.device), indexing='ij')
        uv = torch.stack((xx, yy))[None].expand(rgb.shape[0], -1, -1, -1)
        f = self.fuse(torch.cat((vis, loc, rgb * content, uv * content), 1)) * content
        return f.flatten(2).transpose(1, 2)

    def pool(self, f, content):
        active = content.flatten(1)
        scores = self.k(f) @ self.q(self.queries).T / 16
        if self.condition == 'XYZ_PARTITION':
            assignment = scores.float().softmax(-1) * active[..., None]
            weights = assignment / assignment.sum(1, keepdim=True).clamp_min(1e-20)
        else:
            weights = scores.float().masked_fill(~active[..., None], -torch.inf).softmax(1)
            assignment = weights
        h = weights.transpose(1, 2).to(f.dtype) @ self.v(f)
        h = self.norm(h + self.queries)
        h = h + self.attn(h, h, h, need_weights=False)[0]
        return h, assignment

    def frame(self, rgb, visual, content, xyz, valid, dt, hidden, diagnostic=False):
        f = self.encode(rgb, visual, content, xyz, valid)
        h, assignment = self.pool(f, content)
        h = self.gru(torch.cat((h[0], dt.expand(64, 1)), -1), hidden)
        context = F.scaled_dot_product_attention(self.dq(f)[:, None], self.dk(h)[None, None], self.dv(h)[None, None])[:, 0]
        pixels = self.head(torch.cat((f, context), -1))[0, :, 0].float()
        area = content.flatten().float()
        z = normalized_lse(pixels, area, 0)
        if diagnostic: return z, h, pixels.reshape(rgb.shape[-2:]), assignment[0]
        return z, h

    def decode(self, f, hidden, content):
        context = F.scaled_dot_product_attention(self.dq(f)[:, None], self.dk(hidden)[:, None], self.dv(hidden)[:, None])[:, 0]
        pixels = self.head(torch.cat((f, context), -1))[:, :, 0].float()
        return normalized_lse(pixels, content.flatten(1).float(), 1)

    def sequence(self, batch, hidden=None, checkpoint_frames=False):
        if hidden is None: hidden = batch['rgb'].new_zeros(64, 256)
        # Independent spatial frames batched; only the causal GRU is recurrent.
        chunks=[]
        for start in range(0,len(batch['rgb']),32):
            end=start+32
            args=(batch['rgb'][start:end],batch['visual'][start:end],batch['content'][start:end],
                  batch['xyz'][start:end] if 'xyz' in batch else None,batch['valid'][start:end] if 'valid' in batch else None)
            if checkpoint_frames and self.training:f=checkpoint(self.encode,*args,use_reentrant=False)
            else:f=self.encode(*args)
            if checkpoint_frames and self.training:h=checkpoint(lambda x,m:self.pool(x,m)[0],f,batch['content'][start:end],use_reentrant=False)
            else:h=self.pool(f,batch['content'][start:end])[0]
            local_states=[]
            for j in range(len(f)):
                hidden=self.gru(torch.cat((h[j],batch['dt'][start+j].expand(64,1)),-1),hidden)
                local_states.append(hidden)
            hs=torch.stack(local_states)
            if checkpoint_frames and self.training:z=checkpoint(self.decode,f,hs,batch['content'][start:end],use_reentrant=False)
            else:z=self.decode(f,hs,batch['content'][start:end])
            chunks.append(z)
        return torch.cat(chunks),hidden
