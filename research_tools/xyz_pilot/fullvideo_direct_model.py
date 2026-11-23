"""ADR-0055: all-frame original-block relations and localization correction.

Caller fullvideo_direct_probe. Frozen observations are inputs, not trainable C/R.
CPU-held frame tensors are moved inside bound checkpoint functions to avoid
retaining a full video's dense point activations on GPU.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .joint_model import JointLayer
from .model import route

SCALE = 12.880994319915771
MODES = ('FRAME_ONLY', 'FULL_VIDEO')


class DirectLayer(JointLayer):
    """Same Q/K/V projections and residuals as JointLayer, exact SDPA."""
    def __init__(self, query_chunk=0):
        super().__init__(128, 4)
        self.query_chunk = query_chunk

    def attend(self, x, position):
        if not len(x):
            return x
        content = self.norm(x)
        qk = content + position
        w, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = F.linear(qk, w[:128], b[:128])
        k = F.linear(qk, w[128:256], b[128:256])
        v = F.linear(content, w[256:], b[256:])
        q, k, v = [a.reshape(1, -1, 4, 32).transpose(1, 2) for a in (q, k, v)]
        chunk = self.query_chunk or len(x)
        # Each Q chunk sees ALL keys/values and the full normalization axis.
        outputs = [F.scaled_dot_product_attention(q[:, :, i:i+chunk], k, v,
                   dropout_p=0., is_causal=False) for i in range(0, len(x), chunk)]
        a = torch.cat(outputs, dim=2).transpose(1, 2).reshape(len(x), 128)
        y = x + self.attn.out_proj(a)
        return y + self.ff(y)


def prepare_frame(packet):
    """Route raw provider XYZ once in FP32; labels never determine membership."""
    f = dict(packet)
    assert 'xyz_provider' in f and 'xyz' not in f
    xyz = f['xyz_provider'].float() / SCALE
    assert xyz.shape == (*f['work_hw'], 3)
    assert f['valid'].shape == tuple(f['work_hw'])
    assert torch.isfinite(xyz[f['valid']]).all()
    f['points'], f['keys'], f['index'] = route(xyz, f['valid'], .5)
    assert len(f['points']) == int(f['valid'].sum()) == len(f['index'])
    f['xyz_normalized'] = xyz
    return f


class FullVideoDirect(nn.Module):
    def __init__(self, query_chunk=0, activation_checkpoint=True):
        super().__init__()
        self.activation_checkpoint = activation_checkpoint
        self.visual = nn.Sequential(nn.Conv2d(1024, 128, 1), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 256))
        self.point = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.score = nn.Linear(128, 1)
        self.value = nn.Linear(128, 128)
        self.position = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128))
        self.time = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 128))
        self.layers = nn.ModuleList([DirectLayer(query_chunk) for _ in range(2)])
        self.pixel_context = nn.Linear(128, 32)
        self.correction = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                       nn.GroupNorm(4, 32), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def checked(self, fn, *args):
        if self.activation_checkpoint and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode(self, f):
        device = next(self.parameters()).device
        h, w = f['work_hw']
        local = self.visual(f['patch'].to(device)[None])
        local = F.interpolate(local, size=f['dino_padded_hw'], mode='bilinear',
                              align_corners=False)[0, :, :h, :w]
        points, index = f['points'].to(device), f['index'].to(device)
        keys = f['keys'].to(device)
        if not len(points):
            return local.new_empty((0, 128), dtype=torch.float32)
        absolute = f['xyz_normalized'].to(device).reshape(-1, 3)[points]
        relative = absolute / .5 - torch.floor(absolute / .5) - .5
        gamma, beta = self.geometry(torch.cat((absolute, relative), -1)).chunk(2, -1)
        v = local.permute(1, 2, 0).reshape(-1, 128)[points]
        x = self.point(v * (1 + .1 * gamma.tanh()) + .1 * beta)
        z = self.score(x).float()[:, 0]
        mx = z.new_full((len(keys),), -torch.inf).scatter_reduce(
            0, index, z, reduce='amax', include_self=True)
        a = (z - mx[index]).exp()
        den = a.new_zeros(len(keys)).index_add(0, index, a)
        return x.new_zeros((len(keys), 128), dtype=torch.float32).index_add(
            0, index, self.value(x).float() * (a / den[index])[:, None])

    def relate(self, tokens, position, counts, mode):
        assert mode in MODES and sum(counts) == len(tokens)
        for layer in self.layers:
            if mode == 'FULL_VIDEO':
                tokens = self.checked(layer.attend, tokens, position)
            else:
                xs, ps = tokens.split(counts), position.split(counts)
                tokens = torch.cat([self.checked(layer.attend, x, p) for x, p in zip(xs, ps)])
        return tokens

    def decode(self, f, r):
        device = next(self.parameters()).device
        h, w = f['work_hw']
        ctx = torch.zeros(h*w, 32, device=device, dtype=r.dtype)
        if len(r):
            ctx = ctx.index_copy(0, f['points'].to(device),
                                 self.pixel_context(r)[f['index'].to(device)].to(ctx.dtype))
        detail = F.interpolate(f['detail'].to(device)[None], size=f['dino_padded_hw'],
                               mode='bilinear', align_corners=False)[..., :h, :w]
        delta = self.correction(torch.cat((detail, ctx.T.reshape(1, 32, h, w)), 1))[0, 0]
        return f['z0'].to(device).float() + delta.float()

    def forward(self, frames, mode, progress=None, diagnostics=False):
        assert frames and all(b['pts'] > a['pts'] for a, b in zip(frames, frames[1:]))
        counts = [len(f['keys']) for f in frames]
        cs, ps = [], []
        device = next(self.parameters()).device
        for i, f in enumerate(frames):
            # Bind frame at definition time: checkpoint backward must NOT read
            # the final loop frame. No differentiable tensor is detached.
            def encode_bound(_patch, frame=f):
                return self.encode(frame)
            cs.append(self.checked(encode_bound, f['patch']))
            centers = (f['keys'].to(device).float() + .5) * .5
            dt = centers.new_full((len(centers), 1), f['pts'] - frames[0]['pts'])
            ps.append(self.position(centers) + self.time(dt))
            if progress and (i % 20 == 0 or i+1 == len(frames)):
                progress('encode', frame=i+1, T=len(frames))
        tokens = torch.cat(cs)
        r = self.relate(tokens, torch.cat(ps), counts, mode)
        outputs = []
        for i, (f, current) in enumerate(zip(frames, r.split(counts))):
            def decode_bound(x, frame=f):
                return self.decode(frame, x)
            outputs.append(self.checked(decode_bound, current))
            if progress and (i % 20 == 0 or i+1 == len(frames)):
                progress('decode', frame=i+1, T=len(frames))
        return (outputs, dict(tokens=tokens, relations=r, counts=counts)) if diagnostics else outputs


def video_loss(outputs, frames):
    terms = []
    for z, f in zip(outputs, frames):
        assert z.shape == tuple(f['work_hw'])
        if not f['supervised']:
            continue
        gt = f['gt_work'].to(z.device).float()
        p = z.float().sigmoid()
        terms.append(torch.stack((F.binary_cross_entropy_with_logits(z.float(), gt),
                      1 - (2*(p*gt).sum()+1)/(p.sum()+gt.sum()+1))))
    if not terms:
        raise ValueError('No supervised frames')
    components = torch.stack(terms).mean(0)
    return components.sum(), components
