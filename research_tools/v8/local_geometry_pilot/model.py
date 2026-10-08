"""Trainable patch geometry and the frozen V8 path with content isolation."""
from __future__ import annotations

import torch
from torch import nn

from research_tools.v8.full_coverage_observation_field.model import FullCoverageModel


def geometry_input(xyz, valid, content, mu, absolute_mean, absolute_scale, relative_scale, condition):
    """[B,T,N,H,W,3] -> [B*T*N,7,H,W]; invalid != padding."""
    valid = valid.bool() & content.bool()
    safe = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
    if condition == "MEAN_REPLICATED":
        absolute = mu[..., None, None, :].expand_as(safe)
        relative = torch.zeros_like(safe)
    elif condition == "DENSE_LOCAL":
        absolute = safe
        relative = safe - mu[..., None, None, :]
    else:
        raise ValueError(condition)
    absolute = torch.where(valid[..., None], (absolute - absolute_mean) / absolute_scale, 0.)
    relative = torch.where(valid[..., None], relative / relative_scale, 0.)
    x = torch.cat((absolute, relative, valid[..., None].to(xyz.dtype)), dim=-1)
    x = torch.where(content[..., None], x, 0.)
    return x.reshape(-1, 16, 16, 7).permute(0, 3, 1, 2).contiguous()


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

    def forward(self, x, content):
        m = content.reshape(-1, 1, 16, 16).bool()
        h = torch.relu(self.conv1(torch.where(m, x, 0.)))
        h = torch.where(m, h, 0.)
        h = torch.relu(self.conv2(h))
        h = torch.where(m, h, 0.)
        count = m.sum(dim=(2, 3)).clamp_min(1)
        mean = h.sum(dim=(2, 3)) / count
        maximum = h.masked_fill(~m, -torch.inf).amax(dim=(2, 3))
        maximum = torch.where(m.any(dim=(2, 3)), maximum, 0.)
        return torch.cat((mean, maximum), dim=-1)


class LocalGeometryModel(FullCoverageModel):
    def __init__(self, condition, geometry_stats, patch_chunk=2048):
        super().__init__("FULL", 141, hidden=128)
        if condition not in ("DENSE_LOCAL", "MEAN_REPLICATED"):
            raise ValueError(condition)
        self.pilot_condition = condition
        self.patch_chunk = patch_chunk
        self.patch_encoder = PatchEncoder()
        for key in ("absolute_mean", "absolute_scale", "relative_scale"):
            self.register_buffer(key, torch.tensor(geometry_stats[key], dtype=torch.float32))

    def _spatial_content(self, h, comp, active):
        src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
        edge_mask = (active[:, src] & active[:, dst])[..., None]
        msg = self.edge_mlp(h[:, dst] - h[:, src]) * edge_mask
        agg, deg = torch.zeros_like(h), torch.zeros_like(h[..., :1])
        agg.index_add_(1, src, msg); agg.index_add_(1, dst, -msg)
        deg.index_add_(1, src, edge_mask.to(h.dtype)); deg.index_add_(1, dst, edge_mask.to(h.dtype))
        h = h + agg / deg.clamp_min(1.)
        ids = comp.long()
        ix = ids[..., None].expand_as(h)
        sums, counts = torch.zeros_like(h), torch.zeros_like(h[..., :1])
        sums.scatter_add_(1, ix, torch.where(active[..., None], h, 0.))
        counts.scatter_add_(1, ids[..., None], active[..., None].to(h.dtype))
        means = sums / counts.clamp_min(1.)
        pooled = torch.gather(means, 1, ix)
        h = h + self.component_residual(torch.cat((h, pooled), dim=-1))
        return torch.where(active[..., None], h, 0.)

    def forward(self, batch, return_evidence=False):
        x = batch["x"]
        b, t, n, _ = x.shape
        content = batch["content"]
        patches = geometry_input(batch["xyz"], batch["valid"], content, batch["mu"],
                                 self.absolute_mean, self.absolute_scale, self.relative_scale, self.pilot_condition)
        cm = content.reshape(-1, 16, 16)
        active_patches = cm.any((1, 2))
        selected = patches[active_patches]; selected_mask = cm[active_patches]
        embeddings = [self.patch_encoder(selected[start:start + self.patch_chunk], selected_mask[start:start + self.patch_chunk])
                      for start in range(0, selected.shape[0], self.patch_chunk)]
        flat_geom = torch.zeros((b * t * n, 64), device=x.device, dtype=x.dtype)
        flat_geom[active_patches] = torch.cat(embeddings)
        geom = flat_geom.reshape(b, t, n, 64)
        active = batch["area"] > 0
        x = torch.where(active[..., None], x, 0.)
        h_prev = torch.zeros((b, n, 128), device=x.device, dtype=x.dtype)
        logits = []
        for ti in range(t):
            h = self.cell_encoder(torch.cat((x[:, ti], geom[:, ti]), dim=-1))
            h = self._spatial_content(h, batch["component_ids"][:, ti], active[:, ti])
            if ti:
                pi = batch["predecessor"][:, ti].long()
                raw = batch["association_weight"][:, ti]
                previous_active = torch.gather(active[:, ti - 1], 1, pi.reshape(b, -1)).reshape_as(pi)
                aw = torch.where(previous_active & active[:, ti, :, None], raw, 0.)
                coverage = aw.sum(-1)
                aw = aw / coverage[..., None].clamp_min(1e-8)
                candidates = torch.gather(h_prev, 1, pi.reshape(b, -1)[..., None].expand(-1, -1, 128)).reshape(b, n, 4, 128)
                prev = (candidates * aw[..., None]).sum(2)
                prev = torch.where((coverage > 0)[..., None], prev, self.no_history)
                dt = batch["delta_t"][:, ti, None, None].expand(b, n, 1)
                aux = torch.cat((dt, coverage[..., None], batch["history_available"][:, ti, :, None].float(), 1. - coverage[..., None]), -1)
            else:
                prev = h_prev
                aux = torch.zeros((b, n, 4), device=x.device, dtype=x.dtype)
            temporal = self.temporal_in(torch.cat((h, aux), -1).reshape(b * n, 132))
            h = self.gru(temporal, prev.reshape(b * n, 128)).reshape(b, n, 128)
            h_prev = torch.where(active[:, ti, :, None], h, 0.)
            logits.append(self.cell_head(h_prev).squeeze(-1))
        cell_logits = torch.stack(logits, 1)
        area = batch["area"]
        weights = area / area.sum((1, 2), keepdim=True)
        # Strict exclusion rather than a small positive clamp weight.
        weighted = cell_logits + torch.where(active, weights.clamp_min(1e-30).log(), -torch.inf)
        z = torch.logsumexp(weighted, dim=(1, 2))
        return {"window_logit": z, "cell_logits": cell_logits, "geometry_embedding": geom} if return_evidence else z
