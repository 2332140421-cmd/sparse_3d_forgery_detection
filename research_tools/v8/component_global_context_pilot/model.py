"""Same-frame region attention; only the membership partition is conditional."""
from __future__ import annotations

import torch
from torch import nn

from research_tools.v8.local_geometry_pilot.model import LocalGeometryModel, geometry_input


CONDITIONS = ("GLOBAL_COMPONENT", "GLOBAL_CELL")


def area_lse(logits, area, dims):
    total = area.sum(dims, keepdim=True)
    log_weight = torch.where(area > 0, (area / total.clamp_min(1)).clamp_min(1e-30).log(), -torch.inf)
    return torch.logsumexp(logits + log_weight, dim=dims)


class GlobalContextModel(LocalGeometryModel):
    def __init__(self, condition, statistics, patch_chunk=2048):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        super().__init__("MEAN_REPLICATED", statistics, patch_chunk)
        self.global_condition = condition
        self.region_projection = nn.Sequential(nn.Linear(133, 128), nn.ReLU())
        self.region_attention = nn.MultiheadAttention(128, 4, dropout=0., batch_first=True)
        self.region_norm = nn.LayerNorm(128)

    def region_context(self, h, component, area, mu, geometry_count):
        """[frame_batch,cell,128] -> region context broadcast to the same cells."""
        b, n, d = h.shape
        active = area > 0
        ids = (component.long() if self.global_condition == "GLOBAL_COMPONENT" else
               torch.arange(n, device=h.device)[None].expand(b, n))
        hi = ids[..., None].expand(-1, -1, d)
        sums = torch.zeros_like(h).scatter_add(1, hi, torch.where(active[..., None], h * area[..., None], 0.))
        ra = torch.zeros_like(area).scatter_add(1, ids, area)
        size = torch.zeros_like(area).scatter_add(1, ids, active.to(h.dtype))
        gc = torch.where(active, geometry_count.to(h.dtype), 0.)
        count = torch.zeros_like(area).scatter_add(1, ids, gc)
        xyz = torch.zeros_like(mu).scatter_add(1, ids[..., None].expand(-1, -1, 3),
                                             torch.where((gc > 0)[..., None], mu * gc[..., None], 0.))
        xyz = xyz / count[..., None].clamp_min(1)
        xyz = torch.where((count > 0)[..., None], (xyz - self.absolute_mean) / self.absolute_scale, 0.)
        nodes = torch.cat((sums / ra[..., None].clamp_min(1), xyz,
                           (ra / area.sum(-1, keepdim=True).clamp_min(1))[..., None], (size / 4)[..., None]), -1)
        valid_region = ra > 0
        # A dummy key avoids an all-masked softmax only for fully empty synthetic frames.
        # It is not broadcast and never enters any output area aggregation.
        safe_keys = valid_region.clone()
        safe_keys[:, 0] |= ~valid_region.any(-1)
        u = self.region_projection(nodes)
        context, _ = self.region_attention(u, u, u, key_padding_mask=~safe_keys, need_weights=False)
        context = torch.where(valid_region[..., None], self.region_norm(u + context), 0.)
        broadcast = torch.gather(context, 1, hi)
        return torch.where(active[..., None], broadcast, 0.), nodes, valid_region

    def spatial(self, h, component, area, mu, geometry_count):
        active = area > 0
        src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
        mask = (active[:, src] & active[:, dst])[..., None]
        msg = self.edge_mlp(h[:, dst] - h[:, src]) * mask
        agg, degree = torch.zeros_like(h), torch.zeros_like(h[..., :1])
        agg.index_add_(1, src, msg); agg.index_add_(1, dst, -msg)
        degree.index_add_(1, src, mask.to(h.dtype)); degree.index_add_(1, dst, mask.to(h.dtype))
        h = h + agg / degree.clamp_min(1)
        context, _, _ = self.region_context(h, component, area, mu, geometry_count)
        # Replaces, rather than stacks on, the old component residual.
        h = h + self.component_residual(torch.cat((h, context), -1))
        return torch.where(active[..., None], h, 0.)

    def forward(self, batch, return_evidence=False):
        x, area = batch["x"], batch["area"]
        b, t, n, _ = x.shape
        content, valid, mu = batch["content"], batch["valid"], batch["mu"]
        # MEAN_REPLICATED needs only the saved mean and the identical pixel masks,
        # not a reread/transfer of dense XYZ. Same existing geometry_input formula.
        mean_xyz = mu[..., None, None, :].expand(-1, -1, -1, 16, 16, -1)
        patches = geometry_input(mean_xyz, valid, content, mu, self.absolute_mean,
                                 self.absolute_scale, self.relative_scale, "MEAN_REPLICATED")
        cm = content.reshape(-1, 16, 16)
        selected_mask = cm.any((1, 2))
        px, masks = patches[selected_mask], cm[selected_mask]
        embeddings = [self.patch_encoder(px[i:i + self.patch_chunk], masks[i:i + self.patch_chunk])
                      for i in range(0, len(px), self.patch_chunk)]
        geom = torch.zeros((b * t * n, 64), device=x.device, dtype=x.dtype)
        if embeddings:
            geom[selected_mask] = torch.cat(embeddings)
        active = area > 0
        h = self.cell_encoder(torch.cat((torch.where(active[..., None], x, 0.), geom.reshape(b, t, n, 64)), -1))
        h = self.spatial(h.reshape(b * t, n, 128), batch["component_ids"].reshape(b * t, n),
                         area.reshape(b * t, n), mu.reshape(b * t, n, 3),
                         (valid & content).sum((-1, -2)).reshape(b * t, n)).reshape(b, t, n, 128)
        previous = torch.zeros((b, n, 128), device=x.device, dtype=x.dtype)
        logits = []
        for ti in range(t):
            if ti:
                pi = batch["predecessor"][:, ti].long()
                raw = batch["association_weight"][:, ti]
                pa = torch.gather(active[:, ti - 1], 1, pi.reshape(b, -1)).reshape_as(pi)
                aw = torch.where(pa & active[:, ti, :, None], raw, 0.)
                coverage = aw.sum(-1)
                aw = aw / coverage[..., None].clamp_min(1e-8)
                candidates = torch.gather(previous, 1, pi.reshape(b, -1)[..., None].expand(-1, -1, 128)).reshape(b, n, 4, 128)
                prev = (candidates * aw[..., None]).sum(2)
                prev = torch.where((coverage > 0)[..., None], prev, self.no_history)
                dt = batch["delta_t"][:, ti, None, None].expand(b, n, 1)
                aux = torch.cat((dt, coverage[..., None], batch["history_available"][:, ti, :, None].float(), 1. - coverage[..., None]), -1)
            else:
                prev = previous
                aux = torch.zeros((b, n, 4), device=x.device, dtype=x.dtype)
            inputs = self.temporal_in(torch.cat((h[:, ti], aux), -1).reshape(b * n, 132))
            current = self.gru(inputs, prev.reshape(b * n, 128)).reshape(b, n, 128)
            previous = torch.where(active[:, ti, :, None], current, 0.)
            logits.append(self.cell_head(previous).squeeze(-1))
        cell_logits = torch.stack(logits, 1)
        window = area_lse(cell_logits, area, (1, 2))
        if not return_evidence:
            return window
        return {"window_logit": window, "cell_logits": cell_logits,
                "frame_logits": area_lse(cell_logits, area, (2,))}
