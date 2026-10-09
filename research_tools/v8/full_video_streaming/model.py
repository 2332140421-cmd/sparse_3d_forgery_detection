"""Stateful evaluation of the frozen LocalGeometryModel, without new parameters."""
from __future__ import annotations

import torch

from research_tools.v8.local_geometry_pilot.model import geometry_input


class StreamingModel:
    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def forward_step(self, frame, state=None):
        m = self.model
        x = frame['x']; b, n, _ = x.shape
        if b != 1 or n != 256:
            raise ValueError('STREAM_REQUIRES_ONE_FRAME_256_CELLS')
        active = frame['area'] > 0
        patches = geometry_input(frame['xyz'][:, None], frame['valid'][:, None], frame['content'][:, None],
                                 frame['mu'][:, None], m.absolute_mean, m.absolute_scale, m.relative_scale, m.pilot_condition)
        cm = frame['content'].reshape(-1, 16, 16)
        selected = cm.any((1, 2))
        geom = torch.zeros((n, 64), device=x.device, dtype=x.dtype)
        if selected.any():
            geom[selected] = m.patch_encoder(patches[selected], cm[selected])
        h = m.cell_encoder(torch.cat((torch.where(active[..., None], x, 0.), geom[None]), -1))
        h = m._spatial_content(h, frame['component_ids'], active)
        if state is None:
            prev = torch.zeros((b, n, 128), device=x.device, dtype=x.dtype)
            aux = torch.zeros((b, n, 4), device=x.device, dtype=x.dtype)
            acc = torch.tensor(-torch.inf, device=x.device, dtype=torch.float64)
            total_area = torch.tensor(0., device=x.device, dtype=torch.float64)
            count = 0
        else:
            pi = frame['predecessor'].long()
            pa = torch.gather(state['previous_active'], 1, pi.reshape(b, -1)).reshape_as(pi)
            aw = torch.where(pa & active[..., None], frame['association_weight'], 0.)
            coverage = aw.sum(-1)
            aw = aw / coverage[..., None].clamp_min(1e-8)
            candidates = torch.gather(state['h_prev'], 1, pi.reshape(b, -1)[..., None].expand(-1, -1, 128)).reshape(b, n, 4, 128)
            prev = (candidates * aw[..., None]).sum(2)
            prev = torch.where((coverage > 0)[..., None], prev, m.no_history)
            dt = frame['delta_t'].reshape(b, 1, 1).expand(b, n, 1)
            aux = torch.cat((dt, coverage[..., None], frame['history_available'][..., None].float(), 1.-coverage[..., None]), -1)
            acc, total_area, count = state['log_numerator'], state['total_area'], state['frame_count']
        temporal = m.temporal_in(torch.cat((h, aux), -1).reshape(b*n, 132))
        h = m.gru(temporal, prev.reshape(b*n, 128)).reshape(b, n, 128)
        hidden = torch.where(active[..., None], h, 0.)
        logits = m.cell_head(hidden).squeeze(-1)
        area = frame['area'].double()
        weighted = logits.double() + torch.where(active, area.clamp_min(1).log(), -torch.inf)
        numerator = torch.logsumexp(weighted, (0, 1)); a = area.sum()
        if not torch.isfinite(numerator) or a <= 0:
            raise ValueError('NO_CONTENT_OR_NONFINITE_RESPONSE')
        acc = torch.logaddexp(acc, numerator); total_area = total_area + a
        next_state = {'h_prev': hidden, 'previous_active': active, 'log_numerator': acc,
                      'total_area': total_area, 'frame_count': count+1}
        return {'cell_logits': logits, 'frame_logit': numerator-a.log(),
                'video_logit': acc-total_area.log(), 'state': next_state}


def frame_at(batch, index):
    return {k: v[:, index] for k, v in batch.items() if k in
            ('x', 'xyz', 'valid', 'content', 'mu', 'component_ids', 'predecessor',
             'association_weight', 'history_available', 'delta_t', 'area')}


def state_to(state, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.items()}
