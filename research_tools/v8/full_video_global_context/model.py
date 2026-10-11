"""Reuse the trained global spatial path and the existing causal state update."""
import torch

from research_tools.v8.full_video_streaming.model import StreamingModel
from research_tools.v8.local_geometry_pilot.model import geometry_input


class GlobalStreamingModel(StreamingModel):
    def encode_spatial(self, frame):
        m = self.model
        x, mu, content = frame['x'], frame['mu'], frame['content']
        active = frame['area'] > 0
        mean_xyz = mu[:, None, :, None, None, :].expand(-1, 1, -1, 16, 16, -1)
        patches = geometry_input(mean_xyz, frame['valid'][:, None], content[:, None], mu[:, None],
                                 m.absolute_mean, m.absolute_scale, m.relative_scale, 'MEAN_REPLICATED')
        cm = content.reshape(-1, 16, 16)
        selected = cm.any((1, 2))
        geom = torch.zeros((256, 64), device=x.device, dtype=x.dtype)
        if selected.any():
            geom[selected] = m.patch_encoder(patches[selected], cm[selected])
        h = m.cell_encoder(torch.cat((torch.where(active[..., None], x, 0.), geom[None]), -1))
        # Critical: NOT the old _spatial_content path, which has no global attention.
        return m.spatial(h, frame['component_ids'], frame['area'], mu,
                         (frame['valid'] & content).sum((-1, -2)))
