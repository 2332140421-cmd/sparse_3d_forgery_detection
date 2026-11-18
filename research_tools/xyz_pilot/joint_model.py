"""ADR-0044: replace both old relations with two window attention layers.

Joint experiment caller: cached generic observations -> trainable C0 -> R and
pixel/frame outputs. Reuses the existing joint block encoder and pixel decoder;
no forensic weights or old spatial/context layer participates.
"""
import math
import torch
from torch import nn
from .model import XYZPilot


class JointLayer(nn.Module):
    def __init__(self, d=128, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))

    def attend(self, x, position):
        content = self.norm(x)
        qk = content + position
        a, _ = self.attn(qk[None], qk[None], content[None], need_weights=False)
        y = x + a[0]
        return y + self.ff(y)

    def forward(self, x, position, frame_ids, temporal):
        if temporal:
            return self.attend(x, position)
        # Exact block-diagonal mask, evaluated as separate dense blocks. All
        # tokens update, including duplicate slots of the same observation.
        output = torch.zeros_like(x)
        for identity in frame_ids.unique():
            ids = torch.where(frame_ids == identity)[0]
            output = output.index_copy(0, ids, self.attend(x[ids], position[ids]))
        return output


class XYZJointPilot(XYZPilot):
    def __init__(self, h, d=128):
        super().__init__(h, d)
        del self.space, self.context, self.frame
        self.layers = nn.ModuleList([JointLayer(d, 4) for _ in range(2)])
        self.evidence = nn.Linear(d, 1)

    def encode(self, frame):
        return super().encode(frame, spatial=False)

    def read_window(self, encoded, relative_pts, identities, temporal):
        assert len(encoded) == len(relative_pts) == len(identities)
        assert relative_pts[-1] == 0 and all(t <= 0 for t in relative_pts)
        current = encoded[-1]
        nonempty = [(e, dt, fid) for e, dt, fid in zip(encoded, relative_pts, identities) if len(e['C'])]
        if not len(current['C']):
            # Explicit learned empty relation representation; visual path still
            # predicts all pixels, with no fabricated geometric membership.
            return current['C'], self.evidence(self.empty[0]).squeeze()
        tokens = torch.cat([e['C'] for e, _, _ in nonempty])
        position = torch.cat([e['position'] + self.time(e['C'].new_full((len(e['C']), 1), dt))
                              for e, dt, _ in nonempty])
        frame_ids = torch.cat([torch.full((len(e['C']),), fid, device=tokens.device, dtype=torch.long)
                               for e, _, fid in nonempty])
        for layer in self.layers:
            tokens = layer(tokens, position, frame_ids, temporal)
        current_r = tokens[-len(current['C']):]
        evidence = self.evidence(current_r).float().squeeze(-1)
        frame = torch.logsumexp(evidence, dim=0) - math.log(len(evidence))
        return current_r, frame

    def forward(self, frames, temporal):
        assert frames[-1] is not None
        target = frames[-1]
        encoded, times, identities = [], [], []
        memo = {}
        for frame in frames:
            if frame is None:
                continue
            assert frame['pts'] <= target['pts']
            identity = frame['frame_index']
            if identity not in memo:
                memo[identity] = self.encode(frame)
            encoded.append(memo[identity])
            times.append(frame['pts'] - target['pts'])
            identities.append(identity)
        relation, frame_logit = self.read_window(encoded, times, identities, temporal)
        return dict(pixel=self.decode(target, encoded[-1], relation), frame=frame_logit,
                    blocks=len(encoded[-1]['C']),window_blocks=sum(len(e['C']) for e in encoded))
