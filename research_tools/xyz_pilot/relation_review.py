"""One-frame CPU routing audit; no backbone/frontend or training is called.

Immediate caller: python -m research_tools.xyz_pilot.relation_review.
Output: compact identity/routing JSON for the relation design review. The actual
Cache.get and XYZPilot.forward are required to audit execution, not a toy route.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from . import model as implementation
from .data import Cache, OUT

OUTPUT = OUT.parent / 'pvst_xyz_relation_review_v1'
HISTORICAL = 'a1f320a1586078d7613c3415dc9d5df688040740'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def routing_evidence(xyz, valid, h, points, keys, index):
    expected = torch.where(valid.reshape(-1))[0]
    assert torch.equal(points, expected)
    coordinates = torch.floor(xyz.reshape(-1, 3)[points] / h).long()
    assert torch.equal(keys[index], coordinates)
    counts = torch.bincount(index, minlength=len(keys))
    assert int(counts.sum()) == len(points) and bool((counts > 0).all())
    examples = []
    for z in keys[:, 2].unique().tolist():
        layer = keys[keys[:, 2] == z]
        if len(layer) >= 2:
            examples = layer[:5].tolist()
            break
    return dict(xyz_shape=list(xyz.shape), keys_shape=list(keys.shape),
                valid_points=len(points), block_count=len(keys),
                member_count_sum=int(counts.sum()), membership_exact=True,
                same_z_distinct_xy=examples,
                members_min=int(counts.min()), members_max=int(counts.max()))


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {}
    for name in ['model.py', 'data.py', 'experiment.py']:
        rel = 'research_tools/xyz_pilot/' + name
        old = subprocess.check_output(['git', 'show', HISTORICAL + ':' + rel], cwd=root)
        current = (root / rel).read_bytes()
        assert current == old, rel
        sources[rel] = hashlib.sha256(current).hexdigest()
    config = json.loads((OUT / 'config.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    identity = json.loads((OUT / 'run_identity.json').read_text())
    initial = json.loads((OUT / 'initialization.json').read_text())
    assert sha(OUT / 'config.json') == identity['config_sha256']
    assert sha(OUT / 'manifest.json') == identity['manifest_sha256']
    assert sources['research_tools/xyz_pilot/model.py'] == identity['model_source_sha256']
    checkpoints = {}
    for condition in ['XYZ_CURRENT', 'XYZ_TEMPORAL']:
        path = OUT / (condition + '_last.pt')
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert cp['condition'] == condition and cp['updates'] == 576
        assert cp['identity'] == identity
        assert cp['initial_sha256'] == initial['sha256']
        checkpoints[condition] = dict(path=str(path), sha256=sha(path), updates=cp['updates'])
    model = implementation.XYZPilot(config['h']).eval().requires_grad_(False)
    model.load_state_dict(cp['state_dict'], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    row = next(r for r in manifest['rows'] if r['video_id'] == '9mygy9_fake')
    cache_path = OUT / 'cache' / row['video_id'] / '00100.pt'
    cache_sha = sha(cache_path)
    frame = Cache(config).get(row, 100, device='cpu')
    raw = torch.load(cache_path, map_location='cpu', weights_only=True)
    assert torch.equal(frame['xyz'], raw['xyz_provider'] / config['scale'])
    evidence = {}
    original = implementation.route

    def observed_route(xyz, valid, h):
        result = original(xyz, valid, h)
        evidence.update(routing_evidence(xyz, valid, h, *result))
        return result

    implementation.route = observed_route
    try:
        with torch.no_grad():
            output = model([frame], temporal=True)
    finally:
        implementation.route = original
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert sha(cache_path) == cache_sha
    for condition, item in checkpoints.items():
        assert sha(item['path']) == item['sha256']
    attention = {n: dict(heads=m.num_heads, dimension=m.embed_dim)
                 for n, m in model.named_modules() if isinstance(m, torch.nn.MultiheadAttention)}
    classes = sorted({type(m).__name__ for m in model.modules()})
    assert 'SlabPool' not in classes and 'RGBDSetSlices' not in classes
    assert all(torch.isfinite(output[k]).all() for k in ['pixel', 'frame'])
    import numpy as np
    counts = json.loads((OUT / 'scale_selection.json').read_text())['node_counts']
    window_counts = {}
    for split in ['train', 'dev']:
        totals = []
        largest = None
        for r in manifest['rows']:
            if r['split'] != split:
                continue
            ns = counts[r['video_id']][str(config['h'])]
            for i in range(len(ns)):
                ids = implementation.history_indices(r['pts'], i, config['history_slots'], config['history_step_seconds'])
                total = sum(ns[j] for j in ids if j is not None)
                totals.append(total)
                if largest is None or total > largest['blocks']:
                    largest = dict(blocks=total, video_id=r['video_id'], target=i, slots=ids)
        window_counts[split] = dict(windows=len(totals), median=float(np.median(totals)),
                                    p95=float(np.percentile(totals, 95)), maximum=largest)
    result = dict(status='COMPLETE', exit_code=0, execution_head=head,
                  historical_execution_head=HISTORICAL, audit_source_sha256=sha(__file__),
                  sources=sources, window_counts_from_existing_records=window_counts,
                  checkpoints=checkpoints, model_class=type(model).__name__,
                  definition=str(Path(implementation.__file__).resolve()),
                  cache_path=str(cache_path), cache_sha256=cache_sha,
                  frame_index=100, pts=frame['pts'], native_hw=row['native_hw'],
                  work_hw=row['work_hw'], scale=config['scale'], h=config['h'],
                  normalization_count=1, route='XYZPilot.forward -> encode -> route',
                  attention=attention, module_classes=classes, routing=evidence,
                  pixel_shape=list(output['pixel'].shape), state_unchanged=True,
                  device='cpu', dtype='FP32 computation; stored visual FP16 converted by Cache.get',
                  scope='one cached frame only; B weights, no historical slots; not metric reproduction',
                  new_geometry_frames=0, new_backbone_frames=0,
                  seconds=time.monotonic()-started)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
