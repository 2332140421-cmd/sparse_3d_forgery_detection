"""One lossless 64-frame NPZ format; no dense XYZ, RGB or learned embeddings."""
from pathlib import Path

import numpy as np

from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file, write_npz_atomic

FIELDS = ('x', 'mu', 'area', 'component_ids', 'predecessor', 'association_weight',
          'history_available', 'delta_t', 'valid', 'content')


def write_block(path, arrays, metadata, identity, video_sha):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: np.stack([a[k] for a in arrays]) for k in FIELDS}
    for key in ('valid', 'content'):
        data[key] = np.packbits(data[key].reshape(len(arrays), -1), axis=1, bitorder='little')
    write_npz_atomic(path, **data)
    save_json(path.with_suffix('.json'), {'identity': identity, 'video_sha256': video_sha,
              'sha256': sha256_file(path), 'count': len(arrays), 'frames': metadata})
    return path.stat().st_size + path.with_suffix('.json').stat().st_size


def read_block(path, identity, video_sha):
    path = Path(path); meta = read_json(path.with_suffix('.json'))
    if meta['identity'] != identity or meta['video_sha256'] != video_sha:
        raise ValueError('COMPACT_INPUT_IDENTITY')
    if sha256_file(path) != meta['sha256']: raise ValueError('COMPACT_INPUT_HASH')
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in FIELDS}
    count = meta['count']
    for key in ('valid', 'content'):
        data[key] = np.unpackbits(data[key], axis=1, count=256*16*16, bitorder='little').reshape(count,256,16,16).astype(bool)
    for k, a in data.items():
        if len(a) != count or not np.isfinite(a).all(): raise ValueError(f'COMPACT_INPUT_SHAPE_FINITE:{k}')
    if data['x'].shape != (count,256,77) or data['mu'].shape != (count,256,3):
        raise ValueError('COMPACT_INPUT_FEATURE_SHAPE')
    if not np.array_equal(data['content'].sum((-1,-2)).astype(np.float32),data['area']):
        raise ValueError('COMPACT_AREA_MASK')
    if (data['valid'] & ~data['content']).any(): raise ValueError('COMPACT_VALID_OUTSIDE_CONTENT')
    return [{k: np.asarray(data[k][i]) for k in FIELDS} for i in range(count)], meta['frames']
