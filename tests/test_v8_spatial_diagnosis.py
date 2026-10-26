import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from research_tools.v8.spatial_localization.diagnose import area_summary, audit_masks, lse_example, stopping
from research_tools.v8.spatial_localization.data import mask_to_canvas


def test_area_baseline_excludes_empty_frames_and_is_not_pooled():
    result = area_summary([0, .01, .09])
    assert result['positive_gt_frames'] == 2 and result['empty_gt_frames'] == 1
    assert result['area_median'] == pytest.approx(.05)
    assert result['ap_prevalence_baseline'] == pytest.approx(.05)
    assert area_summary([0, 0])['ap_prevalence_baseline'] is None


def test_lse_math_and_zero_boundary():
    for alpha in [.005, .01, .02, .05, .1]:
        result = lse_example(alpha)
        assert result['frame_logit'] < 0
        required = result['foreground_logit_for_frame_zero']
        assert lse_example(alpha, required)['frame_logit'] == pytest.approx(0, abs=1e-12)
    assert lse_example(.005)['foreground_logit_for_frame_zero'] > lse_example(.1)['foreground_logit_for_frame_zero']


def test_best_and_patience_use_different_improvement_rules():
    values = [.1, .101]+[.1]*7
    history = [{'epoch': i+1, 'dev': {'ANAKIN': {'spatial_dice': v}}} for i, v in enumerate(values)]
    result = stopping(history)
    assert result == {'best_epoch': 2, 'stale': 8, 'patience_reached': True}
    with pytest.raises(ValueError, match='HISTORY_AFTER_EARLY_STOP'):
        stopping(history+[history[-1]])


def test_official_gt_audit_numeric_order_padding_and_corruption(tmp_path):
    cache = tmp_path/'cache'/'v'; cache.mkdir(parents=True)
    mapping = {'source_hw': [2, 4], 'canvas_hw': [512, 512], 'content_hw': [256, 512], 'pad_xy': [0, 128]}
    native = [np.zeros((2, 4), bool), np.array([[1, 0, 0, 0], [0, 0, 0, 0]], bool)]
    archive = tmp_path/'masks.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for i in [1, 0]:
            b = io.BytesIO(); Image.fromarray(native[i].astype('uint8')*255).save(b, format='PNG')
            z.writestr(f'{i}.png', b.getvalue())
    np.save(cache/'gt.npy', np.stack([mask_to_canvas(a, mapping) for a in native]))
    np.save(cache/'frame_label.npy', [0., 1.]); np.save(cache/'pixel_known.npy', [True, True])
    row = {'dataset': 'ANAKIN', 'role': 'fake', 'video_id': 'v', 'pair_id': 'an:v',
           'actual_frames': 2, 'mask_zip': str(archive), 'mask_members': [[0, '0.png'], [1, '1.png']],
           'index_base': 0, 'source_hw': [2, 4], 'mapping512': mapping}
    result = audit_masks(tmp_path, [row], tmp_path)['v']
    assert result['area_median'] == .125
    assert result['canvas512']['area_median'] == .125  # Padding excluded.
    bad = np.load(cache/'gt.npy'); bad[1] = False; np.save(cache/'gt.npy', bad)
    with pytest.raises(AssertionError, match='GT_MAPPING_MISMATCH'):
        audit_masks(tmp_path, [row], tmp_path)
