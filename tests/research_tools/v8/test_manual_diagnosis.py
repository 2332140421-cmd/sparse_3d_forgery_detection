import pytest

from research_tools.v8.full_video_global_context.manual_diagnosis import FIELDS, join_reviews, score_values


def test_join_by_video_identity_and_preserve_notes():
    mapping = [{'review_id': f'V{i:03}', 'video_id': str(i), 'video_sha256': str(i)*64,
                'section': 'test'} for i in (1, 2)]
    records = [{**m, **dict.fromkeys(FIELDS, ''), 'notes': '不是全生成式，\n原文不改'} for m in reversed(mapping)]
    frozen = [{**m, 'role': 'fake', 'generator': 'official'} for m in mapping]
    joined = join_reviews(records, mapping, frozen)
    assert [r['video_id'] for r in joined] == ['1', '2']
    assert joined[0]['notes'] == records[0]['notes']
    assert joined[0]['role'] == 'fake'
    records[0]['video_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='IDENTITY_MISMATCH'):
        join_reviews(records, mapping, frozen)


def test_final_video_logit_mean_and_finite_checks():
    values = {'20260909': -6, '20260910': -3, '20260911': 3, 'MEAN_VIDEO_LOGIT': -2}
    assert score_values(values) == ([-6., -3., 3.], -2.)
    with pytest.raises(ValueError, match='ENSEMBLE_MISMATCH'):
        score_values({**values, 'MEAN_VIDEO_LOGIT': 0})
    with pytest.raises(ValueError, match='NONFINITE'):
        score_values({**values, '20260909': float('nan')})
