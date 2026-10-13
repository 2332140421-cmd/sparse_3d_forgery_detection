from pathlib import Path

from research_tools.v8.full_video_global_context.manual_review import ordered, training_examples


def test_order_ignores_label_score_and_input_order():
    rows = [{'video_id': str(i), 'score': i, 'role': 'fake'} for i in range(40)]
    a = [r['video_id'] for r in ordered(rows)]
    changed = [{**r, 'score': -r['score'], 'role': 'real'} for r in reversed(rows)]
    assert a == [r['video_id'] for r in ordered(changed)]
    assert len(set(a)) == 40


def test_training_first_stable_source_and_exact_real():
    rows = [{'source_id': s, 'video_id': s+role, 'role': role, 'split': 'train',
             'video_path': f'/data/{"02_wan" if role == "fake" else "real"}/{s}.mp4'}
            for s in ('B', 'A') for role in ('fake', 'real')]
    selected, missing = training_examples(rows)
    assert [r['source_id'] for r in selected] == ['A', 'A']
    assert [r['role'] for r in selected] == ['real', 'fake']
    assert missing == ['FCVG: no frozen training fake identity']


def test_offline_page_blind_default_and_no_distortion():
    p = Path(__file__).parents[3]/'research_tools/v8/full_video_global_context/manual_review.html'
    page = p.read_text()
    assert 'id="official" hidden' in page
    assert 'object-fit:contain' in page and 'height:auto' in page
    assert 'fetch(' not in page and ' autoplay' not in page
    assert 'csvEncode' in page and 'csvDecode' in page and 'acceptRecords' in page
    assert "m.video_sha256!==r.video_sha256" in page
    assert 'logit' not in page and 'prediction' not in page
