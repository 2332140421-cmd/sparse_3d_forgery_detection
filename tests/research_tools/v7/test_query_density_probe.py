"""Small contracts for the bounded 04LAX query-density pilot."""

import json

import numpy as np

from research_tools.v7.query_density_probe import pilot
from research_tools.v7.query_density_probe.pilot import nested_axes, query_manifest


def test_midpoint_inserted_grid_is_nested_and_has_expected_counts() -> None:
    original, dense, old_indices = nested_axes()
    assert original.shape == (17,)
    assert dense.shape == (33,)
    assert np.allclose(dense[::2], original, rtol=0.0, atol=0.0)
    assert np.allclose(dense[1::2], (original[:-1] + original[1:]) / 2.0)
    assert np.array_equal(old_indices, np.arange(17) * 2)


def test_manifest_has_289_original_and_800_added_source_coordinates() -> None:
    rows = query_manifest((360, 480))
    assert len(rows) == 1089
    assert sum(row["membership"] == "original" for row in rows) == 289
    assert sum(row["membership"] == "added" for row in rows) == 800
    ids = [row["query_id"] for row in rows]
    assert len(set(ids)) == 1089
    assert all(0 <= row["source_u"] < 480 and 0 <= row["source_v"] < 360 for row in rows)
    assert all(row["original_query_id"] is not None for row in rows if row["membership"] == "original")


def test_roi_envelope_uses_only_confirmed_entries(tmp_path, monkeypatch) -> None:
    roi_dir = tmp_path / "roi_evidence_review_v1"
    roi_dir.mkdir()
    (roi_dir / "roi_annotations.json").write_text(json.dumps({
        "status": "USER_CONFIRMATION_PARTIAL",
        "input": {"entries": [
            {"source_frame_index": 503, "human_confirmation_status": "CONFIRMED"},
            {"source_frame_index": 504, "human_confirmation_status": "SKIP"},
            {"source_frame_index": 505, "human_confirmation_status": "UNCERTAIN"},
        ]},
    }), encoding="utf-8")
    monkeypatch.setattr(pilot, "OLD_CASE", tmp_path)
    assert sorted(pilot._roi_annotations()) == [503]


def test_pair_partition_treats_r17_all_pairs_as_original() -> None:
    manifest = [{"unit_identities": [{"member_slots": [0, 1, 2], "pair_ids": [[0, 1], [0, 2], [1, 2]]}]}]
    stats = pilot._support_density_stats("R17", manifest, set(range(289)))
    assert stats["valid_pair_total"] == 3
    assert stats["valid_pairs_both_original"] == 3
    assert stats["valid_pairs_mixed_original_added"] == 0
    assert stats["valid_pairs_both_added"] == 0
