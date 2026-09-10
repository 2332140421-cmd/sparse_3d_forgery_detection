"""CPU checks for the bounded observation-density diagnostic."""

from types import SimpleNamespace

import numpy as np

from research_tools.v7.observation_density_diagnostic.diagnostics import (
    COMPONENT_CONFIG,
    component_diagnostic,
    compare_sequences,
    qualified_graph_edges,
    review_frame_data,
    nested_query_mapping,
    query_grid_coordinates,
    tracking_validity_rows,
)
from research_tools.v7.local_structural_temporal_probe.representation import build_window_support


def _sequence(track_count: int = 4, *, translation: np.ndarray | None = None):
    times = np.arange(20, dtype=np.float64) * 0.05
    base = np.asarray(
        [[0.0, 0.0, 1.0], [0.6, 0.0, 1.0], [0.0, 0.6, 1.0], [4.0, 4.0, 1.0]],
        dtype=np.float32,
    )
    if track_count > len(base):
        extra = np.stack(
            [np.linspace(0.0, 0.6, track_count - len(base)), np.zeros(track_count - len(base)), np.ones(track_count - len(base))],
            axis=1,
        ).astype(np.float32)
        base = np.concatenate((base, extra), axis=0)
    base = base[:track_count]
    xyz = np.asarray([base + (translation if translation is not None else 0.0) for _ in times], dtype=np.float32)
    valid = np.ones((len(times), track_count), dtype=np.bool_)
    uv = np.zeros((len(times), track_count, 2), dtype=np.float32)
    uv[:] = np.asarray([[[10.0 + i * 5.0, 20.0 + i * 4.0] for i in range(track_count)]], dtype=np.float32)
    return SimpleNamespace(
        frame_indices=np.arange(len(times), dtype=np.int64),
        timestamps_s=times,
        frame_sizes_hw=np.full((len(times), 2), [100, 120], dtype=np.int64),
        track_ids=np.arange(track_count, dtype=np.int64),
        xyz=xyz,
        uv=uv,
        visibility=valid.copy(),
        geometry_validity=valid,
    )


def test_nested_grid_maps_all_64_points_without_duplicate_ids():
    mapping = nested_query_mapping()
    assert len(mapping) == 64
    assert len({row["dense_index"] for row in mapping}) == 64
    base = query_grid_coordinates(256, 8)
    dense = query_grid_coordinates(256, 17)
    for row in mapping:
        np.testing.assert_allclose(base[row["base_index"]], dense[row["dense_index"]], atol=1e-6, rtol=0)


def test_qualified_edges_are_invariant_to_common_rigid_transform():
    sequence = _sequence(3)
    reference = qualified_graph_edges(sequence.xyz, sequence.geometry_validity, COMPONENT_CONFIG)
    theta = 0.4
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    transformed = np.einsum("ab,tib->tia", rotation, sequence.xyz) + np.asarray([2, -1, 0.5], dtype=np.float32)
    assert reference == qualified_graph_edges(transformed, sequence.geometry_validity, COMPONENT_CONFIG)


def test_connected_component_reports_internal_pairs_separately_from_direct_edges():
    sequence = _sequence(4)
    # A-B and B-C are within the one-metre rule; A-C is deliberately too far.
    sequence.xyz[:, 0] = [0.0, 0.0, 1.0]
    sequence.xyz[:, 1] = [0.6, 0.0, 1.0]
    sequence.xyz[:, 2] = [1.2, 0.0, 1.0]
    sequence.xyz[:, 3] = [5.0, 5.0, 1.0]
    config = type(COMPONENT_CONFIG)(1.0, 0.05, minimum_size=3, minimum_overlap=8)
    edges = qualified_graph_edges(sequence.xyz, sequence.geometry_validity, config)
    assert (0, 1) in edges and (1, 2) in edges and (0, 2) not in edges
    support = build_window_support(sequence, window_start_s=0.0, component_config=config)
    diagnostic = component_diagnostic(sequence, window_start_s=0.0)
    assert any(set(row) == {0, 1, 2} for row in diagnostic["component_members"])
    assert diagnostic["qualified_graph_edge_count"] == 2
    assert diagnostic["component_internal_pair_count"] == 3
    assert support["valid_triplet_count"] >= 1


def test_missing_common_member_produces_no_valid_triplet_and_no_zero_fill():
    sequence = _sequence(3)
    sequence.geometry_validity[10:, 0] = False
    sequence.xyz[10:, 0] = np.nan
    sequence.uv[10:, 0] = np.nan
    sequence.visibility[10:, 0] = False
    support = build_window_support(sequence, window_start_s=0.0, component_config=COMPONENT_CONFIG)
    assert support["valid_triplet_count"] == 0
    assert any(item["reason"] == "COMMON_VALID_MEMBERS_LT3" for item in support["invalid_reasons"])
    assert np.isnan(sequence.xyz[10:, 0]).all()


def test_review_data_separates_display_edges_and_marks_nested_base_points():
    sequence = _sequence(4)
    diagnostic = component_diagnostic(sequence, window_start_s=0.0)
    review = review_frame_data(sequence, diagnostic["support"], diagnostic, nested_dense=False)
    assert review["graph_edges_total"] >= len(review["graph_edges_display"])
    assert all(len(point) == 8 for point in review["frames"][0]["points"])
    assert all(point[7] for point in review["frames"][0]["points"])
    # A small synthetic 17x17 track table marks only even/even slots as base.
    dense = _sequence(289)
    dense.uv[:] = 10.0
    d_diag = component_diagnostic(dense, window_start_s=0.0)
    d_review = review_frame_data(dense, d_diag["support"], d_diag, nested_dense=True)
    base_flags = [point[7] for point in d_review["frames"][0]["points"]]
    assert any(base_flags) and not all(base_flags)


def test_matched_sequence_comparison_reports_timestamp_and_visibility_changes():
    old = _sequence(3)
    new = _sequence(3)
    new.visibility[5, 1] = False
    new.uv[5, 1] = np.nan
    new.geometry_validity[5, 1] = False
    new.xyz[5, 1] = np.nan
    summary = compare_sequences(old, new)
    assert summary["source_frame_indices_equal"]
    assert summary["visibility_disagreement_count"] == 1
    assert summary["visibility_disagreement_frame_count"] == 1


def test_component_diagnostic_keeps_three_time_ranges_and_unknown_geometry_cause():
    sequence = _sequence(3)
    sequence.geometry_validity[7:, 0] = False
    sequence.xyz[7:, 0] = np.nan
    sequence.uv[7:, 0] = np.nan
    sequence.visibility[7:, 0] = False
    diagnostic = component_diagnostic(sequence, window_start_s=0.0)
    assert "triplet_range_2d_per_frame" in diagnostic
    rows = tracking_validity_rows(sequence, diagnostic["support"], density_label="density64")
    affected = [row for row in rows if row["array_index"] == 7 and row["track_slot"] == 0][0]
    assert affected["depth_invalid"] == "UNKNOWN"
    assert "depth-vs-pose" in affected["unknown_reason"]


def test_review_dense_grid_marks_64_subset_at_odd_coordinates():
    sequence = _sequence(289)
    diagnostic = component_diagnostic(sequence, window_start_s=0.0)
    review = review_frame_data(sequence, diagnostic["support"], diagnostic, nested_dense=True)
    for point in review["frames"][0]["points"]:
        row, col = divmod(point[0], 17)
        assert point[7] == (row % 2 == 1 and col % 2 == 1)
