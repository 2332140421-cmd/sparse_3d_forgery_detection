import numpy as np

from research_tools.v7.dense_local_structure_pilot.grouping import analysis_grid, assign_groups, fixed_local_edges
from research_tools.v7.dense_local_structure_pilot.representation import fixed_edge_triplet


def test_analysis_grid_is_fixed_three_pixel_centres():
    grid = analysis_grid()
    assert grid.shape == (128 * 128, 2)
    assert np.allclose(grid[:3, 0], [1.5, 4.5, 7.5])
    assert np.allclose(grid[:3, 1], [1.5, 1.5, 1.5])
    assert np.allclose(grid[-1], [382.5, 382.5])


def test_background_groups_are_per_block_and_edges_are_local():
    uv = np.asarray([[1.5, 1.5], [4.5, 1.5], [25.5, 1.5], [200.5, 200.5]], dtype=np.float32)
    groups, summary = assign_groups(uv, None)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    edges = fixed_local_edges(uv, groups, max_neighbors=8)
    assert all(edge["group_id"] == groups[edge["left_query_id"]] for edge in edges)
    assert not any({edge["left_query_id"], edge["right_query_id"]} == {0, 2} for edge in edges)


def test_mask_overlap_uses_small_area_then_index():
    uv = np.asarray([[10.5, 10.5]], dtype=np.float32)
    masks = np.zeros((2, 384, 384), dtype=bool)
    masks[0, :30, :30] = True
    masks[1, :20, :20] = True
    groups, _ = assign_groups(uv, masks)
    assert groups[0].startswith("instance_001_")


def test_fixed_triplet_requires_common_edges_and_uses_history_scale():
    xyz = np.asarray(
        [
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1.1, 0, 1], [0, 1, 1]],
        ],
        dtype=np.float64,
    )
    edges = [
        {"edge_id": 0, "left_query_id": 0, "right_query_id": 1},
        {"edge_id": 1, "left_query_id": 0, "right_query_id": 2},
        {"edge_id": 2, "left_query_id": 1, "right_query_id": 2},
    ]
    triplet = fixed_edge_triplet(xyz, np.asarray([0.0, 0.1, 0.2]), edges, [0, 1], [0, 1, 2])
    assert triplet is not None
    assert triplet["edge_ids"] == [0, 1, 2]
    assert triplet["states"].shape == (3, 4)
