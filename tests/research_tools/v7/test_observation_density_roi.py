from __future__ import annotations

from research_tools.v7.observation_density_diagnostic.roi import (
    NOT_MODEL_SAMPLE,
    VALID,
    compare_density_details,
    coverage_for_roi,
    display_to_source,
    point_in_rect,
    source_to_display,
    validate_annotation,
)


def _detail(density: str, slots: list[tuple[int, float, float, bool, bool, int]]) -> dict:
    return {
        "window_id": "w::real",
        "density": density,
        "frames": [{
            "source_frame_index": 10,
            "timestamp_s": 1.0,
            "height": 100,
            "width": 120,
            "points": [[slot, u, v, visible, geometry, component, True, density == "density64"] for slot, u, v, visible, geometry, component in slots],
        }],
        "triplets": [
            {"triplet_id": 2, "component_index": 9, "source_frame_indices": [9, 10, 11], "common_member_indices": [0, 1, 2], "pair_indices": [[0, 1], [0, 2], [1, 2]]},
            {"triplet_id": 5, "component_index": 4, "source_frame_indices": [8, 10, 12], "common_member_indices": [0, 1, 3], "pair_indices": [[0, 1], [0, 3], [1, 3]]},
        ],
    }


def _annotation() -> dict:
    return {
        "source_id": "S01",
        "window_id": "w::real",
        "role": "real",
        "source_frame_index": 10,
        "timestamp_s": 1.0,
        "frame_size_hw": [100, 120],
        "rect": {"x": 0, "y": 0, "w": 50, "h": 50},
        "observation": "internal_deformation",
        "tracking_observation": "uncertain",
    }


def test_display_source_round_trip_uses_original_pixels():
    source = display_to_source(37.5, 25.0, 300.0, 200.0, 120, 100)
    display = source_to_display(*source, 120, 100, 300.0, 200.0)
    assert display == (37.5, 25.0)


def test_roi_boundary_is_inclusive():
    rect = {"x": 10, "y": 20, "w": 5, "h": 7}
    assert point_in_rect(10, 20, rect)
    assert point_in_rect(15, 27, rect)
    assert not point_in_rect(15.01, 27, rect)


def test_coverage_deduplicates_pairs_across_triplets_and_keeps_components():
    detail = _detail("density64", [(0, 10, 10, True, True, 1), (1, 20, 20, True, True, 1), (2, 80, 80, True, True, 1), (3, 30, 30, True, True, 2)])
    frame, component_rows = coverage_for_roi(detail, _annotation())
    assert frame["valid_triplet_common_points"] == 3
    assert frame["internal_pairs_both_endpoints"] == 3  # (0,1) is shared and counted once.
    assert frame["cross_roi_pairs_one_endpoint"] == 2
    assert frame["component_ids"] == [4, 9]
    assert frame["triplet_ids"] == [2, 5]
    assert len(component_rows) == 2


def test_density_comparison_requires_one_frame_and_allows_different_component_ids():
    dense = _detail("density289", [(10, 10, 10, True, True, 11), (11, 20, 20, True, True, 11), (12, 30, 30, True, True, 12), (13, 40, 40, True, True, 12)])
    dense["triplets"][0]["common_member_indices"] = [10, 11, 12]
    dense["triplets"][0]["pair_indices"] = [[10, 11], [10, 12], [11, 12]]
    dense["triplets"][1]["common_member_indices"] = [10, 11, 13]
    dense["triplets"][1]["pair_indices"] = [[10, 11], [10, 13], [11, 13]]
    comparison, rows, _ = compare_density_details(_detail("density64", [(0, 10, 10, True, True, 1), (1, 20, 20, True, True, 1), (2, 80, 80, True, True, 1), (3, 30, 30, True, True, 2)]), dense, {**_annotation(), "rect": {"x": 0, "y": 0, "w": 50, "h": 50}})
    assert comparison["same_source_frame"] is True
    assert comparison["timestamp_abs_difference_s"] == 0.0
    assert rows[0]["source_frame_index"] == rows[1]["source_frame_index"] == 10
    assert comparison["density289_valid_triplet_common_points"] == 4
    assert comparison["support_status"] == "LOCAL_SUPPORT_INCREASED"


def test_non_model_frame_is_not_reported_as_zero_support():
    detail = _detail("density64", [(0, 10, 10, True, True, 1)])
    detail["triplets"] = []
    frame, _ = coverage_for_roi(detail, _annotation())
    assert frame["model_sample_status"] == NOT_MODEL_SAMPLE
    assert frame["valid_triplet_common_points"] == 0


def test_annotation_validation_checks_source_frame_pts_and_role():
    detail = _detail("density64", [(0, 10, 10, True, True, 1)])
    manifest = {"w::real": {"source_id": "S01", "role": "real", "kind": "MANIP"}}
    valid = validate_annotation(_annotation(), {"w::real": detail}, manifest)
    assert valid["status"] == VALID
    wrong = validate_annotation({**_annotation(), "timestamp_s": 2.0}, {"w::real": detail}, manifest)
    assert wrong["status"] == "UNRESOLVED"
    outside = validate_annotation({**_annotation(), "window_id": "other"}, {"w::real": detail}, manifest)
    assert outside["status"] == "OUTSIDE_DENSITY_POPULATION"
