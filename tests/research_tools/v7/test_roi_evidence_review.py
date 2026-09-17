import numpy as np

from research_tools.v7.local_observation_recovery_probe.probe import condition_frame_status
from research_tools.v7.local_observation_recovery_probe.roi_evidence import canonical_pairs, representation_audit
from research_tools.v7.local_observation_recovery_probe.roi_evidence_phase2 import frame_structure, inside_rect


def test_r_is_not_queried_before_query_start():
    assert condition_frame_status("R", 487, query_start_frame=488) == "NOT_QUERIED"
    assert condition_frame_status("R", 488, query_start_frame=488) == "AVAILABLE"


def test_canonical_pairs_removes_orientation_and_duplicates():
    assert canonical_pairs([[2, 1], [1, 2], [3, 3], [4, 0]]) == [[0, 4], [1, 2]]


def test_saved_triplet_derivatives_are_checked_without_changing_state():
    states = np.asarray([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2]])
    from research_tools.v7.local_structural_temporal_probe.representation import compute_local_derivatives
    center, first, second = compute_local_derivatives(states, [0.0, 0.1, 0.2])
    item = {
        "status": "VALID",
        "local_group_id": 0,
        "target_slots": [0, 1, 2],
        "states": states.tolist(),
        "timestamps_s": [0.0, 0.1, 0.2],
        "pair_ids": [[2, 1]],
        "members_considered": [1, 2, 3],
        "center_state": center.tolist(),
        "first_derivative": first.tolist(),
        "second_derivative": second.tolist(),
    }
    audit = representation_audit({"support": {"triplets": [item], "target_matches": []}})
    assert audit["finite_triplets_checked"] == 1
    assert audit["noncanonical_pair_triplet_count"] == 1
    assert max(audit["derivative_recompute_max_abs"].values()) == 0.0


def test_phase2_uses_finite_roi_uv_and_integer_saved_group_owners():
    uv = np.asarray([[10.0, 10.0], [np.nan, 10.0], [20.0, 20.0]], dtype=np.float32)
    inside = inside_rect(uv, [0, 0, 20, 20])
    assert inside.tolist() == [True, False, True]
    source = {
        "groups": [{"local_group_id": 0, "member_slots": [0, 1], "retained": True}],
        "triplets": [{"frame_indices": [5, 6, 7], "pair_ids": [[1, 0]], "owner_group_ids": [0], "status": "VALID"}],
    }
    structure = frame_structure("R", 6, source)
    assert structure["status"] == "VALID_R_H_TARGET_SUPPORT"
    assert structure["owner_group_ids"] == [0]
    assert structure["pairs"] == [[0, 1]]
