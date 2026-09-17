"""Research-contract checks for the same-support 2D/3D pilot."""

import numpy as np
import torch

from research_tools.v7.geometry_information_pilot.model import ModalSetModel, parameter_count
from research_tools.v7.geometry_information_pilot.runner import _initial_model, _save_features, _summary_from_distances


def test_modal_parameter_counts_and_initial_fusion_degeneracy() -> None:
    assert parameter_count(ModalSetModel(4)) == 569
    assert parameter_count(ModalSetModel(8)) == 633
    base = _initial_model("UV_2D", 20260909).eval()
    fused = _initial_model("UV_XYZ", 20260909).eval()
    states = torch.randn(5, 5, 4)
    intervals = torch.full((5, 4), 0.1)
    duplicated = torch.cat((states, states), dim=-1)
    with torch.no_grad():
        assert torch.allclose(base(states, intervals), fused(duplicated, intervals), atol=1e-6, rtol=0)


def test_normalized_summary_is_invariant_to_global_distance_scale() -> None:
    distances = np.asarray([[1.0, 1.1, 1.2, 1.3, 1.4], [2.0, 2.1, 2.2, 2.3, 2.4]], dtype=np.float64)
    first = _summary_from_distances(distances / np.median(distances))
    second = _summary_from_distances((17.0 * distances) / np.median(17.0 * distances))
    assert np.allclose(first, second, atol=1e-12, rtol=0)


def test_rigid_transform_preserves_same_frame_3d_pair_distances() -> None:
    points = np.asarray([[0.2, 0.1, 1.0], [0.8, 0.2, 1.4], [0.1, 0.9, 2.0]], dtype=np.float64)
    angle = 0.37
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    transformed = points @ rotation.T + np.asarray([3.0, -2.0, 5.0])
    assert np.allclose(np.linalg.norm(points[None, :, :] - points[:, None, :], axis=-1), np.linalg.norm(transformed[None, :, :] - transformed[:, None, :], axis=-1), atol=1e-12, rtol=0)


def test_fixed_uv_can_hide_relative_depth_change_in_3d() -> None:
    uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    depth_a = np.asarray([1.0, 1.0, 1.0])
    depth_b = np.asarray([1.0, 2.0, 1.0])
    xyz_a = np.column_stack((uv * depth_a[:, None], depth_a))
    xyz_b = np.column_stack((uv * depth_b[:, None], depth_b))
    uv_dist_a = np.linalg.norm(uv[1] - uv[0])
    uv_dist_b = np.linalg.norm(uv[1] - uv[0])
    xyz_dist_a = np.linalg.norm(xyz_a[1] - xyz_a[0])
    xyz_dist_b = np.linalg.norm(xyz_b[1] - xyz_b[0])
    assert uv_dist_a == uv_dist_b
    assert not np.isclose(xyz_dist_a, xyz_dist_b)


def test_eight_dimensional_extra_columns_receive_gradient() -> None:
    torch.manual_seed(3)
    model = ModalSetModel(8)
    states = torch.randn(4, 5, 8)
    intervals = torch.full((4, 4), 0.1)
    loss = model(states, intervals).sum()
    loss.backward()
    gradient = model.encoder[0].weight.grad[:, 4:]
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.any(torch.abs(gradient) > 0)


def test_resume_restores_matched_unit_counts_from_manifest(tmp_path) -> None:
    relative = "features/window_inputs/w0.npz"
    feature_path = tmp_path / relative
    feature_path.parent.mkdir(parents=True)
    np.savez(feature_path, s2=np.zeros((1, 5, 4)), s3=np.zeros((1, 5, 4)), intervals=np.ones(4))
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/feature_manifest.json").write_text(
        '[{"window_id":"w0","input_path":"features/window_inputs/w0.npz","matched_unit_count":7}]',
        encoding="utf-8",
    )
    train, validation, info = _save_features(
        tmp_path,
        [{"window_id": "w0", "label": 0}],
        [],
        [],
        resume=True,
    )
    assert info["reused"] is True
    assert train[0]["matched_unit_count"] == 7
    assert validation == []
