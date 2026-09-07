import numpy as np

from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import (
    RasterScale,
    accumulate_world_from_camera,
    backproject_z_depth,
    centered_intrinsics,
    causal_first_frame_intrinsics,
    quasi_static_noise_proxy,
    persistent_track_ids,
    sample_depth_at_uv,
    world_xyz,
)


def test_raster_mapping_round_trip_preserves_pixel_centers():
    mapping = RasterScale((480, 640), (256, 256))
    uv = np.array([[0, 0], [639, 479], [127.25, 88.5]], dtype=np.float32)
    assert np.allclose(mapping.process_to_source(mapping.source_to_process(uv)), uv, atol=1e-6)


def test_track_identity_is_deterministic_and_persistent():
    assert np.array_equal(persistent_track_ids(4), np.array([0, 1, 2, 3], dtype=np.int64))


def test_backprojection_uses_z_depth_and_centered_intrinsics():
    k = centered_intrinsics(3, 5, 2.0)
    xyz = backproject_z_depth(np.array([[2.0, 1.0], [4.0, 1.0]]), np.array([3.0, 3.0]), k)
    assert np.allclose(xyz, [[0, 0, 3], [3, 0, 3]])


def test_intrinsics_use_only_the_first_focal_estimate_in_a_causal_window():
    depths = np.ones((2, 3, 5), dtype=np.float32)
    focal_px = np.array([2.0, 8.0])
    original = focal_px.copy()
    intrinsics, fixed_focal = causal_first_frame_intrinsics(depths, focal_px)
    assert fixed_focal == 2.0
    assert np.allclose(intrinsics[:, 0, 0], [2.0, 2.0])
    assert np.array_equal(focal_px, original)


def test_open3d_source_to_target_convention_accumulates_inverse():
    target_from_source = np.eye(4)[None]
    target_from_source[0, 0, 3] = -2.0
    world_from_camera, valid = accumulate_world_from_camera(target_from_source, np.array([True]))
    assert valid.tolist() == [True, True]
    assert np.allclose(world_from_camera[1, :3, 3], [2, 0, 0])


def test_accumulation_does_not_overwrite_emitted_input_history():
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    original = transforms.copy()
    accumulate_world_from_camera(transforms, np.array([True, True]))
    assert np.array_equal(transforms, original)


def test_pose_failure_propagates_without_identity_fallback():
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    poses, valid = accumulate_world_from_camera(transforms, np.array([False, True]))
    assert valid.tolist() == [True, False, False]
    assert np.isnan(poses[1:]).all()


def test_invalid_depth_and_pose_produce_nan_not_zero():
    depth = np.array([[[2.0, np.nan]], [[2.0, 2.0]]], dtype=np.float32)
    uv = np.array([[[0, 0], [1, 0]], [[0, 0], [1, 0]]], dtype=np.float32)
    sampled, depth_valid = sample_depth_at_uv(depth, uv)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    xyz, geometry_valid = world_xyz(
        uv,
        sampled,
        np.repeat(centered_intrinsics(1, 2, 1.0)[None], 2, axis=0),
        poses,
        depth_valid,
        np.array([True, False]),
    )
    assert geometry_valid.tolist() == [[True, False], [False, False]]
    assert np.allclose(xyz[0, 0], [-1, 0, 2])
    assert np.isnan(xyz[~geometry_valid]).all()
    assert not np.any(xyz[~geometry_valid] == 0)


def test_world_coordinate_accumulation_transforms_camera_points():
    uv = np.array([[[0, 0]], [[0, 0]]], dtype=np.float32)
    depth = np.ones((2, 1), dtype=np.float32)
    k = np.repeat(np.eye(3)[None], 2, axis=0)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 2
    xyz, valid = world_xyz(uv, depth, k, poses, np.ones((2, 1), bool), np.ones(2, bool))
    assert valid.all()
    assert np.allclose(xyz[:, 0], [[0, 0, 1], [2, 0, 1]])


def test_noise_proxy_keeps_its_nonsemantic_quartile_interpretation_explicit():
    proxy = quasi_static_noise_proxy(np.array([0.01, 0.02, 0.10, 0.20]))
    assert proxy["kind"] == "LOWER_VS_UPPER_STEP_QUARTILE_PROXY_NOT_STATIC_GROUND_TRUTH"
    assert proxy["quasi_static_lower_quartile"]["median"] == 0.01
    assert proxy["moving_upper_quartile"]["median"] == 0.2
    assert proxy["jitter_to_motion_median_ratio"] < 1
