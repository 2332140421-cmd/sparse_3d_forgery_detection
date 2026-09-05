import numpy as np

from sparse3d_forgery.frontend.geometry import (
    resize_pad_transform,
    unproject_z_depth_to_world,
)


def test_resize_pad_uv_mapping_round_trips_pixel_centers():
    transform = resize_pad_transform((240, 320), 518)
    uv = np.array([[0.0, 0.0], [319.0, 239.0], [121.25, 87.5]], dtype=np.float32)

    recovered = transform.provider_to_source(transform.source_to_provider(uv))

    np.testing.assert_allclose(recovered, uv, atol=1e-5)


def test_unproject_uses_z_depth_and_inverts_world_to_camera_pose():
    uv = np.array([[[2.0, 1.0], [4.0, 3.0]]], dtype=np.float32)
    depth = np.array([[2.0, 4.0]], dtype=np.float32)
    intrinsics = np.array([[[2.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    world_to_camera = np.array(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )

    world = unproject_z_depth_to_world(uv, depth, intrinsics, world_to_camera)

    expected = np.array([[[-1.0, 0.0, 2.0], [3.0, 4.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(world, expected)
