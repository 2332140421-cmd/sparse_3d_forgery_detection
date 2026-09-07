"""Minimal explicit-geometry helpers for the bounded V7 runtime probe."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RasterScale:
    """Pixel-center mapping between an original raster and a resized raster."""

    source_hw: tuple[int, int]
    process_hw: tuple[int, int]

    def source_to_process(self, uv: np.ndarray) -> np.ndarray:
        source_h, source_w = self.source_hw
        process_h, process_w = self.process_hw
        result = np.asarray(uv, dtype=np.float32).copy()
        result[..., 0] = (result[..., 0] + 0.5) * process_w / source_w - 0.5
        result[..., 1] = (result[..., 1] + 0.5) * process_h / source_h - 0.5
        return result

    def process_to_source(self, uv: np.ndarray) -> np.ndarray:
        source_h, source_w = self.source_hw
        process_h, process_w = self.process_hw
        result = np.asarray(uv, dtype=np.float32).copy()
        result[..., 0] = (result[..., 0] + 0.5) * source_w / process_w - 0.5
        result[..., 1] = (result[..., 1] + 0.5) * source_h / process_h - 0.5
        return result


def persistent_track_ids(count: int) -> np.ndarray:
    """Assign deterministic correspondence identities without semantic meaning."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("track count must be a positive integer")
    return np.arange(count, dtype=np.int64)


def centered_intrinsics(height: int, width: int, focal_px: float) -> np.ndarray:
    """Create the explicitly experimental centered-principal-point K matrix."""

    if height <= 0 or width <= 0 or not np.isfinite(focal_px) or focal_px <= 0:
        raise ValueError("height, width, and focal length must be positive")
    return np.asarray(
        [
            [focal_px, 0.0, (width - 1) / 2],
            [0.0, focal_px, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def causal_first_frame_intrinsics(
    depths: np.ndarray, focal_px: np.ndarray
) -> tuple[np.ndarray, float]:
    """Use the first-frame focal estimate for a causal fixed-intrinsics window.

    A decoded video window has one physical camera calibration unless its source
    explicitly reports a calibration change. Depth Pro infers focal length per
    image, so applying later estimates to an already-emitted history would make
    the geometric input depend on later observations. This bounded probe uses
    only the first estimate and retains all raw estimates for audit.
    """

    depths = np.asarray(depths)
    focal_px = np.asarray(focal_px)
    if depths.ndim != 3 or focal_px.shape != (depths.shape[0],):
        raise ValueError("expected depth [T,H,W] and focal length [T]")
    if depths.shape[0] == 0 or not np.isfinite(focal_px[0]) or focal_px[0] <= 0:
        raise ValueError("the first-frame focal length must be finite and positive")
    focal = float(focal_px[0])
    intrinsics = np.stack(
        [centered_intrinsics(*depth.shape, focal) for depth in depths]
    )
    return intrinsics, focal


def backproject_z_depth(uv: np.ndarray, z_depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Back-project OpenCV pixel coordinates with optical-axis depth."""

    uv = np.asarray(uv)
    z_depth = np.asarray(z_depth)
    k = np.asarray(k)
    if uv.shape[-1:] != (2,) or z_depth.shape != uv.shape[:-1] or k.shape != (3, 3):
        raise ValueError("expected uv [...,2], matching depth [...], and K [3,3]")
    homogeneous = np.concatenate(
        (uv.astype(np.float64), np.ones((*uv.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    rays = np.einsum("ij,...j->...i", np.linalg.inv(k), homogeneous)
    return (rays * z_depth[..., None]).astype(np.float32)


def accumulate_world_from_camera(target_from_source: np.ndarray, success: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate adjacent source-camera to target-camera transforms.

    Open3D legacy RGB-D odometry returns ``target_from_source[t-1]`` for the
    pair ``(t-1, t)``. World is camera zero. Once a link fails, later cameras
    have no valid relation to that world and remain invalid rather than being
    silently replaced by identity.
    """

    transforms = np.asarray(target_from_source, dtype=np.float64)
    success = np.asarray(success)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("target_from_source must have shape [T-1,4,4]")
    if success.dtype != np.bool_ or success.shape != (transforms.shape[0],):
        raise ValueError("success must be bool [T-1]")
    count = transforms.shape[0] + 1
    world_from_camera = np.full((count, 4, 4), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=np.bool_)
    world_from_camera[0] = np.eye(4)
    valid[0] = True
    for t in range(1, count):
        if not valid[t - 1] or not success[t - 1] or not np.all(np.isfinite(transforms[t - 1])):
            continue
        try:
            source_from_target = np.linalg.inv(transforms[t - 1])
        except np.linalg.LinAlgError:
            continue
        world_from_camera[t] = world_from_camera[t - 1] @ source_from_target
        valid[t] = True
    return world_from_camera, valid


def sample_depth_at_uv(depth: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel depth sampling without repairing out-of-bounds observations."""

    depth = np.asarray(depth)
    uv = np.asarray(uv)
    if depth.ndim != 3 or uv.ndim != 3 or uv.shape[0] != depth.shape[0] or uv.shape[2] != 2:
        raise ValueError("expected depth [T,H,W] and uv [T,N,2]")
    count_t, height, width = depth.shape
    values = np.full(uv.shape[:2], np.nan, dtype=np.float32)
    valid = np.zeros(uv.shape[:2], dtype=np.bool_)
    finite = np.all(np.isfinite(uv), axis=-1)
    rounded = np.zeros_like(uv, dtype=np.int64)
    rounded[finite] = np.rint(uv[finite]).astype(np.int64)
    inside = finite & (rounded[..., 0] >= 0) & (rounded[..., 0] < width) & (rounded[..., 1] >= 0) & (rounded[..., 1] < height)
    for t in range(count_t):
        ids = np.flatnonzero(inside[t])
        values[t, ids] = depth[t, rounded[t, ids, 1], rounded[t, ids, 0]]
    valid = inside & np.isfinite(values) & (values > 0)
    values[~valid] = np.nan
    return values, valid


def world_xyz(
    uv: np.ndarray,
    z_depth: np.ndarray,
    intrinsics: np.ndarray,
    world_from_camera: np.ndarray,
    observation_valid: np.ndarray,
    pose_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project observations and propagate all validity without zero filling."""

    uv = np.asarray(uv)
    z_depth = np.asarray(z_depth)
    intrinsics = np.asarray(intrinsics)
    world_from_camera = np.asarray(world_from_camera)
    observation_valid = np.asarray(observation_valid)
    pose_valid = np.asarray(pose_valid)
    count_t, count_n = z_depth.shape
    if uv.shape != (count_t, count_n, 2) or intrinsics.shape != (count_t, 3, 3):
        raise ValueError("incompatible observation or intrinsics shape")
    if world_from_camera.shape != (count_t, 4, 4) or pose_valid.shape != (count_t,):
        raise ValueError("incompatible pose shape")
    if observation_valid.dtype != np.bool_ or observation_valid.shape != (count_t, count_n):
        raise ValueError("observation_valid must be bool [T,N]")
    geometry_valid = observation_valid & pose_valid[:, None]
    xyz = np.full((count_t, count_n, 3), np.nan, dtype=np.float32)
    for t in range(count_t):
        ids = np.flatnonzero(geometry_valid[t])
        if ids.size == 0:
            continue
        camera_xyz = backproject_z_depth(uv[t, ids], z_depth[t, ids], intrinsics[t])
        homogeneous = np.concatenate(
            (camera_xyz.astype(np.float64), np.ones((ids.size, 1), dtype=np.float64)), axis=1
        )
        transformed = (world_from_camera[t] @ homogeneous.T).T[:, :3]
        finite = np.all(np.isfinite(transformed), axis=1)
        xyz[t, ids[finite]] = transformed[finite].astype(np.float32)
        geometry_valid[t, ids[~finite]] = False
    return xyz, geometry_valid


def summarize_distribution(values: np.ndarray) -> dict[str, float | int]:
    """Return compact finite-value diagnostics."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }

def quasi_static_noise_proxy(step_distances: np.ndarray) -> dict[str, object]:
    """Compare low- and high-step geometry subsets without semantic labels.

    This is a runtime stability proxy, not an independently identified static
    region and not an authenticity feature.
    """

    values = np.asarray(step_distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return {"kind": "UNAVAILABLE_NO_VALID_GEOMETRIC_STEPS", "count": 0}
    low_cutoff, high_cutoff = np.percentile(values, [25, 75])
    quasi_static = values[values <= low_cutoff]
    moving = values[values >= high_cutoff]
    static_summary = summarize_distribution(quasi_static)
    moving_summary = summarize_distribution(moving)
    static_median = float(static_summary["median"])
    moving_median = float(moving_summary["median"])
    return {
        "kind": "LOWER_VS_UPPER_STEP_QUARTILE_PROXY_NOT_STATIC_GROUND_TRUTH",
        "quasi_static_lower_quartile": static_summary,
        "moving_upper_quartile": moving_summary,
        "jitter_to_motion_median_ratio": static_median / moving_median if moving_median > 0 else None,
    }
