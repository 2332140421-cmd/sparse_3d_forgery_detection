"""Explicit image-coordinate and camera-coordinate transformations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ResizePadTransform:
    """Mapping between one source raster and a square provider raster."""

    source_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    provider_hw: tuple[int, int]

    def source_to_provider(self, uv: np.ndarray) -> np.ndarray:
        """Map source pixel centers to provider pixel centers."""

        source_h, source_w = self.source_hw
        resized_h, resized_w = self.resized_hw
        left, top, _, _ = self.padding_ltrb
        result = np.asarray(uv, dtype=np.float32).copy()
        result[..., 0] = (result[..., 0] + 0.5) * (resized_w / source_w) - 0.5 + left
        result[..., 1] = (result[..., 1] + 0.5) * (resized_h / source_h) - 0.5 + top
        return result

    def provider_to_source(self, uv: np.ndarray) -> np.ndarray:
        """Map provider pixel centers back to source pixel centers."""

        source_h, source_w = self.source_hw
        resized_h, resized_w = self.resized_hw
        left, top, _, _ = self.padding_ltrb
        result = np.asarray(uv, dtype=np.float32).copy()
        result[..., 0] = (result[..., 0] - left + 0.5) * (source_w / resized_w) - 0.5
        result[..., 1] = (result[..., 1] - top + 0.5) * (source_h / resized_h) - 0.5
        return result


def resize_pad_transform(source_hw: tuple[int, int], target_size: int = 518) -> ResizePadTransform:
    """Match VGGT's aspect-preserving resize and centered square padding."""

    height, width = source_hw
    if height <= 0 or width <= 0 or target_size <= 0 or target_size % 14:
        raise ValueError("raster dimensions must be positive and target_size divisible by 14")
    if width >= height:
        resized_w = target_size
        resized_h = max(14, round(height * target_size / width / 14) * 14)
    else:
        resized_h = target_size
        resized_w = max(14, round(width * target_size / height / 14) * 14)
    left = (target_size - resized_w) // 2
    top = (target_size - resized_h) // 2
    return ResizePadTransform(
        source_hw=source_hw,
        resized_hw=(resized_h, resized_w),
        padding_ltrb=(left, top, target_size - resized_w - left, target_size - resized_h - top),
        provider_hw=(target_size, target_size),
    )


def unproject_z_depth_to_world(
    uv: np.ndarray,
    z_depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    """Lift OpenCV pixels with z-depth and invert world-to-camera poses."""

    if uv.ndim != 3 or uv.shape[-1] != 2 or z_depth.shape != uv.shape[:2]:
        raise ValueError("uv and z_depth shapes must be [T,N,2] and [T,N]")
    frames = uv.shape[0]
    if intrinsics.shape != (frames, 3, 3) or world_to_camera.shape != (frames, 3, 4):
        raise ValueError("camera shapes must be [T,3,3] and [T,3,4]")

    fx = intrinsics[:, 0, 0, None]
    fy = intrinsics[:, 1, 1, None]
    cx = intrinsics[:, 0, 2, None]
    cy = intrinsics[:, 1, 2, None]
    camera = np.stack(
        ((uv[..., 0] - cx) * z_depth / fx, (uv[..., 1] - cy) * z_depth / fy, z_depth),
        axis=-1,
    )
    rotation = world_to_camera[:, :3, :3]
    translation = world_to_camera[:, :3, 3]
    return np.einsum("tji,tnj->tni", rotation, camera - translation[:, None, :]).astype(
        np.float32
    )
