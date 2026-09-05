"""Minimal VGGT adapter producing canonical, camera-compensated particles."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image

from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    ParticleSequence,
    build_particle_sequence,
)
from sparse3d_forgery.video_input import DecodedVideoSample

from .geometry import resize_pad_transform, unproject_z_depth_to_world


VGGT_CODE_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_WEIGHT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_WEIGHT_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"


@dataclass(frozen=True, slots=True)
class VggtFrontendConfig:
    """Explicit settings for one replaceable VGGT feasibility frontend."""

    num_tracks: int = 128
    target_size: int = 518
    visibility_threshold: float = 0.2
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.num_tracks <= 0:
            raise ValueError("num_tracks must be positive")
        if self.target_size <= 0 or self.target_size % 14:
            raise ValueError("target_size must be positive and divisible by 14")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError("visibility_threshold must be in [0, 1]")


def _query_grid(height: int, width: int, count: int) -> np.ndarray:
    """Create a deterministic image-wide grid without semantic selection."""

    columns = max(1, int(np.ceil(np.sqrt(count * width / height))))
    rows = max(1, int(np.ceil(count / columns)))
    u = np.linspace(0.05 * (width - 1), 0.95 * (width - 1), columns, dtype=np.float32)
    v = np.linspace(0.05 * (height - 1), 0.95 * (height - 1), rows, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return np.stack((uu.ravel(), vv.ravel()), axis=-1)[:count]


class VggtFrontend:
    """Run official VGGT heads and convert their outputs to ParticleSequence."""

    def __init__(self, weight_path: str | Path, config: VggtFrontendConfig | None = None):
        import torch
        from vggt.models.vggt import VGGT

        self.config = config or VggtFrontendConfig()
        self.weight_path = Path(weight_path)
        if not self.weight_path.is_file():
            raise FileNotFoundError(self.weight_path)
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the selected frontend configuration")

        self._torch = torch
        self._model = VGGT()
        state = torch.load(self.weight_path, map_location="cpu", weights_only=True, mmap=True)
        self._model.load_state_dict(state)
        self._model.eval().to(self.config.device)

    def _prepare_images(self, decoded: DecodedVideoSample):
        torch = self._torch
        tensors = []
        transforms = []
        for frame in decoded.frames:
            transform = resize_pad_transform(frame.rgb.shape[:2], self.config.target_size)
            resized_h, resized_w = transform.resized_hw
            left, top, _, _ = transform.padding_ltrb
            source = Image.fromarray(frame.rgb, mode="RGB")
            resized = source.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (self.config.target_size, self.config.target_size), (255, 255, 255))
            canvas.paste(resized, (left, top))
            array = np.asarray(canvas, dtype=np.uint8).copy()
            tensors.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0))
            transforms.append(transform)
        return torch.stack(tensors).unsqueeze(0).to(self.config.device), transforms

    def extract(self, decoded: DecodedVideoSample) -> ParticleSequence:
        """Extract one joint-sequence feasibility artifact; output is not causal-training eligible."""

        torch = self._torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images, transforms = self._prepare_images(decoded)
        source_h, source_w = decoded.frames[0].rgb.shape[:2]
        source_query = _query_grid(source_h, source_w, self.config.num_tracks)
        provider_query = transforms[0].source_to_provider(source_query)
        query = torch.from_numpy(provider_query).unsqueeze(0).to(self.config.device)

        if self.config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.config.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(
            enabled=self.config.device.startswith("cuda"), dtype=torch.bfloat16
        ):
            tokens, patch_start = self._model.aggregator(images)
            with torch.cuda.amp.autocast(enabled=False):
                pose_encoding = self._model.camera_head(tokens)[-1]
                depth, _ = self._model.depth_head(tokens, images=images, patch_start_idx=patch_start)
            tracks, visibility_scores, _ = self._model.track_head(
                tokens, images=images, patch_start_idx=patch_start, query_points=query
            )
        elapsed_s = time.perf_counter() - started
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding, image_size_hw=images.shape[-2:]
        )

        provider_uv = tracks[-1][0].float().cpu().numpy().astype(np.float32)
        visibility_score = visibility_scores[0].float().cpu().numpy()
        depth_maps = depth[0, ..., 0].float()
        grid = torch.from_numpy(provider_uv).to(depth_maps.device)
        grid[..., 0] = grid[..., 0].mul(2.0 / (depth_maps.shape[2] - 1)).sub(1.0)
        grid[..., 1] = grid[..., 1].mul(2.0 / (depth_maps.shape[1] - 1)).sub(1.0)
        sampled_depth = torch.nn.functional.grid_sample(
            depth_maps.unsqueeze(1), grid.unsqueeze(2), mode="bilinear", align_corners=True
        )[:, 0, :, 0].float().cpu().numpy()
        intrinsics_np = intrinsics[0].float().cpu().numpy()
        extrinsics_np = extrinsics[0].float().cpu().numpy()

        uv = np.stack(
            [transform.provider_to_source(provider_uv[index]) for index, transform in enumerate(transforms)]
        ).astype(np.float32)
        finite_uv = np.isfinite(uv).all(axis=-1)
        frame_sizes = decoded.frame_sizes_hw
        in_bounds = (
            (uv[..., 0] >= 0)
            & (uv[..., 0] < frame_sizes[:, 1, None])
            & (uv[..., 1] >= 0)
            & (uv[..., 1] < frame_sizes[:, 0, None])
        )
        visibility = (visibility_score >= self.config.visibility_threshold) & finite_uv & in_bounds
        camera_valid = (
            np.isfinite(intrinsics_np).all(axis=(1, 2))
            & np.isfinite(extrinsics_np).all(axis=(1, 2))
            & (intrinsics_np[:, 0, 0] > 0)
            & (intrinsics_np[:, 1, 1] > 0)
        )
        depth_valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
        xyz = unproject_z_depth_to_world(provider_uv, sampled_depth, intrinsics_np, extrinsics_np)
        geometry_validity = visibility & depth_valid & camera_valid[:, None] & np.isfinite(xyz).all(axis=-1)
        uv[~visibility] = np.nan
        xyz[~geometry_validity] = np.nan

        transforms_record = [
            {
                "source_hw": list(transform.source_hw),
                "resized_hw": list(transform.resized_hw),
                "padding_ltrb": list(transform.padding_ltrb),
                "provider_hw": list(transform.provider_hw),
            }
            for transform in transforms
        ]
        peak_bytes = (
            int(torch.cuda.max_memory_allocated(self.config.device))
            if self.config.device.startswith("cuda")
            else 0
        )
        sequence = build_particle_sequence(
            decoded,
            track_ids=np.arange(self.config.num_tracks, dtype=np.int64),
            xyz=xyz,
            uv=uv,
            visibility=visibility.astype(np.bool_),
            geometry_validity=geometry_validity.astype(np.bool_),
            coordinate_system=CoordinateSystem(
                frame_name="vggt_world_first_view_gauge",
                handedness=Handedness.RIGHT,
                axis_directions=(
                    "first-camera-right",
                    "first-camera-down",
                    "first-camera-forward",
                ),
                length_unit=LengthUnit.ARBITRARY_SCALE,
                camera_motion_compensated=True,
                normalization={"applied": False},
            ),
            lineage={
                "provider": "facebookresearch/vggt",
                "code_revision": VGGT_CODE_REVISION,
                "weight_revision": VGGT_WEIGHT_REVISION,
                "weight_sha256": VGGT_WEIGHT_SHA256,
                "frame_indices": decoded.frame_indices.tolist(),
                "resize_and_padding": transforms_record,
                "query_initialization": "deterministic uniform grid in first source frame",
            },
            provenance={
                "depth_semantics": "camera z-depth in VGGT arbitrary scale",
                "pose_semantics": "OpenCV world-to-camera extrinsics",
                "visibility_threshold": self.config.visibility_threshold,
                "num_tracks": self.config.num_tracks,
                "target_size": self.config.target_size,
                "causal_training_eligible": False,
                "causal_limitation": "VGGT jointly aggregates all supplied frames",
                "elapsed_s": elapsed_s,
                "peak_gpu_memory_bytes": peak_bytes,
                "torch_version": torch.__version__,
            },
        )
        del images, tokens, pose_encoding, depth, tracks, visibility_scores
        if self.config.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return sequence
