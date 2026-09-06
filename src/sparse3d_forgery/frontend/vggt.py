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
    validate_particle_sequence,
)
from sparse3d_forgery.video_input import DecodedVideoSample

from .alignment import SimilarityAlignmentError, fit_similarity_transform
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

    def extract_causal_window(
        self, decoded: DecodedVideoSample, history_count: int
    ) -> ParticleSequence:
        """Anchor extension-only future rows to a history-only VGGT gauge."""

        if isinstance(history_count, bool) or not isinstance(history_count, int):
            raise ValueError("history_count must be an integer")
        if history_count <= 0 or history_count >= len(decoded.frames):
            raise ValueError("history_count must leave non-empty history and future rows")
        history_decoded = DecodedVideoSample(
            sample_id=decoded.sample_id,
            source_video_id=decoded.source_video_id,
            frames=decoded.frames[:history_count],
        )
        history = self.extract(history_decoded)
        extension = self.extract(decoded)
        return _construct_history_anchored_window(history, extension, history_count)


def _construct_history_anchored_window(
    history: ParticleSequence,
    extension: ParticleSequence,
    history_count: int,
) -> ParticleSequence:
    """Combine Pass A history and Pass B future using history-only Sim(3)."""

    validate_particle_sequence(history)
    validate_particle_sequence(extension)
    if history_count != history.num_frames or history_count <= 0:
        raise ValueError("history_count must equal the Pass A frame count")
    if history_count >= extension.num_frames:
        raise ValueError("extension must contain at least one future frame")
    if history.source_video_id != extension.source_video_id:
        raise ValueError("Pass A and Pass B source_video_id must match")
    if not np.array_equal(history.frame_indices, extension.frame_indices[:history_count]):
        raise ValueError("Pass A and Pass B history frame indices must match")
    if not np.array_equal(history.timestamps_s, extension.timestamps_s[:history_count]):
        raise ValueError("Pass A and Pass B history timestamps must match")
    if not np.array_equal(history.track_ids, extension.track_ids):
        raise ValueError("Pass A and Pass B track slots must match")

    common = history.geometry_validity & extension.geometry_validity[:history_count]
    source = extension.xyz[:history_count][common]
    destination = history.xyz[common]
    transform = None
    alignment_error = None
    try:
        transform = fit_similarity_transform(source, destination)
    except SimilarityAlignmentError as exc:
        alignment_error = str(exc)

    xyz = np.full(extension.xyz.shape, np.nan, dtype=np.float32)
    xyz[:history_count] = history.xyz
    uv = np.concatenate((history.uv, extension.uv[history_count:]), axis=0).copy()
    visibility = np.concatenate(
        (history.visibility, extension.visibility[history_count:]), axis=0
    ).copy()
    geometry = np.zeros(extension.geometry_validity.shape, dtype=np.bool_)
    geometry[:history_count] = history.geometry_validity
    eligible = transform is not None
    if transform is not None:
        future_valid = extension.geometry_validity[history_count:]
        transformed = transform.apply(extension.xyz[history_count:])
        transformed_finite = np.isfinite(transformed).all(axis=-1)
        geometry[history_count:] = future_valid & transformed_finite
        xyz[history_count:][geometry[history_count:]] = transformed[geometry[history_count:]]

    diagnostics = None
    scale = None
    determinant = None
    rotation = None
    translation = None
    if transform is not None:
        diagnostics = {
            name: getattr(transform.diagnostics, name)
            for name in transform.diagnostics.__dataclass_fields__
        }
        scale = transform.scale
        determinant = float(np.linalg.det(transform.rotation))
        rotation = transform.rotation.tolist()
        translation = transform.translation.tolist()
    lineage = {
        "construction": "history-anchored causal VGGT window",
        "history_count": history_count,
        "history_frame_indices": history.frame_indices.tolist(),
        "future_frame_indices": extension.frame_indices[history_count:].tolist(),
        "pass_a_history_only": history.lineage,
        "pass_b_extension": extension.lineage,
        "alignment_method": "history-only global Sim(3), extension gauge to history gauge",
        "future_points_used_for_alignment": False,
    }
    provenance = {
        "causal_training_eligible": eligible,
        "causal_scope": "only the recorded history cutoff and future frame set",
        "history_count": history_count,
        "causal_cutoff_frame_index": int(history.frame_indices[-1]),
        "causal_cutoff_timestamp_s": float(history.timestamps_s[-1]),
        "common_historical_correspondence_count": int(common.sum()),
        "estimated_scale": scale,
        "rotation_determinant": determinant,
        "rotation": rotation,
        "translation": translation,
        "alignment_diagnostics": diagnostics,
        "alignment_failure": alignment_error,
        "pass_a_runtime_s": history.provenance.get("elapsed_s"),
        "pass_b_runtime_s": extension.provenance.get("elapsed_s"),
        "peak_gpu_memory_bytes": max(
            int(history.provenance.get("peak_gpu_memory_bytes", 0)),
            int(extension.provenance.get("peak_gpu_memory_bytes", 0)),
        ),
        "eligibility_scope": "information flow and gauge construction, not 3D accuracy",
    }
    result = ParticleSequence(
        schema_version=history.schema_version,
        sample_id=f"{extension.sample_id}:history-{history_count}",
        source_video_id=extension.source_video_id,
        frame_indices=extension.frame_indices.copy(),
        timestamps_s=extension.timestamps_s.copy(),
        frame_sizes_hw=extension.frame_sizes_hw.copy(),
        track_ids=history.track_ids.copy(),
        xyz=xyz,
        uv=uv,
        visibility=visibility,
        geometry_validity=geometry,
        coordinate_system=history.coordinate_system,
        lineage=lineage,
        provenance=provenance,
    )
    validate_particle_sequence(result)
    return result
