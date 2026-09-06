"""History-only global similarity alignment for causal VGGT windows."""

from dataclasses import dataclass

import numpy as np


class SimilarityAlignmentError(ValueError):
    """Raised when a proper finite Sim(3) cannot be estimated."""


@dataclass(frozen=True, slots=True)
class SimilarityDiagnostics:
    correspondence_count: int
    raw_residual_mean: float
    raw_residual_max: float
    aligned_residual_mean: float
    aligned_residual_rmse: float
    aligned_residual_max: float
    anchor_rms_radius: float
    normalized_aligned_rmse: float


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    """A source-to-destination mapping: destination = scale R source + t."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    diagnostics: SimilarityDiagnostics

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the single fitted transform without changing its input."""

        points = np.asarray(points)
        return (self.scale * np.einsum("ij,...j->...i", self.rotation, points) + self.translation).astype(
            np.float32
        )


def fit_similarity_transform(source: np.ndarray, destination: np.ndarray) -> SimilarityTransform:
    """Fit a proper global Sim(3) from source to destination with Umeyama SVD.

    The normalized RMSE denominator is the destination points' RMS radius
    around their centroid. It is a label-free scale diagnostic only.
    """

    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or destination.shape != source.shape:
        raise SimilarityAlignmentError("source and destination must have matching shape [K,3]")
    if source.shape[0] < 3:
        raise SimilarityAlignmentError("at least three historical correspondences are required")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise SimilarityAlignmentError("alignment correspondences must be finite")

    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    anchor_rms_radius = float(np.sqrt(np.mean(np.sum(destination_centered**2, axis=1))))
    tolerance = np.finfo(np.float64).eps * max(source_variance, 1.0) * source.shape[0]
    if not np.isfinite(source_variance) or source_variance <= tolerance:
        raise SimilarityAlignmentError("source historical geometry has zero variance")
    if not np.isfinite(anchor_rms_radius) or anchor_rms_radius <= np.sqrt(tolerance):
        raise SimilarityAlignmentError("destination historical geometry has zero radius")
    if np.linalg.matrix_rank(source_centered, tol=np.sqrt(tolerance)) < 2:
        raise SimilarityAlignmentError("source historical geometry is collinear or coincident")

    covariance = destination_centered.T @ source_centered / source.shape[0]
    try:
        left, singular_values, right_t = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        raise SimilarityAlignmentError("alignment SVD did not converge") from exc
    if not all(np.isfinite(value).all() for value in (left, singular_values, right_t)):
        raise SimilarityAlignmentError("alignment SVD produced non-finite values")
    correction = np.ones(3, dtype=np.float64)
    correction[-1] = 1.0 if np.linalg.det(left @ right_t) >= 0 else -1.0
    rotation = left @ np.diag(correction) @ right_t
    scale = float(np.sum(singular_values * correction) / source_variance)
    translation = destination_mean - scale * (rotation @ source_mean)
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(scale) or scale <= 0:
        raise SimilarityAlignmentError("estimated similarity scale is not positive and finite")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise SimilarityAlignmentError("estimated similarity transform is not finite")
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise SimilarityAlignmentError("estimated rotation is not a proper rotation")

    raw = np.linalg.norm(destination - source, axis=1)
    aligned_points = scale * np.einsum("ij,kj->ki", rotation, source) + translation
    aligned = np.linalg.norm(destination - aligned_points, axis=1)
    rmse = float(np.sqrt(np.mean(aligned**2)))
    diagnostics = SimilarityDiagnostics(
        correspondence_count=int(source.shape[0]),
        raw_residual_mean=float(raw.mean()),
        raw_residual_max=float(raw.max()),
        aligned_residual_mean=float(aligned.mean()),
        aligned_residual_rmse=rmse,
        aligned_residual_max=float(aligned.max()),
        anchor_rms_radius=anchor_rms_radius,
        normalized_aligned_rmse=rmse / anchor_rms_radius,
    )
    return SimilarityTransform(
        scale=scale,
        rotation=rotation.astype(np.float64),
        translation=translation.astype(np.float64),
        diagnostics=diagnostics,
    )
