"""Deterministic start-frame grouping and local edge construction.

These functions are deliberately independent of the formal ``src`` data
contract.  They retain every initialized query, including queries that later
become numerically invalid, so an exploratory frontend cannot silently turn
missing observations into a selection rule.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def analysis_grid(size: int = 384, spacing: int = 3) -> np.ndarray:
    """Return the fixed pixel-centre grid ``1.5, 4.5, ...`` as ``[N,2]``."""

    if size <= 0 or spacing <= 0:
        raise ValueError("size and spacing must be positive")
    values = np.arange(spacing / 2.0, size, spacing, dtype=np.float32)
    u, v = np.meshgrid(values, values, indexing="xy")
    return np.stack((u.ravel(), v.ravel()), axis=1)


def _block_id(u: float, v: float, size: int, block_size: int) -> tuple[int, int]:
    bx = min(max(int(np.floor(u / block_size)), 0), (size - 1) // block_size)
    by = min(max(int(np.floor(v / block_size)), 0), (size - 1) // block_size)
    return by, bx


def assign_groups(
    raw_uv_analysis: np.ndarray,
    masks: np.ndarray | None,
    *,
    image_size: int = 384,
    block_size: int = 24,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Assign each query to one fixed block/instance group.

    ``masks`` is a boolean ``[M,H,W]`` raster in the same analysis coordinate
    system.  Overlap is resolved by smaller mask area, then mask index.  A
    query outside every instance remains in its own background block.
    """

    uv = np.asarray(raw_uv_analysis, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.all(np.isfinite(uv)):
        raise ValueError("raw_uv_analysis must be finite [N,2]")
    if image_size <= 0 or block_size <= 0:
        raise ValueError("image_size and block_size must be positive")
    if masks is None:
        mask_array = np.zeros((0, image_size, image_size), dtype=bool)
    else:
        mask_array = np.asarray(masks, dtype=bool)
        if mask_array.ndim != 3 or mask_array.shape[1:] != (image_size, image_size):
            raise ValueError("masks must have shape [M,image_size,image_size]")
    areas = np.sum(mask_array, axis=(1, 2), dtype=np.int64)
    groups: list[str] = []
    for u, v in uv:
        by, bx = _block_id(float(u), float(v), image_size, block_size)
        x, y = int(np.floor(u)), int(np.floor(v))
        hits = [index for index in range(mask_array.shape[0]) if mask_array[index, y, x]]
        if hits:
            chosen = min(hits, key=lambda index: (int(areas[index]), int(index)))
            groups.append(f"instance_{chosen:03d}_block_{by:02d}_{bx:02d}")
        else:
            groups.append(f"background_block_{by:02d}_{bx:02d}")
    unique = sorted(set(groups))
    return np.asarray(groups, dtype="U64"), [
        {
            "group_id": group,
            "query_count": int(np.sum(np.asarray(groups, dtype="U64") == group)),
            "kind": "instance" if group.startswith("instance_") else "background",
        }
        for group in unique
    ]

def fixed_local_edges(
    raw_uv_analysis: np.ndarray,
    group_ids: Iterable[str],
    *,
    max_neighbors: int = 8,
) -> list[dict[str, object]]:
    """Build a frozen undirected nearest-neighbour edge set within each group."""

    uv = np.asarray(raw_uv_analysis, dtype=np.float64)
    groups = np.asarray(list(group_ids), dtype="U64")
    if uv.ndim != 2 or uv.shape[1] != 2 or groups.shape != (uv.shape[0],):
        raise ValueError("uv and group_ids have incompatible shapes")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    by_group: dict[str, list[int]] = defaultdict(list)
    for query_id, group in enumerate(groups.tolist()):
        by_group[str(group)].append(query_id)
    edges: set[tuple[int, int]] = set()
    for group in sorted(by_group):
        members = by_group[group]
        for left in members:
            distances = sorted(
                (
                    float(np.linalg.norm(uv[left] - uv[right])),
                    int(right),
                )
                for right in members
                if right != left
            )
            for _distance, right in distances[:max_neighbors]:
                edge = (min(left, right), max(left, right))
                if edge[0] != edge[1]:
                    edges.add(edge)
    return [
        {
            "edge_id": int(edge_id),
            "left_query_id": int(left),
            "right_query_id": int(right),
            "group_id": str(groups[left]),
            "start_distance_px": float(np.linalg.norm(uv[left] - uv[right])),
        }
        for edge_id, (left, right) in enumerate(sorted(edges))
    ]
