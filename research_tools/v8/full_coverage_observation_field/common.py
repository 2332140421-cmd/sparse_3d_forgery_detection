"""Small, explicit helpers for the isolated V8 pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROTOCOL_VERSION = "v8-full-coverage-observation-field-v1"
CANVAS_HW = (256, 256)
CELL_SIZE = 16
GRID_H = CANVAS_HW[0] // CELL_SIZE
GRID_W = CANVAS_HW[1] // CELL_SIZE
CELL_COUNT = GRID_H * GRID_W
FRAME_COUNT = 16
SEEDS = (20260909, 20260910, 20260911)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(*parts: Any) -> int:
    raw = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_npz_atomic(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        rows = data.get("rows", data.get("windows", data.get("results", [])))
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"unsupported manifest shape: {path}")
    return [dict(r) for r in rows]


def finite_or_zero(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite placeholder and an explicit finite mask; never changes raw data."""
    a = np.asarray(a)
    mask = np.isfinite(a).all(axis=-1) if a.ndim and a.shape[-1] > 1 else np.isfinite(a)
    return np.where(np.isfinite(a), a, 0.0).astype(np.float32), mask.astype(bool)


def make_cell_grid() -> np.ndarray:
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    return np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.int64)


def cell_index(y: int, x: int) -> int:
    return int(y * GRID_W + x)


def cell_neighbors() -> np.ndarray:
    edges: list[tuple[int, int]] = []
    for y in range(GRID_H):
        for x in range(GRID_W):
            i = cell_index(y, x)
            if x + 1 < GRID_W:
                edges.append((i, cell_index(y, x + 1)))
            if y + 1 < GRID_H:
                edges.append((i, cell_index(y + 1, x)))
    return np.asarray(edges, dtype=np.int64)
