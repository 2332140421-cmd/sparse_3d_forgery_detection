"""Small deterministic dynamic-structure representation for the V7 probe."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    max_initial_distance: float
    max_relative_change: float
    minimum_size: int = 3
    minimum_overlap: int = 8


def motion_coherent_components(xyz: np.ndarray, valid: np.ndarray, config: ComponentConfig) -> tuple[tuple[int, ...], ...]:
    """Build label-blind connected components from proximity and relative coherence."""

    xyz = np.asarray(xyz)
    valid = np.asarray(valid)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("expected xyz [T,N,3] and valid [T,N]")
    count_n = xyz.shape[1]
    adjacency = [set() for _ in range(count_n)]
    for i in range(count_n):
        for j in range(i + 1, count_n):
            overlap = valid[:, i] & valid[:, j]
            if int(np.sum(overlap)) < config.minimum_overlap:
                continue
            relative = xyz[overlap, j] - xyz[overlap, i]
            if np.linalg.norm(relative[0]) > config.max_initial_distance:
                continue
            change = np.linalg.norm(relative - np.median(relative, axis=0), axis=1)
            if float(np.median(change)) <= config.max_relative_change:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for seed in range(count_n):
        if seed in seen:
            continue
        stack = [seed]
        group: list[int] = []
        seen.add(seed)
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(group) >= config.minimum_size:
            components.append(tuple(sorted(group)))
    return tuple(components)


def structure_state(xyz: np.ndarray, valid: np.ndarray, component: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Represent component structure by translation-invariant distance statistics."""

    indices = np.asarray(component, dtype=np.int64)
    count_t = xyz.shape[0]
    state = np.full((count_t, 4), np.nan, dtype=np.float64)
    state_valid = np.zeros(count_t, dtype=np.bool_)
    for t in range(count_t):
        ids = indices[valid[t, indices]]
        if ids.size < 3:
            continue
        points = xyz[t, ids].astype(np.float64)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        distances = distances[np.triu_indices(ids.size, k=1)]
        scale = float(np.median(distances))
        if not np.isfinite(scale) or scale <= 0:
            continue
        normalized = distances / scale
        state[t] = [np.mean(normalized), np.std(normalized), np.percentile(normalized, 25), np.percentile(normalized, 75)]
        state_valid[t] = True
    return state, state_valid


def structural_differences(state: np.ndarray, valid: np.ndarray, timestamps_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute timestamp-aware first and second structural changes."""

    state = np.asarray(state, dtype=np.float64)
    valid = np.asarray(valid)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if state.ndim != 2 or valid.shape != (state.shape[0],) or timestamps_s.shape != valid.shape:
        raise ValueError("state, validity, and timestamps have incompatible shapes")
    if timestamps_s.size > 1 and not np.all(np.diff(timestamps_s) > 0):
        raise ValueError("timestamps must be strictly increasing")
    first = np.full_like(state, np.nan)
    first_valid = np.zeros_like(valid)
    for t in range(1, state.shape[0]):
        if valid[t - 1] and valid[t]:
            first[t] = (state[t] - state[t - 1]) / (timestamps_s[t] - timestamps_s[t - 1])
            first_valid[t] = True
    second = np.full_like(state, np.nan)
    second_valid = np.zeros_like(valid)
    for t in range(2, state.shape[0]):
        if first_valid[t - 1] and first_valid[t]:
            second[t] = (first[t] - first[t - 1]) / (timestamps_s[t] - timestamps_s[t - 1])
            second_valid[t] = True
    return first, first_valid, second, second_valid
