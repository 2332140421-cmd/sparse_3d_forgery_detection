from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "review_v8_current_method.py"
SPEC = importlib.util.spec_from_file_location("review_v8_current_method", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_partition_labels_do_not_define_membership() -> None:
    mask = np.array([True, True, True, False])
    a = MODULE.partitions(np.array([0, 0, 1, 3]), mask)
    b = MODULE.partitions(np.array([7, 7, 2, 0]), mask)
    assert {tuple(x) for x in a} == {tuple(x) for x in b} == {(0, 1), (2,)}


def test_stats_are_strict_and_ordered() -> None:
    result = MODULE.stats([4.0, 1.0, 2.0, 3.0])
    assert result == {"min": 1.0, "p25": 1.75, "median": 2.5, "p75": 3.25, "max": 4.0}
    assert all(value is None for value in MODULE.stats([]).values())
