from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_tools.v7.source128_extension.runner import _bootstrap, _stable_key, atomic_json


def test_selection_key_is_stable_and_order_independent() -> None:
    assert _stable_key("N79WJ") == _stable_key("N79WJ")
    assert _stable_key("N79WJ") != _stable_key("HWL2J")


def test_source_bootstrap_uses_paired_sources_and_fixed_seed() -> None:
    left = {"a": 0.5, "b": 1.0}
    right = {"a": 0.25, "b": 0.75}
    result = _bootstrap(left, right)
    assert result["source_count"] == 2
    assert result["mean"] == pytest.approx(0.25)
    assert result["seed"] == 20260909
    assert result["replicates"] == 10000


def test_atomic_json_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "bad.json", {"value": float("nan")})
