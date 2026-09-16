from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_tools.v7.source128_extension.runner import _bootstrap, _stable_key, atomic_json, report


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


def _report_fixture(tmp_path: Path, real_status: str, fake_status: str) -> None:
    atomic_json(tmp_path / "protocol.json", {"selection": {"base_sources": ["S1"], "added_sources": [], "validation_sources": []}})
    atomic_json(tmp_path / "acquisition/media_manifest.json", {"results": [
        {"source_id": "S1", "role": "real", "status": real_status},
        {"source_id": "S1", "role": "fake", "status": fake_status},
    ]})
    atomic_json(tmp_path / "final_status.json", {"status": "MEDIA_INCOMPLETE"})


def test_report_clears_stale_media_incomplete_when_media_is_complete(tmp_path: Path) -> None:
    _report_fixture(tmp_path, "REUSED_EXISTING", "DOWNLOADED")

    result = report(tmp_path)

    final = json.loads((tmp_path / "final_status.json").read_text(encoding="utf-8"))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert result["status"] == "PLANNED"
    assert final["status"] == "PLANNED"
    assert final["media_download_complete"] is True
    assert final["media_manifest_count"] == final["expected_media_count"] == 2
    assert final["download_error_resolved"] is False
    assert "下载状态：COMPLETE（2/2 项）" in text


def test_report_preserves_media_incomplete_when_manifest_is_still_incomplete(tmp_path: Path) -> None:
    _report_fixture(tmp_path, "REUSED_EXISTING", "DOWNLOAD_FAILURE")

    result = report(tmp_path)

    final = json.loads((tmp_path / "final_status.json").read_text(encoding="utf-8"))
    assert result["status"] == "MEDIA_INCOMPLETE"
    assert final["status"] == "MEDIA_INCOMPLETE"
    assert final["media_download_complete"] is False
    assert final["download_error_resolved"] is False
