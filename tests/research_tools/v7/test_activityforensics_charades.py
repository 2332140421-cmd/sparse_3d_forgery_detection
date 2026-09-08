from __future__ import annotations

import csv
import json
from pathlib import Path

from research_tools.v7.data.build_activityforensics_review import HUMAN_FIELDS, build_review
from research_tools.v7.data.remote_zip64 import (
    CentralEntry,
    RangeClient,
    RangeProtocolError,
    _zip64_extra,
    exact_basename,
    extract_member,
    read_central_directory,
)
from research_tools.v7.data.prepare_activityforensics_charades import (
    build_mapping,
    parse_activityforensics_filename,
)
from research_tools.v7.data.target_activityforensics_review import freeze_selection


def test_filename_parser_preserves_multiple_official_segments():
    parsed = parse_activityforensics_filename(
        "00ZCA+10.50=16.40=charades@train_add@00ZCA@483@wan+24.90=42.40=charades@train_delete@00ZCA@480@wan.mp4"
    )
    assert [segment["manipulation_operation"] for segment in parsed["segments"]] == ["add", "delete"]
    assert [segment["start_s"] for segment in parsed["segments"]] == [10.5, 24.9]


def test_hcv1_filename_without_operation_id_is_supported():
    parsed = parse_activityforensics_filename("100_9eAOr_ttXp0+1.03=10.34=hcv1@train_delete@100_9eAOr_ttXp0@ltx.mp4")
    assert parsed["segments"][0]["operation_id"] is None
    assert parsed["segments"][0]["source_dataset"] == "hcv1"


def test_lineage_requires_official_charades_id(tmp_path: Path):
    metadata = tmp_path / "metadata"
    charades = tmp_path / "charades"
    metadata.mkdir()
    charades.mkdir()
    (metadata / "train.csv").write_text("file_name\nX+0.0=1.0=charades@train_add@MISSING@1@wan.mp4\n", encoding="utf-8")
    (metadata / "test.csv").write_text("file_name\n", encoding="utf-8")
    (charades / "Charades_v1_train.csv").write_text("id\nKNOWN\n", encoding="utf-8")
    (charades / "Charades_v1_test.csv").write_text("id\nOTHER\n", encoding="utf-8")
    import zipfile

    annotation_zip = tmp_path / "annot.zip"
    with zipfile.ZipFile(annotation_zip, "w") as archive:
        archive.writestr("annot/train@wan.txt", "X+0.0=1.0=charades@train_add@MISSING@1@wan.mp4 2.0 0.0=1.0\n")
    rows = build_mapping(metadata, annotation_zip, charades)
    assert rows[0]["lineage_status"] == "UNRESOLVED"


def test_review_package_keeps_human_fields_blank_and_symlinks(tmp_path: Path):
    base = tmp_path
    (base / "paired/manifests").mkdir(parents=True)
    (base / "validation").mkdir()
    real = base / "source/charades/videos/SRC.mp4"
    fake = base / "source/activityforensics/raw/video/generator/SRC+0=1=charades@train_add@SRC@1@wan.mp4"
    real.parent.mkdir(parents=True)
    fake.parent.mkdir(parents=True)
    real.write_bytes(b"real")
    fake.write_bytes(b"fake")
    mapping = [{"activityforensics_file": "video/generator/SRC+0=1=charades@train_add@SRC@1@wan.mp4", "lineage_status": "EXACT", "charades_source_id": "SRC", "generator": "wan", "manipulation_operation": "add", "manipulation_intervals": [{"start_s": 0.0, "end_s": 1.0}], "manipulation_segments": [{"start_s": 0.0, "end_s": 1.0, "generator": "wan", "manipulation_operation": "add"}]}]
    (base / "paired/manifests/activityforensics_source_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    media = [{"path": str(real), "status": "MEDIA_VALID", "duration_s": 2.0, "width": 2, "height": 2}, {"path": str(fake), "status": "MEDIA_VALID", "duration_s": 2.0, "width": 2, "height": 2}]
    (base / "validation/media_validation.json").write_text(json.dumps(media), encoding="utf-8")
    result = build_review(base, limit=1)
    assert result["unique_sources"] == 1
    rows = list(csv.DictReader((base / "review/review_manifest.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["V7_CORE_DECISION"] == ""
    assert all(rows[0][field] == "" for field in HUMAN_FIELDS)
    assert (base / "review/batch_001/0001/REAL__SRC.mp4").is_symlink()
    assert (base / "review/batch_001/0001/FAKE_A__SRC+0=1=charades@train_add@SRC@1@wan.mp4").is_symlink()


def test_review_package_is_not_a_model_or_src_dependency():
    assert not any("torch" in field.lower() for field in HUMAN_FIELDS)


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes):
        self.status = status
        self.headers = headers
        self._body = body
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return None
    def read(self):
        return self._body


def test_range_client_rejects_http_200():
    client = RangeClient("https://example.invalid/a.zip", opener=lambda *_args, **_kwargs: _Response(200, {}, b"full"))
    try:
        client.get_range(0, 3)
    except RangeProtocolError:
        pass
    else:
        raise AssertionError("HTTP 200 must not be accepted as a Range response")


def test_range_client_rejects_wrong_content_range():
    client = RangeClient("https://example.invalid/a.zip", opener=lambda *_args, **_kwargs: _Response(206, {"Content-Range": "bytes 0-2/4"}, b"abc"))
    try:
        client.get_range(0, 3)
    except RangeProtocolError:
        pass
    else:
        raise AssertionError("wrong Content-Range must be rejected")


def test_zip64_extra_decodes_64_bit_fields():
    import struct
    extra = struct.pack("<HHQQQ", 0x0001, 24, 1234567890123, 2233445566778, 3344556677889)
    assert _zip64_extra(extra, compressed=0xFFFFFFFF, uncompressed=0xFFFFFFFF, offset=0xFFFFFFFF) == (2233445566778, 1234567890123, 3344556677889)


def test_central_directory_and_exact_member_from_readable_zip(tmp_path: Path):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Charades_v1/SRC.mp4", b"video-bytes")
    payload = buffer.getvalue()
    class MemoryClient:
        size = len(payload)
        def head(self):
            return self.size
        def get_range(self, start, end):
            return payload[start : end + 1]
    entries, meta = read_central_directory(MemoryClient())
    assert meta["entries"] == 1
    entry = exact_basename(entries, "SRC.mp4")
    extract_member(MemoryClient(), entry, tmp_path / "SRC.mp4.part")
    assert (tmp_path / "SRC.mp4").read_bytes() == b"video-bytes"


def test_exact_member_ambiguity_is_rejected():
    entry = CentralEntry("a/SRC.mp4", 0, 0, 0, 0, 0, 0)
    duplicate = CentralEntry("b/SRC.mp4", 0, 0, 0, 0, 0, 0)
    try:
        exact_basename([entry, duplicate], "SRC.mp4")
    except LookupError:
        pass
    else:
        raise AssertionError("duplicate exact basenames must be rejected")


def test_target_selection_is_unique_deterministic_and_round_robin(tmp_path: Path):
    rows = []
    for index, (generator, operation) in enumerate([("fcvg", "delete"), ("vidu", "add"), ("wan", "add"), ("wan", "delete")]):
        for source in (f"S{index}A", f"S{index}B"):
            rows.append({"lineage_status": "EXACT", "charades_source_id": source, "generator": generator, "manipulation_operation": operation, "activityforensics_file": f"video/{source}.mp4", "manipulation_segments": [], "manipulation_intervals": [], "official_split": "train", "source_dataset": "charades"})
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(rows), encoding="utf-8")
    result = freeze_selection(mapping, tmp_path / "selection", primary_count=4, reserve_count=2)
    assert result["primary"] == 4 and result["reserve"] == 2
    selected = json.loads((tmp_path / "selection/selected_sources.json").read_text())
    assert len({row["charades_source_id"] for row in selected["sources"]}) == 4
    assert [tuple(row["selected_cell"]) for row in selected["sources"]] == [("fcvg", "delete"), ("vidu", "add"), ("wan", "add"), ("wan", "delete")]



def test_extract_rejects_crc_and_size_mismatch(tmp_path: Path):
    import dataclasses
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Charades_v1/SRC.mp4", b"video-bytes")
    payload = buffer.getvalue()
    class MemoryClient:
        size = len(payload)
        def get_range(self, start, end):
            return payload[start : end + 1]
    entries, _ = read_central_directory(MemoryClient())
    entry = entries[0]
    try:
        extract_member(MemoryClient(), dataclasses.replace(entry, crc32=entry.crc32 ^ 1), tmp_path / "crc.mp4.part")
    except ValueError as exc:
        assert "CRC32" in str(exc)
    else:
        raise AssertionError("CRC mismatch must fail")
    try:
        extract_member(MemoryClient(), dataclasses.replace(entry, uncompressed_size=entry.uncompressed_size + 1), tmp_path / "size.mp4.part")
    except ValueError as exc:
        assert "size" in str(exc)
    else:
        raise AssertionError("size mismatch must fail")
